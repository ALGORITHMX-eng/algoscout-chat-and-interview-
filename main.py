import os
import json
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from supabase import create_client, Client

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    streaming=True,
)

app = FastAPI(title="AlgoScout LangGraph Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    user_id: str
    session_id: str
    user_message: str
    messages: List[dict]                      # current request messages only
    profile: Optional[dict]
    applied_jobs: Optional[List[dict]]
    rejected_jobs: Optional[List[dict]]
    pending_jobs: Optional[List[dict]]
    recent_interviews: Optional[List[dict]]
    session_history: Optional[List[dict]]     # this session only — not cross-session
    resume: Optional[dict]
    experience_tier: Optional[str]
    skills_gap: Optional[List[str]]
    detected_intent: Optional[str]
    is_emotional: Optional[bool]
    is_off_topic: Optional[bool]
    resume_context: Optional[str]             # only populated by resume_grounder
    career_context: Optional[str]
    previous_conclusions: Optional[dict]      # consistency checker memory
    final_prompt: Optional[str]
    final_response: Optional[str]

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    messages: List[dict]

# ── Identity-only system prompt ───────────────────────────────────────────────
IDENTITY_PROMPT = """You are ALGO — AlgoScout's AI career strategist.
You are the sharp friend who happens to know exactly how hiring works.
You do not perform helpfulness. You deliver results.

RULES:
- Only use data explicitly provided in this prompt. Never invent skills, companies, or history.
- Deliver verdict first, follow up second.
- 3 paragraphs max unless writing a full rewrite or plan.
- No generic advice. Every sentence must trace to their actual data.
- Write like a sharp person talking, not a consultant delivering a report.
- No headers like "Strengths:", "Next Action:", "Assessment:"."""

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Retrieve Profile
# ═══════════════════════════════════════════════════════════════════════════════
async def retrieve_profile(state: AgentState) -> AgentState:
    user_id = state["user_id"]
    try:
        res = supabase.from_("profiles").select("*").eq("user_id", user_id).single().execute()
        state["profile"] = res.data or {}
    except:
        state["profile"] = {}
    try:
        res = supabase.from_("resumes").select("tailored_json, created_at") \
            .eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        state["resume"] = res.data[0] if res.data else None
    except:
        state["resume"] = None
    print(f"[node:retrieve_profile] {state['profile'].get('full_name', 'unknown')}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Analyze History
# ═══════════════════════════════════════════════════════════════════════════════
async def analyze_history(state: AgentState) -> AgentState:
    user_id = state["user_id"]
    session_id = state["session_id"]

    try:
        res = supabase.from_("jobs").select("company, role, score, score_reason, score_breakdown") \
            .eq("user_id", user_id).eq("status", "approved") \
            .order("found_at", desc=True).limit(10).execute()
        state["applied_jobs"] = res.data or []
    except:
        state["applied_jobs"] = []
    try:
        res = supabase.from_("jobs").select("company, role, score") \
            .eq("user_id", user_id).eq("status", "rejected") \
            .order("found_at", desc=True).limit(10).execute()
        state["rejected_jobs"] = res.data or []
    except:
        state["rejected_jobs"] = []
    try:
        res = supabase.from_("jobs").select("company, role, score, location") \
            .eq("user_id", user_id).eq("status", "pending").gte("score", 7) \
            .order("score", desc=True).limit(5).execute()
        state["pending_jobs"] = res.data or []
    except:
        state["pending_jobs"] = []
    try:
        res = supabase.from_("interview_sessions").select("interview_type, created_at") \
            .eq("user_id", user_id).order("created_at", desc=True).limit(5).execute()
        state["recent_interviews"] = res.data or []
    except:
        state["recent_interviews"] = []

    # ✅ FIX: Only load history from THIS session — no cross-session leaking
    try:
        res = supabase.from_("coach_conversations").select("role, content") \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .order("created_at", desc=True).limit(20).execute()
        state["session_history"] = list(reversed(res.data or []))
    except:
        state["session_history"] = []

    # Skills gap from job score breakdowns
    all_missing = []
    for job in state["applied_jobs"]:
        if job.get("score_breakdown"):
            try:
                bd = json.loads(job["score_breakdown"]) if isinstance(job["score_breakdown"], str) else job["score_breakdown"]
                all_missing.extend(bd.get("missing_skills", []))
            except:
                pass
    state["skills_gap"] = [s for s, _ in Counter(all_missing).most_common(10)]

    # ✅ FIX: Load previous_conclusions from Supabase so consistency survives across turns
    try:
        res = supabase.from_("session_conclusions").select("conclusions") \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .single().execute()
        state["previous_conclusions"] = res.data.get("conclusions", {}) if res.data else {}
    except:
        state["previous_conclusions"] = {}

    print(f"[node:analyze_history] applied={len(state['applied_jobs'])} session_history={len(state['session_history'])} conclusions={list(state['previous_conclusions'].keys())}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Emotional Detector
# ═══════════════════════════════════════════════════════════════════════════════
async def emotional_detector(state: AgentState) -> AgentState:
    msg = state["user_message"].lower()
    emotional_signals = [
        "give up", "hopeless", "depressed", "worthless", "tired of",
        "frustrated", "rejected", "nobody wants me", "what's the point",
        "want to quit", "can't do this", "feel like a failure", "unlucky",
        "😭", "😔", "crying", "burnt out", "exhausted"
    ]
    state["is_emotional"] = any(signal in msg for signal in emotional_signals)
    print(f"[node:emotional_detector] is_emotional={state['is_emotional']}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Router (conditional edges live here)
# ═══════════════════════════════════════════════════════════════════════════════
def router(state: AgentState) -> str:
    msg = state["user_message"].lower()

    off_topic_signals = [
        "hack", "exploit", "poem", "weather", "news today", "latest news",
        "stock price", "recipe", "lyrics", "joke", "politics", "sports score",
        "what is the capital", "tell me a story"
    ]
    is_off_topic = any(signal in msg for signal in off_topic_signals)

    if is_off_topic:
        return "off_topic"
    elif state.get("is_emotional"):
        return "emotional"
    else:
        return "career"

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Off Topic Rejector (no LLM call — instant, free)
# ═══════════════════════════════════════════════════════════════════════════════
async def off_topic_rejector(state: AgentState) -> AgentState:
    state["final_response"] = (
        "I'm ALGO — AlgoScout's career assistant. "
        "I only handle career questions: resumes, job strategy, interviews, salary, and positioning. "
        "Try Claude.ai or ChatGPT for everything else."
    )
    print("[node:off_topic_rejector] blocked off-topic request")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 6 — Career Reasoning (seniority + intent — no LLM call)
# ═══════════════════════════════════════════════════════════════════════════════
async def career_reasoning(state: AgentState) -> AgentState:
    profile = state["profile"]
    years_exp = profile.get("years_experience", 0)

    if years_exp == 0:
        tier = "entry-level (0 years)"
    elif years_exp <= 2:
        tier = "junior (1-2 years)"
    elif years_exp <= 5:
        tier = "mid-level (3-5 years)"
    else:
        tier = "senior (5+ years)"
    state["experience_tier"] = tier

    msg = state["user_message"].lower()
    if any(w in msg for w in ["rewrite", "bullet", "resume"]):
        intent = "resume_help"
    elif "cover letter" in msg:
        intent = "cover_letter"
    elif any(w in msg for w in ["interview", "practice", "question"]):
        intent = "interview_prep"
    elif any(w in msg for w in ["learn", "skill", "improve", "roadmap"]):
        intent = "learning_path"
    elif any(w in msg for w in ["salary", "negotiate", "offer", "compensation"]):
        intent = "salary_negotiation"
    elif any(w in msg for w in ["reject", "ghosted", "no response"]):
        intent = "rejection_support"
    elif any(w in msg for w in ["position", "realistic", "candidate", "angle", "chance", "apply", "stand"]):
        intent = "positioning_strategy"
    else:
        intent = "general_career"
    state["detected_intent"] = intent

    print(f"[node:career_reasoning] tier={tier} intent={intent}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 7 — Resume Grounder (only runs when resume is relevant)
# ═══════════════════════════════════════════════════════════════════════════════
async def resume_grounder(state: AgentState) -> AgentState:
    resume_relevant_intents = ["resume_help", "cover_letter", "positioning_strategy", "general_career"]
    if state["detected_intent"] not in resume_relevant_intents or not state["resume"]:
        state["resume_context"] = None
        return state

    try:
        resume_json = state["resume"].get("tailored_json", {})
        skills = ", ".join(resume_json.get("skills", []))
        summary = resume_json.get("summary", "")
        experience = resume_json.get("experience", [])
        exp_text = "\n".join([
            f"• {e.get('role')} at {e.get('company')}: {e.get('summary', '')}"
            for e in experience[:3]
        ])
        state["resume_context"] = f"""
RESUME DATA (ground all claims here):
Summary: {summary}
Skills: {skills}
Experience:
{exp_text}
""".strip()
    except:
        state["resume_context"] = None

    print(f"[node:resume_grounder] resume_context={'loaded' if state['resume_context'] else 'empty'}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 8 — Apply Tooling (build final career context)
# ═══════════════════════════════════════════════════════════════════════════════
async def apply_tooling(state: AgentState) -> AgentState:
    profile = state["profile"]
    intent = state["detected_intent"]
    tier = state["experience_tier"]

    skills = ", ".join(profile.get("skills", [])) or "Not specified"
    target_roles = ", ".join(profile.get("preferred_titles", [])) or "Not specified"

    # ✅ FIX: Only inject skills_gap when actually relevant to intent
    skills_gap_intents = ["learning_path", "rejection_support", "positioning_strategy", "general_career"]
    skills_gap_text = ""
    if intent in skills_gap_intents and state["skills_gap"]:
        skills_gap_text = f"\nRECURRING SKILLS GAP: {', '.join(state['skills_gap'])}"

    applied_summary = "\n".join([
        f"• {j['role']} at {j['company']} — {j['score']}/10"
        for j in state["applied_jobs"]
    ]) or "None yet"

    pending_summary = "\n".join([
        f"• {j['role']} at {j['company']} — {j['score']}/10"
        for j in state["pending_jobs"]
    ]) or "None"

    intent_instructions = {
        "resume_help": "Rewrite their bullets now. Don't ask — just do it. Show BEFORE and AFTER. Name exactly what was weak.",
        "cover_letter": "Write the full cover letter now. Use their most recent applied job as context.",
        "interview_prep": "Give 3 hard, role-specific questions at their experience level. Answer one yourself as a model.",
        "learning_path": "Give a ruthless 30-day plan with specific resources targeting their skills gap.",
        "salary_negotiation": "Give exact salary ranges for their role and level. Tell them word-for-word what to say.",
        "rejection_support": "One sentence acknowledging it. Then diagnose the real reason using their data. Then 3 specific fixes.",
        "positioning_strategy": "Give a direct verdict — realistic or not. No hedging. Then give the exact positioning angle.",
        "general_career": "Answer using their actual data. Reference real skills, companies, scores. No generic advice.",
    }.get(intent, "")

    state["career_context"] = f"""
CANDIDATE:
Name: {profile.get('full_name', 'User')}
Level: {tier}
Skills: {skills}
Target Roles: {target_roles}
Work Preference: {profile.get('work_preference', 'remote')}
Location: {profile.get('location', 'Not specified')}
Background: {profile.get('experience_summary', 'Not provided')}

JOBS APPLIED:
{applied_summary}

HIGH-SCORE PENDING:
{pending_summary}
{skills_gap_text}

INTENT: {intent}
INSTRUCTION: {intent_instructions}
""".strip()

    # Append resume context if loaded
    if state.get("resume_context"):
        state["career_context"] += f"\n\n{state['resume_context']}"

    print(f"[node:apply_tooling] context built for intent={intent}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 9 — Consistency Checker
# ═══════════════════════════════════════════════════════════════════════════════
async def consistency_checker(state: AgentState) -> AgentState:
    # Build a short memory of key conclusions made this session
    # so the responder doesn't contradict itself
    conclusions = state.get("previous_conclusions", {})
    intent = state["detected_intent"]

    # Tag the current intent verdict so future calls can reference it
    conclusions[intent] = {
        "tier": state["experience_tier"],
        "skills_gap": state["skills_gap"],
    }
    state["previous_conclusions"] = conclusions

    # Inject consistency note into career_context if we've already made a verdict
    if len(conclusions) > 1:
        consistency_note = "\nPREVIOUS CONCLUSIONS THIS SESSION: " + json.dumps(
            {k: v for k, v in conclusions.items() if k != intent}
        ) + "\nStay consistent with these. Do not contradict them."
        state["career_context"] += consistency_note

    # ✅ FIX: Persist conclusions to Supabase so next turn loads them correctly
    try:
        existing = supabase.from_("session_conclusions").select("id") \
            .eq("user_id", state["user_id"]) \
            .eq("session_id", state["session_id"]) \
            .execute()
        if existing.data:
            supabase.from_("session_conclusions").update({"conclusions": conclusions}) \
                .eq("user_id", state["user_id"]) \
                .eq("session_id", state["session_id"]).execute()
        else:
            supabase.from_("session_conclusions").insert({
                "user_id": state["user_id"],
                "session_id": state["session_id"],
                "conclusions": conclusions,
            }).execute()
    except Exception as e:
        print(f"[node:consistency_checker] save error: {e}")

    print(f"[node:consistency_checker] conclusions tracked={list(conclusions.keys())}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 10 — Memory Summarizer (runs at end of every turn)
# ═══════════════════════════════════════════════════════════════════════════════
async def memory_summarizer(state: AgentState) -> AgentState:
    # Save this turn to Supabase under the session_id (not just user_id)
    # Actual save happens in responder after streaming completes
    # This node just prepares the summary tag
    print(f"[node:memory_summarizer] session={state['session_id']} ready to persist")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 11 — Responder (LLM call — career path)
# ═══════════════════════════════════════════════════════════════════════════════
async def responder(state: AgentState) -> AgentState:
    # Build messages
    lc_messages = [SystemMessage(content=f"{IDENTITY_PROMPT}\n\n{state['career_context']}")]

    # Session history only (scoped, no cross-session leaking)
    for h in (state["session_history"] or []):
        if h["role"] == "user":
            lc_messages.append(HumanMessage(content=h["content"]))
        else:
            lc_messages.append(AIMessage(content=h["content"]))

    # Current request messages (last 6)
    for m in state["messages"][-6:]:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    # Stream
    full_response = ""
    async for chunk in llm.astream(lc_messages):
        token = chunk.content
        if token:
            full_response += token

    state["final_response"] = full_response
    print(f"[node:responder] response_len={len(full_response)}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 11b — Emotional Responder (separate LLM call with empathy prompt)
# ═══════════════════════════════════════════════════════════════════════════════
async def emotional_responder(state: AgentState) -> AgentState:
    profile = state["profile"]
    name = profile.get("full_name", "").split()[0] if profile.get("full_name") else "hey"

    emotional_prompt = f"""You are ALGO — a career assistant who actually cares.
{name} is going through a hard moment in their job search. 

RULES FOR THIS RESPONSE:
- Acknowledge the emotion in ONE sentence. Be human, not corporate.
- No bullet points. No headers. No action items yet.
- Do NOT say "I can sense your frustration" — that's robotic.
- Do NOT pitch AlgoScout features.
- Do NOT give a numbered list of things to do.
- After acknowledging, ask ONE genuine question to understand what happened.
- Max 3 sentences total.
- Write like a real person who has been there, not a chatbot."""

    lc_messages = [
        SystemMessage(content=emotional_prompt),
        HumanMessage(content=state["user_message"])
    ]

    full_response = ""
    async for chunk in llm.astream(lc_messages):
        token = chunk.content
        if token:
            full_response += token

    state["final_response"] = full_response
    print(f"[node:emotional_responder] empathy response generated")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# Build the Graph
# ═══════════════════════════════════════════════════════════════════════════════
def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("retrieve_profile", retrieve_profile)
    graph.add_node("analyze_history", analyze_history)
    graph.add_node("emotional_detector", emotional_detector)
    graph.add_node("router", lambda state: state)           # router is pure conditional, no logic
    graph.add_node("off_topic_rejector", off_topic_rejector)
    graph.add_node("career_reasoning", career_reasoning)
    graph.add_node("resume_grounder", resume_grounder)
    graph.add_node("apply_tooling", apply_tooling)
    graph.add_node("consistency_checker", consistency_checker)
    graph.add_node("memory_summarizer", memory_summarizer)
    graph.add_node("responder", responder)
    graph.add_node("emotional_responder", emotional_responder)

    # Linear start
    graph.set_entry_point("retrieve_profile")
    graph.add_edge("retrieve_profile", "analyze_history")
    graph.add_edge("analyze_history", "emotional_detector")
    graph.add_edge("emotional_detector", "router")

    # Conditional routing
    graph.add_conditional_edges(
        "router",
        router,
        {
            "off_topic": "off_topic_rejector",
            "emotional": "emotional_responder",
            "career": "career_reasoning",
        }
    )

    # Career path
    graph.add_edge("career_reasoning", "resume_grounder")
    graph.add_edge("resume_grounder", "apply_tooling")
    graph.add_edge("apply_tooling", "consistency_checker")
    graph.add_edge("consistency_checker", "memory_summarizer")
    graph.add_edge("memory_summarizer", "responder")

    # All paths end
    graph.add_edge("responder", END)
    graph.add_edge("emotional_responder", END)
    graph.add_edge("off_topic_rejector", END)

    return graph.compile()

algo_graph = build_graph()

# ═══════════════════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/")
async def root():
    return {"status": "AlgoScout LangGraph backend running — 11 nodes"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.user_id or not req.messages:
        raise HTTPException(status_code=400, detail="user_id and messages required")

    last_user_msg = next(
        (m["content"] for m in reversed(req.messages) if m["role"] == "user"), ""
    )
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found")

    async def stream_response():
        initial_state: AgentState = {
            "user_id": req.user_id,
            "session_id": req.session_id,
            "user_message": last_user_msg,
            "messages": req.messages,
            "profile": None,
            "applied_jobs": None,
            "rejected_jobs": None,
            "pending_jobs": None,
            "recent_interviews": None,
            "session_history": None,
            "resume": None,
            "experience_tier": None,
            "skills_gap": None,
            "detected_intent": None,
            "is_emotional": None,
            "is_off_topic": None,
            "resume_context": None,
            "career_context": None,
            "previous_conclusions": {},
            "final_prompt": None,
            "final_response": None,
        }

        # Run the graph
        final_state = await algo_graph.ainvoke(initial_state)
        final_response = final_state.get("final_response", "")

        # ✅ FIX: Stream in chunks not char-by-char
        CHUNK_SIZE = 12
        for i in range(0, len(final_response), CHUNK_SIZE):
            chunk = final_response[i:i + CHUNK_SIZE]
            yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"

        yield "data: [DONE]\n\n"

        # Persist to Supabase under session_id
        try:
            supabase.from_("coach_conversations").insert({
                "user_id": req.user_id,
                "session_id": req.session_id,
                "role": "user",
                "content": last_user_msg,
            }).execute()
            supabase.from_("coach_conversations").insert({
                "user_id": req.user_id,
                "session_id": req.session_id,
                "role": "assistant",
                "content": final_response,
            }).execute()
        except Exception as e:
            print(f"[chat] save error: {e}")

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)