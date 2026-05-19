import os
import json
import asyncio
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

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
    temperature=0.7,
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

# ── LangGraph State ───────────────────────────────────────────────────────────
class AgentState(TypedDict):
    user_id: str
    user_message: str
    messages: List[dict]  # full chat history from frontend

    # Retrieved context
    profile: Optional[dict]
    applied_jobs: Optional[List[dict]]
    rejected_jobs: Optional[List[dict]]
    pending_jobs: Optional[List[dict]]
    recent_interviews: Optional[List[dict]]
    conversation_history: Optional[List[dict]]
    resume: Optional[dict]

    # Reasoning outputs
    experience_tier: Optional[str]
    skills_gap: Optional[List[str]]
    detected_intent: Optional[str]
    career_context: Optional[str]

    # Final
    final_response: Optional[str]

# ── Request/Response Models ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    user_id: str
    messages: List[dict]  # [{role: "user"|"assistant", content: "..."}]

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Profile Memory Retrieval
# ═══════════════════════════════════════════════════════════════════════════════
async def retrieve_profile(state: AgentState) -> AgentState:
    user_id = state["user_id"]

    try:
        profile_res = supabase.from_("profiles") \
            .select("*").eq("user_id", user_id).single().execute()
        state["profile"] = profile_res.data or {}
    except:
        state["profile"] = {}

    # Resume
    try:
        resume_res = supabase.from_("resumes") \
            .select("tailored_json, created_at") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(1).execute()
        state["resume"] = resume_res.data[0] if resume_res.data else None
    except:
        state["resume"] = None

    print(f"[node:retrieve_profile] done — {state['profile'].get('full_name', 'unknown')}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Application History Analysis
# ═══════════════════════════════════════════════════════════════════════════════
async def analyze_history(state: AgentState) -> AgentState:
    user_id = state["user_id"]

    # Applied jobs
    try:
        applied_res = supabase.from_("jobs") \
            .select("company, role, score, score_reason, score_breakdown, found_at, description") \
            .eq("user_id", user_id).eq("status", "approved") \
            .order("found_at", desc=True).limit(10).execute()
        state["applied_jobs"] = applied_res.data or []
    except:
        state["applied_jobs"] = []

    # Rejected jobs
    try:
        rejected_res = supabase.from_("jobs") \
            .select("company, role, score") \
            .eq("user_id", user_id).eq("status", "rejected") \
            .order("found_at", desc=True).limit(10).execute()
        state["rejected_jobs"] = rejected_res.data or []
    except:
        state["rejected_jobs"] = []

    # Pending high-score jobs
    try:
        pending_res = supabase.from_("jobs") \
            .select("company, role, score, location") \
            .eq("user_id", user_id).eq("status", "pending") \
            .gte("score", 7) \
            .order("score", desc=True).limit(5).execute()
        state["pending_jobs"] = pending_res.data or []
    except:
        state["pending_jobs"] = []

    # Recent interviews
    try:
        interview_res = supabase.from_("interview_sessions") \
            .select("interview_type, messages, created_at") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True).limit(5).execute()
        state["recent_interviews"] = interview_res.data or []
    except:
        state["recent_interviews"] = []

    # Conversation history (last 20 from Supabase — persistent memory)
    try:
        history_res = supabase.from_("coach_conversations") \
            .select("role, content") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True).limit(20).execute()
        state["conversation_history"] = list(reversed(history_res.data or []))
    except:
        state["conversation_history"] = []

    # Extract skills gap from score breakdowns
    all_missing = []
    for job in state["applied_jobs"]:
        if job.get("score_breakdown"):
            try:
                bd = json.loads(job["score_breakdown"]) if isinstance(job["score_breakdown"], str) else job["score_breakdown"]
                all_missing.extend(bd.get("missing_skills", []))
            except:
                pass

    # Count frequency of missing skills
    from collections import Counter
    skill_counts = Counter(all_missing)
    state["skills_gap"] = [skill for skill, _ in skill_counts.most_common(10)]

    print(f"[node:analyze_history] applied={len(state['applied_jobs'])} rejected={len(state['rejected_jobs'])} pending={len(state['pending_jobs'])}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Career Reasoning Agent
# ═══════════════════════════════════════════════════════════════════════════════
async def career_reasoning(state: AgentState) -> AgentState:
    profile = state["profile"]
    years_exp = profile.get("years_experience", 0)

    # Experience tier
    if years_exp == 0:
        tier = "entry-level (0 years) — focus: portfolio, projects, first job strategy"
    elif years_exp <= 2:
        tier = "junior (1-2 years) — focus: skill depth, proving impact, job hopping risks"
    elif years_exp <= 5:
        tier = "mid-level (3-5 years) — focus: specialization, salary negotiation, promotion"
    else:
        tier = "senior (5+ years) — focus: leadership narrative, scope, equity, company selection"

    state["experience_tier"] = tier

    # Detect intent from last user message
    msg = state["user_message"].lower()
    if any(w in msg for w in ["rewrite", "bullet", "resume"]):
        intent = "resume_help"
    elif any(w in msg for w in ["cover letter"]):
        intent = "cover_letter"
    elif any(w in msg for w in ["interview", "practice", "question"]):
        intent = "interview_prep"
    elif any(w in msg for w in ["learn", "skill", "improve", "roadmap"]):
        intent = "learning_path"
    elif any(w in msg for w in ["salary", "negotiate", "offer", "compensation"]):
        intent = "salary_negotiation"
    elif any(w in msg for w in ["reject", "ghosted", "no response", "not hearing"]):
        intent = "rejection_support"
    else:
        intent = "general_career"

    state["detected_intent"] = intent

    # Build structured career context for the LLM
    skills = ", ".join(profile.get("skills", [])) or "Not specified"
    target_roles = ", ".join(profile.get("preferred_titles", [])) or "Not specified"
    skills_gap = ", ".join(state["skills_gap"]) if state["skills_gap"] else "None identified yet"

    applied_summary = "\n".join([
        f"• {j['role']} at {j['company']} — score {j['score']}/10"
        for j in state["applied_jobs"]
    ]) or "None yet"

    rejected_summary = "\n".join([
        f"• {j['role']} at {j['company']} — score {j['score']}/10"
        for j in state["rejected_jobs"]
    ]) or "None"

    pending_summary = "\n".join([
        f"• {j['role']} at {j['company']} — score {j['score']}/10 ({j['location']})"
        for j in state["pending_jobs"]
    ]) or "None"

    interview_summary = "\n".join([
        f"• {i['interview_type']} interview — {i['created_at'][:10]}"
        for i in state["recent_interviews"]
    ]) or "No interviews yet"

    intent_instruction = {
        "resume_help": "User wants resume help. Offer to rewrite specific bullets. Be specific about what's weak.",
        "cover_letter": "User wants a cover letter. Ask which job if not specified. Then generate one.",
        "interview_prep": "User wants interview practice. Give 3 likely questions for their target role and level. Offer to go deeper.",
        "learning_path": f"User wants to improve skills. Their top missing skills are: {skills_gap}. Give a specific 30-day learning plan.",
        "salary_negotiation": "User wants salary help. Give specific numbers based on their role and experience level. Be direct.",
        "rejection_support": "User is dealing with rejection or ghosting. Acknowledge the frustration first, then give a concrete strategy to fix it.",
        "general_career": "General career question. Be specific, reference their actual data.",
    }.get(intent, "")

    state["career_context"] = f"""
CANDIDATE PROFILE:
Name: {profile.get('full_name', 'User')}
Experience Level: {tier}
Skills: {skills}
Target Roles: {target_roles}
Work Preference: {profile.get('work_preference', 'remote')}
Location: {profile.get('location', 'Not specified')}
Summary: {profile.get('experience_summary', 'Not provided')}

JOB SEARCH STATUS:
Applied Jobs:
{applied_summary}

Rejected/Skipped:
{rejected_summary}

High-Score Pending (not acted on):
{pending_summary}

INTERVIEW HISTORY:
{interview_summary}

SKILLS GAP (recurring missing skills across applied jobs):
{skills_gap}

DETECTED INTENT: {intent}
COACHING INSTRUCTION: {intent_instruction}
""".strip()

    print(f"[node:career_reasoning] tier={tier} intent={intent}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Resume / Interview Tooling
# ═══════════════════════════════════════════════════════════════════════════════
async def apply_tooling(state: AgentState) -> AgentState:
    intent = state["detected_intent"]

    # For now: enrich context based on intent
    # Later: this node will call generate-docs, resume-tweak etc via HTTP
    if intent == "resume_help" and state["resume"]:
        resume_json = state["resume"].get("tailored_json", {})
        if resume_json:
            skills_in_resume = resume_json.get("skills", [])
            state["career_context"] += f"\n\nCURRENT RESUME SKILLS: {', '.join(skills_in_resume)}"

    if intent == "learning_path" and state["skills_gap"]:
        state["career_context"] += f"\n\nPRIORITY LEARNING TARGETS: {', '.join(state['skills_gap'][:5])}"

    print(f"[node:apply_tooling] intent={intent} enrichment done")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Final Response (streaming)
# ═══════════════════════════════════════════════════════════════════════════════
async def generate_response(state: AgentState) -> AgentState:
    system_prompt = f"""You are AlgoScout Career Coach — a sharp, emotionally intelligent AI career strategist.
You combine the precision of a senior recruiter with the warmth of a trusted mentor.

{state['career_context']}

COACHING RULES:
1. EXPERIENCE-AWARE: Tailor every response to their experience tier. Never give senior advice to juniors.
2. ACTION-ORIENTED: Every response ends with at least one concrete next action.
   Bad: "You should improve your resume."
   Good: "Your resume lacks deployment examples. Want me to rewrite 3 bullets showing production impact?"
3. USE THEIR ACTUAL DATA: Reference real skills, companies, scores. Never be generic.
4. EMOTIONAL INTELLIGENCE: If they sound discouraged — validate first, then pivot to action.
5. TOOL AWARENESS: Tell them what you can do:
   - "I can rewrite those resume bullets — just say 'rewrite my bullets for [role]'"
   - "I can generate a cover letter — tell me which job"
   - "Head to the Interview tab to practice live"
6. STRICT BOUNDARY: Only career topics. For anything else say:
   "I'm AlgoScout's career assistant and I only help with career-related questions. Try Claude.ai or ChatGPT for other topics 😊"

STYLE: Direct, warm, specific. Use markdown. No fluff. No padding."""

    # Build message history
    lc_messages = [SystemMessage(content=system_prompt)]

    # Add persistent history from Supabase
    for h in (state["conversation_history"] or []):
        if h["role"] == "user":
            lc_messages.append(HumanMessage(content=h["content"]))
        else:
            lc_messages.append(AIMessage(content=h["content"]))

    # Add current turn messages from frontend (last 4 to avoid duplication)
    for m in state["messages"][-4:]:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    # Collect full response for saving
    full_response = ""
    async for chunk in llm.astream(lc_messages):
        token = chunk.content
        if token:
            full_response += token

    state["final_response"] = full_response

    # Save to Supabase
    try:
        # Save user message
        supabase.from_("coach_conversations").insert({
            "user_id": state["user_id"],
            "role": "user",
            "content": state["user_message"],
        }).execute()
        # Save assistant response
        supabase.from_("coach_conversations").insert({
            "user_id": state["user_id"],
            "role": "assistant",
            "content": full_response,
        }).execute()
    except Exception as e:
        print(f"[node:generate_response] save error: {e}")

    print(f"[node:generate_response] response generated — {len(full_response)} chars")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# Build LangGraph
# ═══════════════════════════════════════════════════════════════════════════════
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_profile", retrieve_profile)
    graph.add_node("analyze_history", analyze_history)
    graph.add_node("career_reasoning", career_reasoning)
    graph.add_node("apply_tooling", apply_tooling)
    graph.add_node("generate_response", generate_response)

    graph.set_entry_point("retrieve_profile")
    graph.add_edge("retrieve_profile", "analyze_history")
    graph.add_edge("analyze_history", "career_reasoning")
    graph.add_edge("career_reasoning", "apply_tooling")
    graph.add_edge("apply_tooling", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()

career_graph = build_graph()

# ═══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/")
async def root():
    return {"status": "AlgoScout LangGraph backend running"}

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

    initial_state: AgentState = {
        "user_id": req.user_id,
        "user_message": last_user_msg,
        "messages": req.messages,
        "profile": None,
        "applied_jobs": None,
        "rejected_jobs": None,
        "pending_jobs": None,
        "recent_interviews": None,
        "conversation_history": None,
        "resume": None,
        "experience_tier": None,
        "skills_gap": None,
        "detected_intent": None,
        "career_context": None,
        "final_response": None,
    }

    async def stream_response():
        # Run graph up to generate_response node, stream tokens
        system_prompt_holder = {}

        # We need to stream — run nodes 1-4 first, then stream node 5
        state = initial_state.copy()
        state = await retrieve_profile(state)
        state = await analyze_history(state)
        state = await career_reasoning(state)
        state = await apply_tooling(state)

        # Now stream the LLM response directly
        system_prompt = f"""You are AlgoScout Career Coach — a sharp, emotionally intelligent AI career strategist.
You combine the precision of a senior recruiter with the warmth of a trusted mentor.

{state['career_context']}

COACHING RULES:
1. EXPERIENCE-AWARE: Tailor every response to their experience tier. Never give senior advice to juniors.
2. ACTION-ORIENTED: Every response ends with at least one concrete next action.
3. USE THEIR ACTUAL DATA: Reference real skills, companies, scores. Never be generic.
4. EMOTIONAL INTELLIGENCE: If they sound discouraged — validate first, then pivot to action.
5. TOOL AWARENESS: Tell them what you can do:
   - "I can rewrite those resume bullets — say 'rewrite my bullets for [role]'"
   - "I can generate a cover letter — tell me which job"
   - "Head to the Interview tab to practice live"
6. STRICT BOUNDARY: Only career topics. For anything else:
   "I'm AlgoScout's career assistant and I only help with career-related questions. Try Claude.ai or ChatGPT 😊"

STYLE: Direct, warm, specific. Use markdown. No fluff."""

        lc_messages = [SystemMessage(content=system_prompt)]

        for h in (state["conversation_history"] or []):
            if h["role"] == "user":
                lc_messages.append(HumanMessage(content=h["content"]))
            else:
                lc_messages.append(AIMessage(content=h["content"]))

        for m in state["messages"][-4:]:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_messages.append(AIMessage(content=m["content"]))

        full_response = ""
        async for chunk in llm.astream(lc_messages):
            token = chunk.content
            if token:
                full_response += token
                # SSE format — matches your existing frontend parser
                yield f"data: {json.dumps({'choices': [{'delta': {'content': token}}]})}\n\n"

        yield "data: [DONE]\n\n"

        # Save to Supabase after stream
        try:
            supabase.from_("coach_conversations").insert({
                "user_id": req.user_id,
                "role": "user",
                "content": last_user_msg,
            }).execute()
            supabase.from_("coach_conversations").insert({
                "user_id": req.user_id,
                "role": "assistant",
                "content": full_response,
            }).execute()
        except Exception as e:
            print(f"[chat] save error: {e}")

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

# ── Placeholder for voice interview WebSocket (Phase 2) ──────────────────────
# @app.websocket("/interview")
# async def interview_ws(websocket: WebSocket):
#     pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)