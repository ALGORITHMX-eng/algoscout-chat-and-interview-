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

import os
import json
import asyncio
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
    frequency_penalty=0.8,
    presence_penalty=0.6,
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
    messages: List[dict]
    profile: Optional[dict]
    applied_jobs: Optional[List[dict]]
    rejected_jobs: Optional[List[dict]]
    pending_jobs: Optional[List[dict]]
    recent_interviews: Optional[List[dict]]
    session_history: Optional[List[dict]]
    resume: Optional[dict]
    experience_tier: Optional[str]
    skills_gap: Optional[List[str]]
    detected_intent: Optional[str]
    detected_route: Optional[str]
    resume_context: Optional[str]
    career_context: Optional[str]
    previous_conclusions: Optional[dict]
    final_response: Optional[str]

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    messages: List[dict]

# ── AlgoScout App Navigation Map ─────────────────────────────────────────────
APP_NAVIGATION = """
ALGOSCOUT APP NAVIGATION (use this to direct users):
- Dashboard → view and manage all job leads, approve/reject jobs
- Chat (current) → career coaching, resume advice, job strategy
- Interview tab → practice voice or text interviews for specific jobs
- Profile/Settings → update resume, skills, work preferences, target roles
- Add Job button → manually add a job by URL or company name
- Job detail page → view tailored resume and cover letter for a specific job
"""

# ── Identity System Prompt ────────────────────────────────────────────────────
IDENTITY_PROMPT = """You are ALGO — AlgoScout's AI career strategist and assistant.
You are the sharp, warm friend who knows exactly how hiring works and also knows the AlgoScout app inside out.

{app_navigation}

PERSONALITY:
- When someone greets you casually (hey, hi, hello, what's up), greet them back warmly first. Ask how they're doing. Don't jump straight to career data.
- When someone asks a question, answer it directly and naturally — like a smart friend, not a corporate bot.
- When someone asks where to do something in the app, tell them exactly where to go.
- When someone is venting or frustrated, acknowledge it first before strategy.

CAREER RULES:
- Only use data explicitly provided in this prompt. Never invent skills, companies, or history.
- Deliver verdict first, follow up second.
- 3 paragraphs max unless writing a full rewrite or plan.
- No generic advice. Every sentence must trace to their actual data.
- Write like a sharp person talking, not a consultant delivering a report.
- No headers like "Strengths:", "Next Action:", "Assessment:".

OFF-TOPIC RULE:
- If someone asks for coding help unrelated to their career (write this script, debug my code, etc.), say: "That's outside what I do — try Claude.ai or ChatGPT for that. Anything career-wise I can help with?"
- Everything else career-related, app-related, or conversational is fair game.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Retrieve Profile + History (parallelized)
# ═══════════════════════════════════════════════════════════════════════════════
async def retrieve_profile(state: AgentState) -> AgentState:
    user_id = state["user_id"]
    session_id = state["session_id"]

    def fetch_profile():
        try:
            return supabase.from_("profiles").select("*").eq("user_id", user_id).single().execute()
        except:
            return None

    def fetch_resume():
        try:
            return supabase.from_("resumes").select("tailored_json, created_at") \
                .eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        except:
            return None

    profile_res, resume_res = await asyncio.gather(
        asyncio.to_thread(fetch_profile),
        asyncio.to_thread(fetch_resume),
    )

    state["profile"] = (profile_res.data if profile_res else None) or {}
    state["resume"] = (resume_res.data[0] if resume_res and resume_res.data else None)
    print(f"[node:retrieve_profile] {state['profile'].get('full_name', 'unknown')}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Analyze History (parallelized)
# ═══════════════════════════════════════════════════════════════════════════════
async def analyze_history(state: AgentState) -> AgentState:
    user_id = state["user_id"]
    session_id = state["session_id"]

    def fetch_applied():
        try:
            return supabase.from_("jobs").select("company, role, score, score_reason, score_breakdown") \
                .eq("user_id", user_id).eq("status", "approved") \
                .order("found_at", desc=True).limit(10).execute()
        except: return None

    def fetch_pending():
        try:
            return supabase.from_("jobs").select("company, role, score, location") \
                .eq("user_id", user_id).eq("status", "pending").gte("score", 7) \
                .order("score", desc=True).limit(5).execute()
        except: return None

    def fetch_history():
        try:
            return supabase.from_("coach_conversations").select("role, content") \
                .eq("user_id", user_id).eq("session_id", session_id) \
                .order("created_at", desc=True).limit(20).execute()
        except: return None

    def fetch_conclusions():
        try:
            return supabase.from_("session_conclusions").select("conclusions") \
                .eq("user_id", user_id).eq("session_id", session_id).single().execute()
        except: return None

    applied_res, pending_res, history_res, conclusions_res = await asyncio.gather(
        asyncio.to_thread(fetch_applied),
        asyncio.to_thread(fetch_pending),
        asyncio.to_thread(fetch_history),
        asyncio.to_thread(fetch_conclusions),
    )

    state["applied_jobs"] = (applied_res.data if applied_res else None) or []
    state["pending_jobs"] = (pending_res.data if pending_res else None) or []
    state["session_history"] = list(reversed((history_res.data if history_res else None) or []))
    state["previous_conclusions"] = (conclusions_res.data.get("conclusions", {}) if conclusions_res and conclusions_res.data else {})
    state["rejected_jobs"] = []
    state["recent_interviews"] = []

    all_missing = []
    for job in state["applied_jobs"]:
        if job.get("score_breakdown"):
            try:
                bd = json.loads(job["score_breakdown"]) if isinstance(job["score_breakdown"], str) else job["score_breakdown"]
                all_missing.extend(bd.get("missing_skills", []))
            except: pass
    state["skills_gap"] = [s for s, _ in Counter(all_missing).most_common(10)]

    print(f"[node:analyze_history] applied={len(state['applied_jobs'])} history={len(state['session_history'])}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Smart Classifier (LLM-based routing)
# ═══════════════════════════════════════════════════════════════════════════════
async def smart_classifier(state: AgentState) -> AgentState:
    msg = state["user_message"]

    classify_prompt = f"""Classify this message into exactly one category:

- "greeting" → casual hello, hi, hey, what's up, how are you, good morning — small talk with no specific request
- "emotional" → user is frustrated, sad, giving up, venting about job search, feeling hopeless
- "off_topic" → requests for coding help (write code, debug, script), recipes, weather, news, sports scores, creative writing unrelated to career
- "career" → everything else: job questions, resume help, interview prep, salary, skills, app navigation, AlgoScout features, career strategy

Message: "{msg}"

Reply with ONLY one word: greeting, emotional, off_topic, or career"""

    try:
        result = await llm.ainvoke([
            SystemMessage(content="You are a message classifier. Reply with only one word."),
            HumanMessage(content=classify_prompt)
        ])
        category = result.content.strip().lower().split()[0]
        if category in ["greeting", "emotional", "off_topic", "career"]:
            state["detected_route"] = category
        else:
            state["detected_route"] = "career"
    except:
        state["detected_route"] = "career"

    print(f"[node:smart_classifier] route={state['detected_route']} msg='{msg[:50]}'")
    return state

# ── Router function (reads from state) ────────────────────────────────────────
def router(state: AgentState) -> str:
    return state.get("detected_route", "career")

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Greeting Responder (no career data needed)
# ═══════════════════════════════════════════════════════════════════════════════
async def greeting_responder(state: AgentState) -> AgentState:
    profile = state["profile"] or {}
    name = profile.get("full_name", "").split()[0] if profile.get("full_name") else ""

    greeting_prompt = f"""You are ALGO — a warm, smart career assistant inside the AlgoScout app.
{"The user's name is " + name + "." if name else ""}

The user just sent a casual greeting or small talk message.

RULES:
- Greet them back warmly and naturally. Use their name if you have it.
- Ask how they're doing or what's on their mind — ONE short question.
- Do NOT dump career data or job listings at them unprompted.
- Do NOT say "I'm here to help with your career" or any corporate opening.
- Max 2 sentences. Sound like a real person."""

    lc_messages = [
        SystemMessage(content=greeting_prompt),
        HumanMessage(content=state["user_message"])
    ]

    full_response = ""
    async for chunk in llm.astream(lc_messages):
        token = chunk.content
        if token:
            full_response += token

    state["final_response"] = full_response
    print(f"[node:greeting_responder] done")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Off Topic Rejector (no LLM call)
# ═══════════════════════════════════════════════════════════════════════════════
async def off_topic_rejector(state: AgentState) -> AgentState:
    state["final_response"] = (
        "That's outside what I do — try Claude.ai or ChatGPT for that. "
        "Anything career-wise I can help with?"
    )
    print("[node:off_topic_rejector] blocked")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 6 — Career Reasoning (seniority + intent)
# ═══════════════════════════════════════════════════════════════════════════════
async def career_reasoning(state: AgentState) -> AgentState:
    profile = state["profile"]
    years_exp = profile.get("years_experience", 0)

    if years_exp == 0: tier = "entry-level (0 years)"
    elif years_exp <= 2: tier = "junior (1-2 years)"
    elif years_exp <= 5: tier = "mid-level (3-5 years)"
    else: tier = "senior (5+ years)"
    state["experience_tier"] = tier

    msg = state["user_message"].lower()
    if any(w in msg for w in ["rewrite", "bullet", "resume", "cv"]):
        intent = "resume_help"
    elif "cover letter" in msg:
        intent = "cover_letter"
    elif any(w in msg for w in ["interview", "practice", "question"]):
        intent = "interview_prep"
    elif any(w in msg for w in ["learn", "skill", "improve", "roadmap", "course"]):
        intent = "learning_path"
    elif any(w in msg for w in ["salary", "negotiate", "offer", "compensation", "pay"]):
        intent = "salary_negotiation"
    elif any(w in msg for w in ["reject", "ghosted", "no response", "silence"]):
        intent = "rejection_support"
    elif any(w in msg for w in ["position", "realistic", "candidate", "angle", "chance", "apply", "stand"]):
        intent = "positioning_strategy"
    elif any(w in msg for w in ["where", "how do i", "settings", "profile", "update", "change", "find", "navigate"]):
        intent = "app_navigation"
    else:
        intent = "general_career"
    state["detected_intent"] = intent

    print(f"[node:career_reasoning] tier={tier} intent={intent}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 7 — Resume Grounder
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
        state["resume_context"] = f"""RESUME DATA:
Summary: {summary}
Skills: {skills}
Experience:
{exp_text}""".strip()
    except:
        state["resume_context"] = None
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 8 — Apply Tooling (build career context)
# ═══════════════════════════════════════════════════════════════════════════════
async def apply_tooling(state: AgentState) -> AgentState:
    profile = state["profile"]
    intent = state["detected_intent"]
    tier = state["experience_tier"]

    skills = ", ".join(profile.get("skills", [])) or "Not specified"
    target_roles = ", ".join(profile.get("preferred_titles", [])) or "Not specified"

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
        "learning_path": "Don't give a rigid 30-day plan. Pick ONE skill from their gap that matches what companies in their applied jobs actually want. Tell them exactly what to build with it — a specific mini project. One skill, one project, concrete outcome. Not a syllabus.",
        "salary_negotiation": "Give exact salary ranges for their role and level. Tell them word-for-word what to say.",
        "rejection_support": "One sentence acknowledging it. Then diagnose the real reason using their data. Then 3 specific fixes.",
        "positioning_strategy": "Give a direct verdict. No hedging. Reference session history if one exists.",
        "app_navigation": "Tell them exactly where to go in the app. Be specific — Dashboard, Profile tab, Interview tab, Settings, Add Job button. Don't be vague.",
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

    if state.get("resume_context"):
        state["career_context"] += f"\n\n{state['resume_context']}"

    if state.get("previous_conclusions") and len(state["previous_conclusions"]) > 0:
        state["career_context"] += "\n\nPREVIOUS CONCLUSIONS THIS SESSION: " + json.dumps(state["previous_conclusions"]) + "\nStay consistent with these."

    print(f"[node:apply_tooling] intent={intent}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 9 — Consistency Checker
# ═══════════════════════════════════════════════════════════════════════════════
async def consistency_checker(state: AgentState) -> AgentState:
    conclusions = state.get("previous_conclusions", {})
    intent = state["detected_intent"]
    conclusions[intent] = {"tier": state["experience_tier"], "skills_gap": state["skills_gap"]}
    state["previous_conclusions"] = conclusions

    try:
        existing = supabase.from_("session_conclusions").select("id") \
            .eq("user_id", state["user_id"]).eq("session_id", state["session_id"]).execute()
        if existing.data:
            supabase.from_("session_conclusions").update({"conclusions": conclusions}) \
                .eq("user_id", state["user_id"]).eq("session_id", state["session_id"]).execute()
        else:
            supabase.from_("session_conclusions").insert({
                "user_id": state["user_id"],
                "session_id": state["session_id"],
                "conclusions": conclusions,
            }).execute()
    except Exception as e:
        print(f"[node:consistency_checker] save error: {e}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 10 — Responder (career path LLM call)
# ═══════════════════════════════════════════════════════════════════════════════
async def responder(state: AgentState) -> AgentState:
    system = IDENTITY_PROMPT.format(app_navigation=APP_NAVIGATION)
    lc_messages = [SystemMessage(content=f"{system}\n\n{state['career_context']}")]

    for h in (state["session_history"] or []):
        if h["role"] == "user":
            lc_messages.append(HumanMessage(content=h["content"]))
        else:
            lc_messages.append(AIMessage(content=h["content"]))

    for m in state["messages"][-6:]:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    full_response = ""
    async for chunk in llm.astream(lc_messages):
        token = chunk.content
        if token:
            full_response += token

    state["final_response"] = full_response
    print(f"[node:responder] len={len(full_response)}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 11 — Emotional Responder
# ═══════════════════════════════════════════════════════════════════════════════
async def emotional_responder(state: AgentState) -> AgentState:
    profile = state["profile"] or {}
    name = profile.get("full_name", "").split()[0] if profile.get("full_name") else ""

    prompt = f"""You are ALGO — a career assistant who actually cares.
{"User's name is " + name + "." if name else ""}

RULES:
- Acknowledge the emotion in ONE sentence. Be human, not corporate.
- No bullet points. No headers. No action items yet.
- Do NOT say "I can sense your frustration".
- Ask ONE genuine question to understand what happened.
- Max 3 sentences total."""

    lc_messages = [SystemMessage(content=prompt), HumanMessage(content=state["user_message"])]

    full_response = ""
    async for chunk in llm.astream(lc_messages):
        token = chunk.content
        if token:
            full_response += token

    state["final_response"] = full_response
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# Build the Graph
# ═══════════════════════════════════════════════════════════════════════════════
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_profile", retrieve_profile)
    graph.add_node("analyze_history", analyze_history)
    graph.add_node("smart_classifier", smart_classifier)
    graph.add_node("router", lambda state: state)
    graph.add_node("greeting_responder", greeting_responder)
    graph.add_node("off_topic_rejector", off_topic_rejector)
    graph.add_node("emotional_responder", emotional_responder)
    graph.add_node("career_reasoning", career_reasoning)
    graph.add_node("resume_grounder", resume_grounder)
    graph.add_node("apply_tooling", apply_tooling)
    graph.add_node("consistency_checker", consistency_checker)
    graph.add_node("responder", responder)

    graph.set_entry_point("retrieve_profile")
    graph.add_edge("retrieve_profile", "analyze_history")
    graph.add_edge("analyze_history", "smart_classifier")
    graph.add_edge("smart_classifier", "router")

    graph.add_conditional_edges(
        "router",
        router,
        {
            "greeting": "greeting_responder",
            "emotional": "emotional_responder",
            "off_topic": "off_topic_rejector",
            "career": "career_reasoning",
        }
    )

    graph.add_edge("career_reasoning", "resume_grounder")
    graph.add_edge("resume_grounder", "apply_tooling")
    graph.add_edge("apply_tooling", "consistency_checker")
    graph.add_edge("consistency_checker", "responder")

    graph.add_edge("responder", END)
    graph.add_edge("greeting_responder", END)
    graph.add_edge("emotional_responder", END)
    graph.add_edge("off_topic_rejector", END)

    return graph.compile()

algo_graph = build_graph()

# ═══════════════════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/")
async def root():
    return {"status": "AlgoScout LangGraph backend running — 12 nodes"}

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
            "detected_route": None,
            "resume_context": None,
            "career_context": None,
            "previous_conclusions": {},
            "final_response": None,
        }

        final_state = await algo_graph.ainvoke(initial_state)
        final_response = final_state.get("final_response", "")

        CHUNK_SIZE = 12
        for i in range(0, len(final_response), CHUNK_SIZE):
            chunk = final_response[i:i + CHUNK_SIZE]
            yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"

        yield "data: [DONE]\n\n"

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

# ═══════════════════════════════════════════════════════════════════════════════
# ALGOSCOUT — INTERVIEW GRAPH
# Add this to main.py (paste before the `if __name__ == "__main__":` line)
# ═══════════════════════════════════════════════════════════════════════════════

import json
from typing import TypedDict, List, Optional

# ── Interview State ───────────────────────────────────────────────────────────
class InterviewState(TypedDict):
    user_id: str
    session_id: str
    job_id: str
    user_message: str
    messages: List[dict]
    profile: Optional[dict]
    job: Optional[dict]
    resume: Optional[dict]
    question_count: int
    duration_minutes: int
    elapsed_seconds: int
    running_score: float
    skills_gap: Optional[List[str]]
    last_question_topic: Optional[str]
    difficulty: str               # "easy" | "medium" | "hard"
    interview_context: Optional[str]
    final_response: Optional[str]

# ── Interview System Prompt ───────────────────────────────────────────────────
INTERVIEW_IDENTITY = """You are a senior interviewer at {company}. Your name is ALGO.
You are conducting a {role} interview with {name}.
You are warm but professional — like a real human interviewer who actually enjoys their job.

RULES:
- Ask ONE question at a time. Never two.
- Open with ONE short casual question about their day, max 8 words. No corporate warmth.
- After they answer the small talk, say one short sentence then ask: "Can you tell me about yourself and your experience so far?"
- After they answer that, pick your next question by pulling a specific thread from what they just said. Reference it directly — "You mentioned X, can you walk me through..."
- Use their skills gap to decide which threads to pull harder on.
- If the answer is weak: probe deeper — "Can you give a specific example of that?"
- If the answer is strong: acknowledge briefly in one sentence then pivot to a harder related topic.
- Never give hints, coaching, or feedback during the interview.
- Never break character. You are ALGO, a human interviewer.
- Never say you are an AI.
- After {max_questions} questions, close warmly: "Alright {name}, that's everything I needed — we'll be in touch soon. Take care!"

{interview_context}"""

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Retrieve Profile (reuse existing function signature)
# ═══════════════════════════════════════════════════════════════════════════════
async def interview_retrieve_profile(state: InterviewState) -> InterviewState:
    user_id = state["user_id"]
    try:
        res = supabase.from_("profiles").select("*").eq("user_id", user_id).single().execute()
        state["profile"] = res.data or {}
    except:
        state["profile"] = {}
    try:
        res = supabase.from_("resumes").select("tailored_json") \
            .eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        state["resume"] = res.data[0] if res.data else None
    except:
        state["resume"] = None
    print(f"[interview:retrieve_profile] {state['profile'].get('full_name', 'unknown')}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Load Job Context
# ═══════════════════════════════════════════════════════════════════════════════
async def load_job_context(state: InterviewState) -> InterviewState:
    try:
        res = supabase.from_("jobs").select("*").eq("id", state["job_id"]).single().execute()
        state["job"] = res.data or {}
    except:
        state["job"] = {}

    # Extract skills gap from score_breakdown
    job = state["job"]
    skills_gap = []
    if job.get("score_breakdown"):
        try:
            bd = json.loads(job["score_breakdown"]) if isinstance(job["score_breakdown"], str) else job["score_breakdown"]
            skills_gap = bd.get("missing_skills", [])
        except:
            pass
    state["skills_gap"] = skills_gap
    print(f"[interview:load_job_context] job={job.get('role')} gaps={skills_gap}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Load Interview State (question count, difficulty, score)
# ═══════════════════════════════════════════════════════════════════════════════
async def load_interview_state(state: InterviewState) -> InterviewState:
    # Count how many questions have been asked so far
    assistant_msgs = [m for m in state["messages"] if m["role"] == "assistant"]
    state["question_count"] = len(assistant_msgs)

    # Calculate difficulty based on running score
    score = state.get("running_score", 5.0)
    if score >= 7.5:
        state["difficulty"] = "hard"
    elif score >= 5.0:
        state["difficulty"] = "medium"
    else:
        state["difficulty"] = "easy"

    print(f"[interview:load_interview_state] q={state['question_count']} difficulty={state['difficulty']}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Answer Evaluator (scores user's last answer 1-10)
# ═══════════════════════════════════════════════════════════════════════════════
async def answer_evaluator(state: InterviewState) -> InterviewState:
    # Only evaluate if there's a previous user message (not the first turn)
    user_msgs = [m for m in state["messages"] if m["role"] == "user"]
    if len(user_msgs) < 1:
        state["running_score"] = 5.0
        return state

    last_answer = user_msgs[-1]["content"]
    job = state["job"] or {}

    eval_prompt = f"""Rate this interview answer for a {job.get('role', 'technical')} role on a scale of 1-10.
Answer: "{last_answer}"
Job requires: {job.get('raw_text', '')[:500]}

Respond with ONLY a number between 1 and 10. Nothing else."""

    try:
        eval_res = await llm.ainvoke([
            SystemMessage(content="You are an interview evaluator. Respond with only a number."),
            HumanMessage(content=eval_prompt)
        ])
        score = float(eval_res.content.strip().split()[0])
        score = max(1.0, min(10.0, score))
        # Running average
        prev_score = state.get("running_score", 5.0)
        q_count = state["question_count"]
        state["running_score"] = (prev_score * q_count + score) / (q_count + 1)
    except:
        state["running_score"] = state.get("running_score", 5.0)

    print(f"[interview:answer_evaluator] running_score={state['running_score']:.1f}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Question Router
# ═══════════════════════════════════════════════════════════════════════════════
def interview_router(state: InterviewState) -> str:
    duration = state.get("duration_minutes", 15)
    # Approx questions for duration (~0.7 per minute)
    max_questions = {5: 4, 10: 7, 15: 10, 20: 14, 30: 20}.get(duration, 10)

    if state["question_count"] >= max_questions:
        return "end"

    score = state.get("running_score", 5.0)
    if score < 4.0:
        return "probe"  # dig deeper on same topic
    else:
        return "next"   # move to next topic

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 6 — Build Interview Context
# ═══════════════════════════════════════════════════════════════════════════════
async def build_interview_context(state: InterviewState) -> InterviewState:
    profile = state["profile"] or {}
    job = state["job"] or {}
    resume = state["resume"] or {}
    skills_gap = state.get("skills_gap") or []
    difficulty = state.get("difficulty", "medium")
    route = interview_router(state)

    resume_json = resume.get("tailored_json", {}) if resume else {}
    skills = ", ".join(profile.get("skills", []) or resume_json.get("skills", []))

    gap_instruction = ""
    if skills_gap:
        gap_instruction = f"\nPRIORITY TOPICS (gaps identified in their application): {', '.join(skills_gap[:5])}\nMake sure to probe these areas."

    route_instruction = {
        "probe": "Their last answer was weak. Ask a follow-up that probes deeper on the same topic. Don't move on yet.",
        "next": f"Move to the next topic. Difficulty level: {difficulty}. Ask something that requires depth.",
        "end": "Wrap up the interview professionally. Thank them and say you'll be in touch.",
    }.get(route, "")

    duration = state.get("duration_minutes", 15)
    max_questions = {5: 4, 10: 7, 15: 10, 20: 14, 30: 20}.get(duration, 10)

    state["interview_context"] = f"""
JOB: {job.get('role', 'Unknown')} at {job.get('company', 'Unknown')}
DESCRIPTION: {(job.get('raw_text') or '')[:1000]}

CANDIDATE SKILLS: {skills}
EXPERIENCE: {profile.get('experience_summary', 'Not provided')}
{gap_instruction}

SESSION: Question {state['question_count'] + 1} of {max_questions} · Difficulty: {difficulty}
INSTRUCTION: {route_instruction}
""".strip()

    print(f"[interview:build_interview_context] route={route} difficulty={difficulty}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 7 — Session Saver
# ═══════════════════════════════════════════════════════════════════════════════
async def interview_session_saver(state: InterviewState) -> InterviewState:
    try:
        # Upsert session
        existing = supabase.from_("interview_sessions").select("id") \
            .eq("id", state["session_id"]).execute()

        if existing.data:
            supabase.from_("interview_sessions").update({
                "messages": state["messages"],
                "score": state.get("running_score"),
            }).eq("id", state["session_id"]).execute()
        else:
            supabase.from_("interview_sessions").insert({
                "id": state["session_id"],
                "user_id": state["user_id"],
                "job_id": state["job_id"],
                "interview_type": "technical",
                "messages": state["messages"],
                "score": state.get("running_score"),
            }).execute()
    except Exception as e:
        print(f"[interview:session_saver] error: {e}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 8 — Interview Responder
# ═══════════════════════════════════════════════════════════════════════════════
async def interview_responder(state: InterviewState) -> InterviewState:
    profile = state["profile"] or {}
    job = state["job"] or {}
    duration = state.get("duration_minutes", 15)
    max_questions = {5: 4, 10: 7, 15: 10, 20: 14, 30: 20}.get(duration, 10)

    system_prompt = INTERVIEW_IDENTITY.format(
        company=job.get("company", "the company"),
        name=profile.get("full_name", "candidate").split()[0],
        role=job.get("role", "this role"),
        max_questions=max_questions,
        interview_context=state.get("interview_context", ""),
    )

    lc_messages = [SystemMessage(content=system_prompt)]
    for m in state["messages"][-12:]:  # last 12 messages for context
        if m["role"] == "user":
            # Replace hidden trigger with actual opening instruction
            
            
            content = m["content"]
            if content == "__ALGO_START__":
                content = (
                    "Greet the candidate by first name only. "
                    "Say your name is ALGO and you're from the company. "
                    "Ask how their day is going — ONE short casual question, max 8 words. "
                    "Total response: 2 sentences max. No fluff, no wishes, no corporate warmth. "
                    "Sound like a real person, not a chatbot."
                )
            lc_messages.append(HumanMessage(content=content))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))
            

    full_response = ""
    async for chunk in llm.astream(lc_messages):
        token = chunk.content
        if token:
            full_response += token

    state["final_response"] = full_response
    print(f"[interview:responder] response_len={len(full_response)}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# Build Interview Graph
# ═══════════════════════════════════════════════════════════════════════════════
def build_interview_graph():
    graph = StateGraph(InterviewState)

    graph.add_node("retrieve_profile", interview_retrieve_profile)
    graph.add_node("load_job_context", load_job_context)
    graph.add_node("load_interview_state", load_interview_state)
    graph.add_node("answer_evaluator", answer_evaluator)
    graph.add_node("build_context", build_interview_context)
    graph.add_node("session_saver", interview_session_saver)
    graph.add_node("responder", interview_responder)

    graph.set_entry_point("retrieve_profile")
    graph.add_edge("retrieve_profile", "load_job_context")
    graph.add_edge("load_job_context", "load_interview_state")
    graph.add_edge("load_interview_state", "answer_evaluator")
    graph.add_edge("answer_evaluator", "build_context")
    graph.add_edge("build_context", "session_saver")
    graph.add_edge("session_saver", "responder")
    graph.add_edge("responder", END)

    return graph.compile()

interview_graph = build_interview_graph()

# ═══════════════════════════════════════════════════════════════════════════════
# /interview endpoint
# ═══════════════════════════════════════════════════════════════════════════════
class InterviewRequest(BaseModel):
    user_id: str
    session_id: str
    job_id: str
    messages: List[dict]
    duration_minutes: Optional[int] = 15
    running_score: Optional[float] = 5.0

@app.post("/interview")
async def interview(req: InterviewRequest):
    if not req.user_id or not req.job_id:
        raise HTTPException(status_code=400, detail="user_id and job_id required")

    
    last_user_msg = next(
    (m["content"] for m in reversed(req.messages) if m["role"] == "user"), ""
    )
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found")

    async def stream_response():
        state: InterviewState = {
            "user_id": req.user_id,
            "session_id": req.session_id,
            "job_id": req.job_id,
            "user_message": last_user_msg,
            "messages": req.messages,
            "profile": None,
            "job": None,
            "resume": None,
            "question_count": 0,
            "duration_minutes": req.duration_minutes or 15,
            "elapsed_seconds": 0,
            "running_score": req.running_score or 5.0,
            "skills_gap": None,
            "last_question_topic": None,
            "difficulty": "medium",
            "interview_context": None,
            "final_response": None,
        }

        final_state = await interview_graph.ainvoke(state)
        final_response = final_state.get("final_response", "")

        # Chunk-based streaming
        CHUNK_SIZE = 12
        for i in range(0, len(final_response), CHUNK_SIZE):
            chunk = final_response[i:i + CHUNK_SIZE]
            yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# /interview/feedback endpoint
# ═══════════════════════════════════════════════════════════════════════════════
class FeedbackRequest(BaseModel):
    user_id: str
    job_id: str
    session_id: str
    messages: List[dict]

@app.post("/interview/feedback")
async def interview_feedback(req: FeedbackRequest):
    try:
        # Fetch job for context
        job_res = supabase.from_("jobs").select("role, company").eq("id", req.job_id).single().execute()
        job = job_res.data or {}

        transcript = "\n\n".join([
            f"[{m['role'].upper()}]: {m['content']}"
            for m in req.messages
            if m.get("content") and m["content"] != "__ALGO_START__"
        ])

        feedback_prompt = f"""You are an expert interview coach. Analyze this {job.get('role', 'technical')} interview transcript for {job.get('company', 'the company')}.

TRANSCRIPT:
{transcript}

Return ONLY valid JSON (no markdown, no extra text):
{{
  "overall_score": <0-100>,
  "overall_verdict": "<one sentence>",
  "sections": [
    {{"category": "Communication", "score": <0-100>, "strength": "<what they did well>", "improvement": "<what to fix>"}},
    {{"category": "Technical Knowledge", "score": <0-100>, "strength": "<what they did well>", "improvement": "<what to fix>"}},
    {{"category": "Confidence & Delivery", "score": <0-100>, "strength": "<what they did well>", "improvement": "<what to fix>"}},
    {{"category": "Relevance of Answers", "score": <0-100>, "strength": "<what they did well>", "improvement": "<what to fix>"}},
    {{"category": "Storytelling & Examples", "score": <0-100>, "strength": "<what they did well>", "improvement": "<what to fix>"}}
  ],
  "top_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "critical_gaps": ["<gap 1>", "<gap 2>", "<gap 3>"],
  "recommended_drills": [
    {{"drill": "<name>", "why": "<reason>", "how": "<practice method>"}}
  ],
  "hire_likelihood": "<Strong Yes | Yes | Maybe | No>",
  "coach_note": "<motivational closing note>"
}}"""

        feedback_llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            streaming=False,
        )

        response = await feedback_llm.ainvoke([
            SystemMessage(content="You are an interview evaluator. Return only valid JSON."),
            HumanMessage(content=feedback_prompt),
        ])

        raw = response.content.strip().replace("```json", "").replace("```", "").strip()
        feedback = json.loads(raw)

        # Save to Supabase
        try:
            supabase.from_("interview_sessions").update({
                "feedback": feedback,
                "completed_at": __import__("datetime").datetime.utcnow().isoformat(),
                "messages": req.messages,
            }).eq("id", req.session_id).execute()
        except Exception as e:
            print(f"[feedback] save error: {e}")

        return {"success": True, "feedback": feedback}

    except Exception as e:
        print(f"[feedback] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)