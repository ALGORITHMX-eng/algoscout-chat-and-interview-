import os
import json
from typing import TypedDict, List, Optional
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

# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    user_id: str
    user_message: str
    messages: List[dict]
    profile: Optional[dict]
    applied_jobs: Optional[List[dict]]
    rejected_jobs: Optional[List[dict]]
    pending_jobs: Optional[List[dict]]
    recent_interviews: Optional[List[dict]]
    conversation_history: Optional[List[dict]]
    resume: Optional[dict]
    experience_tier: Optional[str]
    skills_gap: Optional[List[str]]
    detected_intent: Optional[str]
    career_context: Optional[str]

class ChatRequest(BaseModel):
    user_id: str
    messages: List[dict]

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Profile Memory Retrieval
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
# NODE 2 — Application History Analysis
# ═══════════════════════════════════════════════════════════════════════════════
async def analyze_history(state: AgentState) -> AgentState:
    user_id = state["user_id"]
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
    try:
        res = supabase.from_("coach_conversations").select("role, content") \
            .eq("user_id", user_id).order("created_at", desc=True).limit(20).execute()
        state["conversation_history"] = list(reversed(res.data or []))
    except:
        state["conversation_history"] = []

    all_missing = []
    for job in state["applied_jobs"]:
        if job.get("score_breakdown"):
            try:
                bd = json.loads(job["score_breakdown"]) if isinstance(job["score_breakdown"], str) else job["score_breakdown"]
                all_missing.extend(bd.get("missing_skills", []))
            except:
                pass
    from collections import Counter
    state["skills_gap"] = [s for s, _ in Counter(all_missing).most_common(10)]
    print(f"[node:analyze_history] applied={len(state['applied_jobs'])} pending={len(state['pending_jobs'])}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Career Reasoning
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

    skills = ", ".join(profile.get("skills", [])) or "Not specified"
    target_roles = ", ".join(profile.get("preferred_titles", [])) or "Not specified"
    skills_gap = ", ".join(state["skills_gap"]) if state["skills_gap"] else "None identified"

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
        "learning_path": f"Their missing skills: {skills_gap}. Give a ruthless 30-day plan with specific resources.",
        "salary_negotiation": "Give exact salary ranges for their role and level. Tell them word-for-word what to say.",
        "rejection_support": "One sentence acknowledging it. Then diagnose the real reason using their data. Then 3 specific fixes.",
        "positioning_strategy": "Give a direct verdict — realistic or not. No hedging. Then give the exact positioning angle. Name the frame. Name specific companies to target.",
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

RECURRING SKILLS GAP:
{skills_gap}

INTENT: {intent}
INSTRUCTION: {intent_instructions}
""".strip()

    print(f"[node:career_reasoning] tier={tier} intent={intent}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Tooling Enrichment
# ═══════════════════════════════════════════════════════════════════════════════
async def apply_tooling(state: AgentState) -> AgentState:
    intent = state["detected_intent"]
    if intent == "resume_help" and state["resume"]:
        try:
            skills_in_resume = state["resume"].get("tailored_json", {}).get("skills", [])
            if skills_in_resume:
                state["career_context"] += f"\n\nRESUME SKILLS: {', '.join(skills_in_resume)}"
        except:
            pass
    print(f"[node:apply_tooling] intent={intent}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt — Tight, Identity-First, Few-Shot
# ═══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are ALGO — AlgoScout's sharp, no-BS career strategist. 
You think like a top recruiter and talk like a senior brother who wants the user to win.

{career_context}

━━━ CORE RULES ━━━

1. ONLY USE REAL DATA — Never hallucinate skills, tools, or experience.
2. BE DIRECT — Give clear verdicts. No hedging, no corporate fluff.
3. DELIVER FIRST — Give the main answer/rewrite/verdict immediately. Then offer one specific follow-up.
4. BE CONCISE — Default to short, dense replies.

━━━ EXAMPLES ━━━

User: "Should I apply to this AI Architecture Research role?"
ALGO: "Yes, but only through the production angle. Your RAG + LangGraph experience is strong. Pure research angle is weak — no publications. Position yourself as the guy who actually ships what researchers design. Apply."

User: "Rewrite my summary"
ALGO: "Better:

AI Systems Architect with 2 years building production RAG pipelines and agentic systems. Designed and deployed multiple end-to-end solutions using LangGraph and Groq that achieved sub-2s inference and 99.9% uptime."

Now respond in this exact style — sharp, direct, and useful."""

# ═══════════════════════════════════════════════════════════════════════════════
# API
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

    async def stream_response():
        state: AgentState = {
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
        }

        state = await retrieve_profile(state)
        state = await analyze_history(state)
        state = await career_reasoning(state)
        state = await apply_tooling(state)

        system_prompt = SYSTEM_PROMPT.format(career_context=state["career_context"])

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
                yield f"data: {json.dumps({'choices': [{'delta': {'content': token}}]})}\n\n"

        yield "data: [DONE]\n\n"

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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)