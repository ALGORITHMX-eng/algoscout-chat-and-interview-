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
    final_response: Optional[str]

# ── Request Model ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    user_id: str
    messages: List[dict]

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

    try:
        applied_res = supabase.from_("jobs") \
            .select("company, role, score, score_reason, score_breakdown, found_at, description") \
            .eq("user_id", user_id).eq("status", "approved") \
            .order("found_at", desc=True).limit(10).execute()
        state["applied_jobs"] = applied_res.data or []
    except:
        state["applied_jobs"] = []

    try:
        rejected_res = supabase.from_("jobs") \
            .select("company, role, score") \
            .eq("user_id", user_id).eq("status", "rejected") \
            .order("found_at", desc=True).limit(10).execute()
        state["rejected_jobs"] = rejected_res.data or []
    except:
        state["rejected_jobs"] = []

    try:
        pending_res = supabase.from_("jobs") \
            .select("company, role, score, location") \
            .eq("user_id", user_id).eq("status", "pending") \
            .gte("score", 7) \
            .order("score", desc=True).limit(5).execute()
        state["pending_jobs"] = pending_res.data or []
    except:
        state["pending_jobs"] = []

    try:
        interview_res = supabase.from_("interview_sessions") \
            .select("interview_type, messages, created_at") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True).limit(5).execute()
        state["recent_interviews"] = interview_res.data or []
    except:
        state["recent_interviews"] = []

    try:
        history_res = supabase.from_("coach_conversations") \
            .select("role, content") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True).limit(20).execute()
        state["conversation_history"] = list(reversed(history_res.data or []))
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
    skill_counts = Counter(all_missing)
    state["skills_gap"] = [skill for skill, _ in skill_counts.most_common(10)]

    print(f"[node:analyze_history] applied={len(state['applied_jobs'])} rejected={len(state['rejected_jobs'])} pending={len(state['pending_jobs'])}")
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
    elif any(w in msg for w in ["reject", "ghosted", "no response", "not hearing"]):
        intent = "rejection_support"
    elif any(w in msg for w in ["position", "realistic", "candidate", "angle", "stretch"]):
        intent = "positioning_strategy"
    else:
        intent = "general_career"

    state["detected_intent"] = intent

    skills = ", ".join(profile.get("skills", [])) or "Not specified"
    target_roles = ", ".join(profile.get("preferred_titles", [])) or "Not specified"
    skills_gap = ", ".join(state["skills_gap"]) if state["skills_gap"] else "None identified yet"

    applied_summary = "\n".join([
        f"• {j['role']} at {j['company']} — score {j['score']}/10"
        for j in state["applied_jobs"]
    ]) or "None yet"

    rejected_summary = "\n".join([
        f"• {j['role']} at {j['company']}"
        for j in state["rejected_jobs"]
    ]) or "None"

    pending_summary = "\n".join([
        f"• {j['role']} at {j['company']} — score {j['score']}/10"
        for j in state["pending_jobs"]
    ]) or "None"

    intent_instruction = {
        "resume_help": "Rewrite their actual bullets now — don't ask if they want you to. Show before/after. Name exactly what was weak and why the new version is stronger.",
        "cover_letter": "Write the cover letter now. Pick the most relevant job from their applied list and write it. Don't ask for permission.",
        "interview_prep": "Give 3 hard, specific questions for their exact role and level. Not soft questions. Answer one yourself as a model answer.",
        "learning_path": f"Top missing skills: {skills_gap}. Give a ruthless 30-day plan. Name specific courses, projects, not vague advice.",
        "salary_negotiation": "Give exact salary ranges for their role and experience. Tell them word-for-word what to say in the negotiation. No hedging.",
        "rejection_support": "One sentence acknowledging the frustration. Then diagnose exactly why they're getting rejected using their actual data. Then 3 specific fixes — not generic tips.",
        "positioning_strategy": "Give a direct verdict — are they a realistic candidate or not. No fence-sitting. Then give the exact positioning angle that maximizes their shot. Name the frame, name the companies to target.",
        "general_career": "Answer directly using their actual profile. No generic advice. Reference their real skills, companies, scores.",
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
Applied:
{applied_summary}

Rejected/Skipped:
{rejected_summary}

High-Score Pending (not acted on):
{pending_summary}

RECURRING SKILLS GAP:
{skills_gap}

DETECTED INTENT: {intent}
COACHING INSTRUCTION: {intent_instruction}
""".strip()

    print(f"[node:career_reasoning] tier={tier} intent={intent}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Tooling Enrichment
# ═══════════════════════════════════════════════════════════════════════════════
async def apply_tooling(state: AgentState) -> AgentState:
    intent = state["detected_intent"]

    if intent == "resume_help" and state["resume"]:
        resume_json = state["resume"].get("tailored_json", {})
        if resume_json:
            skills_in_resume = resume_json.get("skills", [])
            state["career_context"] += f"\n\nCURRENT RESUME SKILLS: {', '.join(skills_in_resume)}"

    if intent == "learning_path" and state["skills_gap"]:
        state["career_context"] += f"\n\nPRIORITY SKILLS TO LEARN: {', '.join(state['skills_gap'][:5])}"

    print(f"[node:apply_tooling] intent={intent}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are AlgoScout Career Coach — a sharp, no-BS AI career strategist.
You think like a top recruiter who has reviewed 10,000 profiles.
You talk like a mentor who respects the user's time and intelligence.
You never waste words and you never pass the ball.

{career_context}

═══ YOUR RULES ═══

1. BRUTAL HONESTY
   If they're not a realistic candidate, say it plainly.
   Never say "you may face challenges" or "it's not impossible."
   Say: "Honestly, you're not there yet for pure research roles — here's exactly why and the fastest path to fix it."

2. ZERO CORPORATE SPEAK
   Banned phrases: "Key Changes:", "Next Action:", "It's worth noting", "certainly", "it's important to",
   "Would you like me to help?", "Let me know if you need anything.", "I hope this helps."
   You own the solution. Always.

3. SHORT AND DENSE
   Max 3 paragraphs unless they explicitly asked for a long output (rewrite, full plan, cover letter).
   Every sentence must earn its place or it gets cut.

4. ACT FIRST, THEN OFFER ONE SPECIFIC FOLLOW-UP
   Always deliver the output first — the rewrite, the verdict, the plan.
   Then end with ONE tightly scoped follow-up offer. Not generic. Tied to what you just did.
   
   Examples:
   - "Want me to make this positioning more aggressive or pull it back?"
   - "Want me to rewrite the other 2 bullets in the same style?"
   - "Want me to generate the cover letter for this same role?"
   - "Want me to turn this into a 30-day execution plan?"
   
   Never: "Let me know if you need anything." 
   Never: "Would you like me to help with something else?"
   The follow-up must be a direct extension of what you just delivered.

5. USE THEIR REAL DATA
   Reference actual companies they applied to, their actual scores, their actual skills.
   If their score at a company was low — say why. If they have a skills gap — name it.
   Generic advice is a coaching failure.

6. POSITIONING VERDICTS
   When asked if they're a realistic candidate: give a direct verdict.
   "Yes, but only via the production angle — the research angle won't land at your level."
   Then give the exact positioning frame. Name it. Don't describe it vaguely.

7. EMOTIONAL INTELLIGENCE
   If they're frustrated: one sentence acknowledging it, then immediately to strategy.
   Don't dwell. Don't over-empathize. They came for strategy, not therapy.

8. TOPIC BOUNDARY
   Career topics only. For anything else: "I only handle career strategy. Try Claude.ai for that 😊"

TONE: Confident. Direct. Warm but not soft. Sharp but not cold.
FORMAT: Use markdown only when it genuinely helps structure (e.g. before/after bullets, a plan). Never use it decoratively."""

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
            "final_response": None,
        }

        state = await retrieve_profile(state)
        state = await analyze_history(state)
        state = await career_reasoning(state)
        state = await apply_tooling(state)

        system_prompt = SYSTEM_PROMPT.format(
            career_context=state["career_context"]
        )

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
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)