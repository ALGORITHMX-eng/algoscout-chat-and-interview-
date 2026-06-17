import os
import json
import asyncio
import datetime
from typing import TypedDict, List, Optional, Any
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

# ── LLM instances ─────────────────────────────────────────────────────────────
llm_chat = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    streaming=True,
    frequency_penalty=0.8,
    presence_penalty=0.6,
)

llm_interview = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY_INTERVIEW"),
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    streaming=True,
    frequency_penalty=0.8,
    presence_penalty=0.6,
)

llm_fast = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.0,
    streaming=False,
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
    profile_update: Optional[dict]
    final_response: Optional[str]

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    messages: List[dict]

# ── App Navigation ────────────────────────────────────────────────────────────
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
IDENTITY_PROMPT = """You are ALGO — AlgoScout's AI career strategist, built into the app.
You're the sharp friend who knows hiring inside out and knows AlgoScout perfectly.

{app_navigation}

PERSONALITY:
- Casual greeting → greet back warmly with light emoji, ask what's on their mind.
- Direct question → answer it directly. Like a sharp friend, not a corporate bot.
- Venting or frustrated → acknowledge briefly in natural tone (with emoji if it fits), then move straight to useful diagnosis using their real data. Never repeat the same phrase.
- App navigation questions → tell them exactly where to go.

FORMATTING:
- Use **emojis naturally** when it fits (😂, 💪, 😭, 🔥, ✅, 🚀). Max 2-3 per response.
- Match the complexity of the question. Short question = short answer. Big plan request = full structured breakdown.
- Use headers and bullets only when content has 3+ distinct sections...
- For single-topic answers, write flowing prose. No forced structure.
- Emotional replies → plain text only, no formatting whatsoever.
- Bold key terms, scores, company names, and action items when it helps readability.
- Never open with the user's name. Never close with a question.

TONE:
- Lead with the data point, not the feeling. "Your comms score was 20" not "I can see you struggled."
- Call it straight. Bad score = say it's bad.
- Never say: "I think", "great question", "not uncommon", "let's work on this together", "I can see", "I understand."
- No therapy-speak. No corporate softening.
- If they write in pidgin or informal English, match that energy naturally. Don't force it.

NIGERIAN PIDGIN / INFORMAL ENGLISH AWARENESS:
- Words like "jharre", "nah", "oya", "sha", "abeg", "wetin" are emotional fillers or emphasis — NEVER treat them as names or commands.
- Read emotional tone from context, not literal word parsing.

GROUNDING (NON-NEGOTIABLE):
- CANDIDATE section below is the only source of truth. Never invent skills, tools, or experience not listed.
- If something isn't in their profile, say: "I don't have that in your profile — update it in Settings."
- Never recommend on-site US/EU roles to a remote-preference Nigerian candidate unless they explicitly ask.

CAREER RULES:
- Deliver verdict first, follow up second.
- No generic advice. Every sentence must trace to their actual data.
- Write like a sharp person talking, not a consultant delivering a report.

OFF-TOPIC:
- Coding help unrelated to career → "That's outside what I do — try Claude.ai or ChatGPT. Anything career-wise?"
- Everything else: career, app, strategy, conversation — fair game.

SUPPORT EMAIL: algorithmengineer4@gmail.com
Share only if: user asks for contact, reports a bug, or is clearly about to abandon the product.
"""

# ── Updatable profile fields ──────────────────────────────────────────────────
UPDATABLE_FIELDS = {
    "skills":             {"label": "Skills",              "supabase_field": "skills",             "type": "array"},
    "preferred_titles":   {"label": "Target Roles",        "supabase_field": "preferred_titles",    "type": "array"},
    "years_experience":   {"label": "Years of Experience", "supabase_field": "years_experience",    "type": "number"},
    "work_preference":    {"label": "Work Preference",     "supabase_field": "work_preference",     "type": "string"},
    "location":           {"label": "Location",            "supabase_field": "location",            "type": "string"},
    "experience_summary": {"label": "Bio / Summary",       "supabase_field": "experience_summary",  "type": "string"},
    "linkedin":           {"label": "LinkedIn",            "supabase_field": "linkedin",            "type": "string"},
    "github":             {"label": "GitHub",              "supabase_field": "github",              "type": "string"},
    "portfolio":          {"label": "Portfolio URL",       "supabase_field": "portfolio",           "type": "string"},
}

# ── Usage Logger ──────────────────────────────────────────────────────────────
async def log_api_usage(feature: str, model: str, input_tokens: int, output_tokens: int, user_id: str = "system"):
    try:
        await asyncio.to_thread(
            lambda: supabase.from_("api_usage").insert({
                "feature": feature,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "user_id": user_id,
                "created_at": datetime.datetime.utcnow().isoformat(),
            }).execute()
        )
    except Exception as e:
        print(f"[usage_logger] failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Retrieve Profile + Resume
# ═══════════════════════════════════════════════════════════════════════════════
async def retrieve_profile(state: AgentState) -> AgentState:
    user_id = state["user_id"]
    msg = state["user_message"].lower().strip()

    greeting_words = ["hi", "hey", "hello", "what's up", "whats up", "morning", "afternoon", "evening", "yo", "sup"]
    is_greeting = any(msg == w or msg.startswith(w + " ") or msg.startswith(w + "!") for w in greeting_words)
    if is_greeting:
        state["profile"] = {}
        state["resume"] = None
        print(f"[node:retrieve_profile] greeting detected — skipped DB fetch")
        return state

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
# NODE 2 — Analyze History
# FIX: Now fetches interview summaries (not full transcripts) from Supabase
# ═══════════════════════════════════════════════════════════════════════════════
async def analyze_history(state: AgentState) -> AgentState:
    user_id = state["user_id"]

    profile = state.get("profile") or {}
    if not profile and not state.get("user_message", ""):
        state["applied_jobs"] = []
        state["pending_jobs"] = []
        state["session_history"] = []
        state["skills_gap"] = []
        state["previous_conclusions"] = {}
        state["rejected_jobs"] = []
        state["recent_interviews"] = []
        state["profile_update"] = None
        return state

    def fetch_applied():
        try:
            return supabase.from_("jobs").select("company, role, score, score_reason, score_breakdown") \
                .eq("user_id", user_id).eq("status", "approved") \
                .order("found_at", desc=True).limit(5).execute()
        except: return None

    def fetch_pending():
        try:
            return supabase.from_("jobs").select("company, role, score, location") \
                .eq("user_id", user_id).eq("status", "pending").gte("score", 7) \
                .order("score", desc=True).limit(5).execute()
        except: return None

    def fetch_conclusions():
        try:
            return supabase.from_("session_conclusions").select("conclusions") \
                .eq("user_id", user_id).eq("session_id", state["session_id"]).single().execute()
        except: return None

    # FIX: Fetch interview summaries — only score + feedback fields, not full messages
    def fetch_interviews():
        try:
            return supabase.from_("interview_sessions").select(
                "id, created_at, score, feedback"
            ).eq("user_id", user_id).order("created_at", desc=True).limit(3).execute()
        except: return None

    applied_res, pending_res, conclusions_res, interviews_res = await asyncio.gather(
        asyncio.to_thread(fetch_applied),
        asyncio.to_thread(fetch_pending),
        asyncio.to_thread(fetch_conclusions),
        asyncio.to_thread(fetch_interviews),
    )

    state["applied_jobs"] = (applied_res.data if applied_res else None) or []
    state["pending_jobs"] = (pending_res.data if pending_res else None) or []
    state["session_history"] = []
    state["previous_conclusions"] = (conclusions_res.data.get("conclusions", {}) if conclusions_res and conclusions_res.data else {})
    state["rejected_jobs"] = []
    state["profile_update"] = None

    # FIX: Store only compact interview summary — never raw transcript
    raw_interviews = (interviews_res.data if interviews_res else []) or []
    state["recent_interviews"] = [
    {
        "date": s.get("created_at", "")[:10],
        "overall_score": (s.get("feedback") or {}).get("overall_score"),
        "verdict": (s.get("feedback") or {}).get("overall_verdict", ""),
        "hire_likelihood": (s.get("feedback") or {}).get("hire_likelihood"),
        "gaps": (s.get("feedback") or {}).get("critical_gaps", [])[:3],
        "weak_areas": [
            f"{sec['category']}: {sec['score']}/100 — {sec['improvement']}"
            for sec in ((s.get("feedback") or {}).get("sections") or [])
            if sec.get("score", 100) < 60
        ],
    }
    for s in raw_interviews
]

    all_missing = []
    for job in state["applied_jobs"]:
        if job.get("score_breakdown"):
            try:
                bd = json.loads(job["score_breakdown"]) if isinstance(job["score_breakdown"], str) else job["score_breakdown"]
                all_missing.extend(bd.get("missing_skills", []))
            except: pass
    state["skills_gap"] = [s for s, _ in Counter(all_missing).most_common(10)]

    print(f"[node:analyze_history] applied={len(state['applied_jobs'])} interviews={len(state['recent_interviews'])}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Smart Classifier
# FIX: Better Nigerian Pidgin awareness + emotional detection
#      "greeting" only fires on the very first message — not mid-conversation
# ═══════════════════════════════════════════════════════════════════════════════
async def smart_classifier(state: AgentState) -> AgentState:
    msg = state["user_message"]

    # Check if this is truly a first message (no prior assistant turns)
    prior_assistant_msgs = [m for m in state["messages"] if m["role"] == "assistant"]
    is_first_message = len(prior_assistant_msgs) == 0

    classify_prompt = f"""Classify this message into exactly one category.

CRITICAL CONTEXT RULES:
- The user may write in Nigerian Pidgin or informal English
- Words like "jharre", "nah", "oya", "sha", "abeg", "wetin", "e don do" are emotional fillers or emphasis — NEVER names or commands
- Swearing + job/rejection context = "emotional", not "greeting"
- A follow-up message mid-conversation is NEVER "greeting" even if it's short
- This is message #{len(state['messages'])} in the conversation. First assistant message exists: {not is_first_message}

CATEGORIES:
- "greeting" → ONLY if this is the very first message AND it is pure casual small talk with zero career intent (hi, hey, hello, what's up). If there has already been conversation, NEVER use this.
- "emotional" → user is frustrated, venting, swearing, expressing dejection about job search or rejection — includes pidgin expressions like "fuck them jharre", "e don do me like that", "this thing dey pain me"
- "off_topic" → coding help unrelated to career, recipes, weather, sports scores, news
- "profile_update" → wants to add, update, change, or remove something from their profile: skills, target roles, years of experience, work preference, location, bio/summary, LinkedIn, GitHub, portfolio
- "career" → everything else: job questions, resume help, interview prep, salary, skills, app navigation, AlgoScout features, career strategy, rejection analysis

Message: "{msg}"

Reply with ONLY one word: greeting, emotional, off_topic, profile_update, or career"""

    try:
        result = await llm_fast.ainvoke([
            SystemMessage(content="You are a message classifier. Reply with only one word."),
            HumanMessage(content=classify_prompt)
        ])
        category = result.content.strip().lower().split()[0]
        if category in ["greeting", "emotional", "off_topic", "profile_update", "career"]:
            # Extra guard: if not first message, never classify as greeting
            if category == "greeting" and not is_first_message:
                category = "career"
            state["detected_route"] = category
        else:
            state["detected_route"] = "career"
    except:
        state["detected_route"] = "career"

    print(f"[node:smart_classifier] route={state['detected_route']} msg='{msg[:50]}'")
    return state

# ── Router ────────────────────────────────────────────────────────────────────
def router(state: AgentState) -> str:
    return state.get("detected_route", "career")

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Greeting Responder
# ═══════════════════════════════════════════════════════════════════════════════
async def greeting_responder(state: AgentState) -> AgentState:
    profile = state["profile"] or {}
    name = profile.get("full_name", "").split()[0] if profile.get("full_name") else ""

    prompt = f"""You are ALGO — a warm, smart career assistant inside the AlgoScout app.
{"The user's name is " + name + "." if name else ""}
The user just sent a casual greeting or small talk message.
RULES:
- Greet them back warmly and naturally. Use their name if you have it.
- Ask how they're doing or what's on their mind — ONE short question.
- Do NOT dump career data or job listings at them unprompted.
- Do NOT say "I'm here to help with your career" or any corporate opening.
- Max 2 sentences. Sound like a real person."""

    full_response = ""
    async for chunk in llm_fast.astream([SystemMessage(content=prompt), HumanMessage(content=state["user_message"])]):
        if chunk.content:
            full_response += chunk.content

    state["final_response"] = full_response
    asyncio.create_task(log_api_usage("chat_greeting", "llama-3.1-8b-instant", 200, len(full_response) // 4, state["user_id"]))
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Off Topic Rejector (no LLM call)
# ═══════════════════════════════════════════════════════════════════════════════
async def off_topic_rejector(state: AgentState) -> AgentState:
    state["final_response"] = (
        "That's outside what I do — try Claude.ai or ChatGPT for that. "
        "Anything career-wise I can help with?"
    )
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 6 — Profile Update Detector
# ═══════════════════════════════════════════════════════════════════════════════
async def profile_update_detector(state: AgentState) -> AgentState:
    profile = state["profile"] or {}
    msg = state["user_message"]

    extract_prompt = f"""The user wants to update their career profile.
Extract what they want to change and return ONLY valid JSON.

CURRENT PROFILE:
skills: {profile.get('skills', [])}
preferred_titles: {profile.get('preferred_titles', [])}
years_experience: {profile.get('years_experience', 0)}
work_preference: {profile.get('work_preference', '')}
location: {profile.get('location', '')}
experience_summary: {profile.get('experience_summary', '')}
linkedin: {profile.get('linkedin', '')}
github: {profile.get('github', '')}
portfolio: {profile.get('portfolio', '')}

USER MESSAGE: "{msg}"

Return JSON:
{{
  "field": "<one of: skills, preferred_titles, years_experience, work_preference, location, experience_summary, linkedin, github, portfolio>",
  "proposed": <new value — array for skills/preferred_titles, number for years_experience, string for others>,
  "action_type": "<add | remove | replace>",
  "understood": "<one sentence confirming what you understood>"
}}

For "add" to array fields: merge new items with existing.
For "remove" from array fields: remove specified items from existing.
For "replace": use new value directly.
Return ONLY the JSON. No explanation."""

    try:
        result = await llm_fast.ainvoke([
            SystemMessage(content="You extract profile update intentions. Return only valid JSON."),
            HumanMessage(content=extract_prompt)
        ])
        raw = result.content.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        field = parsed.get("field")

        if field not in UPDATABLE_FIELDS:
            state["profile_update"] = None
            return state

        field_meta = UPDATABLE_FIELDS[field]
        current = profile.get(field)
        proposed = parsed.get("proposed")

        if field_meta["type"] == "array":
            current_list = list(current or [])
            if parsed.get("action_type") == "add":
                new_items = proposed if isinstance(proposed, list) else [proposed]
                proposed = list(dict.fromkeys(current_list + new_items))
            elif parsed.get("action_type") == "remove":
                remove_items = proposed if isinstance(proposed, list) else [proposed]
                proposed = [x for x in current_list if x not in remove_items]

        state["profile_update"] = {
            "field": field,
            "supabase_field": field_meta["supabase_field"],
            "field_label": field_meta["label"],
            "current": current,
            "proposed": proposed,
            "understood": parsed.get("understood", ""),
        }
        print(f"[node:profile_update_detector] field={field} proposed={proposed}")
    except Exception as e:
        print(f"[node:profile_update_detector] error: {e}")
        state["profile_update"] = None

    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 7 — Profile Update Responder
# ═══════════════════════════════════════════════════════════════════════════════
async def profile_update_responder(state: AgentState) -> AgentState:
    update = state.get("profile_update")
    if not update:
        state["final_response"] = "I couldn't figure out what you wanted to update. Try being more specific — for example: 'Add React to my skills' or 'Update my location to Lagos'."
        return state

    profile = state["profile"] or {}
    name = profile.get("full_name", "").split()[0] if profile.get("full_name") else ""

    prompt = f"""You are ALGO, a career assistant.
The user wants to update their profile. Confirm what you understood in ONE natural sentence, then say you've prepared the change below for their review.
Say "I've prepared the change for you to review" — NOT "I'll update".
{"User's name is " + name + "." if name else ""}
What they want: {update['understood']}
Field: {update['field_label']}
Max 2 sentences. Sound like a real person."""

    try:
        result = await llm_fast.ainvoke([
            SystemMessage(content="You confirm profile update intentions briefly."),
            HumanMessage(content=prompt)
        ])
        state["final_response"] = result.content.strip()
    except:
        state["final_response"] = f"Got it — I've prepared the change to your {update['field_label']} for you to review below."

    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 8 — Career Reasoning (no LLM)
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
    elif any(w in msg for w in ["where", "how do i", "settings", "profile", "find", "navigate"]):
        intent = "app_navigation"
    else:
        intent = "general_career"
    state["detected_intent"] = intent

    print(f"[node:career_reasoning] tier={tier} intent={intent}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 9 — Resume Grounder
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
# NODE 10 — Apply Tooling
# FIX: Injects interview summary context into career_context
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
        "learning_path": "Pick ONE skill from their gap that matches what companies in their applied jobs actually want. Tell them exactly what to build with it — a specific mini project. One skill, one project, concrete outcome.",
        "salary_negotiation": "Give exact salary ranges for their role and level. Tell them word-for-word what to say.",
        "rejection_support": "One sentence acknowledging it. Then diagnose the real reason using their data. Then 3 specific fixes.",
        "positioning_strategy": "Give a direct verdict. No hedging. Reference session history if one exists.",
        "app_navigation": "Tell them exactly where to go in the app. Be specific — Dashboard, Profile tab, Interview tab, Settings, Add Job button.",
        "general_career": "Answer using their actual data. Reference real skills, companies, scores. No generic advice.",
    }.get(intent, "")

    state["career_context"] = f"""
CANDIDATE:
Name: {profile.get('full_name', 'User')}
Level: {tier}
Location: {profile.get('location', 'Not specified')}
Work Preference: {profile.get('work_preference', 'remote')}
Background: {profile.get('experience_summary', 'Not provided')}

CONFIRMED SKILLS (ONLY these — never add others):
{skills}

TARGET ROLES:
{target_roles}

GEOGRAPHIC CONSTRAINT:
This candidate is based in {profile.get('location', 'Nigeria')}.
Only recommend or discuss roles that are genuinely remote-friendly or match their work preference.
Never suggest on-site US/EU roles unless the candidate explicitly asks.

JOBS APPLIED:
{applied_summary}

HIGH-SCORE PENDING:
{pending_summary}
{skills_gap_text}

INTENT: {intent}
INSTRUCTION: {intent_instructions}
""".strip()

    # FIX: Inject interview summary — compact format, no raw transcript
    if state.get("recent_interviews"):
        interview_lines = "\n".join([
            f"• {i['date']} — Score: {i.get('overall_score', 'N/A')}/100 | "
            f"Gaps: {', '.join((i.get('critical_gaps') or [])[:2])} | "
            f"Hire likelihood: {i.get('hire_likelihood', 'N/A')} | "
            f"Verdict: {i.get('overall_verdict', '')}"
            for i in state["recent_interviews"]
        ])
        state["career_context"] += f"\n\nRECENT INTERVIEW PERFORMANCE (summaries only):\n{interview_lines}"

    if state.get("resume_context"):
        state["career_context"] += f"\n\n{state['resume_context']}"

    if state.get("previous_conclusions") and len(state["previous_conclusions"]) > 0:
        state["career_context"] += "\n\nPREVIOUS CONCLUSIONS THIS SESSION: " + json.dumps(state["previous_conclusions"]) + "\nStay consistent with these."

    print(f"[node:apply_tooling] intent={intent}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 11 — Consistency Checker
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
# NODE 12 — Responder (70b)
# ═══════════════════════════════════════════════════════════════════════════════
async def responder(state: AgentState) -> AgentState:
    system = IDENTITY_PROMPT.format(app_navigation=APP_NAVIGATION)
    lc_messages = [SystemMessage(content=f"{system}\n\n{state['career_context']}")]

    for m in state["messages"][-10:]:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    full_response = ""
    async for chunk in llm_chat.astream(lc_messages):
        if chunk.content:
            full_response += chunk.content

    state["final_response"] = full_response

    input_estimate = sum(len(m.content) for m in lc_messages) // 4
    output_estimate = len(full_response) // 4
    asyncio.create_task(log_api_usage("chat", "llama-3.3-70b-versatile", input_estimate, output_estimate, state["user_id"]))

    print(f"[node:responder] len={len(full_response)}")
    return state

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 13 — Emotional Responder
# FIX: No probing questions. Acknowledge and move. Pidgin-aware.

async def emotional_responder(state: AgentState) -> AgentState:
    profile = state["profile"] or {}
    name = profile.get("full_name", "").split()[0] if profile.get("full_name") else ""
    msg = state["user_message"].lower()

    # Detect "done for today / need a break" signals — don't push, just acknowledge
    done_signals = [
        "i am done", "i'm done", "done for today", "done for now",
        "i need a break", "taking a break", "i'm tired", "i am tired",
        "too tired", "exhausted", "i give up", "giving up", "e don do",
        "i can't anymore", "i cant anymore", "not today", "not doing this today",
        "forget it", "nevermind", "nvm",
    ]
    is_done_signal = any(sig in msg for sig in done_signals)

        if is_done_signal:
        # Pure acknowledgment — no push, no task, no question
        prompt = f"""You are ALGO — a career assistant who actually listens.
{"User's name is " + name + "." if name else ""}
The user is saying they're done for today or need a break from job hunting.

RULES:
- Acknowledge it simply. That's it.
- Do NOT suggest any next steps, tasks, or actions.
- Do NOT ask any questions.
- Do NOT say "I'm here when you're ready" or any variation of that — it's pushy.
- Just let them rest. One short human sentence. Max 10 words.
- Examples of good responses: "Take the rest. You've put in the work." / "Rest up. Job hunting can wait." / "Yeah, step away. It'll still be here."
- Never sound corporate or like a support bot."""

    else:
        # Regular emotional venting
        prompt = f"""You are ALGO — a career assistant who actually cares.
{"User's name is " + name + "." if name else ""}

The user is venting or frustrated about job search (possibly in pidgin).

RULES:
- Acknowledge in one short natural sentence. Sound like a sharp Naija guy.
- Use light emoji where it fits (😂, 😭, 💪, 🔥).
- Be direct. Never say "that one stings" or "that's rough".
- Then one sentence max about next move using their actual data (scores, skills, applied jobs).
- Total max 2 sentences. Match their energy. No questions."""

    full_response = ""
    async for chunk in llm_fast.astream([SystemMessage(content=prompt), HumanMessage(content=state["user_message"])]):
        if chunk.content:
            full_response += chunk.content

    state["final_response"] = full_response
    asyncio.create_task(log_api_usage("chat_emotional", "llama-3.1-8b-instant", 150, len(full_response) // 4, state["user_id"]))
    return state
# ═══════════════════════════════════════════════════════════════════════════════
# Build the Chat Graph
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
    graph.add_node("profile_update_detector", profile_update_detector)
    graph.add_node("profile_update_responder", profile_update_responder)
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
            "profile_update": "profile_update_detector",
            "career": "career_reasoning",
        }
    )

    graph.add_edge("profile_update_detector", "profile_update_responder")
    graph.add_edge("profile_update_responder", END)

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
# Logging
# ═══════════════════════════════════════════════════════════════════════════════
async def log_event(type: str, message: str, source: str, metadata: dict = {}):
    try:
        supabase.from_("monitor_logs").insert({
            "type": type,
            "message": message,
            "source": source,
            "metadata": metadata,
        }).execute()
    except Exception as e:
        print(f"[monitor] failed to log: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# API Routes
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/")
async def root():
    return {"status": "AlgoScout LangGraph backend running — 14 nodes"}


@app.get("/health")
async def health():
    groq_ok = True
    groq_status = "ok"
    supabase_ok = True

    try:
        supabase.from_("profiles").select("id").limit(1).execute()
    except:
        supabase_ok = False
        await log_event("error", "Supabase unreachable", "health_endpoint")

    try:
        import httpx
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=8.0,
        )
        if resp.status_code == 429:
            groq_ok = False
            groq_status = "rate_limited"
            await log_event("warning", "Groq rate limit hit", "health_endpoint")
        elif resp.status_code != 200:
            groq_ok = False
            groq_status = f"error_{resp.status_code}"
            await log_event("error", f"Groq returned {resp.status_code}", "health_endpoint")
    except Exception as e:
        groq_ok = False
        groq_status = "unreachable"
        await log_event("error", f"Groq unreachable: {str(e)}", "health_endpoint")

    status = "ok" if groq_ok and supabase_ok else "degraded"
    return {
        "status": status,
        "groq": groq_ok,
        "groq_status": groq_status,
        "supabase": supabase_ok,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


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
            "profile_update": None,
            "final_response": None,
        }

        final_state = await algo_graph.ainvoke(initial_state)
        final_response = final_state.get("final_response", "")

        CHUNK_SIZE = 12
        for i in range(0, len(final_response), CHUNK_SIZE):
            chunk = final_response[i:i + CHUNK_SIZE]
            yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"

        profile_update = final_state.get("profile_update")
        if profile_update:
            action_payload = {
                "type": "action",
                "action": {
                    "type": "profile_update",
                    "field": profile_update["supabase_field"],
                    "field_label": profile_update["field_label"],
                    "current": profile_update["current"],
                    "proposed": profile_update["proposed"],
                    "summary": profile_update["understood"],
                }
            }
            yield f"data: {json.dumps(action_payload)}\n\n"

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
            await log_event("error", f"Chat save failed: {str(e)}", "chat_endpoint")

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Profile Update Endpoint ───────────────────────────────────────────────────
class ProfileUpdateRequest(BaseModel):
    user_id: str
    field: str
    value: Any

@app.post("/profile/update")
async def update_profile(req: ProfileUpdateRequest):
    valid_fields = [f["supabase_field"] for f in UPDATABLE_FIELDS.values()]
    if req.field not in valid_fields:
        raise HTTPException(status_code=400, detail=f"Field '{req.field}' is not editable via chat")
    try:
        supabase.from_("profiles").update(
            {req.field: req.value}
        ).eq("user_id", req.user_id).execute()
        return {"success": True, "field": req.field, "value": req.value}
    except Exception as e:
        await log_event("error", f"Profile update failed: {str(e)}", "profile_endpoint")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# INTERVIEW GRAPH
# ═══════════════════════════════════════════════════════════════════════════════
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
    difficulty: str
    interview_context: Optional[str]
    final_response: Optional[str]

INTERVIEW_IDENTITY = """You are a senior interviewer at {company}. Your name is ALGO.
You are conducting a {role} interview with {name}.
You are warm but professional — like a real human interviewer who actually enjoys their job.

RULES:
- Ask ONE question at a time. Never two.
- Open with ONE short casual question about their day, max 8 words. No corporate warmth.
- After they answer the small talk, say one short sentence then ask: "Can you tell me about yourself and your experience so far?"
- After they answer that, pick your next question by pulling a specific thread from what they just said.
- Use their skills gap to decide which threads to pull harder on.
- If the answer is weak: probe deeper — "Can you give a specific example of that?"
- If the answer is strong: acknowledge briefly then pivot to a harder related topic.
- Never give hints, coaching, or feedback during the interview.
- Never break character. You are ALGO, a human interviewer.
- Never say you are an AI.
- After {max_questions} questions, close warmly: "Alright {name}, that's everything I needed — we'll be in touch soon. Take care!"

{interview_context}"""

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
    return state

async def load_job_context(state: InterviewState) -> InterviewState:
    try:
        res = supabase.from_("jobs").select("*").eq("id", state["job_id"]).single().execute()
        state["job"] = res.data or {}
    except:
        state["job"] = {}
    job = state["job"]
    skills_gap = []
    if job.get("score_breakdown"):
        try:
            bd = json.loads(job["score_breakdown"]) if isinstance(job["score_breakdown"], str) else job["score_breakdown"]
            skills_gap = bd.get("missing_skills", [])
        except:
            pass
    state["skills_gap"] = skills_gap
    return state

async def load_interview_state(state: InterviewState) -> InterviewState:
    assistant_msgs = [m for m in state["messages"] if m["role"] == "assistant"]
    state["question_count"] = len(assistant_msgs)
    score = state.get("running_score", 5.0)
    if score >= 7.5: state["difficulty"] = "hard"
    elif score >= 5.0: state["difficulty"] = "medium"
    else: state["difficulty"] = "easy"
    return state

async def answer_evaluator(state: InterviewState) -> InterviewState:
    user_msgs = [m for m in state["messages"] if m["role"] == "user"]
    if len(user_msgs) < 1:
        state["running_score"] = 5.0
        return state
    last_answer = user_msgs[-1]["content"]
    job = state["job"] or {}
    eval_prompt = f"""Rate this interview answer for a {job.get('role', 'technical')} role on a scale of 1-10.
Answer: "{last_answer}"
Job requires: {job.get('raw_text', '')[:300]}
Respond with ONLY a number between 1 and 10. Nothing else."""
    try:
        eval_res = await llm_fast.ainvoke([
            SystemMessage(content="You are an interview evaluator. Respond with only a number 1-10."),
            HumanMessage(content=eval_prompt)
        ])
        score = float(eval_res.content.strip().split()[0])
        score = max(1.0, min(10.0, score))
        prev_score = state.get("running_score", 5.0)
        q_count = state["question_count"]
        state["running_score"] = (prev_score * q_count + score) / (q_count + 1)
    except:
        state["running_score"] = state.get("running_score", 5.0)
    return state

def interview_router(state: InterviewState) -> str:
    duration = state.get("duration_minutes", 15)
    max_questions = {5: 4, 10: 7, 15: 10, 20: 14, 30: 20}.get(duration, 10)
    if state["question_count"] >= max_questions: return "end"
    if state.get("running_score", 5.0) < 4.0: return "probe"
    return "next"

async def build_interview_context(state: InterviewState) -> InterviewState:
    profile = state["profile"] or {}
    job = state["job"] or {}
    resume = state["resume"] or {}
    skills_gap = state.get("skills_gap") or []
    difficulty = state.get("difficulty", "medium")
    route = interview_router(state)
    resume_json = resume.get("tailored_json", {}) if resume else {}
    skills = ", ".join(profile.get("skills", []) or resume_json.get("skills", []))
    gap_instruction = f"\nPRIORITY TOPICS (gaps): {', '.join(skills_gap[:5])}\nProbe these areas." if skills_gap else ""
    route_instruction = {
        "probe": "Their last answer was weak. Probe deeper on same topic.",
        "next": f"Move to next topic. Difficulty: {difficulty}. Ask something requiring depth.",
        "end": "Wrap up professionally. Thank them and say you'll be in touch.",
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
    return state

async def interview_session_saver(state: InterviewState) -> InterviewState:
    try:
        existing = supabase.from_("interview_sessions").select("id").eq("id", state["session_id"]).execute()
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
        await log_event("error", f"Session save failed: {str(e)}", "interview_session_saver")
    return state

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
    for m in state["messages"][-12:]:
        if m["role"] == "user":
            content = m["content"]
            if content == "__ALGO_START__":
                content = (
                    "Greet the candidate by first name only. "
                    "Say your name is ALGO and you're from the company. "
                    "Ask how their day is going — ONE short casual question, max 8 words. "
                    "Total response: 2 sentences max. Sound like a real person."
                )
            lc_messages.append(HumanMessage(content=content))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    full_response = ""
    async for chunk in llm_interview.astream(lc_messages):
        if chunk.content:
            full_response += chunk.content

    state["final_response"] = full_response

    input_estimate = sum(len(m.content) for m in lc_messages) // 4
    output_estimate = len(full_response) // 4
    asyncio.create_task(log_api_usage("interview", "llama-3.3-70b-versatile", input_estimate, output_estimate, state["user_id"]))

    return state

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
            "user_id": req.user_id, "session_id": req.session_id,
            "job_id": req.job_id, "user_message": last_user_msg,
            "messages": req.messages, "profile": None, "job": None,
            "resume": None, "question_count": 0,
            "duration_minutes": req.duration_minutes or 15,
            "elapsed_seconds": 0, "running_score": req.running_score or 5.0,
            "skills_gap": None, "last_question_topic": None,
            "difficulty": "medium", "interview_context": None, "final_response": None,
        }
        final_state = await interview_graph.ainvoke(state)
        final_response = final_state.get("final_response", "")
        CHUNK_SIZE = 12
        for i in range(0, len(final_response), CHUNK_SIZE):
            chunk = final_response[i:i + CHUNK_SIZE]
            yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class FeedbackRequest(BaseModel):
    user_id: str
    job_id: str
    session_id: str
    messages: List[dict]

@app.post("/interview/feedback")
async def interview_feedback(req: FeedbackRequest):
    try:
        job_res = supabase.from_("jobs").select("role, company").eq("id", req.job_id).single().execute()
        job = job_res.data or {}
        transcript = "\n\n".join([
            f"[{m['role'].upper()}]: {m['content']}"
            for m in req.messages
            if m.get("content") and m["content"] != "__ALGO_START__"
        ])
        feedback_prompt = f"""You are an expert interview coach. Analyze this {job.get('role', 'technical')} interview for {job.get('company', 'the company')}.
TRANSCRIPT:
{transcript}
Return ONLY valid JSON:
{{
  "overall_score": <0-100>,
  "overall_verdict": "<one sentence>",
  "sections": [
    {{"category": "Communication", "score": <0-100>, "strength": "<str>", "improvement": "<str>"}},
    {{"category": "Technical Knowledge", "score": <0-100>, "strength": "<str>", "improvement": "<str>"}},
    {{"category": "Confidence & Delivery", "score": <0-100>, "strength": "<str>", "improvement": "<str>"}},
    {{"category": "Relevance of Answers", "score": <0-100>, "strength": "<str>", "improvement": "<str>"}},
    {{"category": "Storytelling & Examples", "score": <0-100>, "strength": "<str>", "improvement": "<str>"}}
  ],
  "top_strengths": ["<s1>", "<s2>", "<s3>"],
  "critical_gaps": ["<g1>", "<g2>", "<g3>"],
  "recommended_drills": [{{"drill": "<name>", "why": "<reason>", "how": "<method>"}}],
  "hire_likelihood": "<Strong Yes | Yes | Maybe | No>",
  "coach_note": "<motivational note>"
}}"""

        feedback_llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY_INTERVIEW"),
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

        asyncio.create_task(log_api_usage("interview_feedback", "llama-3.3-70b-versatile", len(feedback_prompt) // 4, len(raw) // 4, req.user_id))

        try:
            supabase.from_("interview_sessions").update({
                "feedback": feedback,
                "completed_at": datetime.datetime.utcnow().isoformat(),
                "messages": req.messages,
            }).eq("id", req.session_id).execute()
        except Exception as e:
            print(f"[feedback] save error: {e}")
            await log_event("error", f"Feedback save failed: {str(e)}", "feedback_endpoint")
        return {"success": True, "feedback": feedback}
    except Exception as e:
        print(f"[feedback] error: {e}")
        await log_event("error", f"Feedback generation failed: {str(e)}", "feedback_endpoint")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Usage Stats Endpoint
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/usage/today")
async def usage_today():
    try:
        today = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        res = supabase.from_("api_usage").select("feature, model, input_tokens, output_tokens, total_tokens") \
            .gte("created_at", today).execute()

        rows = res.data or []
        stats = {}
        for row in rows:
            feature = row["feature"]
            if feature not in stats:
                stats[feature] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            stats[feature]["calls"] += 1
            stats[feature]["input_tokens"] += row.get("input_tokens", 0)
            stats[feature]["output_tokens"] += row.get("output_tokens", 0)
            stats[feature]["total_tokens"] += row.get("total_tokens", 0)

        return {"success": True, "date": today, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)