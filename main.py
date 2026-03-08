from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import os
import json

from openai import OpenAI

# =========================
# APP INIT
# =========================
app = FastAPI(title="HKE Backend – AI Trip Planner & Leads", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# OPENAI CLIENT
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# ROOT + HEALTH
# =========================
@app.get("/")
def root():
    return {"ok": True, "service": "HKE Backend Running"}

@app.get("/health")
def health():
    return {"ok": True}

# =========================
# LEADS (KEEP)
# =========================
class LeadIn(BaseModel):
    source: str = "website"
    name: str
    email: str
    phone: str
    destination: str
    startDate: str
    endDate: str
    days: int
    travellers: int
    rooms: int
    hotelClass: str
    guide: str
    vehicle: str
    subDestinations: List[str] = []

@app.post("/api/leads")
def create_lead(lead: LeadIn):
    try:
        from google_sheets import insert_lead
        insert_lead(lead.dict())
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lead save failed: {str(e)}")

# =========================
# AI REQUEST (ACCEPT BOTH camelCase + snake_case)
# =========================
class AIPlanRequest(BaseModel):
    destination: str
    days: int

    startDate: str = Field(..., alias="start_date")
    travellers: int = 2
    rooms: int = 1

    hotelClass: str = Field("Standard", alias="hotel_category")
    vehicle: str = "SUV"
    guide: str = "Without Guide"

    startPoint: Optional[str] = Field("", alias="start_point")
    endPoint: Optional[str] = Field("", alias="end_point")
    notes: Optional[str] = ""

    # frontend sends interests: [..] ; accept also "places"
    interests: Union[List[str], str, None] = Field(default_factory=list, alias="places")

    class Config:
        populate_by_name = True  # allows sending both alias & field name

class AIPlanResponse(BaseModel):
    itineraryText: str
    itineraryJson: Dict[str, Any]

def _normalize_places(interests: Union[List[str], str, None]) -> List[str]:
    if interests is None:
        return []
    if isinstance(interests, list):
        return [str(x).strip() for x in interests if str(x).strip()]
    if isinstance(interests, str):
        return [x.strip() for x in interests.split(",") if x.strip()]
    return []

SYSTEM_PROMPT = """
You are a senior India tour operations planner for Himalayan Kerala Expeditions.
Create REALISTIC, OPERABLE itineraries (not generic travel content).
Always prioritize practicality: routing, time, fatigue, check-in/out, buffer time.

Rules:
- Use the customer's Start Point and End Point.
- Use the customer's selected places as MUST-include (as many as possible).
- Do NOT invent airports/railways unless user explicitly said it.
- Do NOT write marketing fluff.
- Every day must have:
  1) Start time
  2) Route (From -> To)
  3) Approx drive time range
  4) Sightseeing (max 2-3 major + 1-2 minor)
  5) Meal/rest stops (short)
  6) Night stay location (city)
- Keep driving realistic:
  - Hills: average 25–35 km/h
  - Plains: 45–60 km/h
  - Avoid > 8 hrs hills drive unless it's a transfer day and mention fatigue.
- Add 30–60 min buffer every half day.
- If day count is small, do NOT over-pack. Keep it comfortable.
- If a place cannot fit practically, move it to “Optional if time permits”.
- End the plan with:
  - Package Includes (standard)
  - Package Excludes (standard)
  - Notes (weather, local rules, best time)

Output format STRICT:
Return ONLY JSON with keys:
{
 "title": "...",
 "summary": "...",
 "route_overview": ["Day 1: ...", "Day 2: ..."],
 "day_wise": [
   {
     "day": 1,
     "start_time": "08:00",
     "from": "...",
     "to": "...",
     "drive_time": "X–Y hrs",
     "plan": ["...", "..."],
     "meals_breaks": ["...", "..."],
     "night_stay": "..."
   }
 ],
 "optional_if_time": ["...", "..."],
 "package_includes": ["..."],
 "package_excludes": ["..."],
 "notes": ["..."]
}
""".strip()

def _build_user_prompt(req: AIPlanRequest) -> str:
    places = _normalize_places(req.interests)
    places_text = ", ".join(places) if places else "No specific places selected"

    return f"""
Customer details:
Destination: {req.destination}
Trip Start Point: {req.startPoint}
Trip End Point: {req.endPoint}
Start Date: {req.startDate}
Days: {req.days}
Travellers: {req.travellers}
Rooms: {req.rooms}
Hotel Category: {req.hotelClass}
Vehicle: {req.vehicle}
Guide: {req.guide}
Selected Places (must include): {places_text}
Customer Notes: {req.notes or "None"}

Create a practical day-wise plan following all rules. Return JSON only.
""".strip()

def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None

def _json_to_text(j: Dict[str, Any]) -> str:
    # readable WhatsApp-friendly text from JSON
    title = j.get("title", "Trip Plan")
    summary = j.get("summary", "")
    days = j.get("day_wise", [])
    opt = j.get("optional_if_time", [])
    inc = j.get("package_includes", [])
    exc = j.get("package_excludes", [])
    notes = j.get("notes", [])

    parts = [f"{title}".strip()]
    if summary:
        parts.append(summary.strip())
    parts.append("")

    for d in days:
        day = d.get("day", "")
        st = d.get("start_time", "")
        frm = d.get("from", "")
        to = d.get("to", "")
        drive = d.get("drive_time", "")
        parts.append(f"Day {day} — {frm} → {to} | Start {st} | Drive {drive}".strip())
        for p in d.get("plan", [])[:10]:
            parts.append(f"• {p}")
        mb = d.get("meals_breaks", [])
        if mb:
            parts.append("Meals / breaks:")
            for m in mb[:6]:
                parts.append(f"• {m}")
        ns = d.get("night_stay", "")
        if ns:
            parts.append(f"Night stay: {ns}")
        parts.append("")

    if opt:
        parts.append("Optional if time permits:")
        for o in opt[:10]:
            parts.append(f"• {o}")
        parts.append("")

    parts.append("PACKAGE INCLUDES:")
    for x in inc[:10]:
        parts.append(f"• {x}")
    parts.append("")

    parts.append("PACKAGE EXCLUDES:")
    for x in exc[:12]:
        parts.append(f"• {x}")
    parts.append("")

    if notes:
        parts.append("NOTES:")
        for n in notes[:10]:
            parts.append(f"• {n}")

    return "\n".join(parts).strip()

@app.post("/api/ai/plan", response_model=AIPlanResponse)
def generate_itinerary(req: AIPlanRequest):
    user_prompt = _build_user_prompt(req)

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
            max_output_tokens=1400,
        )
        text = (resp.output_text or "").strip()
        if not text:
            raise HTTPException(status_code=500, detail="OpenAI returned empty itinerary")

        it_json = _safe_json_load(text)

        # If model returned text (rare), fallback into a minimal JSON wrapper
        if not it_json or "day_wise" not in it_json:
            it_json = {
                "title": f"{req.destination} {req.days}D Plan",
                "summary": "Plan generated. Please review with HKE team for final confirmation.",
                "route_overview": [],
                "day_wise": [],
                "optional_if_time": [],
                "package_includes": [
                    "Accommodation as per selected category (final confirmation)",
                    "Private vehicle with driver as per itinerary",
                    "Sightseeing as per route (time & weather permitting)",
                    "Support from HKE team before and during travel"
                ],
                "package_excludes": [
                    "Meals unless included by selected hotel plan",
                    "Entry fees, activities, permits",
                    "Personal expenses and tips",
                    "Anything not mentioned under Includes"
                ],
                "notes": ["This is an AI plan. Final routing may change due to weather/road conditions."]
            }

        # Also provide a readable itineraryText
        itinerary_text = _json_to_text(it_json)

        return {
            "itineraryText": itinerary_text,
            "itineraryJson": it_json
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generate failed: {str(e)}")

# ✅ Alias
@app.post("/api/ai/itinerary", response_model=AIPlanResponse)
def generate_itinerary_alias(req: AIPlanRequest):
    return generate_itinerary(req)

# =========================
# AI CHAT (EDIT ITINERARY) - KEEP SIMPLE
# =========================
class AIChatRequest(BaseModel):
    current_itinerary: str
    user_message: str

class AIChatResponse(BaseModel):
    itinerary: str

@app.post("/api/ai/chat", response_model=AIChatResponse)
def chat_modify_itinerary(req: AIChatRequest):
    prompt = f"""
You are improving an existing itinerary to be MORE PRACTICAL.
Avoid generic lines. Keep route realistic and comfortable.

CURRENT ITINERARY:
{req.current_itinerary}

USER REQUEST:
{req.user_message}

Return the FULL UPDATED itinerary in clean day-wise format.
""".strip()

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=900,
        )
        text = (resp.output_text or "").strip()
        if not text:
            raise HTTPException(status_code=500, detail="OpenAI returned empty updated itinerary")
        return {"itinerary": text}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI chat failed: {str(e)}")

# =========================
# FINALIZE (KEEP)
# =========================
class FinalizeRequest(BaseModel):
    itinerary: str
    context: Dict[str, Any] = Field(default_factory=dict)

@app.post("/api/ai/finalize")
def finalize_itinerary(req: FinalizeRequest):
    return {
        "ok": True,
        "message": "Finalized (stored in browser for next page).",
        "itinerary": req.itinerary,
        "context": req.context,
    }

# =========================
# CUSTOMER CARE CHATBOT (KEEP)
# =========================
class SupportChatRequest(BaseModel):
    message: str

class SupportChatResponse(BaseModel):
    reply: str

@app.post("/api/support/chat", response_model=SupportChatResponse)
def customer_care_chat(req: SupportChatRequest):
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")

    instructions = """
You are the CUSTOMER CARE assistant for Himalayan Kerala Expeditions (HKE).
This chatbot is ONLY for customer support issues:
- payment issues (UPI/QR/transaction pending/failed)
- booking status / confirmation
- cancellation / reschedule process
- refund timelines
- pickup timing / coordination questions
- general assistance
- connect to human agent

STRICT RULES:
- DO NOT create itineraries and DO NOT sell packages here.
- If user asks for itinerary/package/plan, reply: "Please use Plan with AI page for itinerary."
- Be concise and helpful.
- Ask at most ONE follow-up question if required (name/phone/date/UTR).
- If user asks "talk to agent" / "human", immediately give contact details.

HKE Contact:
WhatsApp: +91 97972 94747
Phone: +91 97972 94747
Email: himalayankeralaexpeditions@gmail.com
""".strip()

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            instructions=instructions,
            input=msg,
            max_output_tokens=250,
        )
        reply = (resp.output_text or "").strip()
        if not reply:
            reply = "Please share your issue in 1 line. For urgent help WhatsApp: +91 97972 94747"
        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Support chat failed: {str(e)}")
