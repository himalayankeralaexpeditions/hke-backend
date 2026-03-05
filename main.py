from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, root_validator
from typing import List, Optional, Dict, Any, Union
import os
import json
import re
from datetime import datetime, date

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
# LEADS (BACKWARD + FORWARD COMPATIBLE)
# - Accepts both your older LeadIn shape AND your newer frontend payload
# =========================
class LeadIn(BaseModel):
    # old fields (kept)
    source: str = "website"
    name: Optional[str] = ""
    email: Optional[str] = ""
    phone: Optional[str] = ""
    destination: Optional[str] = ""

    startDate: Optional[str] = None
    endDate: Optional[str] = None
    days: Optional[int] = None
    travellers: Optional[int] = None
    rooms: Optional[int] = None
    hotelClass: Optional[str] = None
    guide: Optional[str] = None
    vehicle: Optional[str] = None
    subDestinations: List[str] = Field(default_factory=list)

    # new frontend payload fields (optional)
    travel_dates: Optional[str] = None
    notes: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"

@app.post("/api/leads")
def create_lead(lead: LeadIn):
    try:
        from google_sheets import insert_lead
        insert_lead(lead.dict())
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lead save failed: {str(e)}")

# =========================
# AI REQUEST (ACCEPT BOTH FRONTEND STYLES)
# - Your backend previously needed startDate (camelCase)
# - Your frontend sometimes sends start_date (snake_case)
# - This model accepts BOTH and normalizes.
# =========================
class AIPlanRequest(BaseModel):
    # required-ish
    destination: str
    days: int

    # accept BOTH styles
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    travellers: int = 2
    rooms: int = 1

    # accept BOTH styles
    hotelClass: str = "Standard"
    hotel_category: Optional[str] = None
    vehicle: str = "SUV"
    guide: str = "Without Guide"

    # route fields
    startPoint: Optional[str] = ""
    endPoint: Optional[str] = ""
    start_point: Optional[str] = ""
    end_point: Optional[str] = ""

    notes: Optional[str] = ""

    # places can come as:
    # - interests: list or comma string (old)
    # - places: list (new)
    # - subDestinations: list (some older UI)
    interests: Union[List[str], str, None] = Field(default_factory=list)
    places: Optional[List[str]] = None
    subDestinations: Optional[List[str]] = None

    budget: Optional[str] = "standard"

    @root_validator(pre=True)
    def _normalize_keys(cls, values: Dict[str, Any]):
        # If frontend sends camelCase only, keep.
        # If frontend sends snake_case, map to camelCase fields used internally.
        sd = values.get("startDate") or values.get("start_date")
        ed = values.get("endDate") or values.get("end_date")
        if sd and not values.get("startDate"):
            values["startDate"] = sd
        if ed and not values.get("endDate"):
            values["endDate"] = ed

        sp = values.get("startPoint") or values.get("start_point")
        ep = values.get("endPoint") or values.get("end_point")
        if sp and not values.get("startPoint"):
            values["startPoint"] = sp
        if ep and not values.get("endPoint"):
            values["endPoint"] = ep

        hc = values.get("hotelClass") or values.get("hotel_category")
        if hc and not values.get("hotelClass"):
            values["hotelClass"] = hc

        # places: unify from places/subDestinations/interests
        if values.get("places") is None:
            if isinstance(values.get("subDestinations"), list) and values["subDestinations"]:
                values["places"] = values["subDestinations"]
            elif isinstance(values.get("interests"), list) and values["interests"]:
                values["places"] = values["interests"]
            elif isinstance(values.get("interests"), str) and values["interests"].strip():
                values["places"] = [x.strip() for x in values["interests"].split(",") if x.strip()]

        return values

# Response supports OLD frontend (itinerary) AND NEW (itineraryJson / itineraryText)
class AIPlanResponse(BaseModel):
    itinerary: str
    itineraryText: Optional[str] = None
    itineraryJson: Optional[Dict[str, Any]] = None

# =========================
# HELPERS
# =========================
def _normalize_places_from_req(req: AIPlanRequest) -> List[str]:
    if req.places and isinstance(req.places, list):
        return [str(x).strip() for x in req.places if str(x).strip()]

    # fallback to interests
    if req.interests is None:
        return []
    if isinstance(req.interests, list):
        return [str(x).strip() for x in req.interests if str(x).strip()]
    if isinstance(req.interests, str):
        return [x.strip() for x in req.interests.split(",") if x.strip()]
    return []

def _phone_10_digits(phone: str) -> str:
    digits = re.sub(r"\D+", "", str(phone or ""))
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    return digits

def _parse_any_date(s: Optional[str]) -> Optional[date]:
    """
    Accepts:
    - YYYY-MM-DD (from <input type="date">)
    - DD-MM-YYYY (your old format)
    - DD/MM/YYYY
    """
    if not s:
        return None
    s = str(s).strip()

    # YYYY-MM-DD
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        pass

    # DD-MM-YYYY
    try:
        return datetime.strptime(s, "%d-%m-%Y").date()
    except Exception:
        pass

    # DD/MM/YYYY
    try:
        return datetime.strptime(s, "%d/%m/%Y").date()
    except Exception:
        pass

    return None

def _fmt_date(d: Optional[date]) -> str:
    return d.strftime("%Y-%m-%d") if d else ""

def _safe_route(req: AIPlanRequest) -> str:
    sp = (req.startPoint or "").strip()
    ep = (req.endPoint or "").strip()
    if sp and ep:
        return f"{sp} → {ep}"
    if sp:
        return f"{sp} → (end as per plan)"
    if ep:
        return f"(start as per plan) → {ep}"
    return "Route as per plan"

# =========================
# NEW: SENSIBLE, OPERABLE ITINERARY PROMPT (JSON OUTPUT)
# =========================
SYSTEM_PROMPT = """
You are a senior India tour operations planner for Himalayan Kerala Expeditions.
Your job is to create REALISTIC, OPERABLE itineraries (not generic travel content).
Always prioritize practicality: routing, time, fatigue, check-in/out, buffer time.

Rules:
- Use the customer's Start Point and End Point if provided.
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
- If a place cannot fit practically, move it to “optional_if_time”.
- Keep the plan "Standard" quality unless user chose Premium/Budget.
- End the plan with standard includes/excludes and practical notes.

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
    places = _normalize_places_from_req(req)
    places_text = ", ".join(places) if places else "Use best highlights for the destination"

    sd = _parse_any_date(req.startDate)
    ed = _parse_any_date(req.endDate)

    # If endDate missing, derive it from startDate + (days-1)
    if sd and not ed and req.days and req.days > 0:
        try:
            ed = sd.fromordinal(sd.toordinal() + max(req.days - 1, 0))
        except Exception:
            ed = None

    # normalize phone if present (not required here, but useful for ops)
    # (not forcing required to avoid 422 for phone)
    return f"""
Customer details:
Destination: {req.destination}
Trip Start Point: {(req.startPoint or "").strip() or "Not provided"}
Trip End Point: {(req.endPoint or "").strip() or "Not provided"}
Start Date: {_fmt_date(sd) or (req.startDate or "")}
End Date: {_fmt_date(ed) or (req.endDate or "")}
Days: {req.days}
Travellers: {req.travellers}
Rooms: {req.rooms}
Hotel Category: {req.hotelClass}
Vehicle: {req.vehicle}
Guide: {req.guide}
Selected Places (must include): {places_text}
Customer Notes: {(req.notes or "None").strip()}

Create a practical day-wise plan following all rules. Return JSON only.
""".strip()

def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()

    # Sometimes model may wrap JSON in ```json ... ```
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```$", "", t).strip()

    try:
        return json.loads(t)
    except Exception:
        return None

def _json_to_pretty_text(data: Dict[str, Any]) -> str:
    # A readable WhatsApp-friendly text for book.html fallback
    # (book.html can parse text into days; but we also provide JSON)
    lines = []
    title = data.get("title") or "Trip Plan"
    summary = data.get("summary") or ""
    lines.append(f"{title}")
    if summary:
        lines.append(summary)
    lines.append("")

    for d in data.get("day_wise", []) or []:
        day = d.get("day", "")
        st = d.get("start_time", "")
        fr = d.get("from", "")
        to = d.get("to", "")
        drive = d.get("drive_time", "")
        lines.append(f"Day {day} — {st} | {fr} → {to} | Drive: {drive}".strip())
        for p in d.get("plan", []) or []:
            lines.append(f"- {p}")
        mb = d.get("meals_breaks", []) or []
        if mb:
            lines.append("Meals/Breaks:")
            for x in mb:
                lines.append(f"- {x}")
        ns = d.get("night_stay", "")
        if ns:
            lines.append(f"Night stay: {ns}")
        lines.append("")

    opt = data.get("optional_if_time", []) or []
    if opt:
        lines.append("Optional if time permits:")
        for x in opt:
            lines.append(f"- {x}")
        lines.append("")

    inc = data.get("package_includes", []) or []
    exc = data.get("package_excludes", []) or []
    notes = data.get("notes", []) or []

    if inc:
        lines.append("PACKAGE INCLUDES:")
        for x in inc:
            lines.append(f"- {x}")
        lines.append("")
    if exc:
        lines.append("PACKAGE EXCLUDES:")
        for x in exc:
            lines.append(f"- {x}")
        lines.append("")
    if notes:
        lines.append("NOTES:")
        for x in notes:
            lines.append(f"- {x}")
        lines.append("")

    return "\n".join(lines).strip()

# =========================
# AI GENERATE (MAIN)
# =========================
@app.post("/api/ai/plan", response_model=AIPlanResponse)
def generate_itinerary(req: AIPlanRequest):
    # basic validations to reduce junk responses
    if not req.destination or not str(req.destination).strip():
        raise HTTPException(status_code=422, detail="destination is required")
    if not req.days or int(req.days) < 2:
        raise HTTPException(status_code=422, detail="days must be at least 2")

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

        itinerary_json = _try_parse_json(text)

        # If JSON parsing fails, try to “recover” by asking model to output JSON only (1 quick retry)
        if itinerary_json is None:
            retry = client.responses.create(
                model="gpt-4.1-mini",
                instructions=SYSTEM_PROMPT + "\n\nIMPORTANT: Output MUST be valid JSON only. No markdown. No extra text.",
                input=user_prompt,
                max_output_tokens=1400,
            )
            text2 = (retry.output_text or "").strip()
            itinerary_json = _try_parse_json(text2)
            text = text2 if text2 else text

        # If still not JSON, return plain itinerary for backward compatibility
        if itinerary_json is None:
            return {
                "itinerary": text,          # old field (kept)
                "itineraryText": text,      # new friendly field
                "itineraryJson": None
            }

        # Convert JSON into pretty text too (book.html can show it nicely)
        pretty = _json_to_pretty_text(itinerary_json)

        return {
            "itinerary": pretty,           # keep old key working
            "itineraryText": pretty,
            "itineraryJson": itinerary_json
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generate failed: {str(e)}")

# ✅ Alias for your frontend
@app.post("/api/ai/itinerary", response_model=AIPlanResponse)
def generate_itinerary_alias(req: AIPlanRequest):
    return generate_itinerary(req)

# =========================
# AI CHAT (EDIT ITINERARY)
# - Accepts either itineraryText or itineraryJson converted to text
# =========================
class AIChatRequest(BaseModel):
    current_itinerary: str
    user_message: str

class AIChatResponse(BaseModel):
    itinerary: str
    itineraryText: Optional[str] = None
    itineraryJson: Optional[Dict[str, Any]] = None

@app.post("/api/ai/chat", response_model=AIChatResponse)
def chat_modify_itinerary(req: AIChatRequest):
    cur = (req.current_itinerary or "").strip()
    msg = (req.user_message or "").strip()

    if not cur:
        raise HTTPException(status_code=422, detail="current_itinerary is required")
    if not msg:
        raise HTTPException(status_code=422, detail="user_message is required")

    prompt = f"""
You will update an existing itinerary plan based on the user's request.

CURRENT ITINERARY (may be text converted from JSON):
{cur}

USER REQUEST:
{msg}

Rules:
- Keep it practical and realistic (commonsense routing, drive times, rest).
- Output ONLY JSON in the same strict schema as before.
""".strip()

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            instructions=SYSTEM_PROMPT,
            input=prompt,
            max_output_tokens=1400,
        )

        text = (resp.output_text or "").strip()
        if not text:
            raise HTTPException(status_code=500, detail="OpenAI returned empty updated itinerary")

        itinerary_json = _try_parse_json(text)
        if itinerary_json is None:
            # fallback: just return the text if parsing fails
            return {"itinerary": text, "itineraryText": text, "itineraryJson": None}

        pretty = _json_to_pretty_text(itinerary_json)
        return {"itinerary": pretty, "itineraryText": pretty, "itineraryJson": itinerary_json}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI chat failed: {str(e)}")

# =========================
# FINALIZE (RECEIVE ANYTHING)
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
# ✅ CUSTOMER CARE CHATBOT (SUPPORT ONLY)
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
