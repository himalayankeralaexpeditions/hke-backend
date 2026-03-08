from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import os
import json
import requests
import hmac
import hashlib

from openai import OpenAI

# =========================
# APP INIT
# =========================
app = FastAPI(title="HKE Backend – AI Trip Planner & Leads", version="2.2.0")

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
# RAZORPAY ENV
# =========================
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")

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
# LEADS
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
# AI REQUEST / RESPONSE
# Accept both camelCase and snake_case from frontend
# =========================
class AIPlanRequest(BaseModel):
    destination: str
    days: int

    # start / end dates
    startDate: str = Field(..., alias="start_date")
    endDate: Optional[str] = Field(default="", alias="end_date")

    # pax / rooms
    travellers: int = 2
    rooms: int = 1

    # trip config
    hotelClass: str = Field("Standard", alias="hotel_category")
    vehicle: str = "SUV"
    guide: str = "Without Guide"

    # route
    startPoint: Optional[str] = Field("", alias="start_point")
    endPoint: Optional[str] = Field("", alias="end_point")

    # customer note
    notes: Optional[str] = ""

    # places list
    interests: Union[List[str], str, None] = Field(default_factory=list, alias="places")

    class Config:
        populate_by_name = True


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
Create REALISTIC, OPERABLE itineraries, not generic travel fluff.

Always prioritize:
- route practicality
- real drive times
- fatigue management
- check-in / check-out feasibility
- buffer time
- commonsense sightseeing order

Rules:
- Use the customer's Start Point and End Point.
- Use selected places as MUST-include where practical.
- Do NOT invent airports, railways, flights, or transfers unless user explicitly mentioned them.
- Do NOT overpack the trip.
- Every day must contain:
  1) start time
  2) from
  3) to
  4) realistic drive time
  5) practical sightseeing steps
  6) short meals / breaks
  7) night stay
- Hill driving: approx 25–35 km/h average
- Plains driving: approx 45–60 km/h average
- Avoid > 8 hrs hill drive unless unavoidable, and mention it as a transfer-heavy day.
- If something cannot fit, move it into optional_if_time.
- Output only valid JSON.
- No markdown, no code block, no extra explanation outside JSON.

STRICT JSON FORMAT:
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
  "notes": ["...", "..."]
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
End Date: {req.endDate}
Days: {req.days}
Travellers: {req.travellers}
Rooms: {req.rooms}
Hotel Category: {req.hotelClass}
Vehicle: {req.vehicle}
Guide: {req.guide}
Selected Places (must include): {places_text}
Customer Notes: {req.notes or "None"}

Create a practical, comfortable, sellable day-wise plan following all rules.
Return JSON only.
""".strip()


def _safe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def _fallback_itinerary_json(req: AIPlanRequest) -> Dict[str, Any]:
    places = _normalize_places(req.interests)
    return {
        "title": f"{req.destination} {req.days} Day Plan",
        "summary": "Practical itinerary prepared by HKE AI. Final routing may adjust slightly based on weather, roads, hotel availability and local conditions.",
        "route_overview": [
            f"Day 1: Arrival and local movement from {req.startPoint or 'start point'}",
            f"Day {req.days}: Return towards {req.endPoint or 'end point'}"
        ],
        "day_wise": [],
        "optional_if_time": places[:3],
        "package_includes": [
            "Accommodation as per selected category (final confirmation at booking stage)",
            "Private vehicle with driver as per itinerary",
            "Sightseeing as per route (time and weather permitting)",
            "Support from HKE team before and during travel"
        ],
        "package_excludes": [
            "Meals unless included in selected hotel plan",
            "Entry fees, activities and permits",
            "Personal expenses, tips and shopping",
            "Anything not mentioned under Includes"
        ],
        "notes": [
            "This is an AI-generated draft itinerary.",
            "Final routing may change due to weather, road, traffic or local restrictions."
        ]
    }


def _json_to_text(j: Dict[str, Any]) -> str:
    title = j.get("title", "Trip Plan")
    summary = j.get("summary", "")
    days = j.get("day_wise", [])
    opt = j.get("optional_if_time", [])
    inc = j.get("package_includes", [])
    exc = j.get("package_excludes", [])
    notes = j.get("notes", [])

    parts = [title]
    if summary:
        parts.append(summary)
    parts.append("")

    for d in days:
        parts.append(
            f"Day {d.get('day','')} — {d.get('from','')} → {d.get('to','')} | "
            f"Start {d.get('start_time','')} | Drive {d.get('drive_time','')}"
        )
        for p in d.get("plan", []):
            parts.append(f"• {p}")
        if d.get("meals_breaks"):
            parts.append("Meals / breaks:")
            for m in d.get("meals_breaks", []):
                parts.append(f"• {m}")
        if d.get("night_stay"):
            parts.append(f"Night stay: {d.get('night_stay')}")
        parts.append("")

    if opt:
        parts.append("Optional if time permits:")
        for x in opt:
            parts.append(f"• {x}")
        parts.append("")

    parts.append("PACKAGE INCLUDES:")
    for x in inc:
        parts.append(f"• {x}")
    parts.append("")

    parts.append("PACKAGE EXCLUDES:")
    for x in exc:
        parts.append(f"• {x}")
    parts.append("")

    if notes:
        parts.append("NOTES:")
        for x in notes:
            parts.append(f"• {x}")

    return "\n".join(parts).strip()


# =========================
# AI GENERATE
# =========================
@app.post("/api/ai/plan", response_model=AIPlanResponse)
def generate_itinerary(req: AIPlanRequest):
    prompt = _build_user_prompt(req)

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            instructions=SYSTEM_PROMPT,
            input=prompt,
            max_output_tokens=1400,
        )

        text = (resp.output_text or "").strip()
        if not text:
            raise HTTPException(status_code=500, detail="OpenAI returned empty itinerary")

        itinerary_json = _safe_json_load(text)

        if not itinerary_json or "day_wise" not in itinerary_json:
            itinerary_json = _fallback_itinerary_json(req)

        # Ensure required sections exist even if model omits them
        itinerary_json.setdefault("title", f"{req.destination} {req.days} Day Plan")
        itinerary_json.setdefault("summary", "Practical itinerary prepared by HKE AI.")
        itinerary_json.setdefault("route_overview", [])
        itinerary_json.setdefault("day_wise", [])
        itinerary_json.setdefault("optional_if_time", [])
        itinerary_json.setdefault("package_includes", [
            "Accommodation as per selected category (final confirmation at booking stage)",
            "Private vehicle with driver as per itinerary",
            "Sightseeing as per route (time and weather permitting)",
            "Support from HKE team before and during travel"
        ])
        itinerary_json.setdefault("package_excludes", [
            "Meals unless included in selected hotel plan",
            "Entry fees, activities and permits",
            "Personal expenses, tips and shopping",
            "Anything not mentioned under Includes"
        ])
        itinerary_json.setdefault("notes", [
            "Final routing may change due to weather, road or local conditions."
        ])

        itinerary_text = _json_to_text(itinerary_json)

        return {
            "itineraryText": itinerary_text,
            "itineraryJson": itinerary_json
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generate failed: {str(e)}")


@app.post("/api/ai/itinerary", response_model=AIPlanResponse)
def generate_itinerary_alias(req: AIPlanRequest):
    return generate_itinerary(req)


# =========================
# AI CHAT (EDIT ITINERARY)
# =========================
class AIChatRequest(BaseModel):
    current_itinerary: str
    user_message: str

class AIChatResponse(BaseModel):
    itinerary: str

@app.post("/api/ai/chat", response_model=AIChatResponse)
def chat_modify_itinerary(req: AIChatRequest):
    prompt = f"""
You are improving an existing itinerary to be more practical and route sensible.
Avoid generic fluff. Keep route realistic, comfortable and sellable.

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
# FINALIZE
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
# RAZORPAY ORDER + VERIFY
# =========================
class RazorpayOrderIn(BaseModel):
    amount_inr: int
    receipt: str
    notes: Dict[str, Any] = Field(default_factory=dict)

class RazorpayOrderOut(BaseModel):
    key_id: str
    order_id: str
    amount: int
    currency: str

@app.post("/api/payments/razorpay/order", response_model=RazorpayOrderOut)
def create_razorpay_order(req: RazorpayOrderIn):
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=500, detail="Razorpay keys not configured")

    amount_paise = int(req.amount_inr) * 100

    try:
        r = requests.post(
            "https://api.razorpay.com/v1/orders",
            auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET),
            json={
                "amount": amount_paise,
                "currency": "INR",
                "receipt": req.receipt,
                "notes": req.notes or {}
            },
            timeout=20
        )

        if r.status_code >= 400:
            raise HTTPException(status_code=500, detail=f"Razorpay order error: {r.text}")

        data = r.json()
        return {
            "key_id": RAZORPAY_KEY_ID,
            "order_id": data["id"],
            "amount": data["amount"],
            "currency": data["currency"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Razorpay order failed: {str(e)}")


class RazorpayVerifyIn(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str

@app.post("/api/payments/razorpay/verify")
def verify_razorpay_signature(req: RazorpayVerifyIn):
    if not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=500, detail="Razorpay secret not configured")

    body = f"{req.razorpay_order_id}|{req.razorpay_payment_id}"
    generated = hmac.new(
        bytes(RAZORPAY_KEY_SECRET, "utf-8"),
        bytes(body, "utf-8"),
        hashlib.sha256
    ).hexdigest()

    ok = hmac.compare_digest(generated, req.razorpay_signature)
    return {"ok": ok}


# =========================
# SUPPORT CHATBOT
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
- payment issues
- booking status / confirmation
- cancellation / reschedule
- refund timelines
- pickup timing / coordination
- general assistance
- connect to human agent

STRICT RULES:
- DO NOT create itineraries and DO NOT sell packages here.
- If user asks for itinerary/package/plan, reply: "Please use Plan with AI page for itinerary."
- Be concise and helpful.
- If user asks for human support, immediately give contact details.

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
