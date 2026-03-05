from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import os
import json
import base64
import hmac
import hashlib
import time
import requests

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
# AI REQUEST (accept camel + snake)
# =========================
class AIPlanRequest(BaseModel):
    destination: str
    days: int

    # Accept both "startDate" and "start_date"
    startDate: str = Field(..., alias="start_date")
    endDate: Optional[str] = Field(default="", alias="end_date")

    travellers: int = 2
    rooms: int = 1
    hotelClass: str = "Standard"
    vehicle: str = "SUV"
    guide: str = "Without Guide"

    startPoint: Optional[str] = ""
    endPoint: Optional[str] = ""
    notes: Optional[str] = ""

    # places can come as array OR comma string
    interests: Union[List[str], str, None] = Field(default_factory=list)

    class Config:
        allow_population_by_field_name = True  # allows startDate input too


class AIPlanResponse(BaseModel):
    itinerary: str
    itineraryJson: Dict[str, Any]

def _normalize_places(interests: Union[List[str], str, None]) -> List[str]:
    if interests is None:
        return []
    if isinstance(interests, list):
        return [str(x).strip() for x in interests if str(x).strip()]
    if isinstance(interests, str):
        return [x.strip() for x in interests.split(",") if x.strip()]
    return []

def _safe_str(x: Optional[str]) -> str:
    return (x or "").strip()

SYSTEM_PROMPT = """
You are a senior India tour operations planner for Himalayan Kerala Expeditions.
Your job is to create REALISTIC, OPERABLE itineraries (not generic travel content).
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
 "destination": "...",
 "startDate": "YYYY-MM-DD",
 "endDate": "YYYY-MM-DD",
 "days": 5,
 "travellers": 2,
 "rooms": 1,
 "hotelClass": "Standard",
 "vehicle": "SUV",
 "guide": "Without Guide",
 "summary": "...",
 "route_overview": ["Day 1: ...", "Day 2: ..."],
 "day_wise": [
   {
     "day": 1,
     "title": "Short day title",
     "start_time": "08:30",
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
    places_text = ", ".join(places) if places else "Best highlights as per destination"

    return f"""
Customer details:
Destination: {req.destination}
Trip Start Point: {_safe_str(req.startPoint)}
Trip End Point: {_safe_str(req.endPoint)}
Start Date: {req.startDate}
End Date: {req.endDate}
Days: {req.days}
Travellers: {req.travellers}
Rooms: {req.rooms}
Hotel Category: {req.hotelClass}
Vehicle: {req.vehicle}
Guide: {req.guide}
Selected Places (must include): {places_text}
Customer Notes: {_safe_str(req.notes) or "None"}

Create a practical day-wise plan following all rules. Return JSON only.
""".strip()

def _standard_includes():
    return [
        "Accommodation as per selected category (standard check-in/check-out times)",
        "Private vehicle with driver as per itinerary (point-to-point as planned)",
        "Sightseeing as per route (time & weather permitting)",
        "Basic trip coordination support from HKE team before and during travel",
    ]

def _standard_excludes():
    return [
        "Meals not mentioned in hotel plan (lunch/dinner usually excluded)",
        "Entry tickets, activities, rides (snow activities, ropeway, rafting, etc.)",
        "Personal expenses (shopping, tips, laundry, room service)",
        "Any changes due to weather/road closure beyond operator control",
    ]

def _standard_notes():
    return [
        "Final order depends on road/time/weather and local conditions.",
        "Hotel selection is as per category and availability at booking time.",
        "Hills driving times vary—start early to keep the plan comfortable.",
    ]

# =========================
# AI GENERATE (MAIN)
# =========================
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

        # Parse JSON strictly
        try:
            itinerary_json = json.loads(text)
        except Exception:
            # If model returns JSON wrapped in text, attempt extraction
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                itinerary_json = json.loads(text[start:end+1])
            else:
                raise

        # Ensure standard sections exist even if model missed
        itinerary_json.setdefault("destination", req.destination)
        itinerary_json.setdefault("startDate", req.startDate)
        itinerary_json.setdefault("endDate", req.endDate)
        itinerary_json.setdefault("days", req.days)
        itinerary_json.setdefault("travellers", req.travellers)
        itinerary_json.setdefault("rooms", req.rooms)
        itinerary_json.setdefault("hotelClass", req.hotelClass)
        itinerary_json.setdefault("vehicle", req.vehicle)
        itinerary_json.setdefault("guide", req.guide)

        itinerary_json.setdefault("package_includes", _standard_includes())
        itinerary_json.setdefault("package_excludes", _standard_excludes())
        itinerary_json.setdefault("notes", _standard_notes())
        itinerary_json.setdefault("optional_if_time", [])

        # Return both: pretty text + json
        pretty = json.dumps(itinerary_json, ensure_ascii=False, indent=2)

        return {
            "itinerary": pretty,
            "itineraryJson": itinerary_json
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generate failed: {str(e)}")

# Alias
@app.post("/api/ai/itinerary", response_model=AIPlanResponse)
def generate_itinerary_alias(req: AIPlanRequest):
    return generate_itinerary(req)

# =========================
# RAZORPAY (Order + Verify)
# =========================
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")

class RazorpayOrderIn(BaseModel):
    amount_inr: int  # in rupees
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
        raise HTTPException(status_code=500, detail="Razorpay keys not set in server env")

    amount_paise = int(req.amount_inr) * 100
    payload = {
        "amount": amount_paise,
        "currency": "INR",
        "receipt": req.receipt,
        "notes": req.notes or {}
    }

    auth = (RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
    try:
        r = requests.post("https://api.razorpay.com/v1/orders", json=payload, auth=auth, timeout=20)
        if r.status_code >= 400:
            raise HTTPException(status_code=500, detail=f"Razorpay order error: {r.text}")
        data = r.json()
        return {
            "key_id": RAZORPAY_KEY_ID,
            "order_id": data["id"],
            "amount": data["amount"],
            "currency": data["currency"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Razorpay order create failed: {str(e)}")

class RazorpayVerifyIn(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str

@app.post("/api/payments/razorpay/verify")
def verify_razorpay_signature(req: RazorpayVerifyIn):
    if not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=500, detail="Razorpay secret not set")

    msg = f"{req.razorpay_order_id}|{req.razorpay_payment_id}".encode("utf-8")
    expected = hmac.new(RAZORPAY_KEY_SECRET.encode("utf-8"), msg, hashlib.sha256).hexdigest()

    ok = hmac.compare_digest(expected, req.razorpay_signature)
    return {"ok": ok}

# =========================
# SUPPORT CHATBOT (KEEP)
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
