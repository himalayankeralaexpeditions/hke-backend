import os
import json
import re
import hmac
import hashlib
import sqlite3
import smtplib
from datetime import datetime
from email.message import EmailMessage
from typing import List, Optional, Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr, field_validator
from openai import OpenAI
import razorpay

load_dotenv()

# =========================================================
# APP
# =========================================================
app = FastAPI(
    title="HKE Backend - AI Planner + Razorpay + Booking Save",
    version="7.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# ENV
# =========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip()

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "").strip()
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "").strip()

DB_PATH = os.getenv("DB_PATH", "hke_bookings.db").strip()

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()
ENQUIRY_RECEIVER = os.getenv("ENQUIRY_RECEIVER", "").strip()

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
rz_client = (
    razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET
    else None
)

# =========================================================
# DB
# =========================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS payments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_name TEXT,
        customer_email TEXT,
        customer_phone TEXT,
        destination TEXT,
        from_location TEXT,
        end_point TEXT,
        start_date TEXT,
        end_date TEXT,
        travellers INTEGER,
        rooms INTEGER,
        trip_name TEXT,
        payment_type TEXT,
        paid_amount REAL,
        total_amount REAL,
        remaining_amount REAL,
        full_payment_deadline TEXT,
        next_schedule_text TEXT,
        razorpay_order_id TEXT UNIQUE,
        razorpay_payment_id TEXT UNIQUE,
        paid_at TEXT,
        raw_customer_json TEXT,
        raw_itinerary_json TEXT,
        raw_pricing_json TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


@app.on_event("startup")
def startup_event():
    init_db()

# =========================================================
# HELPERS
# =========================================================
def clean_phone(value: str) -> str:
    return re.sub(r"\D", "", value or "")[:10]


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def extract_text_from_response(resp: Any) -> str:
    output_text = getattr(resp, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    try:
        parts = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                ctype = getattr(c, "type", "")
                if ctype in ("output_text", "text"):
                    txt = getattr(c, "text", None)
                    if txt:
                        parts.append(txt)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def try_parse_json(text: str) -> Optional[dict]:
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def call_openai_json(prompt: str) -> dict:
    if not client:
        raise RuntimeError("OPENAI_API_KEY not configured")

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        max_output_tokens=3800,
    )

    text = extract_text_from_response(resp)
    parsed = try_parse_json(text)

    if not parsed:
        raise ValueError("Invalid JSON returned by model")

    return parsed


def send_itinerary_enquiry_email(customer_data: dict, itinerary: Optional[dict] = None):
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS or not ENQUIRY_RECEIVER:
        print("Email skipped: SMTP settings not configured")
        return

    name = safe_str(customer_data.get("name"))
    email = safe_str(customer_data.get("email"))
    phone = safe_str(customer_data.get("phone"))
    from_location = safe_str(customer_data.get("fromLocation"))
    destination = safe_str(customer_data.get("destination"))
    end_point = safe_str(customer_data.get("endPoint"))
    start_date = safe_str(customer_data.get("startDate"))
    end_date = safe_str(customer_data.get("endDate"))
    days = safe_str(customer_data.get("days"))
    travellers = safe_str(customer_data.get("travellers"))
    rooms = safe_str(customer_data.get("rooms"))
    budget = safe_str(customer_data.get("budget"))
    travel_type = safe_str(customer_data.get("travelType"))
    hotel_class = safe_str(customer_data.get("hotelClass"))
    vehicle = safe_str(customer_data.get("vehicle"))
    guide = safe_str(customer_data.get("guide"))
    need_food = bool(customer_data.get("needFood"))
    food_preference = safe_str(customer_data.get("foodPreference"))
    travel_style = ", ".join(customer_data.get("travelStyle", []))
    places = ", ".join(customer_data.get("places", []))
    notes = safe_str(customer_data.get("notes"))

    itinerary_title = safe_str((itinerary or {}).get("title"))
    itinerary_summary = safe_str((itinerary or {}).get("summary"))

    subject = f"New AI Planner Enquiry - {destination} - {name}"

    body = f"""
New AI Planner enquiry received from Himalayan Kerala Expeditions.

Customer Details
----------------
Name: {name}
Email: {email}
Phone: {phone}

Trip Details
------------
From Location: {from_location}
Destination / State: {destination}
Trip End Point: {end_point}
Start Date: {start_date}
End Date: {end_date}
Days: {days}
Travellers: {travellers}
Rooms: {rooms}

Preferences
-----------
Budget: {budget}
Travel Type: {travel_type}
Hotel Category: {hotel_class}
Vehicle: {vehicle}
Guide: {guide}
Need Food: {"Yes" if need_food else "No"}
Food Preference: {food_preference}
Travel Style: {travel_style}
Selected Tourist Places: {places}

Special Notes
-------------
{notes}

Generated Itinerary Title
-------------------------
{itinerary_title}

Generated Itinerary Summary
---------------------------
{itinerary_summary}
""".strip()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = ENQUIRY_RECEIVER
    msg["Reply-To"] = email if email else SMTP_USER
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)


def build_itinerary_prompt(data: dict) -> str:
    return f"""
You are a senior luxury travel consultant and itinerary designer for Himalayan Kerala Expeditions, a premium Indian travel company.

Your job is to create a highly polished, customer-facing itinerary that feels like it was prepared by an experienced travel executive with deep destination knowledge.

Customer trip request:
{json.dumps(data, ensure_ascii=False, indent=2)}

Return ONLY valid JSON in this exact structure:
{{
  "title": "string",
  "summary": "A polished 3 to 5 line professional trip introduction written like a premium travel consultant",
  "meta": {{
    "destination": "string",
    "route": "string",
    "dates": "string",
    "travellers": "string",
    "rooms": "string",
    "tripStyle": "string"
  }},
  "extraInfo": {{
    "budget": "string",
    "travelType": "string",
    "hotel": "string",
    "vehicle": "string",
    "guide": "string",
    "food": "string",
    "style": "string",
    "notes": "string"
  }},
  "highlights": [
    "string",
    "string",
    "string",
    "string"
  ],
  "inclusions": [
    "string",
    "string",
    "string",
    "string"
  ],
  "exclusions": [
    "string",
    "string",
    "string",
    "string"
  ],
  "terms": [
    "string",
    "string",
    "string",
    "string"
  ],
  "days": [
    {{
      "day": 1,
      "date": "YYYY-MM-DD",
      "title": "string",
      "route": "string",
      "hotel": "string",
      "meals": "string",
      "activities": [
        "string",
        "string",
        "string",
        "string"
      ],
      "notes": "string"
    }}
  ]
}}

Rules:
- Write like a highly experienced senior travel agent, not like a bot.
- The itinerary must feel premium, practical, and customer-ready.
- The language should sound warm, confident, and professional.
- Use realistic travel flow and route order.
- Reflect the user's selected state, places, travel type, budget, hotel category, vehicle, food preference, and notes.
- Every day must have 4 to 6 meaningful activity points.
- Activities must be detailed, natural, and useful, not short generic lines.
- Mention practical movement, sightseeing pacing, check-in/check-out flow, scenic experiences, local exploration, and comfort planning.
- Notes must sound like genuine travel advisor guidance.
- Summary must feel premium and persuasive.
- Highlights must sound attractive and professionally written.
- Inclusions, exclusions, and terms must sound like a real travel company document.
- Do not include pricing.
- Do not include markdown.
- Do not include explanation outside JSON.
- Day plan must feel ready to send to a customer without further rewriting.
""".strip()


def build_edit_prompt(current_itinerary: str, instruction: str, customer_details: dict) -> str:
    return f"""
You are a senior luxury travel consultant updating an already prepared itinerary for Himalayan Kerala Expeditions.

Customer details:
{json.dumps(customer_details, ensure_ascii=False, indent=2)}

Current itinerary JSON:
{current_itinerary}

Customer edit instruction:
{instruction}

Return ONLY valid JSON in this exact structure:
{{
  "title": "string",
  "summary": "A polished 3 to 5 line professional trip introduction",
  "meta": {{
    "destination": "string",
    "route": "string",
    "dates": "string",
    "travellers": "string",
    "rooms": "string",
    "tripStyle": "string"
  }},
  "extraInfo": {{
    "budget": "string",
    "travelType": "string",
    "hotel": "string",
    "vehicle": "string",
    "guide": "string",
    "food": "string",
    "style": "string",
    "notes": "string",
    "editNote": "string"
  }},
  "highlights": [
    "string",
    "string",
    "string",
    "string"
  ],
  "inclusions": [
    "string",
    "string",
    "string",
    "string"
  ],
  "exclusions": [
    "string",
    "string",
    "string",
    "string"
  ],
  "terms": [
    "string",
    "string",
    "string",
    "string"
  ],
  "days": [
    {{
      "day": 1,
      "date": "YYYY-MM-DD",
      "title": "string",
      "route": "string",
      "hotel": "string",
      "meals": "string",
      "activities": [
        "string",
        "string",
        "string",
        "string"
      ],
      "notes": "string"
    }}
  ]
}}

Rules:
- Apply the customer change properly.
- Keep it polished, premium, and customer-ready.
- Keep the itinerary practical and realistic.
- Do not include pricing.
- Do not include markdown.
- Do not include explanation outside JSON.
""".strip()


def fallback_itinerary(data: dict, edit_note: str = "") -> dict:
    places = data.get("places") or ["Local Sightseeing"]
    days = max(2, int(data.get("days") or 5))
    destination = safe_str(data.get("destination"))
    from_location = safe_str(data.get("fromLocation"))
    end_point = safe_str(data.get("endPoint"))
    start_date = safe_str(data.get("startDate"))
    end_date = safe_str(data.get("endDate"))

    def get_place(i: int) -> str:
        return places[i % len(places)] if places else "Local Sightseeing"

    food_text = (
        f'{safe_str(data.get("foodPreference"), "Flexible")} meals as per selected package'
        if data.get("needFood")
        else "Meals not included unless specifically mentioned"
    )

    day_items = []
    for i in range(days):
        if i == 0:
            title = f"Arrival journey to {destination} and hotel check-in"
            route = f"{from_location} → {destination}"
            activities = [
                f"Begin your journey from {from_location} towards {destination} in a comfortable and well-planned travel flow.",
                f"On arrival at {destination}, complete hotel check-in and settle into your selected stay category.",
                "Take sufficient time to rest after the journey and refresh before stepping out for the evening.",
                "If arrival time permits, enjoy a relaxed local market walk or light nearby sightseeing for a pleasant first impression of the destination.",
                "Return to the hotel for a comfortable overnight stay and prepare for the full sightseeing schedule ahead."
            ]
            hotel = f"{safe_str(data.get('hotelClass'), 'Standard')} stay in {destination}"
            meals = "As per arrival time / package plan"
            notes = "Early check-in depends on hotel availability. Travel fatigue has been considered to keep the first day comfortable."
        elif i == days - 1:
            title = f"Departure from {destination}"
            route = f"{destination} → {end_point}"
            activities = [
                "Enjoy breakfast at the hotel and complete checkout formalities in a relaxed manner.",
                f"Proceed towards {end_point} for your onward journey as per travel timing and route convenience.",
                "Keep buffer travel time in hand for a stress-free transfer, especially during peak season or traffic hours.",
                "The trip concludes with beautiful travel memories and a well-paced experience."
            ]
            hotel = "Checkout day"
            meals = "Breakfast"
            notes = "Departure movement should be aligned with reporting time, traffic conditions, and seasonal road status."
        else:
            p1 = get_place(i)
            p2 = get_place(i + 1)
            p3 = get_place(i + 2)
            title = f"{p1} and nearby sightseeing experience"
            route = f"{destination} local / nearby circuit"
            activities = [
                f"After breakfast, proceed for a full-day excursion covering {p1} with comfortable pacing and scenic travel flow.",
                f"Continue towards {p2}, allowing time for photography, local exploration, and enjoying the major highlights of the area.",
                f"If time, weather, and road conditions are favourable, include an additional stop at {p3} for a more complete sightseeing experience.",
                "Keep time in hand for tea breaks, viewpoint halts, and a smoother family-friendly or couple-friendly travel experience depending on the trip style.",
                "Return to the hotel by evening and unwind after the day’s exploration."
            ]
            hotel = f"{safe_str(data.get('hotelClass'), 'Standard')} stay in {destination}"
            meals = "Breakfast" if not data.get("needFood") else food_text
            notes = "The sightseeing order may be adjusted slightly based on weather, traffic, local restrictions, or guest comfort."

        day_items.append({
            "day": i + 1,
            "date": "",
            "title": title,
            "route": route,
            "hotel": hotel,
            "meals": meals,
            "activities": activities,
            "notes": notes
        })

    extra_info = {
        "budget": safe_str(data.get("budget"), "Standard"),
        "travelType": safe_str(data.get("travelType"), "Family"),
        "hotel": safe_str(data.get("hotelClass"), "Standard"),
        "vehicle": safe_str(data.get("vehicle"), "SUV"),
        "guide": safe_str(data.get("guide"), "Without Guide"),
        "food": food_text,
        "style": ", ".join(data.get("travelStyle") or []) or "Flexible",
        "notes": safe_str(data.get("notes"), "No special notes")
    }

    if edit_note:
        extra_info["editNote"] = edit_note

    return {
        "title": f"{destination} Premium Travel Plan | {days} Days",
        "summary": f"This {days}-day professionally structured journey through {destination} has been designed to offer a comfortable balance of sightseeing, travel convenience, and memorable destination experiences. The itinerary reflects your selected travel style, preferred places, and overall comfort expectations, making it suitable for a smooth and well-managed holiday experience.",
        "meta": {
            "destination": destination,
            "route": f"{from_location} → {destination} → {end_point}",
            "dates": f"{start_date} to {end_date}",
            "travellers": f'{data.get("travellers", 2)} Travellers',
            "rooms": f'{data.get("rooms", 1)} Room(s)',
            "tripStyle": safe_str(data.get("travelType"), "Holiday")
        },
        "extraInfo": extra_info,
        "highlights": [
            f"Professionally planned sightseeing across {destination}",
            "Balanced route design with practical travel pacing",
            "Comfortable stay and transfer flow based on selected preferences",
            "Customer-friendly itinerary suitable for smooth holiday execution"
        ],
        "inclusions": [
            "Accommodation as per selected hotel category",
            f"Transportation by {safe_str(data.get('vehicle'), 'SUV')} for the itinerary movement",
            "Daily sightseeing as per the final confirmed route plan",
            "Travel assistance and coordination support as per package structure"
        ],
        "exclusions": [
            "Airfare, train fare, or any transport not specifically mentioned",
            "Entry tickets, monument fees, activity charges, and personal expenses",
            "Lunch, dinner, snacks, or meals not specifically included in the package",
            "Any cost arising due to weather issues, natural disturbance, or operational changes"
        ],
        "terms": [
            "The itinerary remains subject to final operational feasibility and availability.",
            "Sightseeing flow may change slightly depending on road, weather, and local authority conditions.",
            "Hotel check-in and check-out timings will apply as per hotel policy.",
            "Final travel services are confirmed only after booking amount and availability confirmation."
        ],
        "days": day_items
    }


def verify_razorpay_signature(order_id: str, payment_id: str, signature: str) -> bool:
    if not RAZORPAY_KEY_SECRET:
        return False

    body = f"{order_id}|{payment_id}".encode("utf-8")
    expected_signature = hmac.new(
        RAZORPAY_KEY_SECRET.encode("utf-8"),
        body,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected_signature, signature)

# =========================================================
# MODELS
# =========================================================
class PlannerRequest(BaseModel):
    name: str
    email: EmailStr
    phone: str
    fromLocation: str
    destination: str
    endPoint: str
    startDate: str
    days: int = Field(..., ge=2, le=30)
    endDate: str
    travellers: int = Field(default=2, ge=1, le=50)
    rooms: int = Field(default=1, ge=1, le=20)
    budget: str = "Standard"
    travelType: str = "Family"
    hotelClass: str = "Standard"
    vehicle: str = "SUV"
    guide: str = "Without Guide"
    needFood: bool = False
    foodPreference: Optional[str] = "Flexible"
    travelStyle: List[str] = Field(default_factory=list)
    places: List[str] = Field(default_factory=list)
    notes: Optional[str] = ""

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v):
        digits = clean_phone(v)
        if len(digits) != 10:
            raise ValueError("Phone must be 10 digits")
        return digits

    @field_validator("name", "fromLocation", "destination", "endPoint")
    @classmethod
    def validate_required_strings(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("This field is required")
        return v.strip()

    @field_validator("travelStyle", mode="before")
    @classmethod
    def normalize_travel_style(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        return []

    @field_validator("places", mode="before")
    @classmethod
    def normalize_places(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = [str(x).strip() for x in v if str(x).strip()]
            if not cleaned:
                raise ValueError("At least one tourist place is required")
            return cleaned
        if isinstance(v, str):
            cleaned = [x.strip() for x in v.split(",") if x.strip()]
            if not cleaned:
                raise ValueError("At least one tourist place is required")
            return cleaned
        raise ValueError("At least one tourist place is required")


class ChatEditRequest(BaseModel):
    instruction: Optional[str] = ""
    message: Optional[str] = ""
    current_itinerary: Optional[str] = ""
    itinerary: Optional[str] = ""
    customer_details: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class RazorpayOrderRequest(BaseModel):
    amount: float = Field(..., gt=0)
    currency: str = "INR"
    receipt: Optional[str] = ""
    name: str
    email: EmailStr
    phone: str
    trip_name: str = "HKE Trip Booking"
    payment_type: str = "advance"

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v):
        digits = clean_phone(v)
        if len(digits) != 10:
            raise ValueError("Phone must be 10 digits")
        return digits


class RazorpayVerifyRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


class SavePaymentRequest(BaseModel):
    customer: Dict[str, Any]
    itinerary: Dict[str, Any]
    pricing: Dict[str, Any]
    payment: Dict[str, Any]

# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "HKE Backend Running",
        "version": "7.0.0"
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "openai_configured": bool(client),
        "razorpay_configured": bool(rz_client),
        "email_configured": bool(SMTP_HOST and SMTP_USER and SMTP_PASS and ENQUIRY_RECEIVER),
        "model": OPENAI_MODEL
    }


@app.get("/api/payment/config")
def payment_config():
    return {
        "ok": True,
        "razorpay_key_id": RAZORPAY_KEY_ID,
        "razorpay_enabled": bool(rz_client)
    }


@app.post("/api/ai/itinerary")
def generate_itinerary(payload: PlannerRequest):
    data = payload.model_dump()

    try:
        itinerary = call_openai_json(build_itinerary_prompt(data))
        source = "openai"
    except Exception as e:
        itinerary = fallback_itinerary(data)
        source = "fallback"
        print(f"AI itinerary fallback used: {e}")

    try:
        send_itinerary_enquiry_email(data, itinerary)
    except Exception as email_error:
        print(f"Failed to send enquiry email: {email_error}")

    return {
        "ok": True,
        "source": source,
        "itinerary": itinerary
    }


@app.post("/api/ai/chat")
def edit_itinerary(payload: ChatEditRequest):
    instruction = safe_str(payload.instruction) or safe_str(payload.message)
    current_itinerary = safe_str(payload.current_itinerary) or safe_str(payload.itinerary)
    customer_details = payload.customer_details or payload.context or {}

    if not instruction:
        raise HTTPException(status_code=400, detail="Edit instruction is required")
    if not current_itinerary:
        raise HTTPException(status_code=400, detail="Current itinerary is required")

    try:
        itinerary = call_openai_json(
            build_edit_prompt(current_itinerary, instruction, customer_details)
        )
        return {"ok": True, "source": "openai", "itinerary": itinerary}
    except Exception as e:
        return {
            "ok": True,
            "source": "fallback",
            "warning": str(e),
            "itinerary": fallback_itinerary(customer_details, edit_note=instruction)
        }


@app.post("/api/payment/create-order")
def create_payment_order(payload: RazorpayOrderRequest):
    if not rz_client:
        raise HTTPException(status_code=500, detail="Razorpay is not configured on server")

    amount_rupees = float(payload.amount)
    amount_paise = int(round(amount_rupees * 100))
    receipt = payload.receipt or f"hke_{payload.payment_type}_{clean_phone(payload.phone)}"

    notes = {
        "customer_name": payload.name,
        "customer_email": payload.email,
        "customer_phone": payload.phone,
        "trip_name": payload.trip_name,
        "payment_type": payload.payment_type,
    }

    try:
        order = rz_client.order.create({
            "amount": amount_paise,
            "currency": payload.currency,
            "receipt": receipt[:40],
            "payment_capture": 1,
            "notes": notes
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to create Razorpay order: {str(e)}"
        )

    return {
        "ok": True,
        "order_id": order.get("id"),
        "amount": amount_rupees,
        "currency": payload.currency,
        "razorpay_key_id": RAZORPAY_KEY_ID,
        "name": payload.name,
        "email": payload.email,
        "phone": payload.phone,
        "trip_name": payload.trip_name,
        "payment_type": payload.payment_type
    }


@app.post("/api/payment/verify")
def verify_payment(payload: RazorpayVerifyRequest):
    is_valid = verify_razorpay_signature(
        order_id=payload.razorpay_order_id,
        payment_id=payload.razorpay_payment_id,
        signature=payload.razorpay_signature
    )

    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid Razorpay signature")

    return {
        "ok": True,
        "verified": True,
        "message": "Payment verified successfully"
    }


@app.post("/api/payment/save-confirmation")
def save_payment_confirmation(payload: SavePaymentRequest):
    customer = payload.customer or {}
    itinerary = payload.itinerary or {}
    pricing = payload.pricing or {}
    payment = payload.payment or {}

    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute("""
        INSERT OR REPLACE INTO payments (
            customer_name, customer_email, customer_phone,
            destination, from_location, end_point,
            start_date, end_date, travellers, rooms,
            trip_name, payment_type, paid_amount, total_amount,
            remaining_amount, full_payment_deadline, next_schedule_text,
            razorpay_order_id, razorpay_payment_id, paid_at,
            raw_customer_json, raw_itinerary_json, raw_pricing_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            safe_str(customer.get("name")),
            safe_str(customer.get("email")),
            clean_phone(safe_str(customer.get("phone"))),
            safe_str(customer.get("destination")),
            safe_str(customer.get("fromLocation")),
            safe_str(customer.get("endPoint")),
            safe_str(customer.get("startDate")),
            safe_str(customer.get("endDate")),
            int(customer.get("travellers", 0) or 0),
            int(customer.get("rooms", 0) or 0),
            safe_str(itinerary.get("title")),
            safe_str(payment.get("paymentType")),
            float(payment.get("paidAmount", 0) or 0),
            float(pricing.get("finalFare", 0) or 0),
            float(payment.get("remainingAmount", 0) or 0),
            safe_str(payment.get("fullPaymentDeadline")),
            safe_str(payment.get("nextScheduleText")),
            safe_str(payment.get("razorpayOrderId")),
            safe_str(payment.get("razorpayPaymentId")),
            safe_str(payment.get("paidAt")),
            json.dumps(customer, ensure_ascii=False),
            json.dumps(itinerary, ensure_ascii=False),
            json.dumps(pricing, ensure_ascii=False),
            datetime.utcnow().isoformat()
        ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Unable to save payment confirmation: {str(e)}"
        )
    finally:
        conn.close()

    return {"ok": True, "message": "Payment confirmation saved successfully"}


@app.get("/api/payment/by-payment-id/{payment_id}")
def get_payment_by_payment_id(payment_id: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM payments WHERE razorpay_payment_id = ?", (payment_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Payment not found")

    return {"ok": True, "payment": dict(row)}


@app.get("/api/payments")
def list_payments():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, customer_name, customer_phone, destination, trip_name,
               payment_type, paid_amount, remaining_amount,
               razorpay_payment_id, paid_at
        FROM payments
        ORDER BY id DESC
        LIMIT 100
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    return {"ok": True, "items": rows}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
