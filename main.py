import os
import json
import re
import hmac
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr, field_validator
from openai import OpenAI
import razorpay

# =========================================================
# APP
# =========================================================
app = FastAPI(title="HKE Backend - AI Planner + Razorpay + Booking Save", version="5.0.0")

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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "").strip()
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "").strip()

DB_PATH = os.getenv("DB_PATH", "hke_bookings.db")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
rz_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)) if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET else None

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
        max_output_tokens=2400
    )

    text = extract_text_from_response(resp)
    parsed = try_parse_json(text)

    if not parsed:
        raise ValueError("Invalid JSON returned by model")

    return parsed

def build_itinerary_prompt(data: dict) -> str:
    return f"""
Create a professional travel itinerary for a travel agency website.

Customer data:
{json.dumps(data, ensure_ascii=False, indent=2)}

Return ONLY valid JSON in this exact structure:
{{
  "title": "string",
  "meta": {{
    "destination": "string",
    "route": "string",
    "dates": "string",
    "travellers": "string",
    "rooms": "string"
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
  "days": [
    {{
      "day": 1,
      "date": "YYYY-MM-DD",
      "title": "string",
      "route": "string",
      "activities": ["string", "string", "string"]
    }}
  ]
}}

Rules:
- Match exact number of days requested.
- Use selected tourist places.
- Keep it practical and customer friendly.
- Do not include pricing.
- Do not include markdown.
- Do not include explanation outside JSON.
""".strip()

def build_edit_prompt(current_itinerary: str, instruction: str, customer_details: dict) -> str:
    return f"""
Edit this travel itinerary according to customer instruction.

Customer details:
{json.dumps(customer_details, ensure_ascii=False, indent=2)}

Current itinerary:
{current_itinerary}

Instruction:
{instruction}

Return ONLY valid JSON in this exact structure:
{{
  "title": "string",
  "meta": {{
    "destination": "string",
    "route": "string",
    "dates": "string",
    "travellers": "string",
    "rooms": "string"
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
  "days": [
    {{
      "day": 1,
      "date": "YYYY-MM-DD",
      "title": "string",
      "route": "string",
      "activities": ["string", "string", "string"]
    }}
  ]
}}

Rules:
- Apply the edit instruction properly.
- Keep it travel-agency ready.
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

    day_items = []
    for i in range(days):
        if i == 0:
            title = f"Arrival and transfer to {destination}"
            route = f"{from_location} → {destination}"
            activities = [
                f"Arrival from {from_location} and transfer assistance.",
                f"Check-in at hotel in {destination}.",
                "Relax and enjoy local leisure time if possible."
            ]
        elif i == days - 1:
            title = "Departure day"
            route = f"{destination} → {end_point}"
            activities = [
                "Breakfast at hotel and checkout.",
                f"Proceed towards {end_point} for onward journey.",
                "Trip concludes with beautiful memories."
            ]
        else:
            p1 = get_place(i)
            p2 = get_place(i + 1)
            p3 = get_place(i + 2)
            title = f"{p1} sightseeing"
            route = f"{destination} local / nearby"
            activities = [
                f"After breakfast, visit {p1}.",
                f"Continue sightseeing to {p2}.",
                f"Optional stop at {p3} depending on time and weather.",
                "Return to hotel for overnight stay."
            ]

        day_items.append({
            "day": i + 1,
            "date": "",
            "title": title,
            "route": route,
            "activities": activities
        })

    extra_info = {
        "budget": safe_str(data.get("budget"), "Standard"),
        "travelType": safe_str(data.get("travelType"), "Family"),
        "hotel": safe_str(data.get("hotelClass"), "Standard"),
        "vehicle": safe_str(data.get("vehicle"), "SUV"),
        "guide": safe_str(data.get("guide"), "Without Guide"),
        "food": f'{safe_str(data.get("foodPreference"), "Flexible")} Food Required' if data.get("needFood") else "Food not included",
        "style": ", ".join(data.get("travelStyle") or []) or "Flexible",
        "notes": safe_str(data.get("notes"), "No special notes")
    }

    if edit_note:
        extra_info["editNote"] = edit_note

    return {
        "title": f"{days}-Day {destination} {safe_str(data.get('travelType'), 'Holiday')} Itinerary",
        "meta": {
            "destination": destination,
            "route": f"{from_location} → {destination} → {end_point}",
            "dates": f"{start_date} to {end_date}",
            "travellers": f'{data.get("travellers", 2)} Traveller(s)',
            "rooms": f'{data.get("rooms", 1)} Room(s)',
        },
        "extraInfo": extra_info,
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
        "version": "5.0.0"
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "openai_configured": bool(client),
        "razorpay_configured": bool(rz_client),
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
        return {"ok": True, "source": "openai", "itinerary": itinerary}
    except Exception as e:
        return {"ok": True, "source": "fallback", "warning": str(e), "itinerary": fallback_itinerary(data)}

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
        itinerary = call_openai_json(build_edit_prompt(current_itinerary, instruction, customer_details))
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
        raise HTTPException(status_code=500, detail=f"Unable to create Razorpay order: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Unable to save payment confirmation: {str(e)}")
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
