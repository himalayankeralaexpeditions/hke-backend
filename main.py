import os
import json
import re
import hmac
import hashlib
import logging
import sqlite3
import smtplib
import random
import secrets
import requests
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import List, Optional, Any, Dict

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, EmailStr, field_validator, model_validator
from openai import OpenAI
from passlib.context import CryptContext
from pymongo import MongoClient
import razorpay

load_dotenv()
logger = logging.getLogger("hke.backend")

# =========================================================
# APP
# =========================================================
app = FastAPI(
    title="HKE Backend - AI Planner + Pilgrimage + Razorpay + Booking Save + OTP Login + Admin Login",
    version="8.3.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://himalayankeralaexpeditions.com",
        "https://www.himalayankeralaexpeditions.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$|https://([a-z0-9-]+\.)*godaddysites\.com$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    first_error = exc.errors()[0] if exc.errors() else {}
    message = first_error.get("msg") or "Invalid request"
    return JSONResponse(
        status_code=422,
        content={"ok": False, "detail": message}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    detail = exc.detail
    if exc.status_code >= 500:
        detail = "Something went wrong. Please try again later."
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "detail": detail}
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, _exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"ok": False, "detail": "Something went wrong. Please try again later."}
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

# OTP / MSG91
MSG91_AUTH_KEY = os.getenv("MSG91_AUTH_KEY", "").strip()
MSG91_SMS_FLOW_ID = os.getenv("MSG91_SMS_FLOW_ID", "").strip()
MSG91_DLT_TEMPLATE_ID = os.getenv("MSG91_DLT_TEMPLATE_ID", "").strip()
MSG91_DLT_TEMPLATE_VERSION = os.getenv("MSG91_DLT_TEMPLATE_VERSION", "v1.1").strip()
MSG91_OTP_VARIABLE_NAME = os.getenv("MSG91_OTP_VARIABLE_NAME", "var1").strip()
MSG91_SENDER_ID = os.getenv("MSG91_SENDER_ID", "HKEIND").strip()
OTP_EXPIRY_MINUTES = int(os.getenv("OTP_EXPIRY_MINUTES", "10"))
OTP_MAX_ATTEMPTS = int(os.getenv("OTP_MAX_ATTEMPTS", "5"))
OTP_BYPASS = os.getenv("OTP_BYPASS", "false").lower() == "true"

# Admin login
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123").strip()

# Mongo / partner portal
MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "hke").strip()

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
rz_client = (
    razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET
    else None
)

# Simple in-memory admin session store
ADMIN_SESSIONS: Dict[str, Dict[str, Any]] = {}
PARTNER_SESSIONS: Dict[str, Dict[str, Any]] = {}
PASSWORD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")

mongo_client = MongoClient(MONGODB_URI) if MONGODB_URI else None

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

    cur.execute("""
    CREATE TABLE IF NOT EXISTS otp_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mobile TEXT NOT NULL,
        otp_code TEXT NOT NULL,
        attempts INTEGER DEFAULT 0,
        verified INTEGER DEFAULT 0,
        expires_at TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS itinerary_change_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        booking_ref TEXT,
        customer_phone TEXT,
        customer_name TEXT,
        destination TEXT,
        request_text TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        admin_note TEXT DEFAULT '',
        created_at TEXT,
        updated_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS customer_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phone TEXT NOT NULL UNIQUE,
        name TEXT DEFAULT '',
        email TEXT DEFAULT '',
        consent_marketing INTEGER DEFAULT 1,
        source TEXT DEFAULT '',
        first_login_at TEXT,
        last_login_at TEXT,
        last_activity TEXT,
        notes TEXT DEFAULT ''
    )
    """)

    conn.commit()
    conn.close()


@app.on_event("startup")
def startup_event():
    init_db()


# =========================================================
# MONGO HELPERS
# =========================================================
def get_mongo_db():
    if not mongo_client:
        raise HTTPException(status_code=500, detail="MongoDB is not configured")
    return mongo_client[MONGODB_DB_NAME]


def get_partners_collection():
    return get_mongo_db()["partners"]


def get_partner_rates_collection():
    return get_mongo_db()["partner_rates"]


def ensure_partner_indexes():
    if not mongo_client:
        return

    partners = get_partners_collection()
    rates = get_partner_rates_collection()
    partners.create_index("mobile", unique=True)
    rates.create_index("partner_id")
    rates.create_index("status")
    rates.create_index("available")
    rates.create_index("service_area")
    rates.create_index("available_from")
    rates.create_index("available_to")
    rates.create_index([("partner_id", 1), ("available_from", 1), ("available_to", 1)])
    rates.create_index([("status", 1), ("available", 1), ("partner_type", 1)])
    rates.create_index([("location", 1), ("service_area", 1)])


@app.on_event("startup")
def startup_event_partner_indexes():
    ensure_partner_indexes()

# =========================================================
# HELPERS
# =========================================================
def clean_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) > 10:
        digits = digits[-10:]
    return digits


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def serialize_customer_profile(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "phone": safe_str(row["phone"]),
        "name": safe_str(row["name"]),
        "email": safe_str(row["email"]),
        "consent_marketing": bool(safe_int(row["consent_marketing"], 0)),
        "source": safe_str(row["source"]),
        "first_login_at": safe_str(row["first_login_at"]),
        "last_login_at": safe_str(row["last_login_at"]),
        "last_activity": safe_str(row["last_activity"])
    }


def upsert_customer_profile_after_otp(
    cur: sqlite3.Cursor,
    phone: str,
    login_at: str,
    consent_marketing: bool = True,
    source: str = "website_otp_login"
):
    cur.execute(
        """
        INSERT INTO customer_profiles (
            phone, consent_marketing, source, first_login_at, last_login_at, last_activity
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(phone) DO UPDATE SET
            consent_marketing = excluded.consent_marketing,
            source = excluded.source,
            last_login_at = excluded.last_login_at,
            last_activity = excluded.last_activity
        """,
        (
            phone,
            1 if consent_marketing else 0,
            source,
            login_at,
            login_at,
            login_at
        )
    )


def is_phone_verified(phone: str) -> bool:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT verified, created_at
        FROM otp_sessions
        WHERE mobile = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (phone,)
    )
    row = cur.fetchone()
    conn.close()

    if not row or safe_int(row["verified"], 0) != 1:
        return False

    created_at_raw = safe_str(row["created_at"])
    try:
        created_at = datetime.fromisoformat(created_at_raw)
    except Exception:
        return False

    return created_at >= (datetime.utcnow() - timedelta(hours=24))


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
        max_output_tokens=4200,
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


def generate_otp() -> str:
    return str(random.randint(100000, 999999))


def send_msg91_otp(mobile: str, otp: str):
    if OTP_BYPASS:
        return {"ok": True, "message": "OTP bypass mode enabled", "otp": otp}

    if not MSG91_AUTH_KEY:
        raise RuntimeError("MSG91_AUTH_KEY not configured")

    # MSG91 OneAPI Flow slug created in MSG91 dashboard:
    # Flow name: OTP_LOGIN_FLOW_HKE
    # ID/Slug: otp-login-flow-hke
    url = "https://control.msg91.com/api/v5/oneapi/api/flow/otp-login-flow-hke/run"

    # IMPORTANT:
    # MSG91 OneAPI uses the variable name shown in the flow code sample.
    # Your OneAPI flow shows variable name: OTP
    # Do NOT use ##OTP## here; MSG91 will keep the OTP blank.
    payload = {
        "data": {
            "sendTo": [
                {
                    "to": [
                        {
                            "mobiles": f"91{mobile}",
                            "variables": {
                                "OTP": {
                                    "value": otp
                                }
                            }
                        }
                    ],
                    "variables": {
                        "OTP": {
                            "value": otp
                        }
                    }
                }
            ]
        }
    }

    headers = {
        "authkey": MSG91_AUTH_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
    except requests.RequestException as exc:
        logger.exception("MSG91 OneAPI OTP request transport failure for mobile=%s", mobile)
        raise RuntimeError(f"MSG91 request failed: {exc}") from exc

    try:
        response_data = response.json()
    except Exception:
        response_data = {"raw": response.text}

    logger.info(
        "MSG91 OneAPI OTP response for mobile=%s status=%s response=%s",
        mobile,
        response.status_code,
        json.dumps(response_data, default=str)
    )

    if response.status_code not in (200, 201, 202):
        logger.error(
            "MSG91 OneAPI OTP send failed for mobile=%s status=%s response=%s",
            mobile,
            response.status_code,
            response.text
        )
        raise RuntimeError(f"MSG91 send failed: {response.text}")

    lower_dump = json.dumps(response_data).lower()
    if "error" in lower_dump or "failed" in lower_dump or "company details not found" in lower_dump:
        logger.error(
            "MSG91 OneAPI OTP rejected for mobile=%s response=%s",
            mobile,
            json.dumps(response_data, default=str)
        )
        raise RuntimeError(f"MSG91 rejected OTP: {response_data}")

    return response_data

def build_itinerary_prompt(data: dict, partner_context: Optional[Dict[str, Any]] = None) -> str:
    partner_lines = []
    if partner_context:
        partner_lines = partner_context.get("prompt_lines") or []

    partner_block = ""
    if partner_lines:
        partner_block = (
            "\nVerified HKE partner availability context:\n"
            + "\n".join(partner_lines)
            + "\n\nImportant response rules for verified partner availability:\n"
            + "- Include this exact line in the summary or stay guidance: Stay will be arranged in verified HKE partner property according to selected category.\n"
            + "- Do not reveal partner names, contact numbers, room counts, or rates.\n"
            + "- Present hotel, vehicle, and guide only as HKE verified or approved support.\n"
        )

    return f"""
You are a senior luxury travel consultant and itinerary designer for Himalayan Kerala Expeditions, a premium Indian travel company.

Your job is to create a highly polished, customer-facing itinerary that feels like it was prepared by an experienced travel executive with deep destination knowledge.

Customer trip request:
{json.dumps(data, ensure_ascii=False, indent=2)}{partner_block}

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


def build_pilgrimage_prompt(data: dict, pricing: dict) -> str:
    return f"""
You are a senior pilgrimage travel executive of Himalayan Kerala Expeditions.

Prepare a premium customer-facing pilgrimage itinerary in polished travel-agency language.

Customer request:
{json.dumps(data, ensure_ascii=False, indent=2)}

Live pricing inputs:
{json.dumps(pricing, ensure_ascii=False, indent=2)}

Return ONLY valid JSON in this exact structure:
{{
  "title": "string",
  "summary": "3 to 5 line premium pilgrimage introduction written like a senior travel executive",
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
    "religion": "string"
  }},
  "highlights": ["string", "string", "string", "string"],
  "inclusions": ["string", "string", "string", "string"],
  "exclusions": ["string", "string", "string", "string"],
  "terms": ["string", "string", "string", "string"],
  "pricing": {{
    "hotelTotal": 0,
    "vehicleTotal": 0,
    "foodTotal": 0,
    "subtotal": 0,
    "gst": 0,
    "finalFare": 0,
    "advanceAmount": 0
  }},
  "days": [
    {{
      "day": 1,
      "date": "YYYY-MM-DD",
      "title": "string",
      "route": "string",
      "hotel": "string",
      "meals": "string",
      "activities": ["string", "string", "string", "string"],
      "notes": "string"
    }}
  ]
}}

Rules:
- Write in polished Indian premium travel company style.
- Sound like an experienced senior pilgrimage travel executive.
- Keep route practical and region-correct.
- Mention darshan flow, transfer timing, comfort pacing, meal/rest breaks, and overnight logic.
- If travelType suggests family or senior citizens, keep the itinerary comfort-oriented.
- Use the supplied pricing values exactly. Do not invent pricing.
- Do not include markdown.
- Do not include explanation outside JSON.
""".strip()


def fallback_pilgrimage_itinerary(data: dict, pricing: dict) -> dict:
    destination = safe_str(data.get("destination"))
    from_location = safe_str(data.get("fromLocation"))
    end_point = safe_str(data.get("endPoint"))
    start_date = safe_str(data.get("startDate"))
    end_date = safe_str(data.get("endDate"))
    days = max(2, safe_int(data.get("days"), 5))
    places = data.get("places") or [destination]

    day_items = []
    for i in range(days):
        p1 = places[i % len(places)]
        p2 = places[(i + 1) % len(places)]
        if i == 0:
            activities = [
                f"Arrival from {from_location} and comfortable transfer toward {destination} pilgrimage circuit.",
                f"Check-in and settle according to selected category: {safe_str(data.get('hotelClass'), 'Standard')}.",
                f"Short evening visit / prayer time at {p1}.",
                "Rest and prepare for the main pilgrimage movement.",
                f"Travel assistance planned with {safe_str(data.get('vehicle'), 'SUV')}."
            ]
            title = f"Arrival in {destination} and spiritual introduction"
            route = f"{from_location} → {destination}"
            hotel = f"{safe_str(data.get('hotelClass'), 'Standard')} stay"
            meals = "As per arrival timing / package flow"
            notes = "The first day is kept lighter to maintain comfort and allow smooth acclimatization into the spiritual journey."
        elif i == days - 1:
            activities = [
                f"Morning darshan / spiritual visit at {p1}.",
                "Spend some peaceful time for prayer and local temple/church surroundings.",
                f"Checkout and begin departure toward {end_point}.",
                "Trip concludes with spiritual journey support from HKE."
            ]
            title = "Final darshan and departure"
            route = f"{destination} → {end_point}"
            hotel = "Checkout day"
            meals = "Breakfast"
            notes = "Departure has been kept practical to ensure a smooth end to the pilgrimage."
        else:
            activities = [
                f"Morning darshan / visit at {p1}.",
                f"Continue pilgrimage movement toward {p2}.",
                f"Meal / rest planning based on {safe_str(data.get('foodPreference'), 'flexible food preference')}.",
                "Evening prayer, local spiritual exploration, and overnight stay.",
                f"Travel assistance planned with {safe_str(data.get('vehicle'), 'SUV')}."
            ]
            title = f"{p1} and nearby spiritual circuit"
            route = f"{destination} local pilgrimage movement"
            hotel = f"{safe_str(data.get('hotelClass'), 'Standard')} stay"
            meals = "Breakfast / selected meal support"
            notes = "The day is paced for a meaningful pilgrimage experience without unnecessary rush."

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

    return {
        "title": f"{safe_str(data.get('religion'), 'Pilgrimage')} Pilgrimage - {destination}",
        "summary": f"This pilgrimage journey through {destination} has been structured in a premium and practical way to balance darshan priorities, route comfort, and overall travel convenience. The plan is prepared in a customer-ready format to reflect a professionally managed spiritual experience with thoughtful pacing and support.",
        "meta": {
            "destination": destination,
            "route": f"{from_location} → {destination} → {end_point}",
            "dates": f"{start_date} to {end_date}",
            "travellers": f"{safe_int(data.get('travellers'), 2)} Travellers",
            "rooms": f"{safe_int(data.get('rooms'), 1)} Room(s)",
            "tripStyle": safe_str(data.get("travelType"), "Pilgrimage")
        },
        "extraInfo": {
            "budget": safe_str(data.get("budget"), "Standard"),
            "travelType": safe_str(data.get("travelType"), "Family"),
            "hotel": safe_str(data.get("hotelClass"), "Standard"),
            "vehicle": safe_str(data.get("vehicle"), "SUV"),
            "guide": safe_str(data.get("guide"), "Without Guide"),
            "food": safe_str(data.get("foodPreference"), "Flexible"),
            "style": ", ".join(data.get("travelStyle") or []) or "Comfort focused",
            "notes": safe_str(data.get("notes"), "No special notes"),
            "religion": safe_str(data.get("religion"), "Pilgrimage")
        },
        "highlights": [
            f"Well-paced pilgrimage movement across {destination}",
            "Comfort-oriented darshan flow designed for practical execution",
            "Professional route support with selected stay and vehicle preference",
            "Customer-ready premium spiritual travel document"
        ],
        "inclusions": [
            "Accommodation as per selected hotel category",
            f"Vehicle support by {safe_str(data.get('vehicle'), 'SUV')} as per itinerary movement",
            "Pilgrimage route planning and coordinated day-wise movement",
            "Support from Himalayan Kerala Expeditions before and during the journey"
        ],
        "exclusions": [
            "VIP darshan tickets, helicopter, palki, pony, ropeway or special entry fees unless separately confirmed",
            "Personal expenses, shopping, laundry, tips, and donations",
            "Travel insurance or emergency medical expenses",
            "Anything not specifically listed under inclusions"
        ],
        "terms": [
            "Final confirmation remains subject to hotel and transport availability.",
            "Darshan and local movement timing can be adjusted based on local conditions.",
            "Seasonal demand may affect final confirmation timing and operational sequence.",
            "The final service order will be confirmed after booking payment and availability lock."
        ],
        "pricing": pricing,
        "days": day_items
    }


def calculate_pilgrimage_price(data: dict) -> dict:
    days = max(2, safe_int(data.get("days"), 5))
    travellers = max(1, safe_int(data.get("travellers"), 2))
    rooms = max(1, safe_int(data.get("rooms"), 1))
    hotel_class = safe_str(data.get("hotelClass"), "Standard")
    vehicle = safe_str(data.get("vehicle"), "SUV")
    budget = safe_str(data.get("budget"), "Standard")
    need_food = bool(data.get("needFood"))
    start_date = safe_str(data.get("startDate"))

    hotel_base_map = {
        "Budget": 2800,
        "Standard": 4200,
        "Deluxe": 5800,
        "Premium": 7600,
        "Luxury": 9800,
        "No Hotel": 0
    }

    vehicle_daily_map = {
        "No Cab": 0,
        "No Cab Required": 0,
        "Sedan": 3200,
        "SUV": 4200,
        "Innova": 4700,
        "Innova Crysta": 5200,
        "Tempo Traveller": 7600,
        "Own Vehicle": 1500,
        "Bike Rental": 2200
    }

    budget_factor = {
        "Budget": 1.00,
        "Standard": 1.08,
        "Premium": 1.18,
        "Luxury": 1.30
    }

    hotel_per_room_per_night = hotel_base_map.get(hotel_class, 4200)
    vehicle_per_day = vehicle_daily_map.get(vehicle, 4200)
    food_per_person_per_day = 650 if need_food else 0

    hotel_total = hotel_per_room_per_night * rooms * max(1, days - 1)
    vehicle_total = vehicle_per_day * days
    food_total = food_per_person_per_day * travellers * days

    subtotal = hotel_total + vehicle_total + food_total

    try:
        month = datetime.fromisoformat(start_date).month if start_date else None
    except Exception:
        month = None

    if month in [4, 5, 6, 9, 10, 12]:
        subtotal = int(round(subtotal * 1.15))

    subtotal = int(round(subtotal * budget_factor.get(budget, 1.08)))
    gst = int(round(subtotal * 0.05))
    final_fare = subtotal + gst
    advance_amount = int(round(final_fare * 0.20))

    return {
        "hotelTotal": int(hotel_total),
        "vehicleTotal": int(vehicle_total),
        "foodTotal": int(food_total),
        "subtotal": int(subtotal),
        "gst": int(gst),
        "finalFare": int(final_fare),
        "advanceAmount": int(advance_amount)
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


def require_admin_token(authorization: Optional[str]) -> Dict[str, Any]:
    if not authorization:
        raise HTTPException(status_code=401, detail="Admin authorization required")

    token = authorization.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()

    session = ADMIN_SESSIONS.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired admin session")

    return session


def money_to_paise(value: Any) -> int:
    try:
        return int(round(float(value or 0) * 100))
    except Exception:
        return 0



def parse_iso_date(value: str) -> datetime:
    try:
        return datetime.strptime(safe_str(value), "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {value}. Use YYYY-MM-DD")


def hash_password(password: str) -> str:
    return PASSWORD_CONTEXT.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return PASSWORD_CONTEXT.verify(password, hashed)
    except Exception:
        return False


def serialize_partner(partner: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(partner.get("_id") or ""),
        "partner_type": safe_str(partner.get("partner_type")),
        "business_name": safe_str(partner.get("business_name")),
        "contact_person": safe_str(partner.get("contact_person")),
        "mobile": clean_phone(safe_str(partner.get("mobile"))),
        "email": safe_str(partner.get("email")),
        "created_at": safe_str(partner.get("created_at"))
    }


def serialize_partner_rate(rate: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(rate)
    item["id"] = str(item.get("_id") or item.get("id") or "")
    item["_id"] = item["id"]
    item["partner_id"] = str(item.get("partner_id") or "")
    return item


def require_partner_token(authorization: Optional[str]) -> Dict[str, Any]:
    if not authorization:
        raise HTTPException(status_code=401, detail="Partner authorization required")

    token = authorization.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()

    session = PARTNER_SESSIONS.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired partner session")

    return session


def get_rate_sort_value(rate: Dict[str, Any]) -> float:
    partner_type = safe_str(rate.get("partner_type"))
    if partner_type == "Hotel":
        return safe_float(rate.get("price_per_night"), 10**12)
    if partner_type == "Driver":
        per_day = safe_float(rate.get("per_day_rate"), 10**12)
        per_km = safe_float(rate.get("per_km_rate"), 10**12)
        return per_day if per_day > 0 else per_km
    if partner_type == "Guide":
        return safe_float(rate.get("per_day_rate"), 10**12)
    return 10**12


def get_partner_search_terms(data: Dict[str, Any]) -> List[str]:
    terms = []
    destination = safe_str(data.get("destination"))
    if destination:
        terms.append(destination)
    for place in data.get("places") or []:
        cleaned = safe_str(place)
        if cleaned and cleaned not in terms:
            terms.append(cleaned)
    return terms


def find_best_partner_rate(
    search_terms: List[str],
    start_date: str,
    end_date: str,
    partner_type: str,
    extra_filter: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    if not mongo_client or not search_terms:
        return None

    matchers = []
    for term in search_terms:
        regex = {"$regex": re.escape(term), "$options": "i"}
        matchers.extend([
            {"location": regex},
            {"service_area": regex}
        ])

    query: Dict[str, Any] = {
        "partner_type": partner_type,
        "status": "approved",
        "available": True,
        "available_from": {"$lte": safe_str(end_date)},
        "available_to": {"$gte": safe_str(start_date)},
        "$or": matchers
    }

    if extra_filter:
        query.update(extra_filter)

    items = list(get_partner_rates_collection().find(query))
    if not items:
        return None

    items.sort(key=get_rate_sort_value)
    return items[0]


def get_partner_context_for_itinerary(data: Dict[str, Any]) -> Dict[str, Any]:
    search_terms = get_partner_search_terms(data)
    start_date = safe_str(data.get("startDate"))
    end_date = safe_str(data.get("endDate"))
    hotel_class = safe_str(data.get("hotelClass"), "Standard")
    vehicle = safe_str(data.get("vehicle"), "SUV")
    guide = safe_str(data.get("guide"), "Without Guide")

    result = {
        "hotel": None,
        "driver": None,
        "guide": None,
        "prompt_lines": []
    }

    hotel_rate = find_best_partner_rate(
        search_terms,
        start_date,
        end_date,
        "Hotel",
        {"hotel_category": hotel_class}
    )
    if hotel_rate:
        result["hotel"] = hotel_rate
        result["prompt_lines"].append(f"Hotel: {hotel_class} (HKE Verified Partner)")

    vehicle_lower = vehicle.strip().lower()
    if vehicle_lower not in {"", "no vehicle", "no cab", "no cab required", "own vehicle"}:
        driver_rate = find_best_partner_rate(
            search_terms,
            start_date,
            end_date,
            "Driver",
            {"vehicle_type": vehicle}
        )
        if driver_rate:
            result["driver"] = driver_rate
            result["prompt_lines"].append(f"Vehicle: {vehicle} (HKE Partner Driver)")

    guide_needed = guide.strip().lower() in {"with guide", "guide", "yes", "required"}
    if guide_needed:
        guide_rate = find_best_partner_rate(search_terms, start_date, end_date, "Guide")
        if guide_rate:
            result["guide"] = guide_rate
            result["prompt_lines"].append("Guide: Available")

    return result


def apply_partner_context_to_itinerary(itinerary: Dict[str, Any], partner_context: Dict[str, Any]) -> Dict[str, Any]:
    if not itinerary or not partner_context:
        return itinerary

    extra_info = itinerary.get("extraInfo") or {}
    if partner_context.get("hotel"):
        extra_info["hotel"] = f'{safe_str(extra_info.get("hotel")) or safe_str(partner_context["hotel"].get("hotel_category"), "Standard")} (HKE Verified Partner)'
    if partner_context.get("driver"):
        extra_info["vehicle"] = f'{safe_str(extra_info.get("vehicle")) or safe_str(partner_context["driver"].get("vehicle_type"), "Vehicle")} (HKE Partner Driver)'
    if partner_context.get("guide"):
        extra_info["guide"] = "Available"
    itinerary["extraInfo"] = extra_info

    if partner_context.get("hotel"):
        line = "Stay will be arranged in verified HKE partner property according to selected category."
        summary = safe_str(itinerary.get("summary"))
        if line not in summary:
            itinerary["summary"] = f"{summary} {line}".strip()

    return itinerary

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


class PilgrimageRequest(BaseModel):
    name: str
    email: EmailStr
    phone: str
    religion: str
    destination: str
    destinationState: Optional[str] = ""
    fromLocation: str
    endPoint: str
    startDate: str
    days: int = Field(..., ge=2, le=30)
    endDate: Optional[str] = ""
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

    @field_validator("name", "fromLocation", "destination", "religion", "endPoint")
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
                raise ValueError("At least one pilgrimage place is required")
            return cleaned
        if isinstance(v, str):
            cleaned = [x.strip() for x in v.split(",") if x.strip()]
            if not cleaned:
                raise ValueError("At least one pilgrimage place is required")
            return cleaned
        raise ValueError("At least one pilgrimage place is required")


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
    notes: Optional[Dict[str, Any]] = None

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


class SendOTPRequest(BaseModel):
    mobile: Optional[str] = None
    phone: Optional[str] = None

    @model_validator(mode="after")
    def validate_mobile_or_phone(self):
        value = self.mobile or self.phone
        digits = clean_phone(value or "")
        if len(digits) != 10:
            raise ValueError("Mobile number must be 10 digits")
        self.mobile = digits
        self.phone = digits
        return self


class VerifyOTPRequest(BaseModel):
    mobile: Optional[str] = None
    phone: Optional[str] = None
    otp: str

    @model_validator(mode="after")
    def validate_mobile_or_phone(self):
        value = self.mobile or self.phone
        digits = clean_phone(value or "")
        if len(digits) != 10:
            raise ValueError("Mobile number must be 10 digits")
        self.mobile = digits
        self.phone = digits
        return self

    @field_validator("otp")
    @classmethod
    def validate_otp(cls, v):
        otp = str(v).strip()
        if len(otp) < 4:
            raise ValueError("OTP is invalid")
        return otp


class CustomerProfileUpdateRequest(BaseModel):
    phone: str
    name: Optional[str] = ""
    email: Optional[EmailStr] = None
    consent_marketing: Optional[bool] = None

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v):
        digits = clean_phone(v)
        if len(digits) != 10:
            raise ValueError("Phone must be 10 digits")
        return digits

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        return safe_str(v)


class PartnerRegisterRequest(BaseModel):
    partner_type: str
    business_name: str
    contact_person: str
    mobile: str
    email: EmailStr
    password: str

    @field_validator("partner_type")
    @classmethod
    def validate_partner_type(cls, v):
        value = safe_str(v)
        if value not in {"Hotel", "Driver", "Guide"}:
            raise ValueError("Partner type must be Hotel, Driver, or Guide")
        return value

    @field_validator("business_name", "contact_person")
    @classmethod
    def validate_required_text(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("This field is required")
        return v.strip()

    @field_validator("mobile")
    @classmethod
    def validate_mobile(cls, v):
        digits = clean_phone(v)
        if len(digits) != 10:
            raise ValueError("Mobile number must be 10 digits")
        return digits

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        value = safe_str(v)
        if len(value) < 6:
            raise ValueError("Password must be at least 6 characters")
        return value


class PartnerLoginRequest(BaseModel):
    mobile: str
    password: str

    @field_validator("mobile")
    @classmethod
    def validate_mobile(cls, v):
        digits = clean_phone(v)
        if len(digits) != 10:
            raise ValueError("Mobile number must be 10 digits")
        return digits


class PartnerRateBase(BaseModel):
    partner_type: str
    business_name: str
    location: str
    state: str
    service_area: str
    available_from: str
    available_to: str
    available: bool = True
    notes: Optional[str] = ""
    hotel_category: Optional[str] = None
    room_type: Optional[str] = None
    price_per_night: Optional[float] = None
    meal_plan: Optional[str] = None
    rooms_available: Optional[int] = None
    total_rooms: Optional[int] = None
    vehicle_type: Optional[str] = None
    vehicle_number: Optional[str] = None
    per_day_rate: Optional[float] = None
    per_km_rate: Optional[float] = None
    driver_allowance: Optional[float] = None
    language: Optional[str] = None
    specialty: Optional[str] = None

    @field_validator("partner_type")
    @classmethod
    def validate_partner_type(cls, v):
        value = safe_str(v)
        if value not in {"Hotel", "Driver", "Guide"}:
            raise ValueError("Partner type must be Hotel, Driver, or Guide")
        return value

    @field_validator("business_name", "location", "state", "service_area")
    @classmethod
    def validate_required_text(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("This field is required")
        return v.strip()

    @model_validator(mode="after")
    def validate_dates_and_fields(self):
        start = parse_iso_date(self.available_from)
        end = parse_iso_date(self.available_to)
        if end < start:
            raise ValueError("available_to must be on or after available_from")
        if end > start + timedelta(days=62):
            raise ValueError("available_to cannot exceed 2 months from available_from")

        if self.partner_type == "Hotel":
            if not safe_str(self.hotel_category) or not safe_str(self.room_type) or safe_float(self.price_per_night) <= 0:
                raise ValueError("Hotel rates require hotel_category, room_type, and positive price_per_night")
            if not safe_str(self.meal_plan):
                raise ValueError("Hotel rates require meal_plan")
            rooms_value = self.rooms_available if self.rooms_available is not None else self.total_rooms
            if safe_int(rooms_value) <= 0:
                raise ValueError("Hotel rates require positive rooms_available")
            self.rooms_available = safe_int(rooms_value)

        if self.partner_type == "Driver":
            if not safe_str(self.vehicle_type) or not safe_str(self.vehicle_number):
                raise ValueError("Driver rates require vehicle_type and vehicle_number")
            if safe_float(self.per_day_rate) <= 0 and safe_float(self.per_km_rate) <= 0:
                raise ValueError("Driver rates require per_day_rate or per_km_rate")

        if self.partner_type == "Guide":
            if not safe_str(self.language) or not safe_str(self.specialty) or safe_float(self.per_day_rate) <= 0:
                raise ValueError("Guide rates require language, specialty, and positive per_day_rate")

        return self


class PartnerRateCreateRequest(PartnerRateBase):
    pass


class PartnerRateUpdateRequest(PartnerRateBase):
    pass


class AdminLoginRequest(BaseModel):
    username: str
    password: str


class ItineraryChangeRequestCreate(BaseModel):
    booking_ref: str
    customer_phone: str
    customer_name: Optional[str] = ""
    destination: Optional[str] = ""
    request_text: str

    @field_validator("customer_phone")
    @classmethod
    def validate_phone(cls, v):
        digits = clean_phone(v)
        if len(digits) != 10:
            raise ValueError("Phone must be 10 digits")
        return digits

    @field_validator("booking_ref", "request_text")
    @classmethod
    def validate_required_strings(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("This field is required")
        return v.strip()


class ItineraryChangeRequestReview(BaseModel):
    status: str
    admin_note: Optional[str] = ""

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        allowed = {"pending", "under_review", "approved", "rejected"}
        value = str(v).strip().lower()
        if value not in allowed:
            raise ValueError("Status must be one of: pending, under_review, approved, rejected")
        return value

# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "HKE Backend Running",
        "version": "8.3.2"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ok": True,
        "openai_configured": bool(client),
        "razorpay_configured": bool(rz_client),
        "email_configured": bool(SMTP_HOST and SMTP_USER and SMTP_PASS and ENQUIRY_RECEIVER),
        "msg91_configured": bool(MSG91_AUTH_KEY and MSG91_SMS_FLOW_ID),
        "msg91_dlt_template_configured": bool(MSG91_DLT_TEMPLATE_ID),
        "msg91_dlt_template_version": MSG91_DLT_TEMPLATE_VERSION,
        "msg91_variable_name": MSG91_OTP_VARIABLE_NAME,
        "otp_bypass": OTP_BYPASS,
        "admin_login_enabled": bool(ADMIN_USERNAME and ADMIN_PASSWORD),
        "model": OPENAI_MODEL
    }


@app.get("/api/payment/config")
def payment_config():
    return {
        "ok": True,
        "razorpay_key_id": RAZORPAY_KEY_ID,
        "razorpayKeyId": RAZORPAY_KEY_ID,
        "key": RAZORPAY_KEY_ID,
        "razorpay_enabled": bool(rz_client)
    }


# =========================================================
# ADMIN LOGIN
# =========================================================
@app.post("/api/admin/login")
def admin_login(payload: AdminLoginRequest):
    username = safe_str(payload.username)
    password = safe_str(payload.password)

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    if username != ADMIN_USERNAME or password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin credentials")

    token = secrets.token_hex(32)
    ADMIN_SESSIONS[token] = {
        "username": username,
        "created_at": datetime.utcnow().isoformat()
    }

    return {
        "ok": True,
        "message": "Admin login successful",
        "token": token,
        "admin": username
    }


@app.get("/api/admin/me")
def admin_me(authorization: Optional[str] = Header(default=None)):
    session = require_admin_token(authorization)
    return {
        "ok": True,
        "admin": session.get("username"),
        "created_at": session.get("created_at")
    }


@app.post("/api/admin/logout")
def admin_logout(authorization: Optional[str] = Header(default=None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization required")

    token = authorization.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()

    ADMIN_SESSIONS.pop(token, None)

    return {
        "ok": True,
        "message": "Admin logged out successfully"
    }


# =========================================================
# CUSTOMER OTP LOGIN
# =========================================================
@app.post("/api/auth/send-otp")
def send_otp(payload: SendOTPRequest):
    mobile = payload.mobile or payload.phone or ""
    otp = generate_otp()
    created_at = datetime.utcnow()
    expires_at = created_at + timedelta(minutes=OTP_EXPIRY_MINUTES)

    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute("DELETE FROM otp_sessions WHERE mobile = ?", (mobile,))
        cur.execute("""
            INSERT INTO otp_sessions (
                mobile, otp_code, attempts, verified, expires_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            mobile,
            otp,
            0,
            0,
            expires_at.isoformat(),
            created_at.isoformat()
        ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        raise HTTPException(status_code=500, detail="Unable to create OTP session right now")
    finally:
        conn.close()

    try:
        provider_response = send_msg91_otp(mobile, otp)
    except Exception as e:
        logger.exception("OTP send failed for mobile=%s", mobile)
        raise HTTPException(status_code=500, detail="Unable to send OTP right now")

    resp = {
        "ok": True,
        "message": "OTP sent successfully",
        "mobile": mobile,
        "phone": mobile,
        "expires_in_minutes": OTP_EXPIRY_MINUTES,
        "provider": "msg91"
    }

    if OTP_BYPASS:
        resp["debug_otp"] = otp

    if provider_response:
        resp["provider_response"] = provider_response

    return resp


@app.post("/api/auth/verify-otp")
def verify_otp(payload: VerifyOTPRequest):
    mobile = payload.mobile or payload.phone or ""

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT * FROM otp_sessions
        WHERE mobile = ?
        ORDER BY id DESC
        LIMIT 1
    """, (mobile,))
    row = cur.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="OTP session not found")

    if safe_int(row["verified"], 0) == 1:
        conn.close()
        return {
            "ok": True,
            "message": "Already verified",
            "mobile": mobile,
            "phone": mobile,
            "verified": True
        }

    attempts = safe_int(row["attempts"], 0)
    if attempts >= OTP_MAX_ATTEMPTS:
        conn.close()
        raise HTTPException(status_code=400, detail="Maximum OTP attempts exceeded")

    expires_at_raw = safe_str(row["expires_at"])
    try:
        expires_at = datetime.fromisoformat(expires_at_raw)
    except Exception:
        expires_at = datetime.utcnow() - timedelta(minutes=1)

    if datetime.utcnow() > expires_at:
        conn.close()
        raise HTTPException(status_code=400, detail="OTP expired")

    if safe_str(row["otp_code"]) != safe_str(payload.otp):
        cur.execute("""
            UPDATE otp_sessions
            SET attempts = attempts + 1
            WHERE id = ?
        """, (row["id"],))
        conn.commit()
        conn.close()
        raise HTTPException(status_code=400, detail="Invalid OTP")

    verified_at = datetime.utcnow().isoformat()

    try:
        cur.execute("""
            UPDATE otp_sessions
            SET verified = 1
            WHERE id = ?
        """, (row["id"],))
        upsert_customer_profile_after_otp(cur, mobile, verified_at)
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise HTTPException(status_code=500, detail="Unable to update customer profile right now")
    finally:
        conn.close()

    return {
        "ok": True,
        "message": "Login successful",
        "mobile": mobile,
        "phone": mobile,
        "verified": True
    }


@app.get("/api/customer/profile")
def get_customer_profile(phone: str = Query(...)):
    digits = clean_phone(phone)
    if len(digits) != 10:
        raise HTTPException(status_code=400, detail="Phone must be 10 digits")
    if not is_phone_verified(digits):
        logger.warning("Unauthorized profile access attempt for phone=%s", digits)
        raise HTTPException(status_code=401, detail="Unauthorized")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM customer_profiles WHERE phone = ?", (digits,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Customer profile not found")

    return {
        "ok": True,
        "profile": serialize_customer_profile(row)
    }


@app.post("/api/customer/profile")
def update_customer_profile(payload: CustomerProfileUpdateRequest):
    now_ts = datetime.utcnow().isoformat()
    email = safe_str(payload.email)
    consent_provided = "consent_marketing" in payload.model_fields_set
    consent_value = 1 if payload.consent_marketing else 0
    if not is_phone_verified(payload.phone):
        logger.warning("Unauthorized profile access attempt for phone=%s", payload.phone)
        raise HTTPException(status_code=401, detail="Unauthorized")

    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute("SELECT id FROM customer_profiles WHERE phone = ?", (payload.phone,))
        existing = cur.fetchone()
        if existing:
            cur.execute(
                """
                UPDATE customer_profiles
                SET
                    name = CASE WHEN ? <> '' THEN ? ELSE name END,
                    email = CASE WHEN ? <> '' THEN ? ELSE email END,
                    consent_marketing = CASE WHEN ? THEN ? ELSE consent_marketing END,
                    last_activity = ?,
                    source = ?
                WHERE phone = ?
                """,
                (
                    safe_str(payload.name),
                    safe_str(payload.name),
                    email,
                    email,
                    1 if consent_provided else 0,
                    consent_value,
                    now_ts,
                    "website_profile_update",
                    payload.phone
                )
            )
        else:
            cur.execute(
                """
                INSERT INTO customer_profiles (
                    phone, name, email, consent_marketing, source, last_activity
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.phone,
                    safe_str(payload.name),
                    email,
                    consent_value if consent_provided else 1,
                    "website_profile_update",
                    now_ts
                )
            )
        cur.execute("SELECT * FROM customer_profiles WHERE phone = ?", (payload.phone,))
        row = cur.fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise HTTPException(status_code=500, detail="Unable to save customer profile right now")
    finally:
        conn.close()

    return {
        "ok": True,
        "message": "Customer profile saved successfully",
        "profile": serialize_customer_profile(row)
    }


# =========================================================
# AI ITINERARY
# =========================================================

# =========================================================
# PARTNER PORTAL
# =========================================================
@app.post("/api/partners/register")
def partner_register(payload: PartnerRegisterRequest):
    partners = get_partners_collection()

    if partners.find_one({"mobile": payload.mobile}):
        raise HTTPException(status_code=409, detail="Partner already registered with this mobile number")

    now_ts = datetime.utcnow().isoformat()
    partner_doc = {
        "partner_type": payload.partner_type,
        "business_name": safe_str(payload.business_name),
        "contact_person": safe_str(payload.contact_person),
        "mobile": payload.mobile,
        "email": safe_str(payload.email),
        "password_hash": hash_password(payload.password),
        "created_at": now_ts,
        "updated_at": now_ts
    }

    inserted = partners.insert_one(partner_doc)
    partner_doc["_id"] = inserted.inserted_id

    token = secrets.token_urlsafe(32)
    PARTNER_SESSIONS[token] = {
        "partner_id": str(inserted.inserted_id),
        "mobile": payload.mobile,
        "partner_type": payload.partner_type,
        "created_at": now_ts
    }

    return {
        "ok": True,
        "message": "Partner registered successfully",
        "token": token,
        "partner": serialize_partner(partner_doc)
    }


@app.post("/api/partners/login")
def partner_login(payload: PartnerLoginRequest):
    partners = get_partners_collection()
    partner = partners.find_one({"mobile": payload.mobile})
    if not partner or not verify_password(payload.password, safe_str(partner.get("password_hash"))):
        raise HTTPException(status_code=401, detail="Invalid mobile number or password")

    token = secrets.token_urlsafe(32)
    PARTNER_SESSIONS[token] = {
        "partner_id": str(partner.get("_id")),
        "mobile": payload.mobile,
        "partner_type": safe_str(partner.get("partner_type")),
        "created_at": datetime.utcnow().isoformat()
    }

    return {
        "ok": True,
        "message": "Partner login successful",
        "token": token,
        "partner": serialize_partner(partner)
    }


@app.post("/api/partners/rates")
def create_partner_rate(payload: PartnerRateCreateRequest, authorization: Optional[str] = Header(default=None)):
    session = require_partner_token(authorization)
    partners = get_partners_collection()
    rates = get_partner_rates_collection()

    partner = partners.find_one({"_id": ObjectId(session["partner_id"])})
    if not partner:
        raise HTTPException(status_code=404, detail="Partner account not found")

    if safe_str(partner.get("partner_type")) != payload.partner_type:
        raise HTTPException(status_code=400, detail="Partner type mismatch")

    now_ts = datetime.utcnow().isoformat()
    rate_doc = payload.model_dump()
    rate_doc["business_name"] = safe_str(partner.get("business_name"))
    rate_doc["partner_id"] = str(partner.get("_id"))
    rate_doc["status"] = "pending"
    rate_doc["available"] = bool(payload.available)
    if rate_doc.get("partner_type") == "Hotel":
        rate_doc["rooms_available"] = safe_int(rate_doc.get("rooms_available"), safe_int(rate_doc.get("total_rooms"), 0))
    rate_doc["created_at"] = now_ts
    rate_doc["updated_at"] = now_ts

    inserted = rates.insert_one(rate_doc)
    rate_doc["_id"] = inserted.inserted_id

    return {"ok": True, "message": "Rate saved successfully", "rate": serialize_partner_rate(rate_doc)}


@app.get("/api/partners/rates")
def list_partner_rates(authorization: Optional[str] = Header(default=None)):
    session = require_partner_token(authorization)
    items = list(
        get_partner_rates_collection()
        .find({"partner_id": session["partner_id"]})
        .sort("created_at", -1)
    )
    return {"ok": True, "rates": [serialize_partner_rate(item) for item in items]}


@app.put("/api/partners/rates/{rate_id}")
def update_partner_rate(rate_id: str, payload: PartnerRateUpdateRequest, authorization: Optional[str] = Header(default=None)):
    session = require_partner_token(authorization)
    rates = get_partner_rates_collection()

    try:
        oid = ObjectId(rate_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid rate id")

    existing = rates.find_one({"_id": oid, "partner_id": session["partner_id"]})
    if not existing:
        raise HTTPException(status_code=404, detail="Rate not found")

    if safe_str(existing.get("partner_type")) != payload.partner_type:
        raise HTTPException(status_code=400, detail="Partner type cannot be changed")

    update_doc = payload.model_dump()
    update_doc["partner_id"] = session["partner_id"]
    update_doc["business_name"] = safe_str(existing.get("business_name")) or safe_str(payload.business_name)
    update_doc["status"] = "pending"
    update_doc["available"] = bool(payload.available)
    if update_doc.get("partner_type") == "Hotel":
        update_doc["rooms_available"] = safe_int(update_doc.get("rooms_available"), safe_int(update_doc.get("total_rooms"), 0))
    update_doc["updated_at"] = datetime.utcnow().isoformat()

    rates.update_one({"_id": oid}, {"$set": update_doc})
    updated = rates.find_one({"_id": oid})
    return {"ok": True, "message": "Rate updated successfully", "rate": serialize_partner_rate(updated)}


@app.delete("/api/partners/rates/{rate_id}")
def delete_partner_rate(rate_id: str, authorization: Optional[str] = Header(default=None)):
    session = require_partner_token(authorization)

    try:
        oid = ObjectId(rate_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid rate id")

    result = get_partner_rates_collection().delete_one({"_id": oid, "partner_id": session["partner_id"]})
    if not result.deleted_count:
        raise HTTPException(status_code=404, detail="Rate not found")

    return {"ok": True, "message": "Rate deleted successfully"}


@app.post("/api/ai/itinerary")
def generate_itinerary(payload: PlannerRequest):
    data = payload.model_dump()
    partner_context = get_partner_context_for_itinerary(data)

    try:
        itinerary = call_openai_json(build_itinerary_prompt(data, partner_context=partner_context))
        source = "openai"
    except Exception as e:
        itinerary = fallback_itinerary(data)
        source = "fallback"
        print(f"AI itinerary fallback used: {e}")

    itinerary = apply_partner_context_to_itinerary(itinerary, partner_context)

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


# =========================================================
# PILGRIMAGE
# =========================================================
@app.post("/api/pilgrimage/generate")
def generate_pilgrimage(payload: PilgrimageRequest):
    data = payload.model_dump()

    if not data.get("destination"):
        data["destination"] = data.get("destinationState") or "Pilgrimage"

    if not data.get("endDate"):
        data["endDate"] = data.get("startDate")

    pricing = calculate_pilgrimage_price(data)

    try:
        itinerary = call_openai_json(build_pilgrimage_prompt(data, pricing))
        source = "openai"
        itinerary["pricing"] = pricing
    except Exception as e:
        itinerary = fallback_pilgrimage_itinerary(data, pricing)
        source = "fallback"
        print(f"Pilgrimage AI fallback used: {e}")

    try:
        send_itinerary_enquiry_email(data, itinerary)
    except Exception as email_error:
        print(f"Failed to send pilgrimage enquiry email: {email_error}")

    return {
        "ok": True,
        "source": source,
        "bookingRef": f"HKE-PIL-{int(datetime.now().timestamp())}",
        "pricing": pricing,
        "itinerary": itinerary
    }


# =========================================================
# ITINERARY CHANGE REQUESTS
# =========================================================
@app.post("/api/request-itinerary-change")
def request_itinerary_change(payload: ItineraryChangeRequestCreate):
    conn = get_db()
    cur = conn.cursor()

    now_ts = datetime.utcnow().isoformat()

    try:
        cur.execute("""
            INSERT INTO itinerary_change_requests (
                booking_ref,
                customer_phone,
                customer_name,
                destination,
                request_text,
                status,
                admin_note,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            safe_str(payload.booking_ref),
            clean_phone(payload.customer_phone),
            safe_str(payload.customer_name),
            safe_str(payload.destination),
            safe_str(payload.request_text),
            "pending",
            "",
            now_ts,
            now_ts
        ))
        conn.commit()
        request_id = cur.lastrowid
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Unable to save itinerary change request right now")
    finally:
        conn.close()

    return {
        "ok": True,
        "message": "Itinerary change request submitted successfully",
        "request_id": request_id,
        "status": "pending"
    }


@app.get("/api/itinerary-change-requests")
def get_itinerary_change_requests(phone: str = Query(...)):
    clean = clean_phone(phone)
    if len(clean) != 10:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, booking_ref, customer_phone, customer_name, destination,
               request_text, status, admin_note, created_at, updated_at
        FROM itinerary_change_requests
        WHERE customer_phone = ?
        ORDER BY id DESC
    """, (clean,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    return {"ok": True, "items": rows}


@app.get("/api/admin/itinerary-requests")
def admin_itinerary_requests(authorization: Optional[str] = Header(default=None)):
    require_admin_token(authorization)

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, booking_ref, customer_phone, customer_name, destination,
               request_text, status, admin_note, created_at, updated_at
        FROM itinerary_change_requests
        ORDER BY id DESC
        LIMIT 300
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    return {"ok": True, "items": rows}


@app.post("/api/admin/itinerary-requests/{request_id}/review")
def admin_review_itinerary_request(
    request_id: int,
    payload: ItineraryChangeRequestReview,
    authorization: Optional[str] = Header(default=None)
):
    admin = require_admin_token(authorization)

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT * FROM itinerary_change_requests WHERE id = ?", (request_id,))
    row = cur.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Itinerary change request not found")

    now_ts = datetime.utcnow().isoformat()
    admin_note = safe_str(payload.admin_note)

    try:
        cur.execute("""
            UPDATE itinerary_change_requests
            SET status = ?, admin_note = ?, updated_at = ?
            WHERE id = ?
        """, (
            payload.status,
            admin_note,
            now_ts,
            request_id
        ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Unable to update itinerary request right now")
    finally:
        conn.close()

    return {
        "ok": True,
        "message": "Itinerary request updated successfully",
        "request_id": request_id,
        "status": payload.status,
        "reviewed_by": admin.get("username"),
        "admin_note": admin_note
    }


# =========================================================
# PAYMENT
# =========================================================
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

    if payload.notes:
        for k, v in payload.notes.items():
            notes[str(k)] = safe_str(v)[:255]

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
            detail="Unable to create Razorpay order right now"
        )

    return {
        "ok": True,
        "key": RAZORPAY_KEY_ID,
        "razorpay_key_id": RAZORPAY_KEY_ID,
        "razorpayKeyId": RAZORPAY_KEY_ID,
        "order": order,
        "order_id": order.get("id"),
        "amount": amount_rupees,
        "currency": payload.currency,
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
        "booking_ref": "",
        "booking_status": "received",
        "payment_status": "paid",
        "message": "Payment verified successfully"
    }


@app.post("/api/payment/save-confirmation")
def save_payment_confirmation(payload: SavePaymentRequest):
    customer = payload.customer or {}
    itinerary = payload.itinerary or {}
    pricing = payload.pricing or {}
    payment = payload.payment or {}

    total_amount = safe_float(
        pricing.get("finalFare", pricing.get("total", pricing.get("grand_total", 0)))
    )
    paid_amount = safe_float(
        payment.get("paidAmount", payment.get("advancePaid", payment.get("amount", 0)))
    )
    remaining_amount = safe_float(
        payment.get("remainingAmount", payment.get("remainingBalance", 0))
    )

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
            safe_str(customer.get("fromLocation", customer.get("startPoint"))),
            safe_str(customer.get("endPoint")),
            safe_str(customer.get("startDate")),
            safe_str(customer.get("endDate")),
            int(customer.get("travellers", 0) or 0),
            int(customer.get("rooms", 0) or 0),
            safe_str(itinerary.get("title")),
            safe_str(payment.get("paymentType", payment.get("paymentLabel"))),
            paid_amount,
            total_amount,
            remaining_amount,
            safe_str(payment.get("fullPaymentDeadline", payment.get("dueDate"))),
            safe_str(payment.get("nextScheduleText")),
            safe_str(payment.get("razorpayOrderId")),
            safe_str(payment.get("razorpayPaymentId")),
            safe_str(payment.get("paidAt", datetime.utcnow().isoformat())),
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
            detail="Unable to save payment confirmation right now"
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


# =========================================================
# CUSTOMER ORDERS LOOKUP
# =========================================================
@app.get("/api/orders")
def list_orders(
    phone: Optional[str] = Query(default=None),
    mobile: Optional[str] = Query(default=None),
    booking_ref: Optional[str] = Query(default=None)
):
    conn = get_db()
    cur = conn.cursor()

    if booking_ref:
        cur.execute("""
            SELECT * FROM payments
            WHERE razorpay_order_id = ? OR razorpay_payment_id = ? OR trip_name = ?
            ORDER BY id DESC
        """, (booking_ref, booking_ref, booking_ref))
    else:
        lookup = phone or mobile
        if not lookup:
            conn.close()
            raise HTTPException(status_code=400, detail="Phone or booking_ref is required")

        clean = clean_phone(lookup)
        if len(clean) != 10:
            conn.close()
            raise HTTPException(status_code=400, detail="Invalid phone number")

        cur.execute("""
            SELECT * FROM payments
            WHERE customer_phone = ?
            ORDER BY id DESC
        """, (clean,))

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    orders = []
    for row in rows:
        booking_ref_value = (
            safe_str(row.get("razorpay_order_id"))
            or safe_str(row.get("razorpay_payment_id"))
            or f"HKE-{row.get('id')}"
        )

        orders.append({
            "booking_ref": booking_ref_value,
            "booking_status": "confirmed",
            "payment_status": "paid" if safe_float(row.get("remaining_amount")) <= 0 else "partially paid",
            "destination": safe_str(row.get("destination")),
            "start_date": safe_str(row.get("start_date")),
            "end_date": safe_str(row.get("end_date")),
            "customer_name": safe_str(row.get("customer_name")),
            "phone": safe_str(row.get("customer_phone")),
            "trip_name": safe_str(row.get("trip_name")),
            "full_payment_due_date": safe_str(row.get("full_payment_deadline"), "-"),
            "cancellable_until": safe_str(row.get("full_payment_deadline"), "-"),
            "total_amount_paise": money_to_paise(row.get("total_amount")),
            "paid_amount_paise": money_to_paise(row.get("paid_amount")),
            "remaining_amount_paise": money_to_paise(row.get("remaining_amount")),
            "itinerary": safe_str(row.get("raw_itinerary_json")),
            "pricing": safe_str(row.get("raw_pricing_json")),
            "customer": safe_str(row.get("raw_customer_json")),
        })

    return {"ok": True, "orders": orders}


# =========================================================
# ADMIN-PROTECTED BOOKING LIST
# =========================================================
@app.get("/api/admin/bookings")
def admin_bookings(authorization: Optional[str] = Header(default=None)):
    require_admin_token(authorization)

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, customer_name, customer_email, customer_phone, destination,
               start_date, end_date, travellers, rooms, trip_name, payment_type,
               paid_amount, total_amount, remaining_amount, razorpay_payment_id,
               razorpay_order_id, paid_at
        FROM payments
        ORDER BY id DESC
        LIMIT 200
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
