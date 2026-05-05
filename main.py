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
import threading
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import List, Optional, Any, Dict
from urllib.parse import quote_plus

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
try:
    import gspread
    from google.oauth2.service_account import Credentials as GoogleServiceAccountCredentials
except Exception:  # pragma: no cover - dependency may be unavailable in some environments
    gspread = None
    GoogleServiceAccountCredentials = None

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
ADMIN_CONTENT_STORE_PATH = os.getenv("ADMIN_CONTENT_STORE_PATH", "admin_content_store.json").strip()

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
ENABLE_ITINERARY_IMAGES = os.getenv("ENABLE_ITINERARY_IMAGES", "false").lower() == "true"
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
GOOGLE_ENQUIRY_SHEET_ID = os.getenv("GOOGLE_ENQUIRY_SHEET_ID", "").strip()
GOOGLE_BOOKING_SHEET_ID = os.getenv("GOOGLE_BOOKING_SHEET_ID", "").strip()
GOOGLE_ENQUIRY_TAB = os.getenv("GOOGLE_ENQUIRY_TAB", "Enquiries").strip() or "Enquiries"
GOOGLE_BOOKING_TAB = os.getenv("GOOGLE_BOOKING_TAB", "ConfirmedBookings").strip() or "ConfirmedBookings"
OWNER_WHATSAPP_NUMBER = os.getenv("OWNER_WHATSAPP_NUMBER", "919797294747").strip()

# Admin login
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123").strip()

# Mongo / partner portal
MONGODB_URI = os.getenv("MONGO_URI", os.getenv("MONGODB_URI", "")).strip()
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", os.getenv("MONGO_DB_NAME", "hke_db")).strip()

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

mongo_client = (
    MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    if MONGODB_URI
    else None
)
mongo_ready = False
GOOGLE_SHEETS_CLIENT = None

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


def default_admin_content_store() -> Dict[str, Any]:
    return {
        "homepage_packages": {},
        "images": {},
        "ai_destinations": [],
        "tour_packages": {},
        "pilgrimage_packages": [],
    }


def read_admin_content_store() -> Dict[str, Any]:
    fallback = default_admin_content_store()

    try:
        if not os.path.exists(ADMIN_CONTENT_STORE_PATH):
            return fallback

        with open(ADMIN_CONTENT_STORE_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh) or {}
    except Exception:
        logger.exception("Unable to read admin content store")
        return fallback

    data = fallback.copy()
    for key in data.keys():
        value = raw.get(key)
        if isinstance(data[key], dict):
            data[key] = value if isinstance(value, dict) else {}
        elif isinstance(data[key], list):
            data[key] = value if isinstance(value, list) else []

    return data


def write_admin_content_store(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = default_admin_content_store()

    for key in normalized.keys():
        value = (data or {}).get(key)
        if isinstance(normalized[key], dict):
            normalized[key] = value if isinstance(value, dict) else {}
        elif isinstance(normalized[key], list):
            normalized[key] = value if isinstance(value, list) else []

    try:
        with open(ADMIN_CONTENT_STORE_PATH, "w", encoding="utf-8") as fh:
            json.dump(normalized, fh, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Unable to write admin content store")
        raise HTTPException(status_code=500, detail="Unable to save admin content right now")

    return normalized


@app.on_event("startup")
def startup_event():
    init_db()
    initialize_mongo()


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


def get_collection(name: str):
    return get_mongo_db()[name]


def utc_now() -> datetime:
    return datetime.utcnow()


def utc_now_iso() -> str:
    return utc_now().isoformat()


def hash_otp_value(otp: str) -> str:
    return hashlib.sha256(safe_str(otp).encode("utf-8")).hexdigest()


def normalize_for_mongo(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_for_mongo(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_for_mongo(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_for_mongo(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def mongo_write_enabled() -> bool:
    return bool(mongo_client and mongo_ready)


def initialize_mongo():
    global mongo_ready

    if not mongo_client:
        logger.warning("MongoDB is not configured")
        mongo_ready = False
        return

    try:
        mongo_client.admin.command("ping")
        ensure_partner_indexes()
        ensure_app_mongo_indexes()
        mongo_ready = True
        logger.info("MongoDB connected successfully")
    except Exception:
        mongo_ready = False
        logger.exception("MongoDB connection failed")


def ensure_app_mongo_indexes():
    db = get_mongo_db()
    db["customers"].create_index("phone", unique=True)
    db["otp_sessions"].create_index("phone")
    db["otp_sessions"].create_index("expiresAt")
    db["ai_itineraries"].create_index("phone")
    db["ai_itineraries"].create_index("customerPhone")
    db["bookings"].create_index("phone")
    db["payments"].create_index("bookingId")
    db["payments"].create_index("phone")
    payment_indexes = db["payments"].index_information()
    for field_name in ("razorpayPaymentId", "razorpay_payment_id"):
        index_name = f"{field_name}_1"
        existing = payment_indexes.get(index_name)
        if existing and (not existing.get("sparse") or not existing.get("unique")):
            try:
                db["payments"].drop_index(index_name)
            except Exception:
                logger.exception("Unable to recreate sparse payment index for %s", field_name)
        db["payments"].create_index(field_name, unique=True, sparse=True)
    db["payments"].create_index("internalPaymentId", unique=True, sparse=True)
    db["whatsapp_logs"].create_index("phone")
    db["reviews"].create_index("phone")
    db["itinerary_edits"].create_index("phone")


def safe_mongo_write(action: str, func):
    if not mongo_write_enabled():
        return None

    try:
        return func()
    except Exception:
        logger.exception("MongoDB write failed during %s", action)
        return None


def upsert_customer_mongo(
    phone: str,
    *,
    name: str = "",
    email: str = "",
    last_destination: str = "",
    extra: Optional[Dict[str, Any]] = None
):
    digits = clean_phone(phone)
    if len(digits) != 10:
        return

    now_ts = utc_now_iso()
    set_doc = {
        "phone": digits,
        "updatedAt": now_ts,
    }
    if safe_str(name):
        set_doc["name"] = safe_str(name)
    if safe_str(email):
        set_doc["email"] = safe_str(email)
    if safe_str(last_destination):
        set_doc["lastDestination"] = safe_str(last_destination)
    if extra:
        set_doc.update(normalize_for_mongo(extra))

    safe_mongo_write(
        "upsert_customer",
        lambda: get_collection("customers").update_one(
            {"phone": digits},
            {
                "$set": set_doc,
                "$setOnInsert": {
                    "createdAt": now_ts,
                },
            },
            upsert=True
        )
    )


def save_otp_session_mongo(phone: str, otp: str, purpose: str, expires_at: datetime):
    digits = clean_phone(phone)
    if len(digits) != 10:
        return

    created_at = utc_now_iso()
    safe_mongo_write(
        "save_otp_session",
        lambda: get_collection("otp_sessions").update_many(
            {"phone": digits, "purpose": purpose, "verified": False},
            {"$set": {"superseded": True, "updatedAt": created_at}}
        )
    )
    safe_mongo_write(
        "insert_otp_session",
        lambda: get_collection("otp_sessions").insert_one({
            "phone": digits,
            "otp_hash": hash_otp_value(otp),
            "purpose": purpose,
            "verified": False,
            "createdAt": created_at,
            "updatedAt": created_at,
            "expiresAt": expires_at.isoformat(),
        })
    )


def mark_otp_verified_mongo(phone: str, purpose: str):
    digits = clean_phone(phone)
    if len(digits) != 10:
        return

    verified_at = utc_now_iso()
    session = safe_mongo_write(
        "find_otp_session",
        lambda: get_collection("otp_sessions").find_one(
            {"phone": digits, "purpose": purpose},
            sort=[("createdAt", -1)]
        )
    )
    if not session:
        return

    safe_mongo_write(
        "verify_otp_session",
        lambda: get_collection("otp_sessions").update_one(
            {"_id": session["_id"]},
            {
                "$set": {
                    "verified": True,
                    "verifiedAt": verified_at,
                    "updatedAt": verified_at,
                }
            }
        )
    )


def save_ai_itinerary_mongo(request_payload: Dict[str, Any], itinerary: Dict[str, Any], source: str):
    phone = clean_phone(safe_str(request_payload.get("phone")))
    now_ts = utc_now_iso()
    doc = {
        "name": safe_str(request_payload.get("name")),
        "phone": phone,
        "customerPhone": phone,
        "email": safe_str(request_payload.get("email")),
        "destination": safe_str(request_payload.get("destination")),
        "startDate": safe_str(request_payload.get("startDate")),
        "endDate": safe_str(request_payload.get("endDate")),
        "days": safe_int(request_payload.get("days")),
        "travellers": safe_int(request_payload.get("travellers")),
        "rooms": safe_int(request_payload.get("rooms")),
        "hotelClass": safe_str(request_payload.get("hotelClass")),
        "vehicle": safe_str(request_payload.get("vehicle")),
        "guide": safe_str(request_payload.get("guide")),
        "places": normalize_for_mongo(request_payload.get("places") or []),
        "interests": normalize_for_mongo(request_payload.get("travelStyle") or []),
        "travelStyle": normalize_for_mongo(request_payload.get("travelStyle") or []),
        "requestPayload": normalize_for_mongo(request_payload),
        "generatedItinerary": normalize_for_mongo(itinerary),
        "originalItinerary": normalize_for_mongo(itinerary),
        "latestItinerary": normalize_for_mongo(itinerary),
        "status": "draft",
        "source": source,
        "createdAt": now_ts,
        "updatedAt": now_ts,
    }
    safe_mongo_write(
        "save_ai_itinerary",
        lambda: get_collection("ai_itineraries").insert_one(doc)
    )
    upsert_customer_mongo(
        phone,
        name=safe_str(request_payload.get("name")),
        email=safe_str(request_payload.get("email")),
        last_destination=safe_str(request_payload.get("destination"))
    )


def save_itinerary_edit_mongo(
    customer_details: Dict[str, Any],
    instruction: str,
    current_itinerary: str,
    updated_itinerary: Dict[str, Any],
    source: str
):
    phone = clean_phone(safe_str(customer_details.get("phone")))
    now_ts = utc_now_iso()
    linked_itinerary = None

    if mongo_write_enabled() and len(phone) == 10:
        try:
            linked_itinerary = get_collection("ai_itineraries").find_one(
                {"$or": [{"phone": phone}, {"customerPhone": phone}]},
                sort=[("updatedAt", -1)]
            )
        except Exception:
            logger.exception("MongoDB lookup failed during save_itinerary_edit")

    edit_doc = {
        "phone": phone,
        "customerPhone": phone,
        "name": safe_str(customer_details.get("name")),
        "email": safe_str(customer_details.get("email")),
        "destination": safe_str(customer_details.get("destination")),
        "instruction": safe_str(instruction),
        "currentItinerary": safe_str(current_itinerary),
        "updatedItinerary": normalize_for_mongo(updated_itinerary),
        "customerDetails": normalize_for_mongo(customer_details),
        "source": source,
        "createdAt": now_ts,
        "updatedAt": now_ts,
    }
    if linked_itinerary and linked_itinerary.get("_id"):
        edit_doc["aiItineraryId"] = str(linked_itinerary["_id"])

    safe_mongo_write(
        "save_itinerary_edit",
        lambda: get_collection("itinerary_edits").insert_one(edit_doc)
    )

    if linked_itinerary and linked_itinerary.get("_id"):
        safe_mongo_write(
            "update_ai_itinerary_latest",
            lambda: get_collection("ai_itineraries").update_one(
                {"_id": linked_itinerary["_id"]},
                {
                    "$set": {
                        "latestItinerary": normalize_for_mongo(updated_itinerary),
                        "updatedAt": now_ts,
                    }
                }
            )
        )

    if len(phone) == 10:
        upsert_customer_mongo(
            phone,
            name=safe_str(customer_details.get("name")),
            email=safe_str(customer_details.get("email")),
            last_destination=safe_str(customer_details.get("destination"))
        )


def build_booking_document(
    *,
    booking_id: str,
    phone: str,
    name: str = "",
    email: str = "",
    destination: str = "",
    trip_name: str = "",
    payment_type: str = "",
    amount: float = 0.0,
    currency: str = "INR",
    status: str = "",
    raw_payload: Optional[Dict[str, Any]] = None,
    payment: Optional[Dict[str, Any]] = None,
    itinerary: Optional[Dict[str, Any]] = None,
    pricing: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    now_ts = utc_now_iso()
    doc = {
        "bookingId": safe_str(booking_id),
        "amount": safe_float(amount),
        "currency": safe_str(currency, "INR"),
        "status": safe_str(status),
        "updatedAt": now_ts,
    }
    clean = clean_phone(phone)
    if len(clean) == 10:
        doc["phone"] = clean
    if safe_str(name):
        doc["name"] = safe_str(name)
    if safe_str(email):
        doc["email"] = safe_str(email)
    if safe_str(destination):
        doc["destination"] = safe_str(destination)
    if safe_str(trip_name):
        doc["tripName"] = safe_str(trip_name)
    if safe_str(payment_type):
        doc["paymentType"] = safe_str(payment_type)
    if raw_payload is not None:
        doc["rawPayload"] = normalize_for_mongo(raw_payload)
    if payment is not None:
        doc["payment"] = normalize_for_mongo(payment)
    if itinerary is not None:
        doc["itinerary"] = normalize_for_mongo(itinerary)
    if pricing is not None:
        doc["pricing"] = normalize_for_mongo(pricing)
    if extra:
        doc.update(normalize_for_mongo(extra))
    return doc


def build_booking_ref(value: str = "") -> str:
    raw = safe_str(value)
    if raw:
        return raw[:80]
    return f"HKE-{int(datetime.utcnow().timestamp())}"


def serialize_customer_booking_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    booking_ref = build_booking_ref(
        safe_str(doc.get("bookingRef"))
        or safe_str(doc.get("bookingId"))
        or safe_str(doc.get("razorpayOrderId"))
    )

    return {
        "bookingRef": booking_ref,
        "customerName": safe_str(doc.get("name")),
        "customerPhone": safe_str(doc.get("phone")),
        "customerEmail": safe_str(doc.get("email")),
        "destination": safe_str(doc.get("destination")),
        "packageName": safe_str(doc.get("tripName")),
        "startDate": safe_str(doc.get("startDate")),
        "endDate": safe_str(doc.get("endDate")),
        "travellers": safe_int(doc.get("travellers")),
        "rooms": safe_int(doc.get("rooms")),
        "totalAmount": safe_float(doc.get("totalAmount", doc.get("amount"))),
        "paidAmount": safe_float(doc.get("paidAmount")),
        "remainingAmount": safe_float(doc.get("remainingAmount")),
        "paymentStatus": safe_str(doc.get("paymentStatus")),
        "bookingStatus": safe_str(doc.get("bookingStatus")),
        "createdAt": safe_str(doc.get("createdAt")),
        "updatedAt": safe_str(doc.get("updatedAt")),
        "fullPaymentDeadline": safe_str(doc.get("fullPaymentDeadline")),
        "razorpayOrderId": safe_str(doc.get("razorpayOrderId")),
        "razorpayPaymentId": safe_str(doc.get("razorpayPaymentId")),
        "itineraryStatus": safe_str(doc.get("itineraryStatus")),
        "itineraryGeneratedAt": safe_str(doc.get("itineraryGeneratedAt")),
        "latestItinerary": normalize_for_mongo(doc.get("latestItinerary") or doc.get("itinerary") or {}),
        "routeMap": normalize_for_mongo(doc.get("routeMap") or {}),
        "hotelInfo": normalize_for_mongo(doc.get("hotelInfo") or {}),
        "cabInfo": normalize_for_mongo(doc.get("cabInfo") or {}),
        "itineraryImages": normalize_for_mongo(doc.get("itineraryImages") or []),
    }


def upsert_booking_mongo(booking_doc: Dict[str, Any]):
    booking_id = safe_str(booking_doc.get("bookingId"))
    if not booking_id:
        return

    now_ts = utc_now_iso()
    safe_mongo_write(
        "upsert_booking",
        lambda: get_collection("bookings").update_one(
            {"bookingId": booking_id},
            {
                "$set": booking_doc,
                "$setOnInsert": {
                    "createdAt": now_ts,
                },
            },
            upsert=True
        )
    )


def upsert_payment_mongo(payment_doc: Dict[str, Any]):
    booking_id = safe_str(payment_doc.get("bookingId"))
    if not booking_id:
        return

    payment_doc = dict(payment_doc or {})
    razorpay_payment_id = safe_str(
        payment_doc.get("razorpayPaymentId") or payment_doc.get("razorpay_payment_id")
    )
    payment_doc.pop("razorpay_payment_id", None)
    if razorpay_payment_id:
        payment_doc["razorpayPaymentId"] = razorpay_payment_id
    else:
        payment_doc.pop("razorpayPaymentId", None)
        payment_doc["internalPaymentId"] = safe_str(
            payment_doc.get("internalPaymentId"),
            f"HKE-PAY-{int(datetime.utcnow().timestamp())}"
        )

    now_ts = utc_now_iso()
    safe_mongo_write(
        "upsert_payment",
        lambda: get_collection("payments").update_one(
            {"bookingId": booking_id},
            {
                "$set": payment_doc,
                "$setOnInsert": {
                    "createdAt": now_ts,
                },
            },
            upsert=True
        )
    )


def log_whatsapp_event(phone: str, message_type: str, message: str, status: str):
    digits = clean_phone(phone)
    if len(digits) != 10:
        return

    safe_mongo_write(
        "log_whatsapp_event",
        lambda: get_collection("whatsapp_logs").insert_one({
            "phone": digits,
            "messageType": safe_str(message_type),
            "message": safe_str(message),
            "status": safe_str(status),
            "createdAt": utc_now_iso(),
        })
    )


def generate_and_store_booking_itinerary(booking_doc: Dict[str, Any]) -> Dict[str, Any]:
    booking_id = safe_str(booking_doc.get("bookingId"))
    booking_ref = build_booking_ref(safe_str(booking_doc.get("bookingRef")) or booking_id)
    base_payload = {
        "name": safe_str(booking_doc.get("name")),
        "email": safe_str(booking_doc.get("email")),
        "phone": safe_str(booking_doc.get("phone")),
        "destination": safe_str(booking_doc.get("destination")),
        "packageName": safe_str(booking_doc.get("tripName")),
        "fromLocation": safe_str(booking_doc.get("fromLocation")),
        "endPoint": safe_str(booking_doc.get("endPoint")),
        "startDate": safe_str(booking_doc.get("startDate")),
        "endDate": safe_str(booking_doc.get("endDate")),
        "travellers": safe_int(booking_doc.get("travellers"), 2),
        "rooms": safe_int(booking_doc.get("rooms"), 1),
        "days": safe_int(booking_doc.get("days")),
        "budget": safe_str(booking_doc.get("budget")),
        "travelType": safe_str(booking_doc.get("travelType")),
        "hotelClass": safe_str(booking_doc.get("hotelClass")),
        "vehicle": safe_str(booking_doc.get("vehicle")),
        "guide": safe_str(booking_doc.get("guide")),
        "foodPreference": safe_str(booking_doc.get("foodPreference")),
        "needFood": bool(booking_doc.get("needFood", False)),
        "travelStyle": normalize_for_mongo(booking_doc.get("travelStyle") or []),
        "places": normalize_for_mongo(booking_doc.get("places") or []),
        "notes": safe_str(booking_doc.get("notes")),
    }

    result = {
        "bookingRef": booking_ref,
        "itineraryStatus": "failed",
        "itineraryGeneratedAt": utc_now_iso(),
        "latestItinerary": {},
        "routeMap": build_route_map(base_payload),
        "hotelInfo": build_hotel_info(base_payload),
        "cabInfo": build_cab_info(base_payload),
        "itineraryImages": normalize_for_mongo(booking_doc.get("itineraryImages") or []),
    }

    try:
        assets = generate_booking_itinerary_assets(base_payload)
        result.update({
            "itineraryStatus": "generated",
            "itineraryGeneratedAt": utc_now_iso(),
            "latestItinerary": assets.get("itinerary") or {},
            "routeMap": assets.get("routeMap") or result["routeMap"],
            "hotelInfo": assets.get("hotelInfo") or result["hotelInfo"],
            "cabInfo": assets.get("cabInfo") or result["cabInfo"],
            "itinerarySource": safe_str(assets.get("source")),
        })
        if not result["itineraryImages"]:
            result["itineraryImages"] = generate_itinerary_images_safe(
                {
                    **booking_doc,
                    "bookingRef": booking_ref,
                    "destination": base_payload.get("destination"),
                    "tripName": base_payload.get("packageName"),
                    "hotelClass": base_payload.get("hotelClass"),
                    "vehicle": base_payload.get("vehicle"),
                },
                result["latestItinerary"]
            )
    except Exception:
        logger.exception("Booking itinerary generation failed for %s", booking_ref)

    safe_mongo_write(
        "update_booking_itinerary_assets",
        lambda: get_collection("bookings").update_one(
            {"bookingId": booking_id},
            {"$set": normalize_for_mongo(result)}
        )
    )
    return result


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


def format_display_date_range(start_date: str, end_date: str) -> str:
    start = safe_str(start_date)
    end = safe_str(end_date)
    if start and end:
        return f"{start} to {end}"
    return start or end or "-"


def format_inr_display(value: Any) -> str:
    return f"\u20b9{round(safe_float(value, 0.0)):,}"


def get_google_sheets_client():
    global GOOGLE_SHEETS_CLIENT

    if GOOGLE_SHEETS_CLIENT is not None:
        return GOOGLE_SHEETS_CLIENT

    if not gspread or not GoogleServiceAccountCredentials:
        logger.warning("Google Sheets skipped: google client libraries not available")
        return None

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        if GOOGLE_SERVICE_ACCOUNT_JSON:
            credentials_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
            credentials = GoogleServiceAccountCredentials.from_service_account_info(
                credentials_info,
                scopes=scopes,
            )
        elif GOOGLE_APPLICATION_CREDENTIALS:
            if not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
                logger.warning("Google Sheets skipped: credentials file not found")
                return None
            credentials = GoogleServiceAccountCredentials.from_service_account_file(
                GOOGLE_APPLICATION_CREDENTIALS,
                scopes=scopes,
            )
        else:
            logger.info("Google Sheets skipped: GOOGLE_SERVICE_ACCOUNT_JSON not configured")
            return None
        GOOGLE_SHEETS_CLIENT = gspread.authorize(credentials)
        return GOOGLE_SHEETS_CLIENT
    except Exception as exc:
        logger.exception("Google Sheets skipped: unable to initialize client")
        logger.warning("Google Sheet write failed: %s", exc)
        return None


def get_google_sheet_worksheet(sheet_id: str, tab_name: str):
    if not sheet_id:
        logger.info("Google Sheets skipped: missing sheet id for tab %s", tab_name)
        return None

    client_obj = get_google_sheets_client()
    if not client_obj:
        return None

    try:
        spreadsheet = client_obj.open_by_key(sheet_id)
        return spreadsheet.worksheet(tab_name)
    except Exception as exc:
        logger.exception("Google Sheets skipped: unable to access sheet tab %s", tab_name)
        logger.warning("Google Sheet write failed: %s", exc)
        return None


def append_row_to_google_sheet(sheet_id: str, tab_name: str, row_values: List[Any], success_log: str):
    worksheet = get_google_sheet_worksheet(sheet_id, tab_name)
    if not worksheet:
        return

    try:
        worksheet.append_row([safe_str(value) for value in row_values], value_input_option="USER_ENTERED")
        logger.info(success_log)
    except Exception as exc:
        logger.exception("Google Sheets append failed for tab %s", tab_name)
        logger.warning("Google Sheet write failed: %s", exc)


def append_enquiry_to_sheet(data: Dict[str, Any]):
    if not GOOGLE_ENQUIRY_SHEET_ID:
        logger.info("Google Sheets skipped: GOOGLE_ENQUIRY_SHEET_ID not configured")
        return

    itinerary = data.get("itinerary") or {}
    enquiry = data.get("enquiry") or {}
    row = [
        utc_now_iso(),
        safe_str(enquiry.get("name")),
        clean_phone(safe_str(enquiry.get("phone"))),
        safe_str(enquiry.get("email")),
        safe_str(enquiry.get("fromLocation")),
        safe_str(enquiry.get("destination")),
        safe_str(enquiry.get("endPoint")),
        safe_str(enquiry.get("startDate")),
        safe_str(enquiry.get("endDate")),
        safe_int(enquiry.get("days")),
        safe_int(enquiry.get("travellers")),
        safe_int(enquiry.get("rooms")),
        safe_str(enquiry.get("budget")),
        safe_str(enquiry.get("travelType")),
        safe_str(enquiry.get("hotelCategory") or enquiry.get("hotelClass")),
        safe_str(enquiry.get("vehicle")),
        safe_str(enquiry.get("guide")),
        "Yes" if bool(enquiry.get("foodRequired", enquiry.get("needFood"))) else "No",
        safe_str(enquiry.get("foodPreference")),
        ", ".join([safe_str(place) for place in enquiry.get("places") or [] if safe_str(place)]),
        safe_str(enquiry.get("notes")),
        safe_str(itinerary.get("title") or data.get("generatedTitle")),
        safe_str(data.get("source"), "AI Planner"),
    ]
    append_row_to_google_sheet(
        GOOGLE_ENQUIRY_SHEET_ID,
        GOOGLE_ENQUIRY_TAB,
        row,
        "Enquiry saved to Google Sheet",
    )


def append_booking_to_sheet(data: Dict[str, Any]):
    if not GOOGLE_BOOKING_SHEET_ID:
        logger.info("Google Sheets skipped: GOOGLE_BOOKING_SHEET_ID not configured")
        return

    customer = data.get("customer") or {}
    payment = data.get("payment") or {}
    row = [
        utc_now_iso(),
        safe_str(data.get("bookingRef")),
        safe_str(customer.get("name")),
        clean_phone(safe_str(customer.get("phone"))),
        safe_str(customer.get("email")),
        safe_str(data.get("packageName")),
        safe_str(customer.get("destination")),
        safe_str(customer.get("startDate")),
        safe_str(customer.get("endDate")),
        safe_int(customer.get("travellers")),
        safe_int(customer.get("rooms")),
        safe_str(customer.get("vehicle")),
        safe_str(customer.get("hotelCategory") or customer.get("hotelClass")),
        safe_float(data.get("totalAmount")),
        safe_float(data.get("advancePaid")),
        safe_float(data.get("remainingAmount")),
        safe_str(data.get("paymentStatus")),
        safe_str(payment.get("razorpayOrderId")),
        safe_str(payment.get("razorpayPaymentId")),
    ]
    append_row_to_google_sheet(
        GOOGLE_BOOKING_SHEET_ID,
        GOOGLE_BOOKING_TAB,
        row,
        "Booking saved to Google Sheet",
    )


def send_owner_whatsapp_alert(message_type: str, data: Dict[str, Any]):
    message_type = safe_str(message_type).lower()
    owner_number = safe_str(OWNER_WHATSAPP_NUMBER)

    if not owner_number:
        logger.info("Owner WhatsApp alert skipped: owner number not configured")
        return

    if message_type == "enquiry":
        message = "\n".join([
            "New HKE Enquiry ✅",
            f"Name: {safe_str(data.get('name'))}",
            f"Phone: {clean_phone(safe_str(data.get('phone')))}",
            f"Destination: {safe_str(data.get('destination'))}",
            f"Dates: {format_display_date_range(safe_str(data.get('startDate')), safe_str(data.get('endDate')))}",
            f"Travellers: {safe_str(data.get('travellers'))}",
            f"Budget: {safe_str(data.get('budget'))}",
            f"Places: {', '.join([safe_str(place) for place in data.get('places') or [] if safe_str(place)]) or '-'}",
            "Source: AI Planner",
        ])
    elif message_type == "booking":
        payment = data.get("payment") or {}
        message = "\n".join([
            "New Confirmed Booking ✅",
            f"Booking Ref: {safe_str(data.get('bookingRef'))}",
            f"Name: {safe_str(data.get('name'))}",
            f"Phone: {clean_phone(safe_str(data.get('phone')))}",
            f"Package: {safe_str(data.get('packageName'))}",
            f"Dates: {format_display_date_range(safe_str(data.get('startDate')), safe_str(data.get('endDate')))}",
            f"Total: {format_inr_display(data.get('totalAmount'))}",
            f"Paid: {format_inr_display(data.get('advancePaid'))}",
            f"Remaining: {format_inr_display(data.get('remainingAmount'))}",
            f"Status: {safe_str(data.get('paymentStatus'))}",
        ])
    else:
        logger.info("Owner WhatsApp alert skipped: unsupported message type %s", message_type)
        return

    logger.info("Owner WhatsApp alert skipped: provider not configured")
    log_whatsapp_event(owner_number, f"owner_{message_type}_alert", message, "skipped")


def run_in_background(label: str, func, *args):
    def runner():
        try:
            func(*args)
        except Exception:
            logger.exception("%s failed", label)

    thread = threading.Thread(target=runner, name=label, daemon=True)
    thread.start()


def _append_enquiry_to_sheet_worker(data: Dict[str, Any]):
    try:
        append_enquiry_to_sheet(data)
    except Exception:
        logger.exception("Enquiry Google Sheet hook failed")


def _append_booking_to_sheet_worker(data: Dict[str, Any]):
    try:
        append_booking_to_sheet(data)
    except Exception:
        logger.exception("Booking Google Sheet hook failed")


def _send_owner_whatsapp_alert_worker(message_type: str, data: Dict[str, Any]):
    try:
        send_owner_whatsapp_alert(message_type, data)
    except Exception:
        logger.exception("Owner WhatsApp alert failed")


def safe_append_enquiry_to_sheet(data: Dict[str, Any]):
    run_in_background("safe_append_enquiry_to_sheet", _append_enquiry_to_sheet_worker, data)


def safe_append_booking_to_sheet(data: Dict[str, Any]):
    run_in_background("safe_append_booking_to_sheet", _append_booking_to_sheet_worker, data)


def safe_send_owner_whatsapp_alert(message_type: str, data: Dict[str, Any]):
    run_in_background("safe_send_owner_whatsapp_alert", _send_owner_whatsapp_alert_worker, message_type, data)


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

    url = "https://control.msg91.com/api/v5/oneapi/api/flow/otp-login-flow-hke/run"
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
        "MSG91 OneAPI OTP provider status for mobile=%s status=%s",
        mobile,
        response.status_code
    )

    has_error = response_data.get("hasError") if isinstance(response_data, dict) else None
    provider_status = safe_str(response_data.get("status")).lower() if isinstance(response_data, dict) else ""
    provider_errors = response_data.get("errors") if isinstance(response_data, dict) else None
    errors_list = provider_errors if isinstance(provider_errors, list) else []

    if response.status_code not in (200, 201, 202):
        logger.error(
            "MSG91 OneAPI OTP send failed for mobile=%s status=%s response=%s",
            mobile,
            response.status_code,
            response.text
        )
        raise RuntimeError(f"MSG91 send failed: {response.text}")

    if has_error is True or provider_status in ("error", "failed") or errors_list:
        logger.error(
            "MSG91 OneAPI OTP rejected for mobile=%s response=%s",
            mobile,
            json.dumps(response_data, default=str)
        )
        raise RuntimeError(f"MSG91 rejected OTP: {response_data}")

    if has_error is False or provider_status == "success":
        return response_data

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


def serialize_admin_mongo_value(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): serialize_admin_mongo_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_admin_mongo_value(item) for item in value]
    return value


def sanitize_admin_mongo_doc(doc: Dict[str, Any], blocked_fields: Optional[set] = None) -> Dict[str, Any]:
    blocked = blocked_fields or set()
    clean_doc: Dict[str, Any] = {}

    for key, value in (doc or {}).items():
        if key in blocked:
            continue
        clean_doc[str(key)] = serialize_admin_mongo_value(value)

    return clean_doc


def get_admin_collection_snapshot(
    name: str,
    *,
    blocked_fields: Optional[set] = None,
    sort_fields: Optional[List[Any]] = None,
    limit: int = 100
) -> Dict[str, Any]:
    if not mongo_write_enabled():
        return {"total": 0, "items": []}

    collection = get_collection(name)
    total = collection.count_documents({})
    rows = list(collection.find({}).sort(sort_fields or [("updatedAt", -1), ("createdAt", -1)]).limit(limit))
    items = [sanitize_admin_mongo_doc(row, blocked_fields) for row in rows]
    return {"total": total, "items": items}


def get_public_admin_content() -> Dict[str, Any]:
    return read_admin_content_store()


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


def build_google_maps_search_url(query: str) -> str:
    clean_query = safe_str(query)
    if not clean_query:
        return ""
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(clean_query)}"


def build_google_maps_directions_url(origin: str, destination: str) -> str:
    clean_origin = safe_str(origin)
    clean_destination = safe_str(destination)
    if not clean_origin and not clean_destination:
        return ""

    base = "https://www.google.com/maps/dir/?api=1&travelmode=driving"
    if clean_origin:
        base += f"&origin={quote_plus(clean_origin)}"
    if clean_destination:
        base += f"&destination={quote_plus(clean_destination)}"
    return base


def build_route_map(data: Dict[str, Any]) -> Dict[str, Any]:
    start_point = safe_str(data.get("fromLocation", data.get("startPoint")))
    destination = safe_str(data.get("destination"))
    end_point = safe_str(data.get("endPoint", destination))
    places = [safe_str(item) for item in (data.get("places") or []) if safe_str(item)]

    search_query_parts = [part for part in [destination] + places if part]
    search_query = ", ".join(search_query_parts) or end_point or start_point
    directions_origin = start_point or destination
    directions_destination = end_point or destination or start_point

    return {
        "startPoint": start_point,
        "destination": destination,
        "endPoint": end_point,
        "places": places,
        "googleMapsSearchUrl": build_google_maps_search_url(search_query),
        "googleMapsDirectionsUrl": build_google_maps_directions_url(directions_origin, directions_destination),
    }


def build_hotel_info(data: Dict[str, Any]) -> Dict[str, Any]:
    destination = safe_str(data.get("destination"))
    location = safe_str(data.get("hotelLocation")) or destination
    start_date = safe_str(data.get("startDate"))
    end_date = safe_str(data.get("endDate"))
    return {
        "name": "To be assigned by HKE",
        "location": location,
        "googleMapsUrl": build_google_maps_search_url(location),
        "checkInDate": start_date,
        "checkOutDate": end_date,
        "status": "pending_assignment",
    }


def build_cab_info(data: Dict[str, Any]) -> Dict[str, Any]:
    pickup_location = safe_str(data.get("fromLocation", data.get("startPoint")))
    return {
        "driverName": "To be assigned by HKE",
        "vehicle": safe_str(data.get("vehicle"), "Vehicle to be assigned by HKE"),
        "pickupLocation": pickup_location,
        "pickupDate": safe_str(data.get("startDate")),
        "status": "pending_assignment",
    }


def get_fallback_itinerary_image(destination: str, index: int = 0) -> str:
    dest = safe_str(destination).lower()
    candidates = ["media/hero-banner.jpg"]

    if "kashmir" in dest:
        candidates = ["media/kashmir-package.jpg", "media/kashmir.jpg", "media/kashmir-package.png"]
    elif "manali" in dest:
        candidates = ["media/manali.jpg", "media/manali-package.jpg", "media/manali-package.png", "media/manali1.jpg"]
    elif "kerala" in dest:
        candidates = ["media/kerala-package.jpg", "media/kerala.jpg", "media/kerala-package.png"]
    elif "leh" in dest or "ladakh" in dest:
        candidates = ["media/leh-ladakh.jpg", "media/leh-ladakh2.png"]

    for path in candidates:
        if os.path.exists(os.path.join("godaddy-frontend", path.replace("/", os.sep))):
            return path
    return "media/hero-banner.jpg"


def build_itinerary_image_prompts(booking: Dict[str, Any], itinerary: Dict[str, Any]) -> List[Dict[str, Any]]:
    destination = safe_str(booking.get("destination") or booking.get("tripName"))
    hotel_class = safe_str(booking.get("hotelClass"), "premium")
    vehicle = safe_str(booking.get("vehicle"), "touring vehicle")
    route = safe_str((itinerary.get("meta") or {}).get("route"))

    prompts = [{
        "title": f"{destination or 'HKE Trip'} Cover",
        "prompt": (
            f"Luxury travel brochure cover for Himalayan Kerala Expeditions in {destination or 'India'}, "
            f"cinematic landscape, premium {hotel_class} travel mood, elegant Indian tourism aesthetic, "
            f"scenic route {route or destination}, polished destination marketing photograph."
        ).strip(),
        "fallbackImage": get_fallback_itinerary_image(destination, 0),
    }]

    for idx, day in enumerate((itinerary.get("days") or [])[:3], start=1):
        day_title = safe_str(day.get("title"), f"Day {idx}")
        day_route = safe_str(day.get("route"), destination)
        prompts.append({
            "title": f"Day {idx} - {day_title}",
            "prompt": (
                f"Travel photography for {destination or 'India'} itinerary day {idx}, {day_title}, route {day_route}, "
                f"comfortable {vehicle}, premium guided holiday scene, vibrant natural lighting, realistic destination visuals."
            ).strip(),
            "fallbackImage": get_fallback_itinerary_image(destination, idx),
        })

    return prompts[:4]


def generate_itinerary_images_safe(booking: Dict[str, Any], itinerary: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompts = build_itinerary_image_prompts(booking, itinerary)
    now_ts = utc_now_iso()

    fallback_items = [{
        "title": item["title"],
        "prompt": item["prompt"],
        "imageUrl": item["fallbackImage"],
        "source": "fallback",
        "createdAt": now_ts,
    } for item in prompts]

    if not ENABLE_ITINERARY_IMAGES or not client:
        return fallback_items

    results: List[Dict[str, Any]] = []
    for item in prompts:
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=item["prompt"],
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = safe_str((response.data[0] or {}).url if getattr(response, "data", None) else "")
            if image_url:
                results.append({
                    "title": item["title"],
                    "prompt": item["prompt"],
                    "imageUrl": image_url,
                    "source": "openai",
                    "createdAt": now_ts,
                })
                continue
        except Exception:
            logger.exception("Itinerary image generation failed for booking=%s title=%s", safe_str(booking.get("bookingRef")), item["title"])

        results.append({
            "title": item["title"],
            "prompt": item["prompt"],
            "imageUrl": item["fallbackImage"],
            "source": "fallback",
            "createdAt": now_ts,
        })

    return results


def build_booking_itinerary_request(data: Dict[str, Any]) -> Dict[str, Any]:
    start_date = safe_str(data.get("startDate"))
    end_date = safe_str(data.get("endDate"))
    days_value = safe_int(data.get("days"))
    if days_value <= 0 and start_date and end_date:
        try:
            days_value = max(1, (parse_iso_date(end_date) - parse_iso_date(start_date)).days + 1)
        except Exception:
            days_value = 5
    if days_value <= 0:
        days_value = 5

    destination = safe_str(data.get("destination")) or safe_str(data.get("packageName")) or "HKE Trip"
    places = [safe_str(item) for item in (data.get("places") or []) if safe_str(item)]
    if not places and destination:
        places = [destination]

    payload = {
        "name": safe_str(data.get("name"), "HKE Customer"),
        "email": safe_str(data.get("email"), "guest@hke.local"),
        "phone": clean_phone(safe_str(data.get("phone"))),
        "fromLocation": safe_str(data.get("fromLocation", data.get("startPoint"))) or destination,
        "destination": destination,
        "endPoint": safe_str(data.get("endPoint")) or destination,
        "startDate": start_date or utc_now().strftime("%Y-%m-%d"),
        "days": days_value,
        "endDate": end_date or start_date or utc_now().strftime("%Y-%m-%d"),
        "travellers": max(1, safe_int(data.get("travellers"), 2)),
        "rooms": max(1, safe_int(data.get("rooms"), 1)),
        "budget": safe_str(data.get("budget"), "Standard"),
        "travelType": safe_str(data.get("travelType"), "Family"),
        "hotelClass": safe_str(data.get("hotelClass"), "Standard"),
        "vehicle": safe_str(data.get("vehicle"), "SUV"),
        "guide": safe_str(data.get("guide"), "Without Guide"),
        "needFood": bool(data.get("needFood", False)),
        "foodPreference": safe_str(data.get("foodPreference"), "Flexible"),
        "travelStyle": normalize_for_mongo(data.get("travelStyle") or []),
        "places": places,
        "notes": safe_str(data.get("notes")),
    }
    return payload


def generate_booking_itinerary_assets(data: Dict[str, Any]) -> Dict[str, Any]:
    itinerary_payload = build_booking_itinerary_request(data)
    partner_context = get_partner_context_for_itinerary(itinerary_payload)

    try:
        logger.info("Using OpenAI itinerary system")
        itinerary = call_openai_json(build_itinerary_prompt(itinerary_payload, partner_context=partner_context))
        source = "openai"
    except Exception:
        itinerary = fallback_itinerary(itinerary_payload)
        source = "fallback"

    itinerary = apply_partner_context_to_itinerary(itinerary, partner_context)
    route_map = build_route_map(itinerary_payload)
    hotel_info = build_hotel_info(itinerary_payload)
    cab_info = build_cab_info(itinerary_payload)

    return {
        "source": source,
        "itinerary": itinerary,
        "routeMap": route_map,
        "hotelInfo": hotel_info,
        "cabInfo": cab_info,
    }

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


class BookingItineraryGenerateRequest(BaseModel):
    phone: Optional[str] = None

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v):
        if v in (None, ""):
            return None
        digits = clean_phone(v)
        if len(digits) != 10:
            raise ValueError("Phone must be 10 digits")
        return digits


class BookingImagesGenerateRequest(BaseModel):
    phone: Optional[str] = None

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v):
        if v in (None, ""):
            return None
        digits = clean_phone(v)
        if len(digits) != 10:
            raise ValueError("Phone must be 10 digits")
        return digits


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
    lastDestination: Optional[str] = ""
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
        "mongo_configured": bool(MONGODB_URI),
        "mongo_connected": mongo_write_enabled(),
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
    created_at = utc_now()
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

    save_otp_session_mongo(mobile, otp, "login", expires_at)

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

    verified_at = utc_now_iso()

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

    mark_otp_verified_mongo(mobile, "login")
    upsert_customer_mongo(mobile, extra={"verified": True, "lastLoginAt": verified_at})

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
    now_ts = utc_now_iso()
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

    upsert_customer_mongo(
        payload.phone,
        name=safe_str(payload.name),
        email=email,
        last_destination=safe_str(payload.lastDestination)
    )

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
    logger.info("AI itinerary request received")
    partner_context = get_partner_context_for_itinerary(data)
    route_map = build_route_map(data)

    try:
        logger.info("Using OpenAI itinerary system")
        itinerary = call_openai_json(build_itinerary_prompt(data, partner_context=partner_context))
        source = "openai"
    except Exception as e:
        logger.warning(
            "OpenAI itinerary failed, using fallback: %s: %s",
            type(e).__name__,
            str(e)
        )
        itinerary = fallback_itinerary(data)
        source = "fallback"

    try:
        itinerary = apply_partner_context_to_itinerary(itinerary, partner_context)
    except Exception as e:
        logger.warning(
            "Partner context apply failed for itinerary: %s: %s",
            type(e).__name__,
            str(e)
        )

    try:
        send_itinerary_enquiry_email(data, itinerary)
    except Exception as email_error:
        logger.warning(
            "Failed to send itinerary enquiry email: %s: %s",
            type(email_error).__name__,
            str(email_error)
        )

    save_ai_itinerary_mongo(data, itinerary, source)
    log_whatsapp_event(
        data.get("phone", ""),
        "ai_itinerary",
        f"AI itinerary prepared for {safe_str(data.get('destination'))}",
        "generated"
    )
    safe_append_enquiry_to_sheet({
        "enquiry": data,
        "itinerary": itinerary,
        "source": "AI Planner",
    })
    safe_send_owner_whatsapp_alert("enquiry", data)

    return {
        "ok": True,
        "source": source,
        "itinerary": itinerary,
        "routeMap": route_map,
        "hotelInfo": build_hotel_info(data),
        "cabInfo": build_cab_info(data),
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
        save_itinerary_edit_mongo(customer_details, instruction, current_itinerary, itinerary, "openai")
        return {"ok": True, "source": "openai", "itinerary": itinerary}
    except Exception as e:
        fallback = fallback_itinerary(customer_details, edit_note=instruction)
        save_itinerary_edit_mongo(customer_details, instruction, current_itinerary, fallback, "fallback")
        return {
            "ok": True,
            "source": "fallback",
            "warning": str(e),
            "itinerary": fallback
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

    booking_ref = build_booking_ref(notes.get("booking_ref") or payload.receipt or "")
    total_amount = safe_float(notes.get("total_amount", amount_rupees))
    paid_amount = amount_rupees if safe_str(payload.payment_type).lower() in {"advance", "full", "custom"} else 0.0
    remaining_amount = max(0.0, total_amount - paid_amount)
    booking_status = "payment_pending"
    payment_status = "order_created"

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

    booking_doc = build_booking_document(
        booking_id=safe_str(order.get("id")) or receipt,
        phone=payload.phone,
        name=payload.name,
        email=payload.email,
        destination=safe_str(notes.get("destination")),
        trip_name=payload.trip_name,
        payment_type=payload.payment_type,
        amount=amount_rupees,
        currency=payload.currency,
        status="payment_order_created",
        raw_payload=payload.model_dump(),
        extra={
            "bookingRef": booking_ref,
            "receipt": receipt[:40],
            "razorpayOrderId": safe_str(order.get("id")),
            "razorpayPaymentId": "",
            "startDate": safe_str(notes.get("start_date")),
            "endDate": safe_str(notes.get("end_date")),
            "travellers": safe_int(notes.get("travellers")),
            "rooms": safe_int(notes.get("rooms")),
            "totalAmount": total_amount,
            "paidAmount": paid_amount,
            "remainingAmount": remaining_amount,
            "paymentStatus": payment_status,
            "bookingStatus": booking_status,
            "fullPaymentDeadline": safe_str(notes.get("full_payment_deadline")),
            "notes": notes,
        }
    )
    upsert_booking_mongo(booking_doc)
    upsert_payment_mongo({
        **booking_doc,
        "status": "order_created",
        "bookingRef": booking_ref,
        "amount": amount_rupees,
        "paymentStatus": payment_status,
        "bookingStatus": booking_status,
        "bookingId": safe_str(order.get("id")) or receipt,
    })
    upsert_customer_mongo(payload.phone, name=payload.name, email=payload.email)
    log_whatsapp_event(
        payload.phone,
        "payment_order",
        f"Payment order created for {safe_str(payload.trip_name)}",
        "generated"
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
        "booking_ref": booking_ref,
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

    booking_id = safe_str(payload.razorpay_order_id) or safe_str(payload.razorpay_payment_id)
    existing_booking = None
    if mongo_write_enabled():
        existing_booking = get_collection("bookings").find_one({"bookingId": booking_id})

    booking_doc = build_booking_document(
        booking_id=booking_id,
        phone=safe_str((existing_booking or {}).get("phone")),
        status="payment_verified",
        payment={
            "razorpayOrderId": payload.razorpay_order_id,
            "razorpayPaymentId": payload.razorpay_payment_id,
            "razorpaySignature": payload.razorpay_signature,
            "verified": True,
        },
        extra={
            "bookingRef": build_booking_ref(safe_str((existing_booking or {}).get("bookingRef")) or booking_id),
            "razorpayOrderId": payload.razorpay_order_id,
            "razorpayPaymentId": payload.razorpay_payment_id,
            "paymentStatus": "verified",
            "bookingStatus": safe_str((existing_booking or {}).get("bookingStatus"), "payment_received"),
        }
    )
    upsert_booking_mongo(booking_doc)
    upsert_payment_mongo({
        **booking_doc,
        "status": "verified",
    })

    return {
        "ok": True,
        "verified": True,
        "booking_ref": build_booking_ref(safe_str((existing_booking or {}).get("bookingRef")) or booking_id),
        "booking_status": safe_str((existing_booking or {}).get("bookingStatus"), "payment_received"),
        "payment_status": "verified",
        "message": "Payment verified successfully"
    }


@app.post("/api/payment/save-confirmation")
def save_payment_confirmation(payload: SavePaymentRequest):
    customer = payload.customer or {}
    itinerary = payload.itinerary or {}
    pricing = payload.pricing or {}
    payment = payload.payment or {}
    route_map = build_route_map(customer)
    hotel_info = build_hotel_info(customer)
    cab_info = build_cab_info(customer)

    total_amount = safe_float(
        pricing.get("finalFare", pricing.get("total", pricing.get("grand_total", 0)))
    )
    paid_amount = safe_float(
        payment.get("paidAmount", payment.get("advancePaid", payment.get("amount", 0)))
    )
    remaining_amount = safe_float(
        payment.get("remainingAmount", payment.get("remainingBalance", 0))
    )
    booking_ref = build_booking_ref(
        safe_str(payment.get("bookingRef"))
        or safe_str(customer.get("bookingRef"))
        or safe_str(payment.get("razorpayOrderId"))
        or safe_str(payment.get("razorpayPaymentId"))
    )
    payment_type = safe_str(payment.get("paymentType", payment.get("paymentLabel")))
    booking_status = "confirmed"
    payment_status = "paid" if remaining_amount <= 0 else "partially_paid"

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
            safe_str(itinerary.get("title") or customer.get("packageName") or customer.get("destination")),
            payment_type,
            paid_amount,
            total_amount,
            remaining_amount,
            safe_str(payment.get("fullPaymentDeadline", payment.get("dueDate"))),
            safe_str(payment.get("nextScheduleText")),
            safe_str(payment.get("razorpayOrderId")),
            safe_str(payment.get("razorpayPaymentId")),
            safe_str(payment.get("paidAt", utc_now_iso())),
            json.dumps(customer, ensure_ascii=False),
            json.dumps(itinerary, ensure_ascii=False),
            json.dumps(pricing, ensure_ascii=False),
            utc_now_iso()
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

    booking_id = (
        safe_str(payment.get("razorpayOrderId"))
        or safe_str(payment.get("razorpayPaymentId"))
        or booking_ref
    )
    booking_doc = build_booking_document(
        booking_id=booking_id,
        phone=safe_str(customer.get("phone")),
        name=safe_str(customer.get("name")),
        email=safe_str(customer.get("email")),
        destination=safe_str(customer.get("destination")),
        trip_name=safe_str(itinerary.get("title") or customer.get("packageName") or customer.get("destination")),
        payment_type=payment_type,
        amount=paid_amount,
        currency="INR",
        status="payment_confirmation_saved",
        payment=payment,
        itinerary=itinerary,
        pricing=pricing,
        raw_payload=payload.model_dump(),
        extra={
            "bookingRef": booking_ref,
            "startDate": safe_str(customer.get("startDate")),
            "endDate": safe_str(customer.get("endDate")),
            "travellers": safe_int(customer.get("travellers")),
            "rooms": safe_int(customer.get("rooms")),
            "days": safe_int(customer.get("days")),
            "fromLocation": safe_str(customer.get("fromLocation", customer.get("startPoint"))),
            "endPoint": safe_str(customer.get("endPoint")),
            "razorpayOrderId": safe_str(payment.get("razorpayOrderId")),
            "razorpayPaymentId": safe_str(payment.get("razorpayPaymentId")),
            "totalAmount": total_amount,
            "paidAmount": paid_amount,
            "remainingAmount": remaining_amount,
            "paymentStatus": payment_status,
            "bookingStatus": booking_status,
            "fullPaymentDeadline": safe_str(payment.get("fullPaymentDeadline", payment.get("dueDate"))),
            "budget": safe_str(customer.get("budget")),
            "travelType": safe_str(customer.get("travelType")),
            "hotelClass": safe_str(customer.get("hotelClass")),
            "vehicle": safe_str(customer.get("vehicle")),
            "guide": safe_str(customer.get("guide")),
            "foodPreference": safe_str(customer.get("foodPreference")),
            "needFood": bool(customer.get("needFood", False)),
            "travelStyle": normalize_for_mongo(customer.get("travelStyle") or []),
            "places": normalize_for_mongo(customer.get("places") or []),
            "notes": safe_str(customer.get("notes")),
            "routeMap": route_map,
            "hotelInfo": hotel_info,
            "cabInfo": cab_info,
            "itineraryImages": normalize_for_mongo(customer.get("itineraryImages") or []),
            "latestItinerary": normalize_for_mongo(itinerary),
            "itineraryStatus": "generated" if itinerary else "pending",
            "itineraryGeneratedAt": utc_now_iso() if itinerary else "",
        }
    )
    upsert_booking_mongo(booking_doc)
    upsert_payment_mongo({
        **booking_doc,
        "status": "confirmed",
        "bookingRef": booking_ref,
        "paidAmount": paid_amount,
        "totalAmount": total_amount,
        "remainingAmount": remaining_amount,
        "phone": clean_phone(safe_str(customer.get("phone"))),
        "paymentStatus": payment_status,
        "bookingStatus": booking_status,
        "amount": paid_amount,
    })
    upsert_customer_mongo(
        safe_str(customer.get("phone")),
        name=safe_str(customer.get("name")),
        email=safe_str(customer.get("email")),
        last_destination=safe_str(customer.get("destination"))
    )
    log_whatsapp_event(
        safe_str(customer.get("phone")),
        "payment_confirmation",
        f"Payment confirmation saved for {safe_str(itinerary.get('title') or customer.get('destination'))}",
        "generated"
    )
    safe_append_booking_to_sheet({
        "bookingRef": booking_ref,
        "packageName": safe_str(itinerary.get("title") or customer.get("packageName") or customer.get("destination")),
        "customer": customer,
        "payment": payment,
        "totalAmount": total_amount,
        "advancePaid": paid_amount,
        "remainingAmount": remaining_amount,
        "paymentStatus": payment_status,
    })
    safe_send_owner_whatsapp_alert("booking", {
        "bookingRef": booking_ref,
        "name": safe_str(customer.get("name")),
        "phone": safe_str(customer.get("phone")),
        "packageName": safe_str(itinerary.get("title") or customer.get("packageName") or customer.get("destination")),
        "startDate": safe_str(customer.get("startDate")),
        "endDate": safe_str(customer.get("endDate")),
        "totalAmount": total_amount,
        "advancePaid": paid_amount,
        "remainingAmount": remaining_amount,
        "paymentStatus": payment_status,
        "payment": payment,
    })

    if mongo_write_enabled():
        try:
            has_structured_itinerary = isinstance(itinerary, dict) and bool(itinerary.get("days"))
            if not has_structured_itinerary:
                generated_assets = generate_and_store_booking_itinerary(booking_doc)
                booking_doc.update(generated_assets)
            elif not booking_doc.get("itineraryImages"):
                image_assets = generate_itinerary_images_safe(booking_doc, itinerary)
                booking_doc["itineraryImages"] = image_assets
                safe_mongo_write(
                    "update_booking_images",
                    lambda: get_collection("bookings").update_one(
                        {"bookingId": booking_id},
                        {"$set": {"itineraryImages": normalize_for_mongo(image_assets)}}
                    )
                )
        except Exception:
            logger.exception("Post-payment itinerary generation failed for %s", booking_ref)

    return {"ok": True, "message": "Payment confirmation saved successfully", "bookingRef": booking_ref}


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


@app.post("/api/bookings/{booking_ref}/generate-itinerary")
def generate_booking_itinerary(
    booking_ref: str,
    payload: BookingItineraryGenerateRequest
):
    if not mongo_write_enabled():
        raise HTTPException(status_code=500, detail="MongoDB is not configured")

    booking = get_collection("bookings").find_one({
        "$or": [
            {"bookingRef": booking_ref},
            {"bookingId": booking_ref},
            {"razorpayOrderId": booking_ref},
        ]
    })
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    booking_phone = clean_phone(safe_str(booking.get("phone")))
    requested_phone = clean_phone(safe_str(payload.phone or booking_phone))
    if len(requested_phone) != 10 or requested_phone != booking_phone or not is_phone_verified(requested_phone):
        raise HTTPException(status_code=401, detail="Unauthorized")

    assets = generate_and_store_booking_itinerary(booking)
    return {
        "ok": True,
        "bookingRef": build_booking_ref(safe_str(booking.get("bookingRef")) or safe_str(booking.get("bookingId"))),
        "itineraryStatus": safe_str(assets.get("itineraryStatus")),
        "latestItinerary": normalize_for_mongo(assets.get("latestItinerary") or {}),
        "routeMap": normalize_for_mongo(assets.get("routeMap") or {}),
        "hotelInfo": normalize_for_mongo(assets.get("hotelInfo") or {}),
        "cabInfo": normalize_for_mongo(assets.get("cabInfo") or {}),
    }


@app.post("/api/bookings/{booking_ref}/generate-images")
def generate_booking_images(
    booking_ref: str,
    payload: BookingImagesGenerateRequest,
    authorization: Optional[str] = Header(default=None)
):
    if not mongo_write_enabled():
        raise HTTPException(status_code=500, detail="MongoDB is not configured")

    booking = get_collection("bookings").find_one({
        "$or": [
            {"bookingRef": booking_ref},
            {"bookingId": booking_ref},
            {"razorpayOrderId": booking_ref},
        ]
    })
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    is_admin = False
    if authorization:
        try:
            require_admin_token(authorization)
            is_admin = True
        except HTTPException:
            is_admin = False

    booking_phone = clean_phone(safe_str(booking.get("phone")))
    requested_phone = clean_phone(safe_str(payload.phone or booking_phone))
    if not is_admin and (len(requested_phone) != 10 or requested_phone != booking_phone or not is_phone_verified(requested_phone)):
        raise HTTPException(status_code=401, detail="Unauthorized")

    existing_images = booking.get("itineraryImages") or []
    if existing_images:
        return {"ok": True, "bookingRef": build_booking_ref(safe_str(booking.get("bookingRef")) or safe_str(booking.get("bookingId"))), "itineraryImages": normalize_for_mongo(existing_images)}

    itinerary = booking.get("latestItinerary") or booking.get("itinerary") or {}
    if not itinerary:
        generated = generate_and_store_booking_itinerary(booking)
        itinerary = generated.get("latestItinerary") or {}
        booking = get_collection("bookings").find_one({"bookingId": safe_str(booking.get("bookingId"))}) or booking

    images = generate_itinerary_images_safe(booking, itinerary)
    safe_mongo_write(
        "update_booking_images_manual",
        lambda: get_collection("bookings").update_one(
            {"bookingId": safe_str(booking.get("bookingId"))},
            {"$set": {"itineraryImages": normalize_for_mongo(images), "updatedAt": utc_now_iso()}}
        )
    )
    return {
        "ok": True,
        "bookingRef": build_booking_ref(safe_str(booking.get("bookingRef")) or safe_str(booking.get("bookingId"))),
        "itineraryImages": normalize_for_mongo(images)
    }


@app.get("/api/customer/bookings")
def customer_bookings(phone: str = Query(...)):
    digits = clean_phone(phone)
    if len(digits) != 10:
        raise HTTPException(status_code=400, detail="Invalid phone number")
    if not is_phone_verified(digits):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not mongo_write_enabled():
        return {"ok": True, "items": []}

    rows = list(
        get_collection("bookings")
        .find({"phone": digits})
        .sort([("updatedAt", -1), ("createdAt", -1)])
        .limit(100)
    )

    return {
        "ok": True,
        "items": [serialize_customer_booking_doc(row) for row in rows]
    }


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


@app.get("/api/admin/bookings/mongo")
def admin_bookings_mongo(authorization: Optional[str] = Header(default=None)):
    require_admin_token(authorization)
    if not mongo_write_enabled():
        return {"ok": True, "items": []}

    rows = list(
        get_collection("bookings")
        .find({})
        .sort([("updatedAt", -1), ("createdAt", -1)])
        .limit(100)
    )

    return {
        "ok": True,
        "items": [serialize_customer_booking_doc(row) for row in rows]
    }


@app.get("/api/admin/crm")
def admin_crm(authorization: Optional[str] = Header(default=None)):
    require_admin_token(authorization)

    blocked_common = {
        "otp",
        "otp_code",
        "otp_hash",
        "rawOtp",
        "mongo_uri",
        "mongodb_uri",
        "msg91_auth_key",
        "msg91_key",
        "razorpay_key_secret",
        "secret",
        "authkey",
    }

    customers = get_admin_collection_snapshot(
        "customers",
        blocked_fields=blocked_common,
        sort_fields=[("updatedAt", -1), ("createdAt", -1)]
    )
    ai_itineraries = get_admin_collection_snapshot(
        "ai_itineraries",
        blocked_fields=blocked_common,
        sort_fields=[("updatedAt", -1), ("createdAt", -1)]
    )
    bookings = get_admin_collection_snapshot(
        "bookings",
        blocked_fields=blocked_common,
        sort_fields=[("updatedAt", -1), ("createdAt", -1)]
    )
    payments = get_admin_collection_snapshot(
        "payments",
        blocked_fields=blocked_common,
        sort_fields=[("updatedAt", -1), ("createdAt", -1)]
    )
    whatsapp_logs = get_admin_collection_snapshot(
        "whatsapp_logs",
        blocked_fields=blocked_common,
        sort_fields=[("createdAt", -1), ("updatedAt", -1)]
    )

    return {
        "ok": True,
        "totals": {
            "customers": customers["total"],
            "ai_itineraries": ai_itineraries["total"],
            "bookings": bookings["total"],
            "payments": payments["total"],
            "whatsapp_logs": whatsapp_logs["total"],
        },
        "customers": customers["items"],
        "ai_itineraries": ai_itineraries["items"],
        "bookings": bookings["items"],
        "payments": payments["items"],
        "whatsapp_logs": whatsapp_logs["items"],
    }


@app.get("/api/admin/content")
def admin_content(authorization: Optional[str] = Header(default=None)):
    require_admin_token(authorization)
    content = read_admin_content_store()
    return {"ok": True, **content}


@app.post("/api/admin/content/{section}")
def admin_content_update(
    section: str,
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(default=None)
):
    require_admin_token(authorization)

    allowed_sections = {
        "homepage_packages": dict,
        "images": dict,
        "ai_destinations": list,
        "tour_packages": dict,
        "pilgrimage_packages": list,
    }

    if section not in allowed_sections:
        raise HTTPException(status_code=404, detail="Admin content section not found")

    data = payload.get("data")
    expected_type = allowed_sections[section]
    if not isinstance(data, expected_type):
        raise HTTPException(status_code=400, detail="Invalid admin content payload")

    current = read_admin_content_store()
    current[section] = data
    saved = write_admin_content_store(current)

    return {"ok": True, "section": section, "data": saved.get(section)}


@app.get("/api/public/homepage-packages")
def public_homepage_packages():
    return {"ok": True, "items": get_public_admin_content().get("homepage_packages", {})}


@app.get("/api/public/images")
def public_images():
    return {"ok": True, "items": get_public_admin_content().get("images", {})}


@app.get("/api/public/ai-destinations")
def public_ai_destinations():
    return {"ok": True, "items": get_public_admin_content().get("ai_destinations", [])}


@app.get("/api/public/tour-packages")
def public_tour_packages():
    return {"ok": True, "items": get_public_admin_content().get("tour_packages", {})}


@app.get("/api/public/pilgrimage-packages")
def public_pilgrimage_packages():
    return {"ok": True, "items": get_public_admin_content().get("pilgrimage_packages", [])}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
