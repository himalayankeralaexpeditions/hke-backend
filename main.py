import os
import hmac
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI
import razorpay
from pymongo import MongoClient
from pymongo.collection import Collection


# =========================
# APP INIT
# =========================
app = FastAPI(
    title="HKE Backend – AI Trip Planner, Payments & Bookings",
    version="4.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# ENV
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "").strip()
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "").strip()
MONGO_URI = os.getenv("MONGO_URI", "").strip()

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not found in environment.")
if not RAZORPAY_KEY_ID:
    print("WARNING: RAZORPAY_KEY_ID not found in environment.")
if not RAZORPAY_KEY_SECRET:
    print("WARNING: RAZORPAY_KEY_SECRET not found in environment.")
if not MONGO_URI:
    print("WARNING: MONGO_URI not found in environment.")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

razorpay_client = None
if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
    razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

mongo_client = None
db = None
leads_collection: Optional[Collection] = None
bookings_collection: Optional[Collection] = None
payments_collection: Optional[Collection] = None
support_logs_collection: Optional[Collection] = None
mongo_error_message = ""

if MONGO_URI:
    try:
        mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=10000
        )

        # force real connection
        mongo_client.admin.command("ping")

        db = mongo_client["hke_db"]
        leads_collection = db["leads"]
        bookings_collection = db["bookings"]
        payments_collection = db["payments"]
        support_logs_collection = db["support_logs"]

        bookings_collection.create_index("booking_ref", unique=True)
        bookings_collection.create_index("phone")
        bookings_collection.create_index("email")
        bookings_collection.create_index("razorpay_order_id")
        payments_collection.create_index("razorpay_payment_id", unique=True)
        payments_collection.create_index("razorpay_order_id")

        print("MongoDB connected successfully.")

    except Exception as e:
        mongo_error_message = str(e)
        print(f"WARNING: MongoDB connection failed: {e}")
        mongo_client = None
        db = None
        leads_collection = None
        bookings_collection = None
        payments_collection = None
        support_logs_collection = None


# =========================
# MODELS
# =========================
class LeadIn(BaseModel):
    source: str = "website"
    name: str = ""
    email: str = ""
    phone: str = ""
    message: str = ""


class ItineraryIn(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    destination: str = ""
    startDate: str = ""
    endDate: str = ""
    days: str = "1"
    travellers: str = "2"
    rooms: str = "1"
    hotelClass: str = "Standard"
    guide: str = "Without Guide"
    vehicle: str = "SUV"
    places: str = ""


class ChatUpdateIn(BaseModel):
    message: str
    itinerary: str
    context: Dict[str, Any] = Field(default_factory=dict)


class CreateOrderIn(BaseModel):
    amount: int = Field(..., gt=0, description="Amount in paise")
    currency: str = "INR"
    receipt: str = Field(..., min_length=3)
    notes: Optional[Dict[str, str]] = None


class VerifyPaymentIn(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


class SupportChatIn(BaseModel):
    message: str


# =========================
# HELPERS
# =========================
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_phone(phone: str) -> str:
    digits = "".join(ch for ch in str(phone or "") if ch.isdigit())
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) > 10:
        digits = digits[-10:]
    return digits


def build_booking_ref() -> str:
    return "HKE-" + datetime.now().strftime("%Y%m%d%H%M%S")


def require_mongo() -> None:
    if bookings_collection is None or payments_collection is None:
        raise HTTPException(
            status_code=500,
            detail="MongoDB is configured but connection failed. Check MONGO_URI, Atlas IP access, username, password, and cluster hostname."
        )


def build_itinerary_prompt(data: ItineraryIn) -> str:
    return f"""
Create a practical travel itinerary for Himalayan Kerala Expeditions.

Customer Details:
- Name: {data.name}
- Email: {data.email}
- Phone: {data.phone}

Trip Details:
- Destination: {data.destination}
- Start Date: {data.startDate}
- End Date: {data.endDate}
- Number of Days: {data.days}
- Travellers: {data.travellers}
- Rooms: {data.rooms}
- Hotel Category: {data.hotelClass}
- Guide Requirement: {data.guide}
- Vehicle Type: {data.vehicle}
- Tourist Places: {data.places}

Instructions:
1. Write a professional, customer-facing itinerary.
2. Give a trip title at the top.
3. Add a short overview.
4. Create day-wise plan clearly from Day 1 onward.
5. Keep it practical and realistic.
6. Mention travel flow naturally.
7. Keep wording simple and premium.
8. Do not add fake hotel names unless explicitly given.
9. Add a short "Inclusions" section.
10. Add a short "Exclusions" section.
11. Add a short note for customization.
""".strip()


def build_chat_update_prompt(user_message: str, itinerary: str, context: Dict[str, Any]) -> str:
    return f"""
You are updating a travel itinerary for Himalayan Kerala Expeditions.

Current itinerary:
{itinerary}

Customer context:
{context}

Customer requested this change:
{user_message}

Instructions:
1. Return only the updated full itinerary.
2. Keep it professional and customer-friendly.
3. Preserve useful details from the current itinerary.
4. Apply the requested change clearly.
5. Keep the format easy to read.
""".strip()


def build_support_prompt(user_message: str) -> str:
    return f"""
You are HKE Customer Care for Himalayan Kerala Expeditions.

User message:
{user_message}

Instructions:
1. Reply like customer support, not salesy.
2. Help only with payment issues, booking status, cancellation, reschedule, and general support.
3. Keep the answer short, practical, and easy to understand.
4. If the user wants a human, tell them to contact WhatsApp: +91 97972 94747.
5. Do not invent booking details.
""".strip()


def verify_razorpay_signature(order_id: str, payment_id: str, signature: str) -> bool:
    body = f"{order_id}|{payment_id}"
    generated_signature = hmac.new(
        bytes(RAZORPAY_KEY_SECRET, "utf-8"),
        bytes(body, "utf-8"),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(generated_signature, signature)


def serialize_booking(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return {}
    data = dict(doc)
    if "_id" in data:
        data["_id"] = str(data["_id"])
    return data


def find_booking_by_order_id(order_id: str) -> Optional[Dict[str, Any]]:
    if bookings_collection is None:
        return None
    return bookings_collection.find_one({"razorpay_order_id": order_id})


def make_booking_from_payment(order_id: str, payment_id: str) -> Dict[str, Any]:
    existing = find_booking_by_order_id(order_id)

    if existing:
        update_data = {
            "payment_status": "paid",
            "booking_status": "received",
            "razorpay_payment_id": payment_id,
            "updated_at": utc_now(),
        }
        bookings_collection.update_one(
            {"_id": existing["_id"]},
            {"$set": update_data}
        )
        updated = bookings_collection.find_one({"_id": existing["_id"]})
        return serialize_booking(updated)

    booking_ref = build_booking_ref()
    booking_doc = {
        "booking_ref": booking_ref,
        "customer_name": "",
        "phone": "",
        "email": "",
        "destination": "",
        "start_date": "",
        "end_date": "",
        "days": "",
        "travellers": "",
        "rooms": "",
        "hotel_class": "",
        "vehicle": "",
        "guide": "",
        "places": [],
        "notes": {},
        "itinerary": "",
        "payment_status": "paid",
        "booking_status": "received",
        "advance_amount_paise": 0,
        "currency": "INR",
        "razorpay_order_id": order_id,
        "razorpay_payment_id": payment_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    result = bookings_collection.insert_one(booking_doc)
    booking_doc["_id"] = str(result.inserted_id)
    return booking_doc


# =========================
# ROOT + HEALTH
# =========================
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "HKE Backend Running",
        "version": "4.1.0"
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "openai_configured": bool(OPENAI_API_KEY),
        "razorpay_configured": bool(RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET),
        "mongo_configured": bool(MONGO_URI),
        "mongo_connected": bool(bookings_collection is not None),
        "mongo_error": mongo_error_message if mongo_error_message else ""
    }


# =========================
# LEADS
# =========================
@app.post("/api/leads")
def create_lead(payload: LeadIn):
    doc = {
        "source": payload.source,
        "name": payload.name,
        "email": payload.email,
        "phone": normalize_phone(payload.phone),
        "message": payload.message,
        "created_at": utc_now(),
    }

    inserted_id = None
    if leads_collection is not None:
        result = leads_collection.insert_one(doc)
        inserted_id = str(result.inserted_id)

    return {
        "ok": True,
        "message": "Lead received successfully.",
        "lead_id": inserted_id,
        "lead": doc
    }


# =========================
# AI ITINERARY GENERATION
# =========================
@app.post("/api/ai/itinerary")
def generate_itinerary(payload: ItineraryIn):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing in Render environment.")

    if not payload.destination:
        raise HTTPException(status_code=400, detail="Destination is required.")

    prompt = build_itinerary_prompt(payload)

    try:
        response = openai_client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )

        text = response.output_text.strip() if hasattr(response, "output_text") else ""
        if not text:
            raise HTTPException(status_code=500, detail="OpenAI returned empty itinerary.")

        return {
            "ok": True,
            "itinerary": text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Itinerary generation failed: {str(e)}")


# =========================
# AI ITINERARY CHAT UPDATE
# =========================
@app.post("/api/ai/chat")
def update_itinerary(payload: ChatUpdateIn):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing in Render environment.")

    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message is required.")
    if not payload.itinerary.strip():
        raise HTTPException(status_code=400, detail="Current itinerary is required.")

    prompt = build_chat_update_prompt(
        user_message=payload.message,
        itinerary=payload.itinerary,
        context=payload.context
    )

    try:
        response = openai_client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )

        text = response.output_text.strip() if hasattr(response, "output_text") else ""
        if not text:
            raise HTTPException(status_code=500, detail="OpenAI returned empty updated itinerary.")

        return {
            "ok": True,
            "itinerary": text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Itinerary update failed: {str(e)}")


# =========================
# SUPPORT CHAT
# =========================
@app.post("/api/support/chat")
def support_chat(payload: SupportChatIn):
    user_message = payload.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required.")

    fallback = (
        "Please share your issue clearly. "
        "For urgent help, contact WhatsApp: +91 97972 94747."
    )

    reply = fallback

    if openai_client:
        try:
            response = openai_client.responses.create(
                model="gpt-5-mini",
                input=build_support_prompt(user_message)
            )
            text = response.output_text.strip() if hasattr(response, "output_text") else ""
            if text:
                reply = text
        except Exception:
            reply = fallback

    if support_logs_collection is not None:
        support_logs_collection.insert_one({
            "message": user_message,
            "reply": reply,
            "created_at": utc_now(),
        })

    return {
        "ok": True,
        "reply": reply
    }


# =========================
# PAYMENT CONFIG
# =========================
@app.get("/api/payment/config")
def payment_config():
    if not RAZORPAY_KEY_ID:
        raise HTTPException(status_code=500, detail="RAZORPAY_KEY_ID is missing in Render environment.")

    return {
        "ok": True,
        "razorpayKeyId": RAZORPAY_KEY_ID
    }


# =========================
# CREATE RAZORPAY ORDER
# =========================
@app.post("/api/payment/create-order")
def create_order(payload: CreateOrderIn):
    if not razorpay_client:
        raise HTTPException(status_code=500, detail="Razorpay is not configured in Render environment.")

    require_mongo()

    try:
        order_data = {
            "amount": payload.amount,
            "currency": payload.currency,
            "receipt": payload.receipt,
            "notes": payload.notes or {},
        }

        order = razorpay_client.order.create(data=order_data)

        notes = payload.notes or {}
        booking_ref = notes.get("booking_ref") or payload.receipt or build_booking_ref()

        booking_doc = {
            "booking_ref": booking_ref,
            "customer_name": notes.get("customer_name", ""),
            "phone": normalize_phone(notes.get("customer_phone", "")),
            "email": notes.get("customer_email", ""),
            "destination": notes.get("destination", ""),
            "start_date": notes.get("start_date", ""),
            "end_date": notes.get("end_date", ""),
            "days": notes.get("days", ""),
            "travellers": notes.get("travellers", ""),
            "rooms": notes.get("rooms", ""),
            "hotel_class": notes.get("hotel_class", ""),
            "vehicle": notes.get("vehicle", ""),
            "guide": notes.get("guide", ""),
            "places": [x.strip() for x in notes.get("places", "").split(",")] if notes.get("places") else [],
            "notes": notes,
            "itinerary": notes.get("itinerary", ""),
            "payment_status": "created",
            "booking_status": "pending_payment",
            "advance_amount_paise": payload.amount,
            "currency": payload.currency,
            "razorpay_order_id": order.get("id", ""),
            "razorpay_payment_id": "",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }

        existing = bookings_collection.find_one({"booking_ref": booking_ref})
        if existing:
            bookings_collection.update_one(
                {"_id": existing["_id"]},
                {"$set": booking_doc}
            )
        else:
            bookings_collection.insert_one(booking_doc)

        return {
            "ok": True,
            "key": RAZORPAY_KEY_ID,
            "order": order,
            "booking_ref": booking_ref
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Razorpay order creation failed: {str(e)}")


# =========================
# VERIFY PAYMENT + SAVE BOOKING
# =========================
@app.post("/api/payment/verify")
def verify_payment(payload: VerifyPaymentIn):
    if not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=500, detail="RAZORPAY_KEY_SECRET is missing in Render environment.")

    require_mongo()

    is_valid = verify_razorpay_signature(
        order_id=payload.razorpay_order_id,
        payment_id=payload.razorpay_payment_id,
        signature=payload.razorpay_signature
    )

    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid payment signature.")

    booking = make_booking_from_payment(
        order_id=payload.razorpay_order_id,
        payment_id=payload.razorpay_payment_id
    )

    payment_doc = {
        "booking_ref": booking.get("booking_ref", ""),
        "phone": booking.get("phone", ""),
        "email": booking.get("email", ""),
        "amount_paise": booking.get("advance_amount_paise", 0),
        "currency": booking.get("currency", "INR"),
        "status": "paid",
        "razorpay_order_id": payload.razorpay_order_id,
        "razorpay_payment_id": payload.razorpay_payment_id,
        "created_at": utc_now(),
    }

    existing_payment = payments_collection.find_one(
        {"razorpay_payment_id": payload.razorpay_payment_id}
    )
    if not existing_payment:
        payments_collection.insert_one(payment_doc)

    return {
        "ok": True,
        "message": "Payment verified successfully.",
        "booking_ref": booking.get("booking_ref", ""),
        "booking_status": booking.get("booking_status", "received"),
        "payment_status": booking.get("payment_status", "paid")
    }


# =========================
# BOOKING STATUS
# =========================
@app.get("/api/booking-status")
def booking_status(
    phone: str = "",
    booking_ref: str = "",
    email: str = ""
):
    require_mongo()

    query: Dict[str, Any] = {}

    if booking_ref.strip():
        query["booking_ref"] = booking_ref.strip()
    elif phone.strip():
        query["phone"] = normalize_phone(phone)
    elif email.strip():
        query["email"] = email.strip()
    else:
        raise HTTPException(status_code=400, detail="Provide booking_ref or phone or email.")

    items = list(
        bookings_collection.find(query).sort("created_at", -1)
    )

    return {
        "ok": True,
        "count": len(items),
        "bookings": [serialize_booking(x) for x in items]
    }
