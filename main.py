import os
import hmac
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI
import razorpay
from pymongo import MongoClient
from pymongo.collection import Collection
from twilio.rest import Client as TwilioClient


# =========================
# APP INIT
# =========================
app = FastAPI(
    title="HKE Backend – Hybrid OTP, Orders, Payments & Bookings",
    version="7.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# MSG91
MSG91_AUTH_KEY = os.getenv("MSG91_AUTH_KEY", "").strip()
MSG91_SENDER_ID = os.getenv("MSG91_SENDER_ID", "HKEOTP").strip()
MSG91_OTP_EXPIRY = int(os.getenv("MSG91_OTP_EXPIRY", "10").strip() or "10")
MSG91_TEMPLATE_ID = os.getenv("MSG91_TEMPLATE_ID", "").strip()  # optional, if your account uses template-based OTP

# TWILIO VERIFY
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_VERIFY_SERVICE_SID = os.getenv("TWILIO_VERIFY_SERVICE_SID", "").strip()

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
razorpay_client = (
    razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET else None
)
twilio_client = (
    TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None
)

mongo_client = None
db = None
leads_collection: Optional[Collection] = None
bookings_collection: Optional[Collection] = None
payments_collection: Optional[Collection] = None
support_logs_collection: Optional[Collection] = None
otp_logs_collection: Optional[Collection] = None
mongo_error_message = ""

if MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
        mongo_client.admin.command("ping")

        db = mongo_client["hke_db"]
        leads_collection = db["leads"]
        bookings_collection = db["bookings"]
        payments_collection = db["payments"]
        support_logs_collection = db["support_logs"]
        otp_logs_collection = db["otp_logs"]

        bookings_collection.create_index("booking_ref", unique=True)
        bookings_collection.create_index("phone")
        bookings_collection.create_index("email")
        bookings_collection.create_index("razorpay_order_id")
        payments_collection.create_index("razorpay_payment_id", unique=True)
        payments_collection.create_index("razorpay_order_id")

        if otp_logs_collection is not None:
            otp_logs_collection.create_index("phone")
            otp_logs_collection.create_index("created_at")

        print("MongoDB connected successfully.")
    except Exception as e:
        mongo_error_message = str(e)
        print(f"WARNING: MongoDB connection failed: {e}")


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


class SupportChatIn(BaseModel):
    message: str


class CreateOrderIn(BaseModel):
    amount: int = Field(..., gt=0, description="Amount in paise")
    currency: str = "INR"
    receipt: str = Field(..., min_length=3)
    notes: Optional[Dict[str, str]] = None


class VerifyPaymentIn(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


class CreateBalanceOrderIn(BaseModel):
    booking_ref: str
    amount_paise: int = Field(..., gt=0)
    payment_type: str = Field(default="custom_partial")


class CancelBookingIn(BaseModel):
    booking_ref: str
    reason: str = ""


class SendOtpIn(BaseModel):
    phone: str


class VerifyOtpIn(BaseModel):
    phone: str
    otp: str


# =========================
# HELPERS
# =========================
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def normalize_phone(phone: str) -> str:
    digits = "".join(ch for ch in str(phone or "") if ch.isdigit())
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) > 10:
        digits = digits[-10:]
    return digits


def is_india_number(phone: str) -> bool:
    raw = str(phone or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())

    if raw.startswith("+91"):
        return True
    if digits.startswith("91") and len(digits) >= 12:
        return True
    if len(digits) == 10:
        return True
    return False


def to_india_msg91_number(phone: str) -> str:
    digits = normalize_phone(phone)
    if len(digits) != 10:
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid 10-digit Indian mobile number."
        )
    return "91" + digits


def to_twilio_phone(phone: str) -> str:
    raw = str(phone or "").strip()
    if raw.startswith("+"):
        return raw

    digits = "".join(ch for ch in raw if ch.isdigit())

    if digits.startswith("91") and len(digits) >= 12:
        return "+" + digits
    if len(digits) == 10:
        return "+91" + digits

    return "+" + digits


def parse_date_ymd(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
    except Exception:
        return None


def format_ymd(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m-%d") if dt else ""


def build_booking_ref() -> str:
    return "HKE-" + datetime.now().strftime("%Y%m%d%H%M%S")


def require_mongo() -> None:
    if bookings_collection is None or payments_collection is None:
        raise HTTPException(
            status_code=500,
            detail="MongoDB is configured but connection failed. Check MONGO_URI, Atlas IP access, username, password, and cluster hostname."
        )


def require_msg91() -> None:
    if not MSG91_AUTH_KEY:
        raise HTTPException(
            status_code=500,
            detail="MSG91 is not configured in environment."
        )


def require_twilio_verify() -> None:
    if not twilio_client or not TWILIO_VERIFY_SERVICE_SID:
        raise HTTPException(
            status_code=500,
            detail="Twilio Verify is not configured in environment."
        )


def verify_razorpay_signature(order_id: str, payment_id: str, signature: str) -> bool:
    body = f"{order_id}|{payment_id}"
    generated_signature = hmac.new(
        bytes(RAZORPAY_KEY_SECRET, "utf-8"),
        bytes(body, "utf-8"),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(generated_signature, signature)


def serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return {}
    x = dict(doc)
    if "_id" in x:
        x["_id"] = str(x["_id"])
    return x


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
6. Add short inclusions, exclusions, and customization note.
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
""".strip()


def build_support_prompt(user_message: str) -> str:
    return f"""
You are HKE Customer Care for Himalayan Kerala Expeditions.

User message:
{user_message}

Instructions:
1. Reply like customer support.
2. Help with payment issues, booking status, cancellation, reschedule, and general support.
3. Keep the answer short and practical.
4. If the user wants a human, tell them to contact WhatsApp: +91 97972 94747.
""".strip()


def build_schedule_fields(start_date_str: str) -> Dict[str, Any]:
    start_dt = parse_date_ymd(start_date_str)
    if not start_dt:
        return {
            "full_payment_due_date": "",
            "cancellable_until": "",
            "can_cancel": False
        }

    full_due = start_dt - timedelta(days=10)
    cancel_until = start_dt - timedelta(days=7)
    today = datetime.now().date()

    return {
        "full_payment_due_date": format_ymd(full_due),
        "cancellable_until": format_ymd(cancel_until),
        "can_cancel": today <= cancel_until.date()
    }


def compute_financials(total_amount_paise: int, paid_amount_paise: int) -> Dict[str, int]:
    remaining = max(total_amount_paise - paid_amount_paise, 0)
    return {
        "total_amount_paise": max(total_amount_paise, 0),
        "paid_amount_paise": max(paid_amount_paise, 0),
        "remaining_amount_paise": remaining
    }


def get_booking_or_404(booking_ref: str) -> Dict[str, Any]:
    require_mongo()
    booking = bookings_collection.find_one({"booking_ref": booking_ref.strip()})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found.")
    return booking


def recalc_and_update_booking(booking_ref: str) -> Dict[str, Any]:
    booking = get_booking_or_404(booking_ref)

    payment_items = list(payments_collection.find({
        "booking_ref": booking_ref,
        "status": "paid"
    }))

    paid_total = sum(int(x.get("amount_paise", 0)) for x in payment_items)
    total_amount = int(booking.get("total_amount_paise", 0))
    money = compute_financials(total_amount, paid_total)

    payment_status = "pending"
    booking_status = booking.get("booking_status", "pending_payment")

    if paid_total > 0 and money["remaining_amount_paise"] > 0:
        payment_status = "partially_paid"
        booking_status = "advance_paid"

    if money["remaining_amount_paise"] == 0 and total_amount > 0:
        payment_status = "fully_paid"
        booking_status = "confirmed"

    update_data = {
        **money,
        "payment_status": payment_status,
        "booking_status": booking_status,
        "updated_at": utc_now()
    }

    bookings_collection.update_one(
        {"_id": booking["_id"]},
        {"$set": update_data}
    )

    return serialize_doc(bookings_collection.find_one({"_id": booking["_id"]}))


# =========================
# ROOT + HEALTH
# =========================
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "HKE Backend Running",
        "version": "7.1.0"
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "openai_configured": bool(OPENAI_API_KEY),
        "razorpay_configured": bool(RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET),
        "mongo_configured": bool(MONGO_URI),
        "mongo_connected": bool(bookings_collection is not None),
        "msg91_configured": bool(MSG91_AUTH_KEY),
        "msg91_template_configured": bool(MSG91_TEMPLATE_ID),
        "twilio_verify_configured": bool(
            TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_VERIFY_SERVICE_SID
        ),
        "mongo_error": mongo_error_message if mongo_error_message else ""
    }


# =========================
# OTP AUTH (HYBRID)
# INDIA -> MSG91
# INTERNATIONAL -> TWILIO VERIFY
# =========================
@app.post("/api/auth/send-otp")
def send_otp(payload: SendOtpIn):
    require_mongo()

    if is_india_number(payload.phone):
        require_msg91()

        mobile = to_india_msg91_number(payload.phone)

        # Prefer MSG91 v5 OTP endpoint
        url = "https://control.msg91.com/api/v5/otp"

        headers = {
            "authkey": MSG91_AUTH_KEY,
            "Content-Type": "application/json"
        }

        body: Dict[str, Any] = {
            "mobile": mobile,
            "otp_expiry": MSG91_OTP_EXPIRY
        }

        # If template ID is configured, use it
        if MSG91_TEMPLATE_ID:
            body["template_id"] = MSG91_TEMPLATE_ID

        try:
            r = requests.post(url, json=body, headers=headers, timeout=20)

            try:
                data = r.json()
            except Exception:
                data = {"raw": r.text}

            print("MSG91 SEND RESPONSE:", data)
            print("MSG91 STATUS CODE:", r.status_code)
            print("MSG91 RAW TEXT:", r.text)

            if r.status_code >= 400:
                raise HTTPException(status_code=500, detail=f"MSG91 OTP failed: {data}")

            response_type = str(data.get("type", "")).lower()
            if response_type not in ["success"]:
                raise HTTPException(status_code=500, detail=f"MSG91 OTP failed: {data}")

            if otp_logs_collection is not None:
                otp_logs_collection.insert_one({
                    "provider": "MSG91",
                    "phone": normalize_phone(payload.phone),
                    "phone_raw": payload.phone,
                    "status": "sent",
                    "type": "send",
                    "response": data,
                    "created_at": utc_now()
                })

            return {
                "ok": True,
                "provider": "MSG91",
                "message": "OTP request accepted."
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MSG91 OTP send failed: {str(e)}")

    require_twilio_verify()
    phone_twilio = to_twilio_phone(payload.phone)

    try:
        verification = twilio_client.verify.v2.services(
            TWILIO_VERIFY_SERVICE_SID
        ).verifications.create(
            to=phone_twilio,
            channel="sms"
        )

        if otp_logs_collection is not None:
            otp_logs_collection.insert_one({
                "provider": "TWILIO",
                "phone": payload.phone,
                "phone_raw": payload.phone,
                "status": getattr(verification, "status", "pending"),
                "type": "send",
                "created_at": utc_now()
            })

        return {
            "ok": True,
            "provider": "TWILIO",
            "message": "OTP sent successfully.",
            "status": getattr(verification, "status", "pending")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Twilio OTP send failed: {str(e)}")


@app.post("/api/auth/verify-otp")
def verify_otp(payload: VerifyOtpIn):
    require_mongo()

    otp = str(payload.otp or "").strip()
    if not otp:
        raise HTTPException(status_code=400, detail="OTP is required.")

    if is_india_number(payload.phone):
        require_msg91()

        mobile = to_india_msg91_number(payload.phone)
        url = "https://control.msg91.com/api/v5/otp/verify"

        body = {
            "otp": otp,
            "mobile": mobile
        }

        headers = {
            "authkey": MSG91_AUTH_KEY,
            "Content-Type": "application/json"
        }

        try:
            r = requests.post(url, json=body, headers=headers, timeout=20)

            try:
                data = r.json()
            except Exception:
                data = {"raw": r.text}

            print("MSG91 VERIFY RESPONSE:", data)
            print("MSG91 VERIFY STATUS CODE:", r.status_code)
            print("MSG91 VERIFY RAW TEXT:", r.text)

            success = str(data.get("type", "")).lower() == "success"
            if not success:
                raise HTTPException(status_code=400, detail=f"Invalid or expired OTP. Provider response: {data}")

            phone_normal = normalize_phone(payload.phone)
            orders = list(bookings_collection.find({"phone": phone_normal}).sort("created_at", -1))

            if otp_logs_collection is not None:
                otp_logs_collection.insert_one({
                    "provider": "MSG91",
                    "phone": phone_normal,
                    "phone_raw": payload.phone,
                    "status": "approved",
                    "type": "verify",
                    "response": data,
                    "created_at": utc_now()
                })

            return {
                "ok": True,
                "provider": "MSG91",
                "message": "OTP verified successfully.",
                "phone": phone_normal,
                "orders_count": len(orders)
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MSG91 OTP verification failed: {str(e)}")

    require_twilio_verify()
    phone_twilio = to_twilio_phone(payload.phone)

    try:
        check = twilio_client.verify.v2.services(
            TWILIO_VERIFY_SERVICE_SID
        ).verification_checks.create(
            to=phone_twilio,
            code=otp
        )

        status = getattr(check, "status", "")
        if status != "approved":
            raise HTTPException(status_code=400, detail="Invalid or expired OTP.")

        phone_normal = normalize_phone(payload.phone)
        orders = list(bookings_collection.find({"phone": phone_normal}).sort("created_at", -1))

        if otp_logs_collection is not None:
            otp_logs_collection.insert_one({
                "provider": "TWILIO",
                "phone": phone_normal,
                "phone_raw": payload.phone,
                "status": status,
                "type": "verify",
                "created_at": utc_now()
            })

        return {
            "ok": True,
            "provider": "TWILIO",
            "message": "OTP verified successfully.",
            "phone": phone_normal,
            "orders_count": len(orders)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Twilio OTP verification failed: {str(e)}")


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
# AI ITINERARY
# =========================
@app.post("/api/ai/itinerary")
def generate_itinerary(payload: ItineraryIn):
    if not openai_client:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is missing in environment."
        )
    if not payload.destination:
        raise HTTPException(status_code=400, detail="Destination is required.")

    try:
        response = openai_client.responses.create(
            model="gpt-5-mini",
            input=build_itinerary_prompt(payload)
        )
        text = response.output_text.strip() if hasattr(response, "output_text") else ""
        if not text:
            raise HTTPException(status_code=500, detail="OpenAI returned empty itinerary.")

        return {"ok": True, "itinerary": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Itinerary generation failed: {str(e)}")


@app.post("/api/ai/chat")
def update_itinerary(payload: ChatUpdateIn):
    if not openai_client:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is missing in environment."
        )
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message is required.")
    if not payload.itinerary.strip():
        raise HTTPException(status_code=400, detail="Current itinerary is required.")

    try:
        response = openai_client.responses.create(
            model="gpt-5-mini",
            input=build_chat_update_prompt(
                user_message=payload.message,
                itinerary=payload.itinerary,
                context=payload.context
            )
        )
        text = response.output_text.strip() if hasattr(response, "output_text") else ""
        if not text:
            raise HTTPException(
                status_code=500,
                detail="OpenAI returned empty updated itinerary."
            )

        return {"ok": True, "itinerary": text}

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

    reply = "Please share your issue clearly. For urgent help, contact WhatsApp: +91 97972 94747."

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
            pass

    if support_logs_collection is not None:
        support_logs_collection.insert_one({
            "message": user_message,
            "reply": reply,
            "created_at": utc_now()
        })

    return {"ok": True, "reply": reply}


# =========================
# PAYMENT CONFIG
# =========================
@app.get("/api/payment/config")
def payment_config():
    if not RAZORPAY_KEY_ID:
        raise HTTPException(
            status_code=500,
            detail="RAZORPAY_KEY_ID is missing in environment."
        )
    return {"ok": True, "razorpayKeyId": RAZORPAY_KEY_ID}


# =========================
# CREATE ADVANCE ORDER
# =========================
@app.post("/api/payment/create-order")
def create_order(payload: CreateOrderIn):
    if not razorpay_client:
        raise HTTPException(
            status_code=500,
            detail="Razorpay is not configured in environment."
        )
    require_mongo()

    try:
        notes = payload.notes or {}
        order_data = {
            "amount": payload.amount,
            "currency": payload.currency,
            "receipt": payload.receipt,
            "notes": notes,
        }
        order = razorpay_client.order.create(data=order_data)

        booking_ref = notes.get("booking_ref") or payload.receipt or build_booking_ref()

        total_amount_paise = safe_int(notes.get("total_amount_paise"), 0)
        if total_amount_paise <= 0:
            total_amount_paise = payload.amount * 5

        money = compute_financials(total_amount_paise, 0)
        schedule = build_schedule_fields(notes.get("start_date", ""))

        booking_doc = {
            "booking_ref": booking_ref,
            "customer_name": notes.get("customer_name", ""),
            "phone": normalize_phone(notes.get("customer_phone", "")),
            "email": notes.get("customer_email", ""),
            "destination": notes.get("destination", ""),
            "start_point": notes.get("start_point", ""),
            "end_point": notes.get("end_point", ""),
            "start_date": notes.get("start_date", ""),
            "end_date": notes.get("end_date", ""),
            "days": notes.get("days", ""),
            "travellers": notes.get("travellers", ""),
            "rooms": notes.get("rooms", ""),
            "hotel_class": notes.get("hotel_class", ""),
            "vehicle": notes.get("vehicle", ""),
            "guide": notes.get("guide", ""),
            "places": [x.strip() for x in notes.get("places", "").split(",")] if notes.get("places") else [],
            "itinerary": notes.get("itinerary", ""),
            "currency": payload.currency,
            "advance_amount_paise": payload.amount,
            **money,
            **schedule,
            "payment_status": "pending",
            "booking_status": "pending_payment",
            "can_custom_pay": True,
            "razorpay_order_id": order.get("id", ""),
            "razorpay_payment_id": "",
            "last_balance_order_id": "",
            "last_balance_payment_id": "",
            "cancel_reason": "",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }

        existing = bookings_collection.find_one({"booking_ref": booking_ref})
        if existing:
            bookings_collection.update_one({"_id": existing["_id"]}, {"$set": booking_doc})
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
# VERIFY ADVANCE PAYMENT
# =========================
@app.post("/api/payment/verify")
def verify_payment(payload: VerifyPaymentIn):
    if not RAZORPAY_KEY_SECRET:
        raise HTTPException(
            status_code=500,
            detail="RAZORPAY_KEY_SECRET is missing in environment."
        )
    require_mongo()

    is_valid = verify_razorpay_signature(
        order_id=payload.razorpay_order_id,
        payment_id=payload.razorpay_payment_id,
        signature=payload.razorpay_signature
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid payment signature.")

    booking = bookings_collection.find_one({"razorpay_order_id": payload.razorpay_order_id})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found for this payment.")

    existing_payment = payments_collection.find_one({"razorpay_payment_id": payload.razorpay_payment_id})
    if not existing_payment:
        payments_collection.insert_one({
            "booking_ref": booking.get("booking_ref", ""),
            "phone": booking.get("phone", ""),
            "email": booking.get("email", ""),
            "amount_paise": int(booking.get("advance_amount_paise", 0)),
            "currency": booking.get("currency", "INR"),
            "status": "paid",
            "payment_type": "advance",
            "razorpay_order_id": payload.razorpay_order_id,
            "razorpay_payment_id": payload.razorpay_payment_id,
            "created_at": utc_now(),
        })

    bookings_collection.update_one(
        {"_id": booking["_id"]},
        {"$set": {
            "razorpay_payment_id": payload.razorpay_payment_id,
            "updated_at": utc_now()
        }}
    )

    updated = recalc_and_update_booking(booking["booking_ref"])

    return {
        "ok": True,
        "message": "Payment verified successfully.",
        "booking_ref": updated.get("booking_ref", ""),
        "booking_status": updated.get("booking_status", ""),
        "payment_status": updated.get("payment_status", ""),
        "remaining_amount_paise": updated.get("remaining_amount_paise", 0),
        "full_payment_due_date": updated.get("full_payment_due_date", ""),
        "cancellable_until": updated.get("cancellable_until", ""),
        "can_cancel": updated.get("can_cancel", False)
    }


# =========================
# CREATE BALANCE / CUSTOM ORDER
# =========================
@app.post("/api/payment/create-balance-order")
def create_balance_order(payload: CreateBalanceOrderIn):
    if not razorpay_client:
        raise HTTPException(
            status_code=500,
            detail="Razorpay is not configured in environment."
        )
    require_mongo()

    booking = recalc_and_update_booking(payload.booking_ref)

    if booking.get("booking_status") == "cancelled":
        raise HTTPException(status_code=400, detail="Cancelled booking cannot accept payment.")

    remaining = int(booking.get("remaining_amount_paise", 0))
    if remaining <= 0:
        raise HTTPException(status_code=400, detail="No remaining balance to pay.")

    if payload.amount_paise > remaining:
        raise HTTPException(status_code=400, detail="Custom amount cannot exceed remaining balance.")

    min_amount = 100000  # ₹1,000
    if payload.amount_paise < min_amount:
        raise HTTPException(status_code=400, detail="Minimum custom payment is ₹1,000.")

    if payload.payment_type not in ["balance_full", "custom_partial"]:
        raise HTTPException(status_code=400, detail="Invalid payment type.")

    pay_amount = remaining if payload.payment_type == "balance_full" else payload.amount_paise
    receipt = f"{payload.booking_ref}-BAL-{datetime.now().strftime('%H%M%S')}"

    order_data = {
        "amount": pay_amount,
        "currency": booking.get("currency", "INR"),
        "receipt": receipt,
        "notes": {
            "booking_ref": payload.booking_ref,
            "customer_name": booking.get("customer_name", ""),
            "customer_phone": booking.get("phone", ""),
            "customer_email": booking.get("email", ""),
            "destination": booking.get("destination", ""),
            "payment_type": payload.payment_type
        },
    }

    try:
        order = razorpay_client.order.create(data=order_data)

        bookings_collection.update_one(
            {"booking_ref": payload.booking_ref},
            {"$set": {
                "last_balance_order_id": order.get("id", ""),
                "updated_at": utc_now()
            }}
        )

        return {
            "ok": True,
            "key": RAZORPAY_KEY_ID,
            "order": order,
            "booking_ref": payload.booking_ref,
            "payment_type": payload.payment_type,
            "remaining_amount_paise": remaining
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Balance payment order creation failed: {str(e)}")


# =========================
# VERIFY BALANCE / CUSTOM PAYMENT
# =========================
@app.post("/api/payment/verify-balance")
def verify_balance_payment(payload: VerifyPaymentIn):
    if not RAZORPAY_KEY_SECRET:
        raise HTTPException(
            status_code=500,
            detail="RAZORPAY_KEY_SECRET is missing in environment."
        )
    require_mongo()

    is_valid = verify_razorpay_signature(
        order_id=payload.razorpay_order_id,
        payment_id=payload.razorpay_payment_id,
        signature=payload.razorpay_signature
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid payment signature.")

    booking = bookings_collection.find_one({"last_balance_order_id": payload.razorpay_order_id})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found for balance payment.")

    existing_payment = payments_collection.find_one({"razorpay_payment_id": payload.razorpay_payment_id})
    if not existing_payment:
        try:
            payment_info = razorpay_client.payment.fetch(payload.razorpay_payment_id)
            paid_amount = int(payment_info.get("amount", 0))
        except Exception:
            paid_amount = 0

        payment_type = "custom_partial"
        remaining_before = int(booking.get("remaining_amount_paise", 0))
        if paid_amount >= remaining_before and remaining_before > 0:
            payment_type = "balance_full"

        payments_collection.insert_one({
            "booking_ref": booking.get("booking_ref", ""),
            "phone": booking.get("phone", ""),
            "email": booking.get("email", ""),
            "amount_paise": paid_amount,
            "currency": booking.get("currency", "INR"),
            "status": "paid",
            "payment_type": payment_type,
            "razorpay_order_id": payload.razorpay_order_id,
            "razorpay_payment_id": payload.razorpay_payment_id,
            "created_at": utc_now(),
        })

    bookings_collection.update_one(
        {"_id": booking["_id"]},
        {"$set": {
            "last_balance_payment_id": payload.razorpay_payment_id,
            "updated_at": utc_now()
        }}
    )

    updated = recalc_and_update_booking(booking["booking_ref"])

    return {
        "ok": True,
        "message": "Balance payment verified successfully.",
        "booking_ref": updated.get("booking_ref", ""),
        "booking_status": updated.get("booking_status", ""),
        "payment_status": updated.get("payment_status", ""),
        "paid_amount_paise": updated.get("paid_amount_paise", 0),
        "remaining_amount_paise": updated.get("remaining_amount_paise", 0)
    }


# =========================
# ORDER LOOKUP
# =========================
@app.get("/api/orders")
def get_orders(
    phone: str = "",
    email: str = "",
    booking_ref: str = ""
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

    items = list(bookings_collection.find(query).sort("created_at", -1))

    return {
        "ok": True,
        "count": len(items),
        "orders": [serialize_doc(x) for x in items]
    }


@app.get("/api/orders/{booking_ref}")
def get_order_details(booking_ref: str):
    updated = recalc_and_update_booking(booking_ref)
    payments = list(payments_collection.find({"booking_ref": booking_ref}).sort("created_at", 1))

    return {
        "ok": True,
        "order": updated,
        "payments": [serialize_doc(x) for x in payments]
    }


# =========================
# CANCEL BOOKING
# =========================
@app.post("/api/orders/cancel")
def cancel_booking(payload: CancelBookingIn):
    require_mongo()

    booking = recalc_and_update_booking(payload.booking_ref)

    if booking.get("booking_status") == "cancelled":
        raise HTTPException(status_code=400, detail="Booking is already cancelled.")

    if not booking.get("can_cancel", False):
        raise HTTPException(
            status_code=400,
            detail="Online cancellation is allowed only up to 7 days before trip start."
        )

    bookings_collection.update_one(
        {"booking_ref": payload.booking_ref},
        {"$set": {
            "booking_status": "cancelled",
            "cancel_reason": payload.reason.strip(),
            "updated_at": utc_now()
        }}
    )

    updated = bookings_collection.find_one({"booking_ref": payload.booking_ref})

    return {
        "ok": True,
        "message": "Booking cancelled successfully.",
        "order": serialize_doc(updated)
    }
