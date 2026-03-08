import os
import hmac
import hashlib
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI
import razorpay


# =========================
# APP INIT
# =========================
app = FastAPI(
    title="HKE Backend – AI Trip Planner & Payments",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict to your domain
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

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not found in environment.")
if not RAZORPAY_KEY_ID:
    print("WARNING: RAZORPAY_KEY_ID not found in environment.")
if not RAZORPAY_KEY_SECRET:
    print("WARNING: RAZORPAY_KEY_SECRET not found in environment.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

razorpay_client = None
if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
    razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))


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
    context: Dict[str, Any] = {}


class CreateOrderIn(BaseModel):
    amount: int = Field(..., gt=0, description="Amount in paise")
    currency: str = "INR"
    receipt: str = Field(..., min_length=3)
    notes: Optional[Dict[str, str]] = None


class VerifyPaymentIn(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


# =========================
# HELPERS
# =========================
def safe_int(value: Any, default: int = 1) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


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


def verify_razorpay_signature(order_id: str, payment_id: str, signature: str) -> bool:
    body = f"{order_id}|{payment_id}"
    generated_signature = hmac.new(
        bytes(RAZORPAY_KEY_SECRET, "utf-8"),
        bytes(body, "utf-8"),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(generated_signature, signature)


# =========================
# ROOT + HEALTH
# =========================
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "HKE Backend Running",
        "version": "3.0.0"
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "openai_configured": bool(OPENAI_API_KEY),
        "razorpay_configured": bool(RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET)
    }


# =========================
# LEADS
# =========================
@app.post("/api/leads")
def create_lead(payload: LeadIn):
    return {
        "ok": True,
        "message": "Lead received successfully.",
        "lead": payload.dict()
    }


# =========================
# AI ITINERARY GENERATION
# =========================
@app.post("/api/ai/itinerary")
def generate_itinerary(payload: ItineraryIn):
    if not client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing in Render environment.")

    if not payload.destination:
        raise HTTPException(status_code=400, detail="Destination is required.")

    prompt = build_itinerary_prompt(payload)

    try:
        response = client.responses.create(
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
    if not client:
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
        response = client.responses.create(
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

    try:
        order_data = {
            "amount": payload.amount,
            "currency": payload.currency,
            "receipt": payload.receipt,
            "notes": payload.notes or {},
        }

        order = razorpay_client.order.create(data=order_data)

        return {
            "ok": True,
            "key": RAZORPAY_KEY_ID,
            "order": order
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Razorpay order creation failed: {str(e)}")


# =========================
# VERIFY PAYMENT
# =========================
@app.post("/api/payment/verify")
def verify_payment(payload: VerifyPaymentIn):
    if not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=500, detail="RAZORPAY_KEY_SECRET is missing in Render environment.")

    is_valid = verify_razorpay_signature(
        order_id=payload.razorpay_order_id,
        payment_id=payload.razorpay_payment_id,
        signature=payload.razorpay_signature
    )

    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid payment signature.")

    return {
        "ok": True,
        "message": "Payment verified successfully."
    }
