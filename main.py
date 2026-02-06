from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os

from openai import OpenAI

# =========================
# APP INIT
# =========================
app = FastAPI(
    title="HKE Backend – AI Trip Planner & Leads",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# OPENAI CLIENT
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# ROOT + HEALTH (IMPORTANT)
# =========================
@app.get("/")
def root():
    return {"ok": True, "service": "HKE Backend Running"}

@app.get("/health")
def health():
    return {"ok": True}

# =========================
# GOOGLE SHEETS LEADS (KEEP)
# =========================
class LeadIn(BaseModel):
    source: str
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
    from google_sheets import insert_lead
    insert_lead(lead.dict())
    return {"status": "saved"}

# =========================
# AI ITINERARY REQUEST
# =========================
class AIPlanRequest(BaseModel):
    destination: str
    days: int
    startDate: str
    travellers: int
    hotelClass: str
    budget: Optional[str] = "standard"
    interests: List[str] = []

class AIPlanResponse(BaseModel):
    itinerary: str

# =========================
# AI GENERATE (MAIN LOGIC)
# =========================
@app.post("/api/ai/plan", response_model=AIPlanResponse)
def generate_itinerary(req: AIPlanRequest):
    prompt = f"""
You are a professional travel planner for Himalayan Kerala Expeditions (India).

Create a detailed DAY-WISE itinerary in WhatsApp-friendly format.

Destination: {req.destination}
Days: {req.days}
Start Date: {req.startDate}
Travellers: {req.travellers}
Hotel Class: {req.hotelClass}
Budget: {req.budget}
Important places to include: {", ".join(req.interests)}

STRICT FORMAT:
Day 1 – ...
Day 2 – ...

Add sections:
PACKAGE INCLUDES:
PACKAGE EXCLUDES:

Keep it realistic, sellable, and clear.
Use emojis sparingly.
"""

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=1200,
        )

        return {"itinerary": response.output_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================================================
# 🔁 FRONTEND COMPATIBILITY (THIS FIXES YOUR ISSUE)
# ==================================================
@app.post("/api/ai/itinerary", response_model=AIPlanResponse)
def generate_itinerary_alias(req: AIPlanRequest):
    """
    Alias endpoint for frontend ai-planner.js
    """
    return generate_itinerary(req)

# =========================
# AI CHAT – MODIFY ITINERARY
# =========================
class AIChatRequest(BaseModel):
    current_itinerary: str
    user_message: str

class AIChatResponse(BaseModel):
    itinerary: str

@app.post("/api/ai/chat", response_model=AIChatResponse)
def chat_modify_itinerary(req: AIChatRequest):
    prompt = f"""
You are modifying an existing travel itinerary.

CURRENT ITINERARY:
{req.current_itinerary}

USER REQUEST:
{req.user_message}

Return the FULL UPDATED itinerary
in the SAME WhatsApp-friendly format.
"""

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=1200,
        )

        return {"itinerary": response.output_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# FINALIZE (FUTURE USE)
# =========================
class FinalizeRequest(BaseModel):
    itinerary: str
    context: Dict[str, Any]

@app.post("/api/ai/finalize")
def finalize_itinerary(req: FinalizeRequest):
    """
    Future:
    - Save to Google Sheet
    - Send Email
    - Generate WhatsApp message
    """
    return {
        "ok": True,
        "message": "Itinerary finalized",
        "whatsapp_customer": "✅ Your itinerary is finalized. Thank you for choosing Himalayan Kerala Expeditions!",
        "email_subject": "Your Finalized HKE Itinerary",
        "email_body": req.itinerary
    }
