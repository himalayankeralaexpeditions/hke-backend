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
    version="1.1.0"
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
# ROOT + HEALTH (Render pings HEAD /)
# =========================
@app.get("/")
def root():
    return {"ok": True, "service": "HKE Backend Running"}

@app.head("/")
def root_head():
    return

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
    # your existing google sheet logic stays here
    from google_sheets import insert_lead
    insert_lead(lead.dict())
    return {"status": "saved"}

# =========================
# AI ITINERARY REQUEST
# (Matches your frontend payload)
# =========================
class AIPlanRequest(BaseModel):
    # customer
    name: Optional[str] = ""
    email: Optional[str] = ""
    phone: Optional[str] = ""

    # trip
    destination: str
    tourStartPoint: Optional[str] = ""
    tourEndPoint: Optional[str] = ""

    startDate: str
    endDate: Optional[str] = ""
    days: int

    travellers: int = 2
    rooms: Optional[int] = 1

    hotelClass: str = "Standard"
    guide: Optional[str] = "Without Guide"
    vehicle: Optional[str] = "SUV"

    # places
    places: Optional[str] = ""                 # frontend sends comma-separated string
    interests: Optional[List[str]] = []        # keep for future list use

    # extra
    budget: Optional[str] = "standard"
    customerNotes: Optional[str] = ""

class AIPlanResponse(BaseModel):
    itinerary: str

# =========================
# AI GENERATE (MAIN LOGIC)
# =========================
@app.post("/api/ai/plan", response_model=AIPlanResponse)
def generate_itinerary(req: AIPlanRequest):
    places_text = (req.places or "").strip()
    if not places_text and req.interests:
        places_text = ", ".join([x for x in req.interests if x])

    prompt = f"""
You are a professional travel planner for Himalayan Kerala Expeditions (India).

Create a detailed DAY-WISE itinerary in WhatsApp-friendly format.
Keep it realistic with practical drive times, check-in/out, meals and rest.

Customer: {req.name} | Phone: {req.phone} | Email: {req.email}

Destination: {req.destination}
Tour Start Point: {req.tourStartPoint}
Tour End Point: {req.tourEndPoint}
Days: {req.days}
Start Date: {req.startDate}
End Date: {req.endDate}
Travellers: {req.travellers}
Rooms: {req.rooms}
Hotel Class: {req.hotelClass}
Guide: {req.guide}
Vehicle: {req.vehicle}
Budget: {req.budget}
Important places to include: {places_text}

Customer notes (must follow if relevant):
{req.customerNotes}

STRICT FORMAT:
Day 1 – ...
• Morning: ...
• Afternoon: ...
• Evening: ...
• Stay: (area)

Day 2 – ...
...

Then add:
PACKAGE INCLUDES:
- ...
PACKAGE EXCLUDES:
- ...

Use emojis sparingly (max 1–2 per day).
Do NOT add price unless customer asked.
"""

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=1400,
        )
        text = response.output_text.strip()
        if not text or len(text) < 20:
            raise HTTPException(status_code=500, detail="OpenAI returned empty itinerary.")
        return {"itinerary": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================================================
# 🔁 FRONTEND COMPATIBILITY (YOUR WEBSITE EXPECTS THIS)
# ==================================================
@app.post("/api/ai/itinerary", response_model=AIPlanResponse)
def generate_itinerary_alias(req: AIPlanRequest):
    return generate_itinerary(req)

# =========================
# AI CHAT – MODIFY ITINERARY
# (Supports BOTH payload styles)
# =========================
class AIChatRequest(BaseModel):
    # old style
    current_itinerary: Optional[str] = ""
    user_message: Optional[str] = ""

    # new style (some versions of your ai-planner.js)
    itinerary: Optional[str] = ""
    message: Optional[str] = ""
    context: Optional[Dict[str, Any]] = None

class AIChatResponse(BaseModel):
    itinerary: str

@app.post("/api/ai/chat", response_model=AIChatResponse)
def chat_modify_itinerary(req: AIChatRequest):
    current = (req.current_itinerary or req.itinerary or "").strip()
    user_msg = (req.user_message or req.message or "").strip()

    if not current:
        raise HTTPException(status_code=400, detail="Missing current itinerary.")
    if not user_msg:
        raise HTTPException(status_code=400, detail="Missing user message.")

    ctx = req.context or {}
    ctx_text = "\n".join([f"{k}: {v}" for k, v in ctx.items()]) if ctx else ""

    prompt = f"""
You are modifying an existing travel itinerary for Himalayan Kerala Expeditions.

CONTEXT (if any):
{ctx_text}

CURRENT ITINERARY:
{current}

USER REQUEST:
{user_msg}

Return the FULL UPDATED itinerary in the SAME WhatsApp-friendly format.
Do NOT add pricing unless asked.
"""

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=1400,
        )
        text = response.output_text.strip()
        if not text or len(text) < 20:
            raise HTTPException(status_code=500, detail="OpenAI returned empty update.")
        return {"itinerary": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# FINALIZE (STUB FOR NEXT STEP)
# =========================
class FinalizeRequest(BaseModel):
    itinerary: str
    context: Dict[str, Any]

@app.post("/api/ai/finalize")
def finalize_itinerary(req: FinalizeRequest):
    """
    Next step:
    - Save to Google Sheet
    - Send Email
    - Generate WhatsApp message content
    """
    return {
        "ok": True,
        "message": "Itinerary finalized (stub). Next: Google Sheet + Email + WhatsApp automation.",
        "whatsapp_customer": "✅ Your itinerary is finalized. Thank you for choosing Himalayan Kerala Expeditions!",
        "email_subject": "Your Finalized HKE Itinerary",
        "email_body": req.itinerary
    }
