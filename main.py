from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

from openai import OpenAI

# --------------------
# APP INIT
# --------------------
app = FastAPI(title="HKE Leads & AI Trip Planner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------
# HEALTH
# --------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "HKE FastAPI backend"}

# --------------------
# EXISTING LEADS API (KEEP)
# --------------------
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
    # Your existing google_sheets logic stays here
    from google_sheets import insert_lead
    insert_lead(lead.dict())
    return {"status": "saved"}

# --------------------
# AI PLAN REQUEST
# --------------------
class AIPlanRequest(BaseModel):
    destination: str
    days: int
    startDate: str
    travellers: int
    hotelClass: str
    budget: Optional[str] = "standard"
    interests: List[str] = []

class AIPlanResponse(BaseModel):
    itinerary_text: str

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
Important places to include: {', '.join(req.interests)}

Format STRICTLY like:

Day 1 – ...
Day 2 – ...

Then add:
PACKAGE INCLUDES:
PACKAGE EXCLUDES:

Use emojis sparingly.
Make it realistic and sellable.
"""

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=1200,
        )

        text = response.output_text
        return {"itinerary_text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------
# AI CHAT (MODIFY ITINERARY)
# --------------------
class AIChatRequest(BaseModel):
    current_itinerary: str
    user_message: str

class AIChatResponse(BaseModel):
    updated_itinerary: str

@app.post("/api/ai/chat", response_model=AIChatResponse)
def chat_modify_itinerary(req: AIChatRequest):
    prompt = f"""
You are modifying an existing travel itinerary.

CURRENT ITINERARY:
{req.current_itinerary}

USER REQUEST:
{req.user_message}

Return the FULL UPDATED itinerary in same WhatsApp format.
"""

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=1200,
        )

        return {"updated_itinerary": response.output_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
