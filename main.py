from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import os

from openai import OpenAI

# =========================
# APP INIT
# =========================
app = FastAPI(title="HKE Backend – AI Trip Planner & Leads", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# OPENAI CLIENT
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# ROOT + HEALTH
# =========================
@app.get("/")
def root():
    return {"ok": True, "service": "HKE Backend Running"}

@app.get("/health")
def health():
    return {"ok": True}

# =========================
# LEADS (KEEP)
# =========================
class LeadIn(BaseModel):
    source: str = "website"
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
    try:
        from google_sheets import insert_lead
        insert_lead(lead.dict())
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lead save failed: {str(e)}")

# =========================
# AI REQUEST (ACCEPT FRONTEND FIELDS)
# =========================
class AIPlanRequest(BaseModel):
    # required for itinerary
    destination: str
    days: int
    startDate: str
    travellers: int = 2
    hotelClass: str = "Standard"
    vehicle: str = "SUV"
    guide: str = "Without Guide"

    # new fields (you wanted)
    startPoint: Optional[str] = ""
    endPoint: Optional[str] = ""
    notes: Optional[str] = ""

    # places can come as array OR comma string
    interests: Union[List[str], str, None] = Field(default_factory=list)

    # optional
    budget: Optional[str] = "standard"

class AIPlanResponse(BaseModel):
    itinerary: str

def _normalize_places(interests: Union[List[str], str, None]) -> List[str]:
    if interests is None:
        return []
    if isinstance(interests, list):
        return [str(x).strip() for x in interests if str(x).strip()]
    if isinstance(interests, str):
        # supports "Manali, Solang, Kasol"
        return [x.strip() for x in interests.split(",") if x.strip()]
    return []

def _build_prompt(req: AIPlanRequest) -> str:
    places = _normalize_places(req.interests)
    places_text = ", ".join(places) if places else "Best highlights as per destination"

    sp = (req.startPoint or "").strip()
    ep = (req.endPoint or "").strip()

    route_line = ""
    if sp and ep:
        route_line = f"Trip Route: Start from {sp} and end at {ep}."
    elif sp:
        route_line = f"Trip Start Point: {sp}."
    elif ep:
        route_line = f"Trip End Point: {ep}."

    notes = (req.notes or "").strip()
    notes_line = f"Customer Notes: {notes}" if notes else ""

    return f"""
You are a professional travel planner for Himalayan Kerala Expeditions (India).

Create a detailed DAY-WISE itinerary in WhatsApp-friendly format.

Destination: {req.destination}
Days: {req.days}
Start Date: {req.startDate}
Travellers: {req.travellers}
Hotel Category: {req.hotelClass}
Vehicle: {req.vehicle}
Guide: {req.guide}
Budget: {req.budget}
{route_line}
Must-include places: {places_text}
{notes_line}

STRICT FORMAT:
Day 1 – ...
Day 2 – ...
...
(cover all days)

After day-wise plan add:
PACKAGE INCLUDES:
PACKAGE EXCLUDES:

Rules:
- Keep it realistic and sellable
- No breakfast/lunch/dinner included by default (mention meals are on demand if needed)
- Use emojis sparingly
""".strip()

# =========================
# AI GENERATE (MAIN)
# =========================
@app.post("/api/ai/plan", response_model=AIPlanResponse)
def generate_itinerary(req: AIPlanRequest):
    prompt = _build_prompt(req)

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,              # MUST be a string
            max_output_tokens=1200,
        )
        text = (resp.output_text or "").strip()
        if not text:
            raise HTTPException(status_code=500, detail="OpenAI returned empty itinerary")
        return {"itinerary": text}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generate failed: {str(e)}")

# ✅ Alias for your frontend
@app.post("/api/ai/itinerary", response_model=AIPlanResponse)
def generate_itinerary_alias(req: AIPlanRequest):
    return generate_itinerary(req)

# =========================
# AI CHAT (EDIT ITINERARY)
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
""".strip()

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=1200,
        )
        text = (resp.output_text or "").strip()
        if not text:
            raise HTTPException(status_code=500, detail="OpenAI returned empty updated itinerary")
        return {"itinerary": text}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI chat failed: {str(e)}")

# =========================
# FINALIZE (RECEIVE ANYTHING)
# =========================
class FinalizeRequest(BaseModel):
    itinerary: str
    context: Dict[str, Any] = Field(default_factory=dict)

@app.post("/api/ai/finalize")
def finalize_itinerary(req: FinalizeRequest):
    # For now just acknowledge (frontend expects 200 OK)
    return {
        "ok": True,
        "message": "Finalized (stored in browser for next page).",
        "itinerary": req.itinerary,
        "context": req.context,
    }

# =========================
# ✅ CUSTOMER CARE CHATBOT (NEW)
# Support-only. NOT itinerary/booking sales.
# =========================
class SupportChatRequest(BaseModel):
    message: str

class SupportChatResponse(BaseModel):
    reply: str

@app.post("/api/support/chat", response_model=SupportChatResponse)
def customer_care_chat(req: SupportChatRequest):
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")

    instructions = """
You are the CUSTOMER CARE assistant for Himalayan Kerala Expeditions (HKE).
This chatbot is ONLY for customer support issues:
- payment issues (UPI/QR/transaction pending/failed)
- booking status / confirmation
- cancellation / reschedule process
- refund timelines
- pickup timing / coordination questions
- general assistance
- connect to human agent

STRICT RULES:
- DO NOT create itineraries and DO NOT sell packages here.
- If user asks for itinerary/package/plan, reply: "Please use Plan with AI page for itinerary."
- Be concise and helpful.
- Ask at most ONE follow-up question if required (name/phone/date/UTR).
- If user asks "talk to agent" / "human", immediately give contact details.

HKE Contact:
WhatsApp: +91 97972 94747
Phone: +91 97972 94747
Email: himalayankeralaexpeditions@gmail.com
""".strip()

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            instructions=instructions,
            input=msg,
            max_output_tokens=250,
        )
        reply = (resp.output_text or "").strip()
        if not reply:
            reply = "Please share your issue in 1 line. For urgent help WhatsApp: +91 97972 94747"
        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Support chat failed: {str(e)}")
