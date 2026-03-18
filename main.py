import os
import json
import re
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr, field_validator
from openai import OpenAI

app = FastAPI(title="HKE Backend - AI Trip Planner", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


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

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


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
                f"Hotel check-in at {destination}.",
                "Rest and enjoy a light evening visit if time permits."
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
                f"Optional stop at {p3} depending on time.",
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
        "title": f"{destination} {days} Days / {max(1, days - 1)} Nights Itinerary",
        "meta": {
            "destination": destination,
            "route": f"{from_location} → {destination} → {end_point}",
            "dates": f"{start_date} to {end_date}",
            "travellers": f'{data.get("travellers", 2)} Traveller(s)',
            "rooms": f'{data.get("rooms", 1)} Room(s)'
        },
        "extraInfo": extra_info,
        "days": day_items
    }


def build_itinerary_prompt(data: dict) -> str:
    return f"""
Create a tentative professional travel itinerary for a travel company.

Customer details:
{json.dumps(data, ensure_ascii=False, indent=2)}

Return ONLY valid JSON in this structure:
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
- Match exact requested number of days
- Use selected tourist places
- No pricing
- No markdown
- No explanation outside JSON
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

Return ONLY valid JSON in this structure:
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
- Apply customer instruction properly
- No pricing
- No markdown
- No explanation outside JSON
""".strip()


def call_openai_json(prompt: str) -> dict:
    if not client:
        raise RuntimeError("OPENAI_API_KEY not configured")

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        max_output_tokens=2200
    )

    text = extract_text_from_response(resp)
    parsed = try_parse_json(text)

    if not parsed:
        raise ValueError("Invalid JSON from model")

    return parsed


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


# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def root():
    return {"ok": True, "service": "HKE Backend Running", "version": "3.1.0"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "openai_configured": bool(client),
        "model": OPENAI_MODEL
    }


@app.post("/api/ai/itinerary")
def generate_itinerary(payload: PlannerRequest):
    data = payload.model_dump()

    try:
        itinerary = call_openai_json(build_itinerary_prompt(data))
        return {
            "ok": True,
            "source": "openai",
            "itinerary": itinerary
        }
    except Exception as e:
        return {
            "ok": True,
            "source": "fallback",
            "warning": str(e),
            "itinerary": fallback_itinerary(data)
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
        return {
            "ok": True,
            "source": "openai",
            "itinerary": itinerary
        }
    except Exception as e:
        return {
            "ok": True,
            "source": "fallback",
            "warning": str(e),
            "itinerary": fallback_itinerary(customer_details, edit_note=instruction)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
