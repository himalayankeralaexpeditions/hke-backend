import os
import json
import re
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr, field_validator
from openai import OpenAI

# =========================================================
# APP SETUP
# =========================================================
app = FastAPI(
    title="HKE Backend - AI Trip Planner",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)


# =========================================================
# HELPERS
# =========================================================
def clean_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    return digits[:10]


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def json_dumps_pretty(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def extract_text_from_response(resp: Any) -> str:
    """
    Robust extraction for OpenAI SDK response text.
    """
    # New SDK often supports output_text directly
    output_text = getattr(resp, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    # Fallback: inspect output blocks
    output = getattr(resp, "output", None)
    if output:
        parts: List[str] = []
        try:
            for item in output:
                content = getattr(item, "content", None) or []
                for c in content:
                    ctype = getattr(c, "type", "")
                    if ctype in ("output_text", "text"):
                        txt = getattr(c, "text", None)
                        if txt:
                            parts.append(txt)
        except Exception:
            pass

        if parts:
            return "\n".join(parts).strip()

    # Last fallback
    try:
        return str(resp)
    except Exception:
        return ""


def try_parse_json(text: str) -> Optional[dict]:
    if not text:
        return None

    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # fenced json parse
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # first {...} block parse
    generic = re.search(r"(\{.*\})", text, re.DOTALL)
    if generic:
        try:
            obj = json.loads(generic.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def build_itinerary_prompt(payload: dict) -> str:
    return f"""
Create a practical India travel itinerary for a travel agency website.

Customer details:
{json_dumps_pretty(payload)}

Return ONLY valid JSON in this exact structure:
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
- Write a realistic tentative itinerary.
- Match the exact number of days requested.
- Use the selected tourist places from the customer details.
- Keep it customer-friendly and professional.
- Do not include pricing.
- Do not include markdown.
- Do not include any explanation outside JSON.
""".strip()


def build_edit_prompt(current_itinerary: str, instruction: str, customer_details: Optional[dict]) -> str:
    return f"""
You are editing a travel itinerary for a travel agency.

Customer details:
{json_dumps_pretty(customer_details or {})}

Current itinerary:
{current_itinerary}

Customer edit instruction:
{instruction}

Return ONLY valid JSON in this exact structure:
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
- Apply the user's edit instruction.
- Keep the itinerary practical and travel-agency ready.
- Keep same trip context unless the instruction changes it.
- Do not include pricing.
- Do not include markdown.
- Do not include any explanation outside JSON.
""".strip()


def fallback_itinerary(data: dict, edit_note: str = "") -> dict:
    places = data.get("places") or ["Local Sightseeing"]
    days = max(2, int(data.get("days") or 5))
    destination = safe_str(data.get("destination"))
    from_location = safe_str(data.get("fromLocation"))
    end_point = safe_str(data.get("endPoint"))
    start_date = safe_str(data.get("startDate"))
    end_date = safe_str(data.get("endDate"))

    def get_place(index: int) -> str:
        if not places:
            return "Local Sightseeing"
        return places[index % len(places)]

    items = []
    for i in range(days):
        if i == 0:
            title = f"Arrival and transfer to {destination}"
            route = f"{from_location} → {destination}"
            activities = [
                f"Arrival from {from_location} and assistance by our support team.",
                f"Transfer to {destination} / hotel check-in.",
                "Relax and enjoy a light evening outing if time permits."
            ]
        elif i == days - 1:
            title = "Departure day"
            route = f"{destination} → {end_point}"
            activities = [
                "Breakfast at hotel and checkout.",
                f"Proceed towards {end_point} for onward journey.",
                "Trip ends with beautiful memories."
            ]
        else:
            p1 = get_place(i)
            p2 = get_place(i + 1)
            p3 = get_place(i + 2)
            title = f"{p1} sightseeing"
            route = f"{destination} local / nearby"
            activities = [
                f"After breakfast, explore {p1}.",
                f"Continue sightseeing to {p2}.",
                f"Optional visit to {p3} depending on time and road conditions.",
                "Return to hotel for overnight stay."
            ]

        items.append({
            "day": i + 1,
            "date": "",
            "title": title,
            "route": route,
            "activities": activities
        })

    extra = {
        "budget": safe_str(data.get("budget"), "Standard"),
        "travelType": safe_str(data.get("travelType"), "Family"),
        "hotel": safe_str(data.get("hotelClass"), "Standard"),
        "vehicle": safe_str(data.get("vehicle"), "SUV"),
        "guide": safe_str(data.get("guide"), "Without Guide"),
        "food": (
            f'{safe_str(data.get("foodPreference"), "Flexible")} Food Required'
            if bool(data.get("needFood"))
            else "Food not included"
        ),
        "style": ", ".join(data.get("travelStyle") or []) or "Flexible",
        "notes": safe_str(data.get("notes"), "No special notes"),
    }

    if edit_note:
        extra["editNote"] = edit_note

    return {
        "title": f"{destination} {days} Days / {max(1, days - 1)} Nights Itinerary",
        "meta": {
            "destination": destination,
            "route": f"{from_location} → {destination} → {end_point}",
            "dates": f"{start_date} to {end_date}",
            "travellers": f'{data.get("travellers", 2)} Traveller(s)',
            "rooms": f'{data.get("rooms", 1)} Room(s)',
        },
        "extraInfo": extra,
        "days": items
    }


def call_openai_json(prompt: str) -> dict:
    if not client:
        raise RuntimeError("OPENAI_API_KEY is missing on server.")

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=0.4,
        max_output_tokens=2200,
    )

    text = extract_text_from_response(response)
    parsed = try_parse_json(text)

    if not parsed:
        raise ValueError("Model did not return valid JSON.")

    return parsed


# =========================================================
# REQUEST MODELS
# =========================================================
class PlannerRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    email: EmailStr
    phone: str = Field(..., min_length=10, max_length=20)
    fromLocation: str = Field(..., min_length=2, max_length=120)
    destination: str = Field(..., min_length=2, max_length=120)
    endPoint: str = Field(..., min_length=2, max_length=120)
    startDate: str
    days: int = Field(..., ge=2, le=30)
    endDate: str
    travellers: int = Field(default=2, ge=1, le=50)
    rooms: int = Field(default=1, ge=1, le=25)
    budget: str = Field(default="Standard")
    travelType: str = Field(default="Family")
    hotelClass: str = Field(default="Standard")
    vehicle: str = Field(default="SUV")
    guide: str = Field(default="Without Guide")
    needFood: bool = Field(default=False)
    foodPreference: Optional[str] = Field(default="Flexible")
    travelStyle: List[str] = Field(default_factory=list)
    places: List[str] = Field(default_factory=list)
    notes: Optional[str] = Field(default="")

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        digits = clean_phone(v)
        if len(digits) != 10:
            raise ValueError("Phone must be a valid 10-digit number.")
        return digits

    @field_validator("places")
    @classmethod
    def validate_places(cls, v: List[str]) -> List[str]:
        cleaned = [safe_str(x) for x in v if safe_str(x)]
        if not cleaned:
            raise ValueError("At least one tourist place is required.")
        return cleaned


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
    return {
        "ok": True,
        "service": "HKE Backend Running",
        "version": "3.0.0"
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "openai_configured": bool(client),
        "model": MODEL_NAME
    }


@app.post("/api/ai/itinerary")
def generate_itinerary(payload: PlannerRequest):
    data = payload.model_dump()

    # clean and normalize
    data["name"] = safe_str(data["name"])
    data["fromLocation"] = safe_str(data["fromLocation"])
    data["destination"] = safe_str(data["destination"])
    data["endPoint"] = safe_str(data["endPoint"])
    data["notes"] = safe_str(data.get("notes", ""))
    data["travelStyle"] = [safe_str(x) for x in data.get("travelStyle", []) if safe_str(x)]
    data["places"] = [safe_str(x) for x in data.get("places", []) if safe_str(x)]

    try:
        itinerary = call_openai_json(build_itinerary_prompt(data))
        return {
            "ok": True,
            "source": "openai",
            "itinerary": itinerary
        }
    except Exception as e:
        fallback = fallback_itinerary(data)
        return {
            "ok": True,
            "source": "fallback",
            "warning": f"AI generation fallback used: {str(e)}",
            "itinerary": fallback
        }


@app.post("/api/ai/chat")
def edit_itinerary(payload: ChatEditRequest):
    instruction = safe_str(payload.instruction) or safe_str(payload.message)
    current_itinerary = safe_str(payload.current_itinerary) or safe_str(payload.itinerary)
    customer_details = payload.customer_details or payload.context or {}

    if not instruction:
      raise HTTPException(status_code=400, detail="Edit instruction is required.")

    if not current_itinerary:
      raise HTTPException(status_code=400, detail="Current itinerary is required.")

    try:
        itinerary = call_openai_json(
            build_edit_prompt(
                current_itinerary=current_itinerary,
                instruction=instruction,
                customer_details=customer_details,
            )
        )
        return {
            "ok": True,
            "source": "openai",
            "itinerary": itinerary
        }
    except Exception as e:
        fallback = fallback_itinerary(customer_details or {}, edit_note=instruction)
        return {
            "ok": True,
            "source": "fallback",
            "warning": f"AI edit fallback used: {str(e)}",
            "itinerary": fallback
        }


# =========================================================
# OPTIONAL: LOCAL RUN
# =========================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
