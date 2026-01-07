import os
import json
import logging
import traceback
from typing import Optional, List, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from dotenv import load_dotenv

from google_sheets import insert_lead


# =========================================================
# 🔧 LOGGING (set up first)
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("HKE_BACKEND")


# =========================================================
# 🔧 LOAD ENV (.env overrides OS vars)
# =========================================================
load_dotenv(override=True)
logger.info("Loaded environment variables from .env (override=True)")


# =========================================================
# 🔐 RAILWAY GOOGLE CREDENTIALS WRITER (CRITICAL)
# =========================================================
# Railway stores JSON as ENV string → write it to file at runtime
# Works when you set GOOGLE_SHEETS_JSON in Railway Variables
GOOGLE_SHEETS_JSON = os.getenv("GOOGLE_SHEETS_JSON", "").strip()

if GOOGLE_SHEETS_JSON:
    os.makedirs("credentials", exist_ok=True)
    creds_path = os.path.join("credentials", "google-sheets.json")

    try:
        # Some platforms store escaped JSON; this ensures it's valid JSON
        parsed = json.loads(GOOGLE_SHEETS_JSON)
        normalized_json = json.dumps(parsed, ensure_ascii=False, indent=2)

        # Always write/overwrite to avoid stale creds
        with open(creds_path, "w", encoding="utf-8") as f:
            f.write(normalized_json)

        # If your code uses GOOGLE_APPLICATION_CREDENTIALS, ensure it's set
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials/google-sheets.json"
        logger.info("Railway creds written to credentials/google-sheets.json")

    except Exception:
        logger.error("GOOGLE_SHEETS_JSON is not valid JSON. Fix Railway variable value.")
        logger.error(traceback.format_exc())


# =========================================================
# 🚀 FASTAPI APP
# =========================================================
app = FastAPI(title="HKE Leads API", version="1.0.0")


# =========================================================
# 🌍 CORS
# =========================================================
origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()

if origins_env in ("", "*"):
    allowed_origins: List[str] = ["*"]
else:
    allowed_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

logger.info(f"CORS Allowed Origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 📦 PYDANTIC MODEL — AI TRIP PLANNER
# =========================================================
class LeadIn(BaseModel):
    # Identity
    name: str = Field(..., min_length=1)
    email: Optional[EmailStr] = None

    # Phone (frontend may send phone OR mobile)
    phone: Optional[str] = None
    mobile: Optional[str] = None

    # Trip details
    state: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    days: Optional[Union[int, str]] = None
    travellers: Optional[Union[int, str]] = None
    rooms: Optional[Union[int, str]] = None
    hotel_category: Optional[str] = None
    guide: Optional[str] = None
    vehicle: Optional[str] = None

    # Meta
    package: Optional[str] = None
    source: Optional[str] = "website"
    message: Optional[str] = None
    status: Optional[str] = "New"


# =========================================================
# 🩺 HEALTH CHECK
# =========================================================
@app.get("/")
def health():
    return {"status": "ok", "service": "HKE FastAPI backend"}


# =========================================================
# 📥 CREATE LEAD
# =========================================================
@app.post("/api/leads")
def create_lead(lead: LeadIn):
    """
    Receives AI Trip Planner lead and writes to Google Sheets
    """
    try:
        data = lead.model_dump()

        # Normalize phone number
        phone = (data.get("phone") or data.get("mobile") or "").strip()

        payload = {
            "name": (data.get("name") or "").strip(),
            "phone": phone,
            "email": data.get("email") or "",
            "state": data.get("state") or "",
            "start_date": data.get("start_date") or "",
            "end_date": data.get("end_date") or "",
            "days": data.get("days") if data.get("days") is not None else "",
            "travellers": data.get("travellers") if data.get("travellers") is not None else "",
            "rooms": data.get("rooms") if data.get("rooms") is not None else "",
            "hotel_category": data.get("hotel_category") or "",
            "guide": data.get("guide") or "",
            "vehicle": data.get("vehicle") or "",
            "package": data.get("package") or "",
            "source": data.get("source") or "website",
            "message": data.get("message") or "",
            "status": data.get("status") or "New",
        }

        logger.info("POST /api/leads called")
        logger.info(f"GOOGLE_APPLICATION_CREDENTIALS = {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
        logger.info(f"GOOGLE_SHEET_ID = {os.getenv('GOOGLE_SHEET_ID')}")
        logger.info(f"GOOGLE_SHEET_TAB = {os.getenv('GOOGLE_SHEET_TAB')}")
        logger.info(f"Payload keys = {list(payload.keys())}")

        insert_lead(payload)

        return {"ok": True, "message": "Lead stored successfully"}

    except Exception as e:
        logger.error("ERROR while inserting lead into Google Sheets")
        logger.error(str(e))
        logger.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to store lead", "reason": str(e)},
        )


# =========================================================
# ▶️ LOCAL RUN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
