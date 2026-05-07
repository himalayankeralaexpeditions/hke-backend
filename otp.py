from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from twilio.rest import Client

router = APIRouter()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_VERIFY_SERVICE_SID = os.getenv("TWILIO_VERIFY_SERVICE_SID")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

class SendOTPRequest(BaseModel):
    phone: str

class VerifyOTPRequest(BaseModel):
    phone: str
    code: str

@router.post("/send-otp")
def send_otp(data: SendOTPRequest):
    raise HTTPException(status_code=410, detail="OTP route disabled. Use /api/auth/send-otp")


@router.post("/verify-otp")
def verify_otp(data: VerifyOTPRequest):
    raise HTTPException(status_code=410, detail="OTP route disabled. Use /api/auth/verify-otp")
