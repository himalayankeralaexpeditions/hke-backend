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
    try:
        client.verify.v2.services(
            TWILIO_VERIFY_SERVICE_SID
        ).verifications.create(
            to=data.phone,
            channel="sms"
        )
        return {"status": "success", "message": "OTP sent"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify-otp")
def verify_otp(data: VerifyOTPRequest):
    try:
        result = client.verify.v2.services(
            TWILIO_VERIFY_SERVICE_SID
        ).verification_checks.create(
            to=data.phone,
            code=data.code
        )

        if result.status == "approved":
            return {"status": "success", "message": "OTP verified"}
        else:
            raise HTTPException(status_code=400, detail="Invalid OTP")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
