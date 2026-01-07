# google_sheets.py
import os
import logging
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

logger = logging.getLogger("HKE_BACKEND.google_sheets")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def _get_required_env(key: str) -> str:
    val = os.getenv(key)
    if not val or not val.strip():
        raise RuntimeError(f"Missing environment variable: {key}")
    return val.strip()


def get_sheet():
    """
    Opens the Google Sheet worksheet using a service account json file.

    Requires env:
      GOOGLE_APPLICATION_CREDENTIALS=credentials/google-sheets.json
      GOOGLE_SHEET_ID=xxxx
      GOOGLE_SHEET_TAB=Sheet1
    """
    creds_path = _get_required_env("GOOGLE_APPLICATION_CREDENTIALS")
    sheet_id = _get_required_env("GOOGLE_SHEET_ID")
    sheet_tab = _get_required_env("GOOGLE_SHEET_TAB")

    # Resolve creds file path relative to this project folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    creds_abs = creds_path if os.path.isabs(creds_path) else os.path.join(base_dir, creds_path)

    logger.info("Google Sheets config:")
    logger.info(f"  GOOGLE_APPLICATION_CREDENTIALS = {creds_path}")
    logger.info(f"  Resolved creds path           = {creds_abs}")
    logger.info(f"  GOOGLE_SHEET_ID               = {sheet_id}")
    logger.info(f"  GOOGLE_SHEET_TAB              = {sheet_tab}")

    if not os.path.exists(creds_abs):
        raise RuntimeError(
            f"Google credentials file not found. "
            f"GOOGLE_APPLICATION_CREDENTIALS='{creds_path}' "
            f"Resolved='{creds_abs}'"
        )

    creds = Credentials.from_service_account_file(creds_abs, scopes=SCOPES)
    client = gspread.authorize(creds)

    try:
        ws = client.open_by_key(sheet_id).worksheet(sheet_tab)
        return ws
    except Exception as e:
        raise RuntimeError(
            f"Failed to open Google Sheet tab. "
            f"SHEET_ID={sheet_id}, SHEET_TAB={sheet_tab}, ERROR={repr(e)}"
        )


def insert_lead(data: dict):
    """
    Appends a lead row to Google Sheet in this exact column order:

    timestamp | name | phone | email | state | start_date | end_date | days | travellers | rooms
    | hotel_category | guide | vehicle | package | source | message | status
    """
    logger.info("INSERT_LEAD called")
    logger.info(f"Incoming payload keys: {list(data.keys())}")

    ws = get_sheet()

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # timestamp
        (data.get("name") or "").strip(),              # name
        (data.get("phone") or "").strip(),             # phone
        (data.get("email") or "").strip(),             # email
        (data.get("state") or "").strip(),             # state
        (data.get("start_date") or "").strip(),        # start_date
        (data.get("end_date") or "").strip(),          # end_date
        data.get("days", ""),                          # days
        data.get("travellers", ""),                    # travellers
        data.get("rooms", ""),                         # rooms
        (data.get("hotel_category") or "").strip(),    # hotel_category
        (data.get("guide") or "").strip(),             # guide
        (data.get("vehicle") or "").strip(),           # vehicle
        (data.get("package") or "").strip(),           # package
        (data.get("source") or "website").strip(),     # source
        (data.get("message") or "").strip(),           # message
        (data.get("status") or "New").strip(),         # status
    ]

    try:
        ws.append_row(row, value_input_option="USER_ENTERED")
        logger.info("✅ Lead appended to Google Sheet successfully!")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to append row to Google Sheet: {repr(e)}")
