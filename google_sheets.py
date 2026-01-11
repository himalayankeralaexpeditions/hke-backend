import os
import json
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def _get_creds():
    """
    Reads service account JSON from env var GOOGLE_SERVICE_ACCOUNT_JSON
    (Render-friendly). Returns Google Credentials.
    """
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError("Missing environment variable: GOOGLE_SERVICE_ACCOUNT_JSON")

    try:
        info = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid GOOGLE_SERVICE_ACCOUNT_JSON (must be valid JSON): {e}")

    return Credentials.from_service_account_info(info, scopes=SCOPES)


def _get_sheet():
    """
    Opens the Google Sheet + Worksheet using env vars:
    GOOGLE_SHEET_ID, GOOGLE_SHEET_TAB
    """
    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    tab_name = os.getenv("GOOGLE_SHEET_TAB")

    if not sheet_id:
        raise RuntimeError("Missing environment variable: GOOGLE_SHEET_ID")
    if not tab_name:
        raise RuntimeError("Missing environment variable: GOOGLE_SHEET_TAB")

    client = gspread.authorize(_get_creds())
    ws = client.open_by_key(sheet_id).worksheet(tab_name)
    return ws


def insert_lead(data: dict):
    """
    Insert one lead row into Google Sheet.
    main.py imports this function: from google_sheets import insert_lead
    """

    ws = _get_sheet()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Safe get helper
    def g(key, default=""):
        v = data.get(key, default)
        return "" if v is None else v

    row = [
        now,
        g("name"),
        g("email"),
        g("phone"),
        g("mobile"),
        g("state"),
        g("start_date"),
        g("end_date"),
        g("days"),
        g("travellers"),
        g("rooms"),
        g("hotel_category"),
        g("guide"),
        g("vehicle"),
        g("package"),
        g("source", "website"),
        g("message"),
        g("status", "New"),
    ]

    ws.append_row(row, value_input_option="USER_ENTERED")
    return {"ok": True}
