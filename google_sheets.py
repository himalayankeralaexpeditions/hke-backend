import os
import json
import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def get_sheet():
    service_account_info = json.loads(
        os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    )

    creds = Credentials.from_service_account_info(
        service_account_info,
        scopes=SCOPES
    )

    client = gspread.authorize(creds)

    sheet = client.open_by_key(
        os.environ["GOOGLE_SHEET_ID"]
    ).worksheet(
        os.environ["GOOGLE_SHEET_TAB"]
    )

    return sheet
