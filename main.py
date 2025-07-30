import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ===== LOAD ENV =====
load_dotenv()

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== FIREBASE INIT =====
FIREBASE_CRED_JSON = os.getenv("FIREBASE_CRED_JSON")
if not FIREBASE_CRED_JSON:
    raise RuntimeError("Missing FIREBASE_CRED_JSON environment variable.")

import tempfile

with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_cred_file:
    temp_cred_file.write(FIREBASE_CRED_JSON)
    temp_cred_file.flush()
    cred_path = temp_cred_file.name

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ===== FASTAPI APP =====
app = FastAPI(title="MMTA Backend Simplified Firebase")

# ===== CORS =====
origins = [
    "https://calcue.wuaze.com",
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MODELS =====
class SMSPayload(BaseModel):
    user_id: str
    messages: List[str]

# ===== UTILS =====
def detect_service(msg: str) -> str:
    lower = msg.lower()
    if "mpesa" in lower:
        return "MPESA"
    if "airtel" in lower:
        return "AirtelMoney"
    if "tigo" in lower:
        return "TIGO"
    return "Unknown"

def try_float(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        # Remove commas and anything except digits and dot
        cleaned = re.sub(r"[^\d.]", "", value)
        return float(cleaned)
    except:
        return None

def parse_message(msg: str) -> Dict[str, Any]:
    # Extract transaction id (TID:...)
    tid_match = re.search(r"TID[:\s]*([A-Z0-9.\-]+)", msg, re.I)
    transaction_id = tid_match.group(1) if tid_match else None

    # Determine transaction type: paid/received
    transaction_type = "paid" if "paid" in msg.lower() else "received"

    # Extract amount (e.g. 1,000.00 Tsh)
    amount_match = re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(Tsh|TZS)", msg, re.I)
    amount = try_float(amount_match.group(1)) if amount_match else None

    # Currency hardcoded as TZS
    currency = "TZS" if amount else None

    service = detect_service(msg)

    return {
        "transaction_id": transaction_id,
        "transaction_type": transaction_type,
        "amount": amount,
        "currency": currency,
        "service": service,
        "raw": msg,
        "timestamp": datetime.utcnow()
    }

# ===== FIREBASE SAVE =====
def save_transaction(user_id: str, transaction: Dict[str, Any]) -> dict:
    try:
        doc_ref = db.collection("users").document(user_id).collection("transactions").add(transaction)
        logging.info(f"Saved transaction for user {user_id} to Firebase with ref {doc_ref}")
        return {"status": "success", "firebase_ref": str(doc_ref)}
    except Exception as e:
        logging.error(f"Error saving to Firebase: {e}")
        return {"status": "error", "message": str(e)}

# ===== API ENDPOINT =====
@app.post("/analyze")
async def analyze_sms(payload: SMSPayload):
    if not payload.user_id.strip():
        raise HTTPException(status_code=400, detail="Missing or invalid 'user_id'.")

    results = []
    for msg in payload.messages:
        parsed = parse_message(msg)
        save_result = save_transaction(payload.user_id, parsed)
        results.append({
            "parsed": parsed,
            "save_result": save_result
        })

    return {
        "total_messages": len(payload.messages),
        "results": results
    }

# ===== RUN SERVER =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
