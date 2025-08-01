import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore

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

# ===== LOAD ANALYZE.JSON =====
try:
    with open("analyze.json", "r", encoding="utf-8") as f:
        CLASSIFY_PATTERNS = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load analyze.json: {e}")

# ===== FASTAPI APP =====
app = FastAPI(title="MMTA Backend with Transaction Summary")

# ===== CORS CONFIG =====
origins = [
    "https://calcue.wuaze.com",
    "https://mmta.wuaze.com",
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

class SummaryPayload(BaseModel):
    user_id: str

class AdminPayload(BaseModel):
    admin_id: str
    messages: List[str]

# ===== UTILITIES =====
def validate_user(user_id: str) -> bool:
    try:
        doc = db.collection("users").document(user_id).get()
        return doc.exists
    except Exception as e:
        logging.error(f"Error checking user ID: {e}")
        return False

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
        cleaned = re.sub(r"[^\d.]", "", value)
        return float(cleaned)
    except:
        return None

def parse_message(msg: str) -> Dict[str, Any]:
    tid_match = re.search(r"TID[:\s]*([A-Z0-9.\-]+)", msg, re.I)
    transaction_id = tid_match.group(1) if tid_match else None

    transaction_type = "paid" if "paid" in msg.lower() else "received"
    amount_match = re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(Tsh|TZS)", msg, re.I)
    amount = try_float(amount_match.group(1)) if amount_match else None
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

def classify_transaction(msg: str) -> str:
    msg = msg.lower()
    for word in CLASSIFY_PATTERNS.get("incoming_keywords", []):
        if word in msg:
            return "incoming"
    for word in CLASSIFY_PATTERNS.get("outgoing_keywords", []):
        if word in msg:
            return "outgoing"
    return "unknown"

def save_transaction(user_id: str, transaction: Dict[str, Any]) -> dict:
    try:
        doc_ref = db.collection("users").document(user_id).collection("transactions").add(transaction)
        logging.info(f"Saved transaction for user {user_id}")
        return {"status": "success", "firebase_ref": str(doc_ref)}
    except Exception as e:
        logging.error(f"Error saving to Firebase: {e}")
        return {"status": "error", "message": str(e)}

def summarize_transactions(user_id: str) -> dict:
    ref = db.collection("users").document(user_id).collection("transactions")
    docs = ref.stream()

    summary = {
        "incoming": {"count": 0, "total": 0.0},
        "outgoing": {"count": 0, "total": 0.0},
        "unknown": {"count": 0, "total": 0.0}
    }

    for doc in docs:
        data = doc.to_dict()
        category = classify_transaction(data.get("raw", ""))
        amt = data.get("amount") or 0.0
        summary[category]["count"] += 1
        summary[category]["total"] += amt

    return summary

# ===== ENDPOINTS =====
@app.post("/analyze")
async def analyze_sms(payload: SMSPayload):
    if not payload.user_id.strip():
        raise HTTPException(status_code=400, detail="Missing or invalid 'user_id'.")

    if not validate_user(payload.user_id):
        raise HTTPException(status_code=404, detail="Sorry, no active account. Please create account.")

    results = []
    for msg in payload.messages:
        parsed = parse_message(msg)
        save_result = save_transaction(payload.user_id, parsed)
        results.append({
            "parsed": parsed,
            "save_result": save_result
        })

    return {
        "user_id": payload.user_id,
        "total_messages": len(payload.messages),
        "results": results
    }

@app.post("/analyze-summary")
async def analyze_summary(payload: SummaryPayload):
    if not payload.user_id.strip():
        raise HTTPException(status_code=400, detail="Missing or invalid 'user_id'.")

    if not validate_user(payload.user_id):
        raise HTTPException(status_code=404, detail="Sorry, no active account. Please create account.")

    summary = summarize_transactions(payload.user_id)
    return {
        "user_id": payload.user_id,
        "summary": summary
    }

@app.post("/admin-analyze")
async def admin_analyze(payload: AdminPayload):
    if payload.admin_id != "Calbrs-36":
        raise HTTPException(status_code=403, detail="Unauthorized admin access.")

    results = []
    for msg in payload.messages:
        parsed = parse_message(msg)
        results.append(parsed)

    return {
        "admin_id": payload.admin_id,
        "total_messages": len(payload.messages),
        "results": results
    }

@app.get("/")
async def health_check(request: Request):
    user_id = request.query_params.get("user_id")
    if not user_id or not user_id.strip():
        return {"status": "ok", "message": "MMTA 0.0.3.6", "user": "none_provided"}

    if validate_user(user_id):
        return {"status": "ok", "message": "MMTA 0.0.3.6", "user": "exists"}
    else:
        return {"status": "error", "message": "Please create account", "user": "not_found"}

# ===== RUN SERVER (Optional, for local dev) =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
