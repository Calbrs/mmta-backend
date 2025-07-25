import json
import os
import re
import logging
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MMTA Backend V9 - Manual Pattern Learning")

origins = [
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

DATA_DIR = "data"
USER_DATA_DIR = os.path.join(DATA_DIR, "users")
PATTERNS_FILE = os.path.join(DATA_DIR, "patterns.json")

os.makedirs(USER_DATA_DIR, exist_ok=True)

def load_json(file_path: str) -> Dict[str, Any]:
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load JSON from {file_path}: {e}")
    return {}

def save_json(data: Any, file_path: str):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save JSON to {file_path}: {e}")

patterns = load_json(PATTERNS_FILE)  # Predefined regex patterns for services (you need to create this file)

def try_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(re.sub(r'[^\d.]', '', value)) if value else None
    except ValueError:
        return None

def detect_service(msg: str) -> str:
    lower = msg.lower()
    if re.search(r"^[A-Z0-9]+\s+\(?(?:imethibitishwa|confirmed)\)?\s*", msg, re.I):
        return "MPESA"
    if "mpesa" in lower:
        return "MPESA"
    if "airtel" in lower:
        return "AirtelMoney"
    if re.search(r"TID[:\s]*([A-Z0-9.]+)", msg, re.I) or "tigo" in lower:
        return "TIGO"
    return "Unknown"

def parse_with_patterns(msg: str, service: str) -> Optional[Dict[str, Any]]:
    service_patterns = patterns.get(service)
    if not service_patterns:
        logging.warning(f"No patterns for service: {service}")
        return None

    extracted = {}
    try:
        def extract(regex, message, default=""):
            if not regex:
                return None
            m = re.search(regex, message, re.I)
            val = m.group(1).strip() if m and m.group(1) else default
            if val and val.lower() != "unknown" and val != default:
                return val
            return None

        extracted["transaction_id"] = extract(service_patterns.get("transaction_id"), msg)
        extracted["date_time"] = extract(service_patterns.get("date_time"), msg)
        extracted["transaction_type"] = extract(service_patterns.get("transaction_type"), msg)

        amount_str = extract(service_patterns.get("amount"), msg)
        extracted["amount"] = try_float(amount_str) if amount_str else None

        extracted["currency"] = extract(service_patterns.get("currency"), msg)
        net_amount_str = extract(service_patterns.get("net_amount"), msg)
        extracted["net_amount"] = try_float(net_amount_str) if net_amount_str else None

        # ... add more fields as needed

        extracted["service"] = service
        extracted["raw"] = msg

        # Clean out None or empty fields
        cleaned = {k: v for k, v in extracted.items() if v not in [None, "", {}]}
        return cleaned if cleaned else None
    except re.error as e:
        logging.error(f"Regex error: {e}")
        return None

def get_user_dir(user_id: str) -> str:
    path = os.path.join(USER_DATA_DIR, user_id)
    os.makedirs(path, exist_ok=True)
    return path

def load_transactions(user_id: str) -> List[Dict[str, Any]]:
    path = os.path.join(get_user_dir(user_id), "transactions.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading transactions for {user_id}: {e}")
        return []

def save_transaction(user_id: str, transaction: Dict[str, Any]):
    transactions = load_transactions(user_id)
    transactions.append(transaction)
    path = os.path.join(get_user_dir(user_id), "transactions.json")
    save_json(transactions, path)

def update_user_summary(user_id: str) -> Dict[str, Any]:
    transactions = load_transactions(user_id)
    daily = defaultdict(float)
    weekly = defaultdict(float)
    monthly = defaultdict(float)

    for t in transactions:
        if not t.get("amount") or not t.get("date_time"):
            continue
        try:
            amount = float(t["amount"])
            dt = datetime.strptime(t["date_time"], "%Y-%m-%d %H:%M:%S")
            daily[dt.strftime("%Y-%m-%d")] += amount
            weekly[f"week_{dt.strftime('%U_%Y')}"] += amount
            monthly[dt.strftime("%B_%Y")] += amount
        except Exception:
            continue

    summary = {
        "daily": dict(daily),
        "weekly": dict(weekly),
        "monthly": dict(monthly),
    }
    summary_path = os.path.join(get_user_dir(user_id), "summary.json")
    save_json(summary, summary_path)
    return summary

def get_manual_advice(user_id: str) -> str:
    summary_path = os.path.join(get_user_dir(user_id), "summary.json")
    if not os.path.exists(summary_path):
        return "Hakuna data ya kutosha kutoa ushauri."

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        if summary.get("weekly"):
            latest_week_key = sorted(summary["weekly"].keys(), reverse=True)[0]
            latest_spending = summary["weekly"][latest_week_key]
            if latest_spending > 100000:
                return f"Wiki hii umetumia {latest_spending:,.2f} TZS. Jaribu kupunguza matumizi."
            elif latest_spending > 0:
                return "Matumizi yako wiki hii yako sawa. Endelea kudhibiti gharama zako."
    except Exception:
        return "Tatizo lilitokea katika kusoma data ya ushauri."

    return "Bado hatuna data ya kutosha ya matumizi wiki hii."

class SMSPayload(BaseModel):
    user_id: str
    messages: List[str]

def analyze_message(msg: str, user_id: str) -> Dict[str, Any]:
    service = detect_service(msg)
    parsed = parse_with_patterns(msg, service)
    if parsed:
        save_transaction(user_id, parsed)
        return {"success": True, "data": parsed}
    else:
        logging.info(f"Failed to parse message: '{msg}' with service '{service}'.")
        return {"success": False, "error": "Failed to parse message."}

@app.post("/analyze")
def analyze_sms(payload: SMSPayload):
    results = [analyze_message(msg, payload.user_id) for msg in payload.messages]
    summary = update_user_summary(payload.user_id)
    advice = get_manual_advice(payload.user_id)
    return {
        "analysis": results,
        "summary": summary,
        "advice": advice,
        "total_messages_processed": len(payload.messages)
    }

@app.get("/")
def root():
    return {"message": "MMTA Backend V9 - Manual Pattern Learning"}

# Add other routes as needed
