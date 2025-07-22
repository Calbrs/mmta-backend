import json, os, re, requests
from datetime import datetime
from collections import defaultdict
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware

# ------------------ LOGGING ------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="MMTA Backend V9 - AI Auto-Learning Patterns")

# ------------------ CORS ------------------
origins = [
    "https://mmta-backend.onrender.com",
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ GLOBAL PATHS ------------------
DATA_DIR = "data"
USER_DATA_DIR = os.path.join(DATA_DIR, "users")
PATTERNS_FILE = os.path.join(DATA_DIR, "patterns.json")
PENDING_PATTERNS_FILE = os.path.join(DATA_DIR, "pending_patterns.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(USER_DATA_DIR, exist_ok=True)

# ------------------ LOAD & SAVE ------------------
def load_patterns(file_path: str) -> Dict[str, Any]:
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {file_path}: {e}")
            return {}
    return {}

def save_patterns(data: Dict[str, Any], file_path: str):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        logging.error(f"Error writing to {file_path}: {e}")

patterns = load_patterns(PATTERNS_FILE)
pending_patterns = load_patterns(PENDING_PATTERNS_FILE)

# ------------------ EXAMPLE DEFAULT PATTERNS ------------------
example_tigo_deposit_pattern = (
    r"Received Tsh\s*(?P<amount>[\d,\.]+|[\d,]+)\s*from TIGO\s*-\s*"
    r"(?P<participant>[A-Z\s\.]+)\s*-\s*(?P<phone_number>\d+)\.Balance Tsh\s*"
    r"(?P<new_balance>[\d,\.]+|[\d,]+)\.\s*TID:(?P<transaction_id>[\w\.]+)"
)
patterns.setdefault("TIGO", {}).setdefault("deposit", example_tigo_deposit_pattern)

example_airtel_payment_pattern = (
    r"Airtel Money:\s*TZS\s*(?P<amount>[\d,]+(?:\.\d{2})?)\s*sent to\s*"
    r"(?P<participant>[A-Za-z\s\.]+)\.\s*from\s*(?P<phone_number>[\d+X]+)\.\s*"
    r"Ref No\.\s*(?P<transaction_id>\w+)\.\s*New balance TZS\s*"
    r"(?P<new_balance>[\d,]+(?:\.\d{2})?)\."
)
patterns.setdefault("AirtelMoney", {}).setdefault("payment", example_airtel_payment_pattern)

save_patterns(patterns, PATTERNS_FILE)

# ------------------ GEMINI API ------------------
GEMINI_API_KEY = "AIzaSyAH-0xDNapOWOrCZH1OczyXoOniRhz6jJA"

def clean_ai_response(text: str) -> str:
    """Ondoa ```json``` au ``` kwenye response ya AI."""
    return text.strip().removeprefix("```json").removeprefix("```").strip("` \n")

def ask_ai_for_regex(msg: str) -> Optional[Dict[str, Any]]:
    prompt = (
        f"Analyze this SMS transaction message: '{msg}'.\n"
        f"Extract a JSON containing regex patterns with named capture groups for "
        f"amount, participant, phone_number, transaction_id, new_balance if possible.\n"
        f"Respond in JSON: {{'patterns': {{'type': 'regex_here'}}}}"
    )
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        res.raise_for_status()
        data = res.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        cleaned = clean_ai_response(text)
        logging.info(f"AI raw response: {cleaned}")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"error": "Failed to parse AI response as JSON."}
    except requests.exceptions.RequestException as e:
        logging.error(f"Gemini API request failed: {e}")
        return {"error": f"Gemini API request failed: {e}"}

# ------------------ PATTERN MANAGEMENT ------------------
def add_pending_pattern(service: str, new_patterns: Dict[str, str], raw_msg: str):
    pending_patterns.setdefault(service, []).append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_message": raw_msg,
        "proposed_patterns": new_patterns
    })
    save_patterns(pending_patterns, PENDING_PATTERNS_FILE)

def learn_new_pattern(service: str, approved_patterns: Dict[str, str]):
    patterns.setdefault(service, {}).update(approved_patterns)
    save_patterns(patterns, PATTERNS_FILE)

# ------------------ TRANSACTION PARSING ------------------
class SMSPayload(BaseModel):
    user_id: str
    messages: List[str]

class PatternApprovalPayload(BaseModel):
    service: str
    proposed_patterns_index: int
    approve: bool
    new_type_name: Optional[str] = None

def detect_service(msg: str) -> str:
    msg_low = msg.lower()
    if "airtel" in msg_low:
        return "AirtelMoney"
    if "mpesa" in msg_low or "confirmed" in msg_low:
        return "MPESA"
    if "tigo" in msg_low or "mixx" in msg_low:
        return "TIGO"
    return "Unknown"

def try_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(re.sub(r'[^\d.]', '', value)) if value else None
    except ValueError:
        return None

def parse_with_patterns(msg: str, service: str) -> Optional[Dict[str, Any]]:
    extracted_data = {
        "type": None,
        "amount": None,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phone_number": None,
        "service": service,
        "transaction_id": None,
        "new_balance": None,
        "participant": None,
        "raw": msg
    }
    if service not in patterns:
        return None
    for txn_type, regex_pattern in patterns[service].items():
        try:
            match = re.search(regex_pattern, msg, re.IGNORECASE)
            if match:
                extracted_data["type"] = txn_type
                groups = match.groupdict()
                extracted_data['amount'] = try_float(groups.get('amount'))
                extracted_data['new_balance'] = try_float(groups.get('new_balance'))
                extracted_data['phone_number'] = groups.get('phone_number')
                extracted_data['transaction_id'] = groups.get('transaction_id')
                extracted_data['participant'] = groups.get('participant')
                return extracted_data
        except re.error as e:
            logging.error(f"Regex error: {e}")
    return None

# ------------------ USER DATA ------------------
def save_transaction(user_id: str, transaction: Dict[str, Any]):
    user_dir = os.path.join(USER_DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    path = os.path.join(user_dir, "transactions.json")
    transactions = load_transactions(user_id)
    transactions.append(transaction)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transactions, f, indent=2)

def load_transactions(user_id: str):
    path = os.path.join(USER_DATA_DIR, user_id, "transactions.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_summary(user_id, summary):
    user_dir = os.path.join(USER_DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    path = os.path.join(user_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

def update_user_summary(user_id):
    transactions = load_transactions(user_id)
    daily = defaultdict(float)
    weekly = defaultdict(float)
    monthly = defaultdict(float)
    for t in transactions:
        if not t.get("amount") or not t.get("date"):
            continue
        try:
            amount = float(t["amount"])
            dt = datetime.strptime(t["date"], "%Y-%m-%d %H:%M:%S")
            daily[dt.strftime("%Y-%m-%d")] += amount
            weekly[f"week_{dt.strftime('%U_%Y')}"] += amount
            monthly[dt.strftime("%B_%Y")] += amount
        except (ValueError, TypeError):
            continue
    summary = {"daily": dict(daily), "weekly": dict(weekly), "monthly": dict(monthly)}
    save_summary(user_id, summary)
    return summary

def get_ai_advice(user_id):
    summary_path = os.path.join(USER_DATA_DIR, user_id, "summary.json")
    if not os.path.exists(summary_path):
        return "Hakuna data ya kutosha kutoa ushauri."
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    if summary.get("weekly"):
        latest_week = max(summary["weekly"], key=summary["weekly"].get)
        max_week_spending = summary["weekly"][latest_week]
        if max_week_spending > 100000:
            return f"Wiki hii umetumia {max_week_spending:,.2f} TZS. Jaribu kupunguza matumizi."
        elif max_week_spending > 0:
            return "Matumizi yako wiki hii yako sawa. Endelea kudhibiti gharama zako."
    return "Bado hatuna data ya kutosha ya matumizi wiki hii."

# ------------------ MAIN ANALYSIS ------------------
def analyze_message(msg: str, user_id: str) -> Dict[str, Any]:
    service = detect_service(msg)
    result = parse_with_patterns(msg, service)
    if result:
        save_transaction(user_id, result)
        return {"success": True, "source": "local", "data": result}

    ai_response = ask_ai_for_regex(msg)
    if not ai_response or "error" in ai_response:
        return {"success": False, "error": ai_response.get("error", "AI analysis failed.")}

    proposed_patterns = ai_response.get("patterns", {})
    if proposed_patterns:
        add_pending_pattern(service, proposed_patterns, msg)
        for txn_type, regex in proposed_patterns.items():
            try:
                match = re.search(regex, msg, re.IGNORECASE)
                if match:
                    groups = match.groupdict()
                    data = {
                        "type": txn_type,
                        "amount": try_float(groups.get('amount')),
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "phone_number": groups.get('phone_number'),
                        "service": service,
                        "transaction_id": groups.get('transaction_id'),
                        "new_balance": try_float(groups.get('new_balance')),
                        "participant": groups.get('participant'),
                        "raw": msg
                    }
                    save_transaction(user_id, data)
                    return {"success": True, "source": "AI", "data": data}
            except re.error:
                continue
    return {"success": False, "error": "Failed to parse message even with AI."}

# ------------------ API ENDPOINTS ------------------
@app.get("/")
def root():
    return {"message": "MMTA Backend V9 - AI Auto-Learning Patterns + User Summary"}

@app.post("/analyze")
def analyze_sms(payload: SMSPayload):
    results = [analyze_message(msg, payload.user_id) for msg in payload.messages]
    summary = update_user_summary(payload.user_id)
    return {
        "analysis": results,
        "summary": summary,
        "advice": get_ai_advice(payload.user_id),
        "total_messages_processed": len(payload.messages)
    }

@app.get("/patterns/pending")
def get_pending_patterns():
    return {"pending_patterns": pending_patterns}

@app.post("/patterns/approve")
def approve_or_reject_pattern(payload: PatternApprovalPayload):
    service = payload.service
    idx = payload.proposed_patterns_index
    if service not in pending_patterns or idx >= len(pending_patterns[service]):
        raise HTTPException(status_code=404, detail=f"Proposed patterns for service '{service}' not found.")
    proposed_entry = pending_patterns[service].pop(idx)
    save_patterns(pending_patterns, PENDING_PATTERNS_FILE)
    if payload.approve:
        transaction_type_key = next(iter(proposed_entry["proposed_patterns"]))
        regex_to_add = proposed_entry["proposed_patterns"][transaction_type_key]
        final_type_name = payload.new_type_name or transaction_type_key
        learn_new_pattern(service, {final_type_name: regex_to_add})
        return {"status": "success", "message": "Patterns approved."}
    else:
        return {"status": "success", "message": "Patterns rejected."}

@app.get("/service-names")
def get_service_names():
    data = load_patterns(PATTERNS_FILE)
    return {"service_names": list(data.keys())}

@app.get("/service-names/{service_name}")
def get_service_patterns(service_name: str):
    data = load_patterns(PATTERNS_FILE)
    if service_name not in data:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found.")
    return {service_name: data[service_name]}
