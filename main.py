import json
import os
import re
import requests
import logging
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------- LOGGING ---------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="MMTA Backend V9 - AI Auto-Learning Patterns")

# --------- CORS ---------
origins = [
    "https://calcue.wuaze.com/",
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

# --------- PATHS ---------
DATA_DIR = "data"
USER_DATA_DIR = os.path.join(DATA_DIR, "users")
PATTERNS_FILE = os.path.join(DATA_DIR, "patterns.json")
PENDING_PATTERNS_FILE = os.path.join(DATA_DIR, "pending_patterns.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(USER_DATA_DIR, exist_ok=True)

# --------- LOAD & SAVE UTILITIES ---------
def load_json(file_path: str) -> Dict[str, Any]:
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error in {file_path}: {e}")
            return {}
    return {}

def save_json(data: Dict[str, Any], file_path: str):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        logging.error(f"Failed to write to {file_path}: {e}")

patterns = {
    "transaction_id": r"TID[:\s]*([A-Z0-9.]+)",
    "date_time": r"(\d{2,4}[-/]\d{1,2}[-/]\d{1,2}[\sT]\d{1,2}:\d{2}(:\d{2})?)",
    "transaction_type": r"(Received|Paid|Withdrawn|Sent|Loan|Deposit|Purchase)",
    "amount": r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*Tsh",
    "currency": r"(Tsh|TZS|USD|EUR)",
    "balance": r"Balance\s+([\d,]+\.\d{2})\s*Tsh",
    "charges": {
        "total": r"Charges\s*Tsh\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
        "service_charge": r"Service\scharge\s*Tsh\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
        "gov_levy": r"Govt\sLevy\s*Tsh\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"
    },
    "net_amount": r"Net\s*Amount\s*Tsh\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
    "previous_balance": r"Previous\s*Balance\s*Tsh\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
    "balance_after": r"Balance\s*Tsh\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
    "recipient": {
        "name": r"(?:from|to\s+\d{9,15})\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*) ",
        "phone_number": r"(\d{9,12})",
        "agent_id": r"Agent\sID[:\s]*([A-Z0-9]+)"
    },
    "service_reference": r"Ref[:\s]*([A-Z0-9]+)",
    "service_provider": r"(Airtel|Tigo|Vodacom|Halopesa|M-Pesa|Bank)",
    "channel": r"(USSD|App|ATM|Agent|POS)",
    "payment_method": r"(Cash|Wallet|Card)",
    "location": r"Location[:\s]*([A-Za-z ]+)",
    "status": r"(Successful|Failed|Pending|Declined)",
    "transaction_relation": r"Relation[:\s]*([A-Za-z0-9]+)",
    "user_reference_note": r"Note[:\s]*(.+)",
    "promotions_bonus": r"Bonus[:\s]*Tsh\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
    "auth_method": r"(PIN|OTP|Fingerprint)"
}

pending_patterns = load_json(PENDING_PATTERNS_FILE)

# --------- AI Gemini API ---------
GEMINI_API_KEY = "AIzaSyBzIF1tfjov9O1Lhh7ZdP3WNJiFOd0hPco"

def clean_ai_response(text: str) -> str:
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
        return json.loads(cleaned)
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        logging.error(f"AI API error: {e}")
        return {"error": str(e)}

# --------- PATTERN MANAGEMENT ---------
def add_pending_pattern(service: str, new_patterns: Dict[str, str], raw_msg: str):
    pending_patterns.setdefault(service, []).append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_message": raw_msg,
        "proposed_patterns": new_patterns
    })
    save_json(pending_patterns, PENDING_PATTERNS_FILE)

def learn_new_pattern(service: str, approved_patterns: Dict[str, str]):
    loaded_patterns = load_json(PATTERNS_FILE)
    loaded_patterns.setdefault(service, {}).update(approved_patterns)
    save_json(loaded_patterns, PATTERNS_FILE)

# --------- UTILS ---------
def try_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(re.sub(r'[^\d.]', '', value)) if value else None
    except ValueError:
        return None

def detect_service(msg: str) -> str:
    lower = msg.lower()
    if "airtel" in lower:
        return "AirtelMoney"
    if "mpesa" in lower or "confirmed" in lower:
        return "MPESA"
    if "tigo" in lower or "mixx" in lower:
        return "TIGO"
    return "Unknown"

def parse_with_patterns(msg: str, service: str) -> Optional[Dict[str, Any]]:
    """
    Parses the message by trying to extract all fields via the patterns dict.
    Filters out null, empty, or 'Unknown' values to return only meaningful data.
    """
    if not patterns:
        return None

    extracted_data = {}
    try:
        # Helper to extract and clean values
        def extract_and_clean(regex_pattern, message, default_value=""):
            match = re.search(regex_pattern, message, re.IGNORECASE)
            value = match.group(1).strip() if match else default_value
            # Filter out empty strings, "Unknown", or default empty values
            return value if value and value.lower() != "unknown" and value != default_value else None

        # Extract simple fields
        extracted_data["transaction_id"] = extract_and_clean(patterns["transaction_id"], msg)
        extracted_data["date_time"] = extract_and_clean(patterns["date_time"], msg)
        extracted_data["transaction_type"] = extract_and_clean(patterns["transaction_type"], msg)
        
        amount_str = extract_and_clean(patterns["amount"], msg)
        extracted_data["amount"] = try_float(amount_str) if amount_str else None
        
        extracted_data["currency"] = extract_and_clean(patterns["currency"], msg)
        
        net_amount_str = extract_and_clean(patterns["net_amount"], msg)
        extracted_data["net_amount"] = try_float(net_amount_str) if net_amount_str else None
        
        previous_balance_str = extract_and_clean(patterns["previous_balance"], msg)
        extracted_data["previous_balance"] = try_float(previous_balance_str) if previous_balance_str else None
        
        balance_after_str = extract_and_clean(patterns["balance_after"], msg)
        extracted_data["balance_after"] = try_float(balance_after_str) if balance_after_str else None
        
        extracted_data["service_reference"] = extract_and_clean(patterns["service_reference"], msg)
        extracted_data["service_provider"] = extract_and_clean(patterns["service_provider"], msg)
        extracted_data["channel"] = extract_and_clean(patterns["channel"], msg)
        extracted_data["payment_method"] = extract_and_clean(patterns["payment_method"], msg)
        extracted_data["location"] = extract_and_clean(patterns["location"], msg)
        extracted_data["status"] = extract_and_clean(patterns["status"], msg)
        extracted_data["transaction_relation"] = extract_and_clean(patterns["transaction_relation"], msg)
        extracted_data["user_reference_note"] = extract_and_clean(patterns["user_reference_note"], msg)
        
        promotions_bonus_str = extract_and_clean(patterns["promotions_bonus"], msg)
        extracted_data["promotions_bonus"] = try_float(promotions_bonus_str) if promotions_bonus_str else None
        
        extracted_data["auth_method"] = extract_and_clean(patterns["auth_method"], msg)

        # Extract charges dict
        charges_data = {}
        charges_total_str = extract_and_clean(patterns["charges"]["total"], msg)
        charges_data["total"] = try_float(charges_total_str) if charges_total_str else None
        
        charges_service_charge_str = extract_and_clean(patterns["charges"]["service_charge"], msg)
        charges_data["service_charge"] = try_float(charges_service_charge_str) if charges_service_charge_str else None
        
        charges_gov_levy_str = extract_and_clean(patterns["charges"]["gov_levy"], msg)
        charges_data["gov_levy"] = try_float(charges_gov_levy_str) if charges_gov_levy_str else None
        
        # Only add charges if any sub-field has a value
        if any(v is not None for v in charges_data.values()):
            extracted_data["charges"] = charges_data
        else:
            extracted_data["charges"] = {} # Or omit entirely if you prefer

        # Extract recipient dict
        recipient_data = {}
        recipient_data["name"] = extract_and_clean(patterns["recipient"]["name"], msg)
        recipient_data["phone_number"] = extract_and_clean(patterns["recipient"]["phone_number"], msg)
        recipient_data["agent_id"] = extract_and_clean(patterns["recipient"]["agent_id"], msg)

        # Only add recipient if any sub-field has a value
        if any(v is not None for v in recipient_data.values()):
            extracted_data["recipient"] = recipient_data
        else:
            extracted_data["recipient"] = {} # Or omit entirely if you prefer

        extracted_data["service"] = service
        extracted_data["raw"] = msg

        # Final filtering: remove keys with None or empty string values
        # This will iterate over all extracted_data and its nested dictionaries
        def clean_dict(d):
            return {k: v for k, v in d.items() if v is not None and v != "" and (not isinstance(v, dict) or clean_dict(v))}

        cleaned_result = clean_dict(extracted_data)
        
        return cleaned_result if cleaned_result else None # Return None if nothing was extracted

    except re.error as e:
        logging.error(f"Regex error in parse_with_patterns: {e}")
        return None

# --------- USER DATA ---------
def save_transaction(user_id: str, transaction: Dict[str, Any]):
    user_dir = os.path.join(USER_DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    path = os.path.join(user_dir, "transactions.json")
    transactions = load_transactions(user_id)
    transactions.append(transaction)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transactions, f, indent=2)

def load_transactions(user_id: str) -> List[Dict[str, Any]]:
    path = os.path.join(USER_DATA_DIR, user_id, "transactions.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_summary(user_id: str, summary: Dict[str, Any]):
    user_dir = os.path.join(USER_DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    path = os.path.join(user_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

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
        except (ValueError, TypeError):
            continue

    summary = {"daily": dict(daily), "weekly": dict(weekly), "monthly": dict(monthly)}
    save_summary(user_id, summary)
    return summary

def get_ai_advice(user_id: str) -> str:
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

# --------- MAIN ANALYSIS ---------
class SMSPayload(BaseModel):
    user_id: str
    messages: List[str]

class PatternApprovalPayload(BaseModel):
    service: str
    proposed_patterns_index: int
    approve: bool
    new_type_name: Optional[str] = None

def analyze_message(msg: str, user_id: str) -> Dict[str, Any]:
    service = detect_service(msg)
    parsed = parse_with_patterns(msg, service)

    # Fields muhimu za kuhakikisha zipo kwenye local parsing
    critical_fields = ["amount", "transaction_id", "date_time"]

    # Check kama local parse ni None au kuna critical field haijapatikana
    missing_critical = (
        parsed is None or
        any(parsed.get(field) is None for field in critical_fields)
    )

    if not missing_critical:
        # Local parsing imefanikiwa vizuri, hifadhi na rudisha
        save_transaction(user_id, parsed)
        return {"success": True, "source": "local", "data": parsed}

    # Local parsing haikutosha, tumia AI kama fallback
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
                        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "phone_number": groups.get('phone_number'),
                        "service": service,
                        "transaction_id": groups.get('transaction_id'),
                        "new_balance": try_float(groups.get('new_balance')),
                        "participant": groups.get('participant'),
                        "raw": msg
                    }
                    filtered_data = {k: v for k, v in data.items() if v is not None and v != ""}
                    save_transaction(user_id, filtered_data)
                    return {"success": True, "source": "AI", "data": filtered_data}
            except re.error:
                continue

    return {"success": False, "error": "Failed to parse message even with AI."}

# --------- API ROUTES ---------
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
    save_json(pending_patterns, PENDING_PATTERNS_FILE)

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
    data = load_json(PATTERNS_FILE)
    return {"service_names": list(data.keys())}

@app.get("/service-names/{service_name}")
def get_service_patterns(service_name: str):
    data = load_json(PATTERNS_FILE)
    if service_name not in data:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found.")
    return {service_name: data[service_name]}

@app.get("/user-summary/{user_id}")
def get_user_summary(user_id: str):
    summary_path = os.path.join(USER_DATA_DIR, user_id, "summary.json")
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="User summary not found.")
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f) 

@app.get("/user-transactions/{user_id}")
def get_user_transactions(user_id: str):   
    transactions = load_transactions(user_id)
    if not transactions:
        raise HTTPException(status_code=404, detail="No transactions found for this user.")
    return {"transactions": transactions}

@app.post("/user-transactions/{user_id}")
def save_user_transaction(user_id: str, transaction: Dict[str, Any]):
    if not transaction:
        raise HTTPException(status_code=400, detail="Transaction data is required.")
    save_transaction(user_id, transaction)
    return {"status": "success", "message": "Transaction saved successfully."}

@app.get("/user-advice/{user_id}")
def get_user_advice(user_id: str):
    advice = get_ai_advice(user_id)
    return {"advice": advice}
