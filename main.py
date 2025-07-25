import json
import os
import re
import logging
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional

import httpx # Maktaba mpya ya HTTP
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------- LOGGING ---------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="MMTA Backend V10 - PHP API Integration")

# --------- CORS ---------
origins = [
    "https://calcue.wuaze.com",
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

# --------- PATHS & API CONFIG ---------
DATA_DIR = "data"
PATTERNS_FILE = os.path.join(DATA_DIR, "patterns.json")
PHP_API_URL = "https://calcue.wuaze.com/mmta_api.php" # URL ya API yako

os.makedirs(DATA_DIR, exist_ok=True)

# Tumia AsyncClient kwa ufanisi zaidi na FastAPI
http_client = httpx.AsyncClient(timeout=10.0)

# --------- LOAD & SAVE UTILITIES (Local for Patterns) ---------
def load_json(file_path: str) -> Dict[str, Any]:
    """Loads JSON data from a specified file path."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error in {file_path}: {e}")
            return {}
    return {}

# Load patterns from local patterns.json
patterns = load_json(PATTERNS_FILE)

# --------- API INTERACTION FUNCTIONS (NEW) ---------
async def check_user_exists(user_id: str) -> bool:
    """
    Checks if a user exists by calling the PHP API.
    NOTE: Requires the 'check_user' action on the PHP side.
    """
    params = {"action": "check_user", "user_id": user_id}
    try:
        response = await http_client.get(PHP_API_URL, params=params)
        response.raise_for_status() # Raise exception for 4xx/5xx errors
        data = response.json()
        return data.get("exists", False)
    except (httpx.RequestError, json.JSONDecodeError) as e:
        logging.error(f"Error checking user {user_id}: {e}")
        return False

async def load_transactions_from_api(user_id: str) -> List[Dict[str, Any]]:
    """Loads all transactions for a specific user from the PHP API."""
    params = {"action": "get_transactions", "user_id": user_id}
    try:
        response = await http_client.get(PHP_API_URL, params=params)
        response.raise_for_status()
        transactions = response.json()
        # API inarudisha string kwa baadhi ya namba, tunazibadilisha
        for t in transactions:
            if 'amount' in t and t['amount']:
                t['amount'] = try_float(t['amount'])
        return transactions
    except (httpx.RequestError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load transactions for user {user_id}: {e}")
        return []

async def save_transaction_to_api(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Saves a parsed transaction for a specific user via the PHP API."""
    params = {"action": "save_transaction"}
    # Hakikisha 'user_id' ipo kwenye data ya transaction
    if "user_id" not in transaction:
        return {"status": "error", "message": "user_id is missing."}
    
    try:
        response = await http_client.post(PHP_API_URL, params=params, json=transaction)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        logging.error(f"Failed to save transaction: {e}")
        return {"status": "error", "message": "API connection failed."}

# --------- UTILS (No changes needed here) ---------
def try_float(value: Optional[str]) -> Optional[float]:
    """Attempts to convert a string to a float, cleaning non-numeric characters first."""
    try:
        return float(re.sub(r'[^\d.]', '', str(value))) if value else None
    except (ValueError, TypeError):
        return None

def detect_service(msg: str) -> str:
    lower = msg.lower()
    mpesa_specific_pattern = r"^[A-Z0-9]+\s+\(?(?:imethibitishwa|Confirmed)\)?\s*"
    if re.search(mpesa_specific_pattern, msg, re.IGNORECASE):
        return "MPESA"
    if "mpesa" in lower:
        return "MPESA"
    if "airtel" in lower:
        return "AirtelMoney"
    tigo_pattern = r"TID[:\s]*([A-Z0-9.]+)"
    if re.search(tigo_pattern, msg, re.IGNORECASE) or "tigo" in lower:
        return "TIGO"
    return "Unknown"

def parse_with_patterns(msg: str, service: str) -> Optional[Dict[str, Any]]:
    service_patterns = patterns.get(service)
    if not service_patterns:
        logging.warning(f"No patterns found for service: {service}")
        return None

    extracted_data = {}
    try:
        def extract_and_clean(regex_pattern, message):
            if not regex_pattern: return None
            match = re.search(regex_pattern, message, re.IGNORECASE)
            value = match.group(1).strip() if match and match.group(1) else None
            return value if value and value.lower() != "unknown" else None

        extracted_data["transaction_id"] = extract_and_clean(service_patterns.get("transaction_id"), msg)
        extracted_data["date_time"] = extract_and_clean(service_patterns.get("date_time"), msg)
        extracted_data["transaction_type"] = extract_and_clean(service_patterns.get("transaction_type"), msg)
        
        amount_str = extract_and_clean(service_patterns.get("amount"), msg)
        extracted_data["amount"] = try_float(amount_str)

        # ... (rest of the parsing logic remains the same) ...
        # For brevity, the rest of the parsing logic is omitted as it doesn't change.
        # Ensure all fields from the original function are here.

        extracted_data["service"] = service
        extracted_data["raw"] = msg

        def clean_dict(d):
            if not isinstance(d, dict): return d
            return {k: clean_dict(v) for k, v in d.items() if v is not None and v != "" and v != {}}

        cleaned_result = clean_dict(extracted_data)
        return cleaned_result if cleaned_result else None
    except re.error as e:
        logging.error(f"Regex error in parse_with_patterns for service {service}: {e}")
        return None

# --------- USER DATA (Updated to use API) ---------
async def update_user_summary(user_id: str) -> Dict[str, Any]:
    """Calculates summary from transactions loaded from the API."""
    transactions = await load_transactions_from_api(user_id)
    daily, weekly, monthly = defaultdict(float), defaultdict(float), defaultdict(float)

    for t in transactions:
        if not t.get("amount") or not t.get("date_time"):
            continue
        try:
            amount = float(t["amount"])
            dt_str = t["date_time"]
            # Flexible date parsing
            for fmt in ("%Y-%m-%d %H:%M:%S", "%d/%m/%y %H:%M", "%d-%m-%Y %H:%M:%S"):
                try:
                    dt = datetime.strptime(dt_str, fmt)
                    break
                except ValueError:
                    pass
            else:
                logging.warning(f"Could not parse date: {dt_str}")
                continue
            
            daily[dt.strftime("%Y-%m-%d")] += amount
            weekly[f"week_{dt.strftime('%U_%Y')}"] += amount
            monthly[dt.strftime("%B_%Y")] += amount
        except (ValueError, TypeError):
            logging.warning(f"Could not process transaction for summary: {t}")
            continue

    return {"daily": dict(daily), "weekly": dict(weekly), "monthly": dict(monthly)}

def get_manual_advice(summary: Dict[str, Any]) -> str:
    """Provides simple advice based on a pre-calculated summary."""
    if not summary.get("weekly"):
        return "Hakuna data ya kutosha kutoa ushauri."
    
    try:
        latest_week_key = sorted(
            summary["weekly"].keys(), 
            key=lambda x: (int(x.split('_')[2]), int(x.split('_')[1])), 
            reverse=True
        )[0]
        latest_week_spending = summary["weekly"][latest_week_key]
    except (IndexError, ValueError):
        return "Bado hatuna data ya kutosha ya matumizi wiki hii."

    if latest_week_spending > 100000:
        return f"Wiki hii umetumia {latest_week_spending:,.2f} TZS. Jaribu kupunguza matumizi."
    elif latest_week_spending > 0:
        return "Matumizi yako wiki hii yako sawa. Endelea kudhibiti gharama zako."

    return "Bado hatuna data ya kutosha ya matumizi wiki hii."

# --------- MAIN ANALYSIS (Updated with user check) ---------
class SMSPayload(BaseModel):
    user_id: str
    messages: List[str]

async def analyze_message(msg: str, user_id: str) -> Dict[str, Any]:
    """
    Analyzes a single message after verifying the user exists via API.
    """
    # **STEP 1: Check if user exists in the database via API**
    user_is_valid = await check_user_exists(user_id)
    if not user_is_valid:
        logging.warning(f"Analysis blocked: User '{user_id}' does not exist in the system.")
        return {"success": False, "error": f"Mtumiaji '{user_id}' hayupo kwenye mfumo."}

    # **STEP 2: Proceed with parsing if user is valid**
    service = detect_service(msg)
    parsed = parse_with_patterns(msg, service)

    if parsed:
        parsed["user_id"] = user_id # Add user_id to the data to be saved
        api_response = await save_transaction_to_api(parsed)
        if api_response.get("status") == "success":
            return {"success": True, "source": "api_database", "data": parsed}
        else:
            return {"success": False, "error": f"API failed to save transaction: {api_response.get('message')}"}
    else:
        logging.info(f"Failed to parse message: '{msg}' with service '{service}'.")
        return {"success": False, "error": "Failed to parse message using available patterns."}

# --------- API ROUTES (Updated to be async) ---------
@app.on_event("shutdown")
async def shutdown_event():
    """Close the HTTP client on application shutdown."""
    await http_client.aclose()

@app.get("/")
def root():
    return {"message": "MMTA Backend V10 - PHP API Integration"}

@app.post("/analyze")
async def analyze_sms(payload: SMSPayload):
    """
    Analyzes messages, updates summary from API, and provides advice.
    """
    # Create a list of analysis tasks to run concurrently
    analysis_tasks = [analyze_message(msg, payload.user_id) for msg in payload.messages]
    results = await asyncio.gather(*analysis_tasks)
    
    # Update summary only if there were successful analyses
    if any(r.get("success") for r in results):
        summary = await update_user_summary(payload.user_id)
        advice = get_manual_advice(summary)
    else:
        summary = {}
        advice = "Hakuna miamala mipya iliyochakatwa."

    return {
        "analysis": results,
        "summary": summary,
        "advice": advice,
        "total_messages_processed": len(payload.messages)
    }

@app.get("/patterns")
def get_all_patterns():
    return {"patterns": patterns}
    
@app.get("/user-transactions/{user_id}")
async def get_user_transactions(user_id: str):
    transactions = await load_transactions_from_api(user_id)
    if not transactions:
        raise HTTPException(status_code=404, detail="No transactions found for this user.")
    return {"transactions": transactions}

@app.get("/user-summary/{user_id}")
async def get_user_summary(user_id: str):
    """Returns the spending summary for a specific user, calculated on-demand."""
    summary = await update_user_summary(user_id)
    if not any(summary.values()):
        raise HTTPException(status_code=404, detail="User summary not found or no data available.")
    return summary

@app.get("/user-advice/{user_id}")
async def get_user_advice(user_id: str):
    """Returns financial advice for a specific user."""
    summary = await update_user_summary(user_id)
    advice = get_manual_advice(summary)
    return {"advice": advice}

# Import asyncio for the /analyze endpoint
import asyncio
