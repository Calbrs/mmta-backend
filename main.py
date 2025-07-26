import json
import os
import re
import logging
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# --------- LOGGING ---------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------- FASTAPI APP INIT ---------
app = FastAPI(title="MMTA Backend V9 - Database Integration")

# --------- CORS SETUP ---------
origins = [
    "https://calcue.wuaze.com", # Your frontend domain
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    # Add any other origins your frontend might be running from
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- DIRECTORY SETUP (Still needed for patterns.json, but not user data) ---------
DATA_DIR = "data"
PATTERNS_FILE = os.path.join(DATA_DIR, "patterns.json")
# os.makedirs(USER_DATA_DIR, exist_ok=True) # No longer needed for user data folders

# --------- PHP API URL ---------
PHP_API_BASE_URL = "https://calcue.wuaze.com/mmta_api.php"

# --------- JSON UTILS (Only for patterns.json) ---------
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

# --------- LOAD PATTERNS ---------
patterns = load_json(PATTERNS_FILE)

# --------- HELPERS (No change to parsing logic) ---------
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

        extracted["service"] = service
        extracted["raw"] = msg

        cleaned = {k: v for k, v in extracted.items() if v not in [None, "", {}]}
        return cleaned if cleaned else None
    except re.error as e:
        logging.error(f"Regex error: {e}")
        return None

# --------- DATABASE INTERACTION FUNCTIONS (via PHP API) ---------
async def save_transaction_to_db(user_id: str, transaction: Dict[str, Any]) -> bool:
    """Saves a single transaction to the MySQL DB via PHP API."""
    try:
        # The PHP API expects user_id in the POST body for 'save_transaction' action
        payload = {"user_id": user_id, **transaction}
        logging.info(f"Attempting to save transaction for user {user_id} to PHP API. Payload: {payload}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PHP_API_BASE_URL}?action=save_transaction",
                json=payload,
                timeout=30 # Add a timeout
            )
            response.raise_for_status() # Raises an exception for 4xx/5xx responses
            result = response.json()
            if result.get("status") == "success":
                logging.info(f"Transaction for user {user_id} saved successfully to DB. PHP API response: {result}")
                return True
            else:
                logging.error(f"Failed to save transaction to DB for user {user_id}: {result.get('message', 'Unknown error')}. PHP API response: {result}")
                return False
    except httpx.HTTPStatusError as exc:
        logging.error(f"HTTP error saving transaction for user {user_id}: {exc.response.status_code} - {exc.response.text}")
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Failed to save transaction to database: {exc.response.json().get('error', 'Server error')}"
        )
    except httpx.RequestError as exc:
        logging.error(f"Network error saving transaction for user {user_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not connect to database service. Error: {exc}"
        )
    except Exception as e:
        logging.error(f"Unexpected error saving transaction for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while saving transaction."
        )

async def load_transactions_from_db(user_id: str) -> List[Dict[str, Any]]:
    """Loads all transactions for a user from MySQL DB via PHP API."""
    try:
        logging.info(f"Attempting to load transactions for user {user_id} from PHP API.")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{PHP_API_BASE_URL}?action=get_transactions&user_id={user_id}",
                timeout=30 # Add a timeout
            )
            response.raise_for_status()
            transactions = response.json()
            logging.info(f"Successfully loaded transactions for user {user_id}. PHP API response: {transactions}")
            if isinstance(transactions, list):
                return transactions
            else:
                logging.error(f"Unexpected response format for user {user_id} transactions: {transactions}. Expected a list.")
                return []
    except httpx.HTTPStatusError as exc:
        logging.error(f"HTTP error loading transactions for user {user_id}: {exc.response.status_code} - {exc.response.text}")
        # If user not found (404), return empty list rather than raising error for this specific case
        if exc.response.status_code == 404 and "User with ID" in exc.response.text:
            logging.warning(f"User {user_id} not found in DB, returning empty transactions list.")
            return []
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Failed to load transactions from database: {exc.response.json().get('error', 'Server error')}"
        )
    except httpx.RequestError as exc:
        logging.error(f"Network error loading transactions for user {user_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not connect to database service. Error: {exc}"
        )
    except Exception as e:
        logging.error(f"Unexpected error loading transactions for user {user_id}: {e}")
        return [] # Return empty list on unexpected errors for data loading

async def get_user_summary_from_db(user_id: str) -> Dict[str, Any]:
    """Calculates and returns user summary based on transactions from DB."""
    transactions = await load_transactions_from_db(user_id)
    
    daily = defaultdict(float)
    weekly = defaultdict(float)
    monthly = defaultdict(float)

    for t in transactions:
        if not t.get("amount") or not t.get("date_time"):
            logging.warning(f"Skipping transaction with missing amount or date_time: {t}")
            continue
        try:
            amount = float(t["amount"])
            # Adjust date_time format if needed based on what your PHP API returns
            # Assuming PHP API returns format like 'YYYY-MM-DD HH:MM:SS'
            dt = datetime.strptime(str(t["date_time"]), "%Y-%m-%d %H:%M:%S")
            daily[dt.strftime("%Y-%m-%d")] += amount
            weekly[f"week_{dt.strftime('%U_%Y')}"] += amount
            monthly[dt.strftime("%B_%Y")] += amount
        except Exception as e:
            logging.warning(f"Skipping transaction due to parsing error: {t} - {e}")
            continue

    summary = {
        "daily": dict(daily),
        "weekly": dict(weekly),
        "monthly": dict(monthly),
    }
    return summary

async def get_advice_from_summary(user_id: str) -> str:
    """Provides advice based on the calculated summary."""
    summary = await get_user_summary_from_db(user_id)

    if not summary.get("weekly"):
        return "Hakuna data ya kutosha kutoa ushauri."

    try:
        latest_week_key = sorted(summary["weekly"].keys(), reverse=True)[0]
        latest_spending = summary["weekly"][latest_week_key]
        if latest_spending > 100000:
            return f"Wiki hii umetumia {latest_spending:,.2f} TZS. Jaribu kupunguza matumizi."
        elif latest_spending > 0:
            return "Matumizi yako wiki hii yako sawa. Endelea kudhibiti gharama zako."
    except Exception as e:
        logging.error(f"Tatizo lilitokea katika kusoma data ya ushauri for user {user_id}: {e}")
        return "Tatizo lilitokea katika kusoma data ya ushauri."

    return "Bado hatuna data ya kutosha ya matumizi wiki hii."

# --------- INPUT MODEL ---------
class SMSPayload(BaseModel):
    user_id: str
    messages: List[str]

# --------- CORE LOGIC (Modified to use DB functions) ---------
async def analyze_message(msg: str, user_id: str) -> Dict[str, Any]:
    service = detect_service(msg)
    parsed = parse_with_patterns(msg, service)
    if parsed:
        # Attempt to save to database via PHP API
        logging.info(f"Attempting to save parsed message for user {user_id}: {parsed}")
        success_db = await save_transaction_to_db(user_id, parsed)
        if success_db:
            return {"success": True, "data": parsed, "db_saved": True}
        else:
            return {"success": False, "error": "Failed to save to database.", "data": parsed, "db_saved": False}
    else:
        logging.info(f"Failed to parse message: '{msg}' with service '{service}'.")
        return {"success": False, "error": "Failed to parse message."}

# --------- ENDPOINTS (Modified to use DB functions) ---------
@app.post("/analyze")
async def analyze_sms(payload: SMSPayload):
    # ✅ Validate user_id
    if not payload.user_id.strip():
        raise HTTPException(status_code=400, detail="Missing or invalid 'user_id'.")

    # Check if user exists in the database via PHP API
    try:
        logging.info(f"Checking user existence for user_id: {payload.user_id}")
        async with httpx.AsyncClient() as client:
            user_check_response = await client.get(f"{PHP_API_BASE_URL}?action=check_user&user_id={payload.user_id}", timeout=10)
            user_check_response.raise_for_status()
            user_exists_data = user_check_response.json()
            if not user_exists_data.get("exists"):
                logging.error(f"User '{payload.user_id}' does not exist in the database after check.")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User '{payload.user_id}' does not exist in the database."
                )
            logging.info(f"User {payload.user_id} found in DB. Internal ID: {user_exists_data.get('internal_id')}")
    except httpx.HTTPStatusError as exc:
        logging.error(f"HTTP error checking user existence for {payload.user_id}: {exc.response.status_code} - {exc.response.text}")
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Failed to check user existence: {exc.response.json().get('error', 'Server error')}"
        )
    except httpx.RequestError as exc:
        logging.error(f"Network error checking user existence for {payload.user_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not connect to user verification service. Error: {exc}"
        )
    except Exception as e:
        logging.error(f"Unexpected error checking user existence for {payload.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while checking user existence."
        )

    analysis_results = []
    for msg in payload.messages:
        result = await analyze_message(msg, payload.user_id)
        analysis_results.append(result)
    
    # Get summary and advice from DB
    summary = await get_user_summary_from_db(payload.user_id)
    advice = await get_advice_from_summary(payload.user_id)

    return {
        "analysis": analysis_results,
        "summary": summary,
        "advice": advice,
        "total_messages_processed": len(payload.messages)
    }

@app.get("/")
async def root():
    """
    Checks the status of the FastAPI backend and its connection to the PHP API.
    """
    fastapi_status = {"message": "MMTA Backend V9 - Manual Pattern Learning powered By CALBRS 36 is running."}
    php_api_status = {"status": "unknown", "message": "Could not connect to PHP API."}

    try:
        async with httpx.AsyncClient() as client:
            # Attempt to connect to the PHP API's root endpoint
            response = await client.get(PHP_API_BASE_URL, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            php_api_response_json = response.json()
            php_api_status = {
                "status": "connected",
                "message": php_api_response_json.get("message", "PHP API responded successfully but no specific message."),
                "php_api_raw_response": php_api_response_json
            }
            logging.info(f"Successfully connected to PHP API from root endpoint. Response: {php_api_status}")

    except httpx.HTTPStatusError as exc:
        php_api_status = {
            "status": "error",
            "message": f"PHP API returned HTTP error: {exc.response.status_code} - {exc.response.text}",
            "php_api_raw_response": exc.response.text
        }
        logging.error(f"HTTP error connecting to PHP API from root endpoint: {exc.response.status_code} - {exc.response.text}")
    except httpx.RequestError as exc:
        php_api_status = {
            "status": "error",
            "message": f"Network error connecting to PHP API: {exc}",
            "php_api_raw_response": None
        }
        logging.error(f"Network error connecting to PHP API from root endpoint: {exc}")
    except Exception as e:
        php_api_status = {
            "status": "error",
            "message": f"An unexpected error occurred while connecting to PHP API: {e}",
            "php_api_raw_response": None
        }
        logging.error(f"Unexpected error connecting to PHP API from root endpoint: {e}")
    
    return {
        "fastapi_status": fastapi_status,
        "php_api_connection_status": php_api_status
    }

# New endpoint to create a user in MySQL DB (optional, but useful for initial setup)
@app.post("/create_user_db")
async def create_user_in_db(user_payload: Dict[str, str]):
    user_id = user_payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required in the request body.")
    
    try:
        logging.info(f"Attempting to create user {user_id} in PHP API.")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PHP_API_BASE_URL}?action=create_user",
                json={"user_id": user_id},
                timeout=30 # Add a timeout
            )
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "success":
                logging.info(f"User {user_id} created successfully in DB. PHP API response: {result}")
                return {"message": result.get("message", f"User {user_id} created successfully."), "php_api_response": result}
            else:
                logging.error(f"Failed to create user {user_id} in DB. PHP API response: {result}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=result.get("message", "Failed to create user in database.")
                )
    except httpx.HTTPStatusError as exc:
        logging.error(f"HTTP error creating user {user_id}: {exc.response.status_code} - {exc.response.text}")
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Failed to create user in database: {exc.response.json().get('message', 'Server error')}"
        )
    except httpx.RequestError as exc:
        logging.error(f"Network error creating user {user_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not connect to user creation service. Error: {exc}"
        )
    except Exception as e:
        logging.error(f"Unexpected error creating user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while creating user."
        )
