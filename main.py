import json
import os
import re
import logging
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # Import Field for better validation

# --------- LOGGING ---------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="MMTA Backend V9 - Manual Pattern Learning")

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

# --------- PATHS (Still relevant for patterns.json, but not user data) ---------
DATA_DIR = "data"
PATTERNS_FILE = os.path.join(DATA_DIR, "patterns.json")

os.makedirs(DATA_DIR, exist_ok=True)

# --------- PHP API URL ---------
PHP_API_URL = "https://calcue.wuaze.com/mmta_api.php"

# --------- LOAD & SAVE UTILITIES (for patterns.json only) ---------
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

# Load patterns from patterns.json
patterns = load_json(PATTERNS_FILE)

# --------- UTILS ---------
def try_float(value: Optional[str]) -> Optional[float]:
    """Attempts to convert a string to a float, cleaning non-numeric characters first."""
    try:
        # Remove any characters that are not digits or a decimal point
        return float(re.sub(r'[^\d.]', '', value)) if value else None
    except ValueError:
        return None

def detect_service(msg: str) -> str:
    """
    Detects the service (e.g., MPESA, AirtelMoney, TIGO) based on the message content.
    Prioritizes specific MPESA confirmation pattern and adds TIGO regex.
    """
    lower = msg.lower()

    # --------- MPESA detection ---------
    # Specific check for MPESA confirmation message format:
    mpesa_specific_pattern = r"^[A-Z0-9]+\s+\(?(?:imethibitishwa|Confirmed)\)?\s*"
    if re.search(mpesa_specific_pattern, msg, re.IGNORECASE):
        return "MPESA"

    # General keyword-based MPESA detection
    if "mpesa" in lower:
        return "MPESA"

    # --------- AIRTEL detection ---------
    if "airtel" in lower:
        return "AirtelMoney"

    # --------- TIGO detection ---------
    # Specific pattern for TIGO messages with TID and "Received Tsh..."
    tigo_pattern = r"TID[:\s]*([A-Z0-9.]+)"
    if re.search(tigo_pattern, msg, re.IGNORECASE) or "tigo" in lower:
        return "TIGO"

    return "Unknown"

def parse_with_patterns(msg: str, service: str) -> Optional[Dict[str, Any]]:
    """
    Parses the message by trying to extract all fields via the patterns dict for a specific service.
    Filters out null, empty, or 'Unknown' values to return only meaningful data.
    """
    service_patterns = patterns.get(service)
    if not service_patterns:
        logging.warning(f"No patterns found for service: {service}")
        return None

    extracted_data = {}
    try:
        # Helper to extract and clean values from a regex match
        def extract_and_clean(regex_pattern, message, default_value=""):
            if not regex_pattern: # Handle cases where a specific pattern might be missing for a service
                return None
            match = re.search(regex_pattern, message, re.IGNORECASE)
            value = match.group(1).strip() if match and match.group(1) else default_value
            # Filter out empty strings, "Unknown", or default empty values
            return value if value and value.lower() != "unknown" and value != default_value else None

        # Extract simple fields
        extracted_data["transaction_id"] = extract_and_clean(service_patterns.get("transaction_id"), msg)
        extracted_data["date_time"] = extract_and_clean(service_patterns.get("date_time"), msg)
        extracted_data["transaction_type"] = extract_and_clean(service_patterns.get("transaction_type"), msg)

        amount_str = extract_and_clean(service_patterns.get("amount"), msg)
        extracted_data["amount"] = try_float(amount_str) if amount_str else None

        extracted_data["currency"] = extract_and_clean(service_patterns.get("currency"), msg)

        net_amount_str = extract_and_clean(service_patterns.get("net_amount"), msg)
        extracted_data["net_amount"] = try_float(net_amount_str) if net_amount_str else None

        previous_balance_str = extract_and_clean(service_patterns.get("previous_balance"), msg)
        extracted_data["previous_balance"] = try_float(previous_balance_str) if previous_balance_str else None

        balance_after_str = extract_and_clean(service_patterns.get("balance_after"), msg)
        extracted_data["balance_after"] = try_float(balance_after_str) if balance_after_str else None

        extracted_data["service_reference"] = extract_and_clean(service_patterns.get("service_reference"), msg)
        extracted_data["service_provider"] = extract_and_clean(service_patterns.get("service_provider"), msg)
        extracted_data["channel"] = extract_and_clean(service_patterns.get("channel"), msg)
        extracted_data["payment_method"] = extract_and_clean(service_patterns.get("payment_method"), msg)
        extracted_data["location"] = extract_and_clean(service_patterns.get("location"), msg)
        extracted_data["status"] = extract_and_clean(service_patterns.get("status"), msg)
        extracted_data["transaction_relation"] = extract_and_clean(service_patterns.get("transaction_relation"), msg)
        extracted_data["user_reference_note"] = extract_and_clean(service_patterns.get("user_reference_note"), msg)

        promotions_bonus_str = extract_and_clean(service_patterns.get("promotions_bonus"), msg)
        extracted_data["promotions_bonus"] = try_float(promotions_bonus_str) if promotions_bonus_str else None

        extracted_data["auth_method"] = extract_and_clean(service_patterns.get("auth_method"), msg)
        extracted_data["phone_number"] = extract_and_clean(service_patterns.get("phone_number"), msg)
        extracted_data["participant"] = extract_and_clean(service_patterns.get("participant"), msg)

        # Extract charges dict
        charges_data = {}
        charges_patterns = service_patterns.get("charges", {})
        charges_total_str = extract_and_clean(charges_patterns.get("total"), msg)
        charges_data["total"] = try_float(charges_total_str) if charges_total_str else None

        charges_service_charge_str = extract_and_clean(charges_patterns.get("service_charge"), msg)
        charges_data["service_charge"] = try_float(charges_service_charge_str) if charges_service_charge_str else None

        charges_gov_levy_str = extract_and_clean(charges_patterns.get("gov_levy"), msg)
        charges_data["gov_levy"] = try_float(charges_gov_levy_str) if charges_gov_levy_str else None

        if any(v is not None for v in charges_data.values()):
            extracted_data["charges"] = charges_data
        else:
            extracted_data["charges"] = {}

        # Extract recipient dict
        recipient_data = {}
        recipient_patterns = service_patterns.get("recipient", {})
        recipient_data["name"] = extract_and_clean(recipient_patterns.get("name"), msg)
        recipient_data["phone_number"] = extract_and_clean(recipient_patterns.get("phone_number"), msg)
        recipient_data["agent_id"] = extract_and_clean(recipient_patterns.get("agent_id"), msg)

        if any(v is not None for v in recipient_data.values()):
            extracted_data["recipient"] = recipient_data
        else:
            extracted_data["recipient"] = {}

        extracted_data["service"] = service
        extracted_data["raw"] = msg

        # Final filtering: remove keys with None or empty string values
        def clean_dict(d):
            return {k: v for k, v in d.items() if v is not None and v != "" and (not isinstance(v, dict) or clean_dict(v))}

        cleaned_result = clean_dict(extracted_data)

        return cleaned_result if cleaned_result else None

    except re.error as e:
        logging.error(f"Regex error in parse_with_patterns for service {service}: {e}")
        return None

# --- NEW: PHP API Communication Functions ---
def _make_api_request(action: str, method: str = "GET", data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Any:
    """Helper function to make requests to the PHP API."""
    url = f"{PHP_API_URL}?action={action}"
    try:
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url, params=params)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error communicating with PHP API ({action}): {e}")
        raise HTTPException(status_code=500, detail=f"Backend communication error: {e}")
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from PHP API ({action}): {response.text}")
        raise HTTPException(status_code=500, detail="Invalid JSON response from backend.")

def _save_transaction_to_db(user_id: str, transaction: Dict[str, Any]):
    """Saves a parsed transaction to the database via PHP API."""
    # The PHP API expects certain fields. Let's make sure they are present.
    payload = {
        "user_id": user_id, # Ensure user_id is always present
        "transaction_id": transaction.get("transaction_id"),
        "date_time": transaction.get("date_time"),
        "transaction_type": transaction.get("transaction_type"),
        "amount": transaction.get("amount"),
        "service": transaction.get("service"),
        "raw": transaction.get("raw")
    }
    # Clean payload to remove None values, or explicitly set to empty string for PHP if needed
    cleaned_payload = {k: (v if v is not None else "") for k, v in payload.items()}

    # Ensure amount is treated as a string for PHP API (if float is not directly handled well)
    if isinstance(cleaned_payload['amount'], float):
        cleaned_payload['amount'] = str(cleaned_payload['amount'])

    logging.info(f"Saving transaction to DB: {cleaned_payload}")
    return _make_api_request("save_transaction", method="POST", data=cleaned_payload)

def _load_transactions_from_db(user_id: str) -> List[Dict[str, Any]]:
    """Loads all transactions for a specific user from the database via PHP API."""
    response = _make_api_request("get_transactions", params={"user_id": user_id})
    if isinstance(response, list):
        # Convert numeric strings back to floats for calculations in Python
        for t in response:
            if 'amount' in t and t['amount'] is not None:
                try:
                    t['amount'] = float(t['amount'])
                except (ValueError, TypeError):
                    t['amount'] = None
        return response
    logging.error(f"Unexpected response format for get_transactions: {response}")
    return []

def _get_summary_from_db(user_id: str) -> Dict[str, Any]:
    """
    Calculates and returns daily, weekly, and monthly spending summaries for a user
    by fetching all transactions from the database.
    """
    transactions = _load_transactions_from_db(user_id)
    daily = defaultdict(float)
    weekly = defaultdict(float)
    monthly = defaultdict(float)

    for t in transactions:
        if not t.get("amount") or not t.get("date_time"):
            continue
        try:
            amount = float(t["amount"])
            dt = datetime.strptime(str(t["date_time"]), "%Y-%m-%d %H:%M:%S")
            daily[dt.strftime("%Y-%m-%d")] += amount
            weekly[f"week_{dt.strftime('%U_%Y')}"] += amount
            monthly[dt.strftime("%B_%Y")] += amount
        except (ValueError, TypeError, KeyError) as e:
            logging.warning(f"Could not process transaction for summary due to invalid amount or date_time: {t}, Error: {e}")
            continue

    summary = {"daily": dict(daily), "weekly": dict(weekly), "monthly": dict(monthly)}
    return summary

def get_manual_advice(user_id: str) -> str:
    """Provides simple, non-AI financial advice based on user's spending summary from DB."""
    summary = _get_summary_from_db(user_id) # Get summary by fetching from DB

    if summary.get("weekly"):
        try:
            latest_week_key = sorted(summary["weekly"].keys(), key=lambda x: (int(x.split('_')[2]), int(x.split('_')[1])), reverse=True)
            if not latest_week_key:
                return "Bado hatuna data ya kutosha ya matumizi wiki hii."

            latest_week_spending = summary["weekly"][latest_week_key[0]]
        except (IndexError, ValueError) as e:
            logging.error(f"Error getting latest week spending: {e}")
            return "Bado hatuna data ya kutosha ya matumizi wiki hii."

        if latest_week_spending > 100000:
            return f"Wiki hii umetumia {latest_week_spending:,.2f} TZS. Jaribu kupunguza matumizi."
        elif latest_week_spending > 0:
            return "Matumizi yako wiki hii yako sawa. Endelea kudhibiti gharama zako."

    return "Bado hatuna data ya kutosha ya matumizi wiki hii."


# --------- MAIN ANALYSIS ---------
class SMSPayload(BaseModel):
    user_id: str = Field(..., min_length=1, description="Unique identifier for the user.")
    messages: List[str]

def analyze_message(msg: str, user_id: str) -> Dict[str, Any]:
    """
    Analyzes a single SMS message, detects service, parses it using stored patterns,
    and saves the transaction to the database if successful.
    """
    service = detect_service(msg)
    parsed = parse_with_patterns(msg, service)

    if parsed:
        try:
            _save_transaction_to_db(user_id, parsed)
            return {"success": True, "source": "database", "data": parsed}
        except HTTPException as e:
            logging.error(f"Failed to save transaction to DB for user {user_id}: {e.detail}")
            return {"success": False, "error": f"Failed to save transaction: {e.detail}"}
    else:
        logging.info(f"Failed to parse message: '{msg}' with service '{service}'.")
        return {"success": False, "error": "Failed to parse message using available patterns."}

# --------- API ROUTES ---------
@app.get("/")
def root():
    """Root endpoint for the MMTA Backend."""
    return {"message": "MMTA Backend V9 - Database Integration + User Summary (User ID Required)"}

@app.post("/analyze")
def analyze_sms(payload: SMSPayload):
    """
    Analyzes a list of SMS messages for a given user, updates their summary (calculated on-the-fly),
    and provides financial advice. Requires user_id in the payload.
    """
    # user_id is now automatically validated by Pydantic's SMSPayload
    user_id = payload.user_id

    # First, check if the user exists in the database
    try:
        user_check_response = _make_api_request("check_user", params={"user_id": user_id})
        if not user_check_response.get("exists"):
            raise HTTPException(status_code=404, detail=f"User with ID '{user_id}' does not exist.")
    except HTTPException as e:
        raise e # Re-raise if there was an error communicating with the PHP API
    except Exception as e:
        logging.error(f"An unexpected error occurred during user check: {e}")
        raise HTTPException(status_code=500, detail="Failed to verify user existence.")


    results = [analyze_message(msg, user_id) for msg in payload.messages]

    # After processing messages, calculate and return the updated summary
    summary = _get_summary_from_db(user_id)

    return {
        "analysis": results,
        "summary": summary,
        "advice": get_manual_advice(user_id),
        "total_messages_processed": len(payload.messages)
    }

@app.get("/patterns")
def get_all_patterns():
    """Returns all loaded patterns."""
    return {"patterns": patterns}

@app.get("/service-names")
def get_service_names():
    """Returns a list of all detected service names from patterns.json."""
    data = load_json(PATTERNS_FILE)
    return {"service_names": list(data.keys())}

@app.get("/service-names/{service_name}")
def get_service_patterns(service_name: str):
    """Returns patterns for a specific service name."""
    data = load_json(PATTERNS_FILE)
    if service_name not in data:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found.")
    return {service_name: data[service_name]}

@app.get("/user-summary/{user_id}")
def get_user_summary(user_id: str = Path(..., min_length=1, description="Unique identifier of the user")):
    """Returns the spending summary for a specific user from the database."""
    # First, check if the user exists in the database
    try:
        user_check_response = _make_api_request("check_user", params={"user_id": user_id})
        if not user_check_response.get("exists"):
            raise HTTPException(status_code=404, detail=f"User with ID '{user_id}' does not exist.")
    except HTTPException as e:
        raise e

    summary = _get_summary_from_db(user_id)
    if not summary.get("daily") and not summary.get("weekly") and not summary.get("monthly"):
        raise HTTPException(status_code=404, detail="User summary not found or empty.")
    return {"summary": summary}

@app.get("/user-transactions/{user_id}")
def get_user_transactions(user_id: str = Path(..., min_length=1, description="Unique identifier of the user")):
    """Returns all stored transactions for a specific user from the database."""
    # First, check if the user exists in the database
    try:
        user_check_response = _make_api_request("check_user", params={"user_id": user_id})
        if not user_check_response.get("exists"):
            raise HTTPException(status_code=404, detail=f"User with ID '{user_id}' does not exist.")
    except HTTPException as e:
        raise e

    transactions = _load_transactions_from_db(user_id)
    if not transactions:
        raise HTTPException(status_code=404, detail="No transactions found for this user.")
    return {"transactions": transactions}

@app.post("/user-transactions/{user_id}")
def save_user_transaction(user_id: str = Path(..., min_length=1, description="Unique identifier of the user"), transaction: Dict[str, Any] = Field(..., description="Transaction data to save")):
    """Manually saves a transaction for a specific user to the database."""
    # First, check if the user exists in the database
    try:
        user_check_response = _make_api_request("check_user", params={"user_id": user_id})
        if not user_check_response.get("exists"):
            raise HTTPException(status_code=404, detail=f"User with ID '{user_id}' does not exist.")
    except HTTPException as e:
        raise e

    if not transaction:
        raise HTTPException(status_code=400, detail="Transaction data is required.")
    try:
        _save_transaction_to_db(user_id, transaction)
        return {"status": "success", "message": "Transaction saved successfully to database."}
    except HTTPException as e:
        raise e

@app.delete("/user-transactions/{user_id}")
def delete_user_transactions(user_id: str = Path(..., min_length=1, description="Unique identifier of the user")):
    """Deletes all transactions for a specific user from the database via PHP API."""
    # First, check if the user exists in the database
    try:
        user_check_response = _make_api_request("check_user", params={"user_id": user_id})
        if not user_check_response.get("exists"):
            raise HTTPException(status_code=404, detail=f"User with ID '{user_id}' does not exist.")
    except HTTPException as e:
        raise e

    try:
        response = _make_api_request("delete_user", method="DELETE", params={"user_id": user_id})
        return response
    except HTTPException as e:
        raise e

@app.get("/user-advice/{user_id}")
def get_user_advice_api(user_id: str = Path(..., min_length=1, description="Unique identifier of the user")):
    """Returns financial advice for a specific user."""
    # First, check if the user exists in the database
    try:
        user_check_response = _make_api_request("check_user", params={"user_id": user_id})
        if not user_check_response.get("exists"):
            raise HTTPException(status_code=404, detail=f"User with ID '{user_id}' does not exist.")
    except HTTPException as e:
        raise e

    advice = get_manual_advice(user_id)
    return {"advice": advice}

@app.get("/check-user/{user_id}")
def check_user_exists(user_id: str = Path(..., min_length=1, description="Unique identifier of the user")):
    """Checks if a user exists in the database via PHP API."""
    try:
        # This endpoint directly calls the PHP API to check user existence,
        # so no pre-check needed here.
        response = _make_api_request("check_user", params={"user_id": user_id})
        return response
    except HTTPException as e:
        raise e

@app.post("/create-user/{user_id}")
def create_user(user_id: str = Path(..., min_length=1, description="Unique identifier for the new user")):
    """Creates a new user in the database via PHP API."""
    try:
        response = _make_api_request("create_user", method="POST", data={"user_id": user_id})
        return response
    except HTTPException as e:
        raise e
