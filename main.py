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
import httpx
# Import the AES cipher from the new library
from Crypto.Cipher import AES

# --------- LOGGING ---------
# Set level to DEBUG to get more detailed output when things go wrong
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --------- FASTAPI APP INIT ---------
app = FastAPI(title="MMTA Backend V14 - Challenge Bypass 001")

# --------- CORS SETUP (keep your existing setup) ---------
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

# --------- DIRECTORY SETUP & PHP API (keep your existing setup) ---------
DATA_DIR = "data"
PATTERNS_FILE = os.path.join(DATA_DIR, "patterns.json")
os.makedirs(DATA_DIR, exist_ok=True)
API_KEY = "mmta-backedn-w2J8Smes2V1V44cWRAL4FydxjY43OSW8-calbrs"
PHP_API_BASE_URL = f"https://calcue.wuaze.com/mmta_api.php?api_key={API_KEY}"

# ... (keep all your existing helper functions like load_json, save_json, etc.) ...
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
# --------- HELPERS ---------
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

# --------- 🤖 NEW: ANTI-BOT CHALLENGE SOLVER ---------
# --------- 🤖 REVISED: ANTI-BOT CHALLENGE SOLVER (More Robust) ---------
def solve_js_challenge(html_content: str) -> Optional[Dict[str, str]]:
    """
    Parses the JavaScript challenge HTML to extract parameters,
    decrypts them to find the cookie value, and returns the cookie and redirect URL.
    This version is more robust with better error handling and logging.
    """
    try:
        # Define patterns with optional whitespace handling (\s*) to be more flexible
        patterns = {
            "key": r'var\s+a\s*=\s*toNumbers\("([a-f0-9]+)"\)',
            "iv": r'var\s+b\s*=\s*toNumbers\("([a-f0-9]+)"\)',
            "ciphertext": r'var\s+c\s*=\s*toNumbers\("([a-f0-9]+)"\)',
            "redirect": r'location\.href\s*=\s*"([^"]+)"'
        }

        # Search for each pattern and handle failure gracefully
        key_match = re.search(patterns["key"], html_content)
        if not key_match:
            logging.error("Challenge solve failed: Could not find key pattern 'a'.")
            # The line below is crucial: it will print the HTML that caused the failure to your console.
            logging.debug(f"Problematic HTML content:\n{html_content}")
            return None
        key_hex = key_match.group(1)

        iv_match = re.search(patterns["iv"], html_content)
        if not iv_match:
            logging.error("Challenge solve failed: Could not find IV pattern 'b'.")
            logging.debug(f"Problematic HTML content:\n{html_content}")
            return None
        iv_hex = iv_match.group(1)

        ciphertext_match = re.search(patterns["ciphertext"], html_content)
        if not ciphertext_match:
            logging.error("Challenge solve failed: Could not find ciphertext pattern 'c'.")
            logging.debug(f"Problematic HTML content:\n{html_content}")
            return None
        ciphertext_hex = ciphertext_match.group(1)

        redirect_match = re.search(patterns["redirect"], html_content)
        if not redirect_match:
            logging.error("Challenge solve failed: Could not find redirect URL.")
            logging.debug(f"Problematic HTML content:\n{html_content}")
            return None
        redirect_url = redirect_match.group(1)

        # --- Decryption Logic (same as before) ---
        key = bytes.fromhex(key_hex)
        iv = bytes.fromhex(iv_hex)
        ciphertext = bytes.fromhex(ciphertext_hex)

        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_bytes = cipher.decrypt(ciphertext)
        cookie_value = decrypted_bytes.hex()
        
        logging.info("Successfully solved JS challenge.")
        return {
            "cookie_name": "__test",
            "cookie_value": cookie_value,
            "redirect_url": redirect_url.replace("&amp;", "&")
        }
    except Exception as e:
        # This will catch other errors, e.g., if hex decoding fails
        logging.error(f"An unexpected error occurred in solve_js_challenge: {e}")
        logging.debug(f"Problematic HTML content:\n{html_content}")
        return None


# --------- 🧠 UPDATED: SMART HTTP CLIENT LOGIC ---------
async def make_smart_request(
    method: str,
    url: str,
    **kwargs
) -> httpx.Response:
    """
    Makes an HTTP request that automatically handles the JS challenge.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        # Make the first attempt
        response = await client.request(method, url, **kwargs)
        
        # Check if we received the JS challenge
        if "aes.js" in response.text and "slowAES" in response.text:
            logging.info("JS challenge detected. Attempting to solve...")
            
            # Solve the challenge to get the cookie and redirect URL
            challenge_solution = solve_js_challenge(response.text)
            
            if not challenge_solution:
                raise HTTPException(status_code=500, detail="Failed to solve the anti-bot challenge.")

            # Set the required cookie
            client.cookies.set(
                name=challenge_solution["cookie_name"],
                value=challenge_solution["cookie_value"],
                domain="calcue.wuaze.com"
            )
            
            redirect_url = challenge_solution["redirect_url"]
            logging.info(f"Challenge solved. Following redirect to: {redirect_url}")

            # Make the second request with the cookie to the new URL
            # The original payload (if any) needs to be passed again
            response = await client.request(method, redirect_url, **kwargs)
            
    return response


# --------- DATABASE INTERACTION (UPDATED) ---------
async def save_transaction_to_db(user_id: str, transaction: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = {"user_id": user_id, **transaction}
        response = await make_smart_request(
            "POST",
            f"{PHP_API_BASE_URL}&action=save_transaction",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error saving transaction: {e}")
        return {"status": "error", "message": "Failed to save transaction"}

async def load_transactions_from_db(user_id: str) -> List[Dict[str, Any]]:
    try:
        url = f"{PHP_API_BASE_URL}&action=get_transactions&user_id={user_id}"
        response = await make_smart_request("GET", url)
        response.raise_for_status()
        transactions = response.json()
        return transactions if isinstance(transactions, list) else []
    except Exception as e:
        logging.error(f"Error loading transactions: {e}")
        return []

# ... (The rest of your functions like get_user_summary_from_db and get_advice_from_summary
# do not need changes as they depend on the functions we just updated)
async def get_user_summary_from_db(user_id: str) -> Dict[str, Any]:
    transactions = await load_transactions_from_db(user_id)
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
        except Exception:
            continue
    return {
        "daily": dict(daily),
        "weekly": dict(weekly),
        "monthly": dict(monthly),
    }
async def get_advice_from_summary(user_id: str) -> str:
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
    except Exception:
        return "Tatizo lilitokea katika kusoma data ya ushauri."
    return "Bado hatuna data ya kutosha ya matumizi wiki hii."
# --------- INPUT MODEL ---------
class SMSPayload(BaseModel):
    user_id: str
    messages: List[str]
# --------- CORE LOGIC ---------
async def analyze_message(msg: str, user_id: str) -> Dict[str, Any]:
    service = detect_service(msg)
    parsed = parse_with_patterns(msg, service)
    if parsed:
        db_response = await save_transaction_to_db(user_id, parsed)
        return {"success": True, "data": parsed, "php_response": db_response}
    return {"success": False, "error": "Failed to parse message."}

# --------- ENDPOINTS (UPDATED) ---------
@app.post("/analyze")
async def analyze_sms(payload: SMSPayload):
    if not payload.user_id.strip():
        raise HTTPException(status_code=400, detail="Missing or invalid 'user_id'.")

    analysis_results = []
    for msg in payload.messages:
        result = await analyze_message(msg, payload.user_id)
        analysis_results.append(result)
    
    summary = await get_user_summary_from_db(payload.user_id)
    advice = await get_advice_from_summary(payload.user_id)

    return {
        "analysis": analysis_results,
        "summary": summary,
        "advice": advice,
        "total_messages_processed": len(payload.messages)
    }

@app.get("/php-test")
async def php_test():
    """This endpoint now uses the smart request handler."""
    try:
        response = await make_smart_request("GET", PHP_API_BASE_URL)
        # Check if the final response is JSON or something else
        try:
            php_response_json = response.json()
        except json.JSONDecodeError:
            php_response_json = response.text

        return {
            "status": "success",
            "php_status_code": response.status_code,
            "php_response": php_response_json
        }
    except Exception as e:
        return {"status": "error", "message": f"PHP test failed: {e}"}

@app.get("/")
async def root():
    return {"message": "MMTA Backend V14 - Challenge Bypass"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
