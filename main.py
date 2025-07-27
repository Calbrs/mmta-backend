import os
import re
import json
import logging
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from Crypto.Cipher import AES

# ===== LOGGING SETUP =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== FASTAPI APP =====
app = FastAPI(title="MMTA Backend V15.0.0.0.1 - Fixed Hex Parsing")

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

# ===== CONSTANTS =====
DATA_DIR = "data"
PATTERNS_FILE = os.path.join(DATA_DIR, "patterns.json")
os.makedirs(DATA_DIR, exist_ok=True)

API_KEY = "mmta-backedn-w2J8Smes2V1V44cWRAL4FydxjY43OSW8-calbrs"
PHP_API_BASE_URL = f"https://calcue.wuaze.com/mmta_api.php?api_key={API_KEY}"

# ===== JSON HELPERS =====
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

# ===== LOAD PATTERNS =====
patterns = load_json(PATTERNS_FILE)

# ===== UTILITY FUNCTIONS =====
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

# ===== SMART CHALLENGE SOLVER =====
def sanitize_hex(hex_str: str) -> str:
    # Ondoa whitespaces, newlines, na characters zisizo za hex
    return re.sub(r'[^0-9a-fA-F]', '', hex_str)

def extract_challenge_params(html: str) -> dict:
    params = {}

    # Regex kamili kwa kila param
    key_match = re.search(r'var\s+a\s*=\s*toNumbers\(["\']([0-9a-fA-F]+)["\']\)', html)
    iv_match = re.search(r'var\s+b\s*=\s*toNumbers\(["\']([0-9a-fA-F]+)["\']\)', html)
    cipher_match = re.search(r'var\s+c\s*=\s*toNumbers\(["\']([0-9a-fA-F]+)["\']\)', html)
    redirect_match = re.search(r'location\.href\s*=\s*["\']([^"\']+)["\']', html)

    if key_match:
        params["key"] = key_match.group(1)
    if iv_match:
        params["iv"] = iv_match.group(1)
    if cipher_match:
        params["ciphertext"] = cipher_match.group(1)
    if redirect_match:
        params["redirect"] = redirect_match.group(1)

    return params

def decrypt_cookie(key_hex: str, iv_hex: str, cipher_hex: str) -> str:
    key_hex = sanitize_hex(key_hex)
    iv_hex = sanitize_hex(iv_hex)
    cipher_hex = sanitize_hex(cipher_hex)

    key = bytes.fromhex(key_hex)
    iv = bytes.fromhex(iv_hex)
    cipher = bytes.fromhex(cipher_hex)

    cipher_obj = AES.new(key, AES.MODE_CBC, iv)
    decrypted = cipher_obj.decrypt(cipher)
    return decrypted.hex()

async def solve_smart_challenge(client: httpx.AsyncClient, challenge_html: str) -> dict:
    logging.info("Attempting to solve smart challenge...")
    params = extract_challenge_params(challenge_html)
    if not all(k in params for k in ("key", "iv", "ciphertext", "redirect")):
        with open("debug_challenge.html", "w", encoding="utf-8") as f:
            f.write(challenge_html)
        raise ValueError("Failed to extract challenge params. HTML saved to debug_challenge.html")
    cookie_value = decrypt_cookie(params["key"], params["iv"], params["ciphertext"])
    return {
        "cookie_name": "__test",
        "cookie_value": cookie_value,
        "redirect_url": params["redirect"].replace("&amp;", "&")
    }

# ===== SMART REQUEST HANDLER =====
async def make_smart_request(method: str, url: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.request(method, url, **kwargs)

        # Ikiwa kuna challenge ya JS
        if "aes.js" in response.text and "slowAES" in response.text:
            logging.info("JS challenge detected. Solving...")
            solution = await solve_smart_challenge(client, response.text)
            client.cookies.set(
                name=solution["cookie_name"],
                value=solution["cookie_value"],
                domain="calcue.wuaze.com"
            )
            response = await client.request(method, solution["redirect_url"], **kwargs)
    return response

# ===== DATABASE BRIDGE =====
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
    return {"daily": dict(daily), "weekly": dict(weekly), "monthly": dict(monthly)}

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

# ===== MODELS =====
class SMSPayload(BaseModel):
    user_id: str
    messages: List[str]

# ===== CORE LOGIC =====
async def analyze_message(msg: str, user_id: str) -> Dict[str, Any]:
    service = detect_service(msg)
    parsed = parse_with_patterns(msg, service)
    if parsed:
        db_response = await save_transaction_to_db(user_id, parsed)
        return {"success": True, "data": parsed, "php_response": db_response}
    return {"success": False, "error": "Failed to parse message."}

# ===== ENDPOINTS =====
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
    try:
        response = await make_smart_request("GET", PHP_API_BASE_URL)
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
    return {"message": "MMTA Backend V15.0.0.0.1 - Fixed Hex Parsing running"}

# ===== ENTRY POINT =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
