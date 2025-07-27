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

# --------- LOGGING ---------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------- FASTAPI APP INIT ---------
app = FastAPI(title="MMTA Backend V12 - PHP API Test Added")

# --------- CORS SETUP ---------
origins = [
    "https://calcue.wuaze.com",
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- DIRECTORY SETUP ---------
DATA_DIR = "data"
PATTERNS_FILE = os.path.join(DATA_DIR, "patterns.json")
os.makedirs(DATA_DIR, exist_ok=True)

# --------- PHP API URL ---------
PHP_API_BASE_URL = "https://calcue.wuaze.com/mmta_api.php"

# --------- JSON UTILS ---------
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

# --------- DATABASE INTERACTION FUNCTIONS ---------
async def save_transaction_to_db(user_id: str, transaction: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = {"user_id": user_id, **transaction}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PHP_API_BASE_URL}?action=save_transaction",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        logging.error(f"HTTP error: {exc.response.status_code} - {exc.response.text}")
        return {"status": "error", "message": f"HTTP {exc.response.status_code}"}
    except httpx.RequestError as exc:
        logging.error(f"Network error: {exc}")
        return {"status": "error", "message": "Network error"}
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"status": "error", "message": "Unexpected error"}

async def load_transactions_from_db(user_id: str) -> List[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{PHP_API_BASE_URL}?action=get_transactions&user_id={user_id}",
                timeout=30
            )
            response.raise_for_status()
            transactions = response.json()
            if isinstance(transactions, list):
                return transactions
            return []
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return []
        logging.error(f"HTTP error: {exc.response.status_code} - {exc.response.text}")
        return []
    except httpx.RequestError as exc:
        logging.error(f"Network error: {exc}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
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

# --------- ENDPOINTS ---------
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
    """
    Test connection with the PHP API and return its raw response.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(PHP_API_BASE_URL, timeout=30)
            return {
                "status": "success",
                "php_status_code": response.status_code,
                "php_response": response.text
            }
    except httpx.RequestError as exc:
        logging.error(f"PHP API test failed: {exc}")
        return {"status": "error", "message": "Unable to connect to PHP API."}
    except Exception as e:
        logging.error(f"Unexpected error during PHP test: {e}")
        return {"status": "error", "message": "Unexpected error occurred."}

@app.get("/")
async def root():
    return {"message": "MMTA Backend V12 - PHP API Test Added"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
