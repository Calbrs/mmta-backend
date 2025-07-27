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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="MMTA Backend - Enhanced API Integration Mmta")

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

DATA_DIR = "data"
PATTERNS_FILE = os.path.join(DATA_DIR, "patterns.json")
os.makedirs(DATA_DIR, exist_ok=True)

PHP_API_BASE_URL = "https://calcue.wuaze.com/mmta_api.php"

def load_json(file_path: str) -> Dict[str, Any]:
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"JSON load error: {e}")
    return {}

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

        extracted.update({
            "transaction_id": extract(service_patterns.get("transaction_id"), msg),
            "date_time": extract(service_patterns.get("date_time"), msg),
            "transaction_type": extract(service_patterns.get("transaction_type"), msg),
            "amount": try_float(extract(service_patterns.get("amount"), msg)),
            "currency": extract(service_patterns.get("currency"), msg),
            "net_amount": try_float(extract(service_patterns.get("net_amount"), msg)),
            "service": service,
            "raw": msg
        })

        return {k: v for k, v in extracted.items() if v not in [None, "", {}]}
    except re.error as e:
        logging.error(f"Regex error: {e}")
        return None

async def verify_user(user_id: str) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{PHP_API_BASE_URL}?action=check_user&user_id={user_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("exists", False)
    except Exception as e:
        logging.error(f"User verification failed: {e}")
        return False

async def save_transaction_to_db(user_id: str, transaction: Dict[str, Any]) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PHP_API_BASE_URL}?action=save_transaction",
                json={"user_id": user_id, **transaction},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logging.error(f"Transaction save failed: {e}")
        return {"status": "error", "message": str(e)}

async def load_transactions_from_db(user_id: str) -> List[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{PHP_API_BASE_URL}?action=get_transactions&user_id={user_id}",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
    except Exception as e:
        logging.error(f"Transaction load failed: {e}")
        return []

async def get_user_summary(user_id: str) -> Dict[str, Any]:
    transactions = await load_transactions_from_db(user_id)
    
    daily = defaultdict(float)
    weekly = defaultdict(float)
    monthly = defaultdict(float)

    for t in transactions:
        try:
            amount = float(t.get("amount", 0))
            dt = datetime.strptime(t["date_time"], "%Y-%m-%d %H:%M:%S")
            daily[dt.strftime("%Y-%m-%d")] += amount
            weekly[f"week_{dt.strftime('%U_%Y')}"] += amount
            monthly[dt.strftime("%B_%Y")] += amount
        except Exception:
            continue

    return {
        "daily": dict(daily),
        "weekly": dict(weekly),
        "monthly": dict(monthly),
        "total_transactions": len(transactions)
    }

def generate_advice(summary: Dict[str, Any]) -> str:
    if not summary.get("weekly"):
        return "Insufficient data for analysis"

    try:
        latest_week = next(iter(sorted(summary["weekly"].items(), reverse=True)))
        amount = latest_week[1]
        if amount > 100000:
            return f"High spending alert: {amount:,.2f} TZS this week"
        return "Your spending patterns look normal"
    except Exception:
        return "Analysis unavailable"

class SMSPayload(BaseModel):
    user_id: str
    messages: List[str]

@app.post("/analyze")
async def analyze_sms(payload: SMSPayload):
    if not payload.user_id.strip():
        raise HTTPException(status_code=400, detail="User ID required")

    if not await verify_user(payload.user_id):
        raise HTTPException(status_code=404, detail="User not found")

    analysis = []
    for msg in payload.messages:
        service = detect_service(msg)
        parsed = parse_with_patterns(msg, service)
        if parsed:
            db_response = await save_transaction_to_db(payload.user_id, parsed)
            analysis.append({
                "status": "success",
                "service": service,
                "data": parsed,
                "db_response": db_response
            })
        else:
            analysis.append({
                "status": "failed",
                "message": "Unrecognized transaction format",
                "raw": msg
            })

    summary = await get_user_summary(payload.user_id)
    advice = generate_advice(summary)

    return {
        "user_id": payload.user_id,
        "analysis": analysis,
        "summary": summary,
        "advice": advice,
        "processed": len(payload.messages),
        "successful": sum(1 for a in analysis if a["status"] == "success")
    }

@app.get("/transactions/{user_id}")
async def get_transactions(user_id: str):
    transactions = await load_transactions_from_db(user_id)
    summary = await get_user_summary(user_id)
    return {
        "user_id": user_id,
        "count": len(transactions),
        "transactions": transactions,
        "summary": summary
    }

@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(PHP_API_BASE_URL, timeout=5)
            php_status = response.status_code == 200
        return {
            "fastapi": "healthy",
            "php_api": "healthy" if php_status else "unavailable",
            "patterns_loaded": bool(patterns)
        }
    except Exception:
        return {
            "fastapi": "healthy",
            "php_api": "unreachable",
            "patterns_loaded": bool(patterns)
        }

if __name__ == "__main__":
    patterns = load_json(PATTERNS_FILE)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
