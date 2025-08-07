import os
import re
import json
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import exceptions as firebase_exceptions

import httpx
import asyncio

from typing import List
from fastapi import Query


# ===== LOAD ENV =====
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== FIREBASE INIT =====
FIREBASE_CRED_JSON = os.getenv("FIREBASE_CRED_JSON")
if not FIREBASE_CRED_JSON:
    raise RuntimeError("Missing FIREBASE_CRED_JSON environment variable.")

try:
    cred_dict = json.loads(FIREBASE_CRED_JSON)
    cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    raise RuntimeError(f"Failed to initialize Firebase: {e}")

# ===== LOAD JSON CONFIG FILES =====
PATTERNS_PATH = "data/patterns.json"
APPENDED_JSON_PATH = "data/Appended.json"
try:
    with open("data/detect.json", "r", encoding="utf-8") as f:
        DETECT = json.load(f)
    with open("data/analyze.json", "r", encoding="utf-8") as f:
        ANALYZE = json.load(f)
    with open(PATTERNS_PATH, "r", encoding="utf-8") as f:
        PATTERNS = json.load(f)
    with open("data/core_field.json", "r", encoding="utf-8") as f:
        CORE_FIELDS = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load config files: {e}")

IGNORE_KEYWORDS = ["muda wa maongezi", "airtime", "Transaction failed"]

app = FastAPI(title="MMTA Backend with Transaction Summary")

# ===== CORS CONFIG =====
origins = [
    "https://calcue.wuaze.com",
    "https://mmta.wuaze.com",
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

# ===== MODELS =====
class SMSPayload(BaseModel):
    user_id: str
    messages: List[str]

class SummaryPayload(BaseModel):
    user_id: str

class AdminPayload(BaseModel):
    admin_id: str
    messages: List[str]

class TransactionsByIdPayload(BaseModel):
    user_id: str
    transaction_ids: List[str]

class TransactionListResponse(BaseModel):
    user_id: str
    total_transactions: int
    transactions: List[Dict[str, Any]]
    last_document_id: Optional[str] = None  # For pagination

# ===== UTILITIES =====
def validate_user(user_id: str) -> bool:
    try:
        doc = db.collection("users").document(user_id).get()
        return doc.exists
    except firebase_exceptions.FirebaseError as e:
        logging.error(f"Firebase error checking user ID: {e}")
        return False
    except Exception as e:
        logging.error(f"Error checking user ID: {e}")
        return False

def try_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        cleaned = re.sub(r"[^\d.,]", "", str(value))
        cleaned = cleaned.replace(',', '')
        return float(cleaned)
    except (ValueError, TypeError):
        return None

def sms_contains_ignore_keywords(msg: str) -> bool:
    lower = msg.lower()
    return any(kw in lower for kw in IGNORE_KEYWORDS)

def generate_transaction_id(msg: str) -> str:
    return hashlib.sha1(msg.encode('utf-8')).hexdigest()

def merge_regex(old_regex: str, new_regex: str) -> str:
    if not old_regex:
        return new_regex
    if not new_regex:
        return old_regex
    return f"(?:{old_regex})|(?:{new_regex})"

def save_merged_and_history_regex(
    service: str, 
    direction: str, 
    transaction_id: str, 
    missing_field: str, 
    old_regex: str, 
    new_regex_from_ai: str
):
    try:
        with open(PATTERNS_PATH, "r", encoding="utf-8") as f:
            patterns_data = json.load(f)
        
        if service not in patterns_data:
            patterns_data[service] = {}
        if direction not in patterns_data[service]:
            patterns_data[service][direction] = {}

        merged_regex = merge_regex(old_regex, new_regex_from_ai)
        patterns_data[service][direction][missing_field] = merged_regex
        
        with open(PATTERNS_PATH, "w", encoding="utf-8") as f:
            json.dump(patterns_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Merged regex for '{missing_field}' saved.")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error updating patterns file: {e}")
        return
    
    try:
        try:
            with open(APPENDED_JSON_PATH, "r", encoding="utf-8") as f:
                appended_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            appended_data = {"processed_ids": [], "patterns": {}}
        
        if service not in appended_data["patterns"]:
            appended_data["patterns"][service] = {}
        if direction not in appended_data["patterns"][service]:
            appended_data["patterns"][service][direction] = {}

        history_entry = {
            "old_regex": old_regex,
            "new_regex_from_ai": new_regex_from_ai,
            "timestamp": datetime.utcnow().isoformat()
        }
        if missing_field not in appended_data["patterns"][service][direction]:
            appended_data["patterns"][service][direction][missing_field] = []
        appended_data["patterns"][service][direction][missing_field].append(history_entry)

        if transaction_id not in appended_data["processed_ids"]:
            appended_data["processed_ids"].append(transaction_id)
        
        with open(APPENDED_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(appended_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved regex history for ID '{transaction_id}'.")
    except Exception as e:
        logging.error(f"Error saving history: {e}")

def is_transaction_processed(transaction_id: str) -> bool:
    try:
        with open(APPENDED_JSON_PATH, "r", encoding="utf-8") as f:
            appended_data = json.load(f)
            return transaction_id in appended_data.get("processed_ids", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return False

def check_core_fields(direction: str, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    missing = []
    required_fields = CORE_FIELDS.get(direction, {}).get("required", [])
    for field in required_fields:
        value = parsed_data.get(field)
        if value is None or value == "":
            missing.append(field)
    return {
        "missing": missing,
        "all_present": len(missing) == 0
    }

def detect_service(msg: str) -> str:
    lower = msg.lower()
    for service, keywords in DETECT.get("services", {}).items():
        for word in keywords:
            if word in lower:
                return service
    return "Unknown"

def classify_direction(msg: str) -> str:
    lower = msg.lower()
    for word in DETECT.get("classify", {}).get("incoming", []):
        if word in lower:
            return "incoming"
    for word in DETECT.get("classify", {}).get("outgoing", []):
        if word in lower:
            return "outgoing"
    return "unknown"

def parse_message(msg: str) -> Dict[str, Any]:
    try:
        with open(PATTERNS_PATH, "r", encoding="utf-8") as f:
            current_patterns = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        current_patterns = PATTERNS
        logging.error("Using in-memory patterns due to file error")

    service = detect_service(msg)
    direction = classify_direction(msg)
    transaction_id = generate_transaction_id(msg)

    parsed_data = {
        "service": service,
        "direction": direction,
        "transaction_id": transaction_id,
        "raw": msg,
        "timestamp": datetime.utcnow().isoformat()
    }

    if service in current_patterns and direction in current_patterns[service]:
        patterns = current_patterns[service][direction]
        for field, regex in patterns.items():
            if isinstance(regex, dict):
                parsed_data[field] = {}
                for subfield, subregex in regex.items():
                    match = re.search(subregex, msg, re.I)
                    if match:
                        val = match.group(1).strip()
                        if subfield in ["amount", "balance", "total", "service_charge", "gov_levy", "net_amount"]:
                            val = try_float(val)
                        parsed_data[field][subfield] = val
            else:
                match = re.search(regex, msg, re.I)
                if match:
                    val = match.group(1).strip()
                    if field in ["amount", "balance", "total", "service_charge", "gov_levy", "net_amount"]:
                        val = try_float(val)
                    parsed_data[field] = val

    return parsed_data

# ===== GEMINI AI INTEGRATION =====
async def query_gemini_flash_regex(message: str, missing_fields: List[str]) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        logging.error("Missing GEMINI_API_KEY")
        return {field: "error: API key missing" for field in missing_fields}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    results = {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        async def fetch(field: str):
            prompt = (
                f"From the text below, generate a Python regex pattern to extract '{field}'.\n"
                f"Requirements:\n"
                f"1. Include exactly one capturing group for the value\n"
                f"2. Escape special characters for JSON (double backslashes)\n"
                f"3. Return ONLY the raw regex string\n\n"
                f"Text:\n---\n{message}\n---"
            )
            payload = {"contents": [{"parts": [{"text": prompt}]}]}

            try:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                
                if data.get("candidates") and data["candidates"][0].get("content", {}).get("parts"):
                    text_response = data["candidates"][0]["content"]["parts"][0].get("text", "").strip()
                    cleaned_regex = re.sub(r"^`{1,3}(python|regex)?\s*|`{1,3}$", "", text_response).strip()
                    results[field] = cleaned_regex
                else:
                    results[field] = "error: No valid content"
            except httpx.RequestError as e:
                results[field] = f"error: HTTP request failed - {e}"
            except Exception as e:
                results[field] = f"error: {str(e)}"

        await asyncio.gather(*[fetch(field) for field in missing_fields])

    return results

def save_transaction(user_id: str, transaction: Dict[str, Any]) -> dict:
    try:
        doc_ref = db.collection("users").document(user_id).collection("transactions").add(transaction)
        logging.info(f"Saved transaction for {user_id}")
        return {"status": "success", "firebase_ref": str(doc_ref)}
    except firebase_exceptions.FirebaseError as e:
        logging.error(f"Firebase error: {e}")
        return {"status": "error", "message": f"Firebase error: {e}"}
    except Exception as e:
        logging.error(f"General error: {e}")
        return {"status": "error", "message": str(e)}

def summarize_transactions(user_id: str) -> dict:
    ref = db.collection("users").document(user_id).collection("transactions")
    docs = ref.stream()

    summary = {
        "incoming": {"count": 0, "total": 0.0},
        "outgoing": {"count": 0, "total": 0.0},
        "unknown": {"count": 0, "total": 0.0}
    }

    for doc in docs:
        data = doc.to_dict()
        category = data.get("direction", "unknown")
        amt = data.get("amount") or 0.0
        
        if category in summary:
            summary[category]["count"] += 1
            summary[category]["total"] += amt
        else:
            summary["unknown"]["count"] += 1
            summary["unknown"]["total"] += amt

    return summary



@app.get("/transactions", response_model=TransactionListResponse)
async def get_transactions_by_user(
    user_id: str = Query(..., description="The user ID to fetch transactions for"),
    limit: int = Query(10, description="Number of transactions to return (1-100)", ge=1, le=100),
    last_document_id: Optional[str] = Query(None, description="Last document ID for pagination")
):
    """
    Get paginated list of transactions for a specific user.
    
    Parameters:
    - user_id: Required user ID
    - limit: Number of transactions to return (default: 10, max: 100)
    - last_document_id: Optional last document ID for pagination
    
    Returns:
    - List of transactions with pagination support
    """
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="Missing user_id")
    if not validate_user(user_id):
        raise HTTPException(status_code=404, detail="Account not found")

    try:
        transactions_ref = db.collection("users").document(user_id).collection("transactions")
        
        # Start the query
        query = transactions_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)
        
        # Apply pagination if last_document_id is provided
        if last_document_id:
            last_doc = transactions_ref.document(last_document_id).get()
            if last_doc.exists:
                query = query.start_after(last_doc)
        
        # Execute the query
        docs = query.stream()
        
        transactions = []
        last_doc = None
        for doc in docs:
            transactions.append({
                "id": doc.id,
                "data": doc.to_dict()
            })
            last_doc = doc
        
        return {
            "user_id": user_id,
            "total_transactions": len(transactions),
            "transactions": transactions,
            "last_document_id": last_doc.id if last_doc else None
        }
        
    except Exception as e:
        logging.error(f"Error fetching transactions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch transactions")

# ===== ENDPOINTS =====
@app.post("/analyze")
async def analyze_sms(payload: SMSPayload):
    if not payload.user_id.strip():
        raise HTTPException(status_code=400, detail="Missing user_id")
    if not validate_user(payload.user_id):
        raise HTTPException(status_code=404, detail="Account not found")

    results = []
    for msg in payload.messages:
        if sms_contains_ignore_keywords(msg):
            results.append({
                "status": "ignored",
                "reason": "Contains ignore keywords"
            })
            continue

        parsed = parse_message(msg)
        core_check = check_core_fields(parsed.get("direction", "unknown"), parsed)
        save_result = {"status": "skipped", "reason": "Missing core fields"}
        
        if core_check["all_present"]:
            save_result = save_transaction(payload.user_id, parsed)
        
        results.append({
            "parsed": parsed,
            "core_fields_status": core_check,
            "save_result": save_result
        })

    return {
        "user_id": payload.user_id,
        "total_messages": len(payload.messages),
        "results": results
    }

@app.post("/analyze-summary")
async def analyze_summary(payload: SummaryPayload):
    if not payload.user_id.strip():
        raise HTTPException(status_code=400, detail="Missing user_id")
    if not validate_user(payload.user_id):
        raise HTTPException(status_code=404, detail="Account not found")

    summary = summarize_transactions(payload.user_id)
    return {
        "user_id": payload.user_id,
        "summary": summary
    }

@app.post("/admin-analyze")
async def admin_analyze(payload: AdminPayload):
    if payload.admin_id != "Calbrs-36":
        raise HTTPException(status_code=403, detail="Unauthorized")

    results = []
    for msg in payload.messages:
        if sms_contains_ignore_keywords(msg):
            results.append({"status": "ignored"})
            continue

        parsed = parse_message(msg)
        transaction_id = parsed.get("transaction_id")
        direction = parsed.get("direction", "unknown")
        service = parsed.get("service", "unknown")
        
        if is_transaction_processed(transaction_id):
            results.append({
                "status": "skipped",
                "reason": "Already processed",
                "transaction_id": transaction_id
            })
            continue

        core_check = check_core_fields(direction, parsed)
        gemini_response = None
        re_parsed = None
        save_result = {"status": "not_saved"}

        if not core_check["all_present"]:
            missing_fields = core_check["missing"]
            gemini_response = await query_gemini_flash_regex(msg, missing_fields)
            
            successful_patterns = {k: v for k, v in gemini_response.items() if v and not v.startswith("error:")}
            
            if successful_patterns:
                current_patterns = PATTERNS.get(service, {}).get(direction, {})
                for field, new_regex in successful_patterns.items():
                    old_regex = current_patterns.get(field, "")
                    save_merged_and_history_regex(
                        service, direction, transaction_id, 
                        field, old_regex, new_regex
                    )

                re_parsed = parse_message(msg)
                re_core_check = check_core_fields(direction, re_parsed)
                
                if re_core_check["all_present"]:
                    save_result = save_transaction(payload.admin_id, re_parsed)
        else:
            save_result = save_transaction(payload.admin_id, parsed)

        results.append({
            "parsed": parsed,
            "core_fields_status": core_check,
            "gemini_response": gemini_response,
            "re_parsed": re_parsed,
            "save_result": save_result
        })

    return {
        "admin_id": payload.admin_id,
        "total_messages": len(payload.messages),
        "results": results
    }

@app.post("/transactions-by-ids")
async def get_transactions_by_ids(payload: TransactionsByIdPayload):
    if not payload.user_id.strip():
        raise HTTPException(status_code=400, detail="Missing user_id")
    if not validate_user(payload.user_id):
        raise HTTPException(status_code=404, detail="Account not found")
    if not payload.transaction_ids:
        raise HTTPException(status_code=400, detail="No transaction_ids provided")

    transactions = []
    for doc_id in payload.transaction_ids:
        try:
            doc_ref = db.collection("users").document(payload.user_id).collection("transactions").document(doc_id)
            doc = doc_ref.get()
            if doc.exists:
                transactions.append({
                    "id": doc_id,
                    "data": doc.to_dict()
                })
            else:
                logging.warning(f"Transaction ID not found: {doc_id}")
        except Exception as e:
            logging.error(f"Error retrieving transaction {doc_id}: {e}")

    return {
        "user_id": payload.user_id,
        "found": len(transactions),
        "transactions": transactions
    }

@app.get("/")
async def health_check(request: Request):
    user_id = request.query_params.get("user_id")
    if not user_id or not user_id.strip():
        return {"status": "ok", "message": "MMTA 1.0.0", "user": "none"}

    if validate_user(user_id):
        return {"status": "ok", "message": "MMTA 1.0.0", "user": "exists"}
    else:
        return {"status": "error", "message": "Account required", "user": "not_found"}

# ===== INITIALIZATION =====
if __name__ == "__main__":
    import uvicorn
    # Create data directory if missing
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Initialize required JSON files
    required_files = {
        'detect.json': {},
        'analyze.json': {},
        'patterns.json': {},
        'core_field.json': {
            "incoming": {
                "required": ["amount", "balance", "reference"]
            },
            "outgoing": {
                "required": ["amount", "balance", "recipient"]
            }
        },
        'Appended.json': {"processed_ids": [], "patterns": {}}
    }
    
    for filename, default_content in required_files.items():
        path = f'data/{filename}'
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(default_content, f, indent=2)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
