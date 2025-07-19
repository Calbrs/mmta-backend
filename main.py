from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json, os, re, requests
from datetime import datetime
from collections import defaultdict

app = FastAPI(title="MMTA Backend V7")

# -------------------- GLOBAL PATHS --------------------
DATA_DIR = "data"
USER_DATA_DIR = os.path.join(DATA_DIR, "users")
PATTERNS_FILE = os.path.join(DATA_DIR, "patterns.json")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(USER_DATA_DIR, exist_ok=True)

# -------------------- LOAD / SAVE PATTERNS --------------------
def load_patterns():
    if os.path.exists(PATTERNS_FILE):
        with open(PATTERNS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_patterns():
    with open(PATTERNS_FILE, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=2)

patterns = load_patterns()

# -------------------- GEMINI API --------------------
GEMINI_API_KEY = "AIzaSyAH-0xDNapOWOrCZH1OczyXoOniRhz6jJA"

def ask_ai_for_regex(msg: str) -> Optional[Dict[str, Any]]:
    """
    Iwapo hakuna pattern inayo-match, tumia Gemini AI
    kujaribu kuunda regex mpya.
    """
    prompt = (
        f"Analyze this SMS: {msg}.\n"
        "Return regex patterns for key parts like type and amount in valid JSON.\n"
        "Example: {\"payment\": \"regex_here\"}"
    )
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        data = res.json()
        ai_text = data['candidates'][0]['content']['parts'][0]['text']
        json_match = re.search(r'\{.*\}', ai_text, re.S)
        if json_match:
            return json.loads(json_match.group(0))
        return {"ai_text": ai_text}
    except Exception as e:
        return {"error": str(e)}

def learn_new_pattern(service: str, new_patterns: Dict[str, str]):
    """Ongeza patterns mpya kutoka kwa AI."""
    if service not in patterns:
        patterns[service] = {}
    for key, regex in new_patterns.items():
        patterns[service][key] = regex
    save_patterns()

# -------------------- TRANSACTION PARSING --------------------
class SMSPayload(BaseModel):
    user_id: str
    messages: List[str]

def detect_service(msg: str) -> str:
    msg_low = msg.lower()
    if "airtel" in msg_low or "recharge" in msg_low:
        return "AirtelMoney"
    if "mpesa" in msg_low or "confirmed" in msg_low:
        return "MPESA"
    if "mixx" in msg_low or "yas" in msg_low or "tigo pesa sasa ni mixx" in msg_low:
        return "MIXX"
    return "Unknown"

def parse_with_patterns(msg: str, service: str) -> Optional[Dict[str, Any]]:
    if service not in patterns:
        return None
    for txn_type, regex in patterns[service].items():
        try:
            match = re.search(regex, msg, re.IGNORECASE)
            if match:
                # Kagua kama kuna capture groups
                amount = None
                if match.groups():
                    # Chukua group ya kwanza ikiwa ipo
                    g = match.group(1)
                    if g:
                        amount = g.replace(",", "")
                return {
                    "type": txn_type,
                    "amount": amount,
                    "raw": msg,
                    "service": service,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        except re.error as e:
            print(f"Regex error ({txn_type}): {e}")
            continue
    return None

def save_transaction(user_id: str, txn: Dict[str, Any]):
    user_path = os.path.join(USER_DATA_DIR, user_id)
    os.makedirs(user_path, exist_ok=True)
    file_path = os.path.join(user_path, "transactions.json")

    transactions = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    transactions = json.loads(content)
        except json.JSONDecodeError:
            transactions = []

    transactions.append(txn)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(transactions, f, indent=2)

def analyze_message(msg: str, user_id: str) -> Dict[str, Any]:
    service = detect_service(msg)
    parsed = parse_with_patterns(msg, service)

    if parsed:
        save_transaction(user_id, parsed)
        return parsed

    # Hakuna pattern iliyopatikana - tumia AI
    ai_result = ask_ai_for_regex(msg)
    if ai_result and isinstance(ai_result, dict) and "error" not in ai_result:
        learn_new_pattern(service if service != "Unknown" else "AI_Learned", ai_result)
        return {"type": "unknown", "raw": msg, "ai_learned": ai_result}

    return {"type": "unknown", "raw": msg, "service": service}

# -------------------- USER SUMMARY --------------------
def load_transactions(user_id):
    path = os.path.join(USER_DATA_DIR, user_id, "transactions.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_summary(user_id, summary):
    path = os.path.join(USER_DATA_DIR, user_id, "summary.json")
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
        amount = float(t["amount"])
        dt = datetime.strptime(t["date"], "%Y-%m-%d %H:%M:%S")
        day_key = dt.strftime("%Y-%m-%d")
        week_key = f"week_{dt.strftime('%U_%Y')}"
        month_key = dt.strftime("%B_%Y")

        if t["type"] in ["withdrawal", "payment", "loan_repayment", "airtime"]:
            daily[day_key] += amount
            weekly[week_key] += amount
            monthly[month_key] += amount

    summary = {
        "daily": dict(daily),
        "weekly": dict(weekly),
        "monthly": dict(monthly)
    }
    save_summary(user_id, summary)
    return summary

def get_ai_advice(user_id):
    summary_path = os.path.join(USER_DATA_DIR, user_id, "summary.json")
    if not os.path.exists(summary_path):
        return "Hakuna data ya kutosha kutoa ushauri."

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    latest_week = max(summary["weekly"], default=None, key=lambda x: summary["weekly"][x]) if summary["weekly"] else None
    if latest_week and float(summary["weekly"][latest_week]) > 100000:
        return f"Wiki hii umetumia {summary['weekly'][latest_week]} TZS. Jaribu kuokoa 10% ya mapato yako wiki ijayo."
    
    return "Matumizi yako yako sawa. Endelea kudhibiti gharama zako."

# -------------------- API ENDPOINTS --------------------
@app.get("/")
def root():
    return {"message": "MMTA Backend V7 - AI Auto-Learning Patterns + User Summary"}

@app.post("/analyze")
def analyze_sms(payload: SMSPayload):
    results = [analyze_message(msg, payload.user_id) for msg in payload.messages]
    summary = update_user_summary(payload.user_id)
    return {
        "analysis": results,
        "summary": summary,
        "advice": get_ai_advice(payload.user_id),
        "total_transactions": len(results)
    }
