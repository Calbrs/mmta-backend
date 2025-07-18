from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import re
import json
import os
from datetime import datetime
from analytics import update_user_summary, get_ai_advice

app = FastAPI()

# Data model
class SMSPayload(BaseModel):
    user_id: str
    messages: List[str]

# Regex patterns (updated)
patterns = {
    "withdrawal": r"Withdrawn\s([\d,]+\.?\d*)\sTsh from Agent: (.+?)\. Fee ([\d,]+\.?\d*) Tsh",
    "payment": r"You have paid\s([\d,]+\.?\d*)\sTsh to (.+?)\.",
    "deposit": r"Received Tsh\s([\d,]+\.?\d*) from (.+?)\.",
    "loan_received": r"Received loan of Tsh\s([\d,]+\.?\d*)",
    "loan_repayment": r"You have repaid\s([\d,]+\.?\d*) Tsh",
    "airtime": r"Recharge successful.*?Tsh([\d,]+\.?\d*)",
    "balance": r"Balance\sTsh\s([\d,]+\.?\d*)",
    "failed": r"Transaction failed"
}

DATA_DIR = "data/users"

# Helper to generate transaction ID
def generate_transaction_id(user_id):
    folder = os.path.join(DATA_DIR, user_id)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, "transactions.json")

    if not os.path.exists(file_path):
        return "mmta-00001"

    with open(file_path, "r") as f:
        data = json.load(f)
    return f"mmta-{len(data) + 1:05d}"

# Analyze single message
def analyze_message(message, user_id):
    for txn_type, pattern in patterns.items():
        match = re.search(pattern, message)
        if match:
            txn = {
                "id": generate_transaction_id(user_id),
                "type": txn_type,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            if txn_type == "withdrawal":
                txn.update({"amount": match.group(1).replace(',', ''), "agent": match.group(2), "fee": match.group(3)})
            elif txn_type == "payment":
                txn.update({"amount": match.group(1).replace(',', ''), "recipient": match.group(2)})
            elif txn_type == "deposit":
                txn.update({"amount": match.group(1).replace(',', ''), "sender": match.group(2)})
            elif txn_type == "loan_received":
                txn.update({"amount": match.group(1).replace(',', '')})
            elif txn_type == "loan_repayment":
                txn.update({"amount": match.group(1).replace(',', '')})
            elif txn_type == "airtime":
                txn.update({"amount": match.group(1).replace(',', '')})
            elif txn_type == "balance":
                txn.update({"remaining_balance": match.group(1).replace(',', '')})
            else:
                txn.update({"amount": None})

            return txn
    return {"id": generate_transaction_id(user_id), "type": "unknown", "amount": None, "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# Save transaction
def save_transaction(user_id, transactions):
    folder = os.path.join(DATA_DIR, user_id)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, "transactions.json")

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.extend(transactions)

    with open(file_path, "w") as f:
        json.dump(existing, f, indent=2)

    return len(existing)

@app.get("/")
def root():
    return {"message": "MMTA Backend V2 is running!", "endpoints": ["/analyze"]}

@app.post("/analyze")
def analyze_sms(data: SMSPayload):
    transactions = [analyze_message(msg, data.user_id) for msg in data.messages]
    save_transaction(data.user_id, transactions)
    
    # Update summaries
    update_user_summary(data.user_id)

    # Generate AI advice
    advice = get_ai_advice(data.user_id)

    return {"analysis": transactions, "advice": advice}
