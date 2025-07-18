from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import re
import json
import os

app = FastAPI()

# Data model
class SMSPayload(BaseModel):
    messages: List[str]

# JSON file path
DB_FILE = "transactions.json"

# Regex patterns
patterns = {
    "withdrawal": r"Withdrawn\s([\d,]+\.?\d*)\sTsh from Agent: ([A-Za-z\s]+)\. Fee\s([\d,]+\.?\d*)\sTsh",
    "payment": r"You have paid\s([\d,]+\.?\d*)\sTsh to (.+?)\.",
    "deposit": r"Received Tsh\s([\d,]+\.?\d*) from (.+?)\.",
    "loan_received": r"Received loan of Tsh\s([\d,]+\.?\d*)",
    "loan_repayment": r"You have repaid\s([\d,]+\.?\d*) Tsh",
    "airtime": r"Recharge successful.*?Tsh([\d,]+\.?\d*)",
    "failed": r"Transaction failed"
}

def extract_date(message: str):
    """
    Extracts a date-like pattern from message.
    Examples:
    - CO250610.0910.N30664 -> 2025-06-10
    - 18/07/2025
    """
    # Pattern ya siku/mwezi/mwaka
    date_pattern = re.search(r"(\d{2}/\d{2}/\d{4})", message)
    if date_pattern:
        return date_pattern.group(1)
    
    # Pattern ya mfano: CO250610 (-> 2025-06-10)
    code_pattern = re.search(r"[A-Z]{2}(\d{2})(\d{2})(\d{2})", message)
    if code_pattern:
        year = "20" + code_pattern.group(1)
        month = code_pattern.group(2)
        day = code_pattern.group(3)
        return f"{year}-{month}-{day}"
    
    return None

# Load existing transactions
def load_transactions():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return []

# Save transactions to JSON
def save_transactions(transactions):
    with open(DB_FILE, "w") as f:
        json.dump(transactions, f, indent=4)

# Generate unique ID
def generate_transaction_id(transactions):
    return f"mmta-{len(transactions) + 1:05d}"

# Analyze message
def analyze_message(message, transactions):
    for txn_type, pattern in patterns.items():
        match = re.search(pattern, message)
        if match:
            result = {"id": generate_transaction_id(transactions), "type": txn_type}
            
            # Handle based on type
            if txn_type == "withdrawal":
                result.update({
                    "amount": match.group(1).replace(',', ''),
                    "agent": match.group(2).strip(),
                    "fee": match.group(3).replace(',', '')
                })
            elif txn_type in ["payment", "deposit"]:
                result["amount"] = match.group(1).replace(',', '')
                if len(match.groups()) > 1:
                    recipient = match.group(2).strip()
                    if txn_type == "payment":
                        result["recipient"] = recipient
                    else:
                        result["sender"] = recipient
            else:
                result["amount"] = match.group(1).replace(',', '') if match.groups() else None

            # Add date if found
            result["date"] = extract_date(message)

            return result
    return {"id": generate_transaction_id(transactions), "type": "unknown", "amount": None, "date": extract_date(message)}

@app.post("/analyze")
def analyze_sms(data: SMSPayload):
    transactions = load_transactions()
    analysis = [analyze_message(msg, transactions) for msg in data.messages]
    transactions.extend(analysis)
    save_transactions(transactions)
    return {"analysis": analysis, "total_transactions": len(transactions)}
