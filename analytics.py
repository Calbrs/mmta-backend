import os
import json
from datetime import datetime
from collections import defaultdict

DATA_DIR = "data/users"

def load_transactions(user_id):
    path = os.path.join(DATA_DIR, user_id, "transactions.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def save_summary(user_id, summary):
    path = os.path.join(DATA_DIR, user_id, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

def update_user_summary(user_id):
    transactions = load_transactions(user_id)
    daily = defaultdict(float)
    weekly = defaultdict(float)
    monthly = defaultdict(float)

    for t in transactions:
        if not t.get("amount"):
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
    summary_path = os.path.join(DATA_DIR, user_id, "summary.json")
    if not os.path.exists(summary_path):
        return "Hakuna data ya kutosha kutoa ushauri."

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Simple AI rule: if this week spending > 100,000 TZS, advise saving
    latest_week = max(summary["weekly"], default=None, key=lambda x: summary["weekly"][x]) if summary["weekly"] else None
    if latest_week and float(summary["weekly"][latest_week]) > 100000:
        return f"Wiki hii umetumia {summary['weekly'][latest_week]} TZS. Jaribu kuokoa 10% ya mapato yako wiki ijayo."
    
    return "Matumizi yako yako sawa. Endelea kudhibiti gharama zako."
