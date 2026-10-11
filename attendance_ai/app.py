import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise SystemExit("GEMINI_API_KEY nahi mila. .env file me GEMINI_API_KEY=AIza... likho.")

DATA_FILE = BASE_DIR / "attendance.json"
MODEL_ID = "gemini-3.1-flash-lite"
client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_API_KEY,
)

# ---------- file helpers ----------
def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def find_student(data, rollno):
    for s in data["students"]:
        if s["rollno"] == rollno:
            return s
    raise ValueError(f"Roll no {rollno} wala student nahi mila")

def recalc(student):
    recorded = [v for v in student["attendance"].values() if v is not None]
    student["total_days"] = len(recorded)
    student["days_present"] = recorded.count("P")


# ---------- tools ----------
def get_attendance_percentage(rollno: str):
    data = load_data()
    s = find_student(data, rollno)
    recalc(s)
    pct = (s["days_present"] / s["total_days"] * 100) if s["total_days"] else 0.0
    return {
        "rollno": rollno,
        "name": s["name"],
        "days_present": s["days_present"],
        "total_days": s["total_days"],
        "percentage": round(pct, 2),
    }

def mark_attendance(rollno: str, date: str, status: str):
    if status not in ("P", "A"):
        raise ValueError("status sirf 'P' ya 'A' ho sakta hai")
    data = load_data()
    if date not in data["days"]:
        raise ValueError(f"{date} valid class day nahi hai")
    s = find_student(data, rollno)
    s["attendance"][date] = status
    recalc(s)
    save_data(data)
    return {"success": True, "rollno": rollno, "date": date, "status": status}

def remove_attendance(rollno: str, date: str):
    data = load_data()
    if date not in data["days"]:
        raise ValueError(f"{date} valid class day nahi hai")
    s = find_student(data, rollno)
    s["attendance"][date] = None
    recalc(s)
    save_data(data)
    return {"success": True, "rollno": rollno, "date": date, "status": "removed"}

TOOLS = {
    "get_attendance_percentage": get_attendance_percentage,
    "mark_attendance": mark_attendance,
    "remove_attendance": remove_attendance,
}


# ---------- AI ----------
SYSTEM_PROMPT = """You are an attendance assistant. You do NOT calculate or store data yourself.

Tools:
- get_attendance_percentage(rollno: string)
- mark_attendance(rollno: string, date: string as YYYY-MM-DD, status: "P" or "A")
- remove_attendance(rollno: string, date: string as YYYY-MM-DD)

Reply with ONLY one JSON object, no extra text:
{"tool": "<tool name>", "arguments": {...}}

If no tool fits, reply:
{"tool": "none", "arguments": {"message": "<short reason>"}}
"""
def ask_model(user_text, data):
    system = SYSTEM_PROMPT + f"\nValid dates: {', '.join(data['days'])}"
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        max_tokens=200,
        temperature=0.0,
    )
    return resp.choices[0].message.content
def parse_tool_call(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model output me JSON nahi mila")
    return json.loads(match.group())

def plan_request(user_text):
    data = load_data()
    raw = ask_model(user_text, data)
    return parse_tool_call(raw)

def run_call(call):
    try:
        name = call["tool"]
        args = call.get("arguments", {})
        if name == "none":
            return {"ok": False, "error": args.get("message", "Request supported nahi hai")}
        if name not in TOOLS:
            return {"ok": False, "error": f"Unknown tool: {name}"}
        return {"ok": True, "result": TOOLS[name](**args)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------- main loop ----------
if __name__ == "__main__":
    print("Attendance system ready. Band karne ke liye 'quit' likho.")
    while True:
        text = input("\nAapka instruction: ").strip()
        if text.lower() == "quit":
            break
        if not text:
            continue

        try:
            call = plan_request(text)
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False, indent=2))
            continue

        if call.get("tool") == "remove_attendance":
            args = call.get("arguments", {})
            answer = input(
                f"Confirm karo: roll {args.get('rollno')} ka {args.get('date')} ka record remove karna hai? (yes/no): "
            )
            if answer.strip().lower() != "yes":
                print("Cancel ho gaya, kuch change nahi hua.")
                continue

        print(json.dumps(run_call(call), ensure_ascii=False, indent=2))