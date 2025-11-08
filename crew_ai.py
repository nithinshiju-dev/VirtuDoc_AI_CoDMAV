# crew_ai.py
# Run:
#   uvicorn crew_ai:app --reload --host 127.0.0.1 --port 8000

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import json
import os
import random
import pdfplumber
import google.generativeai as genai
import re
import pytesseract
from PIL import Image
import mysql.connector
import tempfile

# ========= CONFIG =========
# NOTE: You asked to keep your hardcoded key and MySQL creds.
GEMINI_API_KEY = ""
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY env var before starting the backend.")
genai.configure(api_key=GEMINI_API_KEY)

PICKED_MODEL = "gemini-2.5-flash"
MODEL = genai.GenerativeModel(PICKED_MODEL)
GEN_CONFIG = {"max_output_tokens": 320, "temperature": 0.35, "top_p": 0.9, "top_k": 40}
AUTO_FALLBACK_TO_LOCAL_ON_429 = True

# ========= AGENT PROFILES =========
class AgentProfile(BaseModel):
    role: str
    goal: str
    backstory: str


doctor_agent = AgentProfile(
    role="Virtu Doc AI (Conversational Doctor)",
    goal=(
        "Have a warm, human conversation that builds a complete, clear picture of the problem. "
        "Keep a clean record: name, age, disease/symptoms, current_medication, duration, severity, triggers, "
        "relievers, associated symptoms, and relevant history. "
        "Ask ONE short, focused question at a time and keep asking logically connected follow-ups until clarity. "
        "Vary your phrasing, avoid repeating the same question, and acknowledge the patient's updates kindly. "
        "Use everyday language and keep replies brief and supportive. "
        "If the patient describes red-flag or severe symptoms, immediately advise visiting the nearest emergency department or hospital. "
        "If the user asks about any medicine, dosage, drug combinations, or side effects — "
        "DO NOT give medical advice. Instead, politely say that you will review it with the doctor and get back to them soon."
    ),
    backstory=(
        "You are Dr. Virtu, a caring and detail-oriented clinician. "
        "You listen actively, reflect back key points in simple words, and guide the discussion gently. "
        "You avoid sounding robotic by varying your wording and tone, and you never repeat the exact same question if it was asked recently. "
        "You prioritize safety: when severity is high or red-flags appear, you clearly recommend urgent in-person care. "
        "When medicine-related questions arise, you respond cautiously, saying that you’ll check with the doctor and update the patient soon — "
        "you never provide prescriptions, dosages, or drug interaction advice directly."
    )
)



RED_FLAG_PATTERNS = [
    "severe pain", "worst pain", "crushing chest pain", "chest pain",
    "shortness of breath", "breathless", "difficulty breathing",
    "fainting", "passed out", "loss of consciousness",
    "confusion", "disoriented", "seizure",
    "one-sided weakness", "face droop", "slurred speech", "stroke",
    "uncontrolled bleeding", "heavy bleeding",
    "fever 103", "fever 104", "high fever", "very high fever", "39c", "40c",
    "pregnant and bleeding", "severe dehydration",
    "stiff neck with fever", "severe headache sudden", "thunderclap headache",
    "poisoning", "overdose",
]


def is_red_flag(user_text: str) -> bool:
    t = (user_text or "").lower()
    return any(p in t for p in RED_FLAG_PATTERNS)


AFFIRM_PATTERNS = [
    "ok", "okay", "yes", "yep", "yeah", "sure", "please do", "pls do",
    "go ahead", "do it", "book", "inform", "notify", "call them", "proceed"
]
NEGATE_PATTERNS = ["no", "not now", "don't", "do not", "nope", "later", "stop", "cancel"]


def is_affirmative(text: str) -> bool:
    t = (text or "").lower().strip()
    if not t:
        return False
    if any(n in t for n in NEGATE_PATTERNS):
        return False
    return any(a in t for a in AFFIRM_PATTERNS)


def asked_topic_recently(history, keywords, lookback=5) -> bool:
    recent = history[-lookback:] if history else []
    for turn in recent:
        bot = (turn.bot or "").lower()
        if any(k in bot for k in keywords):
            return True
    return False


QUESTION_TEMPLATES = {
    "duration": [
        "How long has this been going on?",
        "When did this start?",
        "Since when are you noticing these symptoms?",
    ],
    "severity": [
        "How intense is it right now (mild, moderate, severe)?",
        "On a scale of 1–10, how painful or bothersome is it?",
        "Would you call it mild, moderate, or severe?",
    ],
    "triggers": [
        "Have you noticed anything that makes it better or worse?",
        "Do certain foods, activities, or times of day change it?",
        "What seems to trigger or relieve it?",
    ],
    "assoc": [
        "Any other symptoms along with this (fever, nausea, cough, etc.)?",
        "Are there any additional symptoms you’ve noticed?",
        "Anything else happening at the same time?",
    ],
}


def pick_variant(topic: str) -> str:
    return random.choice(QUESTION_TEMPLATES.get(topic, ["Could you tell me a bit more?"]))


diagnosis_agent = AgentProfile(
    role="Diagnosing Agent",
    goal=(
        "Read the uploaded medical report text and produce structured insights: probable_condition, key_findings, "
        "risk_level (low|moderate|high), and red_flags. Also produce a short clinician-style narrative."
    ),
    backstory=(
        "You are a careful medical report analyst. You extract factual findings and call out uncertainty clearly. "
        "Always return both a structured JSON and a concise narrative."
    ),
)

final_report_agent = AgentProfile(
    role="Final Report Agent",
    goal=(
        "Convert the patient record and diagnosis into a brief, patient-friendly report: Overview, Key Findings, "
        "Likely Causes, Care Plan (bulleted), and When to Seek Urgent Care."
    ),
    backstory=(
        "You write clearly for laypersons—warm but professional—with an emphasis on next steps and safety."
    ),
)

summary_agent = AgentProfile(
    role="Summary Agent",
    goal=(
        "Create or update a concise clinician-style encounter summary that captures the story so far: "
        "chief complaint, duration, severity, notable positives/negatives, suspected differentials (if any), "
        "current meds or lack thereof, and next steps. Keep it 5–8 lines max, crisp, and readable."
    ),
    backstory=(
        "You are precise and structured. You avoid speculation beyond what was said, but you consolidate "
        "the conversation into a clean medical note suitable for quick review."
    ),
)

# ========= SCHEMAS =========
class Turn(BaseModel):
    user: str
    bot: str


class ChatRequest(BaseModel):
    user_input: str
    history: List[Turn] = []
    patient: Dict[str, str] = {}


class ChatResponse(BaseModel):
    reply: Optional[str] = None
    patient_updates: Dict[str, str] = {}
    used_local_fallback: Optional[bool] = None
    error: Optional[str] = None


class DiagnoseResponse(BaseModel):
    diagnosis_json: Dict[str, str]
    diagnosis_narrative: str


class FinalReportRequest(BaseModel):
    patient: Dict[str, str]
    diagnosis_json: Dict[str, str]
    diagnosis_narrative: str


class FinalReportResponse(BaseModel):
    report_text: str


# ========= HELPERS =========
def is_quota_error(err: Exception) -> bool:
    s = str(err).lower()
    return "429" in s or "quota" in s or "rate" in s or "resourceexhausted" in s


def ask_llm(prompt: str) -> str:
    resp = MODEL.generate_content(prompt, generation_config=GEN_CONFIG)
    return (getattr(resp, "text", "") or "").strip()


def extract_json_block(raw: str) -> Dict[str, str]:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        text = match.group(0).replace("\n", " ").replace("'", '"')
        text = re.sub(r",\s*}", "}", text)
        try:
            return json.loads(text)
        except Exception:
            return {}


def read_report_text(file: UploadFile) -> str:
    name = (file.filename or "").lower()
    data = file.file.read()
    if name.endswith(".pdf"):
        try:
            with pdfplumber.open(BytesIO(data)) as pdf:
                return "\n".join([p.extract_text() or "" for p in pdf.pages]).strip()
        except Exception:
            return ""
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def meds_normalize(v: str) -> str:
    s = _norm(v)
    if not s:
        return ""
    none_phrases = {
        "none", "no meds", "no medicine", "no medication", "not taking any",
        "nil", "-", "no", "no current medication", "no medications",
        "no medication currently", "no current medications"
    }
    return "none" if s in none_phrases else v.strip()


def compute_missing_fields(patient: Dict[str, str]) -> list:
    want = ["name", "age", "disease", "current_medication"]
    return [k for k in want if not (patient.get(k) or "").strip()]


def diff_updates(prev: Dict[str, str], new: Dict[str, str]) -> Dict[str, str]:
    changed = {}
    for k, v in new.items():
        vv = str(v) if not isinstance(v, str) else v
        if vv.strip() and vv.strip() != (prev.get(k, "") or "").strip():
            changed[k] = vv.strip()
    return changed


def merge_patient(old: Dict[str, str], newbits: Dict[str, str]) -> Dict[str, str]:
    out = dict(old)
    if "current_medication" in newbits and isinstance(newbits["current_medication"], str):
        newbits["current_medication"] = meds_normalize(newbits["current_medication"])
    for k, v in (newbits or {}).items():
        if isinstance(v, (int, float)):
            v = str(v)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()
    return out


def generate_patient_summary(history: list, patient: Dict[str, str]) -> str:
    transcript = "\n".join([f"User: {t.user}\nDoctor: {t.bot}" for t in history]) if history else ""
    prompt = f"""
[{summary_agent.role}]
Goal: {summary_agent.goal}
Backstory: {summary_agent.backstory}

Patient JSON:
{json.dumps(patient, indent=2)}

Conversation:
{transcript[-4000:]}

Write a concise 5–8 line summary.
""".strip()
    try:
        text = ask_llm(prompt)
        return (text or "").strip()[:2000]
    except Exception:
        cc = (patient.get("disease") or "Not specified").strip()
        meds = (patient.get("current_medication") or "Not specified").strip()
        return f"Chief complaint: {cc}. Current medication: {meds}. Further details pending."


# === NEW: Refine summary using LLM ===
def refine_summary(summary_text: str) -> str:
    if not summary_text or not summary_text.strip():
        return summary_text
    prompt = f"""
Format this patient summary clearly with bullets and spacing:
{summary_text}
"""
    try:
        return ask_llm(prompt).strip()
    except Exception:
        return summary_text


# ========= APP =========
app = FastAPI(title="Virtu Doc AI — Backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"status": "ok", "model": PICKED_MODEL}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    transcript = "\n".join([f"User: {t.user}\nDoctor: {t.bot}" for t in req.history])
    prev_bot = req.history[-1].bot if req.history else ""
    old_patient = req.patient or {}

    text_now = (req.user_input or "").lower()

    # === NEW: detect lab report mention ===
    if any(k in text_now for k in ["report", "blood test", "scan", "lab", "mri", "x-ray"]):
        reply = "You mentioned a report — please upload it using the 📄 Upload Report section on the right side and once you upload it click on the analyse button to get the diet plan."
        temp_turns = req.history + [Turn(user=req.user_input, bot=reply)]
        patient = merge_patient(old_patient, {})
        raw_summary = generate_patient_summary(temp_turns, patient)
        patient["summary"] = refine_summary(raw_summary)
        return ChatResponse(reply=reply, patient_updates=patient)

    # === Red flag check ===
    if is_red_flag(req.user_input):
        reply = (
            "This could be serious. Please visit the nearest hospital now. "
            "Would you like me to inform the Emergency Department of PES Medical College?"
        )

        patient = merge_patient(old_patient, {})
        raw_summary = generate_patient_summary(req.history + [Turn(user=req.user_input, bot=reply)], patient)
        patient["summary"] = refine_summary(raw_summary)
        return ChatResponse(reply=reply, patient_updates=patient)
    # === Booking confirmation check ===
    # === Booking confirmation check ===
    if is_affirmative(req.user_input) and any(
            k in text_now for k in ["pes", "appointment", "book", "hospital", "inform"]):
        prev_bot_msg = (prev_bot or "").lower()

        # --- If user is replying to the "inform emergency dept" question ---
        if "inform the emergency department" in prev_bot_msg or "visit the nearest hospital" in prev_bot_msg:
            reply = "✅ I’ve informed the PES Medical College Emergency Department. Please proceed safely and take care."
            patient = merge_patient(old_patient, {})
            patient["summary"] = refine_summary(
                generate_patient_summary(req.history + [Turn(user=req.user_input, bot=reply)], patient)
            )
            # Stop conversation here (no further follow-up)
            return ChatResponse(reply=reply, patient_updates=patient)

        # --- Otherwise normal booking flow ---
        patient = merge_patient(old_patient, {})
        pid = patient.get("id") or str(random.randint(1000, 9999))
        pname = patient.get("name", "Unknown")
        page = patient.get("age", 0)
        psymptom = patient.get("disease", "Not specified")

        try:
            add_patient(pid, pname, page, psymptom)
            reply = (
                f"✅ Your appointment at PES Hospital has been successfully booked, {pname}. "
                "Please proceed safely. The medical team will assist you upon arrival."
            )
        except Exception as e:
            reply = f"⚠️ Could not book appointment due to an internal error: {e}"

        patient["summary"] = refine_summary(
            generate_patient_summary(req.history + [Turn(user=req.user_input, bot=reply)], patient)
        )
        return ChatResponse(reply=reply, patient_updates=patient)

    # Extract info
    # === Extract info more reliably ===
    extractor_prompt = f"""
    [{doctor_agent.role} — JSON Extractor Mode]
    User just said: "{req.user_input}"

    Extract as valid JSON ONLY:
    {{
      "name": "",
      "age": "",
      "disease": "",
      "current_medication": ""
    }}

    Rules:
    - Fill what you can. Leave unknowns blank.
    - Output strictly JSON, nothing else.
    """

    try:
        llm_text = ask_llm(extractor_prompt)
        parsed = extract_json_block(llm_text)
    except Exception:
        parsed = {}

    # --- fallback simple extraction if JSON is empty ---
    if not parsed or not any(parsed.values()):
        txt = req.user_input.lower()
        parsed = {}
        # detect name (e.g., "my name is Rahul" or "I'm Rahul")
        name_match = re.search(r"(?:i[' ]?m|my name is)\s+([A-Za-z]+)", txt)
        if name_match:
            parsed["name"] = name_match.group(1).title()

        # detect age (e.g., "I am 25 years old" or "age 25")
        age_match = re.search(r"(?:i am|age is|age)\s+(\d{1,2})", txt)
        if age_match:
            parsed["age"] = age_match.group(1)

        # detect simple disease keywords
        # --- detect likely symptom or disease ---
        symptom_patterns = [
            r"(?:\b|^)(fever|temperature|pyrexia)(?:\b|$)",
            r"(?:\b|^)(headache|migraine|head pain)(?:\b|$)",
            r"(?:\b|^)(cough|cold|sore throat)(?:\b|$)",
            r"(?:\b|^)(vomit|nausea|queasy)(?:\b|$)",
            r"(?:\b|^)(stomach ache|abdominal pain|cramps)(?:\b|$)",
            # 👇 this one handles chest pain in multiple ways
            r"(?:\b|^)(chest pain|pain in chest|chest hurts|tightness in chest|chest discomfort|pain chest)(?:\b|$)",
            r"(?:\b|^)(back pain|body ache|pain)(?:\b|$)",
            r"(?:\b|^)(dizzy|lightheaded|vertigo)(?:\b|$)",
            r"(?:\b|^)(diabetes|blood sugar|sugar problem)(?:\b|$)",
            r"(?:\b|^)(diarrhea|loose motion)(?:\b|$)",
            r"(?:\b|^)(fatigue|tired|weak)(?:\b|$)",
        ]

        for pat in symptom_patterns:
            m = re.search(pat, txt)
            if m:
                parsed["disease"] = m.group(1).lower()
                break

        # detect medication (e.g., "taking paracetamol")
        med_match = re.search(r"(?:taking|on)\s+([A-Za-z0-9\- ]+)", txt)
        if med_match:
            parsed["current_medication"] = med_match.group(1).strip()

    newbits = {k: str(parsed.get(k, "") or "") for k in ["name", "age", "disease", "current_medication"]}
    patient = merge_patient(old_patient, newbits)

    missing = compute_missing_fields(patient)
    next_slot = missing[0] if missing else None

    # === if record complete ===
    if not next_slot:
        if not asked_topic_recently(req.history, ["how long", "duration"]):
            reply = pick_variant("duration")
        elif not asked_topic_recently(req.history, ["intense", "severe", "scale"]):
            reply = pick_variant("severity")
        elif not asked_topic_recently(req.history, ["triggers", "better", "worse"]):
            reply = pick_variant("triggers")
        elif not asked_topic_recently(req.history, ["associated symptoms", "any other symptoms"]):
            reply = pick_variant("assoc")
        else:
            reply = "Thanks, that helps. Is there anything else you'd like me to know?"

        raw_summary = generate_patient_summary(req.history + [Turn(user=req.user_input, bot=reply)], patient)
        patient["summary"] = refine_summary(raw_summary)
        return ChatResponse(reply=reply, patient_updates=patient)

    # === ask next missing slot ===
    doctor_prompt = f"""
[{doctor_agent.role}]
Goal: {doctor_agent.goal}
Conversation:
{transcript}
User said: "{req.user_input}"
Missing: {next_slot}
"""
    try:
        reply = ask_llm(doctor_prompt).strip()
    except Exception as e:
        reply = "Could you share a bit more detail?"

    # === Prevent repeated questions ===
    recent_bots = [t.bot.strip().lower() for t in req.history[-3:]]
    if reply.strip().lower() in recent_bots:
        reply = "Thanks for clarifying. Could you tell me more about when it started?"

    raw_summary = generate_patient_summary(req.history + [Turn(user=req.user_input, bot=reply)], patient)
    patient["summary"] = refine_summary(raw_summary)
    return ChatResponse(reply=reply, patient_updates=patient)


# ========= Agent 2: Diagnosing Agent (report analysis) =========
@app.post("/upload_report", response_model=DiagnoseResponse)
async def upload_report(file: UploadFile = File(...), patient_json: str = Form("{}")):
    try:
        patient = json.loads(patient_json) if patient_json else {}
    except Exception:
        patient = {}

    text = read_report_text(file)
    if not text:
        text = "[No text extracted from report]"

    diag_prompt = f"""
[{diagnosis_agent.role}]
Goal: {diagnosis_agent.goal}
Backstory: {diagnosis_agent.backstory}

Patient JSON:
{json.dumps(patient, indent=2)}

Report text:
{text[:20000]}

Produce TWO parts:
1) JSON object with keys: probable_condition, key_findings, risk_level, red_flags
2) Short clinician-style narrative starting with "narrative:"
"""
    raw = ask_llm(diag_prompt)
    parsed = extract_json_block(raw)
    narrative = ""
    after = raw[raw.rfind("}") + 1:] if "}" in raw else raw
    for line in after.splitlines():
        if line.strip().lower().startswith("narrative:"):
            narrative = line.split(":", 1)[-1].strip()
            break
    if not narrative:
        narrative = after.strip()[:600]

    dx = {
        "probable_condition": str(parsed.get("probable_condition", "") or ""),
        "key_findings": str(parsed.get("key_findings", "") or ""),
        "risk_level": str(parsed.get("risk_level", "") or ""),
        "red_flags": str(parsed.get("red_flags", "") or ""),
    }
    return DiagnoseResponse(diagnosis_json=dx, diagnosis_narrative=narrative)


# ========= Agent 3: Final Report Agent =========
@app.post("/final_report", response_model=FinalReportResponse)
def final_report(req: FinalReportRequest):
    prompt = f"""
[{final_report_agent.role}]
Goal: {final_report_agent.goal}
Backstory: {final_report_agent.backstory}

Patient JSON:
{json.dumps(req.patient, indent=2)}

Diagnosis JSON:
{json.dumps(req.diagnosis_json, indent=2)}

Diagnosis narrative:
{req.diagnosis_narrative}

Write a patient-friendly report with:
- Overview
- Key Findings
- Likely Causes
- Care Plan (bulleted)
- When to Seek Urgent Care
"""
    text = ask_llm(prompt)
    return FinalReportResponse(report_text=text)


@app.post("/final_report_pdf")
def final_report_pdf(req: FinalReportRequest):
    resp = final_report(req)
    report = resp.report_text or "No report."
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    x, y = 50, h - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Virtu Doc AI — Final Report")
    y -= 30
    c.setFont("Helvetica", 12)
    for para in report.split("\n"):
        line_buf = ""
        for word in para.split():
            test = (line_buf + " " + word).strip()
            if c.stringWidth(test, "Helvetica", 12) > (A4[0] - 2 * x):
                c.drawString(x, y, line_buf)
                y -= 16
                line_buf = word
                if y < 80:
                    c.showPage(); y = h - 80; c.setFont("Helvetica", 12)
            else:
                line_buf = test
        if line_buf:
            c.drawString(x, y, line_buf)
            y -= 16
            if y < 80:
                c.showPage(); y = h - 80; c.setFont("Helvetica", 12)
        y -= 4
    c.showPage(); c.save(); buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf",
                             headers={"Content-Disposition": 'attachment; filename="final_report.pdf"'})


# ========= MySQL Integration =========
def add_patient(patient_id, patient_name, patient_age, symptom_summary):
    try:
        conn = mysql.connector.connect(
            host="localhost", user="root", password="Password!123", database="hospital"
        )
        cursor = conn.cursor()
        query = """INSERT INTO emergency_department
                   (patient_id, patient_name, patient_age, symptom_summary)
                   VALUES (%s,%s,%s,%s)"""
        cursor.execute(query, (patient_id, patient_name, patient_age, symptom_summary))
        conn.commit(); print("✅ Patient data added successfully!")
    except mysql.connector.Error as err:
        print("❌ Error:", err)
    finally:
        try:
            if conn.is_connected(): cursor.close(); conn.close()
        except Exception:
            pass


def add_doctor_record(patient_id, patient_name, summary, agent_remarks="", status="Pending"):
    try:
        conn = mysql.connector.connect(
            host="localhost", user="root", password="Password!123", database="hospital"
        )
        cursor = conn.cursor()
        query = """INSERT INTO doctor_chat
                   (patient_id, patient_name, summary, agent_remarks, status)
                   VALUES (%s,%s,%s,%s,%s)"""
        cursor.execute(query, (patient_id, patient_name, summary, agent_remarks, status))
        conn.commit(); cursor.close(); conn.close()
        return f"✅ Record for '{patient_name}' added."
    except mysql.connector.Error as err:
        return f"❌ MySQL Error: {err}"


# ========= Lab Analyzer Integration =========
@app.post("/analyze_lab_report")
async def analyze_lab_report(file: UploadFile = File(...)):
    try:
        data = await file.read()
        fname = file.filename.lower()
        if fname.endswith(".pdf"):
            text = ""
            with pdfplumber.open(BytesIO(data)) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
        elif fname.endswith((".png", ".jpg", ".jpeg", ".webp")):
            img = Image.open(BytesIO(data))
            text = pytesseract.image_to_string(img)
        else:
            text = data.decode("utf-8", errors="ignore")

        patterns = {
            "HbA1c": r"\b(HbA1c|Glycated\s*Hb)\b[^0-9]*([\d.]+)",
            "FPG": r"\b(Fasting(\s*Plasma)?\s*(Glucose|Sugar))\b[^0-9]*([\d.]+)",
            "PPG": r"\b(Post[-\s]*Prandial|PPG)\b[^0-9]*([\d.]+)",
            "Cholesterol": r"\b(Total\s*Cholesterol)\b[^0-9]*([\d.]+)",
            "Triglycerides": r"\b(Triglycerides|TG)\b[^0-9]*([\d.]+)",
        }

        labs = {}
        for key, pat in patterns.items():
            m = re.search(pat, text, flags=re.IGNORECASE)
            labs[key] = float(m.groups()[-1]) if m else None

        triage = "low"
        notes = []
        if labs.get("HbA1c") and labs["HbA1c"] >= 6.5:
            triage = "high"; notes.append("Values suggest diabetes range.")
        elif labs.get("HbA1c") and 5.7 <= labs["HbA1c"] < 6.5:
            triage = "moderate"; notes.append("Prediabetes range.")
        if labs.get("FPG") and labs["FPG"] >= 126: triage = "high"
        if labs.get("PPG") and labs["PPG"] >= 200: triage = "high"

        if triage == "high":
            plan = [
                "Breakfast: 2 moong dal chillas + curd",
                "Lunch: Brown rice + rajma + salad",
                "Snack: Sprout chaat + buttermilk",
                "Dinner: 2 rotis + paneer bhurji + greens"
            ]
        elif triage == "moderate":
            plan = [
                "Breakfast: Oats upma + apple",
                "Lunch: Millet khichdi + curd",
                "Snack: Roasted chana + lemon water",
                "Dinner: Roti + dal + sabzi"
            ]
        else:
            plan = [
                "Breakfast: Poha + fruit",
                "Lunch: Rice + dal + sabzi",
                "Dinner: Roti + sabzi"
            ]

        return {
            "labs_extracted": labs,
            "triage": triage,
            "notes": notes,
            "diet_plan": plan,
            "message": "Diet plan generated successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing report: {e}")


# ========= Lab Summary Agent =========
class LabSummaryRequest(BaseModel):
    analysis_json: Dict[str, Any]


class LabSummaryResponse(BaseModel):
    summary_text: str


lab_summary_agent = AgentProfile(
    role="Lab Summary Agent",
    goal=(
        "Take structured lab analysis JSON and create a concise, readable summary in plain English. "
        "Summarize the triage level, highlight any abnormal findings, and explain what the diet plan means."
    ),
    backstory=(
        "You are an AI medical summarizer who explains lab results in clear and reassuring language, "
        "suitable for both patients and doctors. You write in a professional yet friendly tone."
    ),
)


@app.post("/summarize_lab_analysis", response_model=LabSummaryResponse)
def summarize_lab_analysis(req: LabSummaryRequest):
    """
    Takes JSON output from /analyze_lab_report and turns it into a clean, patient-friendly summary.
    """
    prompt = f"""
[{lab_summary_agent.role}]
Goal: {lab_summary_agent.goal}
Backstory: {lab_summary_agent.backstory}

Lab Analysis JSON:
{json.dumps(req.analysis_json, indent=2)}

Write a short paragraph (5–7 sentences) that clearly explains:
- Whether the results are normal or concerning
- Any key findings
- What the suggested diet plan means for the patient
- Next steps if necessary

Tone: calm, informative, and friendly. Avoid technical jargon.
"""
    try:
        summary_text = ask_llm(prompt)
        return LabSummaryResponse(summary_text=summary_text.strip())
    except Exception as e:
        return LabSummaryResponse(summary_text=f"Error generating summary: {e}")


# ========= Voice endpoints (placeholders) =========
@app.post("/voice_to_text")
async def voice_to_text(file: UploadFile = File(...)):
    """
    Transcribe audio -> text. Placeholder uses a local temp file and returns placeholder text.
    Replace with Whisper/actual provider if desired.
    """
    data = await file.read()
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tmp.write(data); tmp.flush(); tmp.close()
        # TODO: Call Whisper or your provider here. For now return placeholder.
        text = "(transcribed audio placeholder)"
        os.unlink(tmp.name)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speak_reply")
async def speak_reply(text: str = Form(...)):
    """
    Return synthesized speech audio for the given text. Placeholder returns a short silent wav.
    Replace with Coqui TTS / provider call.
    """
    import wave, struct
    samples = 16000
    buf = BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        for i in range(samples):
            w.writeframes(struct.pack('<h', 0))
    buf.seek(0)
    return StreamingResponse(buf, media_type='audio/wav')


# ========= Agent 5: emergency booking / doctor notify =========
class EmergencyBookingRequest(BaseModel):
    patient_id: str
    patient_name: str
    patient_age: Optional[int] = None
    symptom_summary: str


class EmergencyBookingResponse(BaseModel):
    status: str
    message: str


@app.post('/book_emergency', response_model=EmergencyBookingResponse)
def book_emergency(req: EmergencyBookingRequest):
    try:
        add_patient(req.patient_id, req.patient_name, req.patient_age or 0, req.symptom_summary)
        add_doctor_record(req.patient_id, req.patient_name, req.symptom_summary, agent_remarks="Auto-notified", status="Urgent")
        return EmergencyBookingResponse(status="ok", message="Emergency booked and doctor notified.")
    except Exception as e:
        return EmergencyBookingResponse(status="error", message=str(e))


@app.post('/notify_doctor')
def notify_doctor(patient_id: str = Form(...), patient_name: str = Form(...), summary: str = Form(...)):
    res = add_doctor_record(patient_id, patient_name, summary, agent_remarks="Manual notify", status="Pending")
    return {"result": res}


# ========= End of file =========

# If run directly
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("crew_ai:app", host="127.0.0.1", port=8000, reload=True)
