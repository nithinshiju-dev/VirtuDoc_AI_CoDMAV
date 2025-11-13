import base64
from io import BytesIO
from pathlib import Path
import tempfile
import time
import json
import requests
import streamlit as st
from streamlit_mic_recorder import mic_recorder

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Virtu Doc AI", layout="wide")

st.markdown("""
<style>
:root { --card:#141414; --muted:#2a2a2a; --bubble:#262626; --user:#2563eb; }
.chat-shell { display:flex; flex-direction:column; gap:10px; }
.chat-box {
  background:var(--card);
  border:1px solid var(--muted);
  border-radius:10px;
  height:420px;
  padding:12px;
  overflow-y:auto;
  display:flex;
  flex-direction:column;
}
.msg { border-radius:10px; padding:8px 10px; margin:6px 0; word-wrap:break-word; }
.user { background:var(--user); color:white; text-align:right; align-self:flex-end; }
.bot { background:var(--bubble); color:#e5e7eb; text-align:left; align-self:flex-start; }
.input-wrap { background:#0f0f0f; border:1px solid var(--muted); border-radius:10px; padding:10px; }
.small-muted { color:#9ca3af; font-size:12px; }
.badge { padding:2px 8px; border-radius:999px; background:#1f2937; color:#cbd5e1; font-size:12px; }
</style>
""", unsafe_allow_html=True)

def init_state():
    defaults = dict(
        history=[],
        patient={
            "patient_id": "",
            "name": "",
            "age": "",
            "disease": "",
            "current_medication": "",
            "summary": "",
        },
        dx_json={"probable_condition":"", "key_findings":"", "risk_level":"", "red_flags":""},
        dx_narrative="",
        last_reply="",
        chat_input="",
        spoken_upto=0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def api_get(path, **kw):
    return requests.get(f"{API_BASE}{path}", timeout=kw.pop("timeout", 10), **kw)

def api_post(path, **kw):
    return requests.post(f"{API_BASE}{path}", timeout=kw.pop("timeout", 40), **kw)

hl, hr = st.columns([0.6, 0.4])
with hl:
    st.title("🩺 Virtu Doc AI")
with hr:
    st.caption("Mic is UI only — no speech features installed.")

col1, col2, col3 = st.columns([1.55, 1, 1.25])

with col1:
    st.subheader("💬 Chat with Virtu Doc AI")

    t1, t2, t3, t4 = st.columns([1,1,1,1])
    if t1.button("Health"):
        try:
            st.success(api_get("/health").json())
        except Exception as e:
            st.error(e)
    if t2.button("Clear Chat"):
        st.session_state.history = []
        st.session_state.spoken_upto = 0
        st.session_state.chat_input = ""
        st.rerun()
    if t3.button("Models"):
        try:
            r = api_get("/models").json()
            st.info(f"Models: {r.get('count','?')}")
        except Exception as e:
            st.error(e)
    with t4:
        st.checkbox("Auto-send mic (UI only)", value=False, disabled=True)

    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    chat_html = ['<div class="chat-box" id="chatBox">']
    if not st.session_state.history:
        chat_html.append('<div class="msg bot">Hi! I’m Virtu Doc AI. Tell me what’s going on, or tap 🎤 to speak.</div>')
    else:
        for t in st.session_state.history:
            chat_html.append(f'<div class="msg user">{t["user"]}</div>')
            chat_html.append(f'<div class="msg bot">{t["bot"]}</div>')
    chat_html.append("</div>")
    st.markdown("".join(chat_html), unsafe_allow_html=True)
    st.markdown("""
    <script>
      const box = window.parent.document.getElementById('chatBox');
      if (box) { box.scrollTop = box.scrollHeight; }
    </script>
    """, unsafe_allow_html=True)

    st.markdown('<div class="input-wrap">', unsafe_allow_html=True)
    row1 = st.columns([0.6, 0.15, 0.12, 0.13])

    with row1[0]:
        st.session_state.chat_input = st.text_input(
            "",
            value=st.session_state.chat_input,
            key="chat_input_widget",
            placeholder="Type your message..."
        )
    with row1[1]:
        send_clicked = st.button("Send", use_container_width=True)
    with row1[2]:
        st.write(" ")
        audio = mic_recorder(
            start_prompt="🎤 Speak",
            stop_prompt="⏹️ Stop",
            just_once=True,
            format="wav",
            key="mic_key",
            use_container_width=True
        )
        if audio and "bytes" in audio and audio["bytes"]:
            st.caption("Mic pressed (UI only). Audio ignored.")
    with row1[3]:
        send_voice_clicked = st.button("Send Voice", use_container_width=True)

    user_msg_text = None
    if send_clicked:
        user_msg_text = (st.session_state.chat_input or "").strip()
    if send_voice_clicked:
        user_msg_text = (st.session_state.chat_input or "").strip()

    if user_msg_text:
        payload = {
            "user_input": user_msg_text,
            "history": st.session_state.history,
            "patient": st.session_state.patient
        }
        try:
            res = api_post("/chat", json=payload)
            if res.ok:
                data = res.json()
                if data.get("error"):
                    st.error(data["error"])
                else:
                    bot_reply = (data.get("reply") or "").strip()
                    st.session_state.last_reply = bot_reply
                    st.session_state.patient = data.get("patient_updates") or st.session_state.patient
                    st.session_state.history.append({"user": user_msg_text, "bot": bot_reply})
                    st.session_state.chat_input = ""
                    st.session_state.spoken_upto = len(st.session_state.history)
                    st.rerun()
            else:
                st.error(f"HTTP {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Cannot reach backend: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("👤 Patient Details")
    p = st.session_state.patient

    patient_id = st.text_input("Patient ID", value=p.get("patient_id", ""), placeholder="")
    name = st.text_input("Name", value=p.get("name", ""))
    age = st.text_input("Age", value=p.get("age", ""))

    disease = st.text_area(
        "Disease / Symptoms",
        value=p.get("disease", ""),
        height=160,
        placeholder=""
    )

    curr_med = st.text_area(
        "Current Medication",
        value=p.get("current_medication", ""),
        height=80,
        placeholder=""
    )

    st.session_state.patient = {
        **p,
        "patient_id": patient_id,
        "name": name,
        "age": age,
        "disease": disease,
        "current_medication": curr_med,
    }

with col3:
    st.subheader("📄 Upload Report → Diagnose")
    up = st.file_uploader("Upload a report (PDF or TXT)", type=["pdf", "txt"])
    if st.button("Analyze Report") and up is not None:
        try:
            files = {"file": (up.name, up.getvalue(), up.type)}
            r = requests.post(f"{API_BASE}/analyze_lab_report", files=files, timeout=60)
            if r.ok:
                out = r.json()
                summary_text = f"**Lab Analysis Summary:**\n\n"
                summary_text += f"**Triage:** {out.get('triage', '-').title()}\n\n"
                summary_text += "**Extracted Labs:**\n"
                for k, v in (out.get('labs_extracted') or {}).items():
                    summary_text += f"- {k}: {v}\n"
                summary_text += "\n**Diet Plan Suggestion:**\n"
                for item in out.get("diet_plan", []):
                    summary_text += f"- {item}\n"
                summary_text += "\n" + "\n".join(out.get("notes", []))
                st.session_state.patient["summary"] = summary_text
            else:
                st.error(f"HTTP {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"Upload failed: {e}")

    st.caption("Diagnosis Summary")
    st.write(st.session_state.dx_narrative or "—")

    st.subheader("🧾 Summary")
    p = st.session_state.patient
    summary_text = st.text_area(
        "Encounter / Clinician Summary",
        value=p.get("summary", ""),
        height=280,
        placeholder="",
    )
    st.session_state.patient["summary"] = summary_text

