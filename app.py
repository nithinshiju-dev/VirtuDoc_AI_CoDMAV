# app.py
# Run: streamlit run app.py

import base64
from io import BytesIO
from pathlib import Path
import tempfile
import time
import json
import requests
import streamlit as st

# Mic + STT/TTS
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from gtts import gTTS

# Optional offline TTS
try:
    import pyttsx3
    PYTTSX3_OK = True
except Exception:
    PYTTSX3_OK = False

# ---------------- Config ----------------
API_BASE = "http://127.0.0.1:8000"  # change if your backend runs on a different port

st.set_page_config(page_title="Virtu Doc AI", layout="wide")

# ---------------- Styles ----------------
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

# ---------------- State ----------------
def init_state():
    defaults = dict(
        history=[],
        patient={
            "patient_id": "",
            "name": "",
            "age": "",
            "disease": "",
            "current_medication": "",
            "summary": "",  # NEW: used by right-side big Summary box
        },
        dx_json={"probable_condition":"", "key_findings":"", "risk_level":"", "red_flags":""},
        dx_narrative="",
        last_reply="",
        chat_input="",
        auto_send_voice=True,
        voice_replies=True,
        tts_engine="gTTS (online)",   # or "Offline (pyttsx3)"
        tts_lang="en",
        tts_rate=175,                 # pyttsx3 only
        spoken_upto=0,                # index of bot messages already spoken
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ---------------- Helpers ----------------
def play_audio(bytes_data: bytes, mime: str = "audio/mp3"):
    """Auto-play audio without visible download controls."""
    if not bytes_data:
        return
    b64 = base64.b64encode(bytes_data).decode()
    st.markdown(
        f'<audio autoplay src="data:{mime};base64,{b64}"></audio>',
        unsafe_allow_html=True
    )

def transcribe_wav_bytes(wav_bytes: bytes, lang="en-IN") -> str:
    rec = sr.Recognizer()
    with sr.AudioFile(BytesIO(wav_bytes)) as source:
        audio_data = rec.record(source)
    return rec.recognize_google(audio_data, language=lang)

def tts_gtts_chunks(text: str, lang: str = "en"):
    """Yield MP3 chunks (<=3000 chars) using gTTS (in-memory only)."""
    if not text or not text.strip():
        return
    CHUNK = 3000
    for i in range(0, len(text), CHUNK):
        chunk = text[i:i+CHUNK]
        fp = BytesIO()
        gTTS(text=chunk, lang=lang).write_to_fp(fp)
        fp.seek(0)
        yield fp.read()

def tts_pyttsx3_to_file(text: str, rate: int = 175) -> bytes:
    """Synthesize via pyttsx3 to a temp WAV file; return bytes (the temp file is removed)."""
    if not PYTTSX3_OK:
        raise RuntimeError("pyttsx3 not available")
    engine = pyttsx3.init()
    engine.setProperty('rate', int(rate))
    tmpdir = Path(tempfile.mkdtemp())
    wav_path = tmpdir / f"tts_{int(time.time()*1000)}.wav"
    engine.save_to_file(text, str(wav_path))
    engine.runAndWait()
    data = wav_path.read_bytes()
    try: wav_path.unlink(missing_ok=True)
    except Exception: pass
    return data

def speak_text(text: str):
    """Return list of (mime, bytes) according to selected TTS engine."""
    outs = []
    if not text or not text.strip():
        return outs
    if st.session_state.tts_engine == "gTTS (online)":
        try:
            for mp3_bytes in tts_gtts_chunks(text, st.session_state.tts_lang or "en"):
                outs.append(("audio/mp3", mp3_bytes))
        except Exception as e:
            st.warning(f"TTS (gTTS) failed: {e}")
    else:
        if not PYTTSX3_OK:
            st.warning("Offline TTS not available. Install pyttsx3 or switch to gTTS.")
            return outs
        try:
            wav_bytes = tts_pyttsx3_to_file(text, st.session_state.tts_rate)
            outs.append(("audio/wav", wav_bytes))
        except Exception as e:
            st.warning(f"TTS (offline) failed: {e}")
    return outs

def api_get(path, **kw):
    return requests.get(f"{API_BASE}{path}", timeout=kw.pop("timeout", 10), **kw)

def api_post(path, **kw):
    return requests.post(f"{API_BASE}{path}", timeout=kw.pop("timeout", 40), **kw)

# ---------------- Header ----------------
hl, hr = st.columns([0.6, 0.4])
with hl:
    st.title("🩺 Virtu Doc AI")
with hr:
    st.session_state.voice_replies = st.toggle("🔈 Voice Replies", value=st.session_state.voice_replies)
    st.session_state.tts_engine = st.selectbox(
        "TTS Engine", ["gTTS (online)", "Offline (pyttsx3)"],
        index=0 if st.session_state.tts_engine=="gTTS (online)" else 1
    )
    if st.session_state.tts_engine == "gTTS (online)":
        st.session_state.tts_lang = st.text_input("Language (gTTS)", value=st.session_state.tts_lang, help="e.g., en, en-uk, hi, ta, te")
    else:
        st.session_state.tts_rate = st.slider("Voice Speed (offline)", 120, 220, int(st.session_state.tts_rate), 5)

col1, col2, col3 = st.columns([1.55, 1, 1.25])

# ---------------- Left: Chat ----------------
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
        st.checkbox("Auto-send mic input", key="auto_send_voice")

    # Chat window
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

    # Input area + Mic
    st.markdown('<div class="input-wrap">', unsafe_allow_html=True)
    row1 = st.columns([0.6, 0.15, 0.12, 0.13])

    with row1[0]:
        st.session_state.chat_input = st.text_input(
            "Type your message or use the mic",
            value=st.session_state.chat_input,
            key="chat_input_widget",
            placeholder=""
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
    with row1[3]:
        send_voice_clicked = st.button("Send Voice", use_container_width=True)

    # Mic -> STT
    transcript_text = None
    user_msg_text = None
    if audio and "bytes" in audio and audio["bytes"]:
        try:
            transcript_text = transcribe_wav_bytes(audio["bytes"], lang="en-IN")
            st.caption(f'Heard: "{transcript_text}"')
        except sr.UnknownValueError:
            st.warning("I couldn’t understand the audio. Try again.")
        except Exception as e:
            st.error(f"Speech recognition error: {e}")

        if transcript_text:
            if st.session_state.auto_send_voice:
                user_msg_text = transcript_text.strip()
            else:
                st.session_state.chat_input = transcript_text.strip()

    if send_voice_clicked and not st.session_state.auto_send_voice:
        user_msg_text = (st.session_state.chat_input or "").strip()

    if send_clicked:
        user_msg_text = (st.session_state.chat_input or "").strip()

    # Send message
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

                    if st.session_state.voice_replies and bot_reply:
                        for mime, audio_bytes in speak_text(bot_reply):
                            play_audio(audio_bytes, mime=mime)

                    st.session_state.chat_input = ""
                    st.session_state.spoken_upto = len(st.session_state.history)
                    if not st.session_state.voice_replies:
                        st.rerun()
            else:
                st.error(f"HTTP {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Cannot reach backend: {e}")

    st.markdown('</div>', unsafe_allow_html=True)  # end input-wrap
    st.markdown('</div>', unsafe_allow_html=True)  # end chat-shell

# Speak any newly arrived (unsaid) bot messages
if st.session_state.voice_replies and st.session_state.history:
    total = len(st.session_state.history)
    if st.session_state.spoken_upto < total:
        for i in range(st.session_state.spoken_upto, total):
            bot_text = st.session_state.history[i]["bot"]
            for mime, audio_bytes in speak_text(bot_text):
                play_audio(audio_bytes, mime=mime)
        st.session_state.spoken_upto = total

# ---------------- Center: Patient Details ----------------
with col2:
    st.subheader("👤 Patient Details")
    p = st.session_state.patient

    # New: Patient ID on top
    patient_id = st.text_input("Patient ID", value=p.get("patient_id", ""), placeholder="")

    # Demographics
    name = st.text_input("Name", value=p.get("name", ""))
    age = st.text_input("Age", value=p.get("age", ""))

    # Bigger, richer disease/symptom box
    disease = st.text_area(
        "Disease / Symptoms (add more details)",
        value=p.get("disease", ""),
        height=160,
        placeholder=""
        ,
    )

    # Current medications
    curr_med = st.text_area(
        "Current Medication",
        value=p.get("current_medication", ""),
        height=80,
        placeholder="",
    )

    # Auto-sync (no Save button)
    st.session_state.patient = {
        **p,
        "patient_id": patient_id,
        "name": name,
        "age": age,
        "disease": disease,
        "current_medication": curr_med,
    }

# ---------------- Right: Reports & Summary ----------------
with col3:
    st.subheader("📄 Upload Report → Diagnose")
    up = st.file_uploader("Upload a report (PDF or TXT)", type=["pdf", "txt"])
    if st.button("Analyze Report") and up is not None:
        try:
            files = {"file": (up.name, up.getvalue(), up.type)}
            r = requests.post(f"{API_BASE}/analyze_lab_report", files=files, timeout=60)
            if r.ok:
                out = r.json()
                st.success("✅ Lab report analyzed successfully.")
                # Store in Summary box
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

    # Replace Final Report with a big free-form Summary
    st.subheader("🧾 Summary")
    p = st.session_state.patient
    summary_text = st.text_area(
        "Encounter / Clinician Summary",
        value=p.get("summary", ""),
        height=280,
        placeholder="",
    )
    # Auto-sync summary into patient state
    st.session_state.patient["summary"] = summary_text
