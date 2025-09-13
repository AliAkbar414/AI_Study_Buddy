# streamlit_app.py
import os
import re
import json
import sqlite3
import requests
from datetime import datetime
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv

# Try to import Mistral SDK - if it's not available on Streamlit cloud, fallback will show error
try:
    from mistralai import Mistral
except Exception:
    Mistral = None

# Load local .env when running locally (Streamlit Cloud uses secrets)
load_dotenv()

# ---------- Configuration (secrets from Streamlit or env) ----------
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = st.secrets.get("MISTRAL_MODEL") or os.getenv("MISTRAL_MODEL", "mistral-small")
YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY") or os.getenv("YOUTUBE_API_KEY")
ELEVEN_API_KEY = st.secrets.get("ELEVEN_API_KEY") or os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = st.secrets.get("ELEVEN_VOICE_ID") or os.getenv("ELEVEN_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

DB_PATH = "study_buddy.db"

# ---------- Initialize Mistral client (if available) ----------
mistral_client = None
if Mistral and MISTRAL_API_KEY:
    try:
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    except Exception as e:
        mistral_client = None

# ---------- DB helpers ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT,
        language TEXT DEFAULT 'English'
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS searches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        topic TEXT,
        lang TEXT,
        created_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS quizzes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        topic TEXT,
        quiz_text TEXT,
        created_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS quiz_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        quiz_id INTEGER,
        score INTEGER,
        total INTEGER,
        details TEXT,
        created_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS achievements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        name TEXT,
        details TEXT,
        created_at TEXT
    )""")
    conn.commit()
    conn.close()

def save_search(topic, lang=None, user_id=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO searches (user_id, topic, lang, created_at) VALUES (?, ?, ?, ?)",
                    (user_id, topic, lang, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
    except Exception:
        pass

def save_quiz_raw(topic, raw_text, user_id=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO quizzes (user_id, topic, quiz_text, created_at) VALUES (?, ?, ?, ?)",
                    (user_id, topic, raw_text, datetime.utcnow().isoformat()))
        quiz_id = cur.lastrowid
        conn.commit()
        conn.close()
        return quiz_id
    except Exception:
        return None

def save_quiz_result(user_id, quiz_id, score, total, details):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO quiz_results (user_id, quiz_id, score, total, details, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, quiz_id, score, total, json.dumps(details), datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
    except Exception:
        pass

# ---------- Utility: extract text from Mistral response ----------
def extract_text_from_mistral_response(resp):
    try:
        return resp.choices[0].message.content
    except Exception:
        try:
            return resp.choices[0]["message"]["content"]
        except Exception:
            return str(resp)

# ---------- Mistral chat wrapper ----------
def mistral_chat(prompt):
    if not mistral_client:
        raise RuntimeError("Mistral client not configured. Set MISTRAL_API_KEY in Streamlit secrets or .env.")
    resp = mistral_client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role":"user", "content": prompt}]
    )
    return extract_text_from_mistral_response(resp)

# ---------- ElevenLabs TTS ----------
def eleven_tts(text: str):
    if not ELEVEN_API_KEY:
        raise RuntimeError("ElevenLabs key not configured. Set ELEVEN_API_KEY in secrets.")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
    payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.7}}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return BytesIO(r.content)

# ---------- YouTube search ----------
def youtube_search_link(query):
    if not YOUTUBE_API_KEY:
        raise RuntimeError("YouTube key not configured. Set YOUTUBE_API_KEY in secrets.")
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {"part":"snippet", "q": f"{query} explanation tutorial", "type":"video", "maxResults":1, "key": YOUTUBE_API_KEY}
    r = requests.get(url, params=params, timeout=8)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        return None
    video_id = items[0]["id"].get("videoId")
    if not video_id:
        return None
    return f"https://www.youtube.com/watch?v={video_id}"

# ---------- Parsing quizzes (robust) ----------
def parse_quiz_text(raw_text):
    raw = str(raw_text).strip()
    # Prefer splitting on "Q)" anchor; fallback to numeric
    if "Q)" in raw:
        blocks = re.split(r"\n(?=Q\))", raw)
    else:
        blocks = re.split(r"\n(?=\d+\))", raw)
    questions = []
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        first = lines[0]
        if first.startswith("Q)"):
            q_text = first[2:].strip()
        else:
            q_text = re.sub(r'^\d+\)\s*', '', first).strip()
        options = {}
        answer_letter = None
        explanation = ""
        for line in lines[1:]:
            m_opt = re.match(r'^([A-D])\)\s*(.+)$', line)
            if m_opt:
                options[m_opt.group(1)] = m_opt.group(2).strip()
                continue
            if line.lower().startswith("answer"):
                try:
                    answer_letter = line.split(":",1)[1].strip().upper()
                except:
                    answer_letter = None
                continue
            if line.lower().startswith("explanation"):
                try:
                    explanation = line.split(":",1)[1].strip()
                except:
                    explanation = ""
                continue
        if len(options) >= 2:  # accept partial if not perfect; app will handle missing
            questions.append({
                "question": q_text,
                "options": options,
                "answer": answer_letter,
                "explanation": explanation
            })
    return questions

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Study Buddy", layout="centered", initial_sidebar_state="auto")

# Some custom CSS for nicer look
st.markdown(
    """
    <style>
    .big-title { font-size:28px; font-weight:700; }
    .muted { color:#6b7280; }
    .card { padding: 16px; border-radius:10px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); margin-bottom:16px; }
    .option-label { font-weight:500; }
    .stButton>button { min-width:110px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Ensure DB present
init_db()

st.markdown("<div class='big-title'>📚 AI Study Buddy</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Ask questions, generate quizzes (5–7 MCQs), get video suggestions, translate (including Urdu), and play read-aloud audio.</div>", unsafe_allow_html=True)
st.write("")

# Input row
col1, col2, col3 = st.columns([6,1,1])
with col1:
    user_input = st.text_input("Ask a question or topic", value="", placeholder="e.g., Newton's laws or Data Science")

with col2:
    ask_btn = st.button("Ask ✨")
with col3:
    mic_btn = st.button("🎤 Voice (browser upload)")

# Buttons row
c1, c2, c3, c4 = st.columns([1.2,1.2,1.2,2.4])
with c1:
    quiz_btn = st.button("📝 Generate Quiz")
with c2:
    video_btn = st.button("🎥 Suggest Video")
with c3:
    t_lang = st.selectbox("Translate to", ["English","Urdu","Spanish","French","German","Chinese","Arabic","Hindi","Japanese"])
with c4:
    translate_btn = st.button("🌐 Translate Answer")

st.write("---")

# Area placeholders
answer_container = st.container()
quiz_container = st.container()
video_container = st.container()

# ---------- ASK ----------
if ask_btn:
    if not user_input.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Asking the tutor..."):
            try:
                prompt = (
                    "You are an expert tutor. Provide a clear, step-by-step, detailed explanation of the question below. "
                    "Use simple language, give examples, list steps if applicable, and end with a 1-2 sentence summary.\n\n"
                    f"Question: {user_input}"
                )
                answer_text = mistral_chat(prompt)
                save_search(user_input)
                st.session_state['last_answer'] = answer_text
                st.session_state['last_question'] = user_input
                with answer_container:
                    st.markdown("#### 💡 Answer")
                    st.write(answer_text)
                    st.download_button("📋 Copy Answer", data=answer_text, file_name="answer.txt")
            except Exception as e:
                st.error(f"Error producing answer: {e}")

# ---------- TRANSLATE ----------
if translate_btn:
    text_to_translate = st.session_state.get('last_answer', "")
    if not text_to_translate:
        st.warning("Get an answer first to translate.")
    else:
        with st.spinner(f"Translating to {t_lang}..."):
            try:
                prompt = f"Translate the following text into {t_lang} (keep meaning intact):\n\n{text_to_translate}"
                translation = mistral_chat(prompt)
                save_search(text_to_translate, lang=t_lang)
                with answer_container:
                    st.markdown(f"#### 🌐 Translated ({t_lang})")
                    st.write(translation)
                    st.download_button("📋 Copy Translation", data=translation, file_name="translation.txt")
            except Exception as e:
                st.error(f"Translation error: {e}")

# ---------- YOUTUBE ----------
if video_btn:
    if not user_input.strip():
        st.warning("Type a topic to find a video.")
    else:
        with st.spinner("Searching YouTube..."):
            try:
                link = youtube_search_link(user_input)
                if not link:
                    st.info("No good video found.")
                else:
                    save_search(user_input)
                    with video_container:
                        st.markdown("#### 🎥 Suggested Video")
                        # embed
                        embed = f'<iframe width="100%" height="400" src="https://www.youtube.com/embed/{link.split("v=")[-1]}" frameborder="0" allowfullscreen></iframe>'
                        st.components.v1.html(embed, height=420)
            except Exception as e:
                st.error(f"YouTube error: {e}")

# ---------- QUIZ ----------
if quiz_btn:
    if not user_input.strip():
        st.warning("Type a topic to generate a quiz.")
    else:
        with st.spinner("Generating quiz..."):
            try:
                # Strong prompt for consistent format
                prompt = (
                    f"Generate 5-7 multiple-choice questions about: {user_input}.\n"
                    "Use this exact block format for each question:\n"
                    "Q) Question text\nA) Option text\nB) Option text\nC) Option text\nD) Option text\nAnswer: <LETTER>\nExplanation: <short explanation>\n\n"
                    "Return nothing else, strictly follow this format."
                )
                raw_quiz = mistral_chat(prompt)
                questions = parse_quiz_text(raw_quiz)

                # If parsing failed to produce >0 questions, attempt a fallback request (shorter)
                if not questions:
                    raw_quiz = mistral_chat(f"Create 5 short MCQs about {user_input} in the Q)/A)-D) format.")
                    questions = parse_quiz_text(raw_quiz)

                # Save raw text to DB (best-effort)
                quiz_id = save_quiz_raw(user_input, raw_quiz)
                st.session_state['current_quiz'] = {"id": quiz_id, "topic": user_input, "raw": raw_quiz, "questions": questions}

                if not questions:
                    st.info("Could not parse quiz. Showing sample fallback.")
                    # build sample
                    questions = []
                    for i in range(5):
                        questions.append({
                            "question": f"Sample Q{i+1} about {user_input}",
                            "options": {"A":"A opt","B":"B opt","C":"C opt","D":"D opt"},
                            "answer":"A","explanation":"Sample explanation."
                        })
                    st.session_state['current_quiz']['questions'] = questions

                # Render interactive quiz
                with quiz_container:
                    st.markdown(f"#### 📝 Quiz — {user_input}")
                    quiz_form = st.form("quiz_form")
                    q_widgets = []
                    for idx, q in enumerate(questions):
                        quiz_form.markdown(f"**{idx+1}) {q['question']}**")
                        opts = q.get("options", {})
                        # ensure deterministic order A-D
                        choices = []
                        labels = []
                        for k in sorted(opts.keys()):
                            choices.append(f"{k}) {opts[k]}")
                            labels.append(k)
                        # if options less than 4, pad
                        if len(choices) < 4:
                            # fill with placeholders
                            for pad in ["(a)","(b)","(c)","(d)"]:
                                if len(choices) >= 4: break
                                choices.append(pad)
                                labels.append("X")
                        sel = quiz_form.radio(label=f"Choose (Q{idx+1})", options=choices, key=f"q_{idx}")
                        # store mapping of selection back to letter on submit
                        q_widgets.append({"sel": sel, "labels": labels, "choices": choices})
                        quiz_form.write("")  # spacing
                    submitted = quiz_form.form_submit_button("Submit Quiz")
                    if submitted:
                        # gather answers
                        answers = []
                        for i, w in enumerate(q_widgets):
                            sel_text = w['sel']
                            # sel_text like "A) Option"
                            m = re.match(r'^([A-D])\)', sel_text)
                            if m:
                                choice_letter = m.group(1)
                            else:
                                # fallback: pick first matching letter in choices
                                choice_letter = None
                                for k in ["A","B","C","D"]:
                                    if sel_text.startswith(f"{k})"):
                                        choice_letter = k
                                        break
                            answers.append({"q_index": i, "choice": choice_letter})
                        # call submit_quiz logic locally (we can grade using parsed questions)
                        correct_data = []
                        for q in questions:
                            correct_data.append((q.get("answer"), q.get("explanation")))

                        total = len(correct_data)
                        score = 0
                        details = []
                        for i, a in enumerate(answers):
                            choice = (a.get("choice") or "").upper()
                            correct, expl = correct_data[i] if i < len(correct_data) else (None, "")
                            ok = (choice == correct)
                            if ok:
                                score += 1
                            details.append({"index": i, "choice": choice, "correct_answer": correct, "ok": ok, "explanation": expl})
                        save_quiz_result(None, quiz_id, score, total, details)
                        st.success(f"Score: {score} / {total}")
                        # show per-question feedback
                        for d in details:
                            if d["ok"]:
                                st.markdown(f"**Q{d['index']+1}: ✅ Correct** — {d.get('explanation','')}")
                            else:
                                st.markdown(f"**Q{d['index']+1}: ❌ Incorrect** — Correct: **{d.get('correct_answer')}** — {d.get('explanation','')}")
            except Exception as e:
                st.error(f"Quiz error: {e}")

# ---------- Read Aloud (ElevenLabs) ----------
st.write("")
st.markdown("----")
st.markdown("### 🔊 Read Aloud (ElevenLabs TTS)")
ta = st.text_area("Text to speak (or press 'Use last answer')", value=st.session_state.get('last_answer',''), height=120)
if st.button("Use last answer"):
    ta = st.session_state.get('last_answer','')
    st.experimental_rerun()
col1, col2 = st.columns([1,3])
with col1:
    if st.button("🔊 Read Aloud"):
        if not ta.strip():
            st.warning("Enter text to speak.")
        else:
            try:
                audio = eleven_tts(ta)
                st.audio(audio.read(), format="audio/mp3")
            except Exception as e:
                st.error(f"TTS error: {e}")
with col2:
    st.write("ElevenLabs key required for audio. If you don't have it you can skip this step.")

st.write("")
st.markdown("---")
st.caption("Pro tip: If Mistral output is oddly formatted, copy the raw quiz shown in DB or raw field and paste it into the prompt tuning area for easier debugging.")
