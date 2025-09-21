# streamlit_app.py
import streamlit as st
import os
import json
import requests
import datetime
from pathlib import Path
import io
import re

# Optional plotting
import matplotlib.pyplot as plt

# OCR dependencies (optional - app will run without them but doc QA/OCR won't)
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

# Try PDF libraries for notes generation (optional)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# --------- Configuration (use st.secrets or environment variables) ----------
MISTRAL_API_KEY = None
try:
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
except Exception:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

MISTRAL_MODEL = None
try:
    MISTRAL_MODEL = st.secrets.get("MISTRAL_MODEL")
except Exception:
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small")

YOUTUBE_API_KEY = None
try:
    YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY")
except Exception:
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

PROGRESS_FILE = Path("progress.json")

# --------- Try to import the mistralai client (if installed) ----------
mistral_client = None
try:
    from mistralai import Mistral
    if MISTRAL_API_KEY:
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
except Exception:
    mistral_client = None

# --------- Helpers: AI calls and parsing ---------------------------------
def extract_text_from_mistral_response(resp):
    """
    Robustly extract text from various Mistral response shapes.
    Returns a string.
    """
    try:
        # object-like
        return resp.choices[0].message.content
    except Exception:
        try:
            # dict-like
            return resp.choices[0]["message"]["content"]
        except Exception:
            return str(resp)

def extract_json_from_text(text: str) -> str:
    """
    Extract first JSON array/object from a possibly messy model output.
    Returns JSON text (string) or raises ValueError.
    """
    if not text:
        raise ValueError("No text to parse")
    s = text.strip()
    # quick pass - if already JSON
    if s.startswith("{") or s.startswith("["):
        return s
    # try balanced array
    start = s.find("[")
    if start != -1:
        stack = 0
        for i in range(start, len(s)):
            if s[i] == "[":
                stack += 1
            elif s[i] == "]":
                stack -= 1
                if stack == 0:
                    return s[start:i+1]
    # try balanced object
    start = s.find("{")
    if start != -1:
        stack = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                stack += 1
            elif s[i] == "}":
                stack -= 1
                if stack == 0:
                    return s[start:i+1]
    # fallback first [ .. last ]
    a = s.find("[")
    b = s.rfind("]")
    if a != -1 and b != -1 and b > a:
        return s[a:b+1]
    raise ValueError("Couldn't extract JSON from model output")

def call_mistral_for_explanation(topic: str) -> str:
    """Ask Mistral for a structured explanation."""
    if not mistral_client:
        return ("⚠ Mistral client not configured. Please set MISTRAL_API_KEY in Streamlit secrets or environment.")
    prompt = (
        "You are an expert tutor. Provide a thorough, structured explanation for the topic below. "
        "Structure the response with: a short definition, 3–6 key points (bulleted), an example, and a 1-2 sentence summary. "
        "Use clear, student-friendly language.\n\n"
        f"Topic: {topic}"
    )
    try:
        resp = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_text_from_mistral_response(resp)
    except Exception as e:
        return f"⚠ Error while calling Mistral: {e}"

def call_mistral_for_quiz(topic: str, count=10):
    """
    Request exactly count MCQs in JSON array format.
    Each element: {question, options (4), answer_index (0-3), explanation}
    """
    if not mistral_client:
        return None, "Mistral client not configured. Set MISTRAL_API_KEY."
    prompt = (
        f"Create exactly {count} multiple-choice questions about '{topic}'.\n"
        "Return ONLY a JSON array. Each element must be an object with these fields:\n"
        "- question: a single concise question string\n"
        "- options: an array of exactly 4 option strings\n"
        "- answer_index: the index (0-3) of the correct option\n"
        "- explanation: a short explanation for the correct answer\n\n"
        "Important: Output must be valid JSON (no surrounding commentary). Example element:\n"
        '{"question":"What is 2+2?","options":["1","2","3","4"],"answer_index":3,"explanation":"2+2 equals 4."}\n\n'
        "Now produce the JSON array."
    )
    try:
        resp = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = extract_text_from_mistral_response(resp)
        json_text = extract_json_from_text(text)
        quiz = json.loads(json_text)
        normalized = []
        for item in quiz:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            options = item.get("options", []) or []
            explanation = str(item.get("explanation", "")).strip()
            try:
                answer_index = int(item.get("answer_index", 0))
            except Exception:
                answer_index = 0
            # Coerce options to strings & ensure 4
            options = [str(o) for o in options][:4]
            while len(options) < 4:
                options.append("N/A")
            answer_index = max(0, min(3, answer_index))
            normalized.append({
                "question": question,
                "options": options,
                "answer_index": answer_index,
                "explanation": explanation
            })
        # final sanity: if not enough questions, return error
        if len(normalized) < 1:
            return None, "Model returned no valid quiz items."
        return normalized, None
    except Exception as e:
        return None, f"Error parsing Mistral quiz output: {e}"

def call_mistral_for_flashcards(text: str, count=10):
    """Generate concise flashcards (Q/A) from explanation text."""
    if not mistral_client:
        return None, "Mistral client not configured."
    prompt = (
        f"From the explanation below, create up to {count} concise flashcards in JSON array form. "
        "Each item should be an object {\"q\":\"question\",\"a\":\"answer\"}. Return only JSON.\n\n"
        f"Explanation:\n{text}"
    )
    try:
        resp = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = extract_text_from_mistral_response(resp)
        json_text = extract_json_from_text(text)
        cards = json.loads(json_text)
        out = []
        for c in cards:
            if isinstance(c, dict):
                q = str(c.get("q") or c.get("question") or "").strip()
                a = str(c.get("a") or c.get("answer") or "").strip()
                if q and a:
                    out.append({"q": q, "a": a})
        if not out:
            return None, "No flashcards parsed from model output."
        return out, None
    except Exception as e:
        return None, f"Error generating flashcards: {e}"

def call_mistral_for_document_qa(document_text: str, question: str):
    """Answer a question using the provided document text."""
    if not mistral_client:
        return None, "Mistral client not configured."
    prompt = (
        "You are a helpful tutor. Use the document content below to answer the user's question concisely. "
        "If the answer is not in the document, say you can't find it in the provided text.\n\n"
        "DOCUMENT:\n"
        f"{document_text[:15000]}\n\n"  # cap size to avoid huge prompts
        "QUESTION:\n"
        f"{question}\n\n"
        "Answer briefly and list any short pointers if helpful."
    )
    try:
        resp = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role":"user","content":prompt}]
        )
        return extract_text_from_mistral_response(resp), None
    except Exception as e:
        return None, f"Error during document QA: {e}"

# --------- New: call Mistral for Q/A pairs (2-3 line answers) ----------
def call_mistral_for_qapairs(topic: str, count=10):
    """
    Ask the model to return exactly `count` question-answer pairs in JSON array form:
    [{ "q": "...", "a": "..." }, ...]
    The answer 'a' should be 2-3 short sentences (concise).
    """
    if not mistral_client:
        return None, "Mistral client not configured."
    prompt = (
        f"Create exactly {count} concise question-and-answer pairs about '{topic}'.\n"
        "Return ONLY a JSON array. Each element must be an object with these fields:\n"
        "- q: a single concise question string\n"
        "- a: a short answer consisting of 2-3 short sentences (aim for 2-3 lines)\n\n"
        "Important: Output must be valid JSON (no surrounding commentary). Example element:\n"
        '{"q":"What is photosynthesis?","a":"Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs in chloroplasts and produces oxygen as a byproduct."}\n\n'
        "Now produce the JSON array."
    )
    try:
        resp = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role":"user","content":prompt}]
        )
        text = extract_text_from_mistral_response(resp)
        json_text = extract_json_from_text(text)
        arr = json.loads(json_text)
        qas = []
        for item in arr:
            if isinstance(item, dict):
                q = str(item.get("q") or item.get("question") or "").strip()
                a = str(item.get("a") or item.get("answer") or "").strip()
                if q and a:
                    # Clean up whitespace
                    a = re.sub(r'\s+', ' ', a).strip()
                    qas.append({"q": q, "a": a})
        if not qas:
            return None, "Model returned no Q/A pairs."
        return qas, None
    except Exception as e:
        return None, f"Error generating Q/A pairs: {e}"

# --------- New: generate short study Q&A for Notes (returns list of formatted strings) ----------
def generate_short_questions(topic: str, count=10):
    """
    Attempt to generate `count` short study question+answer strings for the given topic.
    Prefers Mistral Q/A pairs (uses call_mistral_for_qapairs); if that fails,
    falls back to using last explanation or naive generation.
    Returns (list_of_formatted_strings, error_or_None)
    Each formatted string: "<question>\n\nAnswer: <answer>"
    """
    # If mistral available, request Q/A pairs
    if mistral_client:
        qas, err = call_mistral_for_qapairs(topic, count=count)
        if qas and not err:
            out = []
            for qa in qas[:count]:
                # Ensure answers are 2-3 sentences long (trim if needed)
                ans = qa["a"]
                # split into sentences (simple split)
                sentences = re.split(r'(?<=[.!?])\s+', ans)
                ans_short = " ".join(sentences[:3]).strip()
                out.append(f"{qa['q']}\n\nAnswer: {ans_short}")
            while len(out) < count:
                out.append(f"Describe a key point about {topic}.\n\nAnswer: {topic} is important because ...")
            return out, None
        # otherwise fall through to fallback

    # Fallback: try to form Q&A from last_explanation in session_state
    explanation = st.session_state.get("last_explanation", "").strip()
    if explanation:
        lines = [l.strip() for l in explanation.split("\n") if l.strip()]
        qas = []
        for ln in lines:
            s = ln.split(".")[0].strip()
            if not s:
                continue
            if len(s.split()) <= 20:
                q = f"What is {s}?"
            else:
                q = f"Explain briefly: {s}"
            # Make a 2-3 line-ish answer: use the sentence + a short follow-up
            first_sent = s if len(s) < 200 else s[:200].rsplit(' ', 1)[0] + "..."
            answer = f"{first_sent}. In short, {topic} involves this concept and is useful in related contexts."
            qas.append(f"{q}\n\nAnswer: {answer}")
            if len(qas) >= count:
                break
        while len(qas) < count:
            qas.append(f"Describe a key point about {topic}.\n\nAnswer: {topic} is important because it helps with ...")
        return qas, None

    # Final fallback: generic Q&A
    out = []
    for i in range(count):
        out.append(f"Question {i+1}. Write a short explanation about {topic}.\n\nAnswer: {topic} can be summarized as follows in a couple of sentences.")
    return out, None

# --------- PDF/Notes generation ----------
def generate_notes_pdf_bytes(topic: str, items: list):
    """
    Generate a PDF (bytes) containing question + answers.
    items may be:
      - list of strings formatted as "Question...\n\nAnswer: ..."
      - list of dicts {"q": "...", "a": "..."}
    Tries reportlab, then fpdf. If neither available, returns txt fallback.
    Returns (bytes, filename, is_pdf_bool, warning_msg)
    """
    title = f"Study Notes - {topic}"
    sanitized_topic = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in topic)[:80].strip()
    filename_pdf = f"{sanitized_topic}_notes.pdf"

    # helper to normalize each item into (q, a)
    def _normalize(it):
        if isinstance(it, dict):
            q = str(it.get("q", "")).strip()
            a = str(it.get("a", "")).strip()
            return q, a
        s = str(it)
        # try split by 'Answer:' or double newline
        parts = re.split(r'\n\s*\nAnswer:\s*', s, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            q = parts[0].strip()
            a = parts[1].strip()
            return q, a
        # fallback: split first sentence as question, rest as answer
        lines = s.splitlines()
        if len(lines) >= 2:
            q = lines[0].strip()
            a = " ".join([l.strip() for l in lines[1:]])
            return q, a
        return s, ""

    # text wrapping helper
    def _wrap_text(text, max_chars=95):
        words = text.split()
        lines = []
        current = []
        cur_len = 0
        for w in words:
            if cur_len + len(w) + (1 if cur_len>0 else 0) > max_chars:
                lines.append(" ".join(current))
                current = [w]
                cur_len = len(w)
            else:
                current.append(w)
                cur_len += (len(w) + (1 if cur_len>0 else 0))
        if current:
            lines.append(" ".join(current))
        return lines

    # Try reportlab
    if REPORTLAB_AVAILABLE:
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        margin = 50
        y = height - margin
        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y, title)
        y -= 28
        c.setFont("Helvetica-Bold", 12)

        for i, it in enumerate(items, start=1):
            q, a = _normalize(it)
            if not q and not a:
                continue
            # Question line(s)
            qtext = f"{i}. {q}"
            q_lines = _wrap_text(qtext, max_chars=95)
            for line in q_lines:
                if y < margin + 40:
                    c.showPage()
                    y = height - margin
                    c.setFont("Helvetica-Bold", 12)
                c.drawString(margin, y, line)
                y -= 14
            # Answer in normal font with indent
            a_lines = []
            # split answer into sentences for nicer line breaks, keep max 3 sentences
            sentences = re.split(r'(?<=[.!?])\s+', a)
            a_short = " ".join(sentences[:3]).strip()
            a_lines = _wrap_text(a_short, max_chars=85)
            c.setFont("Helvetica", 11)
            for line in a_lines:
                if y < margin + 40:
                    c.showPage()
                    y = height - margin
                    c.setFont("Helvetica", 11)
                c.drawString(margin + 18, y, line)
                y -= 13
            y -= 8
            c.setFont("Helvetica-Bold", 12)

        c.save()
        buffer.seek(0)
        return buffer.read(), filename_pdf, True, None

    # Try FPDF
    if FPDF_AVAILABLE:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.ln(4)
        pdf.set_font("Arial", size=12)
        for i, it in enumerate(items, start=1):
            q, a = _normalize(it)
            if not q and not a:
                continue
            pdf.multi_cell(0, 8, f"{i}. {q}")
            # Answer: keep up to 3 sentences
            sentences = re.split(r'(?<=[.!?])\s+', a)
            a_short = " ".join(sentences[:3]).strip()
            pdf.multi_cell(0, 7, f"Answer: {a_short}")
            pdf.ln(2)
        buffer = io.BytesIO()
        buffer.write(pdf.output(dest="S").encode("latin-1"))
        buffer.seek(0)
        return buffer.read(), filename_pdf, True, None

    # Neither library available: fallback to txt bytes
    txt_fname = f"{sanitized_topic}_notes.txt"
    text_content = title + "\n\n"
    for i, it in enumerate(items, start=1):
        q, a = _normalize(it)
        if not q and not a:
            continue
        sentences = re.split(r'(?<=[.!?])\s+', a)
        a_short = " ".join(sentences[:3]).strip()
        text_content += f"{i}. {q}\n\nAnswer: {a_short}\n\n"
    return text_content.encode("utf-8"), txt_fname, False, "No PDF library (reportlab/fpdf) installed. Provided a .txt fallback."

# --------- YouTube helpers ------------------------------------------------
def fetch_youtube_videos(query: str, max_results=3):
    if not YOUTUBE_API_KEY:
        return None, "No YOUTUBE_API_KEY configured; showing fallback search link."
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": f"{query} explanation tutorial",
        "type": "video",
        "maxResults": max_results,
        "key": YOUTUBE_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        items = r.json().get("items", [])
        videos = []
        for it in items:
            vid = it.get("id", {}).get("videoId")
            title = it.get("snippet", {}).get("title")
            if vid:
                videos.append({"id": vid, "title": title or "Untitled"})
        return videos, None
    except Exception as e:
        return None, f"YouTube API error: {e}"

# --------- OCR helpers ----------------------------------------------------
def extract_text_from_uploaded_file(uploaded) -> (str, str):
    """
    Given a Streamlit UploadedFile, extract text.
    Returns (text, error_message_or_None).
    Supports PDF (pdfplumber) and images (PIL + pytesseract).
    """
    if not uploaded:
        return "", "No file."
    name = uploaded.name.lower()
    try:
        if name.endswith(".pdf"):
            if pdfplumber is None:
                return "", "pdfplumber not installed. Install with: pip install pdfplumber"
            # ensure we read bytes/stream
            with pdfplumber.open(io.BytesIO(uploaded.read())) as pdf:
                texts = []
                for p in pdf.pages:
                    txt = p.extract_text()
                    if txt:
                        texts.append(txt)
                return "\n\n".join(texts), None
        else:
            # treat as image
            if Image is None or pytesseract is None:
                return "", "Pillow or pytesseract not installed. Install with: pip install pillow pytesseract and install the Tesseract binary."
            uploaded.seek(0)
            img = Image.open(io.BytesIO(uploaded.read()))
            try:
                img = img.convert("RGB")
            except Exception:
                pass
            try:
                text = pytesseract.image_to_string(img)
            except Exception as e:
                return "", f"pytesseract OCR failed: {e}"
            return text, None
    except Exception as e:
        return "", f"OCR extraction failed: {e}"

# --------- Progress persistence ------------------------------------------
def load_progress():
    """Load progress file and return a list (safe)."""
    if PROGRESS_FILE.exists():
        try:
            raw = PROGRESS_FILE.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                if "entries" in parsed and isinstance(parsed["entries"], list):
                    return parsed["entries"]
                for v in parsed.values():
                    if isinstance(v, list):
                        return v
                return [parsed]
        except Exception:
            return []
    return []

def save_progress_entry(entry: dict):
    """Append an entry to the progress file safely."""
    data = load_progress()
    if not isinstance(data, list):
        data = []
    data.append(entry)
    try:
        PROGRESS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        # if saving fails, don't crash the app
        st.error(f"Failed to save progress: {e}")

# --------- UI / Streamlit App -------------------------------------------
st.set_page_config(page_title="AI Study Buddy", page_icon="📚", layout="wide")

# CSS: gradient background, banner, card, buttons with hover/glow
st.markdown(
    """
    <style>
      :root {
        --accent1: #6ea8fe;
        --accent2: #b388ff;
        --card-bg: rgba(255,255,255,0.9);
      }
      html, body, [data-testid="stAppViewContainer"] > div:first-child {
        background: linear-gradient(180deg, #f2f8ff 0%, #fff8ff 100%);
      }
      .banner {
        background: linear-gradient(90deg, rgba(110,168,254,0.12), rgba(179,136,255,0.12));
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 14px;
        backdrop-filter: blur(6px);
        box-shadow: 0 6px 20px rgba(30,40,60,0.06);
        animation: fadeIn 0.9s ease;
      }
      .banner h1 { margin: 0; font-size: 28px; }
      .banner p { margin: 4px 0 0 0; color: #374151; font-size: 15px; }
      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
      }

      .card {
        background: var(--card-bg);
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(30,40,60,0.06);
        margin-bottom: 14px;
      }

      /* buttons row container */
      .btn-row { display:flex; gap:12px; justify-content:center; margin-top:8px; margin-bottom:12px; flex-wrap:wrap; }
      .app-btn {
        border: none;
        padding: 10px 16px;
        border-radius: 10px;
        font-weight: 700;
        cursor: pointer;
        color: white;
        box-shadow: 0 6px 18px rgba(20,30,60,0.08);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
        display:inline-flex; align-items:center; gap:8px;
        font-size:15px;
      }
      .app-btn:hover { transform: translateY(-4px); box-shadow: 0 12px 34px rgba(20,30,60,0.12); }
      .btn-primary { background: linear-gradient(90deg, #5b8cff, #3bb0ff); }
      .btn-quiz { background: linear-gradient(90deg, #7c5cff, #a15bff); }
      .btn-cards { background: linear-gradient(90deg, #34d399, #10b981); }
      .btn-video { background: linear-gradient(90deg, #f59e0b, #fb923c); }
      .btn-progress { background: linear-gradient(90deg, #ff7b7b, #ff4d4d); }

      /* small icon in button */
      .app-btn .icon { font-size: 18px; }

      /* ensure radio choices and other elements wrap nicely */
      .stRadio > div { gap: 8px; }

      /* make the search input wider and centered */
      .search-row { display:flex; justify-content:center; gap:12px; align-items:center; margin-bottom:6px; }
      .search-box { width: 72%; min-width: 320px; }

      /* table responsive */
      .stTable td, .stTable th { white-space: normal; }

    </style>
    """,
    unsafe_allow_html=True,
)

# Welcome banner - pick one of the taglines randomly per session
import random
taglines = [
    ("Welcome to AI Study Buddy – Your Personal Learning Partner 🤖📘", "Ready to learn something new today? 🚀"),
    ("Welcome back, Student 👋", "Let's explore a topic or test yourself!"),
    ("Hello! Ready to level up your learning? ✨", "Ask a question or upload a document to get started.")
]
if "banner_choice" not in st.session_state:
    st.session_state["banner_choice"] = random.choice(taglines)

banner_main, banner_sub = st.session_state["banner_choice"]
st.markdown(f'<div class="banner"><h1>{banner_main}</h1><p>{banner_sub}</p></div>', unsafe_allow_html=True)

# Session state defaults
if "last_topic" not in st.session_state:
    st.session_state["last_topic"] = ""
if "last_explanation" not in st.session_state:
    st.session_state["last_explanation"] = ""
if "last_quiz" not in st.session_state:
    st.session_state["last_quiz"] = None
if "last_quiz_id" not in st.session_state:
    st.session_state["last_quiz_id"] = None
if "last_flashcards" not in st.session_state:
    st.session_state["last_flashcards"] = None
if "last_study_plan" not in st.session_state:
    st.session_state["last_study_plan"] = None
if "flash_idx" not in st.session_state:
    st.session_state["flash_idx"] = 0
if "uploaded_doc_text" not in st.session_state:
    st.session_state["uploaded_doc_text"] = ""

# ---- Top search row (centered) ----
st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    search_col1, search_col2 = st.columns([4,1])
    with search_col1:
        topic_input = st.text_input("🔎 Enter a topic or question:", value="", key="topic_input", placeholder="e.g., 'Newton's laws', 'binary search', 'photosynthesis' ", help="Type a focused topic or question.")
    with search_col2:
        # small Ask AI button sits beside input (keeps old behavior)
        ask_small_clicked = st.button("📘 Ask AI", key="ask_ai_small")
st.markdown('</div>', unsafe_allow_html=True)

# ---- Buttons row under search bar: use real Streamlit buttons (no hidden buttons) ----
st.markdown('<div class="btn-row">', unsafe_allow_html=True)
# Use columns to layout the main functional buttons (now 6: Ask, Quiz, Flashcards, Videos, Progress, Notes)
btn_cols = st.columns([1,1,1,1,1,1])
with btn_cols[0]:
    ask_clicked = st.button("📘 Ask AI", key="ask_ai_top")
with btn_cols[1]:
    quiz_clicked = st.button("📝 Quiz", key="btn_quiz")
with btn_cols[2]:
    flashcards_clicked = st.button("🃏 Flashcards", key="btn_flash")
with btn_cols[3]:
    video_clicked = st.button("🎥 Videos", key="btn_video")
with btn_cols[4]:
    progress_clicked = st.button("📊 Progress", key="btn_progress")
with btn_cols[5]:
    notes_clicked = st.button("🗒️ Notes (PDF)", key="btn_notes")
st.markdown('</div>', unsafe_allow_html=True)

# Right-side area for uploads (kept in a card below)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📂 Upload document / image for Q&A (optional)")
uploaded_file = st.file_uploader("Upload a PDF or image (png/jpg/tiff) for document Q&A:", type=["pdf", "png", "jpg", "jpeg", "tiff"])
if uploaded_file:
    txt, err = extract_text_from_uploaded_file(uploaded_file)
    if err:
        st.warning(err)
    else:
        st.session_state["uploaded_doc_text"] = txt
        st.success("Document text extracted — expand to preview.")
        with st.expander("Preview extracted text"):
            st.text_area("Document text (truncated)", txt[:3000], height=250)
st.markdown('</div>', unsafe_allow_html=True)

# ---- Ask AI (explanation) ----
# If small ask button or top ask button clicked, trigger explanation immediately
if ask_small_clicked or ask_clicked:
    topic = (topic_input or "").strip()
    if not topic:
        st.warning("Please type a topic or question before clicking Ask AI.")
    else:
        # set last topic and placeholder, then call synchronously (no experimental_rerun)
        st.session_state["last_topic"] = topic
        st.session_state["last_explanation"] = "⏳ Asking AI for a detailed explanation..."
        st.session_state["last_quiz"] = None
        st.session_state["last_quiz_id"] = None
        st.session_state["last_flashcards"] = None
        # call Mistral directly
        st.session_state["last_explanation"] = call_mistral_for_explanation(topic)

# Display explanation in a card
if st.session_state.get("last_explanation"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"📘 Explanation — {st.session_state.get('last_topic','')}")
    st.markdown(st.session_state["last_explanation"], unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Notes (PDF) flow ----
if notes_clicked:
    topic = st.session_state.get("last_topic", "").strip() or topic_input.strip()
    if not topic:
        st.warning("Please provide a topic (type it in the box) before generating notes.")
    else:
        with st.spinner("Generating study questions and creating PDF..."):
            questions, err = generate_short_questions(topic, count=10)
            if err:
                st.error(err)
            else:
                pdf_bytes, fname, is_pdf, warn = generate_notes_pdf_bytes(topic, questions)
                if warn:
                    st.warning(warn)
                # Use st.download_button to offer file
                if is_pdf:
                    st.success("Notes ready — download the PDF below.")
                    st.download_button(label="⬇️ Download Notes (PDF)", data=pdf_bytes, file_name=fname, mime="application/pdf")
                else:
                    # fallback txt
                    st.success("Notes ready — PDF library not available, download plain text file below.")
                    st.download_button(label="⬇️ Download Notes (TXT)", data=pdf_bytes, file_name=fname, mime="text/plain")

        # Optionally show preview of generated questions + answers
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Generated Study Questions")
        for i, q in enumerate(questions, start=1):
            # q is formatted as "Question text\n\nAnswer: ...", show nicely:
            parts = re.split(r'\n\s*\nAnswer:\s*', q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                st.markdown(f"**{i}. {parts[0].strip()}**")
                st.markdown(f"- **Answer:** {parts[1].strip()}")
            else:
                st.markdown(f"**{i}. {q}**")
        st.markdown('</div>', unsafe_allow_html=True)

# ---- QUIZ generation flow ----
if quiz_clicked:
    topic = st.session_state.get("last_topic", "").strip() or topic_input.strip()
    if not topic:
        st.warning("Ask a topic first (Ask AI) or type a topic in the box.")
    else:
        # request quiz (10 questions)
        st.session_state["last_quiz"] = None
        st.session_state["last_quiz_id"] = None
        quiz, err = call_mistral_for_quiz(topic, count=10)
        if err:
            st.error(err)
        else:
            st.session_state["last_quiz"] = quiz
            # set id using datetime correctly
            try:
                st.session_state["last_quiz_id"] = int(datetime.datetime.utcnow().timestamp())
            except Exception:
                st.session_state["last_quiz_id"] = int(datetime.datetime.now().timestamp())

# Render quiz UI if present
if st.session_state.get("last_quiz"):
    quiz = st.session_state["last_quiz"]
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Quiz — Answer the questions and click Submit")
    with st.form("quiz_form"):
        user_answers = {}
        for i, q in enumerate(quiz):
            st.markdown(f"**Q{i+1}. {q.get('question','(no question)')}**")
            labels = q.get("options", [])
            # display radio options
            sel = st.radio("", labels, key=f"q_{i}")
            try:
                selected_index = labels.index(sel)
            except Exception:
                selected_index = None
            user_answers[i] = selected_index
        submitted = st.form_submit_button("Submit Quiz")
    if submitted:
        total = len(quiz)
        correct_count = 0
        details = []
        for i, q in enumerate(quiz):
            correct_idx = int(q.get("answer_index", 0))
            chosen = user_answers.get(i)
            ok = (chosen == correct_idx)
            if ok:
                correct_count += 1
            chosen_text = q["options"][chosen] if (chosen is not None and 0 <= chosen < len(q["options"])) else None
            correct_text = q["options"][correct_idx] if 0 <= correct_idx < len(q["options"]) else None
            details.append({
                "question": q.get("question", ""),
                "chosen_index": chosen,
                "chosen_text": chosen_text,
                "correct_index": correct_idx,
                "correct_text": correct_text,
                "ok": ok,
                "explanation": q.get("explanation", "")
            })
        pct = int((correct_count / total) * 100) if total else 0
        st.success(f"✅ You scored {correct_count} out of {total} ({pct}%)")
        for idx, d in enumerate(details):
            st.markdown(f"**Q{idx+1}. {d['question']}**")
            if d["ok"]:
                st.markdown(f"- ✅ Your answer: **{d['chosen_text']}**")
            else:
                st.markdown(f"- ❌ Your answer: **{d['chosen_text'] or 'No answer'}** — Correct: **{d['correct_text']}**")
            if d["explanation"]:
                st.info(f"Explanation: {d['explanation']}")
        # save progress
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "topic": st.session_state.get("last_topic", ""),
            "score": correct_count,
            "total": total,
            "details": details
        }
        try:
            save_progress_entry(entry)
            st.success("Progress saved.")
        except Exception as e:
            st.error(f"Failed to save progress locally: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Flashcards flow ----
if flashcards_clicked:
    explanation = st.session_state.get("last_explanation", "").strip()
    if not explanation or explanation.startswith("⚠"):
        st.warning("Get a valid AI explanation first (Ask AI) to create flashcards.")
    else:
        st.session_state["last_flashcards"] = None
        cards, err = call_mistral_for_flashcards(explanation, count=12)
        if err:
            st.error(err)
        else:
            st.session_state["last_flashcards"] = cards

if st.session_state.get("last_flashcards"):
    cards = st.session_state["last_flashcards"]
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🃏 Flashcards — Flip to see answers")
    if cards:
        idx = st.session_state.get("flash_idx", 0)
        c = cards[idx]
        st.markdown(f"**Q{idx+1}. {c['q']}**")
        if st.button("Show Answer"):
            st.info(c["a"])
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("◀ Prev"):
                st.session_state["flash_idx"] = max(0, idx - 1)
        with col2:
            if st.button("Next ▶"):
                st.session_state["flash_idx"] = min(len(cards) - 1, idx + 1)
    else:
        st.info("No flashcards generated.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Document Q&A flow (if file uploaded) ----
if st.session_state.get("uploaded_doc_text"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📄 Document Q&A")
    st.write("Ask a question about the uploaded document/image content (OCR).")
    doc_question = st.text_input("Question about document:", key="doc_question")
    if st.button("Ask Document", key="ask_doc"):
        if not doc_question.strip():
            st.warning("Type a question first.")
        else:
            answer, err = call_mistral_for_document_qa(st.session_state["uploaded_doc_text"], doc_question)
            if err:
                st.error(err)
            else:
                st.markdown("**Answer:**")
                st.markdown(answer)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Video suggestions ----
if video_clicked:
    topic = st.session_state.get("last_topic", "").strip() or topic_input.strip()
    if not topic:
        st.warning("Please enter a topic or ask AI first.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎥 Video Suggestions")
        videos, err = fetch_youtube_videos(topic, max_results=4)
        if err and videos is None:
            st.warning(err)
            q = requests.utils.quote(topic + " explanation")
            st.markdown(f"- [Search YouTube for '{topic} explanation'](https://www.youtube.com/results?search_query={q})")
        else:
            if not videos:
                st.info("No videos found. Try again later.")
            else:
                for v in videos:
                    st.markdown(f"**{v['title']}**")
                    st.video(f"https://www.youtube.com/watch?v={v['id']}")
        st.markdown('</div>', unsafe_allow_html=True)

# ---- Progress dashboard ----
if progress_clicked:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Progress Dashboard")
    data = load_progress()
    if not data:
        st.info("No quizzes taken yet. Generate a quiz and submit to see progress here.")
    else:
        # show table (latest first)
        display_rows = []
        for rec in reversed(data):
            if not isinstance(rec, dict):
                rec = {"timestamp": str(rec), "topic": "", "score": 0, "total": 0}
            display_rows.append({
                "timestamp": rec.get("timestamp", ""),
                "topic": rec.get("topic", ""),
                "score": rec.get("score", 0),
                "total": rec.get("total", 0),
            })
        st.table(display_rows)

        # plotting simple metrics
        last_n = display_rows[:20][::-1]
        attempts = list(range(1, len(last_n) + 1))
        percents = []
        scores = []
        totals = []
        for r in last_n:
            s = r.get("score", 0)
            t = r.get("total", 0) or 1
            scores.append(s)
            totals.append(t)
            percents.append(int((s / t) * 100))
        if attempts:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(attempts, percents, marker="o", linestyle="-")
            ax.set_xlabel("Attempt #")
            ax.set_ylabel("Score (%)")
            ax.set_title("Score over Attempts")
            ax.set_xticks(attempts)
            st.pyplot(fig)

            total_correct = sum(scores)
            total_questions = sum(totals)
            total_wrong = total_questions - total_correct
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.pie([total_correct, total_wrong], labels=["Correct", "Wrong"], autopct="%1.1f%%")
            ax2.set_title("Overall Performance (all quizzes)")
            st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- End of app ----
st.markdown("---")
st.caption("Study Buddy — built with ❤")
