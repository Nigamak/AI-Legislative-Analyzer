# ⚖️ AI Legislative Analyzer — Citizen's Dashboard

> **Hackathon Project 3** · Token Compression · Information Density · Indian Law  
> Making Indian parliamentary bills and legal documents understandable for every citizen.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [Token Compression Pipeline](#token-compression-pipeline)
- [Configuration](#configuration)
- [Common Errors & Fixes](#common-errors--fixes)
- [Testing the App](#testing-the-app)

---

## Overview

Indian parliamentary bills and legal documents are dense, verbose, and nearly impossible for the average citizen to understand. This project builds a **Citizen's Dashboard** that:

- Accepts any Indian legal document (PDF or plain text)
- Compresses it using **Token Compression** — a TF-IDF + deduplication technique that strips 50–70% of tokens while retaining maximum information
- Summarises the compressed text using a locally-running **HuggingFace BART model**
- Returns a plain-language summary, key points, and detailed compression statistics

The core innovation is the compression pipeline — documents are filtered *before* reaching the model, directly reducing energy consumption and maximising **Information Density** (value delivered per token consumed).

---

## Features

- 📄 **PDF upload** — drag and drop or browse to upload any legal PDF
- 📝 **Text paste** — paste raw document text directly
- 🗜️ **Token Compression** — TF-IDF extractive compression with boilerplate stripping and deduplication
- 🤖 **Local AI summarisation** — runs entirely on your machine using HuggingFace BART
- 📊 **Compression stats** — see original vs compressed token counts and information density score
- 🔑 **Key points extraction** — auto-extracted bullet points from the summary
- ⚙️ **Adjustable compression ratio** — slider to control how aggressively to compress (10%–90%)
- 🇮🇳 **India-specific boilerplate stripping** — removes legislative formulas like "Be it enacted", Gazette references, etc.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite |
| Backend | FastAPI (Python) |
| AI Model | HuggingFace `facebook/bart-large-cnn` |
| PDF Parsing | pdfplumber |
| Text Processing | NLTK, tiktoken |
| Server | Uvicorn (ASGI) |

---

## Project Structure

```
create_backend/
├── app/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── hf_models.py          # HuggingFace model loader (lazy, cached)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── summarize.py          # POST /api/summarize/text & /pdf-base64
│   │   └── upload.py             # POST /api/upload
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chunking.py           # Sentence-level chunking with stride overlap
│   │   ├── compression.py        # ★ Token Compression (TF-IDF + dedup + boilerplate strip)
│   │   ├── pipeline.py           # Orchestrator: parse → compress → chunk → summarise
│   │   └── summarizer.py         # HF pipeline wrapper + hierarchical re-summarisation
│   └── utils/
│       ├── __init__.py
│       ├── pdf_parser.py         # pdfplumber extraction + header/footer removal
│       └── text_cleaner.py       # Unicode normalisation, artefact removal
├── main.py                       # FastAPI app entry point
├── requirements.txt              # Python dependencies
└── venv/                         # Virtual environment (created during setup)

citizen-dashboard/                # Frontend (create separately)
├── src/
│   └── App.jsx                   # React dashboard (LegislativeAnalyzer.jsx)
├── package.json
└── vite.config.js
```

---

## Prerequisites

Make sure the following are installed on your system before proceeding.

### System Requirements

| Tool | Version | Check |
|---|---|---|
| Python | 3.10 or higher | `python --version` |
| Node.js | 18 or higher | `node --version` |
| npm | 8 or higher | `npm --version` |
| pip | latest | `pip --version` |

### Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Storage | 5 GB free (for model download) | 10 GB |
| GPU | Not required | NVIDIA GPU speeds up inference 5–10× |

---

## Installation

### Backend Setup

**Step 1 — Navigate to your project folder**

```bash
cd create_backend
```

**Step 2 — Create a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

> You should see `(venv)` appear at the start of your terminal prompt.

**Step 3 — Install Python dependencies**

```bash
pip install -r requirements.txt
```

> ⏳ This may take 3–5 minutes. It installs FastAPI, HuggingFace Transformers, PyTorch, pdfplumber, NLTK, and other libraries.

**Step 4 — Download NLTK data** (one-time setup)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

> ✅ You should see `[nltk_data] Downloading package punkt` and `[nltk_data] Downloading package stopwords`.

---

### Frontend Setup

**Step 1 — Create a new React project using Vite**

Open a new terminal window (keep the backend terminal open):

```bash
npm create vite@latest citizen-dashboard -- --template react
cd citizen-dashboard
```

**Step 2 — Replace the default App.jsx**

Copy the contents of `LegislativeAnalyzer.jsx` into `src/App.jsx`:

```bash
# If LegislativeAnalyzer.jsx is in your Downloads folder (Windows):
copy C:\Users\YourName\Downloads\LegislativeAnalyzer.jsx src\App.jsx

# macOS / Linux:
cp ~/Downloads/LegislativeAnalyzer.jsx src/App.jsx
```

Or manually open `src/App.jsx` in VS Code and replace all its contents with the `LegislativeAnalyzer.jsx` code.

**Step 3 — Install Node dependencies**

```bash
npm install
```

---

## Running the Project

You need **two terminals open simultaneously** — one for the backend and one for the frontend.

### Terminal 1 — Start the Backend

```bash
# Navigate to your backend folder
cd create_backend

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Started reloader process using WatchFiles
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

> 🤖 The first time you call the `/summarize` endpoint, HuggingFace will automatically download the BART model (~1.6 GB). This is a one-time download — subsequent runs use the cached model.

### Terminal 2 — Start the Frontend

```bash
# Navigate to your frontend folder
cd citizen-dashboard

# Start the dev server
npm run dev
```

**Expected output:**
```
  VITE v5.x.x  ready in 300 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Open the App

| Service | URL |
|---|---|
| 🖥️ Frontend Dashboard | http://localhost:5173 |
| ⚙️ Backend API | http://localhost:8000 |
| 📚 Auto-generated API Docs | http://localhost:8000/docs |
| 📖 ReDoc API Docs | http://localhost:8000/redoc |

---

## API Reference

### `POST /api/upload`

Upload a PDF or `.txt` file. Returns extracted and cleaned text.

**Request:** `multipart/form-data` with field `file`

**Supported types:** `application/pdf`, `text/plain`

**Max file size:** 50 MB

**Example (curl):**
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@path/to/your/document.pdf"
```

**Response:**
```json
{
  "filename": "draft_bill.pdf",
  "size_mb": 1.2,
  "word_count": 18400,
  "char_count": 112000,
  "text_preview": "THE DIGITAL PERSONAL DATA...",
  "full_text": "..."
}
```

---

### `POST /api/summarize/text`

Summarise plain text directly.

**Request body:**
```json
{
  "text": "Full document text here...",
  "keep_ratio": 0.45
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | string | ✅ Yes | — | Raw document text (min 50 chars) |
| `keep_ratio` | float | No | `0.45` | Compression ratio (0.1 = very compressed, 0.9 = minimal compression) |

**Example (curl):**
```bash
curl -X POST http://localhost:8000/api/summarize/text \
  -H "Content-Type: application/json" \
  -d '{"text": "THE DIGITAL PERSONAL DATA PROTECTION ACT...", "keep_ratio": 0.45}'
```

**Response:**
```json
{
  "summary": "The Digital Personal Data Protection Bill establishes a framework...",
  "key_points": [
    "Citizens have the right to access and erase their personal data.",
    "Data fiduciaries must implement reasonable security safeguards.",
    "A Data Protection Board of India will adjudicate complaints."
  ],
  "word_count": 18400,
  "stats": {
    "original_tokens": 24860,
    "compressed_tokens": 8140,
    "token_reduction_pct": 67.2,
    "char_reduction_pct": 61.8,
    "information_density_score": 0.328
  }
}
```

---

### `POST /api/summarize/pdf-base64`

Summarise a PDF sent as a base64-encoded string.

**Request body:**
```json
{
  "pdf_base64": "<base64 encoded PDF string>",
  "keep_ratio": 0.45
}
```

**Example (Python):**
```python
import base64, requests

with open("document.pdf", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/api/summarize/pdf-base64",
    json={"pdf_base64": b64, "keep_ratio": 0.45}
)
print(response.json())
```

---

## Token Compression Pipeline

This is the core technique required by the problem statement. It enables handling documents exceeding 100k tokens without truncation, while maximising information density.

```
Raw Document (100k+ tokens)
         │
         ▼
┌─────────────────────────────┐
│  Step 1: Boilerplate Strip  │  Removes "Be it enacted", Gazette notices,
│  (compression.py)           │  short-title lines, chapter headings
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Step 2: Deduplication      │  Cosine similarity check — removes sentences
│  (compression.py)           │  that are ≥85% similar to an existing one
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Step 3: TF-IDF Ranking     │  Scores every sentence by term frequency-
│  (compression.py)           │  inverse document frequency, keeps top
│                             │  keep_ratio% most informative sentences
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Step 4: Chunking           │  Splits compressed text into ~900-token
│  (chunking.py)              │  windows with sentence-safe boundaries
│                             │  and stride overlap for context continuity
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Step 5: BART Summarisation │  HuggingFace facebook/bart-large-cnn
│  (summarizer.py)            │  generates a summary per chunk
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Step 6: Hierarchical Merge │  If >1 chunk, chunk summaries are
│  (summarizer.py)            │  combined and re-summarised into one
│                             │  final plain-language summary
└─────────────────────────────┘
         │
         ▼
  Final Summary + Key Points + Stats
```

### Information Density Score

```
Information Density = compressed_tokens / original_tokens
```

A score of `0.33` means only 33% of the original tokens were needed to deliver the same core information. Lower = better efficiency. The system targets below `0.40` on typical Indian legislative documents.

---

## Configuration

### Change the AI Model

Edit `app/models/hf_models.py`, line 10:

```python
# Default — fast, excellent quality, English
MODEL_NAME = "facebook/bart-large-cnn"

# For very long documents (handles up to 16,384 tokens natively)
MODEL_NAME = "allenai/led-base-16384"

# Google's Long-T5 — strong on structured legal text
MODEL_NAME = "google/long-t5-tglobal-base"
```

### Tune Compression Aggressiveness

The `keep_ratio` parameter controls what fraction of sentences survive compression:

| `keep_ratio` | Tokens saved | Best for |
|---|---|---|
| `0.10` – `0.25` | 75–90% | Very large documents, maximum speed |
| `0.35` – `0.50` | 50–65% | **Recommended** — good balance of detail and speed |
| `0.60` – `0.90` | 10–40% | Short documents, high-detail summaries |

---

## Common Errors & Fixes

### `re.error: global flags not at the start of the expression`

**File:** `app/services/compression.py`

**Fix:** Replace the `_BOILERPLATE_PATTERNS` list and `_BOILERPLATE_RE` line with:

```python
_BOILERPLATE_PATTERNS = [
    r"^whereas[\s,]",
    r"be it enacted by",
    r"it is hereby enacted",
    r"passed by both houses",
    r"received the assent of the president",
    r"published in the gazette",
    r"no\.\s+\d+\s+of\s+\d{4}",
    r"the\s+\w+\s+act,?\s+\d{4}$",
    r"arrangement of (?:sections|clauses)",
    r"^chapter\s+[IVXLCDM\d]+\s*$",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE | re.MULTILINE)
```

---

### `ModuleNotFoundError: No module named 'app'`

You are running `uvicorn` from the wrong directory. Always run it from the `create_backend/` root:

```bash
cd create_backend
uvicorn main:app --reload
```

---

### `punkt_tab` not found / NLTK error

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

---

### Model download is very slow

The BART model is ~1.6 GB and downloads only once. It is cached at:
- **Windows:** `C:\Users\YourName\.cache\huggingface\hub`
- **macOS/Linux:** `~/.cache/huggingface/hub`

---

### Frontend shows "Error — make sure the backend is running"

1. Confirm the backend is running on port 8000: visit http://localhost:8000
2. Check that the `API_BASE` constant in `src/App.jsx` matches your backend URL (default: `http://localhost:8000/api`)
3. If running on a different port, update line 3 of `App.jsx`:
   ```javascript
   const API_BASE = "http://localhost:YOUR_PORT/api";
   ```

---

### `torch` installation fails on Windows

Install PyTorch separately before running `pip install -r requirements.txt`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Testing the App

### Quick Text Test

Paste this into the **Paste Text** tab and click **Analyse Document**:

```
THE DIGITAL PERSONAL DATA PROTECTION ACT, 2023 — The Central Government shall establish 
a Data Protection Board of India. Every data fiduciary shall process personal data only 
with the consent of the data principal. Citizens have the right to access, correct, and 
erase their personal data. Failure to implement reasonable security safeguards may result 
in penalties up to two hundred and fifty crore rupees. Data fiduciaries must not retain 
personal data beyond the period necessary for the purpose it was collected.
```

### Expected Output

- **Summary:** A 2–4 sentence plain-language description of the act
- **Key Points:** 2–3 bullet points about citizen rights and penalties
- **Token reduction:** ~40–60% on this sample
- **Information density score:** ~0.40–0.60

---

## Team

Built for the AI Hackathon — Project 3: The AI Legislative Analyzer

**Required Technique:** Token Compression  
**Judging Metric:** Information Density (value delivered per token consumed)
