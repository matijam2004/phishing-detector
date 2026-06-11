# Email Threat Detector

A machine learning–powered phishing email classifier with a real-time WebGL frontend and a FastAPI backend. Trained on 82,000+ emails, it classifies any email as phishing or legitimate in milliseconds and returns calibrated confidence scores.

**Live demo →** [phishing-detector on Vercel](https://phishing-detector-omega-nine.vercel.app/)  
**Backend API →** [phishing-detector-b32k.onrender.com](https://phishing-detector-b32k.onrender.com)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Backend](#backend)
- [Frontend](#frontend)
- [Deployment](#deployment)
- [Local Development](#local-development)
- [API Reference](#api-reference)
- [Tech Stack](#tech-stack)

---

## Overview

Email Threat Detector is a full-stack AI application that analyzes raw email content and predicts whether it is a phishing attempt. The model was trained on a labeled corpus of 82,000+ real emails using TF-IDF feature extraction and Logistic Regression, achieving high confidence on clearly malicious or clearly safe inputs and surfacing uncertainty on ambiguous cases.

The interface is a single-page app with a WebGL deep-nebula background, glassmorphism UI, and instant feedback — no page reloads, no frameworks.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Browser                          │
│                                                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │              frontend/index.html                │   │
│   │                                                 │   │
│   │   WebGL nebula canvas (background)              │   │
│   │   ┌──────────────────────────────────────────┐  │   │
│   │   │  Glassmorphic UI card                    │  │   │
│   │   │  • Textarea  (email input)               │  │   │
│   │   │  • Analyze button  (Ctrl+Enter)          │  │   │
│   │   │  • Result panel  (verdict + prob. bars)  │  │   │
│   │   └──────────────────────────────────────────┘  │   │
│   └────────────────────┬────────────────────────────┘   │
└────────────────────────│────────────────────────────────┘
                         │  POST /predict  {text}
                         ▼
┌────────────────────────────────────────────────────────┐
│                  FastAPI Backend                        │
│                  (Render.com)                          │
│                                                        │
│   clean_text()  →  TF-IDF transform  →  LR predict    │
│                                                        │
│   ┌───────────────┐   ┌──────────┐   ┌─────────────┐  │
│   │  tfidf.pkl    │   │ lr_model │   │ sgd_model   │  │
│   │  5000 features│   │  .pkl    │   │  .pkl       │  │
│   │  (vectorizer) │   │(primary) │   │ (fallback)  │  │
│   └───────────────┘   └──────────┘   └─────────────┘  │
└────────────────────────────────────────────────────────┘
```

**Request lifecycle:**

1. User pastes email text into the textarea and clicks Analyze (or presses `Ctrl+Enter`).
2. The frontend `fetch()`es `POST /predict` with the raw text as JSON.
3. The backend cleans the text, runs it through the fitted TF-IDF vectorizer, and calls `predict_proba()` on the Logistic Regression model.
4. A JSON response is returned containing the label, confidence, and individual class probabilities.
5. The frontend animates the probability bars and displays the verdict.

---

## Project Structure

```
PhishingCheckerAI/
│
├── frontend/
│   └── index.html          # Complete SPA — UI, WebGL shader, API client
│
├── backend/
│   ├── main.py             # FastAPI app — endpoints, preprocessing, inference
│   ├── requirements.txt    # Python dependencies
│   ├── runtime.txt         # Python 3.11
│   ├── feedback.json       # Logged predictions for future retraining
│   └── models/
│       ├── tfidf.pkl       # Fitted TF-IDF vectorizer (5,000 features)
│       ├── lr_model.pkl    # Logistic Regression classifier (primary)
│       └── sgd_model.pkl   # SGDClassifier (supports partial_fit retraining)
│
├── CS166.ipynb             # Full ML pipeline: EDA → training → evaluation
└── render.yaml             # Render.com backend deployment config
```

---

## ML Pipeline

The full pipeline lives in [`CS166.ipynb`](CS166.ipynb).

### Dataset

| Split      | Samples | Share |
|------------|--------:|------:|
| Training   |  65,988 |   80% |
| Test       |  16,498 |   20% |
| **Total**  | **82,486** | **100%** |

Binary labels: `0` = legitimate, `1` = phishing. The split is stratified to preserve class distribution.

### Preprocessing

Each email goes through a deterministic cleaning pipeline before any features are extracted:

```
raw text
  └─ lowercase
  └─ strip URLs  (http / https / www patterns)
  └─ remove punctuation & special characters
  └─ collapse whitespace
  └─ TF-IDF vectorization  (max 5,000 features)
```

The same `clean_text()` function is used at training time (notebook) and inference time (backend), so there is no train/serve skew.

### Feature Extraction

**TF-IDF** (Term Frequency–Inverse Document Frequency) converts cleaned email text into a 5,000-dimensional sparse vector. Each dimension corresponds to a vocabulary token weighted by how distinctive it is across the corpus — common phishing keywords like "verify", "urgent", "account suspended" naturally get high weight.

### Models

| Model | Description | Use |
|---|---|---|
| `LogisticRegression` | L2-regularized linear classifier | Primary inference |
| `SGDClassifier` | Stochastic Gradient Descent | Loaded as fallback; supports `partial_fit()` for incremental retraining from feedback |

Logistic Regression was chosen for its strong baseline performance on high-dimensional sparse text features, fast inference, and calibrated probability outputs.

### Sample Predictions

| Input type | Confidence |
|---|---|
| Obvious phishing ("verify your account now") | ~99% phishing |
| Clear legitimate (work calendar invite) | ~98% legitimate |
| Ambiguous (generic marketing email) | ~64% phishing |

---

## Backend

Built with **FastAPI** and served via **Uvicorn** on Render.com.

### Startup

On startup, the app loads the serialized vectorizer and model from disk once and holds them in memory. All subsequent requests hit in-memory objects — no disk I/O per request.

```python
# Load order: prefer SGDClassifier, fall back to LogisticRegression
model = joblib.load("models/sgd_model.pkl")  # or lr_model.pkl
vectorizer = joblib.load("models/tfidf.pkl")
```

### Inference path (`POST /predict`)

```
request.text
  → clean_text()          # deterministic preprocessing
  → vectorizer.transform()  # sparse TF-IDF matrix
  → model.predict_proba()   # [P(legitimate), P(phishing)]
  → JSON response
```

### CORS

All origins are permitted (`*`) so the Vercel-hosted frontend can reach the Render backend without proxy configuration.

---

## Frontend

A single HTML file — no build step, no npm, no framework.

### WebGL Background

The nebula background is a custom GLSL fragment shader running in a full-screen `<canvas>`. It renders:

- **Three drifting cloud layers** built from 6-octave fractal Brownian motion (fBm) noise, each scrolling in a different direction and tinted deep teal, blue-violet, or indigo.
- **Three star density layers** — faint background, mid-distance, bright foreground — each with a soft halo and an independent per-star twinkle rate derived from a hash function.
- **Vignette** — edge darkening that focuses attention on the center card.

Everything runs at 60 fps with no external libraries.

### UI

| Element | Detail |
|---|---|
| Title | Shimmer gradient animation via `background-position` keyframes |
| Card | Glassmorphism — `backdrop-filter: blur(28px)` + translucent border |
| Textarea | 190 px, resizable, focus ring on active |
| Button | Glass gradient, lift on hover, scale on press |
| Results | Slide-down animation; probability bars transition over 700 ms |
| Keyboard | `Ctrl+Enter` submits from anywhere |
| Responsive | Single breakpoint at 480 px for mobile |

### API client

```js
const API = location.hostname === "localhost"
  ? "http://localhost:8000"
  : "https://phishing-detector-b32k.onrender.com";

fetch(`${API}/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text }),
})
```

Localhost and production are detected automatically — no environment variables needed in the frontend.

---

## Deployment

### Frontend — Vercel

The `frontend/` directory is deployed as a static site on Vercel. Pushes to `main` trigger automatic redeployment.

### Backend — Render.com

Configured via [`render.yaml`](render.yaml):

```yaml
services:
  - type: web
    name: phishing-detector-api
    runtime: python
    rootDir: backend
    buildCommand: pip install --upgrade --no-cache-dir -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Render builds and starts the FastAPI server automatically on every push to `main`.

---

## Local Development

### Prerequisites

- Python 3.11+
- A modern browser (WebGL 1.0 required for the background)

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
# API available at http://localhost:8000
```

### Frontend

Open `frontend/index.html` directly in a browser — or serve it with any static server:

```bash
npx serve frontend
# or
python -m http.server 3000 --directory frontend
```

The frontend auto-detects `localhost` and routes API calls to `http://localhost:8000`.

---

## API Reference

### `GET /health`

Returns server status.

**Response**
```json
{ "status": "ok" }
```

---

### `POST /predict`

Classifies an email as phishing or legitimate.

**Request body**
```json
{
  "text": "Dear customer, your account has been suspended. Click here to verify..."
}
```

**Response**
```json
{
  "label": "phishing",
  "confidence": 98.7,
  "phishing_probability": 98.7,
  "legitimate_probability": 1.3
}
```

| Field | Type | Description |
|---|---|---|
| `label` | `string` | `"phishing"` or `"legitimate"` |
| `confidence` | `float` | Probability of the predicted class (0–100) |
| `phishing_probability` | `float` | Raw phishing class probability (0–100) |
| `legitimate_probability` | `float` | Raw legitimate class probability (0–100) |

**Errors**

| Status | Meaning |
|---|---|
| `422` | Missing or invalid request body |
| `500` | Model not loaded or inference failure |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, vanilla JavaScript, WebGL (GLSL) |
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML | scikit-learn (TF-IDF + Logistic Regression / SGD) |
| Serialization | joblib |
| Frontend hosting | Vercel |
| Backend hosting | Render.com |
| Training notebook | Jupyter (CS166.ipynb) |

---

*CS166 Project — Matija Malisic*
