import re
import json
import joblib
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Phishing Email Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

MODELS_DIR    = Path(__file__).parent / "models"
FEEDBACK_FILE = Path(__file__).parent / "feedback.json"

vectorizer = joblib.load(MODELS_DIR / "tfidf.pkl")

# Prefer SGDClassifier (supports partial_fit); fall back to LogisticRegression
_sgd_path = MODELS_DIR / "sgd_model.pkl"
_lr_path  = MODELS_DIR / "lr_model.pkl"

if _sgd_path.exists():
    try:
        model = joblib.load(_sgd_path)
        incremental = True
    except Exception:
        model = joblib.load(_lr_path)
        incremental = False
elif _lr_path.exists():
    model = joblib.load(_lr_path)
    incremental = False
else:
    raise RuntimeError("No model file found in backend/models/")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class EmailRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)
    correct_label: str = Field(..., pattern="^(phishing|legitimate)$")


@app.get("/health")
def health():
    return {"status": "ok", "incremental_learning": incremental}


@app.post("/predict")
def predict(req: EmailRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Email text cannot be empty.")

    cleaned  = clean_text(req.text)
    features = vectorizer.transform([cleaned]).toarray()
    label    = int(model.predict(features)[0])
    probs    = model.predict_proba(features)[0]
    confidence = round(float(max(probs)) * 100, 2)

    return {
        "label": "phishing" if label == 1 else "legitimate",
        "confidence": confidence,
        "phishing_probability":   round(float(probs[1]) * 100, 2),
        "legitimate_probability": round(float(probs[0]) * 100, 2),
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    # Persist feedback
    entry = {
        "text":  req.text,
        "label": req.correct_label,
        "ts":    datetime.now(timezone.utc).isoformat(),
    }
    entries = json.loads(FEEDBACK_FILE.read_text()) if FEEDBACK_FILE.exists() else []
    entries.append(entry)
    FEEDBACK_FILE.write_text(json.dumps(entries, indent=2))

    # Update model if SGDClassifier is loaded
    if incremental:
        cleaned  = clean_text(req.text)
        features = vectorizer.transform([cleaned]).toarray()
        label    = 1 if req.correct_label == "phishing" else 0
        for _ in range(10):
            model.partial_fit(features, [label], classes=[0, 1])
        joblib.dump(model, _sgd_path)
        return {"status": "model updated", "total_feedback": len(entries)}

    return {"status": "feedback saved (upgrade to SGDClassifier to enable live updates)", "total_feedback": len(entries)}
