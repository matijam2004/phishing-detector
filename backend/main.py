import re
import joblib
from pathlib import Path
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

MODELS_DIR = Path(__file__).parent / "models"

vectorizer = joblib.load(MODELS_DIR / "tfidf.pkl")

_sgd_path = MODELS_DIR / "sgd_model.pkl"
_lr_path  = MODELS_DIR / "lr_model.pkl"

if _sgd_path.exists():
    try:
        model = joblib.load(_sgd_path)
    except Exception:
        model = joblib.load(_lr_path)
elif _lr_path.exists():
    model = joblib.load(_lr_path)
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: EmailRequest):
    cleaned  = clean_text(req.text)
    features = vectorizer.transform([cleaned]).toarray()
    label    = int(model.predict(features)[0])
    probs    = model.predict_proba(features)[0]
    confidence = round(float(max(probs)) * 100, 2)

    return {
        "label":                   "phishing" if label == 1 else "legitimate",
        "confidence":              confidence,
        "phishing_probability":    round(float(probs[1]) * 100, 2),
        "legitimate_probability":  round(float(probs[0]) * 100, 2),
    }
