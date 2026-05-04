import os
import re
import requests
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

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
HF_API    = "https://router.huggingface.co/hf-inference/models/ealvaradob/bert-finetuned-phishing"
HEADERS   = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2000]  # keep well within BERT's 512-token limit


class EmailRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: EmailRequest):
    cleaned = clean_text(req.text)

    try:
        res = requests.post(HF_API, headers=HEADERS, json={"inputs": cleaned}, timeout=30)
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=503, detail="Model is warming up — please try again in a few seconds.")
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=502, detail="Could not reach the model API.")

    if res.status_code == 503:
        raise HTTPException(status_code=503, detail="Model is warming up — please try again in a few seconds.")

    if not res.ok:
        raise HTTPException(status_code=502, detail=f"HF API error {res.status_code}: {res.text[:200]}")

    data = res.json()

    # HF returns [[{label, score}, ...]] or [{label, score}, ...]
    scores = data[0] if isinstance(data[0], list) else data

    phishing_prob   = 0.0
    legitimate_prob = 0.0

    for item in scores:
        lbl = item["label"].upper()
        if lbl in ("LABEL_1", "PHISHING", "1"):
            phishing_prob = item["score"]
        elif lbl in ("LABEL_0", "LEGITIMATE", "0", "SAFE"):
            legitimate_prob = item["score"]

    # Fallback: assume highest score is the prediction
    if phishing_prob == 0.0 and legitimate_prob == 0.0:
        scores_sorted = sorted(scores, key=lambda x: x["score"], reverse=True)
        phishing_prob   = scores_sorted[0]["score"]
        legitimate_prob = scores_sorted[1]["score"] if len(scores_sorted) > 1 else 1 - phishing_prob

    label      = "phishing" if phishing_prob > legitimate_prob else "legitimate"
    confidence = round(max(phishing_prob, legitimate_prob) * 100, 2)

    return {
        "label":                   label,
        "confidence":              confidence,
        "phishing_probability":    round(phishing_prob   * 100, 2),
        "legitimate_probability":  round(legitimate_prob * 100, 2),
    }
