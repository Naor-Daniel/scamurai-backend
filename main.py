from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class AnalyzeRequest(BaseModel):
    subject: str = ""
    sender: str = ""
    body: str = ""


@app.get("/")
def root():
    return {"status": "running"}


def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    text = (req.subject + " " + req.body).lower()

    reasons = []
    risk_score = 0
    verdict = "Safe"

    if "urgent" in text:
        risk_score = 60
        verdict = "Suspicious"
        reasons.append({
            "id": "urgent_language",
            "title": "Urgent language detected",
            "points": 60,
            "evidence": "Found keyword 'urgent'"
        })

    # ✅ This is the key: higher = safer
    safe_score = clamp(100 - risk_score, 0, 100)

    return {
        "score": safe_score,
        "verdict": verdict,
        "reasons": reasons,
        "linkCount": 0
    }