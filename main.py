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


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    text = (req.subject + " " + req.body).lower()

    # Risk score: 0 = safe, 100 = malicious
    risk_score = 0
    verdict = "Safe"
    reasons = []

    if "urgent" in text:
        risk_score = 60
        verdict = "Suspicious"
        reasons.append(
            {
                "title": "Urgent language detected",
                "points": 50,
                "evidence": "Found keyword 'urgent'"
            }
        )

    # Safe score: 100 = safe, 0 = malicious
    safe_score = max(0, min(100, 100 - risk_score))

    return {
        "score": safe_score,
        "verdict": verdict,
        "reasons": reasons
    }