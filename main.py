from fastapi import FastAPI
from pydantic import BaseModel
from uuid import uuid4
import re

app = FastAPI()

VERSION = "1.0.0"


class AnalyzeRequest(BaseModel):
    subject: str = ""
    sender: str = ""
    body: str = ""


@app.get("/")
def root():
    return {"status": "running", "version": VERSION}


def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\bhttps?://[^\s<>\"]+", text, flags=re.IGNORECASE)[:50]


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    trace_id = str(uuid4())
    text = (req.subject + " " + req.body).lower()

    reasons = []
    link_count = len(extract_urls(req.body or ""))

    risk_score = 0
    verdict = "Safe"

    if "urgent" in text:
        risk_score = 60
        verdict = "Suspicious"
        reasons.append(
            {
                "id": "urgent_language",
                "title": "Urgent language detected",
                "points": 60,
                "evidence": "Found keyword 'urgent'"
            }
        )

    safe_score = clamp(100 - risk_score, 0, 100)

    return {
        "score": safe_score,
        "verdict": verdict,
        "reasons": reasons,
        "linkCount": link_count,
        "version": VERSION,
        "traceId": trace_id
    }