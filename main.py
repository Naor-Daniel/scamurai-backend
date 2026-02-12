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
    score = 10
    verdict = "Safe"

    text = (req.subject + " " + req.body).lower()

    if "urgent" in text:
        score = 60
        verdict = "Suspicious"

    return {
        "score": score,
        "verdict": verdict,
        "reasons": [
            {
                "title": "Urgent language detected",
                "points": 50,
                "evidence": "Found keyword 'urgent'"
            }
        ] if verdict != "Safe" else []
    }