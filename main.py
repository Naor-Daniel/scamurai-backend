from __future__ import annotations

import os
import time
import re
import json
from uuid import uuid4
from typing import Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from google.genai import types
from google import genai
from fastapi import FastAPI
from pydantic import BaseModel, Field


# ===================== Config =====================

VERSION = "2.1.0"

RULES_WEIGHT = 0.3
AI_WEIGHT = 0.7

AI_TIMEOUT_SECONDS = 2.0
GEMINI_MODEL_CANDIDATES = [
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
]
_SELECTED_MODEL: str | None = None


# ===================== App =====================

app = FastAPI(title="ScamurAI Backend", version=VERSION)


# ===================== Models =====================

class AnalyzeRequest(BaseModel):
    subject: str = ""
    sender: str = ""
    body: str = ""


class Reason(BaseModel):
    id: str
    title: str
    points: int = Field(ge=0, le=100)  # "risk points" (penalty / contribution)
    evidence: str = ""


class AnalyzeResponse(BaseModel):
    score: int = Field(ge=0, le=100)
    verdict: str
    confidence: str
    risk: dict
    breakdown: dict
    reasons: List[Reason]
    ai: dict
    version: str
    traceId: str


# ===================== Helpers =====================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_text(subject: str, body: str) -> str:
    return (subject + "\n" + body).lower()


def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\bhttps?://[^\s<>\"]+", text, flags=re.IGNORECASE)[:50]


def url_domain(url: str) -> str:
    m = re.match(r"^https?://([^/]+)", url.strip(), flags=re.IGNORECASE)
    return m.group(1).lower() if m else ""


def verdict_from_risk(final_risk: int) -> str:
    if final_risk >= 75:
        return "Malicious"
    if final_risk >= 35:
        return "Suspicious"
    return "Safe"


def confidence_from_diff(risk_rules: int, risk_ai: int, ai_ok: bool) -> str:
    if not ai_ok:
        return "Low"
    diff = abs(risk_rules - risk_ai)
    if diff <= 15:
        return "High"
    if diff <= 35:
        return "Medium"
    return "Low"


# ===================== Rules Engine =====================

def rules_engine(subject: str, sender: str, body: str) -> Tuple[int, dict, list[dict]]:
    text = normalize_text(subject, body)
    urls = extract_urls(body)
    domains = [url_domain(u) for u in urls if u]
    unique_domains = sorted(list(set([d for d in domains if d])))[:20]

    reasons: list[dict] = []
    breakdown = {
        "sender": {"risk": 0, "notes": []},
        "links": {"risk": 0, "notes": [], "linkCount": len(urls), "domains": unique_domains},
        "content": {"risk": 0, "notes": []},
        "attachments": {"risk": 0, "notes": []}
    }

    risk = 0
    urgent_keywords = ["urgent", "immediately", "verify", "password", "suspended", "limited", "invoice", "payment"]
    hit = next((k for k in urgent_keywords if k in text), None)
    if hit:
        pts = 60
        risk += pts
        breakdown["content"]["risk"] = max(breakdown["content"]["risk"], pts)
        breakdown["content"]["notes"].append(f"Urgency/pressure language: '{hit}'")
        reasons.append({
            "id": "urgent_language",
            "title": "Urgent / pressure language detected",
            "points": pts,
            "evidence": f"Found keyword '{hit}'"
        })

    shorteners = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly"]
    shortener_domains = sorted(list(set([d for d in unique_domains if any(s in d for s in shorteners)])))
    if shortener_domains:
        pts = 25
        risk += pts
        breakdown["links"]["risk"] = max(breakdown["links"]["risk"], pts)
        breakdown["links"]["notes"].append("URL shortener detected")
        reasons.append({
            "id": "url_shortener",
            "title": "URL shortener detected",
            "points": pts,
            "evidence": ", ".join(shortener_domains[:5])
        })

    if any(u.lower().startswith("http://") for u in urls):
        pts = 10
        risk += pts
        breakdown["links"]["risk"] = max(breakdown["links"]["risk"], pts)
        breakdown["links"]["notes"].append("Non-HTTPS link detected (http://)")
        reasons.append({
            "id": "http_link",
            "title": "Non-HTTPS link detected",
            "points": pts,
            "evidence": "Found http:// link"
        })

    risk_rules = int(clamp(risk, 0, 100))

    for k in ["sender", "links", "content", "attachments"]:
        breakdown[k]["risk"] = int(clamp(breakdown[k]["risk"], 0, 100))

    return risk_rules, breakdown, reasons


# ===================== Gemini (AI) =====================

_GEMINI_READY = False


def pick_working_model(client: genai.Client) -> str:
    global _SELECTED_MODEL
    if _SELECTED_MODEL:
        return _SELECTED_MODEL

    last_err = ""
    for m in GEMINI_MODEL_CANDIDATES:
        try:
            client.models.generate_content(model=m, contents="ping")
            _SELECTED_MODEL = m
            return m
        except Exception as e:
            last_err = str(e)

    raise RuntimeError(f"No working Gemini model found. Last error: {last_err}")


def _gemini_call(subject: str, sender: str, body: str, domains: list[str]) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1"),
    )

    schema = {
        "ai_risk": 0,
        "confidence": "Low|Medium|High",
        "threatType": "safe|phishing|malware|bec|invoice|other",
        "summary": "max 2 sentences",
        "findings": ["max 5 short bullets"]
    }

    prompt = f"""
You are an email security analyst. Return ONLY valid JSON matching this schema exactly:
{json.dumps(schema)}

Rules:
- ai_risk is 0..100 (0 safe, 100 dangerous).
- Do not invent facts.
- summary <= 2 sentences.
- findings <= 5 items.

Email:
Subject: {subject}
Sender: {sender}

Extracted domains: {", ".join(domains[:20]) if domains else "(none)"}

Body:
{body[:8000]}
""".strip()

    model_name = pick_working_model(client)
    response = client.models.generate_content(model=model_name, contents=prompt)

    text = (response.text or "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise RuntimeError("Gemini did not return JSON")

    data = json.loads(m.group(0))

    ai_risk = int(clamp(float(data.get("ai_risk", 0)), 0, 100))
    confidence = str(data.get("confidence", "Low"))
    threat_type = str(data.get("threatType", "other"))
    summary = str(data.get("summary", "")).strip()

    findings = data.get("findings", [])
    if not isinstance(findings, list):
        findings = []
    findings = [str(x).strip() for x in findings if str(x).strip()][:5]

    if confidence not in ["Low", "Medium", "High"]:
        confidence = "Low"

    return {
        "risk_ai": ai_risk,
        "confidence": confidence,
        "threatType": threat_type,
        "summary": summary,
        "findings": findings,
        "model": model_name
    }


def gemini_analyze(subject: str, sender: str, body: str, domains: list[str]) -> tuple[bool, dict[str, Any]]:
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_gemini_call, subject, sender, body, domains)
            data = fut.result(timeout=AI_TIMEOUT_SECONDS)

        latency_ms = int((time.time() - t0) * 1000)

        return True, {
            "summary": data.get("summary", ""),
            "threatType": data.get("threatType", "other"),
            "findings": data.get("findings", []),
            "model": data.get("model", ""),
            "latencyMs": latency_ms,
            "error": "",
            "_risk_ai": int(data.get("risk_ai", 0)),
            "_confidence": str(data.get("confidence", "Low"))
        }

    except FuturesTimeoutError:
        latency_ms = int((time.time() - t0) * 1000)
        return False, {
            "summary": "",
            "threatType": "other",
            "findings": [],
            "model": "",
            "latencyMs": latency_ms,
            "error": f"AI timeout after {AI_TIMEOUT_SECONDS}s"
        }
    except Exception as ex:
        latency_ms = int((time.time() - t0) * 1000)
        return False, {
            "summary": "",
            "threatType": "other",
            "findings": [],
            "model": "",
            "latencyMs": latency_ms,
            "error": str(ex)
        }


# ===================== Fusion =====================

def fuse(risk_rules: int, risk_ai: int, ai_ok: bool) -> tuple[int, bool]:
    if not ai_ok:
        return int(clamp(risk_rules, 0, 100)), False

    base = RULES_WEIGHT * risk_rules + AI_WEIGHT * risk_ai
    guard_min = 0.5 * risk_rules if risk_rules >= 70 else 0.0
    final_risk = max(base, guard_min)
    guard_applied = (final_risk != base)
    return int(clamp(final_risk, 0, 100)), guard_applied


# ===================== Routes =====================

@app.get("/")
def root():
    return {"status": "running", "version": VERSION}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    trace_id = str(uuid4())

    subject = req.subject or ""
    sender = req.sender or ""
    body = req.body or ""

    risk_rules, breakdown, reasons = rules_engine(subject, sender, body)

    urls = extract_urls(body)
    domains = sorted(list(set([url_domain(u) for u in urls if u])))[:20]

    ai_ok, ai = gemini_analyze(subject, sender, body, domains)

    risk_ai = int(ai.get("_risk_ai", 0)) if ai_ok else 0
    ai_conf = str(ai.get("_confidence", "Low")) if ai_ok else "Low"

    ai.pop("_risk_ai", None)
    ai.pop("_confidence", None)

    final_risk, guard_applied = fuse(risk_rules, risk_ai, ai_ok)
    verdict = verdict_from_risk(final_risk)
    confidence = confidence_from_diff(risk_rules, risk_ai, ai_ok)
    safe_score = int(clamp(100 - final_risk, 0, 100))

    return {
        "score": safe_score,
        "verdict": verdict,
        "confidence": confidence,
        "risk": {
            "final": final_risk,
            "rules": int(clamp(risk_rules, 0, 100)),
            "ai": int(clamp(risk_ai, 0, 100)),
            "weights": {"rules": RULES_WEIGHT, "ai": AI_WEIGHT},
            "guardrailApplied": guard_applied
        },
        "breakdown": breakdown,
        "reasons": reasons,
        "ai": ai,
        "version": VERSION,
        "traceId": trace_id
    }