from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from fastapi import FastAPI
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# =====================================================================
# Application configuration
# =====================================================================

applicationVersion = "3.0.0"

hardChecksWeight = float(os.getenv("HARD_CHECKS_WEIGHT", "0.30"))
aiTimeoutSeconds = float(os.getenv("AI_TIMEOUT_SECONDS", "12.0"))
geminiApiKeyEnvironmentVariable = "GEMINI_API_KEY"

modelCandidates = [
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.0-flash",
]

selectedModel: str | None = None

# =====================================================================
# FastAPI application
# =====================================================================

app = FastAPI(title="ScamurAI Backend", version=applicationVersion)

# =====================================================================
# API models
# =====================================================================

class AuthenticationSummary(BaseModel):
    spf: str = "unknown"
    dkim: str = "unknown"
    dmarc: str = "unknown"


class AnalyzeRequest(BaseModel):
    subject: str = ""
    sender: str = ""
    body: str = ""
    fromDomain: str = ""
    replyToDomain: str = ""
    returnPathDomain: str = ""
    authentication: AuthenticationSummary = Field(default_factory=AuthenticationSummary)


class RiskReason(BaseModel):
    id: str
    title: str
    points: int = Field(ge=0, le=100)
    evidence: str = ""


class AnalyzeResponse(BaseModel):
    score: int = Field(ge=0, le=100)
    verdict: str
    confidence: str
    risk: dict
    breakdown: dict
    reasons: List[RiskReason]
    ai: dict
    ui: dict
    version: str
    traceId: str

# =====================================================================
# Domain types
# =====================================================================

@dataclass(frozen=True)
class EmailFeatures:
    urls: List[str]
    domains: List[str]
    containsHttpLinks: bool
    containsShortenedLinks: bool
    containsMixedScriptText: bool

# =====================================================================
# Utility helpers
# =====================================================================

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def extractUrls(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"\bhttps?://[^\s<>\"]+", text, flags=re.IGNORECASE)[:50]


def extractDomainFromUrl(url: str) -> str:
    match = re.match(r"^https?://([^/]+)", url.strip(), flags=re.IGNORECASE)
    return match.group(1).lower() if match else ""


def containsMixedScript(text: str) -> bool:
    if not text:
        return False
    hasLatin = bool(re.search(r"[A-Za-z]", text))
    hasNonAscii = bool(re.search(r"[^\x00-\x7F]", text))
    return hasLatin and hasNonAscii


def computeFeatures(subject: str, sender: str, body: str) -> EmailFeatures:
    urls = extractUrls(body)
    domains = sorted(set([extractDomainFromUrl(url) for url in urls if url]))[:20]
    containsHttpLinks = any(url.lower().startswith("http://") for url in urls)

    knownShorteners = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly"]
    containsShortenedLinks = any(any(shortener in domain for shortener in knownShorteners) for domain in domains)

    mixedScript = containsMixedScript(subject) or containsMixedScript(sender) or containsMixedScript(body)

    return EmailFeatures(
        urls=urls,
        domains=domains,
        containsHttpLinks=containsHttpLinks,
        containsShortenedLinks=containsShortenedLinks,
        containsMixedScriptText=mixedScript,
    )


def verdictFromRisk(risk: int) -> str:
    if risk >= 75:
        return "Malicious"
    if risk >= 35:
        return "Suspicious"
    return "Safe"

# =====================================================================
# Gemini integration
# =====================================================================

def getGeminiClient(apiKey: str) -> genai.Client:
    return genai.Client(
        api_key=apiKey,
        http_options=types.HttpOptions(api_version="v1"),
    )


def pickWorkingModel(client: genai.Client) -> str:
    global selectedModel
    if selectedModel:
        return selectedModel

    lastError = ""
    for candidate in modelCandidates:
        try:
            client.models.generate_content(model=candidate, contents="ping")
            selectedModel = candidate
            return candidate
        except Exception as exception:
            lastError = str(exception)

    raise RuntimeError(f"No working Gemini model found. Last error: {lastError}")


def parseFirstJsonObject(text: str) -> Dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("AI response did not contain a JSON object.")
    return json.loads(match.group(0))


def buildAiScoringPrompt(
    request: AnalyzeRequest,
    features: EmailFeatures,
) -> str:
    schema = {
        "hardChecks": [
            {
                "id": "string",
                "title": "string",
                "maxPoints": "int",
                "triggered": "bool",
                "riskPoints": "int 0..maxPoints",
                "evidence": "short quote or exact indicator, <= 120 chars"
            }
        ],
        "freeAssessment": {
            "risk": "int 0..100",
            "threatType": "safe|phishing|malware|bec|invoice|other",
            "summary": "max 2 short sentences",
            "keyFindings": ["<= 4 bullets, each <= 14 words"],
            "recommendedAction": "one short sentence"
        },
        "confidence": {
            "label": "Low|Medium|High",
            "score": "int 0..100",
            "rationale": ["<= 4 short bullets, evidence-based"]
        }
    }

    hardChecksCatalog = [
        {"id": "authFails", "title": "Email authentication failed (SPF/DKIM/DMARC)", "maxPoints": 40},
        {"id": "authMissing", "title": "Email authentication missing/unknown", "maxPoints": 15},
        {"id": "replyToMismatch", "title": "Reply-To / Return-Path mismatch", "maxPoints": 20},
        {"id": "lookalikeOrHomograph", "title": "Potential lookalike / homograph spoofing", "maxPoints": 25},
        {"id": "httpLink", "title": "Non-HTTPS link present", "maxPoints": 10},
        {"id": "shortenedLink", "title": "URL shortener present", "maxPoints": 20},
        {"id": "credentialHarvesting", "title": "Asks for password/credit-card/verification", "maxPoints": 40},
        {"id": "pressureLanguage", "title": "Pressure/urgency language", "maxPoints": 20},
    ]

    hardChecksJson = json.dumps(hardChecksCatalog, indent=2)

    prompt = f"""
You are an email security scoring engine.

Return ONLY valid JSON matching this schema exactly:
{json.dumps(schema, indent=2)}

Rules:
- Output must be JSON only. No markdown, no extra keys.
- Do NOT invent facts. Every claim must be directly supported by the provided email text or metadata.
- "this is phishing" inside the email is NOT proof by itself. Treat it as weak evidence unless other indicators exist.
- hardChecks must cover EVERY item in the following catalog exactly once, in the same order.
- For each hardChecks item:
  - triggered is true/false based on the evidence.
  - riskPoints is 0..maxPoints, proportional to evidence strength.
  - evidence must quote or point to the exact indicator (keyword/domain/auth status), <= 120 chars.
- freeAssessment.risk is your holistic risk (0 safe, 100 highly malicious).
- confidence is confidence in the VERDICT class (Safe/Suspicious/Malicious), not confidence in the numeric risk.
  - High confidence requires strong, consistent evidence.
  - Low confidence if evidence is weak, ambiguous, or looks like a test email.

Hard checks catalog (must be used exactly):
{hardChecksJson}

Email metadata:
Subject: {request.subject}
Sender: {request.sender}
fromDomain: {request.fromDomain}
replyToDomain: {request.replyToDomain}
returnPathDomain: {request.returnPathDomain}
authentication: spf={request.authentication.spf}, dkim={request.authentication.dkim}, dmarc={request.authentication.dmarc}

Extracted URLs count: {len(features.urls)}
Extracted URL domains: {", ".join(features.domains) if features.domains else "(none)"}
containsHttpLinks: {features.containsHttpLinks}
containsShortenedLinks: {features.containsShortenedLinks}
containsMixedScriptText: {features.containsMixedScriptText}

Body (truncated):
{request.body[:8000]}
""".strip()

    return prompt


def callGeminiScoring(request: AnalyzeRequest, features: EmailFeatures) -> Dict[str, Any]:
    apiKey = os.getenv(geminiApiKeyEnvironmentVariable, "").strip()
    if not apiKey:
        raise RuntimeError(f"Missing {geminiApiKeyEnvironmentVariable} environment variable.")

    client = getGeminiClient(apiKey)
    modelName = pickWorkingModel(client)

    prompt = buildAiScoringPrompt(request, features)
    response = client.models.generate_content(model=modelName, contents=prompt)

    rawText = (response.text or "").strip()
    parsed = parseFirstJsonObject(rawText)

    parsed["_meta"] = {"model": modelName}
    return parsed


def analyzeWithGemini(request: AnalyzeRequest, features: EmailFeatures) -> Tuple[bool, Dict[str, Any]]:
    start = time.time()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(callGeminiScoring, request, features)
            data = future.result(timeout=aiTimeoutSeconds)

        latencyMs = int((time.time() - start) * 1000)
        data["_meta"]["latencyMs"] = latencyMs
        data["_meta"]["error"] = ""
        return True, data

    except FuturesTimeoutError:
        latencyMs = int((time.time() - start) * 1000)
        return False, {
            "hardChecks": [],
            "freeAssessment": {
                "risk": 0,
                "threatType": "other",
                "summary": "",
                "keyFindings": [],
                "recommendedAction": "",
            },
            "confidence": {"label": "Low", "score": 0, "rationale": [f"AI timeout after {aiTimeoutSeconds}s"]},
            "_meta": {"model": "", "latencyMs": latencyMs, "error": f"AI timeout after {aiTimeoutSeconds}s"},
        }

    except Exception as exception:
        latencyMs = int((time.time() - start) * 1000)
        return False, {
            "hardChecks": [],
            "freeAssessment": {
                "risk": 0,
                "threatType": "other",
                "summary": "",
                "keyFindings": [],
                "recommendedAction": "",
            },
            "confidence": {"label": "Low", "score": 0, "rationale": [str(exception)]},
            "_meta": {"model": "", "latencyMs": latencyMs, "error": str(exception)},
        }

# =====================================================================
# Scoring + UI mapping
# =====================================================================

def computeHardRisk(hardChecks: List[Dict[str, Any]]) -> Tuple[int, List[Dict[str, Any]]]:
    reasons: List[Dict[str, Any]] = []
    total = 0

    for item in hardChecks:
        triggered = bool(item.get("triggered", False))
        maxPoints = int(item.get("maxPoints", 0) or 0)
        riskPoints = int(item.get("riskPoints", 0) or 0)
        riskPoints = int(clamp(riskPoints, 0, maxPoints))

        if triggered and riskPoints > 0:
            total += riskPoints
            reasons.append({
                "id": str(item.get("id", "hardCheck")),
                "title": str(item.get("title", "Hard check")),
                "points": riskPoints,
                "evidence": str(item.get("evidence", "")),
            })

    return int(clamp(total, 0, 100)), reasons


def buildBreakdown(request: AnalyzeRequest, features: EmailFeatures, hardRisk: int, freeRisk: int) -> Dict[str, Any]:
    identityRisk = 0
    if request.authentication.spf == "fail" or request.authentication.dkim == "fail" or request.authentication.dmarc == "fail":
        identityRisk = max(identityRisk, 70)
    elif request.authentication.spf == "unknown" and request.authentication.dkim == "unknown" and request.authentication.dmarc == "unknown":
        identityRisk = max(identityRisk, 25)

    if request.replyToDomain and request.fromDomain and request.replyToDomain != request.fromDomain:
        identityRisk = max(identityRisk, 40)

    linksRisk = 0
    if features.containsHttpLinks:
        linksRisk = max(linksRisk, 20)
    if features.containsShortenedLinks:
        linksRisk = max(linksRisk, 35)

    contentRisk = 0
    if features.containsMixedScriptText:
        contentRisk = max(contentRisk, 30)

    return {
        "identity": {
            "risk": int(clamp(identityRisk, 0, 100)),
            "fromDomain": request.fromDomain,
            "replyToDomain": request.replyToDomain,
            "returnPathDomain": request.returnPathDomain,
            "authentication": {
                "spf": request.authentication.spf,
                "dkim": request.authentication.dkim,
                "dmarc": request.authentication.dmarc,
            },
        },
        "links": {
            "risk": int(clamp(linksRisk, 0, 100)),
            "linkCount": len(features.urls),
            "domains": features.domains,
            "containsHttpLinks": features.containsHttpLinks,
            "containsShortenedLinks": features.containsShortenedLinks,
        },
        "content": {
            "risk": int(clamp(contentRisk, 0, 100)),
            "containsMixedScriptText": features.containsMixedScriptText,
        },
        "engine": {
            "hardRisk": hardRisk,
            "freeRisk": freeRisk,
            "weights": {"hard": hardChecksWeight, "free": 1.0 - hardChecksWeight},
        },
    }


def buildUiSummary(aiPayload: Dict[str, Any], hardReasons: List[Dict[str, Any]]) -> Dict[str, Any]:
    free = aiPayload.get("freeAssessment", {}) if isinstance(aiPayload.get("freeAssessment", {}), dict) else {}
    keyFindings = free.get("keyFindings", [])
    if not isinstance(keyFindings, list):
        keyFindings = []

    hardFindings = [str(r.get("title", "")).strip() for r in sorted(hardReasons, key=lambda x: int(x.get("points", 0)), reverse=True)[:2]]
    merged = []
    for item in hardFindings + [str(x).strip() for x in keyFindings]:
        if item and item not in merged:
            merged.append(item)
        if len(merged) >= 4:
            break

    recommendedAction = str(free.get("recommendedAction", "")).strip()

    return {
        "keyFindings": merged,
        "recommendedAction": recommendedAction,
    }

# =====================================================================
# FastAPI routes
# =====================================================================

@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "running", "version": applicationVersion}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    traceId = str(uuid4())

    features = computeFeatures(request.subject, request.sender, request.body)

    aiAvailable, aiPayload = analyzeWithGemini(request, features)

    hardChecks = aiPayload.get("hardChecks", [])
    if not isinstance(hardChecks, list):
        hardChecks = []

    hardRisk, hardReasons = computeHardRisk(hardChecks)

    freeAssessment = aiPayload.get("freeAssessment", {}) if isinstance(aiPayload.get("freeAssessment", {}), dict) else {}
    freeRisk = int(clamp(float(freeAssessment.get("risk", 0) or 0), 0, 100))

    finalRisk = hardChecksWeight * hardRisk + (1.0 - hardChecksWeight) * freeRisk
    finalRisk = int(clamp(finalRisk, 0, 100))

    verdict = verdictFromRisk(finalRisk)
    safetyScore = int(clamp(100 - finalRisk, 0, 100))

    confidence = aiPayload.get("confidence", {}) if isinstance(aiPayload.get("confidence", {}), dict) else {}
    confidenceLabel = str(confidence.get("label", "Low"))
    if confidenceLabel not in ["Low", "Medium", "High"]:
        confidenceLabel = "Low"
    confidenceScore = int(clamp(float(confidence.get("score", 0) or 0), 0, 100))

    rationale = confidence.get("rationale", [])
    if not isinstance(rationale, list):
        rationale = []
    rationale = [str(x).strip() for x in rationale if str(x).strip()][:4]

    breakdown = buildBreakdown(request, features, hardRisk, freeRisk)
    ui = buildUiSummary(aiPayload, hardReasons)

    response = {
        "score": safetyScore,
        "verdict": verdict,
        "confidence": confidenceLabel,
        "risk": {
            "final": finalRisk,
            "hard": hardRisk,
            "free": freeRisk,
            "weights": {"hard": hardChecksWeight, "free": 1.0 - hardChecksWeight},
            "confidenceScore": confidenceScore,
            "confidenceRationale": rationale,
        },
        "breakdown": breakdown,
        "reasons": hardReasons,
        "ai": {
            "summary": str(freeAssessment.get("summary", "")).strip(),
            "threatType": str(freeAssessment.get("threatType", "other")).strip(),
            "keyFindings": freeAssessment.get("keyFindings", []),
            "recommendedAction": str(freeAssessment.get("recommendedAction", "")).strip(),
            "model": aiPayload.get("_meta", {}).get("model", ""),
            "latencyMs": aiPayload.get("_meta", {}).get("latencyMs", 0),
            "error": aiPayload.get("_meta", {}).get("error", ""),
            "hardChecks": hardChecks,
        },
        "ui": ui,
        "version": applicationVersion,
        "traceId": traceId,
    }

    return response