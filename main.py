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

applicationVersion = "3.1.0"

defaultHardChecksWeight = float(os.getenv("HARD_CHECKS_WEIGHT", "0.30"))
aiTimeoutSeconds = float(os.getenv("AI_TIMEOUT_SECONDS", "20.0"))
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


class UserSettings(BaseModel):
    """
    Settings are user-controlled and sent by the add-on on every /analyze call.
    The backend treats them as preferences, not as trusted facts.
    """
    sensitivity: str = "balanced"  # lenient|balanced|strict
    hardChecksWeight: float = Field(defaultHardChecksWeight, ge=0.0, le=1.0)

    allowlistedDomains: List[str] = Field(default_factory=list)
    blocklistedDomains: List[str] = Field(default_factory=list)

    treatAuthUnknownAsRisk: bool = True
    assumeTestEmailIfContainsTestingWords: bool = True

    language: str = "en"  # en|he


class AnalyzeRequest(BaseModel):
    subject: str = ""
    sender: str = ""
    body: str = ""

    fromDomain: str = ""
    replyToDomain: str = ""
    returnPathDomain: str = ""

    authentication: AuthenticationSummary = Field(default_factory=AuthenticationSummary)
    settings: UserSettings = Field(default_factory=UserSettings)


class RiskReason(BaseModel):
    id: str
    title: str
    points: int = Field(ge=0, le=100)
    evidence: str = ""


class AnalyzeResponse(BaseModel):
    score: int = Field(ge=0, le=100)  # safety score
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


def normalizeDomain(value: str) -> str:
    return str(value or "").strip().lower()


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


def sanitizeSettings(settings: UserSettings) -> UserSettings:
    sensitivity = settings.sensitivity.strip().lower()
    if sensitivity not in ["lenient", "balanced", "strict"]:
        sensitivity = "balanced"

    hardWeight = float(clamp(float(settings.hardChecksWeight), 0.0, 1.0))

    allowlisted = sorted(set([normalizeDomain(x) for x in settings.allowlistedDomains if normalizeDomain(x)]))[:50]
    blocklisted = sorted(set([normalizeDomain(x) for x in settings.blocklistedDomains if normalizeDomain(x)]))[:50]

    language = settings.language.strip().lower()
    if language not in ["en", "he"]:
        language = "en"

    return UserSettings(
        sensitivity=sensitivity,
        hardChecksWeight=hardWeight,
        allowlistedDomains=allowlisted,
        blocklistedDomains=blocklisted,
        treatAuthUnknownAsRisk=bool(settings.treatAuthUnknownAsRisk),
        assumeTestEmailIfContainsTestingWords=bool(settings.assumeTestEmailIfContainsTestingWords),
        language=language,
    )


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


def hardChecksCatalog() -> List[Dict[str, Any]]:
    return [
        {"id": "authFails", "title": "Email authentication failed (SPF/DKIM/DMARC)", "maxPoints": 40},
        {"id": "authMissing", "title": "Email authentication missing/unknown", "maxPoints": 15},
        {"id": "replyToMismatch", "title": "Reply-To / Return-Path mismatch", "maxPoints": 20},
        {"id": "lookalikeOrHomograph", "title": "Potential lookalike / homograph spoofing", "maxPoints": 25},
        {"id": "httpLink", "title": "Non-HTTPS link present", "maxPoints": 10},
        {"id": "shortenedLink", "title": "URL shortener present", "maxPoints": 20},
        {"id": "credentialHarvesting", "title": "Asks for password/credit-card/verification", "maxPoints": 40},
        {"id": "pressureLanguage", "title": "Pressure/urgency language", "maxPoints": 20},
    ]


def buildAiScoringPrompt(request: AnalyzeRequest, features: EmailFeatures, settings: UserSettings) -> str:
    schema = {
        "hardChecks": [
            {
                "id": "string",
                "title": "string",
                "maxPoints": "int",
                "triggered": "bool",
                "riskPoints": "int 0..maxPoints",
                "evidence": "short quote or exact indicator, <= 120 chars",
            }
        ],
        "freeAssessment": {
            "risk": "int 0..100",
            "threatType": "safe|phishing|malware|bec|invoice|other",
            "summary": "max 2 short sentences",
            "keyFindings": ["<= 4 bullets, each <= 14 words"],
            "recommendedAction": "one short sentence",
        },
        "confidence": {
            "label": "Low|Medium|High",
            "score": "int 0..100",
            "rationale": ["<= 4 short bullets, evidence-based"],
        },
    }

    catalog = hardChecksCatalog()
    catalogJson = json.dumps(catalog, indent=2)

    promptLanguageRule = "Write summary/findings/action in English."
    if settings.language == "he":
        promptLanguageRule = "Write summary/findings/action in Hebrew (natural, concise)."

    promptSensitivityRule = ""
    if settings.sensitivity == "strict":
        promptSensitivityRule = (
            "- Sensitivity is STRICT: prefer flagging risk when evidence is moderate.\n"
            "- If in doubt between Safe and Suspicious, lean Suspicious.\n"
        )
    elif settings.sensitivity == "lenient":
        promptSensitivityRule = (
            "- Sensitivity is LENIENT: avoid over-flagging weak signals.\n"
            "- If evidence is weak/ambiguous, prefer Safe or low Suspicious.\n"
        )
    else:
        promptSensitivityRule = (
            "- Sensitivity is BALANCED: weigh evidence proportionally.\n"
        )

    allowlisted = ", ".join(settings.allowlistedDomains) if settings.allowlistedDomains else "(none)"
    blocklisted = ", ".join(settings.blocklistedDomains) if settings.blocklistedDomains else "(none)"

    prompt = f"""
You are an email security scoring engine.

Return ONLY valid JSON matching this schema exactly:
{json.dumps(schema, indent=2)}

General rules:
- Output JSON only. No markdown, no extra keys.
- Do NOT invent facts. Every claim must be supported by provided text/metadata.
- The phrase "this is phishing" in the email is NOT proof by itself. Treat it as weak evidence unless corroborated.
- If the email looks like a self-test (e.g., "testing", "does it work", "what's your score"), reduce confidence.

Hard checks rules:
- hardChecks MUST cover EVERY item in the catalog exactly once, same order, same ids/titles/maxPoints.
- For each hardChecks item:
  - triggered true/false based on direct evidence.
  - riskPoints is 0..maxPoints proportional to evidence strength.
  - evidence must reference the exact indicator (keyword/auth status/domain), <= 120 chars.

Free assessment rules:
- freeAssessment.risk: holistic risk 0..100 (0 safe, 100 highly malicious).
- threatType: best-fit category.
- summary: <= 2 short sentences.
- keyFindings: <= 4 bullets, each <= 14 words, evidence-based.
- recommendedAction: exactly 1 short sentence, user-facing.

Confidence rules (about the VERDICT class, not the numeric risk):
- label: Low/Medium/High, score: 0..100.
- High confidence requires strong, consistent evidence.
- Low confidence if evidence is weak, contradictory, ambiguous, or likely a test email.

User preferences:
- {promptLanguageRule}
{promptSensitivityRule.strip()}

Allowlist and blocklist:
- If fromDomain or any URL domain is in blocklistedDomains, treat as strong risk evidence.
- If fromDomain is allowlisted AND no other strong indicators exist, risk may decrease.
- Never mark Safe with High confidence if blocklisted domain appears.

treatAuthUnknownAsRisk: {settings.treatAuthUnknownAsRisk}
assumeTestEmailIfContainsTestingWords: {settings.assumeTestEmailIfContainsTestingWords}
allowlistedDomains: {allowlisted}
blocklistedDomains: {blocklisted}

Hard checks catalog (use exactly):
{catalogJson}

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


def normalizeAiPayload(ai: Dict[str, Any]) -> Dict[str, Any]:
    catalog = hardChecksCatalog()

    hardChecks = ai.get("hardChecks", [])
    if not isinstance(hardChecks, list):
        hardChecks = []

    normalizedHard: List[Dict[str, Any]] = []
    for i in range(len(catalog)):
        c = catalog[i]
        item = hardChecks[i] if i < len(hardChecks) and isinstance(hardChecks[i], dict) else {}

        maxPoints = int(c.get("maxPoints", 0))
        riskPoints = int(clamp(float(item.get("riskPoints", 0) or 0), 0, maxPoints))

        normalizedHard.append({
            "id": c["id"],
            "title": c["title"],
            "maxPoints": maxPoints,
            "triggered": bool(item.get("triggered", False)),
            "riskPoints": riskPoints,
            "evidence": str(item.get("evidence", ""))[:120],
        })

    free = ai.get("freeAssessment", {})
    if not isinstance(free, dict):
        free = {}

    freeRisk = int(clamp(float(free.get("risk", 0) or 0), 0, 100))
    threatType = str(free.get("threatType", "other")).strip().lower()
    if threatType not in ["safe", "phishing", "malware", "bec", "invoice", "other"]:
        threatType = "other"

    keyFindings = free.get("keyFindings", [])
    if not isinstance(keyFindings, list):
        keyFindings = []
    keyFindings = [str(x).strip() for x in keyFindings if str(x).strip()][:4]

    normalizedFree = {
        "risk": freeRisk,
        "threatType": threatType,
        "summary": str(free.get("summary", "")).strip(),
        "keyFindings": keyFindings,
        "recommendedAction": str(free.get("recommendedAction", "")).strip(),
    }

    conf = ai.get("confidence", {})
    if not isinstance(conf, dict):
        conf = {}

    label = str(conf.get("label", "Low")).strip()
    if label not in ["Low", "Medium", "High"]:
        label = "Low"

    score = int(clamp(float(conf.get("score", 0) or 0), 0, 100))

    rationale = conf.get("rationale", [])
    if not isinstance(rationale, list):
        rationale = []
    rationale = [str(x).strip() for x in rationale if str(x).strip()][:4]

    normalizedConfidence = {"label": label, "score": score, "rationale": rationale}

    meta = ai.get("_meta", {})
    if not isinstance(meta, dict):
        meta = {}

    return {
        "hardChecks": normalizedHard,
        "freeAssessment": normalizedFree,
        "confidence": normalizedConfidence,
        "_meta": meta,
    }


def callGeminiScoring(request: AnalyzeRequest, features: EmailFeatures, settings: UserSettings) -> Dict[str, Any]:
    apiKey = os.getenv(geminiApiKeyEnvironmentVariable, "").strip()
    if not apiKey:
        raise RuntimeError(f"Missing {geminiApiKeyEnvironmentVariable} environment variable.")

    client = getGeminiClient(apiKey)
    modelName = pickWorkingModel(client)

    prompt = buildAiScoringPrompt(request, features, settings)
    response = client.models.generate_content(model=modelName, contents=prompt)

    rawText = (response.text or "").strip()
    parsed = parseFirstJsonObject(rawText)
    if not isinstance(parsed, dict):
        raise ValueError("AI response JSON root is not an object.")

    parsed["_meta"] = {"model": modelName}
    return normalizeAiPayload(parsed)


def analyzeWithGemini(request: AnalyzeRequest, features: EmailFeatures, settings: UserSettings) -> Tuple[bool, Dict[str, Any]]:
    start = time.time()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(callGeminiScoring, request, features, settings)
            data = future.result(timeout=aiTimeoutSeconds)

        latencyMs = int((time.time() - start) * 1000)
        data["_meta"]["latencyMs"] = latencyMs
        data["_meta"]["error"] = ""
        return True, data

    except FuturesTimeoutError:
        latencyMs = int((time.time() - start) * 1000)
        return False, {
            "hardChecks": normalizeAiPayload({"hardChecks": []}).get("hardChecks", []),
            "freeAssessment": {"risk": 50, "threatType": "other", "summary": "", "keyFindings": [], "recommendedAction": ""},
            "confidence": {"label": "Low", "score": 0, "rationale": [f"AI timeout after {aiTimeoutSeconds}s"]},
            "_meta": {"model": "", "latencyMs": latencyMs, "error": f"AI timeout after {aiTimeoutSeconds}s"},
        }

    except Exception as exception:
        latencyMs = int((time.time() - start) * 1000)
        return False, {
            "hardChecks": normalizeAiPayload({"hardChecks": []}).get("hardChecks", []),
            "freeAssessment": {"risk": 50, "threatType": "other", "summary": "", "keyFindings": [], "recommendedAction": ""},
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
        maxPoints = int(item.get("maxPoints", 0) or 0)
        riskPoints = int(item.get("riskPoints", 0) or 0)
        triggered = bool(item.get("triggered", False))
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


def buildBreakdown(request: AnalyzeRequest, features: EmailFeatures, hardRisk: int, freeRisk: int, settings: UserSettings) -> Dict[str, Any]:
    return {
        "identity": {
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
            "linkCount": len(features.urls),
            "domains": features.domains,
            "containsHttpLinks": features.containsHttpLinks,
            "containsShortenedLinks": features.containsShortenedLinks,
        },
        "content": {
            "containsMixedScriptText": features.containsMixedScriptText,
        },
        "engine": {
            "hardRisk": hardRisk,
            "freeRisk": freeRisk,
            "weights": {"hard": settings.hardChecksWeight, "free": 1.0 - settings.hardChecksWeight},
            "sensitivity": settings.sensitivity,
            "treatAuthUnknownAsRisk": settings.treatAuthUnknownAsRisk,
            "assumeTestEmailIfContainsTestingWords": settings.assumeTestEmailIfContainsTestingWords,
            "allowlistedDomains": settings.allowlistedDomains,
            "blocklistedDomains": settings.blocklistedDomains,
            "language": settings.language,
        },
    }


def buildUiSummary(aiPayload: Dict[str, Any], hardReasons: List[Dict[str, Any]]) -> Dict[str, Any]:
    free = aiPayload.get("freeAssessment", {}) if isinstance(aiPayload.get("freeAssessment", {}), dict) else {}
    keyFindings = free.get("keyFindings", [])
    if not isinstance(keyFindings, list):
        keyFindings = []

    hardFindings = [
        str(r.get("title", "")).strip()
        for r in sorted(hardReasons, key=lambda x: int(x.get("points", 0)), reverse=True)[:2]
    ]

    merged: List[str] = []
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

    settings = sanitizeSettings(request.settings)
    features = computeFeatures(request.subject, request.sender, request.body)

    aiAvailable, aiPayload = analyzeWithGemini(request, features, settings)

    hardChecks = aiPayload.get("hardChecks", [])
    if not isinstance(hardChecks, list):
        hardChecks = []

    hardRisk, hardReasons = computeHardRisk(hardChecks)

    freeAssessment = aiPayload.get("freeAssessment", {}) if isinstance(aiPayload.get("freeAssessment", {}), dict) else {}
    freeRisk = int(clamp(float(freeAssessment.get("risk", 0) or 0), 0, 100))

    finalRisk = settings.hardChecksWeight * hardRisk + (1.0 - settings.hardChecksWeight) * freeRisk
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

    breakdown = buildBreakdown(request, features, hardRisk, freeRisk, settings)
    ui = buildUiSummary(aiPayload, hardReasons)

    if not aiAvailable:
        ui["recommendedAction"] = ui.get("recommendedAction") or "Unable to analyze reliably. Verify manually before acting."

    response = {
        "score": safetyScore,
        "verdict": verdict if aiAvailable else "Suspicious",
        "confidence": confidenceLabel if aiAvailable else "Low",
        "risk": {
            "final": finalRisk,
            "hard": hardRisk,
            "free": freeRisk,
            "weights": {"hard": settings.hardChecksWeight, "free": 1.0 - settings.hardChecksWeight},
            "confidenceScore": confidenceScore,
            "confidenceRationale": rationale,
            "aiAvailable": aiAvailable,
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
            "confidence": {"label": confidenceLabel, "score": confidenceScore, "rationale": rationale},
        },
        "ui": ui,
        "version": applicationVersion,
        "traceId": traceId,
    }

    return response