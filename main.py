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

applicationVersion = "2.2.0"

rulesWeight = 0.30
aiWeight = 0.70

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

class AnalyzeRequest(BaseModel):
    """Request payload sent by the Gmail add-on."""
    subject: str = ""
    sender: str = ""
    body: str = ""


class RiskReason(BaseModel):
    """
    A single deterministic signal from the rules engine.

    points is a contribution to risk (0..100). It is not a probability.
    """
    id: str
    title: str
    points: int = Field(ge=0, le=100)
    evidence: str = ""


class AnalyzeResponse(BaseModel):
    """Response payload returned to the Gmail add-on."""
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
    """Derived features extracted from the email for rules + AI prompt conditioning."""
    urls: List[str]
    domains: List[str]
    containsHttpLinks: bool
    containsShortenedLinks: bool
    containsMixedScriptText: bool


# =====================================================================
# Utility helpers
# =====================================================================

def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp numeric value to [minimum, maximum]."""
    return max(minimum, min(maximum, value))


def normalizeText(subject: str, body: str) -> str:
    """Normalize email content for deterministic keyword checks."""
    return (subject + "\n" + body).lower().strip()


def extractUrls(text: str) -> List[str]:
    """Extract up to 50 URLs from the email body."""
    if not text:
        return []
    return re.findall(r"\bhttps?://[^\s<>\"]+", text, flags=re.IGNORECASE)[:50]


def extractDomain(url: str) -> str:
    """Extract host portion from a URL."""
    match = re.match(r"^https?://([^/]+)", url.strip(), flags=re.IGNORECASE)
    return match.group(1).lower() if match else ""


def containsMixedScript(text: str) -> bool:
    """
    Heuristic for common spoofing: detect a mix of Latin and non-ASCII characters.
    Conservative by design to avoid false positives.
    """
    if not text:
        return False
    hasLatin = bool(re.search(r"[A-Za-z]", text))
    hasNonAscii = bool(re.search(r"[^\x00-\x7F]", text))
    return hasLatin and hasNonAscii


def computeFeatures(subject: str, sender: str, body: str) -> EmailFeatures:
    """Compute structured signals used by rules and prompt conditioning."""
    urls = extractUrls(body)
    domains = sorted(set([extractDomain(url) for url in urls if url]))[:20]

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
    """Map risk score to a coarse verdict."""
    if risk >= 75:
        return "Malicious"
    if risk >= 35:
        return "Suspicious"
    return "Safe"


def confidenceFromClassification(
    finalRisk: int,
    rulesRisk: int,
    aiRisk: int,
    aiAvailable: bool,
    guardrailApplied: bool
) -> str:
    """
    Confidence is about classification stability (Safe/Suspicious/Malicious),
    not about the raw risk magnitude.

    Signals:
    - distance from decision boundaries (35, 75)
    - agreement between rules and AI (if AI available)
    - guardrailApplied indicates disagreement strong enough to trigger a floor
    """
    def level_from_distance(distance: int) -> int:
        if distance >= 25:
            return 2  # High
        if distance >= 12:
            return 1  # Medium
        return 0  # Low

    def clamp_level(level: int) -> int:
        return max(0, min(2, level))

    if finalRisk < 35:
        distance = 35 - finalRisk
    elif finalRisk >= 75:
        distance = finalRisk - 75
    else:
        distance = min(finalRisk - 35, 75 - finalRisk)

    level = level_from_distance(int(distance))

    if not aiAvailable:
        return ["Low", "Medium", "High"][min(level, 1)]

    disagreement = abs(int(rulesRisk) - int(aiRisk))
    if disagreement > 60:
        level -= 2
    elif disagreement > 40:
        level -= 1

    if guardrailApplied:
        level -= 1

    level = clamp_level(level)
    return ["Low", "Medium", "High"][level]

def confidenceFromAgreement(rulesRisk: int, aiRisk: int, aiAvailable: bool) -> str:
    """Compute a user-facing confidence label."""
    if not aiAvailable:
        return "Low"
    difference = abs(rulesRisk - aiRisk)
    if difference <= 15:
        return "High"
    if difference <= 35:
        return "Medium"
    return "Low"


# =====================================================================
# Rules engine
# =====================================================================

def runRulesEngine(subject: str, sender: str, body: str, features: EmailFeatures) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Produce a deterministic rules-based risk score and structured breakdown.

    Returns:
        rulesRisk: int 0..100
        breakdown: dict with component risks and short notes
        reasons: list of signals (id/title/points/evidence)
    """
    normalized = normalizeText(subject, body)

    reasons: List[Dict[str, Any]] = []
    breakdown: Dict[str, Any] = {
        "sender": {"risk": 0, "notes": []},
        "links": {"risk": 0, "notes": [], "linkCount": len(features.urls), "domains": features.domains},
        "content": {"risk": 0, "notes": []},
        "attachments": {"risk": 0, "notes": []},
    }

    totalRisk = 0

    urgencyKeywords = ["urgent", "immediately", "verify", "password", "suspended", "limited", "invoice", "payment"]
    matchedUrgency = next((keyword for keyword in urgencyKeywords if keyword in normalized), None)
    if matchedUrgency:
        points = 60
        totalRisk += points
        breakdown["content"]["risk"] = max(breakdown["content"]["risk"], points)
        breakdown["content"]["notes"].append(f"Pressure language detected: '{matchedUrgency}'")
        reasons.append({
            "id": "pressureLanguage",
            "title": "Pressure or urgency language",
            "points": points,
            "evidence": f"Keyword '{matchedUrgency}' appears in subject/body",
        })

    if features.containsShortenedLinks:
        points = 25
        totalRisk += points
        breakdown["links"]["risk"] = max(breakdown["links"]["risk"], points)
        breakdown["links"]["notes"].append("Link shortener detected")
        reasons.append({
            "id": "shortenedLinks",
            "title": "URL shortener detected",
            "points": points,
            "evidence": "At least one URL uses a known shortener domain",
        })

    if features.containsHttpLinks:
        points = 10
        totalRisk += points
        breakdown["links"]["risk"] = max(breakdown["links"]["risk"], points)
        breakdown["links"]["notes"].append("Non-HTTPS link detected")
        reasons.append({
            "id": "nonHttpsLinks",
            "title": "Non-HTTPS link",
            "points": points,
            "evidence": "At least one URL starts with http://",
        })

    if features.containsMixedScriptText:
        points = 15
        totalRisk += points
        breakdown["sender"]["risk"] = max(breakdown["sender"]["risk"], points)
        breakdown["sender"]["notes"].append("Mixed-script text may indicate spoofing")
        reasons.append({
            "id": "mixedScript",
            "title": "Potential homograph / spoofing indicators",
            "points": points,
            "evidence": "Latin characters appear together with non-ASCII characters",
        })

    rulesRisk = int(clamp(totalRisk, 0, 100))
    for sectionName in ["sender", "links", "content", "attachments"]:
        breakdown[sectionName]["risk"] = int(clamp(breakdown[sectionName]["risk"], 0, 100))

    return rulesRisk, breakdown, reasons


# =====================================================================
# Gemini integration
# =====================================================================

def buildAiPrompt(subject: str, sender: str, body: str, features: EmailFeatures) -> str:
    """
    Build a strict prompt asking Gemini to produce machine-parseable JSON only.

    The JSON is used directly in a user-facing security UI, so the model must:
      - Avoid hallucinations
      - Be concise
      - Provide evidence-based findings
    """
    schema = {
        "aiRisk": "integer 0..100",
        "threatType": "one of: safe|phishing|malware|bec|invoice|other",
        "summary": "max 2 short sentences",
        "keyFindings": ["max 4 bullets, each <= 14 words, evidence-based"],
        "recommendedAction": "one short sentence",
    }

    prompt = f"""
You are an email security analyst.

Return ONLY valid JSON that matches this schema exactly:
{json.dumps(schema, indent=2)}

Hard rules:
- Output must be JSON only. No markdown, no commentary.
- aiRisk is 0..100 where 0 is safe, 100 is highly malicious.
- Do NOT invent facts. If something cannot be verified from the email text, do not claim it.
- summary: <= 2 short sentences.
- keyFindings: <= 4 items, each <= 14 words, focus on what matters for a user.
- recommendedAction: exactly 1 short sentence, user-facing.

Email context:
Subject: {subject}
Sender: {sender}

Extracted domains: {", ".join(features.domains) if features.domains else "(none)"}
URL count: {len(features.urls)}

Body (truncated):
{body[:8000]}
""".strip()

    return prompt


def getGeminiClient(apiKey: str) -> genai.Client:
    """Create a Gemini client pinned to the stable v1 API."""
    return genai.Client(
        api_key=apiKey,
        http_options=types.HttpOptions(api_version="v1"),
    )


def pickWorkingModel(client: genai.Client) -> str:
    """
    Pick the first model that supports generateContent in the current project.

    Caches the selection in-process to avoid repeated test calls.
    """
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
    """Extract and parse the first JSON object from a string."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("AI response did not contain a JSON object.")
    return json.loads(match.group(0))


def callGemini(subject: str, sender: str, body: str, features: EmailFeatures) -> Dict[str, Any]:
    """Call Gemini and return the parsed JSON payload."""
    apiKey = os.getenv(geminiApiKeyEnvironmentVariable, "").strip()
    if not apiKey:
        raise RuntimeError(f"Missing {geminiApiKeyEnvironmentVariable} environment variable.")

    client = getGeminiClient(apiKey)
    modelName = pickWorkingModel(client)

    prompt = buildAiPrompt(subject, sender, body, features)
    response = client.models.generate_content(model=modelName, contents=prompt)

    rawText = (response.text or "").strip()
    parsed = parseFirstJsonObject(rawText)

    aiRisk = int(clamp(float(parsed.get("aiRisk", 0)), 0, 100))
    threatType = str(parsed.get("threatType", "other"))
    summary = str(parsed.get("summary", "")).strip()

    keyFindingsValue = parsed.get("keyFindings", [])
    if not isinstance(keyFindingsValue, list):
        keyFindingsValue = []
    keyFindings = [str(item).strip() for item in keyFindingsValue if str(item).strip()][:4]

    recommendedAction = str(parsed.get("recommendedAction", "")).strip()

    return {
        "aiRisk": aiRisk,
        "threatType": threatType,
        "summary": summary,
        "keyFindings": keyFindings,
        "recommendedAction": recommendedAction,
        "model": modelName,
    }


def analyzeWithGemini(subject: str, sender: str, body: str, features: EmailFeatures) -> Tuple[bool, Dict[str, Any]]:
    """
    Analyze email with Gemini using a hard timeout.

    Returns:
        (aiAvailable, aiPayload)
    """
    start = time.time()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(callGemini, subject, sender, body, features)
            aiData = future.result(timeout=aiTimeoutSeconds)

        latencyMs = int((time.time() - start) * 1000)
        aiData["latencyMs"] = latencyMs
        aiData["error"] = ""
        return True, aiData

    except FuturesTimeoutError:
        latencyMs = int((time.time() - start) * 1000)
        return False, {
            "aiRisk": 0,
            "threatType": "other",
            "summary": "",
            "keyFindings": [],
            "recommendedAction": "",
            "model": "",
            "latencyMs": latencyMs,
            "error": f"AI timeout after {aiTimeoutSeconds}s",
        }

    except Exception as exception:
        latencyMs = int((time.time() - start) * 1000)
        return False, {
            "aiRisk": 0,
            "threatType": "other",
            "summary": "",
            "keyFindings": [],
            "recommendedAction": "",
            "model": "",
            "latencyMs": latencyMs,
            "error": str(exception),
        }


# =====================================================================
# Risk fusion (rules + AI)
# =====================================================================

def fuseRisk(rulesRisk: int, aiRisk: int, aiAvailable: bool) -> Tuple[int, bool]:
    """
    Fuse risk into a single score with a guardrail.

    Guardrail: if rules detect very high risk, the final risk cannot drop too low.
    """
    if not aiAvailable:
        return int(clamp(rulesRisk, 0, 100)), False

    base = rulesWeight * rulesRisk + aiWeight * aiRisk
    minimum = 0.50 * rulesRisk if rulesRisk >= 70 else 0.0
    finalRisk = max(base, minimum)
    guardrailApplied = finalRisk != base

    return int(clamp(finalRisk, 0, 100)), guardrailApplied


def buildUiSummary(verdict: str, confidence: str, reasons: List[Dict[str, Any]], aiData: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build compact UI fields for the add-on.

    This is what the user should see first: short findings and one action.
    """
    topReasons = sorted(reasons, key=lambda item: int(item.get("points", 0)), reverse=True)[:3]
    ruleFindings = [str(r.get("title", "")).strip() for r in topReasons if str(r.get("title", "")).strip()]

    aiFindings = aiData.get("keyFindings", [])
    if not isinstance(aiFindings, list):
        aiFindings = []

    mergedFindings = []
    for item in ruleFindings + [str(x).strip() for x in aiFindings if str(x).strip()]:
        if item and item not in mergedFindings:
            mergedFindings.append(item)
        if len(mergedFindings) >= 4:
            break

    recommendedAction = aiData.get("recommendedAction", "")
    if not recommendedAction:
        if verdict == "Malicious":
            recommendedAction = "Do not click anything; report as phishing and verify via official channel."
        elif verdict == "Suspicious":
            recommendedAction = "Verify sender and links before taking any action."
        else:
            recommendedAction = "No action needed; stay cautious with unexpected requests."

    return {
        "keyFindings": mergedFindings,
        "recommendedAction": recommendedAction,
    }


# =====================================================================
# FastAPI routes
# =====================================================================

@app.get("/")
def health() -> Dict[str, str]:
    """Health endpoint used by Render and manual checks."""
    return {"status": "running", "version": applicationVersion}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    """Analyze a single email and return a structured security assessment."""
    traceId = str(uuid4())

    subject = request.subject or ""
    sender = request.sender or ""
    body = request.body or ""

    features = computeFeatures(subject, sender, body)
    rulesRisk, breakdown, reasons = runRulesEngine(subject, sender, body, features)

    aiAvailable, aiData = analyzeWithGemini(subject, sender, body, features)
    aiRisk = int(aiData.get("aiRisk", 0)) if aiAvailable else 0

    finalRisk, guardrailApplied = fuseRisk(rulesRisk, aiRisk, aiAvailable)
    verdict = verdictFromRisk(finalRisk)
    confidence = confidenceFromClassification(
        finalRisk=finalRisk,
        rulesRisk=rulesRisk,
        aiRisk=aiRisk,
        aiAvailable=aiAvailable,
        guardrailApplied=guardrailApplied
    )
    safetyScore = int(clamp(100 - finalRisk, 0, 100))

    ui = buildUiSummary(verdict, confidence, reasons, aiData)

    response = {
        "score": safetyScore,
        "verdict": verdict,
        "confidence": confidence,
        "risk": {
            "final": finalRisk,
            "rules": int(clamp(rulesRisk, 0, 100)),
            "ai": int(clamp(aiRisk, 0, 100)),
            "weights": {"rules": rulesWeight, "ai": aiWeight},
            "guardrailApplied": guardrailApplied,
        },
        "breakdown": breakdown,
        "reasons": reasons,
        "ai": {
            "summary": aiData.get("summary", ""),
            "threatType": aiData.get("threatType", "other"),
            "keyFindings": aiData.get("keyFindings", []),
            "recommendedAction": aiData.get("recommendedAction", ""),
            "model": aiData.get("model", ""),
            "latencyMs": aiData.get("latencyMs", 0),
            "error": aiData.get("error", ""),
        },
        "ui": ui,
        "version": applicationVersion,
        "traceId": traceId,
    }

    return response