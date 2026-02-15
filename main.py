from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from fastapi import FastAPI
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from datetime import datetime, timezone

# =====================================================================
# Application configuration
# =====================================================================

applicationVersion = "3.4.0"

defaultHardChecksWeight = float(os.getenv("HARD_CHECKS_WEIGHT", "0.30"))
aiTimeoutSeconds = float(os.getenv("AI_TIMEOUT_SECONDS", "20.0"))

geminiApiKeyEnvironmentVariable = "GEMINI_API_KEY"
safeBrowsingApiKeyEnvironmentVariable = "GOOGLE_SAFE_BROWSING_API_KEY"

modelCandidates = [
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.0-flash",
]

selectedModel: str | None = None

app = FastAPI(title="ScamurAI Backend", version=applicationVersion)

# =====================================================================
# API models
# =====================================================================

class AuthenticationSummary(BaseModel):
    spf: str = "unknown"
    dkim: str = "unknown"
    dmarc: str = "unknown"

class UserSettings(BaseModel):
    sensitivity: str = "balanced"
    hardChecksWeight: float = Field(defaultHardChecksWeight, ge=0.0, le=1.0)

    allowlistedDomains: List[str] = Field(default_factory=list)
    blocklistedDomains: List[str] = Field(default_factory=list)

    treatAuthUnknownAsRisk: bool = True
    assumeTestEmailIfContainsTestingWords: bool = True

    language: str = "en"
    aiEnabled: bool = True
    viewMode: str = "basic"

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

@dataclass(frozen=True)
class UrlReputation:
    status: str
    maliciousUrls: List[str]
    checkedCount: int
    error: str = ""

# =====================================================================
# Utility helpers
# =====================================================================

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))

def normalizeDomain(value: str) -> str:
    return str(value or "").strip().lower()

def normalizeText(subject: str, body: str) -> str:
    return (subject + "\n" + body).lower().strip()

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

    knownShorteners = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "tiny.cc", "is.gd", "cutt.ly", "rebrand.ly"]
    containsShortenedLinks = any(any(shortener == domain or domain.endswith("." + shortener) for shortener in knownShorteners) for domain in domains)

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

    viewMode = str(settings.viewMode or "basic").strip().lower()
    if viewMode not in ["basic", "advanced"]:
        viewMode = "basic"

    return UserSettings(
        sensitivity=sensitivity,
        hardChecksWeight=hardWeight,
        allowlistedDomains=allowlisted,
        blocklistedDomains=blocklisted,
        treatAuthUnknownAsRisk=bool(settings.treatAuthUnknownAsRisk),
        assumeTestEmailIfContainsTestingWords=bool(settings.assumeTestEmailIfContainsTestingWords),
        language=language,
        aiEnabled=bool(settings.aiEnabled),
        viewMode=viewMode,
    )

def domainIsListed(domain: str, listed: List[str]) -> bool:
    d = normalizeDomain(domain)
    if not d:
        return False
    for item in listed:
        item = normalizeDomain(item)
        if not item:
            continue
        if d == item or d.endswith("." + item):
            return True
    return False

def sanitizeAiErrorReason(reason: Any) -> str:
    s = str(reason or "").strip()
    if not s:
        return ""

    if "RESOURCE_EXHAUSTED" in s or "Quota exceeded" in s or "rate-limits" in s or "generate_content" in s:
        return "Quota exceeded; retry later."

    if "429" in s:
        return "Quota exceeded; retry later."

    if "401" in s or "PERMISSION_DENIED" in s or "UNAUTHENTICATED" in s:
        return "Authentication failed; check API key or permissions."

    if "503" in s or "UNAVAILABLE" in s:
        return "Service temporarily unavailable; retry later."

    if "400" in s or "INVALID_ARGUMENT" in s:
        return "Invalid request; check input and configuration."

    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                err = obj.get("error")
                if isinstance(err, dict):
                    msg = str(err.get("message") or "").strip()
                    if msg:
                        return msg.split("\n")[0].strip()
        except Exception:
            pass

    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 120:
        s = s[:117] + "..."
    return s

# =====================================================================
# Optional enrichment: Google Safe Browsing (URL reputation)
# =====================================================================

def checkUrlReputation(urls: List[str]) -> UrlReputation:
    apiKey = os.getenv(safeBrowsingApiKeyEnvironmentVariable, "").strip()
    if not apiKey:
        return UrlReputation(status="unavailable", maliciousUrls=[], checkedCount=0, error="Missing API key")

    urls = [u for u in urls if isinstance(u, str) and u.strip().lower().startswith(("http://", "https://"))]
    urls = urls[:50]
    if not urls:
        return UrlReputation(status="ok", maliciousUrls=[], checkedCount=0, error="")

    endpoint = "https://safebrowsing.googleapis.com/v4/threatMatches:find?key=" + apiKey

    payload = {
        "client": {"clientId": "ScamurAI", "clientVersion": applicationVersion},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": u} for u in urls],
        },
    }

    requestData = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=requestData,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(raw) if raw else {}

        matches = data.get("matches", [])
        malicious = []
        if isinstance(matches, list):
            for m in matches:
                url = (m or {}).get("threat", {}).get("url", "")
                if url:
                    malicious.append(str(url))

        malicious = sorted(set(malicious))[:20]
        return UrlReputation(status="ok", maliciousUrls=malicious, checkedCount=len(urls), error="")

    except urllib.error.HTTPError as e:
        code = getattr(e, "code", 0)
        if code == 429:
            return UrlReputation(status="rate_limited", maliciousUrls=[], checkedCount=len(urls), error="Rate limited")
        return UrlReputation(status="error", maliciousUrls=[], checkedCount=len(urls), error=f"HTTP {code}")

    except Exception as e:
        return UrlReputation(status="error", maliciousUrls=[], checkedCount=len(urls), error=str(e))

# =====================================================================
# Gemini integration (AI-active path)
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
        {"id": "maliciousUrlReputation", "title": "URL reputation indicates malicious or phishing", "maxPoints": 50},
        {"id": "brandImpersonation", "title": "Brand impersonation or fake login workflow", "maxPoints": 30},
    ]

def buildAiScoringPrompt(
    request: AnalyzeRequest,
    features: EmailFeatures,
    settings: UserSettings,
    urlRep: UrlReputation,
) -> str:
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
        promptSensitivityRule = "- Sensitivity is BALANCED: weigh evidence proportionally.\n"

    allowlisted = ", ".join(settings.allowlistedDomains) if settings.allowlistedDomains else "(none)"
    blocklisted = ", ".join(settings.blocklistedDomains) if settings.blocklistedDomains else "(none)"

    urlRepLine = f"urlReputation: status={urlRep.status}, maliciousUrls={len(urlRep.maliciousUrls)}"
    if urlRep.maliciousUrls:
        urlRepLine += f", sample={urlRep.maliciousUrls[:3]}"

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
  - evidence must reference the exact indicator (keyword/auth status/domain/reputation), <= 120 chars.

Free assessment rules:
- freeAssessment.risk: holistic risk 0..100 (0 safe, 100 highly malicious).
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
{urlRepLine}

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

def callGeminiScoring(request: AnalyzeRequest, features: EmailFeatures, settings: UserSettings, urlRep: UrlReputation) -> Dict[str, Any]:
    apiKey = os.getenv(geminiApiKeyEnvironmentVariable, "").strip()
    if not apiKey:
        raise RuntimeError(f"Missing {geminiApiKeyEnvironmentVariable} environment variable.")

    client = getGeminiClient(apiKey)
    modelName = pickWorkingModel(client)

    prompt = buildAiScoringPrompt(request, features, settings, urlRep)
    response = client.models.generate_content(model=modelName, contents=prompt)

    rawText = (response.text or "").strip()
    parsed = parseFirstJsonObject(rawText)
    if not isinstance(parsed, dict):
        raise ValueError("AI response JSON root is not an object.")

    parsed["_meta"] = {"model": modelName}
    return normalizeAiPayload(parsed)

def analyzeWithGemini(request: AnalyzeRequest, features: EmailFeatures, settings: UserSettings, urlRep: UrlReputation) -> Tuple[bool, Dict[str, Any]]:
    start = time.time()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(callGeminiScoring, request, features, settings, urlRep)
            data = future.result(timeout=aiTimeoutSeconds)

        latencyMs = int((time.time() - start) * 1000)
        data["_meta"]["latencyMs"] = latencyMs
        data["_meta"]["error"] = ""
        return True, data

    except FuturesTimeoutError:
        latencyMs = int((time.time() - start) * 1000)
        return False, {
            "hardChecks": [],
            "freeAssessment": {"risk": 0, "threatType": "other", "summary": "", "keyFindings": [], "recommendedAction": ""},
            "confidence": {"label": "Low", "score": 0, "rationale": [f"AI timeout after {aiTimeoutSeconds}s"]},
            "_meta": {"model": "", "latencyMs": latencyMs, "error": f"AI timeout after {aiTimeoutSeconds}s"},
        }

    except Exception as exception:
        latencyMs = int((time.time() - start) * 1000)
        return False, {
            "hardChecks": [],
            "freeAssessment": {"risk": 0, "threatType": "other", "summary": "", "keyFindings": [], "recommendedAction": ""},
            "confidence": {"label": "Low", "score": 0, "rationale": [str(exception)]},
            "_meta": {"model": "", "latencyMs": latencyMs, "error": str(exception)},
        }


# =====================================================================
# Fallback deterministic scoring (AI-unavailable path)
# =====================================================================

def fallbackHardChecks(
    request: AnalyzeRequest,
    features: EmailFeatures,
    settings: UserSettings,
    urlRep: UrlReputation,
) -> List[Dict[str, Any]]:
    normalized = normalizeText(request.subject, request.body)

    testingWords = [
        "test", "testing", "does this work", "what is your score", "scamurai", "sandbox", "demo",
        "phishing simulation", "security training", "awareness training",
    ]

    credentialWords = [
        "password", "passcode", "one-time password", "otp", "verify your account", "verify your identity",
        "confirm your identity", "security check", "login", "sign in", "reset your password",
        "credit card", "card number", "cvv", "billing", "payment details",
        "bank account", "ssn", "social security",
    ]

    pressureWords = [
        "urgent", "immediately", "act now", "within 24 hours", "today", "final notice",
        "account suspended", "account locked", "limited access", "unusual activity",
        "your mailbox is full", "verify now", "failure to comply",
    ]

    brandWords = [
        "microsoft", "google", "gmail", "outlook", "apple", "icloud", "amazon", "paypal", "bank",
        "dhl", "fedex", "ups", "netflix",
    ]

    def hasAny(needles: List[str]) -> str:
        for n in needles:
            if n in normalized:
                return n
        return ""

    fromDomain = normalizeDomain(request.fromDomain)
    urlDomains = features.domains

    allowlistedFrom = domainIsListed(fromDomain, settings.allowlistedDomains)
    blocklistedFrom = domainIsListed(fromDomain, settings.blocklistedDomains)
    blocklistedAnyUrl = any(domainIsListed(d, settings.blocklistedDomains) for d in urlDomains)

    authFail = request.authentication.spf == "fail" or request.authentication.dkim == "fail" or request.authentication.dmarc == "fail"
    authAllUnknown = request.authentication.spf == "unknown" and request.authentication.dkim == "unknown" and request.authentication.dmarc == "unknown"

    replyMismatch = bool(request.replyToDomain) and bool(request.fromDomain) and normalizeDomain(request.replyToDomain) != normalizeDomain(request.fromDomain)
    returnPathMismatch = bool(request.returnPathDomain) and bool(request.fromDomain) and normalizeDomain(request.returnPathDomain) != normalizeDomain(request.fromDomain)

    testingHit = hasAny(testingWords) if settings.assumeTestEmailIfContainsTestingWords else ""

    checks = []
    for c in hardChecksCatalog():
        checks.append({
            "id": c["id"],
            "title": c["title"],
            "maxPoints": int(c["maxPoints"]),
            "triggered": False,
            "riskPoints": 0,
            "evidence": "",
        })

    def setCheck(idValue: str, triggered: bool, points: int, evidence: str) -> None:
        for item in checks:
            if item["id"] == idValue:
                item["triggered"] = bool(triggered)
                item["riskPoints"] = int(clamp(points, 0, int(item["maxPoints"])))
                item["evidence"] = str(evidence)[:120]
                return

    if authFail:
        setCheck("authFails", True, 40, f"auth: spf={request.authentication.spf}, dkim={request.authentication.dkim}, dmarc={request.authentication.dmarc}")

    if authAllUnknown and settings.treatAuthUnknownAsRisk:
        setCheck("authMissing", True, 12, "auth: spf=unknown, dkim=unknown, dmarc=unknown")

    if replyMismatch or returnPathMismatch:
        ev = []
        if replyMismatch:
            ev.append(f"replyToDomain={request.replyToDomain} != fromDomain={request.fromDomain}")
        if returnPathMismatch:
            ev.append(f"returnPathDomain={request.returnPathDomain} != fromDomain={request.fromDomain}")
        setCheck("replyToMismatch", True, 18, "; ".join(ev))

    if features.containsMixedScriptText:
        setCheck("lookalikeOrHomograph", True, 18, "mixed-script characters detected in subject/sender/body")

    if features.containsHttpLinks:
        setCheck("httpLink", True, 10, "at least one URL starts with http://")

    if features.containsShortenedLinks:
        setCheck("shortenedLink", True, 18, "URL shortener domain detected")

    credHit = hasAny(credentialWords)
    if credHit:
        setCheck("credentialHarvesting", True, 34, f"keyword: {credHit}")

    pressureHit = hasAny(pressureWords)
    if pressureHit:
        setCheck("pressureLanguage", True, 18, f"keyword: {pressureHit}")

    if urlRep.status == "ok" and urlRep.maliciousUrls:
        setCheck("maliciousUrlReputation", True, 50, f"reputation flagged: {urlRep.maliciousUrls[0]}")

    brandHit = hasAny(brandWords)
    if brandHit:
        setCheck("brandImpersonation", True, 18, f"brand keyword: {brandHit}")

    if blocklistedFrom or blocklistedAnyUrl:
        setCheck("maliciousUrlReputation", True, 50, "blocklisted domain matched (user blocklist)")

    if allowlistedFrom and not authFail and not urlRep.maliciousUrls and not credHit and not pressureHit and not features.containsShortenedLinks:
        for item in checks:
            if item["riskPoints"] > 0:
                item["riskPoints"] = int(clamp(item["riskPoints"] - 6, 0, item["maxPoints"]))
        setCheck("maliciousUrlReputation", False, 0, "")

    if testingHit:
        for item in checks:
            item["riskPoints"] = int(clamp(item["riskPoints"] * 0.6, 0, item["maxPoints"]))
            if item["riskPoints"] > 0 and item["evidence"]:
                item["evidence"] = (item["evidence"][:90] + " (test-like)")

    if settings.sensitivity == "strict":
        for item in checks:
            if item["riskPoints"] > 0:
                item["riskPoints"] = int(clamp(item["riskPoints"] + 2, 0, item["maxPoints"]))
    elif settings.sensitivity == "lenient":
        for item in checks:
            if item["riskPoints"] > 0:
                item["riskPoints"] = int(clamp(item["riskPoints"] - 2, 0, item["maxPoints"]))

    return checks


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


# =====================================================================
# UI mapping
# =====================================================================

def utcNowIso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def buildBreakdown(
    request: AnalyzeRequest,
    features: EmailFeatures,
    urlRep: UrlReputation,
    hardRisk: int,
    freeRisk: int,
    settings: UserSettings,
) -> Dict[str, Any]:
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
            "urlReputation": {
                "status": urlRep.status,
                "checkedCount": urlRep.checkedCount,
                "maliciousCount": len(urlRep.maliciousUrls),
                "maliciousUrlsSample": urlRep.maliciousUrls[:3],
                "error": urlRep.error,
            },
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
            "aiEnabled": settings.aiEnabled,
            "viewMode": settings.viewMode,
        },
    }


def buildUiSummary(aiPayload: Dict[str, Any], hardReasons: List[Dict[str, Any]], language: str) -> Dict[str, Any]:
    free = aiPayload.get("freeAssessment", {}) if isinstance(aiPayload.get("freeAssessment", {}), dict) else {}
    keyFindings = free.get("keyFindings", [])
    if not isinstance(keyFindings, list):
        keyFindings = []

    hardFindings = [
        str(r.get("title", "")).strip()
        for r in sorted(hardReasons, key=lambda x: int(x.get("points", 0)), reverse=True)[:3]
    ]

    merged: List[str] = []
    for item in hardFindings + [str(x).strip() for x in keyFindings]:
        if item and item not in merged:
            merged.append(item)
        if len(merged) >= 4:
            break

    if not merged:
        merged = ["No meaningful risk indicators detected."] if language != "he" else ["לא זוהו אינדיקציות סיכון משמעותיות."]

    recommendedAction = str(free.get("recommendedAction", "")).strip()

    return {
        "keyFindings": merged,
        "recommendedAction": recommendedAction,
    }


def defaultActionForFallback(language: str) -> str:
    if language == "he":
        return "ה-AI לא זמין כרגע; מומלץ לא ללחוץ ולאמת מול ערוץ רשמי."
    return "AI is unavailable; do not click and verify via an official channel."


# =====================================================================
# Routes
# =====================================================================

@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "running", "version": applicationVersion}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    traceId = str(uuid4())
    analyzedAt = utcNowIso()

    settings = sanitizeSettings(request.settings)
    features = computeFeatures(request.subject, request.sender, request.body)

    urlRep = checkUrlReputation(features.urls)

    aiAvailable = False
    aiPayload: Dict[str, Any] = {}

    aiDisabledByUser = not settings.aiEnabled
    if not aiDisabledByUser:
        aiAvailable, aiPayload = analyzeWithGemini(request, features, settings, urlRep)

    aiStatus = "on" if aiAvailable else "off"

    if aiDisabledByUser:
        aiOffReasonRaw = "Disabled in Settings"
    else:
        aiOffReasonRaw = str((aiPayload.get("_meta", {}) or {}).get("error", "")).strip() if isinstance(aiPayload,
                                                                                                        dict) else ""
        if not aiOffReasonRaw:
            aiOffReasonRaw = "AI unavailable"

    aiOffReason = sanitizeAiErrorReason(aiOffReasonRaw) or "AI unavailable"

    if aiAvailable:
        hardChecks = aiPayload.get("hardChecks", [])
        if not isinstance(hardChecks, list):
            hardChecks = []
        else:
            hardChecks = fallbackHardChecks(request, features, settings, urlRep)

            if aiDisabledByUser:
                aiReason = "AI disabled by user."
            else:
                aiReason = "AI unavailable; used deterministic fallback checks only."

            aiPayload = {
                "hardChecks": hardChecks,
                "freeAssessment": {
                    "risk": 0,
                    "threatType": "other",
                    "summary": "",
                    "keyFindings": [],
                    "recommendedAction": "",
                },
                "confidence": {
                    "label": "Low",
                    "score": 0,
                    "rationale": [aiReason],
                },
                "_meta": {
                    "model": "",
                    "latencyMs": 0,
                    "error": aiOffReason,
                },
            }

    hardRisk, hardReasons = computeHardRisk(hardChecks)

    freeAssessment = aiPayload.get("freeAssessment", {}) if isinstance(aiPayload.get("freeAssessment", {}), dict) else {}
    freeRisk = int(clamp(float(freeAssessment.get("risk", 0) or 0), 0, 100))

    if not aiAvailable:
        finalRisk = hardRisk
    else:
        finalRisk = settings.hardChecksWeight * hardRisk + (1.0 - settings.hardChecksWeight) * freeRisk
        finalRisk = int(clamp(finalRisk, 0, 100))

    verdict = verdictFromRisk(finalRisk)
    safetyScore = int(clamp(100 - finalRisk, 0, 100))

    confidence = aiPayload.get("confidence", {}) if isinstance(aiPayload.get("confidence", {}), dict) else {}
    confidenceLabel = str(confidence.get("label", "Low"))
    if confidenceLabel not in ["Low", "Medium", "High"]:
        confidenceLabel = "Low"
    if not aiAvailable:
        confidenceLabel = "Low"

    confidenceScore = int(clamp(float(confidence.get("score", 0) or 0), 0, 100))
    if not aiAvailable:
        confidenceScore = 0

    rationale = confidence.get("rationale", [])
    if not isinstance(rationale, list):
        rationale = []
    rationale = [str(x).strip() for x in rationale if str(x).strip()][:4]

    breakdown = buildBreakdown(request, features, urlRep, hardRisk, freeRisk, settings)
    ui = buildUiSummary(aiPayload, hardReasons, settings.language)

    if not ui.get("recommendedAction"):
        ui["recommendedAction"] = defaultActionForFallback(settings.language) if not aiAvailable else ""

    ui["meta"] = {
        "analyzedAt": analyzedAt,
        "ai": {
            "status": aiStatus,
            "reason": "" if aiAvailable else aiOffReason,
            "model": str((aiPayload.get("_meta", {}) or {}).get("model", "")) if isinstance(aiPayload, dict) else "",
            "latencyMs": int((aiPayload.get("_meta", {}) or {}).get("latencyMs", 0)) if isinstance(aiPayload, dict) else 0,
        },
        "safeBrowsing": {
            "status": urlRep.status,
            "checkedCount": urlRep.checkedCount,
            "maliciousCount": len(urlRep.maliciousUrls),
            "error": urlRep.error,
        },
        "confidenceHelp": "Confidence measures the stability of the verdict (Safe/Suspicious/Malicious), not the exact number.",
    }
    if settings.language == "he":
        ui["meta"]["confidenceHelp"] = "ביטחון (Confidence) מודד יציבות של הסיווג (Safe/Suspicious/Malicious), לא את הדיוק של המספר."

    response = {
        "score": safetyScore,
        "verdict": verdict,
        "confidence": confidenceLabel,
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

            "error": "" if aiAvailable else aiOffReason,
            "hardChecks": hardChecks,
            "confidence": {"label": confidenceLabel, "score": confidenceScore, "rationale": rationale},
            "status": aiStatus,
            "statusReason": "" if aiAvailable else aiOffReason,
        },
        "ui": ui,
        "version": applicationVersion,
        "traceId": traceId,
    }

    return response