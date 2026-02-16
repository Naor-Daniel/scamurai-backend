# ScamurAI

Hybrid Gmail Security Analysis System  
Deterministic Engine + LLM Assessment + Google Safe Browsing

Author: Naor Daniel  
Year: 2026  

---

## Overview

ScamurAI is a Gmail Add-on security analysis system that evaluates inbound emails using a hybrid architecture combining:

- Deterministic rule-based security checks  
- LLM-based semantic threat assessment (Gemini)  
- Google Safe Browsing URL reputation validation  
- Explicit confidence modeling  
- Stable per-message analysis snapshots  

The system is designed to balance explainability, deterministic guarantees, and AI-assisted reasoning, with controlled degradation under failure conditions.

---

## Architecture

ScamurAI consists of two layers:

### Frontend – Gmail Add-on (Apps Script)

Responsibilities:
- Fetch Gmail message content and metadata  
- Build structured analysis payload  
- Render basic and advanced views  
- Maintain per-message analysis snapshot  
- Manage user settings and history  
- Enforce cache-first navigation  

The frontend does not compute risk. All scoring logic resides in the backend.

---

### Backend – FastAPI

Responsibilities:
- Deterministic risk evaluation  
- Gemini-based semantic assessment  
- Google Safe Browsing integration  
- Risk aggregation and confidence modeling  
- Structured JSON normalization  

The backend is stateless and returns a fully normalized analysis object.

---

## Analysis Pipeline

Each email is evaluated across three independent signal layers.

### 1. Deterministic Engine (“Hard Checks”)

Evaluates:
- SPF / DKIM / DMARC authentication  
- Domain alignment (From / Reply-To / Return-Path)  
- Credential harvesting patterns  
- Urgency / pressure language  
- Lookalike or spoofing indicators  
- Structural anomalies  

This layer always runs.

Outputs:
- Hard risk score  
- Structured signals  
- Explicit evidence  

---

### 2. LLM Semantic Assessment (Optional)

When enabled, Gemini evaluates contextual and semantic threat indicators.

Outputs:
- Threat classification  
- Summary and key findings  
- Recommended action  
- Confidence rationale  

If AI fails (quota, timeout, authentication, etc.), the system:
- Falls back to deterministic-only scoring  
- Marks AI as unavailable  
- Sanitizes error messages  

---

### 3. Google Safe Browsing

Extracted URLs are validated via Safe Browsing.

Outputs:
- Reputation status  
- Checked URL count  
- Malicious URL count  

Safe Browsing acts as an additional signal and does not override core logic.

---

## Risk Aggregation Model

Final risk is computed as:

    finalRisk = hardWeight * hardRisk + (1 - hardWeight) * aiRisk  
    safetyScore = 100 - finalRisk  

Where:
- hardWeight ∈ [0,1]  
- aiRisk = 0 if AI unavailable  

Outputs:
- Safety score (0–100)  
- Verdict (Safe / Suspicious / Malicious)  
- Confidence score and label  

---

## Confidence Model

Confidence represents classification stability, not probability.

Confidence decreases when:
- Evidence is weak or conflicting  
- AI is unavailable  
- Signals resemble synthetic/test patterns  

Confidence is explicitly separated from the safety score.

---

## Stability & Reproducibility

Each messageId receives a stored analysis snapshot.

Navigation never triggers re-analysis.  
Only explicit “Refresh”:
- Clears cache  
- Removes snapshot  
- Re-runs backend analysis  

This guarantees deterministic UI behavior and prevents unintended quota consumption.

---

## Failure Handling

Under AI or API failure:
- Deterministic engine continues  
- AI status is explicitly marked  
- Errors are sanitized before display  
- No raw API responses are exposed  

---

## Storage Model

Per Google account:
- Settings → User Properties  
- History → User Properties  
- Message snapshots → User Properties  
- Short-term cache → User Cache (TTL-based)  

History is display-only and does not influence scoring.

---

## Design Goals

- Deterministic fallback under AI failure  
- Explicit AI state visibility  
- Separation of concerns  
- Explainable scoring  
- Stable per-message behavior  
- Production-oriented structure  

---
