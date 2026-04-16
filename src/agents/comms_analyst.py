"""
Comms Analyst Agent.

Prende sms.json + mails.json + (opzionale) trascrizioni audio,
produce un JSON summary con:
- phishing_signals
- legitimate_counterparties (aiuta il Reviewer a non flaggare tx verso questi)
- suspicious_keywords
- notes

Gira su modello "ctx" (Gemini Flash): finestra lunga, economico.
"""
from __future__ import annotations
import json
from typing import List, Dict, Any

from src.agents.llm_client import llm_call
from src.utils.io import safe_json_loads
from src.utils.logging import log


SYSTEM = """You are a communications intelligence analyst for a fraud-detection pipeline.
You receive samples of SMS, emails and (optionally) call transcripts exchanged
by citizens of the digital metropolis "Reply Mirror".

Your task: extract fraud-relevant signals. Return ONLY valid JSON, no prose:
{
  "phishing_signals": ["brief description", ...],
  "legitimate_counterparties": ["company/person names mentioned in normal context"],
  "suspicious_keywords": ["word/phrase", ...],
  "victims_suspected": ["user_id/name of users seemingly targeted by scams"],
  "notes": "2-4 sentences summary"
}
Keep each list ≤ 12 items. Be precise, do not invent names not present in the data."""


def _truncate(s: str, n: int) -> str:
    s = str(s)
    return s[:n]


def summarize_comms(sms: List[Dict[str, Any]],
                    mails: List[Dict[str, Any]],
                    audio_transcripts: List[Dict[str, Any]] | None,
                    session_id: str) -> Dict[str, Any]:
    if not (sms or mails or audio_transcripts):
        return {"phishing_signals": [], "legitimate_counterparties": [],
                "suspicious_keywords": [], "victims_suspected": [], "notes": ""}

    payload = {
        "sms_samples":   [_truncate(s.get("sms",  ""), 400) for s in (sms   or [])[:60]],
        "mail_samples":  [_truncate(m.get("mail", ""), 900) for m in (mails or [])[:20]],
        "call_samples":  [{"user_hint": a.get("user_hint", ""),
                            "text": _truncate(a.get("transcript", ""), 600)}
                          for a in (audio_transcripts or [])[:20]],
    }
    user_content = json.dumps(payload, ensure_ascii=False)

    raw = llm_call("ctx", SYSTEM, user_content, session_id=session_id,
                   name="comms-analyst", json_mode=True)
    parsed = safe_json_loads(raw, default={})
    if not parsed:
        log.warning("Comms analyst returned invalid JSON, using empty summary")
        return {"phishing_signals": [], "legitimate_counterparties": [],
                "suspicious_keywords": [], "victims_suspected": [], "notes": raw[:200]}
    return parsed
