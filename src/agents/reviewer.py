"""
Reviewer Agent.

Riceve il subset di transazioni marcate come REVIEW dal triage,
insieme a profili utente e summary delle comunicazioni, e decide
quali flaggare come fraudolente.

Modello: "big" (Claude Sonnet) se il dossier è piccolo,
         "cheap" (Gemini Flash Lite) in batch per dossier grandi.

Strategia:
- Single-pass: ≤ 300 tx, un unico LLM call con Claude Sonnet
- Batch mode: > 300 tx, chunking verso worker cheap + arbiter finale
"""
from __future__ import annotations
import json
import math
from typing import List, Dict, Any

import pandas as pd

from src.agents.llm_client import llm_call
from src.utils.io import df_to_records_clean, safe_json_loads
from src.utils.logging import log


REVIEWER_SYS = """You are the lead fraud-detection agent in the "Reply Mirror" system.
You receive a DOSSIER containing:
- user_profiles: citizens with salary, job, residence, description
- comms_summary: signals from SMS/emails/calls (phishing, legitimate counterparties, victims)
- transactions: a pre-screened subset flagged as suspicious by deterministic rules

Each transaction carries pre-computed features:
- rule_hits: deterministic rules triggered
- iforest_score: ML anomaly score 0-1 (higher = more anomalous)
- social_vulnerability_score: 0-1, how targeted by social engineering the sender is
- geo_mismatch, impossible_travel_flag, velocity_* flags
- amount_z, amount_vs_monthly_salary, balance_negative

YOUR JOB: decide which of these are GENUINELY fraudulent.
- Prioritize PRECISION. A false positive blocks a legitimate customer.
- geo_mismatch=true, impossible_travel_flag=true, and velocity_exceeded are strong signals.
- A high social_vulnerability_score means the user may have been tricked → be MORE willing to flag
  transactions where the recipient is unfamiliar or the amount is unusual.
- If a recipient appears in comms_summary.legitimate_counterparties, LOWER suspicion.
- Coherence with user profile matters: a pensioner wiring crypto at 3am is suspicious.

Output ONLY JSON, no prose:
{
  "flagged": [
    {"tx_id": "<transaction_id>", "confidence": 0.0-1.0, "reason": "<≤15 words>"}
  ]
}

Constraints:
- Never flag 0 transactions UNLESS the dossier is truly clean.
- Never flag >70% of the dossier (the dossier is already pre-filtered).
- Aim for the subset that maximises precision × recall."""


WORKER_SYS = """You are a junior fraud-detection worker handling a chunk of pre-screened transactions.
You receive user profiles, comms summary and a batch of transactions.
Each transaction has pre-computed anomaly features.
FLAG any transaction that looks suspicious. A senior arbiter will filter false positives afterwards.

Pay strong attention to: impossible_travel_flag, geo_mismatch, velocity_card_exceeded,
velocity_amount_exceeded, high amount_z (>3), social_vulnerability_score>0.7.

Output ONLY JSON:
{
  "flagged": [
    {"tx_id": "...", "confidence": 0.0-1.0, "reason": "<≤10 words>"}
  ]
}"""


TX_COLS = [
    "transaction_id", "sender_id", "recipient_id", "transaction_type",
    "amount", "location", "sender_iban", "recipient_iban",
    "balance_after", "description", "timestamp",
    "hour", "is_night", "amount_z", "balance_negative",
    "iban_cross_border", "sender_is_citizen", "recipient_is_citizen",
    "geo_mismatch", "impossible_travel_flag", "impossible_travel_kmh_val",
    "velocity_tx_15min", "velocity_amt_2h",
    "velocity_card_exceeded", "velocity_amount_exceeded",
    "amount_vs_monthly_salary",
    "social_vulnerability_score",
    "iforest_score", "risk_score_heuristic",
    "rule_hits", "rule_reasons",
]


def _build_payload(dossier: pd.DataFrame,
                   user_profiles: Any,
                   comms_summary: Any) -> str:
    tx_records = df_to_records_clean(dossier, TX_COLS)
    payload = {
        "user_profiles": user_profiles,
        "comms_summary": comms_summary,
        "transactions": tx_records,
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def run_single_pass(dossier: pd.DataFrame,
                    user_profiles: Any,
                    comms_summary: Any,
                    session_id: str) -> List[Dict[str, Any]]:
    """Una sola chiamata al modello 'big'."""
    if len(dossier) == 0:
        return []
    payload = _build_payload(dossier, user_profiles, comms_summary)
    raw = llm_call("big", REVIEWER_SYS, payload, session_id=session_id,
                   name="reviewer-single-pass", json_mode=True)
    data = safe_json_loads(raw, default={"flagged": []})
    flagged = data.get("flagged", []) if isinstance(data, dict) else []
    log.info(f"Reviewer (single): {len(flagged)}/{len(dossier)} flagged")
    return flagged


def run_batch_mode(dossier: pd.DataFrame,
                   user_profiles: Any,
                   comms_summary: Any,
                   session_id: str,
                   chunk_size: int = 150) -> List[Dict[str, Any]]:
    """
    Worker (cheap) su chunk + Arbiter (arb) su unione dei risultati.
    L'arbiter è importato lazy per evitare circular import.
    """
    from src.agents.arbiter import run_arbiter

    if len(dossier) == 0:
        return []

    # --- Fase Worker ---
    n_chunks = math.ceil(len(dossier) / chunk_size)
    log.info(f"Batch mode: {n_chunks} chunks of ~{chunk_size} tx")
    worker_flags: List[Dict[str, Any]] = []
    for i in range(0, len(dossier), chunk_size):
        chunk = dossier.iloc[i:i + chunk_size]
        payload = _build_payload(chunk, user_profiles, comms_summary)
        chunk_idx = i // chunk_size
        try:
            raw = llm_call("cheap", WORKER_SYS, payload, session_id=session_id,
                           name=f"worker-chunk-{chunk_idx}", json_mode=True)
            data = safe_json_loads(raw, default={"flagged": []})
            chunk_flags = data.get("flagged", []) if isinstance(data, dict) else []
            worker_flags.extend(chunk_flags)
        except Exception as e:
            log.warning(f"Worker chunk {chunk_idx} failed: {e}")

    log.info(f"Workers flagged {len(worker_flags)} tx, running arbiter...")
    if not worker_flags:
        return []

    # --- Fase Arbiter ---
    return run_arbiter(worker_flags, dossier, user_profiles, comms_summary, session_id)
