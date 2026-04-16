"""
Arbiter Agent.

Riceve la lista dei flag sollevati dai worker + il dossier originale delle
transazioni candidate, e decide quali MANTENERE come veramente fraudolente.

Filosofia: i worker sono volutamente aggressivi (vedi WORKER_SYS),
l'arbiter ottimizza per PRECISION tagliando i falsi positivi.

Modello: "arb" = gpt-5-mini (cost-sensitive, buono nel filtering).
"""
from __future__ import annotations
import json
from typing import List, Dict, Any

import pandas as pd

from src.agents.llm_client import llm_call
from src.utils.io import df_to_records_clean, safe_json_loads
from src.utils.logging import log
from src.agents.reviewer import TX_COLS


ARBITER_SYS = """You are the Lead Fraud Arbiter in the "Reply Mirror" pipeline.
Junior workers have flagged the transactions below as potentially fraudulent.
They are deliberately aggressive — YOUR job is to FILTER OUT false positives.

Review each flagged transaction against:
- user_profiles (coherence with job, salary, residence)
- comms_summary (is the recipient a legitimate counterparty?)
- pre-computed features (impossible_travel_flag, geo_mismatch, velocity, etc.)

Keep ONLY the genuinely suspicious ones. Aim for HIGH PRECISION.
Be willing to cut 30-60% of the worker flags if they lack substance.

Output ONLY JSON:
{
  "flagged": [
    {"tx_id": "...", "confidence": 0.0-1.0, "reason": "<≤15 words>"}
  ]
}"""


def run_arbiter(worker_flags: List[Dict[str, Any]],
                dossier: pd.DataFrame,
                user_profiles: Any,
                comms_summary: Any,
                session_id: str) -> List[Dict[str, Any]]:
    """Filtra worker_flags tenendo solo quelli confermati dall'arbiter."""
    if not worker_flags:
        return []

    # Deduplica per tx_id mantenendo la confidence max
    by_id: Dict[str, Dict[str, Any]] = {}
    for f in worker_flags:
        tx_id = f.get("tx_id")
        if not tx_id:
            continue
        if tx_id not in by_id or f.get("confidence", 0) > by_id[tx_id].get("confidence", 0):
            by_id[tx_id] = f

    flagged_ids = set(by_id.keys())
    arbiter_subset = dossier[dossier["transaction_id"].isin(flagged_ids)].copy()

    if arbiter_subset.empty:
        log.warning("Arbiter: no matching tx in dossier for worker flags")
        return list(by_id.values())

    payload = json.dumps({
        "user_profiles": user_profiles,
        "comms_summary": comms_summary,
        "worker_flags": [{"tx_id": k, "worker_reason": v.get("reason", "")}
                         for k, v in by_id.items()],
        "flagged_transactions_to_review": df_to_records_clean(arbiter_subset, TX_COLS),
    }, ensure_ascii=False, default=str)

    try:
        raw = llm_call("arb", ARBITER_SYS, payload, session_id=session_id,
                       name="arbiter-review", json_mode=True)
        data = safe_json_loads(raw, default=None)
        if isinstance(data, dict) and isinstance(data.get("flagged"), list):
            log.info(f"Arbiter kept {len(data['flagged'])}/{len(by_id)} worker flags")
            return data["flagged"]
        log.warning("Arbiter returned invalid JSON; falling back to worker flags")
        return list(by_id.values())
    except Exception as e:
        log.warning(f"Arbiter failed: {e}; falling back to worker flags")
        return list(by_id.values())
