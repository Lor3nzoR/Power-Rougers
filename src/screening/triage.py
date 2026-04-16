"""
Triage: decide per ogni transazione se è
- "PASS"      → legittima, nessun output
- "HARD_FLAG" → frode con alta confidenza (bypassa LLM)
- "REVIEW"    → ambigua, va al Reviewer LLM

Questa è la cerniera tra Layer Muscolare e Layer Cerebrale.
"""
from __future__ import annotations
import pandas as pd

from src.config import THRESHOLDS
from src.utils.logging import log


def triage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- HARD_FLAG: almeno una regola HARD colpita ---
    hard_mask = df["rule_severity"] == "HARD"

    # --- REVIEW: regole SUSPECT OPPURE iforest flag ---
    review_mask = (
        (df["rule_severity"] == "SUSPECT")
        | df.get("iforest_flag", False)
    ) & ~hard_mask

    # Se tutti i REVIEW superano il limite configurato, teniamo i top-scored.
    max_review = int(len(df) * THRESHOLDS.max_llm_review_fraction)
    review_idx = df.index[review_mask]
    if len(review_idx) > max_review and max_review > 0:
        # Ordina per combined score desc
        combined = (
            df.loc[review_idx, "iforest_score"].fillna(0.0)
            + df.loc[review_idx, "risk_score_heuristic"].fillna(0.0) / 20.0
        )
        top = combined.sort_values(ascending=False).head(max_review).index
        review_mask = df.index.isin(top)

    df["triage"] = "PASS"
    df.loc[review_mask, "triage"] = "REVIEW"
    df.loc[hard_mask,   "triage"] = "HARD_FLAG"

    n_hard   = int((df["triage"] == "HARD_FLAG").sum())
    n_review = int((df["triage"] == "REVIEW").sum())
    n_pass   = int((df["triage"] == "PASS").sum())
    log.info(f"Triage: HARD={n_hard} · REVIEW={n_review} · PASS={n_pass} "
             f"(tot={len(df)})")
    return df


def dossier_subset(df: pd.DataFrame, triage_label: str = "REVIEW") -> pd.DataFrame:
    return df[df["triage"] == triage_label].copy()
