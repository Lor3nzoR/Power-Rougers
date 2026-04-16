"""
Rule engine deterministico.
Ogni regola ritorna (bool_mask, reason_str).
Le regole "HARD" producono flag diretti senza passare dall'LLM.
Le regole "SUSPECT" alimentano il triage (dossier verso Reviewer).

Riferimento: PDF sez. 2-5 (Hard deterministic rules).
"""
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import List, Callable, Tuple

from src.config import THRESHOLDS


@dataclass
class Rule:
    name: str
    severity: str              # "HARD" | "SUSPECT"
    fn: Callable[[pd.DataFrame], pd.Series]
    reason: str


def _impossible_travel(df: pd.DataFrame) -> pd.Series:
    return df.get("impossible_travel_flag", pd.Series(False, index=df.index)).astype(bool)

def _overdraft_large(df: pd.DataFrame) -> pd.Series:
    # saldo negativo + amount grande
    return df["balance_negative"] & (df["amount_z"].abs() > THRESHOLDS.amount_z_soft)

def _salary_catastrophic(df: pd.DataFrame) -> pd.Series:
    return df["amount_vs_monthly_salary"].fillna(0) > THRESHOLDS.salary_ratio_hard

def _amount_z_extreme(df: pd.DataFrame) -> pd.Series:
    return df["amount_z"].abs() > THRESHOLDS.amount_z_hard

def _velocity_combo(df: pd.DataFrame) -> pd.Series:
    return df.get("velocity_card_exceeded", False) | df.get("velocity_amount_exceeded", False)

def _social_high_plus_anomaly(df: pd.DataFrame) -> pd.Series:
    """Utente molto vulnerabile + transazione con almeno qualche anomalia"""
    high_social = df["social_vulnerability_score"] > THRESHOLDS.social_vuln_hard
    some_anomaly = (
        (df["amount_z"].abs() > THRESHOLDS.amount_z_soft)
        | df["iban_cross_border"]
        | df["is_night"]
        | df["geo_mismatch"]
    )
    return high_social & some_anomaly

def _geo_mismatch(df: pd.DataFrame) -> pd.Series:
    return df["geo_mismatch"].astype(bool)


RULES: List[Rule] = [
    Rule("impossible_travel",     "HARD",    _impossible_travel,
         "Impossible travel: velocity > 900 km/h between locations"),
    Rule("salary_catastrophic",   "HARD",    _salary_catastrophic,
         "Amount exceeds full monthly salary"),
    Rule("velocity_combo",        "HARD",    _velocity_combo,
         "Velocity check exceeded (card count or amount sum)"),
    Rule("amount_z_extreme",      "SUSPECT", _amount_z_extreme,
         "Amount z-score > 4σ from sender baseline"),
    Rule("overdraft_large",       "SUSPECT", _overdraft_large,
         "Negative balance + large amount deviation"),
    Rule("social_high_plus",      "SUSPECT", _social_high_plus_anomaly,
         "High social vulnerability + behavioral anomaly"),
    Rule("geo_mismatch",          "SUSPECT", _geo_mismatch,
         "Location mismatch vs last known GPS city"),
]


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge al df:
    - rule_hits: lista dei nomi di regole triggerate
    - rule_severity: "HARD" se almeno una HARD, "SUSPECT" se almeno SUSPECT, else ""
    - rule_reasons: stringa concatenata human-readable
    """
    df = df.copy()
    hits_per_row: List[List[str]] = [[] for _ in range(len(df))]
    reasons_per_row: List[List[str]] = [[] for _ in range(len(df))]
    severity = [""] * len(df)

    for rule in RULES:
        try:
            mask = rule.fn(df).fillna(False).astype(bool)
        except Exception:
            continue
        idx = df.index[mask]
        for i in idx:
            pos = df.index.get_loc(i)
            hits_per_row[pos].append(rule.name)
            reasons_per_row[pos].append(rule.reason)
            # Promuove la severity
            if rule.severity == "HARD":
                severity[pos] = "HARD"
            elif severity[pos] != "HARD":
                severity[pos] = "SUSPECT"

    df["rule_hits"]     = hits_per_row
    df["rule_reasons"]  = ["; ".join(r) for r in reasons_per_row]
    df["rule_severity"] = severity
    return df
