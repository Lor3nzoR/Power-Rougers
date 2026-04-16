"""
Feature economiche / finanziarie.

- amount_z: z-score dell'importo rispetto al sender (baseline)
- balance_negative: saldo post-tx < 0 (overdraft)
- amount_vs_monthly_salary: rapporto con lo stipendio mensile dell'utente
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any


def add_economic_features(tx: pd.DataFrame,
                          iban_to_user: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    tx = tx.copy()

    # --- Baseline per-sender (media/std/count) ---
    stats = tx.groupby("sender_id")["amount"].agg(["mean", "std", "count"])
    stats.columns = ["sender_mean", "sender_std", "sender_count"]
    tx = tx.merge(stats, left_on="sender_id", right_index=True, how="left")

    std_fallback = tx["amount"].std() or 1.0
    tx["amount_z"] = (
        (tx["amount"] - tx["sender_mean"])
        / tx["sender_std"].replace(0, np.nan).fillna(std_fallback)
    ).fillna(0.0)
    tx["amount_z_abs"] = tx["amount_z"].abs()

    # --- Overdraft ---
    tx["balance_negative"] = tx["balance_after"].fillna(0.0) < 0

    # --- Salary ratio ---
    def _salary(iban: Any) -> float | None:
        if not isinstance(iban, str):
            return None
        u = iban_to_user.get(iban)
        return u.get("salary") if u else None

    tx["sender_salary"] = tx["sender_iban"].apply(_salary) if "sender_iban" in tx else None
    tx["amount_vs_monthly_salary"] = tx.apply(
        lambda r: (r["amount"] / (r["sender_salary"] / 12.0))
        if r.get("sender_salary") else np.nan,
        axis=1,
    )
    return tx
