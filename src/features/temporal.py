"""
Feature temporali + Velocity checks.

Riferimento: PDF sez. 3 "Architetture di Monitoraggio Frequenziale".
- Card Velocity: conteggio tx per sender in finestra scorrevole
- Amount Velocity: cumulativo importi in finestra scorrevole
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.config import THRESHOLDS


def add_temporal_features(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["hour"]     = tx["timestamp"].dt.hour.fillna(-1).astype(int)
    tx["dow"]      = tx["timestamp"].dt.dayofweek.fillna(-1).astype(int)
    tx["is_night"] = tx["hour"].between(0, 5) | (tx["hour"] == 23)
    tx["is_weekend"] = tx["dow"].isin([5, 6])
    return tx


def add_velocity_features(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Calcolo vettoriale delle velocity checks via rolling window su sender_id.
    - velocity_tx_15min: # di tx del sender negli ultimi N minuti
    - velocity_amt_2h:   somma importi del sender nelle ultime N ore
    """
    tx = tx.copy()
    if "timestamp" not in tx or "sender_id" not in tx:
        tx["velocity_tx_15min"] = 0
        tx["velocity_amt_2h"]   = 0.0
        return tx

    card_win = f"{THRESHOLDS.card_velocity_window_min}min"
    amt_win  = f"{THRESHOLDS.amount_velocity_window_hours}h"

    # Ordiniamo per sender + timestamp per rolling coerente
    sorted_tx = tx.sort_values(["sender_id", "timestamp"]).copy()
    # rolling con time-based window richiede index datetime
    sorted_tx = sorted_tx.set_index("timestamp", drop=False)

    # Count tx
    tx_count = (
        sorted_tx.groupby("sender_id")["amount"]
        .rolling(card_win, closed="both").count()
        .reset_index(level=0, drop=True)
    )
    # Sum amount
    amt_sum = (
        sorted_tx.groupby("sender_id")["amount"]
        .rolling(amt_win, closed="both").sum()
        .reset_index(level=0, drop=True)
    )

    sorted_tx["velocity_tx_15min"] = tx_count.fillna(1).astype(int)
    sorted_tx["velocity_amt_2h"]   = amt_sum.fillna(0.0).astype(float)

    # Rimontiamo sull'indice originale
    sorted_tx = sorted_tx.reset_index(drop=True)
    return (
        sorted_tx.sort_values("transaction_id")
        .reset_index(drop=True)
        if "transaction_id" in sorted_tx else sorted_tx
    )


def velocity_flags(tx: pd.DataFrame) -> pd.DataFrame:
    """Boolean flags basati sui threshold configurati."""
    tx = tx.copy()
    tx["velocity_card_exceeded"] = (
        tx.get("velocity_tx_15min", 0) > THRESHOLDS.card_velocity_max_tx
    )
    tx["velocity_amount_exceeded"] = (
        tx.get("velocity_amt_2h", 0.0) > THRESHOLDS.amount_velocity_max_eur
    )
    return tx
