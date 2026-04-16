"""
Isolation Forest per anomaly scoring non supervisionato.

Riferimento: PDF sez. 7 "Motore di Isolamento".
L'idea: le frodi sono istanze "poche e diverse" nello spazio vettoriale.
In un albero casuale si isolano con pochi split → path corto → anomaly score alto.

L'output (0=normale, 1=anomalia) viene aggiunto come colonna `iforest_score`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.config import IFOREST_FEATURES, THRESHOLDS
from src.utils.logging import log


def fit_score_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log.warning("scikit-learn not available; iforest_score fallback to 0.")
        df["iforest_score"] = 0.0
        df["iforest_flag"]  = False
        return df

    cols = [c for c in IFOREST_FEATURES if c in df.columns]
    if not cols:
        df["iforest_score"] = 0.0
        df["iforest_flag"]  = False
        return df

    X = df[cols].copy()
    # Coerce bool → int, fill NaN → 0, clip infinite
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iforest = IsolationForest(
        n_estimators=THRESHOLDS.iforest_n_estimators,
        contamination=THRESHOLDS.iforest_contamination,
        random_state=THRESHOLDS.iforest_random_state,
        n_jobs=-1,
    )
    iforest.fit(Xs)
    # decision_function: più basso = più anomalo.
    # Convertiamo a 0-1 con anomalia alta = 1.
    raw = -iforest.decision_function(Xs)
    # Normalizzazione min-max
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-9:
        score = np.zeros_like(raw)
    else:
        score = (raw - lo) / (hi - lo)
    df["iforest_score"] = score
    # predict restituisce -1 per anomalie
    df["iforest_flag"] = iforest.predict(Xs) == -1

    n_flag = int(df["iforest_flag"].sum())
    log.info(f"Isolation Forest: {n_flag}/{len(df)} flagged "
             f"(contamination={THRESHOLDS.iforest_contamination})")
    return df
