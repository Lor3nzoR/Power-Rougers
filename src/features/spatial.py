"""
Feature spaziali: Haversine distance + Impossible Travel + geo_mismatch.

Riferimento: PDF sez. 2.
- Formula Haversine per distanza ortodromica
- Velocità implicita = distanza / delta_tempo
- Soglia impossible travel ~900 km/h (jet commerciali)
- Allineamento asincrono tx ↔ locations via merge_asof
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.config import THRESHOLDS
from src.utils.logging import log

EARTH_RADIUS_KM = 6371.0


def haversine_vec(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distanza ortodromica in km fra coppie di coordinate (numpy vectorized)."""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def add_spatial_features(tx: pd.DataFrame, loc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggancia a ciascuna tx l'ultima location GPS nota del sender (biotag = sender_id),
    quindi calcola:
    - impossible_travel_km: distanza dalla location precedente
    - impossible_travel_kmh_val: velocità implicita km/h
    - impossible_travel_flag: bool, velocità > soglia
    - geo_mismatch: city della location vs città nel campo tx.location
    """
    tx = tx.copy()
    # Default columns
    tx["impossible_travel_km"]     = 0.0
    tx["impossible_travel_kmh_val"] = 0.0
    tx["impossible_travel_flag"]   = False
    tx["geo_mismatch"]             = False

    if loc_df is None or not len(loc_df):
        return tx
    required_cols = {"biotag", "timestamp", "lat", "lng"}
    if not required_cols.issubset(loc_df.columns):
        log.warning(f"locations.json missing cols: {required_cols - set(loc_df.columns)}")
        return tx

    # --- Step 1: asof merge tra tx e loc_df (ultima loc nota prima della tx) ---
    # Ci serve il sender_id che coincida con biotag.
    left = tx[["transaction_id", "sender_id", "timestamp", "location"]].sort_values("timestamp").copy()
    right = loc_df[["biotag", "timestamp", "lat", "lng", "city"]].sort_values("timestamp").copy() \
        if "city" in loc_df.columns else \
        loc_df[["biotag", "timestamp", "lat", "lng"]].sort_values("timestamp").copy()

    # merge_asof richiede key ordinata e left-right con stesso dtype timestamp
    merged = pd.merge_asof(
        left, right,
        left_on="timestamp", right_on="timestamp",
        left_by="sender_id", right_by="biotag",
        direction="backward",
    )
    # Rinomina le colonne fuse
    merged = merged.rename(columns={
        "lat": "tx_loc_lat", "lng": "tx_loc_lng",
        "city": "tx_loc_city",
    })

    # --- Step 2: per ogni sender, distanza dalla loc PRECEDENTE nota ---
    # Calcoliamo distanza e velocità rispetto alla previous fix dello stesso sender.
    merged = merged.sort_values(["sender_id", "timestamp"]).reset_index(drop=True)
    merged["prev_lat"]  = merged.groupby("sender_id")["tx_loc_lat"].shift(1)
    merged["prev_lng"]  = merged.groupby("sender_id")["tx_loc_lng"].shift(1)
    merged["prev_time"] = merged.groupby("sender_id")["timestamp"].shift(1)

    valid = merged[["tx_loc_lat", "tx_loc_lng", "prev_lat", "prev_lng"]].notna().all(axis=1)
    km = np.zeros(len(merged))
    km[valid] = haversine_vec(
        merged.loc[valid, "prev_lat"], merged.loc[valid, "prev_lng"],
        merged.loc[valid, "tx_loc_lat"], merged.loc[valid, "tx_loc_lng"],
    )
    dt_hours = (merged["timestamp"] - merged["prev_time"]).dt.total_seconds() / 3600.0
    kmh = np.where(dt_hours > 1e-6, km / dt_hours.replace(0, np.nan), 0.0)
    kmh = np.nan_to_num(kmh, nan=0.0, posinf=0.0, neginf=0.0)

    merged["impossible_travel_km"]     = km
    merged["impossible_travel_kmh_val"] = kmh
    merged["impossible_travel_flag"]   = kmh > THRESHOLDS.impossible_travel_kmh

    # --- Step 3: geo_mismatch (city vs tx.location) ---
    def _mismatch(row):
        tx_loc = row.get("location")
        city   = row.get("tx_loc_city")
        if not isinstance(tx_loc, str) or not isinstance(city, str):
            return False
        a, b = tx_loc.lower().strip(), city.lower().strip()
        if not a or not b:
            return False
        return a not in b and b not in a
    merged["geo_mismatch"] = merged.apply(_mismatch, axis=1)

    # --- Step 4: rimerge sulle tx originali su transaction_id ---
    cols_to_bring = [
        "transaction_id",
        "impossible_travel_km", "impossible_travel_kmh_val",
        "impossible_travel_flag", "geo_mismatch",
    ]
    tx = tx.drop(columns=[c for c in cols_to_bring if c in tx and c != "transaction_id"], errors="ignore")
    tx = tx.merge(merged[cols_to_bring], on="transaction_id", how="left")
    # Fill
    tx["impossible_travel_km"]     = tx["impossible_travel_km"].fillna(0.0)
    tx["impossible_travel_kmh_val"] = tx["impossible_travel_kmh_val"].fillna(0.0)
    tx["impossible_travel_flag"]   = tx["impossible_travel_flag"].fillna(False).astype(bool)
    tx["geo_mismatch"]             = tx["geo_mismatch"].fillna(False).astype(bool)
    return tx
