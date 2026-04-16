"""
Orchestrator di feature engineering. Espone `build_features()` che prende
un Dataset e ritorna il DataFrame arricchito.
"""
from __future__ import annotations
import pandas as pd

from src.loaders.dataset import Dataset
from src.features.temporal import add_temporal_features, add_velocity_features, velocity_flags
from src.features.spatial import add_spatial_features
from src.features.economic import add_economic_features
from src.features.behavioral import add_behavioral_features
from src.features.nlp_local import compute_social_vulnerability, attach_social_scores
from src.utils.logging import log


def build_features(ds: Dataset,
                   audio_transcripts: list | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Ritorna (feats_df, social_vuln_dict).
    `social_vuln_dict` è user_id -> dict con score e pattern counts (utile per dossier LLM).
    """
    log.info("Building features...")
    feats = ds.tx.copy()
    feats = add_temporal_features(feats)
    feats = add_behavioral_features(feats, ds.user_biotags)
    feats = add_economic_features(feats, ds.iban_to_user)
    feats = add_velocity_features(feats)
    feats = velocity_flags(feats)
    feats = add_spatial_features(feats, ds.locations)

    svi = compute_social_vulnerability(ds.users, ds.sms, ds.mails, audio_transcripts)
    feats = attach_social_scores(feats, svi)

    # ---- Heuristic risk score composito (usato come feature + fallback) ----
    feats["risk_score_heuristic"] = (
        (feats["amount_z"].abs() > 2).astype(int) * 2
      + (feats["amount_z"].abs() > 4).astype(int) * 2
      + feats["is_night"].astype(int)
      + feats["balance_negative"].astype(int) * 3
      + feats["iban_cross_border"].astype(int)
      + feats["geo_mismatch"].astype(int) * 4
      + feats["impossible_travel_flag"].astype(int) * 5
      + feats.get("velocity_card_exceeded", False).astype(int) * 3
      + feats.get("velocity_amount_exceeded", False).astype(int) * 3
      + ((feats["amount_vs_monthly_salary"].fillna(0) > 0.5).astype(int)) * 2
      + (feats["social_vulnerability_score"] > 0.5).astype(int) * 3
      + (feats["social_vulnerability_score"] > 0.8).astype(int) * 3
    )

    log.info(f"Features built: {feats.shape[0]} rows × {feats.shape[1]} cols")
    return feats, svi
