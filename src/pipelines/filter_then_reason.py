"""
Pipeline orchestrator: Filter-then-Reason.

Flow:
  1. LOAD         → tx + users + locations + sms + mails + (audio)
  2. FEATURES     → temporal, spatial, economic, behavioral, NLP-local
  3. SCREENING    → rule engine + Isolation Forest
  4. TRIAGE       → PASS / HARD_FLAG / REVIEW
  5. LLM REVIEW   → solo sui REVIEW (single-pass o batch)
  6. MERGE        → HARD_FLAG ∪ LLM_flagged
  7. SAFETY NETS  → min/max output bounds, fallback
  8. WRITE        → file output challenge-compliant
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Set

import pandas as pd

from src.config import THRESHOLDS
from src.loaders.dataset import load_dataset, Dataset
from src.loaders.audio import transcribe_audio_files
from src.features import build_features
from src.screening.rule_engine import apply_rules
from src.screening.isolation_forest import fit_score_isolation_forest
from src.screening.triage import triage, dossier_subset
from src.agents.comms_analyst import summarize_comms
from src.agents.user_profiles import build_user_profiles
from src.agents.reviewer import run_single_pass, run_batch_mode
from src.agents.llm_client import flush as llm_flush
from src.utils.io import write_flagged_ids
from src.utils.logging import log


@dataclass
class PipelineConfig:
    dataset_path: Path
    output_path: Path
    session_id: str
    mode: str = "auto"              # auto | single | batch
    enable_audio: bool = True       # trascrizione MP3 se presenti
    single_pass_max: int = 300      # cutoff auto-mode


def _apply_safety_nets(final_ids: List[str],
                       total_tx: int,
                       feats: pd.DataFrame) -> List[str]:
    """
    Protezioni finali contro output invalidi (regole challenge):
    - mai 0 flag
    - mai ~tutti flaggati
    - se l'LLM ha fallito (0 flag), fallback a top-risk
    """
    # Dedup preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for x in final_ids:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)

    # Fallback: empty output → top 10% per risk_score
    if len(uniq) < THRESHOLDS.min_flag_count:
        log.warning("No flags produced; falling back to top-risk heuristic")
        n = max(THRESHOLDS.min_flag_count, total_tx // 10)
        uniq = (feats.nlargest(n, "risk_score_heuristic")["transaction_id"]
                .astype(str).tolist())

    # Cap: troppi flag → tieni i top ranked
    max_n = max(1, int(total_tx * THRESHOLDS.max_flag_fraction))
    if len(uniq) > max_n:
        log.warning(f"{len(uniq)}/{total_tx} flagged, truncating to {max_n}")
        keep = set(
            feats[feats["transaction_id"].astype(str).isin(uniq)]
            .assign(
                _rank=lambda d: d["risk_score_heuristic"].fillna(0)
                              + d["iforest_score"].fillna(0) * 10
            )
            .nlargest(max_n, "_rank")["transaction_id"].astype(str).tolist()
        )
        uniq = [x for x in uniq if x in keep]

    return uniq


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    log.info(f"=== Mirror Agent Pipeline ===")
    log.info(f"Dataset : {cfg.dataset_path}")
    log.info(f"Output  : {cfg.output_path}")
    log.info(f"Session : {cfg.session_id}")

    # -------- 1. LOAD --------
    ds: Dataset = load_dataset(cfg.dataset_path)
    total_tx = ds.n_tx

    # -------- 1b. AUDIO (if Deus Ex) --------
    audio_transcripts = []
    if cfg.enable_audio and ds.audio_files:
        log.info(f"Transcribing {len(ds.audio_files)} audio files (cached)")
        audio_transcripts = transcribe_audio_files(ds.audio_files)

    # -------- 2. FEATURES --------
    feats, social_vuln = build_features(ds, audio_transcripts)

    # -------- 3. SCREENING --------
    feats = apply_rules(feats)
    feats = fit_score_isolation_forest(feats)

    # -------- 4. TRIAGE --------
    feats = triage(feats)
    hard_ids: List[str] = (
        feats.loc[feats["triage"] == "HARD_FLAG", "transaction_id"].astype(str).tolist()
    )
    review_df = dossier_subset(feats, "REVIEW")

    # -------- 5. LLM REVIEW --------
    llm_flagged: List[Dict[str, Any]] = []
    if len(review_df) > 0:
        profiles = build_user_profiles(ds.users, social_vuln)
        comms = summarize_comms(ds.sms, ds.mails, audio_transcripts, cfg.session_id)

        mode = cfg.mode
        if mode == "auto":
            mode = "single" if len(review_df) <= cfg.single_pass_max else "batch"
        log.info(f"Review mode: {mode} on {len(review_df)} tx")

        if mode == "single":
            llm_flagged = run_single_pass(review_df, profiles, comms, cfg.session_id)
        else:
            llm_flagged = run_batch_mode(review_df, profiles, comms, cfg.session_id)

    # -------- 6. MERGE --------
    valid_ids = set(ds.tx["transaction_id"].astype(str))
    llm_ids = [f.get("tx_id") for f in llm_flagged if f.get("tx_id") in valid_ids]
    final_ids = list(dict.fromkeys([*hard_ids, *llm_ids]))

    # -------- 7. SAFETY --------
    final_ids = _apply_safety_nets(final_ids, total_tx, feats)

    # -------- 8. WRITE --------
    n_written = write_flagged_ids(cfg.output_path, final_ids)

    # -------- 9. LANGFUSE FLUSH --------
    try:
        llm_flush()
    except Exception:
        pass

    # -------- 10. REPORT --------
    stats = {
        "total_tx": total_tx,
        "hard_flags": len(hard_ids),
        "review_dossier": int((feats["triage"] == "REVIEW").sum()),
        "llm_flags": len(llm_ids),
        "final_flags": n_written,
        "output_path": str(cfg.output_path),
        "session_id": cfg.session_id,
    }
    log.info(f"=== DONE: {n_written}/{total_tx} flagged → {cfg.output_path} ===")
    log.info(f"Stats: {stats}")

    # Top reasons preview
    if llm_flagged:
        log.info("Top LLM reasons:")
        for f in llm_flagged[:5]:
            log.info(f"  {str(f.get('tx_id',''))[:12]}…  "
                     f"conf={f.get('confidence', '?')}  {f.get('reason','')}")
    return stats
