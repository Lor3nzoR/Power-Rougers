"""
CLI entry point del Mirror Agent.

Esempi d'uso:
    python -m src.main --dataset "datasets/The Truman Show - train" --out outputs/truman_train.txt
    python -m src.main --dataset "datasets/Deus Ex - train" --out outputs/deus_ex.txt --mode batch
    python -m src.main --dataset "datasets/Brave New World - train" --out outputs/bnw.txt --no-audio
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Carichiamo .env PRIMA di qualunque import che tocchi env vars
load_dotenv()

from src.pipelines.filter_then_reason import PipelineConfig, run_pipeline
from src.utils.logging import log


def _build_session_id(explicit: str | None, dataset_path: Path) -> str:
    if explicit:
        sid = explicit
    else:
        team = os.environ.get("TEAM_NAME", "team").replace(" ", "-")
        ds_name = dataset_path.name.replace(" ", "-")
        sid = f"{team}-{ds_name}"
    # Challenge requirement: nessuno spazio
    sid = sid.replace(" ", "-")
    if " " in sid:
        raise ValueError("session-id non deve contenere spazi")
    return sid


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Mirror Agent - Fraud detection multi-agent pipeline (Reply Challenge 2026)"
    )
    ap.add_argument("--dataset", required=True, help="Path al dataset folder")
    ap.add_argument("--out",     required=True, help="Path al file di output")
    ap.add_argument("--session-id", default=None,
                    help="Langfuse session_id (default: <TEAM>-<dataset>)")
    ap.add_argument("--mode", choices=["auto", "single", "batch"], default="auto",
                    help="Modalità LLM review")
    ap.add_argument("--no-audio", action="store_true",
                    help="Disabilita trascrizione audio anche se presenti MP3")
    ap.add_argument("--single-pass-max", type=int, default=300,
                    help="Soglia tx per passare da single a batch in modalità auto")
    args = ap.parse_args(argv)

    dataset_path = Path(args.dataset)
    output_path  = Path(args.out)
    session_id   = _build_session_id(args.session_id, dataset_path)

    cfg = PipelineConfig(
        dataset_path=dataset_path,
        output_path=output_path,
        session_id=session_id,
        mode=args.mode,
        enable_audio=not args.no_audio,
        single_pass_max=args.single_pass_max,
    )

    try:
        stats = run_pipeline(cfg)
        return 0
    except Exception as e:
        log.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
