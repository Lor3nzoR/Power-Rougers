"""Test end-to-end del layer muscolare senza LLM."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders.dataset import load_dataset
from src.features import build_features
from src.screening.rule_engine import apply_rules
from src.screening.isolation_forest import fit_score_isolation_forest
from src.screening.triage import triage

ds = load_dataset("/tmp/fake_ds")
print(f"\n--- DATASET ---\n{ds.summary()}\n")

feats, svi = build_features(ds, audio_transcripts=None)
print("--- SOCIAL VULNERABILITY ---")
for uid, info in svi.items():
    print(f"  {uid}: score={info['score']} hits={info['total_hits']} counts={info['pattern_counts']}")

feats = apply_rules(feats)
feats = fit_score_isolation_forest(feats)
feats = triage(feats)

print("\n--- TRIAGE RESULT ---")
cols = ["transaction_id", "sender_id", "amount", "location",
        "triage", "rule_severity", "rule_hits",
        "impossible_travel_kmh_val", "geo_mismatch",
        "velocity_tx_15min", "iforest_score",
        "social_vulnerability_score", "risk_score_heuristic"]
import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(feats[cols].to_string(index=False))

print(f"\n--- SUMMARY ---")
print(f"Total:     {len(feats)}")
print(f"HARD_FLAG: {(feats['triage']=='HARD_FLAG').sum()}")
print(f"REVIEW:    {(feats['triage']=='REVIEW').sum()}")
print(f"PASS:      {(feats['triage']=='PASS').sum()}")
