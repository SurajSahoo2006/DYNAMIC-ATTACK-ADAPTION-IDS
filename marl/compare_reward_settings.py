"""
marl/compare_reward_settings.py
===============================
Merges and compares MAPPO evaluation results across all 3 reward settings:
  1. balanced
  2. security_first
  3. availability_first

Prints a comparison table and saves to marl/results/reward_settings_comparison.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MARL_RESULTS_DIR = PROJECT_ROOT / "marl" / "results"

SETTINGS = ["balanced", "security_first", "availability_first"]


def main():
    MARL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison: Dict[str, Any] = {}

    print("\n" + "=" * 88)
    print("                MAPPO REWARD SETTINGS COMPARISON TABLE")
    print("=" * 88)
    print(f"{'Setting':<20} | {'Aggregate Return':<18} | {'Attacks Contained':<18} | {'Benign Disrupted':<18} | {'Truncated':<10}")
    print("-" * 88)

    for setting in SETTINGS:
        file_path = MARL_RESULTS_DIR / f"eval_{setting}.json"
        if not file_path.exists():
            print(f"WARNING: File {file_path.name} not found. Filling with nulls.")
            comparison[setting] = {
                "setting": setting,
                "aggregate_return": None,
                "attacks_contained_pct": None,
                "benign_disrupted_pct": None,
                "truncated": None,
                "status": "missing",
            }
            print(f"{setting:<20} | {'null':<18} | {'null':<18} | {'null':<18} | {'null':<10}")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            agg_ret = data.get("aggregate_return")
            att_pct = data.get("attacks_contained_pct")
            ben_pct = data.get("benign_disrupted_pct")
            trunc = data.get("truncated", False)

            raw_checkpoint_path = data.get("checkpoint")
            if raw_checkpoint_path:
                try:
                    checkpoint_path = os.path.relpath(raw_checkpoint_path, PROJECT_ROOT).replace("\\", "/")
                except Exception:
                    checkpoint_path = os.path.relpath(raw_checkpoint_path).replace("\\", "/")
            else:
                checkpoint_path = None

            comparison[setting] = {
                "setting": setting,
                "aggregate_return": round(float(agg_ret), 4) if agg_ret is not None else None,
                "attacks_contained_pct": round(float(att_pct), 2) if att_pct is not None else None,
                "benign_disrupted_pct": round(float(ben_pct), 2) if ben_pct is not None else None,
                "truncated": bool(trunc),
                "episodes_evaluated": data.get("episodes_evaluated"),
                "checkpoint": checkpoint_path,
            }

            agg_str = f"{agg_ret:>8.2f}" if agg_ret is not None else "null"
            att_str = f"{att_pct:>6.2f}%" if att_pct is not None else "null"
            ben_str = f"{ben_pct:>6.2f}%" if ben_pct is not None else "null"
            trunc_str = str(trunc)

            print(f"{setting:<20} | {agg_str:<18} | {att_str:<18} | {ben_str:<18} | {trunc_str:<10}")
        except Exception as e:
            print(f"WARNING: Error reading {file_path.name}: {e}. Filling with nulls.")
            comparison[setting] = {
                "setting": setting,
                "aggregate_return": None,
                "attacks_contained_pct": None,
                "benign_disrupted_pct": None,
                "truncated": None,
                "error": str(e),
            }
            print(f"{setting:<20} | {'error':<18} | {'error':<18} | {'error':<18} | {'null':<10}")

    print("=" * 88 + "\n")

    out_file = MARL_RESULTS_DIR / "reward_settings_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"[OK] Comparison saved to {out_file.name}")


if __name__ == "__main__":
    main()
