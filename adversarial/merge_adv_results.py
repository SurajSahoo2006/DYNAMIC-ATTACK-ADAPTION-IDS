"""
adversarial/merge_adv_results.py
================================
Merges adversarial stage results into adversarial/results/adversarial_summary.json.

Top-level keys:
  - "fgsm"
  - "hsj"
  - "zoo"
  - "defense"
  - "pool_info": {"total_flows": N, "benign_flows": M}

Prints a 4-row summary table to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ADV_RESULTS_DIR = PROJECT_ROOT / "adversarial" / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_file_with_fallback(primary_name: str, fallback_name: str) -> Any:
    p_path = ADV_RESULTS_DIR / primary_name
    f_path = REPORTS_DIR / fallback_name
    if p_path.exists():
        with open(p_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif f_path.exists():
        with open(f_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"WARNING: Neither {p_path} nor {f_path} exists. Skipping.")
        return None


def main():
    ADV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}

    # 1. FGSM
    fgsm_data = load_file_with_fallback("fgsm_results.json", "adversarial_fgsm.json")
    if fgsm_data is not None:
        summary["fgsm"] = fgsm_data
        pool_info = fgsm_data.get("pool_info")
        if not pool_info:
            samples = fgsm_data.get("zero_attack_control", {}).get("samples", 2000)
            pool_info = {"total_flows": samples, "benign_flows": min(200, samples)}
        summary["pool_info"] = pool_info
    else:
        summary["pool_info"] = {"total_flows": None, "benign_flows": None}

    # 2. HSJ
    hsj_data = load_file_with_fallback("hsj_results.json", "adversarial_hsj.json")
    if hsj_data is not None:
        summary["hsj"] = hsj_data

    # 3. ZOO
    zoo_data = load_file_with_fallback("zoo_results.json", "adversarial_zoo.json")
    if zoo_data is not None:
        summary["zoo"] = zoo_data

    # 4. Defense
    defense_data = load_file_with_fallback("defense_results.json", "adversarial_defense.json")
    if defense_data is not None:
        summary["defense"] = defense_data

    # Save merged summary
    out_path = ADV_RESULTS_DIR / "adversarial_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Merged adversarial summary written to {out_path.name}")

    # Print 4-row summary table
    print("\n" + "=" * 80)
    print("                  ADVERSARIAL EVALUATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Attack Stage':<18} | {'Flows (N)':<10} | {'Clean Acc':<12} | {'Adversarial Metric / Key Finding':<32}")
    print("-" * 80)

    # Row 1: FGSM
    if "fgsm" in summary:
        fg = summary["fgsm"]
        cnt = summary["pool_info"]["total_flows"]
        clean_acc = fg.get("zero_attack_control", {}).get("ensemble_clean_pipeline_acc", 0.0) * 100
        # find eps 0.10 result
        eps10 = next((r for r in fg.get("fgsm_and_random_noise_results", []) if abs(r["epsilon"] - 0.10) < 1e-3), None)
        adv_desc = f"eps=0.10 Adv Acc: {eps10['ensemble_adversarial_acc']*100:.1f}%" if eps10 else "Evaluated"
        print(f"{'1. FGSM Transfer':<18} | {cnt:<10} | {clean_acc:>10.2f}% | {adv_desc:<32}")
    else:
        print(f"{'1. FGSM Transfer':<18} | {'N/A':<10} | {'N/A':<12} | {'Skipped / Missing':<32}")

    # Row 2: HSJ
    if "hsj" in summary:
        hs = summary["hsj"]
        b1 = hs.get("Budget 1 (Low: 3 iters, 10 evals)", {})
        b2 = hs.get("Budget 2 (High: 6 iters, 20 evals)", {})
        samples = b1.get("samples", 100)
        clean_acc = b1.get("clean_accuracy", 0.0) * 100
        b1_acc = b1.get("adversarial_accuracy", 0.0) * 100
        b2_acc = b2.get("adversarial_accuracy", 0.0) * 100
        desc = f"B1 Adv Acc: {b1_acc:.1f}%, B2: {b2_acc:.1f}%"
        print(f"{'2. HopSkipJump':<18} | {samples:<10} | {clean_acc:>10.2f}% | {desc:<32}")
    else:
        print(f"{'2. HopSkipJump':<18} | {'N/A':<10} | {'N/A':<12} | {'Skipped / Missing':<32}")

    # Row 3: ZOO
    if "zoo" in summary:
        zo = summary["zoo"]
        b1 = zo.get("Budget 1 (Low: 4 iters)", {})
        b2 = zo.get("Budget 2 (High: 8 iters)", {})
        samples = b1.get("samples", 40)
        clean_acc = b1.get("clean_accuracy", 0.0) * 100
        b1_acc = b1.get("adversarial_accuracy", 0.0) * 100
        b2_acc = b2.get("adversarial_accuracy", 0.0) * 100
        desc = f"B1 Adv Acc: {b1_acc:.1f}%, B2: {b2_acc:.1f}%"
        print(f"{'3. ZOO Attack':<18} | {samples:<10} | {clean_acc:>10.2f}% | {desc:<32}")
    else:
        print(f"{'3. ZOO Attack':<18} | {'N/A':<10} | {'N/A':<12} | {'Skipped / Missing':<32}")

    # Row 4: Defense
    if "defense" in summary:
        df = summary["defense"]
        samples = df.get("test_sample_size", 500)
        arms = df.get("arms", {})
        arm_d = arms.get("Arm D: Retrained Clean + Adv (3k+3k)", {})
        fg_acc = arm_d.get("fgsm_eps010", {}).get("acc", 0.0) * 100
        clean_d = arm_d.get("clean", {}).get("acc", 0.0) * 100
        desc = f"Arm D FGSM 0.10 Acc: {fg_acc:.1f}%"
        print(f"{'4. 4-Arm Defense':<18} | {samples:<10} | {clean_d:>10.2f}% | {desc:<32}")
    else:
        print(f"{'4. 4-Arm Defense':<18} | {'N/A':<10} | {'N/A':<12} | {'Skipped / Missing':<32}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
