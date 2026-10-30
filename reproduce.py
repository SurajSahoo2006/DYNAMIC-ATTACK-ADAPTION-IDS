"""
reproduce.py
============
Single entry point to reproduce and verify key results and artifacts across
Phases 1-6 of the Dynamic Attack Adaptation IDS project.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()


def main():
    # 2a. Header
    print("=== Dynamic Attack Adaptation IDS -- Reproduce ===")

    # 2b. Environment Variable Check
    raw_dir = os.environ.get("CIC_DATA_RAW_DIR")
    if not raw_dir or not os.path.isdir(raw_dir):
        print(
            "ERROR: Set CIC_DATA_RAW_DIR to the folder containing CICIoT2023\n"
            "       CSV files, then rerun."
        )
        sys.exit(1)

    print(f"CIC_DATA_RAW_DIR: {raw_dir} (verified)\n")

    # 2c. Phase Artifact Verification
    print("--- Phase Output Artifacts Status ---")
    
    # Phase 1-2
    p1_scaler = (PROJECT_ROOT / "artifacts/scaler.pkl").exists() or (PROJECT_ROOT / "artifacts/standard_scaler.pkl").exists()
    p1_feat = (PROJECT_ROOT / "artifacts/selected_features.json").exists()
    status_p1 = "FOUND" if (p1_scaler and p1_feat) else "MISSING"
    print(f"Phase 1-2 : {status_p1} (artifacts/scaler.pkl, artifacts/selected_features.json)")

    # Phase 3
    p3_rf = (PROJECT_ROOT / "models/rf_model.pkl").exists() or (PROJECT_ROOT / "models/rf_model.joblib").exists() or (PROJECT_ROOT / "models/rf_8class_b2.joblib").exists()
    p3_xgb = (PROJECT_ROOT / "models/xgb_model.pkl").exists() or (PROJECT_ROOT / "models/xgb_model.joblib").exists() or (PROJECT_ROOT / "models/xgb_8class_b2.joblib").exists()
    status_p3 = "FOUND" if (p3_rf and p3_xgb) else "MISSING"
    print(f"Phase 3   : {status_p3} (models/rf_model.pkl, models/xgb_model.pkl)")

    # Phase 4
    p4_adv = (PROJECT_ROOT / "adversarial/results/adversarial_summary.json").exists()
    p4_shap = (PROJECT_ROOT / "xai/results/shap_summary.json").exists() or (PROJECT_ROOT / "reports/shap_metrics.json").exists()
    status_p4 = "FOUND" if (p4_adv and p4_shap) else "MISSING"
    print(f"Phase 4   : {status_p4} (adversarial/results/adversarial_summary.json, xai/results/shap_summary.json)")

    # Phase 5
    p5_mappo = (PROJECT_ROOT / "marl/results/mappo_balanced.json").exists()
    p5_dash = (PROJECT_ROOT / "dashboard/dashboard.html").exists() or (PROJECT_ROOT / "reports/dashboard.html").exists()
    status_p5 = "FOUND" if (p5_mappo and p5_dash) else "MISSING"
    print(f"Phase 5   : {status_p5} (marl/results/mappo_balanced.json, dashboard/dashboard.html)")

    # Phase 6
    p6_man = (PROJECT_ROOT / "artifacts/model_manifest.json").exists()
    p6_comp = (PROJECT_ROOT / "marl/results/reward_settings_comparison.json").exists()
    status_p6 = "FOUND" if (p6_man and p6_comp) else "MISSING"
    print(f"Phase 6   : {status_p6} (artifacts/model_manifest.json, marl/results/reward_settings_comparison.json)")
    print()

    # 2d. Execution Instructions
    print("--- Instructions to Rerun Phases ---")
    print("To rerun Phase 1-2: python run_phase1_2.py")
    print("To rerun Phase 3 ensemble: python run_phase3_ensemble.py")
    print("To rerun Phase 3 LLM: python run_phase3_llm.py  (requires Ollama)")
    print("To rerun Phase 4 XAI: python xai/run_phase4_shap.py")
    print("To rerun Phase 4 adversarial: python adversarial/run_phase4_adversarial.py --stage fgsm")
    print("To rerun Phase 5: python marl/mappo.py --reward_setting balanced")
    print("To rerun Phase 6 packaging: python package_model.py")
    print()

    # 2e. Key Result Numbers from JSON files
    print("--- Key Empirical Results ---")

    # Phase 3 metrics
    p3_path = PROJECT_ROOT / "reports/phase3_metrics.json"
    if p3_path.exists():
        try:
            with open(p3_path, "r", encoding="utf-8") as f:
                p3 = json.load(f)
            acc_8 = p3.get("8class", {}).get("models", {}).get("weighted", {}).get("accuracy")
            c34_models = p3.get("34class", {}).get("models", {})
            acc_34 = c34_models.get("weighted", {}).get("accuracy") or c34_models.get("xgb", {}).get("accuracy")
            f1_34 = c34_models.get("weighted", {}).get("macro_f1") or c34_models.get("xgb", {}).get("macro_f1")
            
            p3_8_str = f"{acc_8 * 100:.2f}%" if acc_8 is not None else "N/A"
            p3_34_str = f"{acc_34 * 100:.2f}%" if acc_34 is not None else "N/A"
            f1_34_str = f"{f1_34:.4f}" if f1_34 is not None else "N/A"
            print(f"Phase 3 Test Accuracy (8-class)  : {p3_8_str}")
            print(f"Phase 3 Test Accuracy (34-class) : {p3_34_str}")
            print(f"Phase 3 Macro F1 (34-class)      : {f1_34_str}")
        except Exception:
            print("Phase 3 Results                  : N/A (error reading reports/phase3_metrics.json)")
    else:
        print("Phase 3 Results                  : N/A (run the phase first)")

    # Phase 5 metrics
    p5_path = PROJECT_ROOT / "marl/results/eval_balanced.json"
    if not p5_path.exists():
        p5_path = PROJECT_ROOT / "reports/marl_eval.json"
    if p5_path.exists():
        try:
            with open(p5_path, "r", encoding="utf-8") as f:
                p5 = json.load(f)
            if "aggregate_return" in p5:
                agg_ret = p5.get("aggregate_return")
                att_pct = p5.get("attacks_contained_pct")
                ben_pct = p5.get("benign_disrupted_pct")
            else:
                bagg = p5.get("settings", {}).get("balanced", {}).get("mappo_aggregate", {})
                agg_ret = bagg.get("mean_return")
                att_pct = bagg.get("attacks_contained", {}).get("percentage")
                ben_pct = bagg.get("benigns_disrupted", {}).get("percentage")

            ret_str = f"{agg_ret:.2f}" if agg_ret is not None else "N/A"
            att_str = f"{att_pct:.2f}%" if att_pct is not None else "N/A"
            ben_str = f"{ben_pct:.2f}%" if ben_pct is not None else "N/A"
            print(f"Phase 5 MAPPO Return (balanced)  : {ret_str}")
            print(f"Phase 5 Attacks Contained        : {att_str}")
            print(f"Phase 5 Benign Disrupted         : {ben_str}")
        except Exception:
            print("Phase 5 Results                  : N/A (error reading marl/results/eval_balanced.json)")
    else:
        print("Phase 5 Results                  : N/A (run the phase first)")

    # 2f. Footer
    print("\n=== END ===")


if __name__ == "__main__":
    main()
