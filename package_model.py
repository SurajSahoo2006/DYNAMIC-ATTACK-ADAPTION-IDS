"""
package_model.py
================
Verifies all Phase 1-5 artifacts exist, extracts empirical metrics from Phase 3-5
result JSON files, and writes artifacts/model_manifest.json.

Prints:
  - Artifact verification table (FOUND / MISSING)
  - Pretty JSON manifest to stdout
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

PROJECT_ROOT = Path(__file__).parent.resolve()

REQUIRED_ITEMS: List[Union[str, Tuple[str, ...]]] = [
    "artifacts/scaler.pkl",
    "artifacts/label_encoder_8.pkl",
    "artifacts/label_encoder_34.pkl",
    "artifacts/selected_features.json",
    ("models/rf_model.pkl", "models/rf_model.joblib"),
    ("models/xgb_model.pkl", "models/xgb_model.joblib", "models/xgb_model.ubj", "models/xgb_model.json"),
    "marl/results/mappo_balanced.json",
    "reports/phase3_summary.md",
    "reports/phase4_summary.md",
    "reports/phase5_summary.md",
]


def check_artifacts() -> Tuple[List[str], List[str]]:
    missing: List[str] = []
    found: List[str] = []

    print("\n" + "=" * 70)
    print("                 PHASE 1-5 ARTIFACT VERIFICATION")
    print("=" * 70)

    for item in REQUIRED_ITEMS:
        if isinstance(item, tuple):
            matched = None
            for opt in item:
                if (PROJECT_ROOT / opt).exists():
                    matched = opt
                    break
            if matched:
                print(f"  [FOUND]   {matched}")
                found.append(matched)
            else:
                target = item[0]
                print(f"  [MISSING] {target} (none of {item} found)")
                missing.append(target)
        else:
            p = PROJECT_ROOT / item
            if p.exists():
                print(f"  [FOUND]   {item}")
                found.append(item)
            else:
                print(f"  [MISSING] {item}")
                missing.append(item)

    print("=" * 70 + "\n")
    return found, missing


def build_manifest(missing_artifacts: List[str]) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "project": "Dynamic Attack Adaptation IDS",
        "dataset": "CICIoT2023",
        "phase1_2": {
            "n_features_selected": 19,
            "feature_list": None,
        },
        "phase3": {
            "ensemble": "RF(0.60)+XGB(0.40)",
            "test_acc_8class": None,
            "test_acc_34class": None,
            "macro_f1_34class": None,
        },
        "phase4": {
            "shap_top_feature": None,
            "fgsm_attack_success_rate": None,
        },
        "phase5": {
            "mappo_aggregate_return_balanced": None,
            "attacks_contained_pct": None,
            "benign_disrupted_pct": None,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "missing_artifacts": missing_artifacts,
    }

    # 1. Phase 1-2: Selected Features
    feat_path = PROJECT_ROOT / "artifacts" / "selected_features.json"
    if feat_path.exists():
        try:
            with open(feat_path, "r", encoding="utf-8") as f:
                features = json.load(f)
            if isinstance(features, list):
                manifest["phase1_2"]["feature_list"] = features
                manifest["phase1_2"]["n_features_selected"] = len(features)
            elif isinstance(features, dict) and "selected_features" in features:
                flist = features["selected_features"]
                manifest["phase1_2"]["feature_list"] = flist
                manifest["phase1_2"]["n_features_selected"] = len(flist)
        except Exception as e:
            print(f"WARNING: Error reading selected_features.json: {e}")

    # 2. Phase 3: Metrics (phase3_metrics.json)
    p3_path = PROJECT_ROOT / "reports" / "phase3_metrics.json"
    if p3_path.exists():
        try:
            with open(p3_path, "r", encoding="utf-8") as f:
                p3 = json.load(f)
            # 8-class test accuracy
            acc_8 = p3.get("8class", {}).get("models", {}).get("weighted", {}).get("accuracy")
            if acc_8 is not None:
                manifest["phase3"]["test_acc_8class"] = round(float(acc_8), 4)

            # 34-class test accuracy & macro F1
            c34_models = p3.get("34class", {}).get("models", {})
            acc_34 = c34_models.get("weighted", {}).get("accuracy") or c34_models.get("xgb", {}).get("accuracy")
            f1_34 = c34_models.get("weighted", {}).get("macro_f1") or c34_models.get("xgb", {}).get("macro_f1")
            if acc_34 is not None:
                manifest["phase3"]["test_acc_34class"] = round(float(acc_34), 4)
            if f1_34 is not None:
                manifest["phase3"]["macro_f1_34class"] = round(float(f1_34), 4)
        except Exception as e:
            print(f"WARNING: Error reading phase3_metrics.json: {e}")

    # 3. Phase 4: SHAP top feature & FGSM attack success rate
    shap_path = PROJECT_ROOT / "reports" / "shap_metrics.json"
    if shap_path.exists():
        try:
            with open(shap_path, "r", encoding="utf-8") as f:
                sdata = json.load(f)
            g_imp = sdata.get("global_importance", {})
            if g_imp:
                top_feat = next(iter(g_imp.keys()))
                manifest["phase4"]["shap_top_feature"] = top_feat
        except Exception as e:
            print(f"WARNING: Error reading shap_metrics.json: {e}")

    fgsm_path = PROJECT_ROOT / "adversarial" / "results" / "fgsm_results.json"
    if not fgsm_path.exists():
        fgsm_path = PROJECT_ROOT / "reports" / "adversarial_fgsm.json"
    if fgsm_path.exists():
        try:
            with open(fgsm_path, "r", encoding="utf-8") as f:
                fg_data = json.load(f)
            # Find eps = 0.10 transfer ASR
            results_list = fg_data.get("fgsm_and_random_noise_results", [])
            eps10 = next((r for r in results_list if abs(r.get("epsilon", 0) - 0.10) < 1e-3), None)
            if eps10 and "transfer_asr" in eps10:
                manifest["phase4"]["fgsm_attack_success_rate"] = round(float(eps10["transfer_asr"]), 4)
            elif eps10 and "ensemble_adversarial_acc" in eps10:
                clean_acc = eps10.get("ensemble_clean_acc", 0.944)
                adv_acc = eps10["ensemble_adversarial_acc"]
                computed_asr = (clean_acc - adv_acc) / max(1e-6, clean_acc)
                manifest["phase4"]["fgsm_attack_success_rate"] = round(float(computed_asr), 4)
        except Exception as e:
            print(f"WARNING: Error reading fgsm results: {e}")

    # 4. Phase 5: MAPPO aggregate metrics for 'balanced'
    marl_eval_path = PROJECT_ROOT / "marl" / "results" / "eval_balanced.json"
    if not marl_eval_path.exists():
        marl_eval_path = PROJECT_ROOT / "reports" / "marl_eval.json"
    if marl_eval_path.exists():
        try:
            with open(marl_eval_path, "r", encoding="utf-8") as f:
                mdata = json.load(f)
            if "aggregate_return" in mdata:
                manifest["phase5"]["mappo_aggregate_return_balanced"] = round(float(mdata["aggregate_return"]), 4)
                manifest["phase5"]["attacks_contained_pct"] = round(float(mdata["attacks_contained_pct"]), 2)
                manifest["phase5"]["benign_disrupted_pct"] = round(float(mdata["benign_disrupted_pct"]), 2)
            elif "settings" in mdata and "balanced" in mdata["settings"]:
                bagg = mdata["settings"]["balanced"].get("mappo_aggregate", {})
                if "mean_return" in bagg:
                    manifest["phase5"]["mappo_aggregate_return_balanced"] = round(float(bagg["mean_return"]), 4)
                if "attacks_contained" in bagg:
                    manifest["phase5"]["attacks_contained_pct"] = round(float(bagg["attacks_contained"]["percentage"]), 2)
                if "benigns_disrupted" in bagg:
                    manifest["phase5"]["benign_disrupted_pct"] = round(float(bagg["benigns_disrupted"]["percentage"]), 2)
        except Exception as e:
            print(f"WARNING: Error reading marl evaluation: {e}")

    return manifest


def main():
    found, missing = check_artifacts()
    manifest = build_manifest(missing)

    out_path = PROJECT_ROOT / "artifacts" / "model_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 70)
    print("                    MODEL MANIFEST (PRETTY JSON)")
    print("=" * 70)
    print(json.dumps(manifest, indent=2))
    print("=" * 70 + "\n")
    print(f"[OK] Manifest saved to {out_path}")


if __name__ == "__main__":
    main()
