"""
xai/run_phase4_shap.py
======================
Phase 4 - SHAP Explainability Runner

Executes TreeExplainer on B2 tree models:
1. Samples 2,000 stratified test flows (seed=42).
2. Computes XGBoost SHAP values (model weight in B2 ensemble: 0.40).
3. Evaluates Random Forest SHAP values on a stratified stability sample (model weight: 0.60).
4. Generates:
   - Global feature importance bar chart (reports/shap_global_importance.png)
   - Per-class top-5 feature summary chart and table (reports/shap_per_class_importance.png)
   - Local explanations (top 5 features with sign and direction) for the 40 demo flows
     (saved to artifacts/shap_local_40flows.json).
5. Assesses stability between XGBoost and Random Forest feature rankings.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ARTIFACTS_DIR,
    ENCODER_8_PATH,
    RANDOM_SEED,
    REPORTS_DIR,
    TEST_PARQUET,
)
from orchestrator.llm_orchestrator import FeatureUnscaler
from run_phase3_llm import select_balanced_random_flows
from xai.shap_explain import SHAPExplainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_phase4_shap")


def sample_stratified_test_flows(df_test: pd.DataFrame, n_samples: int = 2000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Samples stratified test flows ensuring rare classes are properly represented.
    """
    rng = np.random.default_rng(seed)
    class_counts = df_test["label_8"].value_counts()
    total = len(df_test)
    sampled_indices = []

    for label, count in class_counts.items():
        # Minimum of 10 samples or proportional share
        quota = max(10, int(round((count / total) * n_samples)))
        sub = df_test[df_test["label_8"] == label]
        chosen = rng.choice(sub.index, size=min(len(sub), quota), replace=False)
        sampled_indices.extend(chosen.tolist())

    # Adjust to exact n_samples if slight rounding difference
    if len(sampled_indices) > n_samples:
        sampled_indices = rng.choice(sampled_indices, size=n_samples, replace=False).tolist()

    return df_test.loc[sampled_indices].copy()


def run_shap_analysis(n_samples: int = 2000, rf_sample_size: int = 200) -> dict:
    proc = psutil.Process(os.getpid())
    t0 = time.time()
    init_ram = proc.memory_info().rss / (1024.0 * 1024.0)

    print("=" * 70)
    print("  PHASE 4: SHAP EXPLAINABILITY PIPELINE (TREE EXPLAINER)")
    print("=" * 70)
    print(f"Initial Process RAM: {init_ram:.1f} MB")
    print("Architecture Note: Ensemble comprises RF (0.60) + XGBoost (0.40) + DNN (0.00).")
    print("TreeExplainer explains XGBoost (0.40) globally and checks stability against RF (0.60).")

    # 1. Initialize Explainer
    print("\n[1/5] Loading B2 models and initializing SHAP TreeExplainer ...")
    explainer = SHAPExplainer(target="8class", model_type="b2")
    le_8 = joblib.load(ENCODER_8_PATH)
    class_names = list(le_8.classes_)

    # 2. Load & sample test data
    print(f"\n[2/5] Sampling {n_samples} stratified test flows (seed={RANDOM_SEED}) ...")
    df_test = pd.read_parquet(TEST_PARQUET)
    df_sample = sample_stratified_test_flows(df_test, n_samples=n_samples, seed=RANDOM_SEED)
    print(f"Sampled {len(df_sample)} flows. Class distribution:")
    for c_idx, count in df_sample["label_8"].value_counts().sort_index().items():
        print(f"  Class {c_idx} ({class_names[c_idx]}): {count} flows")

    # 3. Compute SHAP on XGBoost
    print(f"\n[3/5] Computing TreeExplainer SHAP values for XGBoost on {len(df_sample)} flows ...")
    t_xgb_start = time.time()
    shap_xgb = explainer.explain_xgb(df_sample)
    t_xgb = time.time() - t_xgb_start
    print(f"XGBoost SHAP completed in {t_xgb:.2f} s ({t_xgb / len(df_sample) * 1000:.2f} ms/flow)")
    ram_after_xgb = proc.memory_info().rss / (1024.0 * 1024.0)
    print(f"Process RAM after XGBoost SHAP: {ram_after_xgb:.1f} MB")

    # Global importance
    global_imp = explainer.get_global_importance(shap_xgb)
    print("\n--- Global Feature Importance (Top 10 - Mean |SHAP|) ---")
    for feat, val in global_imp.head(10).items():
        print(f"  {feat:<20}: {val:.4f}")

    # Plot Global Importance
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    global_plot_path = REPORTS_DIR / "shap_global_importance.png"
    plt.figure(figsize=(10, 6))
    plt.barh(global_imp.index[::-1], global_imp.values[::-1], color="#1f77b4", edgecolor="black")
    plt.xlabel("Mean |SHAP value| (Average Impact on Model Output Magnitude)")
    plt.title("Phase 4: Global Feature Importance (XGBoost B2, 2,000 Stratified Test Flows)")
    plt.tight_layout()
    plt.savefig(global_plot_path, dpi=200)
    plt.close()
    print(f"[SAVE] Global importance plot saved --> {global_plot_path.name}")

    # Per-class Top-5 Features
    per_class_imp = explainer.get_per_class_importance(shap_xgb)
    per_class_summary = {}
    print("\n--- Per-Class Top-5 Features (8-Class Family Breakdown) ---")
    for c_name, s in per_class_imp.items():
        top5 = s.head(5).to_dict()
        per_class_summary[c_name] = top5
        print(f"[{c_name}]")
        for f_name, f_val in top5.items():
            print(f"    {f_name:<18}: {f_val:.4f}")

    # Plot Per-Class Summary Heatmap/Grouped Bar
    class_plot_path = REPORTS_DIR / "shap_per_class_importance.png"
    df_per_class = pd.DataFrame(per_class_imp).T[global_imp.head(8).index]
    plt.figure(figsize=(12, 6))
    im = plt.imshow(df_per_class.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, label="Mean |SHAP Value|")
    plt.xticks(range(len(df_per_class.columns)), df_per_class.columns, rotation=35, ha="right")
    plt.yticks(range(len(df_per_class.index)), df_per_class.index)
    plt.title("Phase 4: Top Features Mean Impact per Attack Class (8 Classes)")
    plt.tight_layout()
    plt.savefig(class_plot_path, dpi=200)
    plt.close()
    print(f"[SAVE] Per-class importance heatmap saved --> {class_plot_path.name}")

    # 4. Check Stability with Random Forest
    print(f"\n[4/5] Checking Feature Stability: XGBoost vs Random Forest (Sample: {rf_sample_size} flows) ...")
    df_stability_sample = sample_stratified_test_flows(df_test, n_samples=rf_sample_size, seed=RANDOM_SEED)
    t_stab_start = time.time()
    stability_metrics = explainer.check_stability(df_stability_sample)
    t_stab = time.time() - t_stab_start
    print(f"Stability computation completed in {t_stab:.2f} s")
    print(f"  Spearman Rank Correlation: rho = {stability_metrics['spearman_rho']:.4f} (p = {stability_metrics['spearman_pval']:.2e})")
    print(f"  Top-5 XGBoost features:    {stability_metrics['top5_xgb']}")
    print(f"  Top-5 Random Forest feats: {stability_metrics['top5_rf']}")
    print(f"  Top-5 Feature Overlap:     {stability_metrics['top5_overlap_count']}/5 ({stability_metrics['top5_overlap_pct']}%) -> {stability_metrics['top5_overlap']}")

    # 5. Local Explanations for the 40 Demo Flows
    print("\n[5/5] Generating local SHAP explanations for the 40 seeded demo flows ...")
    df_40 = select_balanced_random_flows(df_test, seed=RANDOM_SEED, samples_per_family=5)
    shap_40 = explainer.explain_xgb(df_40)
    unscaled_40 = explainer.unscaler.unscale(df_40[explainer.selected_features])

    # Run predictions on the 40 flows
    prob_dict = explainer.ensemble.predict_probabilities(df_40[explainer.selected_features])
    preds_40 = prob_dict["weighted"]
    pred_classes_40 = np.argmax(preds_40, axis=1)

    local_explanations_40 = {}
    for i, (idx, row) in enumerate(df_40.iterrows()):
        unscaled_dict = unscaled_40.loc[idx].to_dict()
        pred_c_idx = int(pred_classes_40[i])
        pred_c_name = class_names[pred_c_idx]

        local_top5 = explainer.get_local_explanation(
            shap_values_single=shap_40[i],
            unscaled_features_dict=unscaled_dict,
            target_class_idx=pred_c_idx,
            top_k=5,
        )
        local_explanations_40[str(idx)] = {
            "flow_index": int(idx),
            "true_class": class_names[int(row["label_8"])],
            "predicted_class": pred_c_name,
            "predicted_confidence": round(float(preds_40[i, pred_c_idx]), 4),
            "top_shap_features": local_top5,
        }

    # Save to artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = ARTIFACTS_DIR / "shap_local_40flows.json"
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(local_explanations_40, f, indent=2)
    print(f"[SAVE] Local SHAP attributions for 40 demo flows saved --> {local_path.name}")

    total_time = time.time() - t0
    peak_ram = proc.memory_info().rss / (1024.0 * 1024.0)

    print("\n" + "=" * 70)
    print("  PHASE 4 SHAP EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Execution Time: {total_time:.2f} s ({total_time / 60.0:.2f} min)")
    print(f"Peak Process RAM:     {peak_ram:.1f} MB ({peak_ram / 1024.0:.2f} GB) [Ceiling: < 6.0 GB]")
    print(f"XGBoost-RF Stability: Spearman rho = {stability_metrics['spearman_rho']:.4f}, Top-5 Overlap = {stability_metrics['top5_overlap_count']}/5")
    print("=" * 70)

    results = {
        "n_samples": n_samples,
        "rf_sample_size": rf_sample_size,
        "runtime_sec": round(total_time, 2),
        "peak_ram_mb": round(peak_ram, 1),
        "global_importance": global_imp.to_dict(),
        "per_class_summary": per_class_summary,
        "stability": {
            "spearman_rho": stability_metrics["spearman_rho"],
            "spearman_pval": stability_metrics["spearman_pval"],
            "top5_xgb": stability_metrics["top5_xgb"],
            "top5_rf": stability_metrics["top5_rf"],
            "top5_overlap": stability_metrics["top5_overlap"],
            "top5_overlap_count": stability_metrics["top5_overlap_count"],
        },
    }

    with open(REPORTS_DIR / "shap_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE] Metrics saved --> {REPORTS_DIR / 'shap_metrics.json'}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 4 SHAP Analysis")
    parser.add_argument("--samples", type=int, default=2000, help="Test flows for XGBoost SHAP")
    parser.add_argument("--rf-samples", type=int, default=200, help="Test flows for RF stability check")
    args = parser.parse_args()

    run_shap_analysis(n_samples=args.samples, rf_sample_size=args.rf_samples)
