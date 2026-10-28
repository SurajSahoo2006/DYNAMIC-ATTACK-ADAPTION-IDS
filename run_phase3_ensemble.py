"""
run_phase3_ensemble.py
======================
Phase 3 - Ensemble Classifier Pipeline

Executes:
  1. Primary Target: 8-class classification (Benign + 7 attack families)
  2. Secondary Target: 34-class classification (fine-grained)
  3. Validation-based early stopping & ensemble weight tuning (no SMOTE leak)
  4. Test set evaluation: RF vs XGBoost vs PyTorch DNN vs Soft & Weighted Ensemble
  5. Confusion matrix plot for 8-class model saved to reports/confusion_matrix_8class.png
  6. Memory tracking and latency measurement
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import (
    TRAIN_PARQUET,
    TEST_PARQUET,
    CONFUSION_MATRIX_8CLASS_PATH,
    REPORTS_DIR,
    ENCODER_8_PATH,
    ENCODER_34_PATH,
    SELECTED_FEATURES_PATH,
)
from models.load_validation import load_validation_data
from models.ensemble import EnsembleClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase3_ensemble")


def get_current_ram_mb() -> float:
    """Returns current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


def plot_confusion_matrix_8class(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    """Plots and saves normalized confusion matrix for the 8-class ensemble."""
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Ratio"},
    )
    plt.title("Phase 3 Ensemble Classifier - 8-Class Confusion Matrix (Test Set)", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] Confusion matrix saved to: {out_path.name}")


def run_pipeline(args: argparse.Namespace) -> None:
    t_start = time.time()
    peak_ram_mb = get_current_ram_mb()

    print("=" * 70)
    print("  PHASE 3: ENSEMBLE CLASSIFIER (RF + XGBOOST + PYTORCH DNN)")
    print("=" * 70)
    print(f"Initial RAM: {peak_ram_mb:.1f} MB")

    # 1. Load artifacts
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_features = json.load(f)
    le_8 = joblib.load(ENCODER_8_PATH)
    le_34 = joblib.load(ENCODER_34_PATH)
    classes_8 = list(le_8.classes_)
    classes_34 = list(le_34.classes_)

    print(f"Features: {len(selected_features)} features selected from Phase 2")

    # 2. Load Processed Parquet Data
    print(f"\n[DATA] Loading Train Parquet: {TRAIN_PARQUET.name} ...")
    df_train = pd.read_parquet(TRAIN_PARQUET)
    X_train = df_train[selected_features]
    y_train_8 = df_train["label_8"].to_numpy(dtype=np.int64)
    y_train_34 = df_train["label_34"].to_numpy(dtype=np.int64)
    print(f"       Train rows: {len(df_train):,} (SMOTE-balanced)")

    print(f"[DATA] Loading Untouched Test Parquet: {TEST_PARQUET.name} ...")
    df_test = pd.read_parquet(TEST_PARQUET)
    X_test = df_test[selected_features]
    y_test_8 = df_test["label_8"].to_numpy(dtype=np.int64)
    y_test_34 = df_test["label_34"].to_numpy(dtype=np.int64)
    print(f"       Test rows:  {len(df_test):,} (untouched real distribution)")

    # 3. Load Pure Validation Data (NO SMOTE)
    print(f"\n[DATA] Loading Pure Validation Data (NO SMOTE) ...")
    X_val, y_val_8, y_val_34 = load_validation_data(
        sample_per_class=args.val_sample,
        force_reload=args.force_reload_val,
    )
    peak_ram_mb = max(peak_ram_mb, get_current_ram_mb())
    print(f"       Validation rows: {len(X_val):,}")
    print(f"Current RAM: {get_current_ram_mb():.1f} MB")

    all_metrics: Dict[str, Any] = {}

    # ==================================================================
    # STEP 4: PRIMARY TARGET - 8-CLASS ENSEMBLE
    # ==================================================================
    print("\n" + "#" * 70)
    print("  STEP 4: PRIMARY TARGET - 8-CLASS ENSEMBLE")
    print("#" * 70)

    ens_8 = EnsembleClassifier(
        target_name="8class",
        num_classes=8,
        class_names=classes_8,
    )
    ens_8.fit(X_train, y_train_8, X_val, y_val_8)
    peak_ram_mb = max(peak_ram_mb, get_current_ram_mb())

    # Save 8-class models
    ens_8.save()

    # Evaluate on test set
    print("\n[EVAL] Evaluating 8-class models on untouched test set ...")
    eval_8 = ens_8.evaluate(X_test, y_test_8)
    eval_8["training_telemetry"] = ens_8.training_telemetry
    all_metrics["8class"] = eval_8

    print("\n--- 8-Class Test Results Summary ---")
    print(f"{'Model':<30} {'Accuracy':>10} {'Macro Prec':>12} {'Macro Rec':>12} {'Macro F1':>10}")
    print("-" * 76)
    for mk, mstats in eval_8["models"].items():
        print(
            f"{mstats['name']:<30} "
            f"{mstats['accuracy'] * 100:>9.2f}% "
            f"{mstats['macro_precision'] * 100:>11.2f}% "
            f"{mstats['macro_recall'] * 100:>11.2f}% "
            f"{mstats['macro_f1'] * 100:>9.2f}%"
        )
    print("-" * 76)
    print(f"Test Inference Time: {eval_8['inference_time_sec']:.3f} s ({eval_8['ms_per_sample']:.4f} ms/sample)")

    print("\n--- 8-Class Per-Class Metrics (Weighted Ensemble) ---")
    print(f"{'#':<3} {'Class Name':<16} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>10}  {'Notes'}")
    print("-" * 76)
    for row in eval_8["per_class_table"]:
        print(
            f"{row['class_idx']:<3} {row['class_name']:<16} "
            f"{row['precision'] * 100:>9.2f}% {row['recall'] * 100:>9.2f}% {row['f1'] * 100:>7.2f}% "
            f"{row['support']:>10,}  {row['flag']}"
        )

    # Plot Confusion Matrix
    cm_8 = np.array(eval_8["confusion_matrix"])
    plot_confusion_matrix_8class(cm_8, classes_8, CONFUSION_MATRIX_8CLASS_PATH)

    # Free memory before 34-class model
    del ens_8
    gc.collect()
    peak_ram_mb = max(peak_ram_mb, get_current_ram_mb())

    # ==================================================================
    # STEP 5: SECONDARY TARGET - 34-CLASS ENSEMBLE
    # ==================================================================
    if not args.skip_34:
        print("\n" + "#" * 70)
        print("  STEP 5: SECONDARY TARGET - 34-CLASS ENSEMBLE")
        print("#" * 70)

        ens_34 = EnsembleClassifier(
            target_name="34class",
            num_classes=34,
            class_names=classes_34,
        )
        ens_34.fit(X_train, y_train_34, X_val, y_val_34)
        peak_ram_mb = max(peak_ram_mb, get_current_ram_mb())

        # Save 34-class models
        ens_34.save()

        # Evaluate on test set
        print("\n[EVAL] Evaluating 34-class models on untouched test set ...")
        eval_34 = ens_34.evaluate(X_test, y_test_34)
        eval_34["training_telemetry"] = ens_34.training_telemetry
        all_metrics["34class"] = eval_34

        print("\n--- 34-Class Test Results Summary ---")
        print(f"{'Model':<30} {'Accuracy':>10} {'Macro Prec':>12} {'Macro Rec':>12} {'Macro F1':>10}")
        print("-" * 76)
        for mk, mstats in eval_34["models"].items():
            print(
                f"{mstats['name']:<30} "
                f"{mstats['accuracy'] * 100:>9.2f}% "
                f"{mstats['macro_precision'] * 100:>11.2f}% "
                f"{mstats['macro_recall'] * 100:>11.2f}% "
                f"{mstats['macro_f1'] * 100:>9.2f}%"
            )
        print("-" * 76)
        print(f"Test Inference Time: {eval_34['inference_time_sec']:.3f} s ({eval_34['ms_per_sample']:.4f} ms/sample)")

        print("\n--- 34-Class Per-Class Metrics (Weighted Ensemble - sample view) ---")
        print(f"{'#':<3} {'Class Name':<28} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>8}  {'Notes'}")
        print("-" * 84)
        for row in eval_34["per_class_table"]:
            print(
                f"{row['class_idx']:<3} {row['class_name']:<28} "
                f"{row['precision'] * 100:>9.2f}% {row['recall'] * 100:>9.2f}% {row['f1'] * 100:>7.2f}% "
                f"{row['support']:>8,}  {row['flag']}"
            )
        del ens_34
        gc.collect()

    peak_ram_mb = max(peak_ram_mb, get_current_ram_mb())
    total_time = time.time() - t_start

    # Clean predictions array from JSON serialization
    for tgt in all_metrics:
        for m in all_metrics[tgt]["models"]:
            if "predictions" in all_metrics[tgt]["models"][m]:
                del all_metrics[tgt]["models"][m]["predictions"]

    all_metrics["execution_telemetry"] = {
        "total_runtime_sec": round(total_time, 2),
        "peak_ram_mb": round(peak_ram_mb, 1),
        "peak_ram_gb": round(peak_ram_mb / 1024.0, 2),
    }

    # Save metrics JSON for reporting
    metrics_path = REPORTS_DIR / "phase3_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[SAVE] Full evaluation metrics saved to: {metrics_path.name}")

    print("\n" + "=" * 70)
    print("  PHASE 3 ENSEMBLE PIPELINE COMPLETE")
    print(f"  Total Runtime: {total_time:.1f} s ({total_time / 60.0:.2f} min)")
    print(f"  Measured Peak RAM: {peak_ram_mb:.1f} MB ({peak_ram_mb / 1024.0:.2f} GB)  [Safety limit: 6.0 GB]")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 3 Ensemble Training & Evaluation")
    parser.add_argument("--val-sample", type=int, default=5000, help="Per-class sample cap for validation data")
    parser.add_argument("--force-reload-val", action="store_true", help="Force re-reading validation.csv")
    parser.add_argument("--skip-34", action="store_true", help="Skip secondary 34-class training (8-class only)")
    args = parser.parse_args()

    run_pipeline(args)
