"""
marl/build_cache.py
===================
Stage 0: Environment Data Cache Generation.

Runs the frozen B2 ensemble (RF 0.6 + XGB 0.4) over val_phase2.parquet and
test_phase2.parquet to precompute 8-class probability distributions, predictions,
confidences, and severities. Splits validation into rl_train (70%) and rl_val (30%).

Outputs:
  - data/processed/marl_cache_val.parquet
  - data/processed/marl_cache_test.parquet
  - data/processed/marl_rl_train_indices.json
  - data/processed/marl_rl_val_indices.json
  - reports/marl_cache_metrics.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import psutil

import sys
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ENCODER_8_PATH,
    TEST_PARQUET,
    VAL_PARQUET,
    get_flow_severity,
)
from config_marl import (
    CACHE_METRICS_PATH,
    CACHE_SPLIT_SEED,
    CACHE_TEST_PATH,
    CACHE_VAL_PATH,
    RL_TRAIN_INDICES_PATH,
    RL_TRAIN_SPLIT_RATIO,
    RL_VAL_INDICES_PATH,
    RL_VAL_SPLIT_RATIO,
)
from predict import get_or_load_model, get_selected_features

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
log = logging.getLogger("marl_cache")


def get_peak_ram_mb() -> float:
    """Returns current process RSS in MB."""
    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


def compute_ensemble_cache(
    df: pd.DataFrame,
    class_names: List[str],
    selected_features: List[str],
    batch_size: int = 25000,
) -> pd.DataFrame:
    """
    Computes ensemble predictions and probabilities in batches to strictly manage memory.
    """
    model = get_or_load_model(target="8class", model_type="b2")
    n_rows = len(df)
    
    probs_list = []
    preds_list = []
    confs_list = []

    X_full = df[selected_features]

    for start_idx in range(0, n_rows, batch_size):
        end_idx = min(start_idx + batch_size, n_rows)
        X_batch = X_full.iloc[start_idx:end_idx]
        probs_dict = model.predict_probabilities(X_batch)
        p_weighted = probs_dict["weighted"]  # shape (batch, 8)
        
        batch_preds = np.argmax(p_weighted, axis=1)
        batch_confs = np.max(p_weighted, axis=1)

        probs_list.append(p_weighted)
        preds_list.append(batch_preds)
        confs_list.append(batch_confs)

    all_probs = np.vstack(probs_list)
    all_preds = np.concatenate(preds_list)
    all_confs = np.concatenate(confs_list)

    cache_df = pd.DataFrame(index=df.index)

    # 8-class probability columns
    for idx, cname in enumerate(class_names):
        cache_df[f"prob_{cname}"] = all_probs[:, idx].astype(np.float32)

    cache_df["pred_class_idx"] = all_preds.astype(np.int32)
    cache_df["pred_class"] = [class_names[i] for i in all_preds]
    cache_df["confidence"] = np.round(all_confs, 4).astype(np.float32)

    # Deterministic severity mapping
    severities = [get_flow_severity(cname, conf) for cname, conf in zip(cache_df["pred_class"], all_confs)]
    cache_df["severity"] = severities

    # Ground truth labels
    cache_df["true_label_idx"] = df["label_8"].to_numpy().astype(np.int32)
    cache_df["true_label"] = [class_names[i] for i in cache_df["true_label_idx"]]

    return cache_df


def build_and_save_cache() -> Dict:
    start_time = time.time()
    log.info("Starting Stage 0 cache generation...")

    # Load 8-class encoder class names
    le_8 = joblib.load(ENCODER_8_PATH)
    class_names = [str(c) for c in le_8.classes_]
    features = get_selected_features()
    log.info(f"Target classes ({len(class_names)}): {class_names}")
    log.info(f"Selected features count: {len(features)}")

    # 1. Process Validation Data
    log.info(f"Loading validation parquet: {VAL_PARQUET}")
    df_val = pd.read_parquet(VAL_PARQUET)
    val_rows = len(df_val)
    log.info(f"Validation dataset rows: {val_rows}")

    val_cache = compute_ensemble_cache(df_val, class_names, features)
    val_acc = float((val_cache["pred_class_idx"] == val_cache["true_label_idx"]).mean())
    log.info(f"Ensemble B2 Accuracy on Validation: {val_acc:.4f} ({val_cache['pred_class_idx'].eq(val_cache['true_label_idx']).sum()} / {val_rows})")

    # Save validation cache
    val_cache.to_parquet(CACHE_VAL_PATH, index=True)
    log.info(f"Saved validation cache to {CACHE_VAL_PATH}")

    # 2. Process Test Data
    log.info(f"Loading test parquet: {TEST_PARQUET}")
    df_test = pd.read_parquet(TEST_PARQUET)
    test_rows = len(df_test)
    log.info(f"Test dataset rows: {test_rows}")

    test_cache = compute_ensemble_cache(df_test, class_names, features)
    test_acc = float((test_cache["pred_class_idx"] == test_cache["true_label_idx"]).mean())
    log.info(f"Ensemble B2 Accuracy on Test: {test_acc:.4f} ({test_cache['pred_class_idx'].eq(test_cache['true_label_idx']).sum()} / {test_rows})")

    # Save test cache
    test_cache.to_parquet(CACHE_TEST_PATH, index=True)
    log.info(f"Saved test cache to {CACHE_TEST_PATH}")

    # 3. Create rl_train (70%) and rl_val (30%) stratified/random row index splits
    rng = np.random.default_rng(CACHE_SPLIT_SEED)
    shuffled_indices = rng.permutation(val_rows)
    split_point = int(val_rows * RL_TRAIN_SPLIT_RATIO)

    rl_train_idx = shuffled_indices[:split_point].tolist()
    rl_val_idx = shuffled_indices[split_point:].tolist()

    with open(RL_TRAIN_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(rl_train_idx, f)
    with open(RL_VAL_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(rl_val_idx, f)
    log.info(f"Saved rl_train indices ({len(rl_train_idx)}) to {RL_TRAIN_INDICES_PATH}")
    log.info(f"Saved rl_val indices ({len(rl_val_idx)}) to {RL_VAL_INDICES_PATH}")

    # Calculate accuracies on the splits
    rl_train_preds = val_cache.iloc[rl_train_idx]["pred_class_idx"].to_numpy()
    rl_train_trues = val_cache.iloc[rl_train_idx]["true_label_idx"].to_numpy()
    rl_train_acc = float((rl_train_preds == rl_train_trues).mean())

    rl_val_preds = val_cache.iloc[rl_val_idx]["pred_class_idx"].to_numpy()
    rl_val_trues = val_cache.iloc[rl_val_idx]["true_label_idx"].to_numpy()
    rl_val_acc = float((rl_val_preds == rl_val_trues).mean())

    # Calculate class counts
    def get_counts(series: pd.Series) -> Dict[str, int]:
        c = series.value_counts().to_dict()
        return {str(k): int(v) for k, v in c.items()}

    metrics = {
        "description": "Stage 0 MARL cache generation and validation split",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(time.time() - start_time, 2),
        "peak_ram_mb": round(get_peak_ram_mb(), 2),
        "cache_split_seed": CACHE_SPLIT_SEED,
        "class_names": class_names,
        "row_counts": {
            "validation_total": val_rows,
            "rl_train": len(rl_train_idx),
            "rl_val": len(rl_val_idx),
            "test": test_rows,
        },
        "split_ratios": {
            "rl_train": RL_TRAIN_SPLIT_RATIO,
            "rl_val": RL_VAL_SPLIT_RATIO,
        },
        "accuracies": {
            "val_total": round(val_acc, 6),
            "rl_train": round(rl_train_acc, 6),
            "rl_val": round(rl_val_acc, 6),
            "test": round(test_acc, 6),
        },
        "class_distribution_true": {
            "rl_train": get_counts(val_cache.iloc[rl_train_idx]["true_label"]),
            "rl_val": get_counts(val_cache.iloc[rl_val_idx]["true_label"]),
            "test": get_counts(test_cache["true_label"]),
        },
        "class_distribution_pred": {
            "rl_train": get_counts(val_cache.iloc[rl_train_idx]["pred_class"]),
            "rl_val": get_counts(val_cache.iloc[rl_val_idx]["pred_class"]),
            "test": get_counts(test_cache["pred_class"]),
        },
        "severity_distribution": {
            "rl_train": get_counts(val_cache.iloc[rl_train_idx]["severity"]),
            "rl_val": get_counts(val_cache.iloc[rl_val_idx]["severity"]),
            "test": get_counts(test_cache["severity"]),
        },
    }

    with open(CACHE_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved cache metrics to {CACHE_METRICS_PATH}")

    log.info(f"Stage 0 finished in {metrics['runtime_seconds']}s, peak RAM {metrics['peak_ram_mb']} MB.")
    return metrics


if __name__ == "__main__":
    build_and_save_cache()
