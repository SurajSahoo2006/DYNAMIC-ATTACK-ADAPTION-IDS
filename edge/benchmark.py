"""
edge/benchmark.py
=================
Phase 11 Edge Compression and Latency Benchmark.
Measures model footprint (uncompressed and compressed), per-flow latency,
throughput, and peak memory under strict single-thread CPU execution.
"""

from __future__ import annotations

import os
# Enforce single-thread execution for all math/linear algebra backends
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy
import gc
import json
import logging
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, f1_score

import sys
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODELS_DIR,
    TEST_PARQUET,
    SELECTED_FEATURES_PATH,
    RANDOM_SEED,
)
from models.load_validation import load_validation_data

RESULTS_DIR = PROJECT_ROOT / "edge" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)


def get_environment_info() -> Dict[str, Any]:
    """
    Returns environment metadata without machine-specific or personal identifiers
    (no username, no hostname).
    """
    import sklearn
    import xgboost
    import torch

    return {
        "cpu_model": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": platform.python_version(),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "psutil": psutil.__version__,
        },
    }


def measure_model_sizes_mb(model_files_or_objs: List[Any]) -> Tuple[float, float]:
    """
    Measures total size in MB for model files or in-memory model objects
    under joblib compress=0 and compress=3.
    """
    total_uncomp_bytes = 0
    total_comp_bytes = 0

    for item in model_files_or_objs:
        if isinstance(item, (str, Path)) and Path(item).exists():
            obj = joblib.load(item)
        else:
            obj = item

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tname = f.name
        try:
            # compress=0
            joblib.dump(obj, tname, compress=0)
            total_uncomp_bytes += os.path.getsize(tname)

            # compress=3
            joblib.dump(obj, tname, compress=3)
            total_comp_bytes += os.path.getsize(tname)
        finally:
            if os.path.exists(tname):
                try:
                    os.remove(tname)
                except OSError:
                    pass

    size_mb_uncomp = round(total_uncomp_bytes / (1024 * 1024), 2)
    size_mb_comp = round(total_comp_bytes / (1024 * 1024), 2)
    return size_mb_uncomp, size_mb_comp


def benchmark(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: Union[np.ndarray, pd.DataFrame],
    model_files: List[Any],
    X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    y_val: Optional[np.ndarray] = None,
    n_runs: int = 5,
    warmup_flows: int = 50,
    eval_flows: int = 500,
    throughput_batch: int = 1000,
) -> Dict[str, Any]:
    """
    Reusable benchmark function used by every variant.

    Parameters
    ----------
    predict_fn : callable taking a 2D numpy array and returning predicted labels or probabilities
    X : test inputs (2,000 real scaled test rows, 19 features)
    model_files : list of model files or model objects to measure size on disk
    X_val : optional validation features for accuracy/macro F1
    y_val : optional validation true labels
    n_runs : number of repeated runs (default 5)
    warmup_flows : warm-up flows before timing (default 50)
    eval_flows : number of flows timed per run (default 500)
    throughput_batch : batch size for throughput measurement (default 1000)

    Returns
    -------
    dict with environment, size, latency (median, p95, min-max), throughput, peak memory,
    and validation metrics.
    """
    X_np = X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float32)

    # 1. Size on disk
    size_uncomp, size_comp = measure_model_sizes_mb(model_files)

    # 2. Per-flow latency (batch size 1, 50 warmup + 500 flows, 5 runs)
    per_flow_runs = []
    for r in range(n_runs):
        # Warmup
        for i in range(warmup_flows):
            _ = predict_fn(X_np[i : i + 1])

        # Timed flows
        flow_times = []
        for i in range(warmup_flows, warmup_flows + eval_flows):
            flow = X_np[i : i + 1]
            t0 = time.perf_counter()
            _ = predict_fn(flow)
            t1 = time.perf_counter()
            flow_times.append((t1 - t0) * 1000.0)

        run_median = float(np.median(flow_times))
        run_p95 = float(np.percentile(flow_times, 95))
        per_flow_runs.append({
            "run": r + 1,
            "median_ms": round(run_median, 3),
            "p95_ms": round(run_p95, 3),
            "min_ms": round(float(np.min(flow_times)), 3),
            "max_ms": round(float(np.max(flow_times)), 3),
        })

    median_latencies = [r["median_ms"] for r in per_flow_runs]
    p95_latencies = [r["p95_ms"] for r in per_flow_runs]

    latency_summary = {
        "median_ms": round(float(np.median(median_latencies)), 3),
        "p95_ms": round(float(np.median(p95_latencies)), 3),
        "min_ms": round(float(np.min(median_latencies)), 3),
        "max_ms": round(float(np.max(median_latencies)), 3),
        "range_ms": [round(float(np.min(median_latencies)), 3), round(float(np.max(median_latencies)), 3)],
        "runs": per_flow_runs,
    }

    # 3. Throughput (batch size 1,000, 5 runs)
    throughput_runs = []
    X_batch = X_np[:throughput_batch]
    for r in range(n_runs):
        t0 = time.perf_counter()
        _ = predict_fn(X_batch)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        fps = throughput_batch / elapsed if elapsed > 0 else 0.0
        throughput_runs.append(round(fps, 2))

    throughput_summary = {
        "median_fps": round(float(np.median(throughput_runs)), 2),
        "min_fps": round(float(np.min(throughput_runs)), 2),
        "max_fps": round(float(np.max(throughput_runs)), 2),
        "range_fps": [round(float(np.min(throughput_runs)), 2), round(float(np.max(throughput_runs)), 2)],
        "runs": throughput_runs,
    }

    # 4. Peak memory: resident memory delta while predicting batch of 1,000
    process = psutil.Process(os.getpid())
    gc.collect()
    rss_before = process.memory_info().rss
    _ = predict_fn(X_batch)
    rss_after = process.memory_info().rss
    mem_delta_mb = round(max(0.0, (rss_after - rss_before) / (1024 * 1024)), 4)

    # 5. Validation evaluation if provided
    val_accuracy = None
    val_macro_f1 = None
    if X_val is not None and y_val is not None:
        X_val_np = X_val.to_numpy(dtype=np.float32) if hasattr(X_val, "to_numpy") else np.asarray(X_val, dtype=np.float32)
        val_preds_raw = predict_fn(X_val_np)
        if val_preds_raw.ndim > 1:
            val_preds = np.argmax(val_preds_raw, axis=1)
        else:
            val_preds = val_preds_raw
        val_accuracy = round(float(accuracy_score(y_val, val_preds)), 4)
        val_macro_f1 = round(float(f1_score(y_val, val_preds, average="macro", zero_division=0)), 4)

    results = {
        "environment": get_environment_info(),
        "size_mb_uncompressed": size_uncomp,
        "size_mb_compressed": size_comp,
        "latency_per_flow_ms": latency_summary,
        "throughput_fps": throughput_summary,
        "peak_memory_delta_mb": mem_delta_mb,
    }
    if val_accuracy is not None:
        results["validation_accuracy"] = val_accuracy
        results["validation_macro_f1"] = val_macro_f1

    return results


def run_baseline_benchmark() -> Dict[str, Any]:
    """Runs baseline footprint and latency benchmark for Phase 3 B2 ensemble."""
    print("=" * 70)
    print("PHASE 11: BASELINE FOOTPRINT AND LATENCY BENCHMARK")
    print("=" * 70)

    # 1. Load test data: 2,000 real scaled test rows (seed 42), 19 features
    print("[1/4] Loading test inputs (2,000 samples, seed 42) ...")
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_features = json.load(f)

    df_test = pd.read_parquet(TEST_PARQUET)
    df_2k = df_test[selected_features].sample(n=2000, random_state=RANDOM_SEED).reset_index(drop=True)
    X_2k = df_2k.to_numpy(dtype=np.float32)
    print(f"  Test samples shape: {X_2k.shape}")

    # 2. Load validation data
    print("[2/4] Loading validation set for macro F1 and accuracy ...")
    X_val, y_val_8, _ = load_validation_data()
    X_val_np = X_val[selected_features].to_numpy(dtype=np.float32)
    print(f"  Validation samples shape: {X_val_np.shape}")

    # 3. Load baseline B2 models
    print("[3/4] Loading baseline Phase 3 B2 ensemble (RF 0.60 + XGBoost 0.40) ...")
    rf_path = MODELS_DIR / "rf_8class_b2.joblib"
    xgb_path = MODELS_DIR / "xgb_8class_b2.joblib"
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)

    # Single-thread enforcement
    rf_model.n_jobs = 1
    xgb_model.set_params(n_jobs=1)

    model_files = [rf_path, xgb_path]

    # Predict function
    def baseline_predict_fn(x: np.ndarray) -> np.ndarray:
        p_rf = rf_model.predict_proba(x)
        p_xgb = xgb_model.predict_proba(x)
        probs = 0.60 * p_rf + 0.40 * p_xgb
        return np.argmax(probs, axis=1)

    # 4. Run benchmark
    print("[4/4] Executing benchmark suite ...")
    bench_res = benchmark(
        predict_fn=baseline_predict_fn,
        X=X_2k,
        model_files=model_files,
        X_val=X_val_np,
        y_val=y_val_8,
    )

    baseline_output = {
        "variant": "baseline",
        "description": "Phase 3 B2 Ensemble (RF 0.60 + XGBoost 0.40, 19 features)",
        "model_files": [
            "models/rf_8class_b2.joblib",
            "models/xgb_8class_b2.joblib",
        ],
        **bench_res,
    }

    out_file = RESULTS_DIR / "baseline.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(baseline_output, f, indent=2)

    print(f"\n[DONE] Baseline benchmark saved to {out_file}")
    print(f"  Size uncompressed: {baseline_output['size_mb_uncompressed']} MB")
    print(f"  Size compressed:   {baseline_output['size_mb_compressed']} MB")
    print(f"  Per-flow latency:  median={baseline_output['latency_per_flow_ms']['median_ms']:.2f} ms | p95={baseline_output['latency_per_flow_ms']['p95_ms']:.2f} ms")
    print(f"  Throughput:        {baseline_output['throughput_fps']['median_fps']:.1f} flows/sec")
    print(f"  Peak memory delta: {baseline_output['peak_memory_delta_mb']:.4f} MB")
    print(f"  Val accuracy:      {baseline_output['validation_accuracy']:.4f}")
    print(f"  Val macro F1:      {baseline_output['validation_macro_f1']:.4f}")

    return baseline_output


if __name__ == "__main__":
    run_baseline_benchmark()
