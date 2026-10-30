"""
edge/benchmark.py
=================
Phase 11 Edge Compression and Latency Benchmark.
Measures model footprint (uncompressed and compressed), per-flow latency,
throughput, and true resident memory (psutil RSS) under single-thread execution.
Supports running variants in isolated fresh subprocesses.
"""

from __future__ import annotations

import os
# Enforce single-thread execution for all math/linear algebra backends
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import copy
import gc
import json
import logging
import platform
import tempfile
import threading
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

EDGE_MODELS_DIR = PROJECT_ROOT / "edge" / "models"
EDGE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def get_environment_info() -> Dict[str, Any]:
    """Returns environment metadata without machine-specific or personal identifiers."""
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
    initial_rss_mb: Optional[float] = None,
    load_rss_mb: Optional[float] = None,
    n_runs: int = 5,
    warmup_flows: int = 50,
    eval_flows: int = 500,
    throughput_batch: int = 1000,
) -> Dict[str, Any]:
    """
    Reusable benchmark function used by every variant.
    Measures size, per-flow latency, throughput, peak resident memory (psutil RSS),
    and validation metrics.
    """
    proc = psutil.Process()
    if initial_rss_mb is None:
        initial_rss_mb = round(proc.memory_info().rss / (1024 * 1024), 2)
    if load_rss_mb is None:
        load_rss_mb = round(proc.memory_info().rss / (1024 * 1024), 2)

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

    # 3. Throughput and Peak Resident Memory (RSS sampled every 5 ms during batch prediction)
    throughput_runs = []
    X_batch = X_np[:throughput_batch]

    stop_event = threading.Event()
    peak_rss_holder = [proc.memory_info().rss]

    def rss_sampler():
        while not stop_event.is_set():
            try:
                curr_rss = proc.memory_info().rss
                if curr_rss > peak_rss_holder[0]:
                    peak_rss_holder[0] = curr_rss
            except Exception:
                pass
            time.sleep(0.005)

    sampler_thread = threading.Thread(target=rss_sampler, daemon=True)
    sampler_thread.start()

    for r in range(n_runs):
        t0 = time.perf_counter()
        _ = predict_fn(X_batch)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        fps = throughput_batch / elapsed if elapsed > 0 else 0.0
        throughput_runs.append(round(fps, 2))

    stop_event.set()
    sampler_thread.join(timeout=1.0)

    # Final check of peak RSS
    final_rss = proc.memory_info().rss
    if final_rss > peak_rss_holder[0]:
        peak_rss_holder[0] = final_rss

    raw_peak_mb = round(peak_rss_holder[0] / (1024 * 1024), 2)
    peak_rss_mb = max(load_rss_mb, raw_peak_mb)

    # Verify memory values are non-zero
    if load_rss_mb <= 0:
        log.error(f"ERROR: load_rss_mb is not positive ({load_rss_mb} MB)!")
        raise ValueError(f"ERROR: load_rss_mb is not positive ({load_rss_mb} MB)!")
    if peak_rss_mb <= 0:
        log.error(f"ERROR: peak_rss_mb is not positive ({peak_rss_mb} MB)!")
        raise ValueError(f"ERROR: peak_rss_mb is not positive ({peak_rss_mb} MB)!")

    peak_delta_mb = round(max(0.0, peak_rss_mb - initial_rss_mb), 2)

    throughput_summary = {
        "median_fps": round(float(np.median(throughput_runs)), 2),
        "min_fps": round(float(np.min(throughput_runs)), 2),
        "max_fps": round(float(np.max(throughput_runs)), 2),
        "range_fps": [round(float(np.min(throughput_runs)), 2), round(float(np.max(throughput_runs)), 2)],
        "runs": throughput_runs,
    }

    # 4. Validation evaluation if provided
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
        "initial_rss_mb": initial_rss_mb,
        "load_rss_mb": load_rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "peak_memory_delta_mb": peak_delta_mb,
    }
    if val_accuracy is not None:
        results["validation_accuracy"] = val_accuracy
        results["validation_macro_f1"] = val_macro_f1

    return results


def run_single_variant_process(
    variant_name: str,
    rf_path: str,
    xgb_path: str,
    out_json_path: str,
    category: str = "Variant",
    description: str = "",
    float32_cast: bool = False,
    extra_fields: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Executes a complete benchmark in the current process (designed to be invoked
    in a fresh subprocess for each variant).
    """
    proc = psutil.Process()

    # 1. Load test data inputs first (small memory footprint ~150 KB)
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_features = json.load(f)

    df_test = pd.read_parquet(TEST_PARQUET)
    df_2k = df_test[selected_features].sample(n=2000, random_state=RANDOM_SEED).reset_index(drop=True)
    X_2k = df_2k.to_numpy(dtype=np.float32)

    # Record initial resident memory before model loading
    initial_rss_mb = round(proc.memory_info().rss / (1024 * 1024), 2)

    # 2. Load models into memory
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)
    rf_model.n_jobs = 1
    xgb_model.set_params(n_jobs=1)

    # Record resident memory after model loading
    load_rss_mb = round(proc.memory_info().rss / (1024 * 1024), 2)

    # 3. Define predict_fn
    def predict_fn(x: np.ndarray) -> np.ndarray:
        if float32_cast:
            x_in = np.asarray(x, dtype=np.float32)
        else:
            x_in = x
        p_rf = rf_model.predict_proba(x_in)
        p_xgb = xgb_model.predict_proba(x_in)
        probs = 0.60 * p_rf + 0.40 * p_xgb
        return np.argmax(probs, axis=1)

    # 4. Execute latency, throughput, and memory benchmark
    model_files = [Path(rf_path), Path(xgb_path)]
    bench_res = benchmark(
        predict_fn=predict_fn,
        X=X_2k,
        model_files=model_files,
        initial_rss_mb=initial_rss_mb,
        load_rss_mb=load_rss_mb,
    )

    # 5. Load validation data and evaluate accuracy / macro F1 AFTER memory benchmarks
    X_val_df, y_val_8, _ = load_validation_data()
    X_val_np = X_val_df[selected_features].to_numpy(dtype=np.float32)
    val_preds_raw = predict_fn(X_val_np)
    val_preds = np.argmax(val_preds_raw, axis=1) if val_preds_raw.ndim > 1 else val_preds_raw
    val_acc = round(float(accuracy_score(y_val_8, val_preds)), 4)
    val_f1 = round(float(f1_score(y_val_8, val_preds, average="macro", zero_division=0)), 4)

    bench_res["validation_accuracy"] = val_acc
    bench_res["validation_macro_f1"] = val_f1

    record = {
        "variant": variant_name,
        "category": category,
        "description": description,
        "model_files": [str(Path(rf_path).relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        str(Path(xgb_path).relative_to(PROJECT_ROOT)).replace("\\", "/")],
        **bench_res,
    }
    if extra_fields:
        record.update(extra_fields)

    out_p = Path(out_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    log.info(
        f"[DONE] {variant_name}: size={record['size_mb_uncompressed']}MB (comp: {record['size_mb_compressed']}MB), "
        f"lat_p95={record['latency_per_flow_ms']['p95_ms']:.2f}ms, "
        f"load_rss={record['load_rss_mb']:.2f}MB, peak_rss={record['peak_rss_mb']:.2f}MB, "
        f"val_f1={record.get('validation_macro_f1', 0):.4f}"
    )
    return record


def run_baseline_benchmark() -> Dict[str, Any]:
    """Runs baseline footprint and latency benchmark for Phase 3 B2 ensemble."""
    rf_path = MODELS_DIR / "rf_8class_b2.joblib"
    xgb_path = MODELS_DIR / "xgb_8class_b2.joblib"
    out_file = RESULTS_DIR / "baseline.json"

    return run_single_variant_process(
        variant_name="baseline",
        rf_path=str(rf_path),
        xgb_path=str(xgb_path),
        out_json_path=str(out_file),
        category="Reference",
        description="Phase 3 B2 Ensemble (RF 0.60 + XGBoost 0.40, 19 features)",
    )


def main():
    parser = argparse.ArgumentParser(description="Edge Benchmark Module")
    parser.add_argument("--variant", default="baseline", help="Variant name")
    parser.add_argument("--rf-path", default=None, help="Path to RF joblib file")
    parser.add_argument("--xgb-path", default=None, help="Path to XGBoost joblib file")
    parser.add_argument("--out-json", default=None, help="Path to save output JSON")
    parser.add_argument("--category", default="Variant", help="Variant category")
    parser.add_argument("--description", default="", help="Variant description")
    parser.add_argument("--float32", action="store_true", help="Cast input to float32")
    parser.add_argument("--extra-json", default=None, help="Extra JSON key-values to include")
    args = parser.parse_args()

    if args.variant == "baseline" and args.rf_path is None:
        run_baseline_benchmark()
        return

    rf_p = args.rf_path or str(MODELS_DIR / "rf_8class_b2.joblib")
    xgb_p = args.xgb_path or str(MODELS_DIR / "xgb_8class_b2.joblib")
    out_p = args.out_json or str(RESULTS_DIR / f"variant_{args.variant}.json")

    extra = json.loads(args.extra_json) if args.extra_json else None

    run_single_variant_process(
        variant_name=args.variant,
        rf_path=rf_p,
        xgb_path=xgb_p,
        out_json_path=out_p,
        category=args.category,
        description=args.description,
        float32_cast=args.float32,
        extra_fields=extra,
    )


if __name__ == "__main__":
    main()
