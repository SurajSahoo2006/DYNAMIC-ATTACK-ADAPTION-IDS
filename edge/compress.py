"""
edge/compress.py
================
Phase 11 Compression Variants Evaluator.
Evaluates V1 (RF tree truncation), V2 (Real XGBoost booster truncation & reload),
V3 (RF depth-capped retrain), V4 (float32 cast), and V5 (combined best).
Runs each variant in an isolated fresh subprocess for accurate resident memory (RSS).
Writes individual edge/results/variant_<name>.json files and supports --resume.
"""

from __future__ import annotations

import os
# Single-thread execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import copy
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODELS_DIR,
    DATA_PROCESSED_DIR,
    SELECTED_FEATURES_PATH,
    RANDOM_SEED,
)
from edge.benchmark import (
    RESULTS_DIR,
    EDGE_MODELS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def create_and_save_truncated_rf(rf_base: RandomForestClassifier, n_trees: int) -> Path:
    """Creates a truncated RF model and saves it to edge/models/."""
    save_path = EDGE_MODELS_DIR / f"rf_v1_{n_trees}.joblib"
    rf_copy = copy.deepcopy(rf_base)
    rf_copy.estimators_ = rf_copy.estimators_[:n_trees]
    rf_copy.n_estimators = n_trees
    rf_copy.n_jobs = 1
    joblib.dump(rf_copy, save_path, compress=0)
    log.info(f"Saved truncated RF ({n_trees} trees) to {save_path.name}")
    return save_path


def create_and_save_truncated_xgb(xgb_base: Any, n_rounds: int) -> Path:
    """
    FIX 2: Creates a truncated booster (booster[:rounds]), saves it to a new file,
    so that file size and latency can be measured directly from the reloaded model.
    """
    save_path = EDGE_MODELS_DIR / f"xgb_v2_{n_rounds}.joblib"
    xgb_copy = copy.deepcopy(xgb_base)
    xgb_copy._Booster = xgb_base.get_booster()[:n_rounds]
    xgb_copy.n_estimators = n_rounds
    xgb_copy.set_params(n_jobs=1)
    joblib.dump(xgb_copy, save_path, compress=0)
    log.info(f"Saved truncated XGBoost ({n_rounds} rounds) to {save_path.name}")
    return save_path


def get_or_train_rf_depth(
    depth: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    force_retrain: bool = False,
) -> Path:
    """Loads cached retrained depth-capped RF or trains a new one, returning file path."""
    save_path = EDGE_MODELS_DIR / f"rf_v3_depth{depth}.joblib"
    if save_path.exists() and not force_retrain:
        log.info(f"Found cached RF (max_depth={depth}) at {save_path.name}")
        return save_path

    log.info(f"Retraining RF (200 trees, max_depth={depth}, balanced, seed={RANDOM_SEED}) on unsmoted data...")
    rf_capped = RandomForestClassifier(
        n_estimators=200,
        max_depth=depth,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=1,
        random_state=RANDOM_SEED,
    )
    rf_capped.fit(X_train, y_train)
    joblib.dump(rf_capped, save_path, compress=0)
    log.info(f"Saved retrained RF to {save_path.name}")
    return save_path


def run_variant_subprocess(
    variant_name: str,
    rf_path: Path,
    xgb_path: Path,
    category: str,
    description: str,
    float32_cast: bool = False,
    extra_fields: Dict[str, Any] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    FIX 1: Launches each variant in an isolated fresh Python subprocess to prevent
    shared heap memory and obtain true resident memory (psutil RSS) measurements.
    """
    out_file = RESULTS_DIR / f"variant_{variant_name}.json"
    if resume and out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Ensure valid non-zero memory
            if data.get("load_rss_mb", 0) > 0 and data.get("peak_rss_mb", 0) > 0:
                log.info(f"[RESUME] Skipping already evaluated variant '{variant_name}'")
                return data
            else:
                log.warning(f"Existing {out_file.name} has zero memory values; re-running.")
        except Exception as e:
            log.warning(f"Error reading existing {out_file.name}, re-running: {e}")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "edge" / "benchmark.py"),
        "--variant", variant_name,
        "--rf-path", str(rf_path),
        "--xgb-path", str(xgb_path),
        "--out-json", str(out_file),
        "--category", category,
        "--description", description,
    ]
    if float32_cast:
        cmd.append("--float32")
    if extra_fields:
        cmd.extend(["--extra-json", json.dumps(extra_fields)])

    log.info(f"\n---> Launching fresh subprocess for variant '{variant_name}' ...")
    ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if ret.returncode != 0:
        log.error(f"ERROR: Subprocess for variant '{variant_name}' failed with returncode {ret.returncode}!")
        raise RuntimeError(f"Variant '{variant_name}' benchmarking failed.")

    with open(out_file, "r", encoding="utf-8") as f:
        record = json.load(f)

    if record.get("load_rss_mb", 0) <= 0 or record.get("peak_rss_mb", 0) <= 0:
        log.error(f"ERROR: Variant '{variant_name}' produced zero memory: load_rss={record.get('load_rss_mb')}, peak_rss={record.get('peak_rss_mb')}")
        raise ValueError(f"Variant '{variant_name}' produced zero resident memory!")

    return record


def main():
    parser = argparse.ArgumentParser(description="Evaluate Edge Compression Variants")
    parser.add_argument("--resume", action="store_true", help="Resume from previously completed variants")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 11: EDGE COMPRESSION VARIANTS EVALUATION (FIX 1 & FIX 2)")
    print(f"Resume mode: {args.resume}")
    print("=" * 70)

    # 1. Load baseline models
    rf_base_path = MODELS_DIR / "rf_8class_b2.joblib"
    xgb_base_path = MODELS_DIR / "xgb_8class_b2.joblib"
    rf_base = joblib.load(rf_base_path)
    xgb_base = joblib.load(xgb_base_path)
    rf_base.n_jobs = 1
    xgb_base.set_params(n_jobs=1)

    # 2. Load training data for RF retrain (V3)
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_features = json.load(f)
    df_tr = pd.read_parquet(DATA_PROCESSED_DIR / "train_unsmoted_19.parquet")
    X_train = df_tr[selected_features].to_numpy(dtype=np.float32)
    y_train_8 = df_tr["label_8"].to_numpy(dtype=np.int64)

    all_variant_results: Dict[str, Dict[str, Any]] = {}
    rf_pool: Dict[str, Tuple[Path, float]] = {}
    xgb_pool: Dict[str, Tuple[Path, float]] = {}

    # -------------------------------------------------------------
    # V1: RF Tree Truncation (100, 50, 25 trees) + baseline XGBoost
    # -------------------------------------------------------------
    for n_trees in [100, 50, 25]:
        vname = f"v1_rf_{n_trees}"
        rf_path = create_and_save_truncated_rf(rf_base, n_trees)
        res = run_variant_subprocess(
            variant_name=vname,
            rf_path=rf_path,
            xgb_path=xgb_base_path,
            category="V1",
            description=f"RF truncated to {n_trees} trees + Baseline XGBoost",
            extra_fields={"rf_trees": n_trees, "xgb_rounds": 250},
            resume=args.resume,
        )
        all_variant_results[vname] = res
        rf_pool[vname] = (rf_path, res["validation_macro_f1"])

    # -------------------------------------------------------------
    # V2: Real XGBoost Booster Truncation (125, 60, 30 rounds) + baseline RF
    # -------------------------------------------------------------
    for n_rounds in [125, 60, 30]:
        vname = f"v2_xgb_{n_rounds}"
        xgb_path = create_and_save_truncated_xgb(xgb_base, n_rounds)
        res = run_variant_subprocess(
            variant_name=vname,
            rf_path=rf_base_path,
            xgb_path=xgb_path,
            category="V2",
            description=f"Baseline RF + Real truncated XGBoost ({n_rounds} rounds reloaded)",
            extra_fields={"rf_trees": 200, "xgb_rounds": n_rounds},
            resume=args.resume,
        )
        all_variant_results[vname] = res
        xgb_pool[vname] = (xgb_path, res["validation_macro_f1"])

    # -------------------------------------------------------------
    # V3: RF Retrained with Depth Capping (max_depth 15, 10) + baseline XGBoost
    # -------------------------------------------------------------
    for depth in [15, 10]:
        vname = f"v3_rf_depth{depth}"
        rf_path = get_or_train_rf_depth(depth, X_train, y_train_8)
        res = run_variant_subprocess(
            variant_name=vname,
            rf_path=rf_path,
            xgb_path=xgb_base_path,
            category="V3",
            description=f"RF retrained max_depth={depth} (200 trees) + Baseline XGBoost",
            extra_fields={"rf_trees": 200, "rf_max_depth": depth, "xgb_rounds": 250},
            resume=args.resume,
        )
        all_variant_results[vname] = res
        rf_pool[vname] = (rf_path, res["validation_macro_f1"])

    # -------------------------------------------------------------
    # V4: float32 Input Cast
    # -------------------------------------------------------------
    vname = "v4_float32"
    # Measure prediction differences on validation set
    from models.load_validation import load_validation_data
    X_val_df, _, _ = load_validation_data()
    X_val_np = X_val_df[selected_features].to_numpy()
    p_f64 = 0.60 * rf_base.predict_proba(X_val_np) + 0.40 * xgb_base.predict_proba(X_val_np)
    X_val_f32 = X_val_np.astype(np.float32)
    p_f32 = 0.60 * rf_base.predict_proba(X_val_f32) + 0.40 * xgb_base.predict_proba(X_val_f32)
    pred_diffs = int(np.sum(np.argmax(p_f64, axis=1) != np.argmax(p_f32, axis=1)))

    res_v4 = run_variant_subprocess(
        variant_name=vname,
        rf_path=rf_base_path,
        xgb_path=xgb_base_path,
        category="V4",
        description="Baseline Ensemble with float32 input cast",
        float32_cast=True,
        extra_fields={
            "prediction_differences_count": pred_diffs,
            "prediction_differences_total": len(X_val_df),
            "prediction_difference_rate": round(pred_diffs / len(X_val_df), 6),
        },
        resume=args.resume,
    )
    all_variant_results[vname] = res_v4

    # -------------------------------------------------------------
    # V5: Combined Best (Best RF from V1/V3 + Best XGBoost from V2)
    # -------------------------------------------------------------
    best_rf_name = max(rf_pool.keys(), key=lambda k: rf_pool[k][1])
    best_rf_path, best_rf_f1 = rf_pool[best_rf_name]

    best_xgb_name = max(xgb_pool.keys(), key=lambda k: xgb_pool[k][1])
    best_xgb_path, best_xgb_f1 = xgb_pool[best_xgb_name]

    log.info(f"Selecting best components on validation macro F1:")
    log.info(f"  Best RF:  {best_rf_name} (val macro F1: {best_rf_f1:.4f}) -> {best_rf_path.name}")
    log.info(f"  Best XGB: {best_xgb_name} (val macro F1: {best_xgb_f1:.4f}) -> {best_xgb_path.name}")

    vname = "v5_combined"
    res_v5 = run_variant_subprocess(
        variant_name=vname,
        rf_path=best_rf_path,
        xgb_path=best_xgb_path,
        category="V5",
        description=f"Combined Best ({best_rf_name} + {best_xgb_name})",
        extra_fields={
            "selected_rf_variant": best_rf_name,
            "selected_xgb_variant": best_xgb_name,
            "selected_rf_val_macro_f1": best_rf_f1,
            "selected_xgb_val_macro_f1": best_xgb_f1,
            "selected_rf_model_file": str(best_rf_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "selected_xgb_model_file": str(best_xgb_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        resume=args.resume,
    )
    all_variant_results[vname] = res_v5

    # Summary JSON
    variants_summary_path = RESULTS_DIR / "variants_summary.json"
    with open(variants_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_variant_results, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPRESSION VARIANTS EVALUATION SUMMARY (FIX 1 & FIX 2)")
    print("=" * 70)
    print(f"{'Variant':<16} {'Size(MB)':<10} {'Comp(MB)':<10} {'p95(ms)':<10} {'Thru(fps)':<12} {'LoadRSS(MB)':<12} {'PeakRSS(MB)':<12} {'Val F1':<8}")
    print("-" * 88)
    for name, v in all_variant_results.items():
        print(
            f"{name:<16} "
            f"{v['size_mb_uncompressed']:<10.2f} "
            f"{v['size_mb_compressed']:<10.2f} "
            f"{v['latency_per_flow_ms']['p95_ms']:<10.2f} "
            f"{v['throughput_fps']['median_fps']:<12.1f} "
            f"{v['load_rss_mb']:<12.2f} "
            f"{v['peak_rss_mb']:<12.2f} "
            f"{v['validation_macro_f1']:<8.4f}"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()
