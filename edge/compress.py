"""
edge/compress.py
================
Phase 11 Compression Variants Evaluator.
Evaluates V1 (RF tree truncation), V2 (XGBoost boosting rounds),
V3 (RF depth-capped retrain), V4 (float32 cast), and V5 (combined best).
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
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import sys
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODELS_DIR,
    TEST_PARQUET,
    DATA_PROCESSED_DIR,
    SELECTED_FEATURES_PATH,
    RANDOM_SEED,
)
from models.load_validation import load_validation_data
from edge.benchmark import (
    benchmark,
    RESULTS_DIR,
    measure_model_sizes_mb,
)

EDGE_MODELS_DIR = PROJECT_ROOT / "edge" / "models"
EDGE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def truncate_rf(rf_model: RandomForestClassifier, n_trees: int) -> RandomForestClassifier:
    """Returns a deepcopy of RF with only the first n_trees estimators."""
    rf_copy = copy.deepcopy(rf_model)
    rf_copy.estimators_ = rf_copy.estimators_[:n_trees]
    rf_copy.n_estimators = n_trees
    rf_copy.n_jobs = 1
    return rf_copy


def slice_xgb(xgb_model: Any, n_rounds: int) -> Any:
    """Returns a deepcopy of XGBoost classifier with only the first n_rounds trees."""
    xgb_copy = copy.deepcopy(xgb_model)
    xgb_copy._Booster = xgb_copy.get_booster()[:n_rounds]
    xgb_copy.n_estimators = n_rounds
    xgb_copy.set_params(n_jobs=1)
    return xgb_copy


def get_or_train_rf_depth(
    depth: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    force_retrain: bool = False,
) -> RandomForestClassifier:
    """Loads cached retrained depth-capped RF or trains a new one."""
    cache_path = EDGE_MODELS_DIR / f"rf_v3_depth{depth}.joblib"
    if cache_path.exists() and not force_retrain:
        log.info(f"Loading cached RF (max_depth={depth}) from {cache_path.name}")
        rf_capped = joblib.load(cache_path)
        rf_capped.n_jobs = 1
        return rf_capped

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
    joblib.dump(rf_capped, cache_path, compress=0)
    log.info(f"Saved retrained RF to {cache_path.name}")
    return rf_capped


def make_ensemble_predict_fn(rf_model: Any, xgb_model: Any, float32_cast: bool = False) -> Callable[[np.ndarray], np.ndarray]:
    """Returns ensemble prediction function (RF 0.60 + XGBoost 0.40)."""
    def predict_fn(x: np.ndarray) -> np.ndarray:
        if float32_cast:
            x_in = np.asarray(x, dtype=np.float32)
        else:
            x_in = x
        p_rf = rf_model.predict_proba(x_in)
        p_xgb = xgb_model.predict_proba(x_in)
        probs = 0.60 * p_rf + 0.40 * p_xgb
        return np.argmax(probs, axis=1)

    return predict_fn


def run_variant(
    variant_name: str,
    category: str,
    description: str,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    model_objects: List[Any],
    X_2k: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    extra_fields: Dict[str, Any] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """Runs benchmark for a single variant and saves result JSON."""
    out_file = RESULTS_DIR / f"variant_{variant_name}.json"
    if resume and out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            log.info(f"[RESUME] Skipping already evaluated variant '{variant_name}'")
            return data
        except Exception as e:
            log.warning(f"Error reading existing {out_file.name}, re-running: {e}")

    log.info(f"\n---> Benchmarking {variant_name} ({description}) ...")
    res = benchmark(
        predict_fn=predict_fn,
        X=X_2k,
        model_files=model_objects,
        X_val=X_val,
        y_val=y_val,
    )

    variant_record = {
        "variant": variant_name,
        "category": category,
        "description": description,
        **res,
    }
    if extra_fields:
        variant_record.update(extra_fields)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(variant_record, f, indent=2)

    log.info(
        f"[OK] {variant_name}: size={variant_record['size_mb_uncompressed']}MB (comp: {variant_record['size_mb_compressed']}MB), "
        f"lat_p95={variant_record['latency_per_flow_ms']['p95_ms']:.2f}ms, "
        f"thru={variant_record['throughput_fps']['median_fps']:.1f}fps, "
        f"val_acc={variant_record['validation_accuracy']:.4f}, "
        f"val_f1={variant_record['validation_macro_f1']:.4f}"
    )
    return variant_record


def main():
    parser = argparse.ArgumentParser(description="Evaluate Edge Compression Variants")
    parser.add_argument("--resume", action="store_true", help="Resume from previously completed variants")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 11: EDGE COMPRESSION VARIANTS EVALUATION")
    print(f"Resume mode: {args.resume}")
    print("=" * 70)

    # 1. Load test inputs (2,000 real scaled test rows, seed 42)
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_features = json.load(f)

    df_test = pd.read_parquet(TEST_PARQUET)
    df_2k = df_test[selected_features].sample(n=2000, random_state=RANDOM_SEED).reset_index(drop=True)
    X_2k = df_2k.to_numpy(dtype=np.float32)

    # 2. Load validation data
    X_val_df, y_val_8, _ = load_validation_data()
    X_val = X_val_df[selected_features].to_numpy(dtype=np.float32)

    # 3. Load baseline models
    rf_base_path = MODELS_DIR / "rf_8class_b2.joblib"
    xgb_base_path = MODELS_DIR / "xgb_8class_b2.joblib"
    rf_base = joblib.load(rf_base_path)
    xgb_base = joblib.load(xgb_base_path)
    rf_base.n_jobs = 1
    xgb_base.set_params(n_jobs=1)

    # 4. Load training data for RF retrain (V3)
    df_tr = pd.read_parquet(DATA_PROCESSED_DIR / "train_unsmoted_19.parquet")
    X_train = df_tr[selected_features].to_numpy(dtype=np.float32)
    y_train_8 = df_tr["label_8"].to_numpy(dtype=np.int64)

    all_variant_results: Dict[str, Dict[str, Any]] = {}
    rf_variants_pool: Dict[str, Tuple[Any, float]] = {}
    xgb_variants_pool: Dict[str, Tuple[Any, float]] = {}

    # -------------------------------------------------------------
    # V1: RF Tree Truncation (100, 50, 25 trees) + baseline XGBoost
    # -------------------------------------------------------------
    for n_trees in [100, 50, 25]:
        vname = f"v1_rf_{n_trees}"
        rf_trunc = truncate_rf(rf_base, n_trees)
        pred_fn = make_ensemble_predict_fn(rf_trunc, xgb_base)
        res = run_variant(
            variant_name=vname,
            category="V1",
            description=f"RF truncated to {n_trees} trees + Baseline XGBoost",
            predict_fn=pred_fn,
            model_objects=[rf_trunc, xgb_base],
            X_2k=X_2k,
            X_val=X_val,
            y_val=y_val_8,
            extra_fields={"rf_trees": n_trees, "xgb_rounds": 250},
            resume=args.resume,
        )
        all_variant_results[vname] = res
        rf_variants_pool[vname] = (rf_trunc, res["validation_macro_f1"])

    # -------------------------------------------------------------
    # V2: XGBoost Boosting Rounds (125, 60, 30 rounds) + baseline RF
    # -------------------------------------------------------------
    for n_rounds in [125, 60, 30]:
        vname = f"v2_xgb_{n_rounds}"
        xgb_sliced = slice_xgb(xgb_base, n_rounds)
        pred_fn = make_ensemble_predict_fn(rf_base, xgb_sliced)
        res = run_variant(
            variant_name=vname,
            category="V2",
            description=f"Baseline RF + XGBoost {n_rounds} boosting rounds",
            predict_fn=pred_fn,
            model_objects=[rf_base, xgb_sliced],
            X_2k=X_2k,
            X_val=X_val,
            y_val=y_val_8,
            extra_fields={"rf_trees": 200, "xgb_rounds": n_rounds},
            resume=args.resume,
        )
        all_variant_results[vname] = res
        xgb_variants_pool[vname] = (xgb_sliced, res["validation_macro_f1"])

    # -------------------------------------------------------------
    # V3: RF Retrained with Depth Capping (max_depth 15, 10) + baseline XGBoost
    # -------------------------------------------------------------
    for depth in [15, 10]:
        vname = f"v3_rf_depth{depth}"
        rf_capped = get_or_train_rf_depth(depth, X_train, y_train_8)
        pred_fn = make_ensemble_predict_fn(rf_capped, xgb_base)
        res = run_variant(
            variant_name=vname,
            category="V3",
            description=f"RF retrained max_depth={depth} (200 trees) + Baseline XGBoost",
            predict_fn=pred_fn,
            model_objects=[rf_capped, xgb_base],
            X_2k=X_2k,
            X_val=X_val,
            y_val=y_val_8,
            extra_fields={"rf_trees": 200, "rf_max_depth": depth, "xgb_rounds": 250},
            resume=args.resume,
        )
        all_variant_results[vname] = res
        rf_variants_pool[vname] = (rf_capped, res["validation_macro_f1"])

    # -------------------------------------------------------------
    # V4: float32 Input Cast
    # -------------------------------------------------------------
    vname = "v4_float32"
    pred_fn_f32 = make_ensemble_predict_fn(rf_base, xgb_base, float32_cast=True)
    # Check if predictions change relative to float64
    preds_val_f64 = make_ensemble_predict_fn(rf_base, xgb_base, float32_cast=False)(X_val)
    preds_val_f32 = pred_fn_f32(X_val)
    pred_diffs = int(np.sum(preds_val_f64 != preds_val_f32))

    res_v4 = run_variant(
        variant_name=vname,
        category="V4",
        description="Baseline Ensemble with float32 input cast",
        predict_fn=pred_fn_f32,
        model_objects=[rf_base, xgb_base],
        X_2k=X_2k,
        X_val=X_val,
        y_val=y_val_8,
        extra_fields={
            "prediction_differences_count": pred_diffs,
            "prediction_differences_total": len(X_val),
            "prediction_difference_rate": round(pred_diffs / len(X_val), 6),
        },
        resume=args.resume,
    )
    all_variant_results[vname] = res_v4

    # -------------------------------------------------------------
    # V5: Combined Best (Best RF from V1/V3 + Best XGBoost from V2)
    # -------------------------------------------------------------
    # Select best RF variant strictly on validation macro F1
    best_rf_name = max(rf_variants_pool.keys(), key=lambda k: rf_variants_pool[k][1])
    best_rf_model = rf_variants_pool[best_rf_name][0]
    best_rf_f1 = rf_variants_pool[best_rf_name][1]

    # Select best XGBoost variant strictly on validation macro F1
    best_xgb_name = max(xgb_variants_pool.keys(), key=lambda k: xgb_variants_pool[k][1])
    best_xgb_model = xgb_variants_pool[best_xgb_name][0]
    best_xgb_f1 = xgb_variants_pool[best_xgb_name][1]

    log.info(f"Selecting best components on validation macro F1:")
    log.info(f"  Best RF:  {best_rf_name} (val macro F1: {best_rf_f1:.4f})")
    log.info(f"  Best XGB: {best_xgb_name} (val macro F1: {best_xgb_f1:.4f})")

    vname = "v5_combined"
    pred_fn_v5 = make_ensemble_predict_fn(best_rf_model, best_xgb_model)
    res_v5 = run_variant(
        variant_name=vname,
        category="V5",
        description=f"Combined Best ({best_rf_name} + {best_xgb_name})",
        predict_fn=pred_fn_v5,
        model_objects=[best_rf_model, best_xgb_model],
        X_2k=X_2k,
        X_val=X_val,
        y_val=y_val_8,
        extra_fields={
            "selected_rf_variant": best_rf_name,
            "selected_xgb_variant": best_xgb_name,
            "selected_rf_val_macro_f1": best_rf_f1,
            "selected_xgb_val_macro_f1": best_xgb_f1,
        },
        resume=args.resume,
    )
    all_variant_results[vname] = res_v5

    # Also save a summary of all variants
    variants_summary_path = RESULTS_DIR / "variants_summary.json"
    with open(variants_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_variant_results, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPRESSION VARIANTS EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<18} {'Size (MB)':<12} {'Comp (MB)':<12} {'p95 Lat (ms)':<14} {'Thru (fps)':<12} {'Val Acc':<10} {'Val F1':<10}")
    print("-" * 88)
    for name, v in all_variant_results.items():
        print(
            f"{name:<18} "
            f"{v['size_mb_uncompressed']:<12.2f} "
            f"{v['size_mb_compressed']:<12.2f} "
            f"{v['latency_per_flow_ms']['p95_ms']:<14.2f} "
            f"{v['throughput_fps']['median_fps']:<12.1f} "
            f"{v['validation_accuracy']:<10.4f} "
            f"{v['validation_macro_f1']:<10.4f}"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()
