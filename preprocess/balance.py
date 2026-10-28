"""
preprocess/balance.py
=====================
Phase 2 - Step 5: Class Balancing with SMOTE
Applied to TRAIN set ONLY.  Test set is never resampled.

Why SMOTE instead of random oversampling?
  Random duplication of minority rows risks overfitting to those exact
  points.  SMOTE generates NEW synthetic points by interpolating between
  existing minority-class neighbours, producing a more generalisable
  training distribution.

Dynamic Capped Strategy & Safety Cap
-------------------------------------
- Target count: size of the LARGEST class in the training split (never above
  SAMPLE_PER_CLASS). No class is oversampled beyond that.
- Safety cap: no class may have more than 10x its real training rows created
  as synthetic. If a class would exceed that, it stops at 10x and a warning
  is logged naming the class.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RANDOM_SEED
import config

log = logging.getLogger(__name__)


def balance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_count: int | None = None,
    max_synthetic_multiplier: int = 10,
    label_encoder: Any = None,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Oversample minority classes in (X_train, y_train) using SMOTE.

    Parameters
    ----------
    X_train                  : scaled feature DataFrame
    y_train                  : integer label Series (34-class or 8-class)
    target_count             : desired count per class. If None, uses max class count
                               in y_train (bounded by config.SAMPLE_PER_CLASS).
    max_synthetic_multiplier : maximum synthetic rows = multiplier * real rows (default 10)
    label_encoder            : optional LabelEncoder to resolve class names for warnings & logs
    seed                     : random seed

    Returns
    -------
    (X_resampled, y_resampled, balance_stats_dict)
    """
    counts = y_train.value_counts().sort_index()

    # Determine target count dynamically
    max_train_count = int(counts.max())
    if target_count is None:
        if config.SAMPLE_PER_CLASS is not None:
            target_count = min(max_train_count, config.SAMPLE_PER_CLASS)
        else:
            target_count = max_train_count
    else:
        if config.SAMPLE_PER_CLASS is not None:
            target_count = min(target_count, config.SAMPLE_PER_CLASS)

    print(f"\n[BALANCE] SMOTE target count: {target_count:,} rows (bounded by largest train class / SAMPLE_PER_CLASS)")
    print(f"[BALANCE] Safety cap: max {max_synthetic_multiplier}x synthetic rows per class")

    # Build sampling strategy with 10x safety cap
    strategy: dict[int, int] = {}
    capped_classes: list[dict[str, Any]] = []

    for cls, cnt in counts.items():
        cls_int = int(cls)
        if cnt < target_count:
            desired_synthetic = target_count - cnt
            max_synthetic = max_synthetic_multiplier * cnt
            if desired_synthetic > max_synthetic:
                final_target = cnt + max_synthetic
                cls_name = (
                    label_encoder.classes_[cls_int]
                    if label_encoder is not None and cls_int < len(label_encoder.classes_)
                    else str(cls_int)
                )
                capped_classes.append({
                    "class_id": cls_int,
                    "class_name": cls_name,
                    "real_count": cnt,
                    "synthetic_count": max_synthetic,
                    "final_target": final_target,
                    "requested_target": target_count,
                })
                strategy[cls_int] = final_target
            else:
                strategy[cls_int] = target_count

    # Print warnings for capped classes
    if capped_classes:
        print(f"\n[BALANCE] WARNING: {len(capped_classes)} minority class(es) hit the {max_synthetic_multiplier}x synthetic cap:")
        for cap in capped_classes:
            print(
                f"  - Class '{cap['class_name']}' ({cap['class_id']}): "
                f"{cap['real_count']:,} real rows --> capped at +{cap['synthetic_count']:,} synthetic "
                f"({cap['final_target']:,} total; desired was {cap['requested_target']:,})"
            )

    if not strategy:
        print("[BALANCE] All classes already meet or exceed target. No oversampling needed.")
        X_res = X_train.copy()
        y_res = y_train.copy()
        method_used = "None"
    else:
        print(f"[BALANCE] Classes being oversampled: {len(strategy)} of {len(counts)}")
        try:
            smote = SMOTE(
                sampling_strategy=strategy,
                random_state=seed,
                k_neighbors=min(5, max(1, counts.min() - 1)),
            )
            X_res, y_res = smote.fit_resample(X_train, y_train)
            method_used = "SMOTE"
        except Exception as exc:
            log.warning("SMOTE failed (%s). Falling back to RandomOverSampler.", exc)
            print(f"[BALANCE] WARNING  SMOTE failed: {exc}")
            print("[BALANCE] Falling back to RandomOverSampler ...")
            ros = RandomOverSampler(sampling_strategy=strategy, random_state=seed)
            X_res, y_res = ros.fit_resample(X_train, y_train)
            method_used = "RandomOverSampler"

    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    # Compute real vs synthetic breakdown table
    counts_after = y_res.value_counts()
    breakdown: list[dict[str, Any]] = []

    for cls in sorted(counts.index):
        cls_int = int(cls)
        real_cnt = int(counts[cls_int])
        tot_cnt = int(counts_after.get(cls_int, real_cnt))
        syn_cnt = tot_cnt - real_cnt
        syn_pct = (syn_cnt / tot_cnt * 100.0) if tot_cnt > 0 else 0.0
        cls_name = (
            label_encoder.classes_[cls_int]
            if label_encoder is not None and cls_int < len(label_encoder.classes_)
            else str(cls_int)
        )
        breakdown.append({
            "class_id": cls_int,
            "class_name": cls_name,
            "real_count": real_cnt,
            "synthetic_count": syn_cnt,
            "total_count": tot_cnt,
            "synthetic_pct": syn_pct,
            "capped": any(c["class_id"] == cls_int for c in capped_classes),
        })

    total_real = sum(b["real_count"] for b in breakdown)
    total_syn = sum(b["synthetic_count"] for b in breakdown)
    total_all = len(X_res)
    overall_syn_pct = (total_syn / total_all * 100.0) if total_all > 0 else 0.0

    print(f"\n[BALANCE] Real vs Synthetic breakdown after {method_used}:")
    print(f"  {'Class':<32} {'Real':>8} {'Synthetic':>10} {'Total':>10} {'% Syn':>8}")
    print(f"  {'-'*32} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for b in breakdown:
        cap_flag = " *" if b["capped"] else ""
        print(f"  {b['class_name']:<32} {b['real_count']:>8,} {b['synthetic_count']:>10,} {b['total_count']:>10,} {b['synthetic_pct']:>7.1f}%{cap_flag}")
    print(f"  {'-'*32} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    print(f"  {'OVERALL TOTAL':<32} {total_real:>8,} {total_syn:>10,} {total_all:>10,} {overall_syn_pct:>7.1f}%")
    print(f"[BALANCE] Synthetic rows generated: {total_syn:,} ({overall_syn_pct:.1f}% of balanced train set)")
    print("[BALANCE] [OK]  Test set was NOT resampled.")

    stats = {
        "method": method_used,
        "target_count": target_count,
        "max_synthetic_multiplier": max_synthetic_multiplier,
        "total_real": total_real,
        "total_synthetic": total_syn,
        "total_balanced": total_all,
        "overall_synthetic_pct": overall_syn_pct,
        "breakdown": breakdown,
        "capped_classes": capped_classes,
    }

    return X_res, y_res, stats
