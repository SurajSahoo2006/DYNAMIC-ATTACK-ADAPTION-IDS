"""
experiments/eval_utils.py
=========================
Standardized evaluation utilities for rare-class experiments.
Tracks:
  - Overall accuracy
  - Overall macro F1
  - Precision, Recall, F1, and Support for:
      - Backdoor_Malware (idx 0)
      - Recon-PingSweep  (idx 28)
      - Uploading_Attack (idx 31)
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

RARE_CLASSES = ["Backdoor_Malware", "Recon-PingSweep", "Uploading_Attack"]


def evaluate_34class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    split_name: str = "Validation",
) -> Dict[str, Any]:
    """
    Computes accuracy, macro F1, and precision/recall/F1 for the 3 rare classes.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)

    acc = float(accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    report = classification_report(
        y_t,
        y_p,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    rare_metrics: Dict[str, Dict[str, float]] = {}
    for cname in RARE_CLASSES:
        metrics = report.get(cname, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        rare_metrics[cname] = {
            "precision": round(metrics["precision"] * 100.0, 2),
            "recall": round(metrics["recall"] * 100.0, 2),
            "f1": round(metrics["f1-score"] * 100.0, 2),
            "support": int(metrics["support"]),
        }

    return {
        "split": split_name,
        "accuracy": round(acc * 100.0, 2),
        "macro_f1": round(macro_f1 * 100.0, 2),
        "rare_metrics": rare_metrics,
    }


def print_experiment_result(exp_name: str, res_val: Dict[str, Any], res_test: Dict[str, Any] | None = None):
    print(f"\n{'='*75}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*75}")
    print(f"Validation: Acc = {res_val['accuracy']:.2f}% | Macro F1 = {res_val['macro_f1']:.2f}%")
    for cname in RARE_CLASSES:
        m = res_val["rare_metrics"][cname]
        print(f"  [VAL]  {cname:<18} -> P: {m['precision']:>5.1f}% | R: {m['recall']:>5.1f}% | F1: {m['f1']:>5.1f}% (Supp: {m['support']:>4})")

    if res_test is not None:
        print(f"Test Set:   Acc = {res_test['accuracy']:.2f}% | Macro F1 = {res_test['macro_f1']:.2f}%")
        for cname in RARE_CLASSES:
            m = res_test["rare_metrics"][cname]
            print(f"  [TEST] {cname:<18} -> P: {m['precision']:>5.1f}% | R: {m['recall']:>5.1f}% | F1: {m['f1']:>5.1f}% (Supp: {m['support']:>4})")
