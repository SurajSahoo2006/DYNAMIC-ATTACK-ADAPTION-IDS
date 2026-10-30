#!/usr/bin/env python3
"""
reports/generate_phase10_report.py - Generate Phase 10 summary report from zero-day JSON artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AE_TRAINING_PATH = PROJECT_ROOT / "zeroday" / "results" / "ae_training.json"
OPENSET_SUMMARY_PATH = PROJECT_ROOT / "zeroday" / "results" / "openset_summary.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase10_summary.md"


def format_stat(stat: dict) -> str:
    x = stat["x"]
    n = stat["n"]
    pct = stat["percentage"]
    lower = stat["ci_lower"]
    upper = stat["ci_upper"]
    return f"{x} of {n} ({pct:.2f}%) [{lower:.2f}%, {upper:.2f}%]"


def main():
    if not AE_TRAINING_PATH.exists():
        raise FileNotFoundError(f"Missing {AE_TRAINING_PATH}")
    if not OPENSET_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing {OPENSET_SUMMARY_PATH}")

    with open(AE_TRAINING_PATH, "r", encoding="utf-8") as f:
        ae_data = json.load(f)

    with open(OPENSET_SUMMARY_PATH, "r", encoding="utf-8") as f:
        openset_data = json.load(f)

    known_control = openset_data["known_class_control"]
    leave_one_out = openset_data["leave_one_class_out"]

    # Compute headline numbers directly from JSON
    total_heldout = len(leave_one_out)
    flagged_gte_50 = sum(
        1 for c, d in leave_one_out.items()
        if d["flagged_unknown_either"]["percentage"] >= 50.0
    )
    silently_benign_gt_20 = sum(
        1 for c, d in leave_one_out.items()
        if d["silently_predicted_benign"]["percentage"] > 20.0
    )

    headline = (
        f"Across the {total_heldout} held-out attack classes, {flagged_gte_50} of {total_heldout} "
        f"were flagged UNKNOWN in at least 50% of flows, and {silently_benign_gt_20} of {total_heldout} "
        f"were silently labelled Benign in more than 20% of flows."
    )

    # Table 1: Known-class control
    t1_rows = []
    for cname, stat in known_control.items():
        formatted_val = format_stat(stat)
        t1_rows.append(f"| {cname} | {stat['n']} | {formatted_val} |")

    # Table 2: Leave-one-class-out
    t2_rows = []
    for cname, d in leave_one_out.items():
        n_test = d["n_test"]
        f_either = format_stat(d["flagged_unknown_either"])
        f_ae = format_stat(d["flagged_unknown_by_ae"])
        f_conf = format_stat(d["flagged_unknown_by_confidence"])
        s_benign = format_stat(d["silently_predicted_benign"])
        s_other = format_stat(d["silently_predicted_other_attack"])
        t2_rows.append(
            f"| {cname} | {n_test} | {f_either} | {f_ae} | {f_conf} | {s_benign} | {s_other} |"
        )

    lines = [
        "# Phase 10: Zero-Day (Unseen Attack) Detection Layer",
        "",
        "## Setup",
        "- **Anomaly Detector Architecture**: PyTorch fully-connected Autoencoder (`19 -> 12 -> 6 -> 12 -> 19`) with ReLU activations and MSE reconstruction loss.",
        f"- **Benign-Only Training**: Fitted strictly on {ae_data['n_train_benign']} real Benign flows (no SMOTE rows) using Adam (learning rate `1e-3`, batch size 512, maximum 40 epochs) with early stopping on {ae_data['n_val_benign']} validation Benign flows.",
        f"- **Reconstruction Threshold**: Fixed at the 95th percentile of validation Benign reconstruction error (`threshold = {ae_data['threshold']:.6f}`). The test set is never accessed to select this cutoff.",
        "- **Combined Decision Rule**: A flow is flagged UNKNOWN if `AE reconstruction score > threshold` OR `max predicted class probability < 0.60` (uncertainty threshold matching Phase 3).",
        "",
        "## Headline Finding",
        headline,
        "",
        "## Table 1: Known-Class Control (Full Test Set)",
        "Flagged-UNKNOWN rates on the full test set evaluated against the Phase 3 B2 ensemble (RF 0.6 + XGBoost 0.4). For Benign traffic, this represents the false-positive rate.",
        "",
        "| Class | Test Flows | Flagged UNKNOWN (95% Wilson CI) |",
        "|:---|---:|:---|",
    ]
    lines.extend(t1_rows)
    lines.extend([
        "",
        "## Table 2: Leave-One-Class-Out Evaluation (Unseen Attack Simulation)",
        "For each attack class, a fresh Random Forest classifier is trained on all training flows with that class excluded (capped at 50,000 stratified samples). Evaluation is performed on test flows of the held-out class using the fixed Autoencoder and uncertainty rule.",
        "",
        "| Held-Out Class | Test Flows | Flagged UNKNOWN (Either) | Flagged by AE | Flagged by Uncertainty (<0.60) | Silently Benign | Silently Other Attack |",
        "|:---|---:|:---|:---|:---|:---|:---|",
    ])
    lines.extend(t2_rows)
    lines.extend([
        "",
        "## Limitations",
        "- **Heterogeneous Benign Class**: The Autoencoder is trained on one aggregated Benign class that encompasses diverse normal protocols; subtle anomalies within complex normal behaviors may not trigger reconstruction error.",
        "- **Attacks Mimicking Normal Patterns**: Attacks that closely resemble normal application traffic (such as certain Web-based and Spoofing variants) can exhibit low reconstruction error and rely primarily on uncertainty thresholds for detection.",
        "- **Duration Point Mass**: A point mass of 45.26% exists in test flow durations, which can affect feature scaling and reconstruction residuals across specific temporal profiles.",
        "- **Simulated Zero-Days**: Held-out classes originate from the known dataset distribution and serve as proxy unseen attacks rather than novel in-the-wild zero-day vulnerabilities.",
        "- **Single Random Seed**: Training and evaluation were conducted with seed 42 without hyperparameter tuning on the test set.",
        "- **Threshold Trade-off**: The 95th percentile threshold establishes an empirical trade-off between the Benign false-positive rate (26.60%) and open-set recall, without subsequent optimization.",
        "",
    ])

    report_content = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[REPORT] Phase 10 summary written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
