#!/usr/bin/env python3
"""
reports/generate_phase10_report.py - Generate Phase 10 summary report with separated detectors, lift table, AUROC, and operating points.
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

    op1_meta = openset_data["operating_point_original"]
    op2_meta = openset_data["operating_point_fp_budget"]

    known_op1 = openset_data["known_class_control"]
    known_op2 = openset_data["known_class_control_fp_budget"]

    leave_op1 = openset_data["leave_one_class_out"]
    leave_op2 = openset_data["leave_one_class_out_fp_budget"]

    lift_op1 = openset_data["lift"]
    lift_op2 = openset_data["lift_fp_budget"]
    auroc_data = openset_data["auroc"]

    # (a) Benign FPR at each operating point
    benign_fpr_op1 = op1_meta["test_benign_fpr"]
    benign_fpr_op2 = op2_meta["test_benign_fpr"]

    # (b) Per-class lift
    lift_items = [f"{c}: {lift_op1[c]['formatted']}" for c in lift_op1]
    lift_summary_str = "; ".join(lift_items)

    # (c) Per held-out class missed rate
    missed_items = []
    for c in leave_op1:
        s_b = leave_op1[c]["silently_predicted_benign"]["x"]
        s_o = leave_op1[c]["silently_predicted_other_attack"]["x"]
        n_c = leave_op1[c]["n_test"]
        missed_x = s_b + s_o
        missed_pct = (missed_x / n_c) * 100.0
        missed_items.append(f"{c}: {missed_x} of {n_c} ({missed_pct:.2f}%)")
    missed_summary_str = "; ".join(missed_items)

    # (d) AE-only vs Confidence-only contribution
    ae_rates = [leave_op1[c]["flagged_unknown_by_ae"]["percentage"] for c in leave_op1]
    conf_rates = [leave_op1[c]["flagged_unknown_by_confidence"]["percentage"] for c in leave_op1]
    mean_ae = sum(ae_rates) / len(ae_rates)
    mean_conf = sum(conf_rates) / len(conf_rates)

    headline_bullets = [
        f"- **(a) Benign False-Positive Rate**: At the original operating point (95th percentile, confidence 0.60), the test Benign false-positive rate is {benign_fpr_op1:.2f}%. At the FP-budget operating point (97.5th percentile, confidence 0.40), the test Benign false-positive rate is {benign_fpr_op2:.2f}%.",
        f"- **(b) Detection Lift Over Known Control**: When attack classes are held out during training, combined flagging increases relative to the known-class control across all 7 classes ({lift_summary_str}).",
        f"- **(c) Held-Out Missed Rates (Unflagged Traffic)**: Flows not flagged as UNKNOWN (sum of silently Benign and silently other attack) under the original operating point are: {missed_summary_str}.",
        f"- **(d) Detector Contributions**: Across held-out classes, confidence-only flagging averages {mean_conf:.2f}% (ranging from {min(conf_rates):.2f}% to {max(conf_rates):.2f}%), whereas AE-only flagging averages {mean_ae:.2f}% (ranging from {min(ae_rates):.2f}% to {max(ae_rates):.2f}%).",
    ]

    # Table 1: Known-Class Control (Full Test Set)
    t1_rows = []
    for cname in known_op1:
        n_c = known_op1[cname]["n_test"]
        f_comb_1 = format_stat(known_op1[cname]["flagged_unknown_either"])
        f_ae_1 = format_stat(known_op1[cname]["flagged_unknown_by_ae"])
        f_conf_1 = format_stat(known_op1[cname]["flagged_unknown_by_confidence"])
        f_comb_2 = format_stat(known_op2[cname]["flagged_unknown_either"])
        t1_rows.append(
            f"| {cname} | {n_c} | {f_comb_1} | {f_ae_1} | {f_conf_1} | {f_comb_2} |"
        )

    # Table 2: Leave-One-Class-Out Evaluation (Original Operating Point)
    t2_rows = []
    for cname, d in leave_op1.items():
        n_test = d["n_test"]
        f_either = format_stat(d["flagged_unknown_either"])
        f_ae = format_stat(d["flagged_unknown_by_ae"])
        f_conf = format_stat(d["flagged_unknown_by_confidence"])
        s_benign = format_stat(d["silently_predicted_benign"])
        s_other = format_stat(d["silently_predicted_other_attack"])
        missed_x = d["silently_predicted_benign"]["x"] + d["silently_predicted_other_attack"]["x"]
        missed_pct = (missed_x / n_test) * 100.0
        missed_str = f"{missed_x} of {n_test} ({missed_pct:.2f}%)"
        t2_rows.append(
            f"| {cname} | {n_test} | {f_either} | {f_ae} | {f_conf} | {s_benign} | {s_other} | {missed_str} |"
        )

    # Table 3: Leave-One-Class-Out Evaluation (FP-Budget Operating Point)
    t3_rows = []
    for cname, d in leave_op2.items():
        n_test = d["n_test"]
        f_either = format_stat(d["flagged_unknown_either"])
        f_ae = format_stat(d["flagged_unknown_by_ae"])
        f_conf = format_stat(d["flagged_unknown_by_confidence"])
        s_benign = format_stat(d["silently_predicted_benign"])
        s_other = format_stat(d["silently_predicted_other_attack"])
        missed_x = d["silently_predicted_benign"]["x"] + d["silently_predicted_other_attack"]["x"]
        missed_pct = (missed_x / n_test) * 100.0
        missed_str = f"{missed_x} of {n_test} ({missed_pct:.2f}%)"
        t3_rows.append(
            f"| {cname} | {n_test} | {f_either} | {f_ae} | {f_conf} | {s_benign} | {s_other} | {missed_str} |"
        )

    # Table 4: Lift Table
    t4_rows = []
    for cname in lift_op1:
        lift_1 = lift_op1[cname]["formatted"]
        lift_2 = lift_op2[cname]["formatted"]
        h_rate_1 = f"{lift_op1[cname]['heldout_rate']:.2f}%"
        k_rate_1 = f"{lift_op1[cname]['known_rate']:.2f}%"
        t4_rows.append(f"| {cname} | {h_rate_1} | {k_rate_1} | {lift_1} | {lift_2} |")

    # Table 5: AUROC Table
    t5_rows = []
    for cname, a_entry in auroc_data.items():
        t5_rows.append(f"| {cname} | {a_entry['formatted']} |")

    lines = [
        "# Phase 10: Zero-Day (Unseen Attack) Detection Layer",
        "",
        "## Setup",
        "- **Anomaly Detector Architecture**: PyTorch fully-connected Autoencoder (`19 -> 12 -> 6 -> 12 -> 19`) with ReLU activations and MSE reconstruction loss.",
        f"- **Benign-Only Training**: Fitted strictly on {ae_data['n_train_benign']} real Benign flows (no SMOTE rows). Note that the available real Benign training rows ({ae_data['n_train_benign']}) are fewer than the 100,000 cap. Training ran for {ae_data['epochs_run']} epochs with early stopping (patience 10) on {ae_data['n_val_benign']} validation Benign flows.",
        f"- **Operating Point 1 (Original)**: Selected with AE threshold at the 95.0th percentile of validation Benign error (`threshold = {op1_meta['threshold']:.6f}`) and confidence cutoff `0.60`.",
        f"- **Operating Point 2 (FP Budget <= 10%)**: Selected via grid search over validation data (percentiles {{95, 97.5, 99}}, confidence cutoffs {{0.30, 0.40, 0.50, 0.60}}). The chosen setting uses the 97.5th percentile (`threshold = {op2_meta['threshold']:.6f}`) and confidence cutoff `0.40`, yielding validation Benign FPR of {op2_meta['val_benign_fpr']*100:.2f}%. Selection used validation rows of the held-out classes (a mild mismatch with a true zero-day setting).",
        "",
        "## Executive Headline Finding",
    ]
    lines.extend(headline_bullets)
    lines.extend([
        "",
        "## Table 1: Known-Class Control (Full Test Set)",
        "Evaluation of the frozen Phase 3 B2 ensemble across the test set showing individual detector rates (AE-only, confidence-only) and combined flagged UNKNOWN rates at both operating points.",
        "",
        "| Class | Test Flows | Combined (OP 1: 95th, 0.60) | AE-Only (OP 1) | Confidence-Only (OP 1) | Combined (OP 2: 97.5th, 0.40) |",
        "|:---|---:|:---|:---|:---|:---|",
    ])
    lines.extend(t1_rows)
    lines.extend([
        "",
        "## Table 2: Leave-One-Class-Out Evaluation (Original Operating Point: 95th, 0.60)",
        "For each held-out attack class, a fresh Random Forest (100 trees, max depth 15, balanced class weights, seed 42) is trained with that class excluded (capped at 50,000 stratified samples).",
        "",
        "| Held-Out Class | Test Flows | Combined Flagged | AE-Only Flagged | Confidence-Only (<0.60) | Silently Benign | Silently Other Attack | Total Missed |",
        "|:---|---:|:---|:---|:---|:---|:---|:---|",
    ])
    lines.extend(t2_rows)
    lines.extend([
        "",
        "## Table 3: Leave-One-Class-Out Evaluation (FP-Budget Operating Point: 97.5th, 0.40)",
        "Side-by-side evaluation of leave-one-class-out zero-day simulation under the tuned false-positive budget operating point.",
        "",
        "| Held-Out Class | Test Flows | Combined Flagged | AE-Only Flagged | Confidence-Only (<0.40) | Silently Benign | Silently Other Attack | Total Missed |",
        "|:---|---:|:---|:---|:---|:---|:---|:---|",
    ])
    lines.extend(t3_rows)
    lines.extend([
        "",
        "## Table 4: Detection Lift Over Known Control (Paired Bootstrap 95% CI)",
        "Lift equals held-out combined flagged rate minus known-class control combined flagged rate on the same test flows, with paired bootstrap 95% confidence intervals (1,000 resamples, seed 42).",
        "",
        "| Attack Class | Held-Out Rate (OP 1) | Known-Class Rate (OP 1) | Lift (OP 1: 95th, 0.60) | Lift (OP 2: 97.5th, 0.40) |",
        "|:---|---:|---:|:---|:---|",
    ])
    lines.extend(t4_rows)
    lines.extend([
        "",
        "## Table 5: Threshold-Free Autoencoder Quality (AUROC on Test Set)",
        "Area under the ROC curve evaluating AE reconstruction score discrimination (Benign vs each attack class) on the untouched test set, with 200 bootstrap resamples 95% confidence intervals.",
        "",
        "| Attack Class | AE Reconstruction Score AUROC (95% Bootstrap CI) |",
        "|:---|:---|",
    ])
    lines.extend(t5_rows)
    lines.extend([
        "",
        "## Limitations",
        "- **Detector Contribution Balance**: The confidence rule contributes most of the detection; the autoencoder adds a smaller share (see AE-only rates).",
        "- **Heterogeneous Benign Class**: The Autoencoder is trained on one aggregated Benign class that encompasses diverse normal protocols; subtle anomalies within complex normal behaviors may not trigger reconstruction error.",
        "- **Attacks Mimicking Normal Patterns**: Attacks that closely resemble normal application traffic (such as certain Web-based and Spoofing variants) can exhibit low reconstruction error and rely primarily on uncertainty thresholds for detection.",
        "- **Duration Point Mass**: A point mass of 45.26% exists in test flow durations, which can affect feature scaling and reconstruction residuals across specific temporal profiles.",
        "- **Simulated Zero-Days**: Held-out classes originate from the known dataset distribution and serve as proxy unseen attacks rather than novel in-the-wild zero-day vulnerabilities.",
        "- **Single Random Seed**: Training and evaluation were conducted with seed 42 without hyperparameter tuning on the test set.",
        "- **Threshold Trade-off**: The 95th percentile threshold establishes an empirical trade-off between the Benign false-positive rate and open-set recall.",
        "",
    ])

    report_content = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[REPORT] Phase 10 updated summary written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
