"""
ids_system/run_integration.py
=============================
Phase 13: Consolidated Pipeline Integration & Parity Verification.

Executes:
  1. Reusable parity tests (P1 to P7) comparing the unified IDSPipeline
     against individual Phase 1-12 subsystems.
  2. End-to-end integration run on 2,000 stratified held-out test rows:
     - 8-class accuracy & macro F1 with 95% Wilson CI
     - Comparison with Phase 11 B2 edge baseline
     - UNKNOWN rate per true class with Wilson CI & comparison with Phase 12 known control
     - Single-threaded throughput and p95 per-flow latency
     - Peak resident memory (RSS MB)
     - MARL tactical response actions on 500 flows
     - Cryptographic forensic alert logging & verification on 200 flows
  3. Writes results to ids_system/results/integration_report.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch

# Enforce single-threaded deterministic CPU execution
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alerts.alert_logger import ForensicAlertLogger
from alerts.verify_log import verify_alert_log
from config import ENCODER_8_PATH, get_flow_severity
from ids_system.pipeline import IDSPipeline
from predict import get_or_load_model
from xai.shap_explain import SHAPExplainer
from zeroday.train_autoencoder import BenignAutoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_integration")

RESULTS_DIR = PROJECT_ROOT / "ids_system" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INTEGRATION_JSON_PATH = RESULTS_DIR / "integration_report.json"


def wilson_ci(x: int, n: int, confidence: float = 0.95) -> Dict[str, Any]:
    """Computes 95% Wilson score confidence interval."""
    if n <= 0:
        return {
            "x": int(x),
            "n": int(n),
            "pct": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "formatted": "0 of 0 (0.00% [0.00%, 0.00%])",
        }
    p = float(x) / float(n)
    z = 1.9599639845400542
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    hw = (z * math.sqrt((p * (1.0 - p) / n) + (z**2) / (4.0 * (n**2)))) / denom
    low = max(0.0, center - hw)
    high = min(1.0, center + hw)
    return {
        "x": int(x),
        "n": int(n),
        "pct": round(p * 100.0, 2),
        "ci_lower": round(low * 100.0, 2),
        "ci_upper": round(high * 100.0, 2),
        "formatted": f"{x} of {n} ({p*100.0:.2f}%) [{low*100.0:.2f}%, {high*100.0:.2f}%]",
    }


def load_test_dataset(n_samples: Optional[int] = None, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Loads test data. Inverts StandardScaler on data/processed/test.parquet
    to obtain real unscaled feature inputs, preserving repo-relative containment.
    Returns (raw_df_19, scaled_df_19, y_test_8).
    """
    test_pq = PROJECT_ROOT / "data" / "processed" / "test.parquet"
    if not test_pq.exists():
        raise FileNotFoundError(f"Missing {test_pq}")

    df_full = pd.read_parquet(test_pq)
    pipeline = IDSPipeline()
    features = pipeline.features

    if n_samples is not None and n_samples < len(df_full):
        _, df_sample = train_test_split(
            df_full,
            test_size=n_samples,
            stratify=df_full["label_8"],
            random_state=seed,
        )
    else:
        df_sample = df_full

    scaled_df_19 = df_sample[features].copy()
    y_test_8 = df_sample["label_8"].copy()

    means, scales = pipeline.scaler_params_19
    raw_arr = scaled_df_19.to_numpy(dtype=np.float64) * scales + means
    raw_df_19 = pd.DataFrame(raw_arr, columns=features, index=df_sample.index)

    return raw_df_19, scaled_df_19, y_test_8


def run_parity_tests(n_samples: int = 500, seed: int = 42) -> Dict[str, Any]:
    """
    Executes parity checks P1 to P7 against individual Phase 1-12 subsystems.
    Used by pytest test suite and CLI selftest.
    """
    log.info(f"Running Parity Tests P1-P7 on {n_samples} test rows (seed={seed}) ...")
    pipeline = IDSPipeline()
    raw_df_19, scaled_df_19, y_test_8 = load_test_dataset(n_samples=n_samples, seed=seed)

    results: Dict[str, Any] = {}

    # 1. Pipeline batch analysis
    t0 = time.time()
    pipe_out = pipeline.analyze_batch(raw_df_19)
    elapsed_an = time.time() - t0

    # -------------------------------------------------------------
    # P1: Pipeline Raw Class Probabilities == predict.py Output
    # -------------------------------------------------------------
    tol = 1e-6
    pred_model = get_or_load_model("8class", "b2")
    probs_predict = pred_model.predict_probabilities(scaled_df_19)["weighted"].astype(np.float32)
    pipe_probs = pipe_out[[f"prob_{c}" for c in pipeline.class_names]].to_numpy(dtype=np.float32)
    max_p1_diff = float(np.max(np.abs(pipe_probs - probs_predict)))
    p1_pass = max_p1_diff <= tol
    results["P1"] = {
        "name": "Pipeline Raw Probabilities == predict.py Output",
        "status": "passed" if p1_pass else "failed",
        "max_diff": max_p1_diff,
        "tolerance": tol,
        "n_samples": n_samples,
    }

    # -------------------------------------------------------------
    # P2: Pipeline Calibrated Confidence == Calibration Code Output
    # -------------------------------------------------------------
    calib_path = PROJECT_ROOT / "calibration" / "models" / "chosen_calibrator.joblib"
    calibrator = joblib.load(calib_path)
    P_cal_direct = calibrator.predict_proba(probs_predict)
    cal_conf_direct = np.max(P_cal_direct, axis=1).astype(np.float32)
    pipe_cal_conf = pipe_out["calibrated_confidence"].to_numpy(dtype=np.float32)
    max_p2_diff = float(np.max(np.abs(pipe_cal_conf - cal_conf_direct)))
    p2_pass = max_p2_diff <= tol
    results["P2"] = {
        "name": "Pipeline Calibrated Confidence == Calibration Code Output",
        "status": "passed" if p2_pass else "failed",
        "max_diff": max_p2_diff,
        "tolerance": tol,
        "n_samples": n_samples,
    }

    # -------------------------------------------------------------
    # P3: Pipeline AE Score == Phase 10 AE Scoring Code Output
    # -------------------------------------------------------------
    ae_pt_path = PROJECT_ROOT / "zeroday" / "models" / "autoencoder.pt"
    checkpoint = torch.load(ae_pt_path, map_location="cpu", weights_only=False)
    ae_model = BenignAutoencoder(in_features=19)
    ae_model.load_state_dict(checkpoint["state_dict"])
    ae_model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(scaled_df_19.to_numpy(dtype=np.float32))
        rec = ae_model(x_t)
        ae_scores_direct = ((x_t - rec) ** 2).mean(dim=1).cpu().numpy().astype(np.float32)
    pipe_ae_scores = pipe_out["ae_score"].to_numpy(dtype=np.float32)
    max_p3_diff = float(np.max(np.abs(pipe_ae_scores - ae_scores_direct)))
    p3_pass = max_p3_diff <= tol
    results["P3"] = {
        "name": "Pipeline AE Score == Phase 10 AE Scoring Output",
        "status": "passed" if p3_pass else "failed",
        "max_diff": max_p3_diff,
        "tolerance": tol,
        "n_samples": n_samples,
    }

    # -------------------------------------------------------------
    # P4: Severity == Phase 3 Severity Table for Predicted Class
    # -------------------------------------------------------------
    expected_severities = [
        get_flow_severity(cls, conf)
        for cls, conf in zip(pipe_out["predicted_class"], pipe_out["raw_confidence"])
    ]
    p4_matches = (pipe_out["severity"].tolist() == expected_severities)
    results["P4"] = {
        "name": "Severity == Phase 3 Deterministic Severity Table",
        "status": "passed" if p4_matches else "failed",
        "matched_count": int(sum(a == b for a, b in zip(pipe_out["severity"], expected_severities))),
        "total_count": n_samples,
    }

    # -------------------------------------------------------------
    # P5: Pipeline SHAP Top-5 Feature Names == xai/shap_explain.py (20 rows)
    # -------------------------------------------------------------
    n_shap_rows = min(20, n_samples)
    explainer = SHAPExplainer()
    shap_mismatches = 0
    for i in range(n_shap_rows):
        row_raw = raw_df_19.iloc[i].to_dict()
        exp_pipe = pipeline.explain(row_raw, generate_brief=False)
        pipe_top5 = [f["feature"] for f in exp_pipe["top_features"]]

        scaled_row = scaled_df_19.iloc[[i]]
        shap_matrix = explainer.explain_xgb(scaled_row)
        pred_idx = exp_pipe["predicted_class_idx"]
        top5_direct = explainer.get_local_explanation(
            shap_values_single=shap_matrix[0],
            unscaled_features_dict=row_raw,
            target_class_idx=pred_idx,
            top_k=5,
        )
        direct_top5 = [f["feature"] for f in top5_direct]
        if pipe_top5 != direct_top5:
            shap_mismatches += 1

    p5_pass = (shap_mismatches == 0)
    results["P5"] = {
        "name": "Pipeline SHAP Top-5 Feature Names == xai/shap_explain.py Output (20 rows)",
        "status": "passed" if p5_pass else "failed",
        "tested_rows": n_shap_rows,
        "mismatches": shap_mismatches,
    }

    # -------------------------------------------------------------
    # P6: Alert Logging Chain Integrity & Tamper Rejection
    # -------------------------------------------------------------
    # Clean previous live log for deterministic isolated check
    test_live_log = PROJECT_ROOT / "logs" / "alerts_live_test.jsonl"
    test_tail_hash = PROJECT_ROOT / "logs" / "alerts_live_test_tail.txt"
    if test_live_log.exists():
        test_live_log.unlink()
    if test_tail_hash.exists():
        test_tail_hash.unlink()

    test_logger = ForensicAlertLogger(log_path=test_live_log, tail_hash_path=test_tail_hash)
    for i in range(5):
        test_logger.log_alert(
            flow_index=i,
            predicted_class="DDoS",
            confidence=0.98,
            severity="Critical",
            explanation="Test volumetric flow",
            recommended_action="Block flow",
            source="rule_based_fallback",
            model_version="Phase13-IDSPipeline",
        )

    valid_clean, _, _ = verify_alert_log(test_live_log, test_tail_hash, enforce_anchor=True)

    # Tamper a copy
    tamper_copy = PROJECT_ROOT / "logs" / "alerts_tampered_test.jsonl"
    shutil.copyfile(test_live_log, tamper_copy)
    with open(tamper_copy, "r", encoding="utf-8") as f:
        lines = f.readlines()
    bad_rec = json.loads(lines[2])
    bad_rec["severity"] = "Low"
    lines[2] = json.dumps(bad_rec) + "\n"
    with open(tamper_copy, "w", encoding="utf-8") as f:
        f.writelines(lines)

    valid_tampered, _, _ = verify_alert_log(tamper_copy, test_tail_hash, enforce_anchor=True)

    # Clean up test files
    if test_live_log.exists():
        test_live_log.unlink()
    if test_tail_hash.exists():
        test_tail_hash.unlink()
    if tamper_copy.exists():
        tamper_copy.unlink()

    p6_pass = valid_clean and (not valid_tampered)
    results["P6"] = {
        "name": "Forensic Alert Logger Pass on Chain & Rejection on Tampered Copy",
        "status": "passed" if p6_pass else "failed",
        "clean_log_valid": bool(valid_clean),
        "tampered_copy_rejected": bool(not valid_tampered),
    }

    # -------------------------------------------------------------
    # P7: analyze_batch Rejects Missing Feature and Names Column
    # -------------------------------------------------------------
    col_dropped = pipeline.features[0]
    bad_df = raw_df_19.drop(columns=[col_dropped])
    p7_pass = False
    caught_name = ""
    try:
        pipeline.analyze_batch(bad_df)
    except ValueError as ex:
        caught_name = str(ex)
        if col_dropped in caught_name:
            p7_pass = True

    results["P7"] = {
        "name": "analyze_batch Rejects Missing Feature and Names Missing Column",
        "status": "passed" if p7_pass else "failed",
        "dropped_column": col_dropped,
        "named_in_error": col_dropped in caught_name,
    }

    return results


def run_end_to_end_integration() -> Dict[str, Any]:
    """
    Executes full end-to-end integration run on 2,000 stratified held-out test rows.
    """
    log.info("Starting Phase 13 End-to-End Integration Evaluation (2,000 rows, seed=42) ...")
    proc = psutil.Process()
    rss_before_mb = round(proc.memory_info().rss / (1024.0 * 1024.0), 2)

    pipeline = IDSPipeline()
    health_status = pipeline.health()
    raw_df_2000, scaled_df_2000, y_test_8 = load_test_dataset(n_samples=2000, seed=42)

    # Measure batch analysis throughput and per-flow latencies
    latencies_ms = []
    t_start = time.perf_counter()
    # Batch analyze
    analyzed_df = pipeline.analyze_batch(raw_df_2000)
    batch_elapsed = time.perf_counter() - t_start
    throughput_fps = round(2000.0 / max(1e-6, batch_elapsed), 2)

    # Single-thread per-flow latency measurement on 100 individual samples
    single_samples = raw_df_2000.head(100)
    for idx, row in single_samples.iterrows():
        t0 = time.perf_counter()
        _ = pipeline.analyze_flow(row.to_dict())
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p95_latency_ms = round(float(np.percentile(latencies_ms, 95)), 3)
    p50_latency_ms = round(float(np.percentile(latencies_ms, 50)), 3)
    peak_rss_mb = round(proc.memory_info().rss / (1024.0 * 1024.0), 2)

    # Classification Metrics (8 classes)
    pred_classes = analyzed_df["predicted_class"].to_numpy()
    if np.issubdtype(y_test_8.dtype, np.number):
        true_classes = np.array([pipeline.class_names[int(i)] for i in y_test_8])
    else:
        true_classes = y_test_8.to_numpy()

    acc = float(accuracy_score(true_classes, pred_classes))
    correct_count = int(np.sum(true_classes == pred_classes))
    acc_ci = wilson_ci(correct_count, 2000)
    macro_f1 = float(f1_score(true_classes, pred_classes, average="macro"))

    # Comparison with Phase 11 B2 Baseline
    edge_final_path = PROJECT_ROOT / "edge" / "results" / "final_comparison.json"
    phase11_baseline_acc = None
    phase11_baseline_f1 = None
    if edge_final_path.exists():
        with open(edge_final_path, "r", encoding="utf-8") as f:
            edge_data = json.load(f)
        b_info = edge_data.get("test_evaluations", {}).get("baseline", {})
        phase11_baseline_acc = b_info.get("test_accuracy")
        phase11_baseline_f1 = b_info.get("test_macro_f1")

    acc_diff_p11 = round(acc - phase11_baseline_acc, 4) if phase11_baseline_acc is not None else None
    f1_diff_p11 = round(macro_f1 - phase11_baseline_f1, 4) if phase11_baseline_f1 is not None else None

    # UNKNOWN Flagging Rate per True Class with Wilson CIs
    phase12_summary_path = PROJECT_ROOT / "calibration" / "results" / "openset_calibrated_summary.json"
    phase12_known_control = {}
    if phase12_summary_path.exists():
        with open(phase12_summary_path, "r", encoding="utf-8") as f:
            p12_d = json.load(f)
        phase12_known_control = p12_d.get("known_class_control", {})

    unknown_per_class: Dict[str, Any] = {}
    unique_classes = pipeline.class_names

    for c_name in unique_classes:
        mask = (true_classes == c_name)
        n_c = int(np.sum(mask))
        n_unk = int(np.sum(analyzed_df.loc[mask, "unknown_flag"]))
        ci_dict = wilson_ci(n_unk, n_c)

        # Retrieve Phase 12 control rate
        p12_ctrl = phase12_known_control.get(c_name, {})
        p12_cal_flag = p12_ctrl.get("calibrated_flagged", {})
        p12_rate_pct = p12_cal_flag.get("percentage")

        unknown_per_class[c_name] = {
            "n_samples": n_c,
            "n_unknown": n_unk,
            "rate_pct": ci_dict["pct"],
            "ci_lower": ci_dict["ci_lower"],
            "ci_upper": ci_dict["ci_upper"],
            "formatted": ci_dict["formatted"],
            "phase12_control_pct": p12_rate_pct,
            "difference_vs_phase12": round(ci_dict["pct"] - p12_rate_pct, 2) if p12_rate_pct is not None else None,
        }

    # MARL Tactical Response Simulation on 500 consecutive flows
    log.info("Executing MARL Tactical Response on 500 consecutive flows ...")
    flows_500 = analyzed_df.head(500).copy()
    marl_out_500 = pipeline.respond(flows_500)

    triage_counts = collections.Counter(marl_out_500["marl_triage"])
    contain_counts = collections.Counter(marl_out_500["marl_containment"])
    esc_counts = collections.Counter(marl_out_500["marl_escalation"])

    marl_shares = {
        "n_flows": 500,
        "triage_action_shares": {
            act: round(cnt / 500.0, 4) for act, cnt in triage_counts.items()
        },
        "containment_action_shares": {
            act: round(cnt / 500.0, 4) for act, cnt in contain_counts.items()
        },
        "escalation_action_shares": {
            act: round(cnt / 500.0, 4) for act, cnt in esc_counts.items()
        },
    }

    # Forensic Alert Logging & Verification on 200 flows
    log.info("Executing Forensic Alert Logging & Verification on 200 flows ...")
    pipeline.reset_alert_log()

    flows_200 = marl_out_500.head(200).copy()
    for idx, row in flows_200.iterrows():
        r_dict = row.to_dict()
        r_dict["flow_index"] = idx
        r_dict["true_class"] = y_test_8.loc[idx]
        pipeline.log_alert(r_dict)

    verify_report = pipeline.verify_log()
    log.info(f"Forensic log verification: {verify_report['status']} ({verify_report['lines_verified']} lines)")

    # Parity Tests
    parity_summary = run_parity_tests(n_samples=500, seed=42)

    report_data = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "health": health_status,
        "parity_tests": parity_summary,
        "metrics_2000_test": {
            "n_samples": 2000,
            "accuracy": round(acc, 4),
            "accuracy_wilson_ci": [acc_ci["ci_lower"], acc_ci["ci_upper"]],
            "accuracy_formatted": acc_ci["formatted"],
            "macro_f1": round(macro_f1, 4),
            "phase11_baseline_accuracy": phase11_baseline_acc,
            "phase11_baseline_macro_f1": phase11_baseline_f1,
            "accuracy_difference_vs_phase11": acc_diff_p11,
            "macro_f1_difference_vs_phase11": f1_diff_p11,
            "accuracy_difference_explanation": "Evaluated on 2,000 stratified samples (test.parquet) versus full 26,292 test split in Phase 11.",
        },
        "unknown_flagging_by_class": unknown_per_class,
        "performance": {
            "throughput_fps": throughput_fps,
            "latency_p50_ms": p50_latency_ms,
            "latency_p95_ms": p95_latency_ms,
            "rss_before_mb": rss_before_mb,
            "peak_rss_mb": peak_rss_mb,
            "execution_threads": 1,
        },
        "marl_response_500_flows": marl_shares,
        "forensic_alert_logging_200_flows": {
            "n_logged": 200,
            "verification_status": verify_report["status"],
            "lines_verified": verify_report["lines_verified"],
            "tail_hash": verify_report["tail_hash"],
        },
    }

    with open(INTEGRATION_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    log.info(f"Integration report successfully written to {INTEGRATION_JSON_PATH}")
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 13 Integration Tests & Report")
    parser.add_argument("--parity-only", action="store_true", help="Run only P1-P7 parity checks")
    parser.add_argument("--samples", type=int, default=500, help="Number of rows for parity check")
    args = parser.parse_args()

    if args.parity_only:
        p_res = run_parity_tests(n_samples=args.samples)
        print(json.dumps(p_res, indent=2))
    else:
        rep = run_end_to_end_integration()
        print("\n" + "=" * 70)
        print("          PHASE 13 INTEGRATION REPORT SUMMARY")
        print("=" * 70)
        print(f"Overall Health:       {rep['health']['overall_status'].upper()}")
        print(f"8-Class Accuracy:     {rep['metrics_2000_test']['accuracy'] * 100:.2f}% (Wilson 95% CI: [{rep['metrics_2000_test']['accuracy_wilson_ci'][0]}%, {rep['metrics_2000_test']['accuracy_wilson_ci'][1]}%])")
        print(f"8-Class Macro F1:     {rep['metrics_2000_test']['macro_f1']:.4f}")
        print(f"Phase 11 Baseline Acc:{rep['metrics_2000_test']['phase11_baseline_accuracy']}")
        print(f"Throughput:           {rep['performance']['throughput_fps']:.1f} flows/sec")
        print(f"p95 Flow Latency:     {rep['performance']['latency_p95_ms']:.3f} ms")
        print(f"Peak Memory (RSS):    {rep['performance']['peak_rss_mb']} MB")
        print(f"Alert Log Status:     {rep['forensic_alert_logging_200_flows']['verification_status']}")
        print("=" * 70)
        all_passed = all(
            v["status"] in ("passed", "known_mismatch")
            for v in rep["parity_tests"].values()
        )
        print("PARITY TESTS:", "ALL PASSED" if all_passed else "SOME FAILED")
        for k, v in rep["parity_tests"].items():
            print(f"  {k}: {v['status'].upper()} - {v['name']}")
        print("=" * 70)
