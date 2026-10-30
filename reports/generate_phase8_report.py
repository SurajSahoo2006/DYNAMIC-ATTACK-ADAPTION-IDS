"""
reports/generate_phase8_report.py
=================================
Generates reports/phase8_summary.md reading strictly from:
  - drift/drift_metadata.json (or data/drift/drift_metadata.json)
  - drift/drift_detection_results.json
  - drift/retrain_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_REPORT = REPORTS_DIR / "phase8_summary.md"


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load metadata
    meta_path = PROJECT_ROOT / "drift" / "drift_metadata.json"
    if not meta_path.exists():
        meta_path = PROJECT_ROOT / "data" / "drift" / "drift_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # 2. Load detection results
    det_path = PROJECT_ROOT / "drift" / "drift_detection_results.json"
    with open(det_path, "r", encoding="utf-8") as f:
        det_data = json.load(f)

    # 3. Load retrain results
    retrain_path = PROJECT_ROOT / "drift" / "retrain_results.json"
    with open(retrain_path, "r", encoding="utf-8") as f:
        retrain_data = json.load(f)

    # Build report lines
    lines = [
        "# Phase 8: Concept Drift Detection and Incremental Adaptation",
        "",
        "## Experimental Setup",
        f"- {meta_data.get('n_windows', 5)} temporal windows from held-out test set",
        "- Drift injected in window 2: DDoS flow_duration x8, IAT x6",
        f"- Drifted rows in window 2: {meta_data.get('drifted_rows_in_window_2', 'N/A')}",
        "- Detector: Page-Hinkley (min_instances=30, threshold=50.0)",
        "",
        "## Drift Detection Results",
        "| Window | Samples | Accuracy | Drift Detected | First Drift Index |",
        "|--------|---------|----------|----------------|-------------------|",
    ]

    for w in det_data.get("windows", []):
        wid = w.get("window_id")
        samples = w.get("n_samples")
        acc = w.get("accuracy", 0.0) * 100
        detected = "Yes" if w.get("drift_detected") else "No"
        idx = str(w.get("first_drift_index")) if w.get("first_drift_index") is not None else "-"
        lines.append(f"| {wid} | {samples} | {acc:.2f}% | {detected} | {idx} |")

    trig = retrain_data.get("retrain_triggered_by", "scheduled")
    n_retrain = retrain_data.get("n_retrain_samples", "N/A")
    orig_acc = retrain_data.get("original_rf_accuracy_window3", 0.0) * 100
    ret_acc = retrain_data.get("retrained_rf_accuracy_window3", 0.0) * 100
    delta = retrain_data.get("accuracy_delta", 0.0) * 100
    delta_sign = "+" if delta >= 0 else ""

    lines.extend([
        "",
        "## Incremental Retraining",
        f"- Retrain triggered by: {trig}",
        f"- Training windows: 0, 1, 2 ({n_retrain} total samples)",
        f"- Original RF accuracy on window 3 : {orig_acc:.2f}%",
        f"- Retrained RF accuracy on window 3: {ret_acc:.2f}%",
        f"- Delta: {delta_sign}{delta:.2f}%",
        "",
        "## Limitations",
        "- Drift simulation is synthetic (feature multiplication), not real traffic change",
        "- Retrained model uses RF only; XGBoost and ensemble not yet incrementally updated",
        "- Page-Hinkley threshold tuned for demonstration; production use requires calibration on held-out drift examples",
        "",
    ])

    report_content = "\n".join(lines)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Phase 8 report written to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
