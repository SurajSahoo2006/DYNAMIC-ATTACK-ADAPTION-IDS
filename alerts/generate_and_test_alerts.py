"""
alerts/generate_and_test_alerts.py
==================================
Phase 4 - Alert Log Generator and Tamper Test Suite

1. Ingests the 40 demo flows from Phase 3/4 and appends them to logs/alerts.jsonl
   with full forensic telemetry, ISO UTC timestamps, top SHAP features, and hash chaining.
2. Appends 5 sample adversarial flows from artifacts/adversarial_demo_flows.parquet
   marked explicitly with adversarial=True.
3. Runs cryptographic verification on the valid log using verify_log.py.
4. Creates a tampered copy of the log, alters a field in line 15, and proves that
   verify_log.py detects and pinpoints the tamper event.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from alerts.alert_logger import ForensicAlertLogger
from alerts.verify_log import verify_alert_log
from config import ARTIFACTS_DIR, REPORTS_DIR


def generate_forensic_alerts_log():
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    alerts_file = logs_dir / "alerts.jsonl"

    # Reset log for clean generation
    if alerts_file.exists():
        alerts_file.unlink()

    logger = ForensicAlertLogger(log_path=alerts_file)

    # 1. Load 40 demo flows and SHAP local attributions
    demo_df = pd.read_csv(REPORTS_DIR / "llm_demo_outputs_shap.csv")
    shap_path = ARTIFACTS_DIR / "shap_local_40flows.json"
    shap_local = {}
    if shap_path.exists():
        with open(shap_path, "r", encoding="utf-8") as f:
            shap_local = json.load(f)

    print(f"[ALERTS] Logging 40 standard test flows to {alerts_file.name} ...")
    for _, row in demo_df.iterrows():
        flow_idx = int(row["flow_index"])
        flow_shap = shap_local.get(str(flow_idx), {}).get("top_shap_features", [])

        logger.log_alert(
            flow_index=flow_idx,
            predicted_class=str(row["ensemble_predicted_class"]),
            true_class=str(row["true_8class"]),
            confidence=float(row["confidence"]),
            severity=str(row["severity"]),
            explanation=str(row["explanation"]),
            recommended_action=str(row["recommended_action"]),
            source=str(row.get("source", "ollama")),
            model_version="B2-Ensemble-8class",
            top_shap_features=flow_shap,
            adversarial=False,
        )

    # 2. Append 5 adversarial flows
    adv_path = ARTIFACTS_DIR / "adversarial_demo_flows.parquet"
    if adv_path.exists():
        adv_df = pd.read_parquet(adv_path)
        print(f"[ALERTS] Logging 5 adversarial evasion flows with adversarial=True ...")
        for idx in adv_df.index[:5]:
            row = adv_df.loc[idx]
            true_cls = row["true_class"]
            pred_cls = row["adversarial_pred_class"]

            logger.log_alert(
                flow_index=int(idx),
                predicted_class=str(pred_cls),
                true_class=str(true_cls),
                confidence=0.55,  # Evasion shift
                severity="High",
                explanation=f"Adversarial evasion flow: traffic features were perturbed using constrained FGSM. True label '{true_cls}' classified as '{pred_cls}'.",
                recommended_action="Route flow to quarantine VLAN and inspect packet size variance for artificial padding anomalies.",
                source="art_adversarial_simulation",
                model_version="B2-Ensemble-8class",
                top_shap_features=None,
                adversarial=True,
                custom_metadata={
                    "attack_vector": "Constrained-FGSM-Transfer",
                    "perturbation_budget_epsilon": 0.10,
                },
            )

    print(f"[ALERTS] Log populated: {len(demo_df) + 5} total records written to {alerts_file.name}.")

    # 3. Test verification on pristine log
    print("\n--- Test 1: Verifying Pristine Alert Log ---")
    valid, count, msg = verify_alert_log(alerts_file)
    print(f"Validation Result: {'PASS' if valid else 'FAIL'}")
    print(f"Details: {msg}")
    assert valid, "Pristine log verification must pass!"

    # 4. Test verification on tampered copy
    print("\n--- Test 2: Verifying Tampered Alert Log Copy ---")
    tampered_file = logs_dir / "alerts_tampered_test.jsonl"
    shutil.copyfile(alerts_file, tampered_file)

    # Modify line 15 (alter predicted severity)
    lines = []
    with open(tampered_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    target_record = json.loads(lines[14])
    print(f"Tampering Line 15 (Alert {target_record['alert_id']}): changing severity from '{target_record['severity']}' to 'Low' ...")
    target_record["severity"] = "Low"
    lines[14] = json.dumps(target_record) + "\n"

    with open(tampered_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    valid_tamper, count_tamper, msg_tamper = verify_alert_log(tampered_file)
    print(f"Validation Result on Tampered File: {'PASS (Undetected)' if valid_tamper else 'CORRECTLY DETECTED TAMPER'}")
    print(f"Details:\n{msg_tamper}")
    assert not valid_tamper, "Tampered log must be detected and rejected!"

    return {
        "pristine_verified_count": count,
        "tamper_detected": not valid_tamper,
        "tamper_detection_message": msg_tamper,
    }


if __name__ == "__main__":
    generate_forensic_alerts_log()
