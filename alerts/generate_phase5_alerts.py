"""
alerts/generate_phase5_alerts.py
================================
Stage 7 Part 1: Generates logs/alerts_phase5.jsonl and audits tamper-evidence.

Augments the 45 alerts from Phase 4 with MARL tactical decision fields:
  - marl_triage_action
  - marl_containment_action
  - marl_escalation_action
  - marl_policy_version

Maintains cryptographic SHA-256 hash chaining and updates the external tail anchor
at logs/alerts_phase5_tail_hash.txt.

Audits integrity:
  1. Pristine log: must PASS with full hash continuity and tail anchor match.
  2. Modified copy: must FAIL with content modification tamper alert.
  3. Truncated copy: must FAIL with tail anchor mismatch alert.

Saves verification log to reports/phase5_log_verification.txt.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import tempfile
from typing import Dict, List

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from alerts.alert_logger import ForensicAlertLogger
from alerts.verify_log import verify_alert_log
from config_marl import (
    ALERTS_PHASE5_JSONL_PATH,
    ALERTS_PHASE5_TAIL_HASH_PATH,
    CONTAINMENT_ACTIONS,
    ESCALATION_ACTIONS,
    LOGS_DIR,
    MARL_MODELS_DIR,
    PHASE5_LOG_VERIFICATION_TXT_PATH,
    TRIAGE_ACTIONS,
)
from marl.mappo import MAPPOTeam

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
log = logging.getLogger("phase5_alerts")


def generate_and_verify_alerts() -> Dict:
    log.info("Starting Phase 5 alert generation and verification...")
    phase4_log_path = LOGS_DIR / "alerts.jsonl"
    if not phase4_log_path.exists():
        raise FileNotFoundError(f"Missing Phase 4 alerts log at {phase4_log_path}")

    # Remove existing phase 5 log and anchor if present to ensure fresh chain
    if ALERTS_PHASE5_JSONL_PATH.exists():
        ALERTS_PHASE5_JSONL_PATH.unlink()
    if ALERTS_PHASE5_TAIL_HASH_PATH.exists():
        ALERTS_PHASE5_TAIL_HASH_PATH.unlink()

    # Load trained MAPPO team
    best_ckpt = MARL_MODELS_DIR / "mappo_balanced_seed123_best.pt"
    if not best_ckpt.exists():
        best_ckpt = MARL_MODELS_DIR / "mappo_balanced_seed42_best.pt"
    log.info(f"Loading MAPPO policy from {best_ckpt}...")
    team = MAPPOTeam()
    team.load_checkpoint(best_ckpt)

    logger = ForensicAlertLogger(
        log_path=ALERTS_PHASE5_JSONL_PATH,
        tail_hash_path=ALERTS_PHASE5_TAIL_HASH_PATH,
    )

    with open(phase4_log_path, "r", encoding="utf-8") as f:
        p4_records = [json.loads(line.strip()) for line in f if line.strip()]

    log.info(f"Loaded {len(p4_records)} records from Phase 4 alert log.")

    # Generate MARL actions and write to Phase 5 log
    class_map = {
        "Benign": 0, "BruteForce": 1, "DDoS": 2, "DoS": 3,
        "Mirai": 4, "Recon": 5, "Spoofing": 6, "Web-based": 7,
    }
    sev_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

    for idx, rec in enumerate(p4_records):
        # Synthesize observation vector for MAPPO decision
        probs = np.zeros(8, dtype=np.float32)
        pred_cls = rec.get("predicted_class", "Benign")
        pred_idx = class_map.get(pred_cls, 0)
        conf = float(rec.get("confidence", 0.9))
        probs[pred_idx] = conf
        rem_p = (1.0 - conf) / 7.0
        for i in range(8):
            if i != pred_idx:
                probs[i] = rem_p

        sev_str = rec.get("severity", "Low")
        sev_onehot = np.zeros(4, dtype=np.float32)
        sev_onehot[sev_map.get(sev_str, 0)] = 1.0

        step_frac = float(idx) / float(len(p4_records))
        esc_left = 1.0
        posture_act = 0.0
        posture_rem = 0.0
        hist_counts = np.zeros(8, dtype=np.float32)
        hist_counts[pred_idx] = 0.2

        env_obs = np.concatenate([
            probs,
            np.array([conf], dtype=np.float32),
            sev_onehot,
            np.array([step_frac, esc_left, posture_act, posture_rem], dtype=np.float32),
            hist_counts,
        ], dtype=np.float32)

        obs_matrix = np.zeros((3, 28), dtype=np.float32)
        for a_i in range(3):
            obs_matrix[a_i, :25] = env_obs
            obs_matrix[a_i, 25 + a_i] = 1.0

        actions, _, _ = team.select_actions(obs_matrix, deterministic=True)
        triage_act = TRIAGE_ACTIONS[actions[0]]
        contain_act = CONTAINMENT_ACTIONS[actions[1]]
        esc_act = ESCALATION_ACTIONS[actions[2]]

        # Write alert to Phase 5 log
        logger.log_alert(
            flow_index=rec["flow_index"],
            predicted_class=rec["predicted_class"],
            confidence=rec["confidence"],
            severity=rec["severity"],
            explanation=rec["explanation"],
            recommended_action=rec["recommended_action"],
            source=rec.get("source", "ollama"),
            model_version=rec.get("model_version", "B2-Ensemble-8class"),
            top_shap_features=rec.get("top_shap_features"),
            adversarial=rec.get("adversarial", False),
            true_class=rec.get("true_class"),
            marl_triage_action=triage_act,
            marl_containment_action=contain_act,
            marl_escalation_action=esc_act,
            marl_policy_version="MAPPO-v1.0-balanced",
        )

    log.info(f"Successfully generated {len(p4_records)} records into {ALERTS_PHASE5_JSONL_PATH}")

    # -------------------------------------------------------------------------
    # Verification Audits
    # -------------------------------------------------------------------------
    verification_lines = []
    verification_lines.append("=" * 76)
    verification_lines.append("  PHASE 5 FORENSIC LOG CRYPTOGRAPHIC VERIFICATION SUITE")
    verification_lines.append("=" * 76)

    # Test 1: Pristine Log Verification
    valid_p, count_p, msg_p = verify_alert_log(
        log_path=ALERTS_PHASE5_JSONL_PATH,
        tail_hash_path=ALERTS_PHASE5_TAIL_HASH_PATH,
        enforce_anchor=True,
    )
    verification_lines.append(f"\n[TEST 1] Pristine Log Verification ({ALERTS_PHASE5_JSONL_PATH.name}):")
    verification_lines.append(f"  Status: {'PASS' if valid_p else 'FAIL'}")
    verification_lines.append(f"  Records Verified: {count_p}")
    verification_lines.append(f"  Verifier Output: {msg_p}")
    assert valid_p, f"Pristine log verification failed! {msg_p}"

    # Test 2: Content Modification Tamper Detection
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
        tampered_path = Path(tf.name)
        with open(ALERTS_PHASE5_JSONL_PATH, "r", encoding="utf-8") as src:
            for l_idx, line in enumerate(src):
                if l_idx == 15:
                    entry = json.loads(line)
                    entry["confidence"] = 0.9999  # Tampered field
                    tf.write(json.dumps(entry) + "\n")
                else:
                    tf.write(line)

    valid_t, count_t, msg_t = verify_alert_log(
        log_path=tampered_path,
        tail_hash_path=ALERTS_PHASE5_TAIL_HASH_PATH,
        enforce_anchor=True,
    )
    tampered_path.unlink()
    verification_lines.append(f"\n[TEST 2] Content Modification Tamper Test (Altered record 15):")
    verification_lines.append(f"  Tamper Detected: {'YES (PASS)' if not valid_t else 'NO (FAIL)'}")
    verification_lines.append(f"  Verifier Output: {msg_t}")
    assert not valid_t, "Verifier failed to detect content tampering!"

    # Test 3: Tail Truncation Tamper Detection
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
        truncated_path = Path(tf.name)
        with open(ALERTS_PHASE5_JSONL_PATH, "r", encoding="utf-8") as src:
            all_l = src.readlines()
            tf.writelines(all_l[:-1])  # Drop last line

    valid_tr, count_tr, msg_tr = verify_alert_log(
        log_path=truncated_path,
        tail_hash_path=ALERTS_PHASE5_TAIL_HASH_PATH,
        enforce_anchor=True,
    )
    truncated_path.unlink()
    verification_lines.append(f"\n[TEST 3] Tail Truncation Tamper Test (Dropped last record without updating anchor):")
    verification_lines.append(f"  Truncation Detected: {'YES (PASS)' if not valid_tr else 'NO (FAIL)'}")
    verification_lines.append(f"  Verifier Output: {msg_tr}")
    assert not valid_tr, "Verifier failed to detect tail truncation!"

    verification_lines.append("\n" + "=" * 76)
    verification_lines.append("  VERIFICATION RESULT: 3/3 TESTS PASSED")
    verification_lines.append("=" * 76 + "\n")

    report_text = "\n".join(verification_lines)
    with open(PHASE5_LOG_VERIFICATION_TXT_PATH, "w", encoding="utf-8") as rf:
        rf.write(report_text)
    log.info(f"Saved verification log to {PHASE5_LOG_VERIFICATION_TXT_PATH}")
    print(report_text)


if __name__ == "__main__":
    generate_and_verify_alerts()
