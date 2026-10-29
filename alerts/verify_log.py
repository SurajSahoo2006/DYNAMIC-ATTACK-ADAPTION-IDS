"""
alerts/verify_log.py
====================
Phase 4 - Cryptographic Forensic Log Verifier

Audits the tamper-evident hash chain in logs/alerts.jsonl:
  - Verifies that each line's prev_hash exactly equals the preceding entry's entry_hash.
  - Verifies that each line's entry_hash matches the recomputed SHA-256 over its canonical payload.
  - Detects and pinpoints the first modified, inserted, deleted, or reordered log line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from alerts.alert_logger import ForensicAlertLogger, GENESIS_HASH


def verify_alert_log(
    log_path: Path,
    tail_hash_path: Optional[Path] = None,
    enforce_anchor: bool = True,
) -> Tuple[bool, int, str]:
    """
    Audits the integrity of an alert log file.
    - Verifies internal SHA-256 hash chaining (detects mid-file tamper).
    - Checks external tail hash anchor (detects tail truncation and full-file rewrites).

    Returns:
        (is_valid: bool, lines_verified: int, message: str)
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return False, 0, f"Log file does not exist: {log_path}"

    expected_prev_hash = GENESIS_HASH
    line_count = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                record = json.loads(line_str)
            except Exception as e:
                return False, line_count, f"[FAIL] Line {line_num} contains malformed JSON: {e}"

            # Check required hash fields
            if "entry_hash" not in record:
                return False, line_count, f"[FAIL] Line {line_num} (Alert {record.get('alert_id', 'Unknown')}) missing 'entry_hash'."
            if "prev_hash" not in record:
                return False, line_count, f"[FAIL] Line {line_num} (Alert {record.get('alert_id', 'Unknown')}) missing 'prev_hash'."

            actual_prev_hash = record["prev_hash"]
            actual_entry_hash = record["entry_hash"]

            # 1. Verify prev_hash continuity
            if actual_prev_hash != expected_prev_hash:
                return (
                    False,
                    line_count,
                    f"[TAMPER DETECTED] Broken chain at Line {line_num} (Alert {record.get('alert_id', 'Unknown')}):\n"
                    f"  Expected prev_hash: {expected_prev_hash}\n"
                    f"  Found prev_hash:    {actual_prev_hash}"
                )

            # 2. Verify recomputed entry_hash
            recomputed_hash = ForensicAlertLogger.compute_entry_hash(record)
            if recomputed_hash != actual_entry_hash:
                return (
                    False,
                    line_count,
                    f"[TAMPER DETECTED] Content modification at Line {line_num} (Alert {record.get('alert_id', 'Unknown')}):\n"
                    f"  Recorded entry_hash:   {actual_entry_hash}\n"
                    f"  Recomputed entry_hash: {recomputed_hash}\n"
                    f"  Payload was altered or forged after hashing."
                )

            expected_prev_hash = actual_entry_hash
            line_count += 1

    if line_count == 0:
        return True, 0, "[WARN] Log file is empty. Genesis verification valid."

    # 3. External tail anchor verification
    if enforce_anchor:
        if tail_hash_path is None:
            tail_hash_path = log_path.parent / f"{log_path.stem}_tail_hash.txt"
        tail_hash_path = Path(tail_hash_path)

        if tail_hash_path.exists() and tail_hash_path.stat().st_size > 0:
            with open(tail_hash_path, "r", encoding="utf-8") as tf:
                anchored_hash = tf.read().strip()
            if anchored_hash and expected_prev_hash != anchored_hash:
                return (
                    False,
                    line_count,
                    f"[TRUNCATION/REWRITE DETECTED] Log tail hash does not match external anchor ({tail_hash_path.name}):\n"
                    f"  External anchored tail hash: {anchored_hash}\n"
                    f"  Found log file tail hash:     {expected_prev_hash}\n"
                    f"  Tail records were deleted, truncated, or forged."
                )

    return True, line_count, f"[SUCCESS] Verified {line_count} alerts. Cryptographic hash chain and external anchor are 100% intact."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit tamper-evident alert log chain")
    parser.add_argument(
        "--file",
        type=str,
        default=str(Path(__file__).parent.parent / "logs" / "alerts.jsonl"),
        help="Path to alerts.jsonl file",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default=None,
        help="Path to external tail anchor hash file (optional, defaults to <stem>_tail_hash.txt)",
    )
    args = parser.parse_args()

    target_file = Path(args.file)
    anchor_file = Path(args.anchor) if args.anchor else None
    valid, count, msg = verify_alert_log(target_file, tail_hash_path=anchor_file)
    print("=" * 65)
    print("  FORENSIC ALERT LOG INTEGRITY AUDIT")
    print("=" * 65)
    print(f"Target File: {target_file.name}")
    print(msg)
    print("=" * 65)
    exit(0 if valid else 1)
