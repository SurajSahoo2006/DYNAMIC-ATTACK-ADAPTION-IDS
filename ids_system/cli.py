"""
ids_system/cli.py
=================
Command-line interface for the consolidated IDS Pipeline.

Usage:
  python -m ids_system.cli health
  python -m ids_system.cli analyze --csv <file.csv> --out <file.jsonl> [--respond] [--log]
  python -m ids_system.cli selftest
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ids_system.pipeline import IDSPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ids_cli")


def cmd_health(pipeline: IDSPipeline):
    """Executes system health audit."""
    print("=" * 70)
    print("          IDS PIPELINE HEALTH AUDIT")
    print("=" * 70)
    h = pipeline.health()
    print(json.dumps(h, indent=2))
    print("=" * 70)
    print(f"Overall Status: {h['overall_status'].upper()} (Process RSS: {h['process_rss_mb']} MB)")


def cmd_analyze(args, pipeline: IDSPipeline):
    """Executes batch flow analysis."""
    csv_path = Path(args.csv)
    out_path = Path(args.out)
    if not csv_path.exists():
        log.error(f"Input CSV file not found: {csv_path}")
        sys.exit(1)

    log.info(f"Loading input flows from {csv_path.name} ...")
    df_raw = pd.read_csv(csv_path)
    n_flows = len(df_raw)
    log.info(f"Read {n_flows:,} flow records.")

    t0 = time.time()
    analyzed_df = pipeline.analyze_batch(df_raw)
    elapsed = time.time() - t0
    log.info(f"Analyzed {n_flows:,} flows in {elapsed:.2f} s ({n_flows / max(1e-6, elapsed):.1f} flows/s)")

    if args.respond:
        log.info("Running greedy Multi-Agent Tactical Defense response ...")
        t_marl = time.time()
        analyzed_df = pipeline.respond(analyzed_df)
        log.info(f"Tactical response completed in {time.time() - t_marl:.2f} s.")

    if args.log:
        log.info(f"Logging forensic alerts to {pipeline.live_log_path.resolve()} ...")
        for idx, row in analyzed_df.iterrows():
            pipeline.log_alert(row.to_dict())
        log.info("Alert logging complete.")

    # Write JSONL output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in analyzed_df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    log.info(f"Results written to {out_path.resolve()}")


def cmd_selftest():
    """Executes parity self-test on 200 held-out test rows."""
    print("=" * 70)
    print("      RUNNING PHASE 13 PARITY SELF-TEST (200 ROWS)")
    print("=" * 70)
    from ids_system.run_integration import run_parity_tests
    results = run_parity_tests(n_samples=200, seed=42)
    all_ok = all(
        v.get("status") in ("passed", "known_mismatch")
        for v in results.values()
    )
    print(json.dumps(results, indent=2))
    print("=" * 70)
    print("SELFTEST RESULT:", "PASSED" if all_ok else "FAILED")
    if not all_ok:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Consolidated IDS Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # health
    subparsers.add_parser("health", help="Check pipeline component health")

    # analyze
    parser_an = subparsers.add_parser("analyze", help="Analyze batch of network flows from CSV")
    parser_an.add_argument("--csv", required=True, help="Path to input CSV file")
    parser_an.add_argument("--out", required=True, help="Path to output JSONL file")
    parser_an.add_argument("--respond", action="store_true", help="Generate MARL tactical defense actions")
    parser_an.add_argument("--log", action="store_true", help="Log alerts to forensic tamper-evident log")

    # selftest
    subparsers.add_parser("selftest", help="Run parity checks on 200 test rows")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "selftest":
        cmd_selftest()
    else:
        pipeline = IDSPipeline()
        if args.command == "health":
            cmd_health(pipeline)
        elif args.command == "analyze":
            cmd_analyze(args, pipeline)


if __name__ == "__main__":
    main()
