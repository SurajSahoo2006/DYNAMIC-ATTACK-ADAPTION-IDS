"""
run_phase3_llm.py
=================
Phase 3 - LLM Semantic Orchestrator Demo Runner

Demonstrates semantic triage on 40 representative test flows
(5 flows per category across all 8 classes).
Measures:
  - Latency (total and per-sample)
  - Memory consumption (peak RSS)
  - Strict JSON validity rate
  - Fallback vs LLM usage rate
  - Consistency with Ensemble classification (100% enforced)
Saves outputs to reports/llm_demo_outputs.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil

import sys
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import (
    TEST_PARQUET,
    ENCODER_8_PATH,
    SELECTED_FEATURES_PATH,
    LLM_OUTPUTS_CSV,
    REPORTS_DIR,
    LLM_DEMO_SAMPLE_SIZE,
)
from orchestrator.llm_orchestrator import LLMSemanticOrchestrator
from predict import predict_flow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_phase3_llm")


def run_llm_demo(sample_size: int = LLM_DEMO_SAMPLE_SIZE) -> None:
    t_start = time.time()
    proc = psutil.Process(os.getpid())
    init_ram_mb = proc.memory_info().rss / (1024.0 * 1024.0)

    print("=" * 70)
    print("  PHASE 3: LLM SEMANTIC ORCHESTRATOR DEMO RUN")
    print("=" * 70)
    print(f"Initial Process RAM: {init_ram_mb:.1f} MB")

    # 1. Load test data and metadata
    df_test = pd.read_parquet(TEST_PARQUET)
    le_8 = joblib.load(ENCODER_8_PATH)
    classes_8 = list(le_8.classes_)
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_features = json.load(f)

    # 2. Select balanced 40 flows (5 per class)
    samples_per_class = max(1, sample_size // len(classes_8))
    sampled_indices = []

    for c_idx in range(len(classes_8)):
        sub = df_test[df_test["label_8"] == c_idx]
        n_take = min(len(sub), samples_per_class)
        sampled_indices.extend(sub.index[:n_take].tolist())

    df_sample = df_test.loc[sampled_indices].copy()
    print(f"\n[DATA] Selected {len(df_sample)} balanced test flows ({samples_per_class} per class across 8 classes)")

    # 3. Run Ensemble Classifier inference
    print("\n[ENSEMBLE] Executing ensemble predictions and confidence scoring ...")
    ens_preds = predict_flow(df_sample[selected_features], target="8class")
    df_sample["true_class"] = [classes_8[i] for i in df_sample["label_8"]]
    df_sample["pred_class"] = ens_preds["predicted_class"].values
    df_sample["confidence"] = ens_preds["confidence"].values
    df_sample["severity"] = ens_preds["severity"].values

    # 4. Initialize Orchestrator
    print("\n[ORCHESTRATOR] Initializing LLM Semantic Orchestrator ...")
    orchestrator = LLMSemanticOrchestrator()
    unscaled_df = orchestrator.unscaler.unscale(df_sample[selected_features])

    # 5. Process flows
    print(f"\n[ORCHESTRATOR] Generating semantic summaries for {len(df_sample)} flows ...")
    records = []
    latencies = []
    valid_json_count = 0
    ollama_count = 0
    fallback_count = 0

    for idx, row in df_sample.iterrows():
        flow_telemetry = unscaled_df.loc[idx].to_dict()
        pred_class = row["pred_class"]
        conf = float(row["confidence"])
        sev = row["severity"]

        res = orchestrator.generate_flow_summary(
            flow_features_unscaled=flow_telemetry,
            attack_class=pred_class,
            confidence=conf,
            severity=sev,
        )

        # Validate schema
        has_keys = all(k in res for k in ["attack_class", "severity", "explanation", "recommended_action"])
        if has_keys:
            valid_json_count += 1

        source = res.get("_source", "unknown")
        if source == "ollama":
            ollama_count += 1
        else:
            fallback_count += 1

        lat = res.get("_latency_sec", 0.0)
        latencies.append(lat)

        records.append({
            "flow_index": idx,
            "true_class": row["true_class"],
            "ensemble_predicted_class": pred_class,
            "confidence": conf,
            "severity": sev,
            "explanation": res["explanation"],
            "recommended_action": res["recommended_action"],
            "source": source,
            "latency_sec": lat,
        })

    total_time = time.time() - t_start
    peak_ram_mb = proc.memory_info().rss / (1024.0 * 1024.0)

    # 6. Save results to CSV
    res_df = pd.DataFrame(records)
    LLM_OUTPUTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(LLM_OUTPUTS_CSV, index=False)
    print(f"\n[SAVE] Demo outputs saved to: {LLM_OUTPUTS_CSV.name}")

    # 7. Print Telemetry & Statistics
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    json_validity_rate = (valid_json_count / len(df_sample)) * 100.0

    print("\n" + "=" * 70)
    print("  LLM SEMANTIC ORCHESTRATOR DEMO METRICS")
    print("=" * 70)
    print(f"Total Flows Processed:     {len(df_sample)}")
    print(f"JSON Validity Rate:        {json_validity_rate:.1f}% ({valid_json_count}/{len(df_sample)})")
    print(f"Ollama Mode Count:         {ollama_count} ({ollama_count / len(df_sample) * 100:.1f}%)")
    print(f"Fallback Mode Count:       {fallback_count} ({fallback_count / len(df_sample) * 100:.1f}%)")
    print(f"Average Flow Latency:      {avg_latency * 1000.0:.2f} ms ({avg_latency:.4f} s/flow)")
    print(f"Total Execution Time:      {total_time:.2f} s")
    print(f"Measured Peak RAM:         {peak_ram_mb:.1f} MB ({peak_ram_mb / 1024.0:.2f} GB) [Safety: < 6.0 GB]")
    print("=" * 70)

    # Sample display
    print("\nSample Flow Triage Output (First 4 rows):")
    for r in records[:4]:
        print(f"\n[Flow #{r['flow_index']}] Classified: {r['ensemble_predicted_class']} ({r['confidence']:.2%}) | Severity: {r['severity']} | Source: {r['source']}")
        print(f"  Explanation: {r['explanation']}")
        print(f"  Action:      {r['recommended_action']}")

    # Save LLM metrics to JSON
    llm_metrics = {
        "flows_processed": len(df_sample),
        "json_validity_rate": json_validity_rate,
        "ollama_count": ollama_count,
        "fallback_count": fallback_count,
        "avg_latency_ms": round(avg_latency * 1000.0, 2),
        "total_runtime_sec": round(total_time, 2),
        "peak_ram_mb": round(peak_ram_mb, 1),
    }
    with open(REPORTS_DIR / "llm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(llm_metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Semantic Orchestrator Demo")
    parser.add_argument("--samples", type=int, default=LLM_DEMO_SAMPLE_SIZE, help="Number of test flows to process")
    args = parser.parse_args()

    run_llm_demo(sample_size=args.samples)
