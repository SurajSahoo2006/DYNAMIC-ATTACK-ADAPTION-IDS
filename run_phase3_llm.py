"""
run_phase3_llm.py
=================
Phase 3 - LLM Semantic Orchestrator Demo Runner

Demonstrates semantic triage on 40 representative test flows
(5 flows per 8-class category, selected randomly with fixed seed=42).
Guarantees representation of rare classes: Backdoor_Malware, Recon-PingSweep, and Uploading_Attack.

Uses the B2 classifier for predictions.
Calls local Ollama (Llama-3.2-3B-Instruct) with temperature 0.
Records separate metrics for Ollama vs Fallback, Ollama-only latency,
Ollama JSON validity, peak RAM, and field override events.
Saves outputs to reports/llm_demo_outputs.csv and reports/llm_metrics.json.
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
    ENCODER_34_PATH,
    SELECTED_FEATURES_PATH,
    LLM_OUTPUTS_CSV,
    REPORTS_DIR,
    LLM_DEMO_SAMPLE_SIZE,
    RANDOM_SEED,
)
from orchestrator.llm_orchestrator import LLMSemanticOrchestrator
from predict import predict_flow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_phase3_llm")


def select_balanced_random_flows(
    df_test: pd.DataFrame,
    seed: int = RANDOM_SEED,
    samples_per_family: int = 5,
) -> pd.DataFrame:
    """
    Randomly selects 5 flows per 8-class family with seed=42.
    Ensures rare 34-class attacks (Backdoor_Malware, Recon-PingSweep, Uploading_Attack)
    are included in their respective family selections.
    """
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []

    # Map rare class labels
    # Backdoor_Malware is in Web-based (label_8 == 7, label_34 == 0)
    # Uploading_Attack is in Web-based (label_8 == 7, label_34 == 31)
    # Recon-PingSweep is in Recon (label_8 == 5, label_34 == 28)

    for fam_idx in range(8):
        sub_fam = df_test[df_test["label_8"] == fam_idx]
        chosen_in_fam: list[int] = []

        if fam_idx == 7:  # Web-based: ensure Backdoor_Malware and Uploading_Attack
            sub_backdoor = sub_fam[sub_fam["label_34"] == 0]
            if len(sub_backdoor) > 0:
                chosen_in_fam.append(int(rng.choice(sub_backdoor.index)))

            sub_upload = sub_fam[sub_fam["label_34"] == 31]
            if len(sub_upload) > 0:
                chosen_in_fam.append(int(rng.choice(sub_upload.index)))

            remaining_needed = samples_per_family - len(chosen_in_fam)
            pool = [idx for idx in sub_fam.index if idx not in chosen_in_fam]
            if pool and remaining_needed > 0:
                chosen_in_fam.extend(rng.choice(pool, size=min(len(pool), remaining_needed), replace=False).tolist())

        elif fam_idx == 5:  # Recon: ensure Recon-PingSweep
            sub_pingsweep = sub_fam[sub_fam["label_34"] == 28]
            if len(sub_pingsweep) > 0:
                chosen_in_fam.append(int(rng.choice(sub_pingsweep.index)))

            remaining_needed = samples_per_family - len(chosen_in_fam)
            pool = [idx for idx in sub_fam.index if idx not in chosen_in_fam]
            if pool and remaining_needed > 0:
                chosen_in_fam.extend(rng.choice(pool, size=min(len(pool), remaining_needed), replace=False).tolist())

        else:
            # Standard random sample for other classes
            chosen = rng.choice(sub_fam.index, size=min(len(sub_fam), samples_per_family), replace=False).tolist()
            chosen_in_fam.extend(chosen)

        selected_indices.extend(chosen_in_fam)

    return df_test.loc[selected_indices].copy()


def run_llm_demo(sample_size: int = LLM_DEMO_SAMPLE_SIZE) -> None:
    proc = psutil.Process(os.getpid())
    init_ram_mb = proc.memory_info().rss / (1024.0 * 1024.0)

    print("=" * 70)
    print("  PHASE 3: LLM SEMANTIC ORCHESTRATOR DEMO RUN")
    print("=" * 70)
    print(f"Initial Process RAM: {init_ram_mb:.1f} MB")

    # 1. Load test data and metadata
    df_test = pd.read_parquet(TEST_PARQUET)
    le_8 = joblib.load(ENCODER_8_PATH)
    le_34 = joblib.load(ENCODER_34_PATH)
    classes_8 = list(le_8.classes_)
    classes_34 = list(le_34.classes_)
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_features = json.load(f)

    # 2. Select 40 balanced flows randomly with seed 42
    df_sample = select_balanced_random_flows(df_test, seed=RANDOM_SEED, samples_per_family=5)
    print(f"\n[DATA] Selected {len(df_sample)} balanced test flows (5 per 8-class family, seed=42)")
    rare_counts = {
        "Backdoor_Malware": int((df_sample["label_34"] == 0).sum()),
        "Recon-PingSweep": int((df_sample["label_34"] == 28).sum()),
        "Uploading_Attack": int((df_sample["label_34"] == 31).sum()),
    }
    print(f"[DATA] Rare classes included: {rare_counts}")

    # 3. Run B2 Ensemble Classifier inference
    print("\n[ENSEMBLE] Executing B2 ensemble predictions and confidence scoring ...")
    ens_preds = predict_flow(df_sample[selected_features], target="8class", model_type="b2")
    df_sample["true_class"] = [classes_8[i] for i in df_sample["label_8"]]
    df_sample["true_34_class"] = [classes_34[i] for i in df_sample["label_34"]]
    df_sample["pred_class"] = ens_preds["predicted_class"].values
    df_sample["confidence"] = ens_preds["confidence"].values
    df_sample["severity"] = ens_preds["severity"].values

    # 4. Initialize Orchestrator
    print("\n[ORCHESTRATOR] Initializing LLM Semantic Orchestrator (Ollama / Llama-3.2-3B) ...")
    orchestrator = LLMSemanticOrchestrator()
    unscaled_df = orchestrator.unscaler.unscale(df_sample[selected_features])

    # 5. Process flows
    print(f"\n[ORCHESTRATOR] Generating semantic summaries for {len(df_sample)} flows ...")
    records = []
    ollama_latencies = []
    ollama_valid_json_count = 0
    ollama_count = 0
    fallback_count = 0
    peak_ram_mb = init_ram_mb

    total_llm_start = time.time()
    n_flows = len(df_sample)

    for i, (idx, row) in enumerate(df_sample.iterrows()):
        is_last = (i == n_flows - 1)
        flow_telemetry = unscaled_df.loc[idx].to_dict()
        pred_class = row["pred_class"]
        conf = float(row["confidence"])
        sev = row["severity"]

        res = orchestrator.generate_flow_summary(
            flow_features_unscaled=flow_telemetry,
            attack_class=pred_class,
            confidence=conf,
            severity=sev,
            is_last_call=is_last,
        )

        source = res.get("source", "unknown")
        lat = res.get("latency_sec", 0.0)

        if source == "ollama":
            ollama_count += 1
            ollama_latencies.append(lat)
            # Check JSON schema validity for Ollama outputs
            has_keys = all(k in res and isinstance(res[k], str) and len(res[k].strip()) > 0
                           for k in ["attack_class", "severity", "explanation", "recommended_action"])
            if has_keys:
                ollama_valid_json_count += 1
        else:
            fallback_count += 1

        # Track process RAM
        current_ram = proc.memory_info().rss / (1024.0 * 1024.0)
        peak_ram_mb = max(peak_ram_mb, current_ram)

        conf_band = res.get("confidence_band", "low" if conf < 0.60 else ("medium" if conf < 0.85 else "high"))
        records.append({
            "flow_index": idx,
            "true_8class": row["true_class"],
            "true_34class": row["true_34_class"],
            "ensemble_predicted_class": pred_class,
            "confidence": conf,
            "confidence_band": conf_band,
            "severity": sev,
            "explanation": res["explanation"],
            "recommended_action": res["recommended_action"],
            "source": source,
            "fallback": res.get("fallback", False),
            "overrode_fields": res.get("overrode_fields", False),
            "latency_sec": lat,
        })

        if (i + 1) % 5 == 0 or is_last:
            print(f"  Processed {i + 1:>2}/{n_flows} flows | Last latency: {lat:.1f} s | Source: {source} | RAM: {current_ram:.1f} MB")

    total_llm_time = time.time() - total_llm_start

    # 6. Save results to CSV
    res_df = pd.DataFrame(records)
    LLM_OUTPUTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(LLM_OUTPUTS_CSV, index=False)
    print(f"\n[SAVE] Overwritten demo outputs --> {LLM_OUTPUTS_CSV.name}")

    # 7. Metrics calculation
    avg_sec_ollama = float(np.mean(ollama_latencies)) if ollama_latencies else 0.0
    ollama_json_validity_rate = (ollama_valid_json_count / max(1, ollama_count)) * 100.0 if ollama_count > 0 else 0.0
    field_override_count = orchestrator.model_overrode_fields_count

    low_conf_count = sum(1 for r in records if r["confidence"] < 0.60)
    print("\n" + "=" * 70)
    print("  LLM SEMANTIC ORCHESTRATOR DEMO METRICS")
    print("=" * 70)
    print(f"Total Flows Processed:          {n_flows}")
    print(f"Low-Confidence Flows (< 0.60):  {low_conf_count} ({low_conf_count / n_flows * 100:.1f}%)")
    print(f"Real Ollama Outputs:            {ollama_count} ({ollama_count / n_flows * 100:.1f}%)")
    print(f"Fallback Outputs:               {fallback_count} ({fallback_count / n_flows * 100:.1f}%)")
    print(f"Ollama JSON Validity Rate:      {ollama_json_validity_rate:.1f}% ({ollama_valid_json_count}/{ollama_count})")
    print(f"Ollama Avg Seconds per Flow:    {avg_sec_ollama:.2f} s/flow")
    print(f"Total Triage Generation Time:   {total_llm_time:.1f} s ({total_llm_time / 60.0:.2f} min)")
    print(f"Model Overrode Fields Events:   {field_override_count}")
    print(f"Measured Peak Process RAM:      {peak_ram_mb:.1f} MB ({peak_ram_mb / 1024.0:.2f} GB) [Safety: < 6.0 GB]")
    print("=" * 70)

    # Display 5 sample real Ollama outputs
    ollama_records = [r for r in records if r["source"] == "ollama"]
    print("\nSample Real Ollama Triage Outputs (First 5 Ollama rows):")
    for r in ollama_records[:5]:
        print(f"\n[Flow #{r['flow_index']} | {r['true_34class']}] Ensemble Pred: {r['ensemble_predicted_class']} ({r['confidence']:.2%}) | Severity: {r['severity']}")
        print(f"  Explanation: {r['explanation']}")
        print(f"  Action:      {r['recommended_action']}")

    # Save metrics JSON
    llm_metrics = {
        "flows_processed": n_flows,
        "low_confidence_count": low_conf_count,
        "ollama_count": ollama_count,
        "fallback_count": fallback_count,
        "ollama_json_validity_rate": round(ollama_json_validity_rate, 2),
        "ollama_avg_seconds_per_flow": round(avg_sec_ollama, 2),
        "total_triage_generation_sec": round(total_llm_time, 2),
        "peak_ram_mb": round(peak_ram_mb, 1),
        "peak_ram_gb": round(peak_ram_mb / 1024.0, 2),
        "model_overrode_fields_count": field_override_count,
        "rare_classes_tested": rare_counts,
    }
    with open(REPORTS_DIR / "llm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(llm_metrics, f, indent=2)
    print(f"[SAVE] Overwritten metrics --> {REPORTS_DIR / 'llm_metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Semantic Orchestrator Demo")
    parser.add_argument("--samples", type=int, default=LLM_DEMO_SAMPLE_SIZE, help="Number of test flows to process")
    args = parser.parse_args()

    run_llm_demo(sample_size=args.samples)
