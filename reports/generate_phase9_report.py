#!/usr/bin/env python3
"""
reports/generate_phase9_report.py - Generate Phase 9 summary report from federated JSON artifacts.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARTITION_META_PATH = PROJECT_ROOT / "federated" / "data" / "partition_metadata.json"
LOCAL_RESULTS_PATH = PROJECT_ROOT / "federated" / "results" / "local_results.json"
FEDAVG_RESULTS_PATH = PROJECT_ROOT / "federated" / "results" / "fedavg_results.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase9_summary.md"


def main():
    if not PARTITION_META_PATH.exists():
        raise FileNotFoundError(f"Missing {PARTITION_META_PATH}")
    if not LOCAL_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing {LOCAL_RESULTS_PATH}")
    if not FEDAVG_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing {FEDAVG_RESULTS_PATH}")

    with open(PARTITION_META_PATH, "r", encoding="utf-8") as f:
        partition_data = json.load(f)

    with open(LOCAL_RESULTS_PATH, "r", encoding="utf-8") as f:
        local_data = json.load(f)

    with open(FEDAVG_RESULTS_PATH, "r", encoding="utf-8") as f:
        fedavg_data = json.load(f)

    # 1. Setup & Partition Table
    client_partition_rows = []
    for c in partition_data["clients"]:
        cid = c["client_id"]
        dom = c["dominant_class"]
        rows = c["n_rows"]
        client_partition_rows.append(f"| Client {cid} | {dom} | {rows:,} |")

    # 2. Performance Table
    dominant_map = {c["client_id"]: c["dominant_class"] for c in partition_data["clients"]}
    perf_rows = []
    for cm in fedavg_data["client_models"]:
        cid = cm["client_id"]
        dom = dominant_map.get(cid, "Client")
        acc = cm["test_accuracy"] * 100.0
        f1 = cm["test_macro_f1"] * 100.0
        perf_rows.append(f"| Client {cid} ({dom}) | {acc:.2f}% | {f1:.2f}% |")

    g_acc = fedavg_data["global_federated_model"]["test_accuracy"] * 100.0
    g_f1 = fedavg_data["global_federated_model"]["test_macro_f1"] * 100.0
    perf_rows.append(f"| Global FedAvg | {g_acc:.2f}% | {g_f1:.2f}% |")

    c_acc = fedavg_data["centralized_baseline"]["test_accuracy"] * 100.0
    c_f1 = fedavg_data["centralized_baseline"]["test_macro_f1"] * 100.0
    perf_rows.append(f"| Centralized (Phase 3) | {c_acc:.2f}% | {c_f1:.2f}% |")

    gap = c_acc - g_acc
    if gap <= 5.0:
        key_finding = (
            f"Global FedAvg achieves {g_acc:.2f}% accuracy vs {c_acc:.2f}% centralized "
            f"(gap: {gap:.2f}%), demonstrating that collaborative training without "
            f"sharing raw data retains most predictive performance."
        )
    else:
        key_finding = (
            f"Global FedAvg achieves {g_acc:.2f}% accuracy vs {c_acc:.2f}% centralized "
            f"(gap: {gap:.2f}%)."
        )

    lines = [
        "# Phase 9: Federated Learning Simulation",
        "",
        "## Setup",
        "- 5 simulated IoT gateway clients (non-IID class distribution)",
        "- Max 8,000 rows per client; no data shared between clients",
        "- Aggregation: FedAvg (tree pooling)",
        "",
        "## Client Partition Summary",
        "| Client | Dominant Class | Rows |",
        "|:---|:---|---:|",
    ]
    lines.extend(client_partition_rows)
    lines.extend([
        "",
        "## Local vs Global Performance on Full Test Set",
        "| Model | Test Accuracy | Macro F1 |",
        "|:---|---:|---:|",
    ])
    lines.extend(perf_rows)
    lines.extend([
        "",
        "## Key Finding",
        key_finding,
        "",
        "## Limitations",
        "- Tree pooling is not true gradient-based FedAvg; it is a simulation approximation suitable for tree ensembles",
        "- Clients trained for one round only (no multi-round convergence)",
        "- Non-IID partition is synthetic, not from real distributed IoT deployments",
        "- No differential privacy or secure aggregation applied",
        "",
    ])

    report_content = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[REPORT] Phase 9 summary written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
