"""
export/elk_export.py
====================
Stage 7 Part 2: Elastic Common Schema (ECS) ndjson Alert Exporter.

Reads logs/alerts_phase5.jsonl and transforms each record into Elastic Common Schema (ECS)
compliant JSON structures formatted as ndjson for direct ingestion into Elasticsearch / Kibana.

Target Output:
  - logs/alerts_phase5_ecs.ndjson
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import sys
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config_marl import (
    ALERTS_PHASE5_ECS_NDJSON_PATH,
    ALERTS_PHASE5_JSONL_PATH,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
log = logging.getLogger("elk_export")

SEVERITY_NUMERIC_MAP = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4,
}


def convert_record_to_ecs(record: Dict[str, Any]) -> Dict[str, Any]:
    """Transforms an internal forensic alert record to Elastic Common Schema (ECS)."""
    sev_str = record.get("severity", "Medium")
    sev_num = SEVERITY_NUMERIC_MAP.get(sev_str, 2)
    pred_cls = record.get("predicted_class", "Unknown")

    ecs_doc = {
        "@timestamp": record.get("timestamp_utc"),
        "event": {
            "kind": "alert",
            "category": ["network", "intrusion_detection"],
            "type": ["info", "indicator"],
            "severity": sev_num,
            "severity_name": sev_str,
            "id": record.get("alert_id"),
            "action": record.get("recommended_action", "monitor"),
        },
        "rule": {
            "name": f"IDS-ML-{pred_cls}-Detection",
            "category": "Threat Adaptation",
            "version": record.get("model_version", "B2-Ensemble-8class"),
            "description": f"Automated B2 Ensemble and MARL detection for {pred_cls}",
        },
        "threat": {
            "indicator": {
                "name": pred_cls,
                "type": "attack_flow",
            }
        },
        "labels": {
            "flow_index": record.get("flow_index"),
            "predicted_class": pred_cls,
            "true_class": record.get("true_class", "Unknown"),
            "confidence": record.get("confidence"),
            "confidence_band": record.get("confidence_band"),
            "is_low_confidence": record.get("is_low_confidence", False),
            "is_adversarial": record.get("adversarial", False),
            "marl_triage_action": record.get("marl_triage_action", "none"),
            "marl_containment_action": record.get("marl_containment_action", "none"),
            "marl_escalation_action": record.get("marl_escalation_action", "none"),
            "marl_policy_version": record.get("marl_policy_version", "none"),
            "entry_hash": record.get("entry_hash"),
            "prev_hash": record.get("prev_hash"),
        },
        "message": record.get("explanation", ""),
        "tags": ["ciciot2023", "marl_tactical_defense", "ids_phase5"],
    }

    # Include SHAP attributes if present
    if "top_shap_values" in record:
        ecs_doc["labels"]["top_shap_features"] = record["top_shap_values"]

    return ecs_doc


def export_alerts_to_ecs(
    input_path: Path = ALERTS_PHASE5_JSONL_PATH,
    output_path: Path = ALERTS_PHASE5_ECS_NDJSON_PATH,
) -> int:
    log.info(f"Exporting alerts from {input_path} to ECS ndjson: {output_path}...")
    if not input_path.exists():
        raise FileNotFoundError(f"Input alert log not found at {input_path}")

    exported_count = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            ecs_entry = convert_record_to_ecs(rec)
            fout.write(json.dumps(ecs_entry) + "\n")
            exported_count += 1

    log.info(f"Successfully exported {exported_count} ECS records to {output_path}")
    return exported_count


if __name__ == "__main__":
    export_alerts_to_ecs()
