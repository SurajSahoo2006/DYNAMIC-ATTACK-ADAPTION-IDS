"""
alerts/alert_logger.py
======================
Phase 4 - Cryptographic Tamper-Evident Forensic Alert Logger

Writes flattened, ELK/Kibana-ready forensic alert records to logs/alerts.jsonl.
Features:
  - Cryptographic SHA-256 hash chaining (prev_hash -> entry_hash)
  - ISO 8601 UTC processing timestamps (with explicit note that CICIoT2023 lacks flow timestamps)
  - Flattened fields for easy Elasticsearch / OpenSearch indexing
  - Full model telemetry, low-confidence flag, top SHAP features, and adversarial indicator
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

GENESIS_HASH = "0" * 64


class ForensicAlertLogger:
    """
    Appends forensic alert entries to a tamper-evident JSONL audit log.
    Each entry is cryptographically linked to the previous entry via SHA-256.
    """

    def __init__(self, log_path: Optional[Path] = None, tail_hash_path: Optional[Path] = None):
        if log_path is None:
            log_path = Path(__file__).parent.parent / "logs" / "alerts.jsonl"
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if tail_hash_path is None:
            tail_hash_path = self.log_path.parent / f"{self.log_path.stem}_tail_hash.txt"
        self.tail_hash_path = Path(tail_hash_path)
        self.last_hash = self._get_tail_hash()

    def _get_tail_hash(self) -> str:
        """Reads the last entry hash from the existing log, or returns GENESIS_HASH."""
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return GENESIS_HASH

        last_line = ""
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    last_line = line_str

        if not last_line:
            return GENESIS_HASH

        try:
            record = json.loads(last_line)
            return record.get("entry_hash", GENESIS_HASH)
        except Exception:
            return GENESIS_HASH

    @staticmethod
    def compute_entry_hash(entry_dict: Dict[str, Any]) -> str:
        """
        Computes SHA-256 hash over deterministic JSON representation of entry excluding entry_hash.
        """
        hashable = {k: v for k, v in entry_dict.items() if k != "entry_hash"}
        canonical_str = json.dumps(hashable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def log_alert(
        self,
        flow_index: int,
        predicted_class: str,
        confidence: float,
        severity: str,
        explanation: str,
        recommended_action: str,
        source: str = "ollama",
        model_version: str = "B2-Ensemble-8class",
        top_shap_features: Optional[List[Dict[str, Any]]] = None,
        adversarial: bool = False,
        true_class: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Creates, hashes, and persists a single forensic alert record.
        """
        conf_band = "low" if confidence < 0.60 else ("medium" if confidence < 0.85 else "high")
        now_utc = datetime.now(timezone.utc).isoformat()

        # Format and separate SHAP attribution values and raw flow telemetry values
        shap_summary_parts = []
        clean_shap_features = []
        top_shap_values: Dict[str, float] = {}
        top_raw_feature_values: Dict[str, float] = {}

        if top_shap_features:
            for item in top_shap_features:
                feat = item.get("feature", "unknown")
                shap_val = round(float(item.get("shap_value", 0.0)), 4)
                raw_val = round(float(item.get("raw_value", item.get("unscaled_value", 0.0))), 4)
                dir_label = str(item.get("direction", ""))
                if "decreases" in dir_label:
                    impact = "decreases_prediction"
                elif "increases" in dir_label:
                    impact = "increases_prediction"
                else:
                    impact = "positive_attribution" if shap_val >= 0 else "negative_attribution"

                clean_shap_features.append({
                    "feature": feat,
                    "shap_value": shap_val,
                    "raw_value": raw_val,
                    "impact": impact,
                })
                top_shap_values[feat] = shap_val
                top_raw_feature_values[feat] = raw_val
                shap_summary_parts.append(f"{feat} [SHAP: {shap_val:+.4f}, Raw: {raw_val}]")

        shap_summary_str = ", ".join(shap_summary_parts)

        entry = {
            "alert_id": f"ALT-{uuid.uuid4().hex[:12].upper()}",
            "timestamp_utc": now_utc,
            "timestamp_note": "Processing time (UTC). CICIoT2023 raw benchmark lacks ground-truth packet timestamps.",
            "flow_index": int(flow_index),
            "predicted_class": str(predicted_class),
            "true_class": str(true_class) if true_class else "Unknown",
            "confidence": round(float(confidence), 4),
            "confidence_band": conf_band,
            "is_low_confidence": bool(confidence < 0.60),
            "severity": str(severity),
            "model_version": str(model_version),
            "adversarial": bool(adversarial),
            "source": str(source),
            "top_shap_summary": shap_summary_str,
            "top_shap_features": clean_shap_features,
            "top_shap_values": top_shap_values,
            "top_raw_feature_values": top_raw_feature_values,
            "explanation": str(explanation),
            "recommended_action": str(recommended_action),
            "prev_hash": self.last_hash,
        }

        if custom_metadata:
            entry.update(custom_metadata)

        # Compute tamper-evident hash
        entry_hash = self.compute_entry_hash(entry)
        entry["entry_hash"] = entry_hash

        # Write JSON line atomically
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        self.last_hash = entry_hash

        # Persist external tail hash anchor to detect tail truncation / full rewrites
        try:
            with open(self.tail_hash_path, "w", encoding="utf-8") as tf:
                tf.write(entry_hash.strip() + "\n")
        except Exception:
            pass

        return entry
