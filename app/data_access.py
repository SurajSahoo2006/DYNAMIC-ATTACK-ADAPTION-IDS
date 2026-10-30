"""
app/data_access.py
==================
Phase 14: Read-only Data Access Layer for Streamlit Frontend.

Provides cached access to artifacts and evaluation JSON files across
Phases 1 through 13.
Does not import machine learning model binaries or evaluation engines directly.
Every read is wrapped in error handling and returns None if files are missing.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_json_file(relative_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Safely reads a JSON file; returns None if missing or unreadable."""
    abs_path = PROJECT_ROOT / relative_path
    if not abs_path.exists():
        return None
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_jsonl_file(relative_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Safely reads a JSONL file line-by-line; returns empty list if missing."""
    abs_path = PROJECT_ROOT / relative_path
    if not abs_path.exists():
        return []
    records = []
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    records.append(json.loads(s))
    except Exception:
        return []
    return records


@st.cache_data
def load_manifest() -> Optional[Dict[str, Any]]:
    return _read_json_file("artifacts/model_manifest.json")


@st.cache_data
def load_integration_report() -> Optional[Dict[str, Any]]:
    return _read_json_file("ids_system/results/integration_report.json")


@st.cache_data
def load_adversarial_results() -> Optional[Dict[str, Any]]:
    return _read_json_file("adversarial/results/adversarial_summary.json")


@st.cache_data
def load_xai_results() -> Optional[Dict[str, Any]]:
    return _read_json_file("xai/results/shap_summary.json")


@st.cache_data
def load_marl_results() -> Optional[Dict[str, Any]]:
    return _read_json_file("marl/results/reward_settings_comparison.json")


@st.cache_data
def load_drift_results() -> Optional[Dict[str, Any]]:
    det = _read_json_file("drift/drift_detection_results.json")
    ret = _read_json_file("drift/retrain_results.json")
    meta = _read_json_file("drift/drift_metadata.json")
    if not det and not ret:
        return None
    return {
        "detection": det,
        "retrain": ret,
        "metadata": meta,
    }


@st.cache_data
def load_federated_results() -> Optional[Dict[str, Any]]:
    fedavg = _read_json_file("federated/results/fedavg_results.json")
    local = _read_json_file("federated/results/local_results.json")
    part = _read_json_file("federated/data/partition_metadata.json")
    if not fedavg and not local:
        return None
    return {
        "fedavg": fedavg,
        "local": local,
        "partition": part,
    }


@st.cache_data
def load_zeroday_results() -> Optional[Dict[str, Any]]:
    ae = _read_json_file("zeroday/results/ae_training.json")
    openset = _read_json_file("zeroday/results/openset_summary.json")
    if not ae and not openset:
        return None
    return {
        "ae_training": ae,
        "openset_summary": openset,
    }


@st.cache_data
def load_edge_results() -> Optional[Dict[str, Any]]:
    base = _read_json_file("edge/results/baseline.json")
    comp = _read_json_file("edge/results/final_comparison.json")
    var = _read_json_file("edge/results/variants_summary.json")
    if not comp and not base:
        return None
    return {
        "baseline": base,
        "final_comparison": comp,
        "variants_summary": var,
    }


@st.cache_data
def load_calibration_results() -> Optional[Dict[str, Any]]:
    fit = _read_json_file("calibration/results/fit.json")
    sel = _read_json_file("calibration/results/selective.json")
    summary = _read_json_file("calibration/results/openset_calibrated_summary.json")
    raw_m = _read_json_file("calibration/results/raw_metrics.json")
    if not fit and not sel and not summary:
        return None
    return {
        "fit": fit,
        "selective": sel,
        "openset_summary": summary,
        "raw_metrics": raw_m,
    }


@st.cache_data
def load_alerts(source_name: str = "live") -> List[Dict[str, Any]]:
    """Loads alert records from logs/alerts_live.jsonl or logs/alerts_phase5.jsonl."""
    if source_name == "live":
        return _read_jsonl_file("logs/alerts_live.jsonl")
    elif source_name == "phase5":
        return _read_jsonl_file("logs/alerts_phase5.jsonl")
    return []


@st.cache_data
def load_demo_flows() -> Optional[pd.DataFrame]:
    """Loads demo flows from artifacts/adversarial_demo_flows.parquet or test set."""
    p_demo = PROJECT_ROOT / "artifacts" / "adversarial_demo_flows.parquet"
    if p_demo.exists():
        try:
            return pd.read_parquet(p_demo)
        except Exception:
            pass

    p_test = PROJECT_ROOT / "data" / "processed" / "test.parquet"
    if p_test.exists():
        try:
            return pd.read_parquet(p_test).head(100)
        except Exception:
            pass

    return None


@st.cache_data
def load_architecture_table() -> str:
    """Extracts the 8-stage architecture table from reports/phase13_summary.md."""
    p13_md = PROJECT_ROOT / "reports" / "phase13_summary.md"
    if not p13_md.exists():
        return "Architecture specification unavailable (reports/phase13_summary.md missing)."
    text = p13_md.read_text(encoding="utf-8")
    m = re.search(r"(\| Stage # \|.*?\n\n)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return "Architecture table unavailable."


@st.cache_data
def load_limitations(phase: Union[int, str]) -> str:
    """Extracts the Limitations section of reports/phaseN_summary.md."""
    phase_str = str(phase)
    report_file = PROJECT_ROOT / "reports" / f"phase{phase_str}_summary.md"
    if not report_file.exists():
        return f"Limitations documentation for Phase {phase_str} is not available."

    try:
        text = report_file.read_text(encoding="utf-8")
        m = re.search(
            r"#+\s*(?:\d+\.\s*)?(?:Honest\s+)?Limitations.*?\n(.*?)(?=\n#+ |\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        return "Limitations section header not found in report."
    except Exception as ex:
        return f"Error loading limitations: {ex}"


def load_downloadable_files() -> Dict[str, Path]:
    """Discovers available report, artifact, and data files for download."""
    candidates = [
        ("Final Report (Markdown)", PROJECT_ROOT / "reports" / "final_report.md"),
        ("Phase 13 Summary (Markdown)", PROJECT_ROOT / "reports" / "phase13_summary.md"),
        ("Model Manifest (JSON)", PROJECT_ROOT / "artifacts" / "model_manifest.json"),
        ("Integration Report (JSON)", PROJECT_ROOT / "ids_system" / "results" / "integration_report.json"),
    ]

    # Processed CSVs if present
    data_dir = PROJECT_ROOT / "data" / "processed"
    if data_dir.exists():
        for csv_file in data_dir.glob("*.csv"):
            candidates.append((f"Processed Data: {csv_file.name}", csv_file))

    available = {label: path for label, path in candidates if path.exists()}
    return available
