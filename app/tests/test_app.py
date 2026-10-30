"""
app/tests/test_app.py
=====================
Pytest suite verifying the Streamlit Frontend (Phase 14) using AppTest.
Ensures zero exceptions across all 6 views, verified predictions,
validation of missing columns, and zero direct model imports.
"""

from __future__ import annotations

import io
from pathlib import Path
import re

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_SCRIPT = str(PROJECT_ROOT / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def app_runner():
    """Initializes and runs AppTest once."""
    at = AppTest.from_file(APP_SCRIPT)
    at.run(timeout=30)
    return at


def test_pages_render_without_exception(app_runner):
    """Verifies that all 6 views render without any unhandled exceptions."""
    at = app_runner
    assert not at.exception, f"System Overview threw exception: {at.exception}"

    pages = [
        "Live Triage",
        "Batch Analysis",
        "Alert Log",
        "Results",
        "Downloads",
        "System Overview",
    ]

    for p in pages:
        at.sidebar.radio[0].set_value(p).run(timeout=30)
        assert not at.exception, f"Page '{p}' threw exception: {at.exception}"


def test_live_triage_predicts_class(app_runner):
    """Verifies Live Triage prediction output on a demo flow."""
    at = app_runner
    at.sidebar.radio[0].set_value("Live Triage").run(timeout=30)
    assert not at.exception

    # Trigger prediction button
    button_found = False
    for b in at.button:
        if "Run Telemetry Analysis" in b.label:
            b.click().run(timeout=30)
            button_found = True
            break

    assert button_found, "Analyze button not found in Live Triage"
    assert not at.exception, f"Prediction execution threw exception: {at.exception}"
    metric_labels = [m.label for m in at.metric]
    assert "Predicted Family" in metric_labels
    assert "Raw Confidence" in metric_labels
    assert "Threat Severity" in metric_labels


def test_batch_analysis_rejects_missing_columns():
    """Verifies that a CSV missing required feature columns is rejected with the column name."""
    from ids_system.pipeline import IDSPipeline

    pipeline = IDSPipeline()
    dropped_col = pipeline.features[0]

    # Create dummy dataframe with one missing feature
    partial_features = [f for f in pipeline.features if f != dropped_col]
    bad_df = pd.DataFrame([{f: 1.0 for f in partial_features}])

    with pytest.raises(ValueError) as excinfo:
        pipeline.analyze_batch(bad_df)

    err_msg = str(excinfo.value)
    assert dropped_col in err_msg, f"Expected '{dropped_col}' in error message: {err_msg}"


def test_no_direct_model_imports():
    """Ensures app/*.py contains no direct model loading or library imports."""
    banned_keywords = ["joblib.load", "xgboost", "torch.load", "sklearn"]
    app_dir = PROJECT_ROOT / "app"

    for py_file in app_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for kw in banned_keywords:
            assert kw not in text, f"Forbidden direct import/usage '{kw}' detected in {py_file.name}"
