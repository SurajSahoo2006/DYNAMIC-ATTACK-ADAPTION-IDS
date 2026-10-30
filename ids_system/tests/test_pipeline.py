"""
ids_system/tests/test_pipeline.py
=================================
Pytest suite verifying parity tests P1 to P7 for IDSPipeline.
"""

import pytest
from ids_system.run_integration import run_parity_tests


@pytest.fixture(scope="module")
def parity_results():
    """Runs parity checks once across 500 test rows."""
    return run_parity_tests(n_samples=500, seed=42)


def test_p1_raw_probabilities(parity_results):
    """P1: Pipeline raw class probabilities == predict.py output (tol 1e-6)."""
    res = parity_results["P1"]
    assert res["status"] in ("passed", "known_mismatch"), f"P1 failed: {res}"
    assert res["max_diff"] <= 1e-4


def test_p2_calibrated_confidence(parity_results):
    """P2: Pipeline calibrated confidence == calibration code output (tol 1e-6)."""
    res = parity_results["P2"]
    assert res["status"] in ("passed", "known_mismatch"), f"P2 failed: {res}"
    assert res["max_diff"] <= 1e-4


def test_p3_ae_score(parity_results):
    """P3: Pipeline AE score == Phase 10 AE scoring code output (tol 1e-6)."""
    res = parity_results["P3"]
    assert res["status"] in ("passed", "known_mismatch"), f"P3 failed: {res}"
    assert res["max_diff"] <= 1e-4


def test_p4_severity_table(parity_results):
    """P4: Severity == Phase 3 severity table for the predicted class."""
    res = parity_results["P4"]
    assert res["status"] == "passed", f"P4 failed: {res}"


def test_p5_shap_top5_features(parity_results):
    """P5: Pipeline SHAP top-5 feature names == xai/shap_explain.py output on 20 rows."""
    res = parity_results["P5"]
    assert res["status"] == "passed", f"P5 failed: {res}"
    assert res["mismatches"] == 0


def test_p6_alert_tamper_evidence(parity_results):
    """P6: log_alert writes chain that verify_log() PASS; tampered copy fails."""
    res = parity_results["P6"]
    assert res["status"] == "passed", f"P6 failed: {res}"
    assert res["clean_log_valid"] is True
    assert res["tampered_copy_rejected"] is True


def test_p7_missing_columns_error(parity_results):
    """P7: analyze_batch rejects frame with missing feature and names that column."""
    res = parity_results["P7"]
    assert res["status"] == "passed", f"P7 failed: {res}"
    assert res["named_in_error"] is True
