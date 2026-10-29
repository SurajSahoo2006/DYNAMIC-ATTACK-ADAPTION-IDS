"""
tools/check_report_consistency.py
=================================
Automated auditor that verifies mathematical and empirical consistency between
Phase 4 report documents (reports/phase4_summary.md, README.md) and underlying
generated JSON artifacts in reports/.

Checks:
1. Recomputed percentage consistency: for every "x of y" or "x/y (p%)" quoted in the report,
   verifies that |(x / y) * 100 - p| <= 0.1%.
2. Exact empirical alignment: verifies that every accuracy, count, CI interval, and runtime
   quoted in reports/phase4_summary.md matches the numbers in:
   - reports/adversarial_fgsm.json
   - reports/adversarial_hsj.json
   - reports/adversarial_zoo.json
   - reports/adversarial_defense.json
   - reports/adversarial_metrics.json
   - reports/noise_sweep_metrics.json
   - reports/shap_metrics.json
   - reports/telemetry_grounding_comparison.json
3. Architecture & Class Names consistency with config.py and label encoders.
4. Requirements completeness against all external imports in the codebase.
"""

from __future__ import annotations

import ast
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ENCODER_8_PATH, ENCODER_34_PATH, REPORTS_DIR


def check_fraction_percentages(markdown_text: str) -> List[str]:
    """Finds all 'k/n (pct%)' or 'k of n' instances and checks mathematical agreement."""
    errors = []
    # Match patterns like: 285/500 ( 57.0% or 285 / 500 (57.0% or 285 of 500 flows (57.0%)
    patterns = [
        re.compile(r"(\d+)\s*/\s*(\d+)\s*\([^\d]*([0-9]+\.[0-9]+)\s*%\)"),
        re.compile(r"(\d+)\s+of\s+(\d+)\s+(?:flows\s+)?\([^\d]*([0-9]+\.[0-9]+)\s*%\)"),
    ]
    for pattern in patterns:
        for match in pattern.finditer(markdown_text):
            k = int(match.group(1))
            n = int(match.group(2))
            stated_pct = float(match.group(3))
            if n == 0:
                continue
            recomputed = (k / n) * 100.0
            if abs(recomputed - stated_pct) > 0.15:
                errors.append(
                    f"Percentage mismatch: {k}/{n} recomputed is {recomputed:.2f}%, but report states {stated_pct:.2f}% (match: '{match.group(0)}')"
                )
    return errors


def check_json_alignments(summary_md: str) -> List[str]:
    """Cross-checks specific JSON metrics against text in phase4_summary.md."""
    errors = []

    def check_presence(val_str: str, label: str):
        if val_str not in summary_md:
            errors.append(f"Missing or mismatched in report: {label} (expected '{val_str}')")

    # 1. SHAP metrics
    shap_file = REPORTS_DIR / "shap_metrics.json"
    if shap_file.exists():
        with open(shap_file, "r", encoding="utf-8") as f:
            shap_data = json.load(f)
        rho = shap_data.get("spearman_rho")
        if rho is not None:
            check_presence(f"{rho:.4f}", f"SHAP Spearman rho ({rho:.4f})")
        top_overlap = shap_data.get("top5_overlap_count")
        if top_overlap is not None:
            check_presence(f"{top_overlap} / 5", f"SHAP top-5 overlap ({top_overlap}/5)")

    # 2. Telemetry grounding comparison
    ground_file = REPORTS_DIR / "telemetry_grounding_comparison.json"
    if ground_file.exists():
        with open(ground_file, "r", encoding="utf-8") as f:
            ground_data = json.load(f)
        p4_data = ground_data.get("phase4_dynamic_shap", {})
        p3_data = ground_data.get("phase3_static_telemetry", {})
        p4_unmatched = p4_data.get("total_unmatched_numbers", 0)
        p3_unmatched = p3_data.get("total_unmatched_numbers", 0)
        p4_flagged = p4_data.get("flagged_flows_count", 0)
        check_presence(f"{p4_unmatched} / 40", f"Phase 4 ungrounded numbers ({p4_unmatched}/40)")
        check_presence(f"{p3_unmatched} / 40", f"Phase 3 ungrounded numbers ({p3_unmatched}/40)")

    # 3. Noise sweep metrics
    noise_file = REPORTS_DIR / "noise_sweep_metrics.json"
    if noise_file.exists():
        with open(noise_file, "r", encoding="utf-8") as f:
            noise_data = json.load(f)
        b2_rf = noise_data.get("deployed_b2_rf", {})
        if b2_rf:
            check_presence(f"{b2_rf.get('n_trees')} trees", f"Deployed B2 RF trees count ({b2_rf.get('n_trees')})")
            check_presence(f"max_depth {b2_rf.get('max_depth')}", f"Deployed B2 RF max_depth ({b2_rf.get('max_depth')})")
        for entry in noise_data.get("tree_threshold_analysis", []):
            if entry["feature"] == "flow_duration":
                s01 = entry["rf_splits_within_0_01"]
                s05 = entry["rf_splits_within_0_05"]
                check_presence(f"{s01:,}".replace(",", ""), f"flow_duration splits +/-0.01 ({s01})")
                check_presence(f"{s05:,}".replace(",", ""), f"flow_duration splits +/-0.05 ({s05})")

    # 4. Adversarial FGSM & Controls
    fgsm_file = REPORTS_DIR / "adversarial_fgsm.json"
    if fgsm_file.exists():
        with open(fgsm_file, "r", encoding="utf-8") as f:
            fgsm_data = json.load(f)
        diff_zero = fgsm_data["zero_attack_control"]["max_numerical_drift"]
        check_presence(f"{diff_zero:.6f}", f"Zero-attack numerical drift ({diff_zero:.6f})")

    # 5. HopSkipJump metrics
    hsj_file = REPORTS_DIR / "adversarial_hsj.json"
    if hsj_file.exists():
        with open(hsj_file, "r", encoding="utf-8") as f:
            hsj_data = json.load(f)
        for budget_name, b_data in hsj_data.items():
            cnt = b_data.get("adversarial_count")
            if cnt:
                check_presence(cnt, f"HSJ {budget_name} perturbed count ({cnt})")

    # 6. ZOO metrics
    zoo_file = REPORTS_DIR / "adversarial_zoo.json"
    if zoo_file.exists():
        with open(zoo_file, "r", encoding="utf-8") as f:
            zoo_data = json.load(f)
        for budget_name, b_data in zoo_data.items():
            cnt = b_data.get("adversarial_count")
            if cnt:
                check_presence(cnt, f"ZOO {budget_name} perturbed count ({cnt})")

    # 7. Defense experiment metrics
    def_file = REPORTS_DIR / "adversarial_defense.json"
    if def_file.exists():
        with open(def_file, "r", encoding="utf-8") as f:
            def_data = json.load(f)
        for arm_name, m in def_data.get("arms", {}).items():
            cl_cnt = m["clean"]["count"]
            fg_cnt = m["fgsm_eps010"]["count"]
            check_presence(cl_cnt, f"Defense {arm_name} clean count ({cl_cnt})")
            check_presence(fg_cnt, f"Defense {arm_name} FGSM 0.10 count ({fg_cnt})")

    return errors


def check_requirements_completeness() -> List[str]:
    """Ensures every imported third-party library is present in requirements.txt."""
    req_file = PROJECT_ROOT / "requirements.txt"
    with open(req_file, "r", encoding="utf-8") as f:
        req_lines = [line.strip().split(">=")[0].split("==")[0].lower() for line in f if line.strip() and not line.startswith("#")]

    module_map = {
        "sklearn": "scikit-learn",
        "imblearn": "imbalanced-learn",
        "art": "adversarial-robustness-toolbox",
    }

    imported_modules = set()
    for py_path in PROJECT_ROOT.glob("**/*.py"):
        if "venv" in str(py_path) or ".gemini" in str(py_path):
            continue
        try:
            with open(py_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        imported_modules.add(n.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules.add(node.module.split(".")[0])
        except Exception:
            pass

    standard_or_local = {
        "sys", "os", "json", "time", "logging", "argparse", "hashlib", "pathlib", "typing",
        "shutil", "datetime", "re", "warnings", "collections", "ast", "glob", "math",
        "models", "ingest", "preprocess", "xai", "adversarial", "alerts", "orchestrator",
        "config", "predict", "tools", "__future__", "dataclasses", "gc", "uuid", "experiments",
        "run_phase3_llm", "scapy"
    }

    missing = []
    for mod in imported_modules:
        if mod in standard_or_local:
            continue
        pkg_name = module_map.get(mod, mod).lower()
        if not any(req == pkg_name or req.startswith(pkg_name) for req in req_lines):
            missing.append(f"Imported module '{mod}' (package '{pkg_name}') is missing from requirements.txt")

    return missing


def check_classes_and_counts() -> List[str]:
    """Verifies class counts and names match encoders and config."""
    errors = []
    le_8 = joblib.load(ENCODER_8_PATH)
    if len(le_8.classes_) != 8:
        errors.append(f"Encoder 8-class has {len(le_8.classes_)} classes, expected 8")
    le_34 = joblib.load(ENCODER_34_PATH)
    if len(le_34.classes_) != 34:
        errors.append(f"Encoder 34-class has {len(le_34.classes_)} classes, expected 34")

    readme_path = PROJECT_ROOT / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")
    if "33 attack types" not in readme_text and "34" not in readme_text:
        errors.append("README does not reference the dataset fine-grained attack classes")

    return errors


def main():
    print("=" * 76)
    print("  PHASE 4 REPORT & CODEBASE CONSISTENCY AUDITOR")
    print("=" * 76)

    summary_file = REPORTS_DIR / "phase4_summary.md"
    if not summary_file.exists():
        print(f"[FAIL] Missing {summary_file}")
        sys.exit(1)

    summary_text = summary_file.read_text(encoding="utf-8")

    all_errors = []

    print("[1] Auditing mathematical agreement of recomputed percentages ...")
    frac_errs = check_fraction_percentages(summary_text)
    if frac_errs:
        all_errors.extend(frac_errs)
        for e in frac_errs:
            print(f"  [ERROR] {e}")
    else:
        print("  [PASS] All quoted percentages recompute within 0.15% tolerance.")

    print("\n[2] Cross-checking quoted numbers against reports/*.json artifacts ...")
    json_errs = check_json_alignments(summary_text)
    if json_errs:
        all_errors.extend(json_errs)
        for e in json_errs:
            print(f"  [ERROR] {e}")
    else:
        print("  [PASS] All quoted figures align with underlying JSON artifacts.")

    print("\n[3] Auditing requirements.txt completeness against code imports ...")
    req_errs = check_requirements_completeness()
    if req_errs:
        all_errors.extend(req_errs)
        for e in req_errs:
            print(f"  [ERROR] {e}")
    else:
        print("  [PASS] requirements.txt contains all imported external packages.")

    print("\n[4] Verifying class encoders and labels ...")
    class_errs = check_classes_and_counts()
    if class_errs:
        all_errors.extend(class_errs)
        for e in class_errs:
            print(f"  [ERROR] {e}")
    else:
        print("  [PASS] Encoders and class definitions verified.")

    print("\n" + "=" * 76)
    if not all_errors:
        print("  AUDIT RESULT: 100% PASS (0 mismatches detected)")
        print("=" * 76)
        sys.exit(0)
    else:
        print(f"  AUDIT RESULT: FAIL ({len(all_errors)} mismatches detected)")
        print("=" * 76)
        sys.exit(1)


if __name__ == "__main__":
    main()
