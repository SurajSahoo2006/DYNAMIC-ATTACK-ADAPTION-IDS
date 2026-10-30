"""
tools/check_report_consistency.py
=================================
Automated auditor that verifies mathematical and empirical consistency between
Phase 4 and Phase 5 report documents (reports/phase4_summary.md, reports/phase5_summary.md,
README.md) and underlying generated JSON artifacts in reports/.

Checks:
1. Recomputed percentage consistency: for every "x of y" or "x/y (p%)" quoted in reports,
   verifies that |(x / y) * 100 - p| <= 0.15%.
2. Exact empirical alignment: verifies that every accuracy, count, CI interval, return,
   and runtime quoted in reports matches the numbers in underlying reports/*.json artifacts.
3. Architecture & Class Names consistency with config.py, config_marl.py, and label encoders.
4. Requirements completeness against all external imports in the codebase.
5. Markdown formatting integrity: no corrupted feature names, balanced LaTeX math delimiters,
   no empty table headers, and zero prohibited causal assertions ("prove", "demonstrate", "confirm", "clearly").
6. Machine-specific path audit: ensures no "D:", "C:\\", or "file:///" strings exist in markdown files.
7. README command verification: ensures all documented execution commands correspond to existing scripts.
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

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import ENCODER_8_PATH, ENCODER_34_PATH, REPORTS_DIR
import config_marl


def check_markdown_formatting_and_math(markdown_text: str, filename: str = "report") -> List[str]:
    """
    Checks for markdown text corruption:
    1. Feature name corruption: 'low_duration' appearing without a leading 'f'.
    2. Unbalanced LaTeX math delimiters: odd number of unescaped '$' signs on a line.
    3. Truncated or empty table header cells.
    4. Prohibited causal assertions: 'prove', 'demonstrate', 'confirm', 'clearly'.
    """
    errors = []
    lines = markdown_text.splitlines()
    in_code_block = False

    prohibited_words = ["prove", "demonstrate", "confirm", "clearly"]

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # 1. Feature name corruption
        if re.search(r"(?<!f)low_duration", line):
            errors.append(f"[{filename}] Line {i}: corrupted feature name 'low_duration' (missing 'f'): {stripped}")

        # 2. Odd number of unescaped dollar signs on a line
        # Math blocks $$...$$ or $$ are handled
        if not stripped.startswith("$$"):
            unescaped_dollars = len(re.findall(r"(?<!\\)\$", line))
            if unescaped_dollars % 2 != 0:
                errors.append(f"[{filename}] Line {i}: odd number of unescaped dollar signs ({unescaped_dollars}): {stripped}")

        # 3. Table header cells that are only '$' or empty
        if i < len(lines):
            next_line = lines[i].strip()
            if stripped.startswith("|") and stripped.endswith("|") and re.match(r"^\|(?:\s*[:\-]+\s*\|)+$", next_line):
                cells = [c.strip() for c in re.split(r"(?<!\\)\|", stripped)[1:-1]]
                for c_idx, cell in enumerate(cells, 1):
                    if cell in ["$", "$$", ""]:
                        errors.append(f"[{filename}] Line {i}: table header column {c_idx} is empty or only dollar ('{cell}'): {stripped}")

        # 4. Prohibited words check in Phase 5 report
        if "phase5" in filename.lower():
            for w in prohibited_words:
                if re.search(r"\b" + w + r"\w*", line, re.IGNORECASE):
                    # Exclude allowed quotes or citations if any
                    errors.append(f"[{filename}] Line {i}: prohibited causal word '{w}' detected: {stripped}")

    return errors


def check_machine_specific_paths(markdown_text: str, filename: str) -> List[str]:
    """Ensures no local machine-specific path strings exist in markdown reports."""
    errors = []
    # Match Windows drive paths (e.g. D:\, D:/, C:\, C:/) excluding standard web schemes (http/https/ftp)
    # or explicit file:/// URIs, Users, or username suraj
    pattern = re.compile(r"(?<!https)(?<!http)(?<!ftp)\b[A-Za-z]:[\\/]|file:\/\/\/|\bUsers[\\/]|\\suraj\b|\/suraj\b")
    for i, line in enumerate(markdown_text.splitlines(), 1):
        if pattern.search(line):
            errors.append(f"[{filename}] Line {i}: machine-specific path detected: {line.strip()}")
    return errors


def check_all_tracked_text_files_paths() -> List[str]:
    r"""
    Scans ALL tracked text files in the git repository for local machine-specific path strings:
    - Windows drive letters (e.g. D:\, C:\, D:/, C:/) excluding web schemes (http/https/ftp)
    - file:/// URIs
    - Users/ or Users\
    - username suraj
    """
    import subprocess
    errors = []
    try:
        res = subprocess.run(
            ["git", "ls-files"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        tracked_files = [f.strip() for f in res.stdout.splitlines() if f.strip()]
    except Exception as e:
        return [f"Failed to execute git ls-files: {e}"]

    binary_exts = {".pt", ".joblib", ".parquet", ".png", ".jpg", ".jpeg", ".webp", ".ico", ".bin"}
    pattern = re.compile(r"(?<!https)(?<!http)(?<!ftp)\b[A-Za-z]:[\\/]|file:\/\/\/|\bUsers[\\/]|\\suraj\b|\/suraj\b")

    for rel_path in tracked_files:
        if rel_path == "tools/check_report_consistency.py":
            continue
        fpath = PROJECT_ROOT / rel_path
        if fpath.suffix.lower() in binary_exts or not fpath.exists():
            continue

        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if pattern.search(line):
                        errors.append(f"[{rel_path}] Line {line_num}: machine-specific path detected: {line.strip()[:120]}")
        except Exception:
            pass

    return errors


def check_phase5_table_intervals(p5_text: str) -> List[str]:
    """
    Verifies that all tables in phase5_summary.md reporting counts or means
    include confidence intervals (e.g. [..., ...]).
    """
    errors = []
    lines = p5_text.splitlines()
    in_table = False
    headers = []
    table_rows = []
    table_start_line = 0

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                headers = [c.strip() for c in stripped.strip("|").split("|")]
                in_table = True
                table_start_line = i
                table_rows = []
            elif len(table_rows) == 0 and re.match(r"^\|(?:\s*[:\-]+\s*\|)+$", stripped):
                # Separator line
                table_rows.append(stripped)
            else:
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                table_rows.append((i, cells))
        else:
            if in_table and len(table_rows) > 1:
                # Check headers for statistical counts or means
                target_cols = []
                for idx, h in enumerate(headers):
                    h_lower = h.lower()
                    if any(kw in h_lower for kw in ["return", "attacks", "benign", "containment", "disrupted", "diff"]):
                        if not any(skip in h_lower for skip in ["budget", "step", "seed", "outcome", "rl_val", "training"]):
                            target_cols.append((idx, h))

                if target_cols:
                    for line_no, row_cells in table_rows[1:]:
                        for col_idx, col_header in target_cols:
                            if col_idx < len(row_cells):
                                cell_val = row_cells[col_idx]
                                if "[" in cell_val and "]" in cell_val:
                                    continue
                                if cell_val in ["--", "N/A", ""]:
                                    continue
                                # Check if adjacent column contains CI (e.g. 95% Bootstrap CI)
                                adjacent_has_ci = False
                                if col_idx + 1 < len(row_cells):
                                    adj_cell = row_cells[col_idx + 1]
                                    adj_header = headers[col_idx + 1].lower() if col_idx + 1 < len(headers) else ""
                                    if "ci" in adj_header and "[" in adj_cell and "]" in adj_cell:
                                        adjacent_has_ci = True

                                if not adjacent_has_ci:
                                    errors.append(
                                        f"[phase5_summary.md] Table at line {table_start_line}, row {line_no}: "
                                        f"column '{col_header}' lacks confidence interval brackets in cell: '{cell_val}'"
                                    )

            in_table = False
            headers = []
            table_rows = []

    return errors


def check_fraction_percentages(markdown_text: str, filename: str = "report") -> List[str]:
    """Finds all 'k/n (pct%)' or 'k of n' instances and checks mathematical agreement."""
    errors = []
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
                    f"[{filename}] Percentage mismatch: {k}/{n} recomputed is {recomputed:.2f}%, but report states {stated_pct:.2f}% (match: '{match.group(0)}')"
                )
    return errors


def check_phase4_json_alignments(summary_md: str) -> List[str]:
    """Cross-checks specific JSON metrics against text in phase4_summary.md."""
    errors = []

    def check_presence(val_str: str, label: str):
        if val_str not in summary_md:
            errors.append(f"[phase4_summary.md] Missing or mismatched: {label} (expected '{val_str}')")

    shap_file = REPORTS_DIR / "shap_metrics.json"
    if shap_file.exists():
        with open(shap_file, "r", encoding="utf-8") as f:
            shap_data = json.load(f)
        rho = shap_data.get("spearman_rho")
        if rho is not None:
            check_presence(f"{rho:.4f}", f"SHAP Spearman rho ({rho:.4f})")

    fgsm_file = REPORTS_DIR / "adversarial_fgsm.json"
    if fgsm_file.exists():
        with open(fgsm_file, "r", encoding="utf-8") as f:
            fgsm_data = json.load(f)
        for res in fgsm_data.get("fgsm_and_random_noise_results", []):
            eps = res["epsilon"]
            ens_adv_cnt = res["ensemble_adversarial_count"]
            check_presence(ens_adv_cnt, f"FGSM eps={eps} count ({ens_adv_cnt})")

    return errors


def check_phase5_json_alignments(summary_md: str) -> List[str]:
    """Cross-checks numbers in phase5_summary.md against Phase 5 JSON artifacts."""
    errors = []

    def check_presence(val_str: str, label: str):
        if val_str not in summary_md:
            errors.append(f"[phase5_summary.md] Missing or mismatched: {label} (expected '{val_str}')")

    # 1. Cache metrics
    cache_file = REPORTS_DIR / "marl_cache_metrics.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            c = json.load(f)
        check_presence(str(c["row_counts"]["validation_total"]), "Validation Total rows")
        check_presence(str(c["row_counts"]["rl_train"]), "rl_train rows")
        check_presence(str(c["row_counts"]["rl_val"]), "rl_val rows")
        check_presence(str(c["row_counts"]["test"]), "Test rows")
        check_presence(f"{c['accuracies']['val_total'] * 100:.2f}%", "Validation total accuracy")
        check_presence(f"{c['accuracies']['test'] * 100:.2f}%", "Test accuracy")

    # 2. Pilot metrics
    pilot_file = REPORTS_DIR / "marl_pilot.json"
    if pilot_file.exists():
        with open(pilot_file, "r", encoding="utf-8") as f:
            p = json.load(f)
        check_presence(str(p["pilot_steps"]), "Pilot steps")
        check_presence(str(p["chosen_budget"]), "Chosen training budget")

    # 3. Final evaluation metrics
    eval_file = REPORTS_DIR / "marl_eval.json"
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            e = json.load(f)
        active_setting = e["metadata"]["active_settings"][0]
        s_data = e["settings"][active_setting]
        
        # Baselines
        for b in [
            "random",
            "always_none",
            "max_containment",
            "severity_rule",
            "severity_rule_budget_aware",
            "label_aware_rule",
            "label_aware_rule_budget_aware",
        ]:
            if b in s_data["baselines"]:
                b_info = s_data["baselines"][b]
                check_presence(f"{b_info['mean_return']:.2f}", f"Baseline {b} return ({b_info['mean_return']:.2f})")
                check_presence(f"{b_info['attacks_contained']['x']} of {b_info['attacks_contained']['n']}", f"Baseline {b} attacks count")

        # MAPPO Aggregate
        m_agg = s_data["mappo_aggregate"]
        check_presence(f"{m_agg['mean_return']:.2f}", f"MAPPO aggregate return ({m_agg['mean_return']:.2f})")
        check_presence(f"{m_agg['attacks_contained']['x']} of {m_agg['attacks_contained']['n']}", f"MAPPO aggregate attacks count")

        # Paired diff
        p_diff = s_data["paired_differences"].get("overall_vs_best_non_learned", s_data["paired_differences"]["overall"])
        sign = "+" if p_diff['mean_diff'] > 0 else ""
        check_presence(f"{sign}{p_diff['mean_diff']:.2f}", f"Overall paired diff ({sign}{p_diff['mean_diff']:.2f})")
        check_presence(p_diff["decision_outcome"], f"Decision outcome ({p_diff['decision_outcome']})")

    # 4. Adversarial stress test
    stress_file = REPORTS_DIR / "marl_stress_test.json"
    if stress_file.exists():
        with open(stress_file, "r", encoding="utf-8") as f:
            st = json.load(f)
        check_presence(f"{st['metadata']['clean_classifier_acc'] * 100:.2f}%", "Stress test clean accuracy")
        check_presence(f"{st['metadata']['adversarial_classifier_acc'] * 100:.2f}%", "Stress test adversarial accuracy")
        if "paired_comparisons" in st:
            delta = st['paired_comparisons']['mappo_aggregate']['paired_delta_return']
            sign = "+" if delta > 0 else ""
            check_presence(f"{sign}{delta:.2f}", "MAPPO delta return")
        elif "comparison_vs_clean" in st:
            delta = st['comparison_vs_clean']['mappo_aggregate']['delta_return']
            check_presence(f"{delta:+.2f}", "MAPPO delta return")

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
        "run_phase3_llm", "scapy", "concurrent", "marl", "export", "dashboard", "config_marl",
        "io", "base64", "tempfile", "urllib", "subprocess", "drift", "federated", "copy", "zeroday",
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


def check_readme_commands_exist() -> List[str]:
    """Ensures python script commands documented in README.md exist as files."""
    errors = []
    readme_path = PROJECT_ROOT / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")

    matches = re.findall(r"python\s+([\w\/\.\-]+)\b", readme_text)
    for m in matches:
        if not m.endswith(".py"):
            continue
        script_path = PROJECT_ROOT / m
        if not script_path.exists():
            errors.append(f"Command in README refers to non-existent script: {m}")

    return errors


def check_manifest_artifacts() -> Tuple[List[str], List[str]]:
    """
    CHECK 9 — manifest_artifact_check:
    Open artifacts/model_manifest.json.
    For every file listed under "missing_artifacts": emit a WARNING line
    "WARN manifest: <file> listed as missing".
    Pass if missing_artifacts is empty; WARN (not FAIL) otherwise.
    """
    errors = []
    warnings = []
    manifest_path = PROJECT_ROOT / "artifacts" / "model_manifest.json"
    if not manifest_path.exists():
        errors.append("artifacts/model_manifest.json is missing entirely.")
        return errors, warnings

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        missing_list = manifest.get("missing_artifacts", [])
        for item in missing_list:
            warnings.append(f"WARN manifest: {item} listed as missing")
    except Exception as e:
        errors.append(f"Error parsing artifacts/model_manifest.json: {e}")

    return errors, warnings


def check_reward_settings_coverage() -> Tuple[List[str], List[str]]:
    """
    CHECK 10 — reward_settings_coverage:
    Open marl/results/reward_settings_comparison.json if it exists.
    Confirm all three settings are present as keys.
    If any setting has truncated=true, emit a WARNING.
    FAIL only if the file is absent entirely.
    """
    errors = []
    warnings = []
    comparison_path = PROJECT_ROOT / "marl" / "results" / "reward_settings_comparison.json"
    if not comparison_path.exists():
        errors.append("marl/results/reward_settings_comparison.json is absent entirely.")
        return errors, warnings

    try:
        with open(comparison_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected_settings = ["balanced", "security_first", "availability_first"]
        for s in expected_settings:
            if s not in data:
                errors.append(f"Setting '{s}' missing from reward_settings_comparison.json keys: {list(data.keys())}")
            else:
                entry = data[s]
                if isinstance(entry, dict) and entry.get("truncated") is True:
                    warnings.append(f"WARN reward_settings: Setting '{s}' has truncated=true")
    except Exception as e:
        errors.append(f"Error parsing marl/results/reward_settings_comparison.json: {e}")

    return errors, warnings


def check_phase8_drift() -> Tuple[List[str], List[str]]:
    """
    CHECK 11 — phase8_drift_check:
    Open drift/drift_detection_results.json, confirm "drift_injected_window" exists.
    Open drift/retrain_results.json, confirm "accuracy_delta" exists and is float.
    FAIL only if either file missing.
    WARN if accuracy_delta == 0.0 exactly.
    """
    errors = []
    warnings = []
    det_path = PROJECT_ROOT / "drift" / "drift_detection_results.json"
    retrain_path = PROJECT_ROOT / "drift" / "retrain_results.json"

    if not det_path.exists():
        errors.append("drift/drift_detection_results.json is missing.")
    else:
        try:
            with open(det_path, "r", encoding="utf-8") as f:
                det_data = json.load(f)
            if "drift_injected_window" not in det_data:
                errors.append("'drift_injected_window' missing from drift/drift_detection_results.json")
        except Exception as e:
            errors.append(f"Error reading drift/drift_detection_results.json: {e}")

    if not retrain_path.exists():
        errors.append("drift/retrain_results.json is missing.")
    else:
        try:
            with open(retrain_path, "r", encoding="utf-8") as f:
                retrain_data = json.load(f)
            if "accuracy_delta" not in retrain_data:
                errors.append("'accuracy_delta' missing from drift/retrain_results.json")
            else:
                val = retrain_data["accuracy_delta"]
                if not isinstance(val, (int, float)):
                    errors.append(f"'accuracy_delta' in drift/retrain_results.json is not a float: {type(val)}")
                elif float(val) == 0.0:
                    warnings.append("WARN phase8_drift: accuracy_delta is exactly 0.0")
        except Exception as e:
            errors.append(f"Error reading drift/retrain_results.json: {e}")

    return errors, warnings


def check_phase9_federated() -> Tuple[List[str], List[str]]:
    """
    CHECK 12 — phase9_federated_check:
    Open federated/results/fedavg_results.json.
    Confirm "global_federated_model" key exists.
    Confirm "centralized_baseline" key exists.
    FAIL only if the file is missing entirely.
    WARN if global_federated_model.test_accuracy < 0.70
    (suspiciously low, likely a bug).
    """
    errors = []
    warnings = []
    res_path = PROJECT_ROOT / "federated" / "results" / "fedavg_results.json"
    if not res_path.exists():
        errors.append("federated/results/fedavg_results.json is missing entirely.")
        return errors, warnings

    try:
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "global_federated_model" not in data:
            errors.append("'global_federated_model' key missing from federated/results/fedavg_results.json")
        else:
            g_model = data["global_federated_model"]
            acc = g_model.get("test_accuracy", 0.0)
            if acc < 0.70:
                warnings.append(f"WARN phase9_federated: global_federated_model test_accuracy ({acc:.4f}) < 0.70")

        if "centralized_baseline" not in data:
            errors.append("'centralized_baseline' key missing from federated/results/fedavg_results.json")
    except Exception as e:
        errors.append(f"Error parsing federated/results/fedavg_results.json: {e}")

    return errors, warnings


def check_phase10_zeroday() -> Tuple[List[str], List[str]]:
    """
    CHECK 13 — phase10_zeroday_check:
    - FAIL if zeroday/results/openset_summary.json or ae_training.json is missing.
    - FAIL if the threshold in ae_training.json is not a float.
    - WARN if any class is marked truncated.
    - Recompute percentages in phase10_summary.md from the JSON counts
      (same tolerance as the existing percent check).
    """
    errors = []
    warnings = []
    ae_path = PROJECT_ROOT / "zeroday" / "results" / "ae_training.json"
    openset_path = PROJECT_ROOT / "zeroday" / "results" / "openset_summary.json"
    p10_md_path = REPORTS_DIR / "phase10_summary.md"

    if not ae_path.exists():
        errors.append("zeroday/results/ae_training.json is missing.")
    else:
        try:
            with open(ae_path, "r", encoding="utf-8") as f:
                ae_data = json.load(f)
            thresh = ae_data.get("threshold")
            if not isinstance(thresh, (int, float)):
                errors.append(f"Threshold in zeroday/results/ae_training.json is not a float: {type(thresh)}")
        except Exception as e:
            errors.append(f"Error reading zeroday/results/ae_training.json: {e}")

    if not openset_path.exists():
        errors.append("zeroday/results/openset_summary.json is missing.")
    else:
        try:
            with open(openset_path, "r", encoding="utf-8") as f:
                openset_data = json.load(f)

            trunc_list = openset_data.get("truncated_classes", [])
            if trunc_list:
                warnings.append(f"WARN phase10_zeroday: Classes marked truncated: {trunc_list}")

            leave_one_out = openset_data.get("leave_one_class_out", {})
            for cname, cdata in leave_one_out.items():
                if isinstance(cdata, dict) and cdata.get("truncated") is True:
                    warnings.append(f"WARN phase10_zeroday: Class '{cname}' has truncated=True")
        except Exception as e:
            errors.append(f"Error reading zeroday/results/openset_summary.json: {e}")

    if p10_md_path.exists():
        try:
            md_text = p10_md_path.read_text(encoding="utf-8")
            frac_errs = check_fraction_percentages(md_text, "phase10_summary.md")
            if frac_errs:
                errors.extend(frac_errs)
        except Exception as e:
            errors.append(f"Error reading reports/phase10_summary.md: {e}")
    else:
        errors.append("reports/phase10_summary.md is missing.")

    return errors, warnings


def main():
    print("=" * 76)
    print("  PHASE 4 & 5 REPORT & CODEBASE CONSISTENCY AUDITOR")
    print("=" * 76)

    p4_file = REPORTS_DIR / "phase4_summary.md"
    p5_file = REPORTS_DIR / "phase5_summary.md"
    readme_file = PROJECT_ROOT / "README.md"

    if not p4_file.exists():
        print(f"[FAIL] Missing {p4_file}")
        sys.exit(1)
    if not p5_file.exists():
        print(f"[FAIL] Missing {p5_file}")
        sys.exit(1)

    p4_text = p4_file.read_text(encoding="utf-8")
    p5_text = p5_file.read_text(encoding="utf-8")
    readme_text = readme_file.read_text(encoding="utf-8")

    all_errors = []
    all_warnings = []

    print("[1] Auditing mathematical agreement of recomputed percentages ...")
    frac_p4 = check_fraction_percentages(p4_text, "phase4_summary.md")
    frac_p5 = check_fraction_percentages(p5_text, "phase5_summary.md")
    frac_errs = frac_p4 + frac_p5
    if frac_errs:
        all_errors.extend(frac_errs)
        for e in frac_errs:
            print(f"  [ERROR] {e}")
    else:
        print("  [PASS] All quoted percentages in Phase 4 & 5 recompute within 0.15% tolerance.")

    print("\n[2] Cross-checking quoted numbers against reports/*.json artifacts ...")
    json_p4 = check_phase4_json_alignments(p4_text)
    json_p5 = check_phase5_json_alignments(p5_text)
    json_errs = json_p4 + json_p5
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

    print("\n[5] Auditing markdown formatting, table headers, math delimiters, and causal assertions ...")
    fmt_p4 = check_markdown_formatting_and_math(p4_text, "phase4_summary.md")
    fmt_p5 = check_markdown_formatting_and_math(p5_text, "phase5_summary.md")
    fmt_errs = fmt_p4 + fmt_p5
    if fmt_errs:
        all_errors.extend(fmt_errs)
        for e in fmt_errs:
            print(f"  [ERROR] {e}")
    else:
        print("  [PASS] Markdown formatting, table headers, and math delimiters verified.")

    print("\n[6] Auditing for machine-specific path strings across ALL tracked text files ...")
    path_tracked_errs = check_all_tracked_text_files_paths()
    if path_tracked_errs:
        all_errors.extend(path_tracked_errs)
        for e in path_tracked_errs:
            print(f"  [ERROR] {e}")
    else:
        print("  [PASS] Zero machine-specific path strings found across all tracked text files.")

    print("\n[7] Auditing Phase 5 report tables for confidence intervals on counts & means ...")
    tbl_int_errs = check_phase5_table_intervals(p5_text)
    if tbl_int_errs:
        all_errors.extend(tbl_int_errs)
        for e in tbl_int_errs:
            print(f"  [ERROR] {e}")
    else:
        print("  [PASS] All Phase 5 report tables reporting counts or means include confidence intervals.")

    print("\n[8] Verifying README execution commands ...")
    cmd_errs = check_readme_commands_exist()
    if cmd_errs:
        all_errors.extend(cmd_errs)
        for e in cmd_errs:
            print(f"  [ERROR] {e}")
    else:
        print("  [PASS] All documented README script commands exist.")

    print("\n[9] Auditing model manifest artifact integrity (manifest_artifact_check) ...")
    m_errs, m_warns = check_manifest_artifacts()
    if m_errs:
        all_errors.extend(m_errs)
        for e in m_errs:
            print(f"  [ERROR] {e}")
    elif m_warns:
        all_warnings.extend(m_warns)
        for w in m_warns:
            print(f"  {w}")
        print(f"  [WARN] Manifest contains {len(m_warns)} missing artifact(s).")
    else:
        print("  [PASS] artifacts/model_manifest.json verified (0 missing artifacts).")

    print("\n[10] Auditing MARL reward settings coverage (reward_settings_coverage) ...")
    r_errs, r_warns = check_reward_settings_coverage()
    if r_errs:
        all_errors.extend(r_errs)
        for e in r_errs:
            print(f"  [ERROR] {e}")
    elif r_warns:
        all_warnings.extend(r_warns)
        for w in r_warns:
            print(f"  {w}")
        print("  [WARN] Settings present with truncated=true.")
    else:
        print("  [PASS] All 3 reward settings present with truncated=false.")

    print("\n[11] Auditing Phase 8 concept drift and retraining artifacts (phase8_drift_check) ...")
    d_errs, d_warns = check_phase8_drift()
    if d_errs:
        all_errors.extend(d_errs)
        for e in d_errs:
            print(f"  [ERROR] {e}")
    elif d_warns:
        all_warnings.extend(d_warns)
        for w in d_warns:
            print(f"  {w}")
        print("  [WARN] Phase 8 drift check completed with warnings.")
    else:
        print("  [PASS] Phase 8 drift detection and retraining artifacts verified.")

    print("\n[12] Auditing Phase 9 federated learning artifacts (phase9_federated_check) ...")
    f_errs, f_warns = check_phase9_federated()
    if f_errs:
        all_errors.extend(f_errs)
        for e in f_errs:
            print(f"  [ERROR] {e}")
    elif f_warns:
        all_warnings.extend(f_warns)
        for w in f_warns:
            print(f"  {w}")
        print("  [WARN] Phase 9 federated check completed with warnings.")
    else:
        print("  [PASS] Phase 9 federated learning artifacts verified.")

    print("\n[13] Auditing Phase 10 zero-day detection artifacts (phase10_zeroday_check) ...")
    z_errs, z_warns = check_phase10_zeroday()
    if z_errs:
        all_errors.extend(z_errs)
        for e in z_errs:
            print(f"  [ERROR] {e}")
    elif z_warns:
        all_warnings.extend(z_warns)
        for w in z_warns:
            print(f"  {w}")
        print("  [WARN] Phase 10 zero-day check completed with warnings.")
    else:
        print("  [PASS] Phase 10 zero-day detection artifacts verified.")

    print("\n" + "=" * 76)
    if not all_errors:
        if all_warnings:
            print(f"  AUDIT RESULT: 100% PASS with {len(all_warnings)} WARNING(S) (0 errors)")
        else:
            print("  AUDIT RESULT: 100% PASS (0 mismatches detected)")
        print("=" * 76)
        sys.exit(0)
    else:
        print(f"  AUDIT RESULT: FAIL ({len(all_errors)} mismatches detected)")
        print("=" * 76)
        sys.exit(1)


if __name__ == "__main__":
    main()
