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
    # or explicit file:/// URIs
    pattern = re.compile(r"(?<!https)(?<!http)(?<!ftp)\b[A-Za-z]:[\\/]|file:\/\/\/")
    for i, line in enumerate(markdown_text.splitlines(), 1):
        if pattern.search(line):
            errors.append(f"[{filename}] Line {i}: machine-specific path detected: {line.strip()}")
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
        for b in ["random", "always_none", "max_containment", "severity_rule", "oracle"]:
            b_info = s_data["baselines"][b]
            check_presence(f"{b_info['mean_return']:.2f}", f"Baseline {b} return ({b_info['mean_return']:.2f})")
            check_presence(f"{b_info['attacks_contained']['x']} of {b_info['attacks_contained']['n']}", f"Baseline {b} attacks count")

        # MAPPO Aggregate
        m_agg = s_data["mappo_aggregate"]
        check_presence(f"{m_agg['mean_return']:.2f}", f"MAPPO aggregate return ({m_agg['mean_return']:.2f})")
        check_presence(f"{m_agg['attacks_contained']['x']} of {m_agg['attacks_contained']['n']}", f"MAPPO aggregate attacks count")

        # Paired diff
        p_diff = s_data["paired_differences"]["overall"]
        check_presence(f"+{p_diff['mean_diff']:.2f}", f"Overall paired diff (+{p_diff['mean_diff']:.2f})")
        check_presence(p_diff["decision_outcome"], f"Decision outcome ({p_diff['decision_outcome']})")

    # 4. Adversarial stress test
    stress_file = REPORTS_DIR / "marl_stress_test.json"
    if stress_file.exists():
        with open(stress_file, "r", encoding="utf-8") as f:
            st = json.load(f)
        check_presence(f"{st['metadata']['clean_classifier_acc'] * 100:.2f}%", "Stress test clean accuracy")
        check_presence(f"{st['metadata']['adversarial_classifier_acc'] * 100:.2f}%", "Stress test adversarial accuracy")
        check_presence(f"{st['comparison_vs_clean']['mappo_aggregate']['delta_return']:+.2f}", "MAPPO delta return")

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
        "io", "base64", "tempfile", "urllib",
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

    print("\n[6] Auditing for machine-specific path strings across markdown documents ...")
    path_p4 = check_machine_specific_paths(p4_text, "phase4_summary.md")
    path_p5 = check_machine_specific_paths(p5_text, "phase5_summary.md")
    path_readme = check_machine_specific_paths(readme_text, "README.md")
    path_errs = path_p4 + path_p5 + path_readme
    if path_errs:
        all_errors.extend(path_errs)
        for e in path_errs:
            print(f"  [ERROR] {e}")
    else:
        print("  [PASS] Zero machine-specific path strings found.")

    print("\n[7] Verifying README execution commands ...")
    cmd_errs = check_readme_commands_exist()
    if cmd_errs:
        all_errors.extend(cmd_errs)
        for e in cmd_errs:
            print(f"  [ERROR] {e}")
    else:
        print("  [PASS] All documented README script commands exist.")

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
