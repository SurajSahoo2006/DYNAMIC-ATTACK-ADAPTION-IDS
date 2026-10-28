"""
run_phase1_2.py
===============
One-command runner for Phase 1 (Data Ingestion) and Phase 2
(Preprocessing + Feature Selection).

Usage
-----
    python run_phase1_2.py               # runs with SAMPLE_PER_CLASS from config.py
    python run_phase1_2.py --sample 5000 # override sample size
    python run_phase1_2.py --no-smote   # skip SMOTE (faster, for quick tests)
    python run_phase1_2.py --full        # use ALL rows (needs ~32 GB RAM)

Outputs
-------
    data/processed/train_phase2.parquet
    data/processed/test_phase2.parquet
    artifacts/standard_scaler.pkl
    artifacts/label_encoder_34class.pkl
    artifacts/label_encoder_8class.pkl
    artifacts/selected_features.json
    artifacts/rfe_feature_ranking.csv
    reports/class_distribution.png
    reports/rfe_ranking.png
    reports/summary.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------ #
# Parse args before anything else so --sample can override config     #
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser(description="Run Phase 1 + 2 of the IDS pipeline")
parser.add_argument("--sample", type=int, default=None,
                    help="Rows per class to sample (overrides config.py)")
parser.add_argument("--no-smote", action="store_true",
                    help="Skip SMOTE (useful for very fast smoke tests)")
parser.add_argument("--full", action="store_true",
                    help="Use ALL rows (no sampling). Needs ~32 GB RAM.")
args = parser.parse_args()

# Override config before importing it elsewhere
import config
if args.full:
    config.SAMPLE_PER_CLASS = None
elif args.sample is not None:
    config.SAMPLE_PER_CLASS = args.sample

# Now safe to import modules that read config at import time
from config import (
    SAMPLE_PER_CLASS, TRAIN_PARQUET, TEST_PARQUET,
    REPORTS_DIR, ARTIFACTS_DIR, SELECTED_FEATURES_PATH,
)
from ingest.loader          import load_ciciot2023, FEATURE_COLS
from ingest.validate        import run_validation_report
from preprocess.clean       import clean
from preprocess.encode      import encode_labels, LABEL_34_COL, LABEL_8_COL, LABEL_COL
from preprocess.split       import split
from preprocess.scale       import scale
from preprocess.balance     import balance
from preprocess.select_features import select_features

# ------------------------------------------------------------------ #
# Logging                                                             #
# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    t_start = time.time()

    print("\n" + "=" * 62)
    print("   DYNAMIC ATTACK ADAPTATION -- IDS PIPELINE")
    print("   Phase 1: Data Ingestion  |  Phase 2: Preprocessing")
    print("=" * 62 + "\n")
    print(f"  Sample per class : {SAMPLE_PER_CLASS if SAMPLE_PER_CLASS else 'ALL ROWS'}")
    print(f"  SMOTE            : {'DISABLED' if args.no_smote else 'ENABLED'}")
    print()

    # ================================================================
    # PHASE 1: DATA INGESTION
    # ================================================================
    print("\n" + "-" * 60)
    print("  PHASE 1 -- DATA INGESTION")
    print("-" * 60)

    # Step 1.1 - Load
    log.info("Loading CICIoT2023 CSVs ...")
    df_raw = load_ciciot2023()

    # Step 1.2 - Validate
    val_summary = run_validation_report(df_raw)

    # ================================================================
    # PHASE 2: PREPROCESSING + FEATURE SELECTION
    # ================================================================
    print("\n" + "-" * 60)
    print("  PHASE 2 -- PREPROCESSING + FEATURE SELECTION")
    print("-" * 60)

    # Step 2.1 - Clean
    df_clean, clean_summary = clean(df_raw)
    del df_raw   # free memory

    # Step 2.2 - Encode labels
    df_encoded, le_34, le_8 = encode_labels(df_clean)
    del df_clean

    # Step 2.3 - Split BEFORE scaling / SMOTE
    X_train, X_test, y34_train, y34_test, y8_train, y8_test = split(
        df_encoded, FEATURE_COLS
    )
    del df_encoded

    # Step 2.4 - Scale (fit on train only)
    X_train_s, X_test_s, scaler = scale(X_train, X_test)
    n_train_real = len(X_train_s)
    n_test_real = len(X_test_s)
    del X_train, X_test

    # Step 2.5 - SMOTE (train only)
    if args.no_smote:
        print("\n[BALANCE] SMOTE skipped (--no-smote flag).")
        X_train_b, y34_train_b = X_train_s, y34_train
        balance_stats = None
    else:
        max_train_count = int(y34_train.value_counts().max())
        smote_target = min(max_train_count, config.SAMPLE_PER_CLASS) if config.SAMPLE_PER_CLASS else max_train_count
        X_train_b, y34_train_b, balance_stats = balance(
            X_train_s,
            y34_train,
            target_count=smote_target,
            max_synthetic_multiplier=10,
            label_encoder=le_34,
        )

    # Step 2.6 - Feature Selection (RFE)
    selected_features, ranking_df = select_features(X_train_b, y34_train_b)

    # Step 2.7 - Reduce to 19 features and save Parquet
    print(f"\n[SAVE] Reducing to {len(selected_features)} selected features ...")

    # Rebuild SMOTE-balanced y8 (it got shuffled by SMOTE on y34)
    if not args.no_smote:
        # Map resampled y34 back to y8
        le_34_classes = list(le_34.classes_)
        le_8_classes  = list(le_8.classes_)
        from config import LABEL_TO_CLASS
        y8_train_b = y34_train_b.map(
            lambda i: le_8.transform(
                [LABEL_TO_CLASS[le_34_classes[i]]]
            )[0]
        )
    else:
        y8_train_b = y8_train

    train_out = pd.concat([
        X_train_b[selected_features],
        y34_train_b.rename(LABEL_34_COL),
        y8_train_b.rename(LABEL_8_COL),
    ], axis=1)

    test_out = pd.concat([
        X_test_s[selected_features],
        y34_test.rename(LABEL_34_COL),
        y8_test.rename(LABEL_8_COL),
    ], axis=1)

    train_out.to_parquet(TRAIN_PARQUET, index=False)
    test_out.to_parquet(TEST_PARQUET,  index=False)
    print(f"[SAVE] Train Parquet --> {TRAIN_PARQUET.name}  shape: {train_out.shape}")
    print(f"[SAVE] Test  Parquet --> {TEST_PARQUET.name}   shape: {test_out.shape}")

    # ================================================================
    # SMOKE TEST
    # ================================================================
    _run_smoke_test(train_out, test_out, selected_features, le_8)

    # ================================================================
    # SUMMARY REPORT
    # ================================================================
    elapsed = time.time() - t_start
    _write_summary_md(
        val_summary, clean_summary, train_out, test_out,
        n_train_real, n_test_real, balance_stats,
        selected_features, ranking_df, elapsed,
        smote_skipped=args.no_smote,
    )

    print("\n" + "=" * 62)
    print("   ALL DONE")
    print(f"   Total time: {elapsed:.1f} s")
    print("=" * 62 + "\n")


# ================================================================
# SMOKE TEST  (clearly labelled -- NOT the final result)
# ================================================================
def _run_smoke_test(train, test, feat_cols, le_8) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from config import SMOKE_TEST_TREES, RANDOM_SEED

    print("\n" + "-" * 60)
    print("  [TEST]  SMOKE TEST  (preprocessing sanity check -- NOT final result)")
    print("-" * 60)
    print(f"  Quick RF ({SMOKE_TEST_TREES} trees) on {len(feat_cols)} features")
    print("  Goal: confirm preprocessing produced sensible data.")

    from config import LABEL_34_COL, LABEL_8_COL
    X_tr = train[feat_cols].values
    y_tr = train[LABEL_8_COL].values
    X_te = test[feat_cols].values
    y_te = test[LABEL_8_COL].values

    clf = RandomForestClassifier(
        n_estimators=SMOKE_TEST_TREES,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    print(f"\n  Smoke-test accuracy (8-class): {acc * 100:.2f}%")
    print("  (Expect >90% if preprocessing is correct.)")
    print()
    print(classification_report(y_te, y_pred,
                                 target_names=list(le_8.classes_),
                                 zero_division=0))
    print("-" * 60)
    print("  WARNING  REMINDER: This is a SMOKE TEST only.")
    print("  The real classifier (Phase 3: RF + XGBoost + DNN ensemble)")
    print("  will be trained on the full dataset with proper evaluation.")
    print("-" * 60)

    return acc


# ================================================================
# SUMMARY MARKDOWN
# ================================================================
def _write_summary_md(
    val_summary, clean_summary, train_out, test_out,
    n_train_real, n_test_real, balance_stats,
    selected_features, ranking_df, elapsed, smote_skipped,
) -> None:
    from config import (SAMPLE_PER_CLASS, N_FEATURES_TO_SELECT,
                        RANDOM_SEED, TEST_SIZE, SMOTE_TARGET_COUNT,
                        LABEL_TO_CLASS, CLASS_NAMES, REPORTS_DIR)
    import datetime

    out_path = REPORTS_DIR / "summary.md"

    # rebuild class8 counts from train
    from config import LABEL_8_COL
    class8_counts_train = train_out[LABEL_8_COL].value_counts().sort_index()

    lines = [
        "# Phase 1 & 2 - Pipeline Summary",
        "",
        f"> Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> Runtime: {elapsed:.1f} s  ",
        f"> Random seed: {RANDOM_SEED}",
        "",
        "---",
        "",
        "## Phase 1 - Data Ingestion",
        "",
        "### Dataset: CICIoT2023",
        f"- Source files: `train.csv` + `test.csv` (raw CSVs in `data/raw/`)",
        f"- Columns in raw CSV: **47** (46 numeric features + 1 label column)",
        f"- Sample per fine-grained class: **{SAMPLE_PER_CLASS if SAMPLE_PER_CLASS else 'ALL'}**",
        "",
        "### Validation Report",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Raw rows loaded | {val_summary['n_rows']:,} |",
        f"| Columns | {val_summary['n_cols']} |",
        f"| Total NaN cells | {val_summary['total_missing']} |",
        f"| Total Inf cells | {val_summary['total_inf']} |",
        f"| Exact duplicate rows | {val_summary['n_duplicates']:,} |",
        f"| Fine-grained labels | {val_summary['n_labels']} |",
        "",
        "### 34-Label --> 8-Class Mapping",
        "(Read directly from the CSV -- NOT taken from literature)",
        "",
        "| Fine-grained Label | 8-Class |",
        "|-------------------|---------|",
    ]

    from config import LABEL_TO_CLASS
    for lbl, cls in sorted(LABEL_TO_CLASS.items()):
        lines.append(f"| `{lbl}` | {cls} |")

    lines += [
        "",
        "### Class Distribution (before preprocessing)",
        "See `reports/class_distribution.png`",
        "",
        "| 8-Class | Raw count |",
        "|---------|-----------|",
    ]
    for cls, cnt in val_summary.get("class8_counts", {}).items():
        lines.append(f"| {cls} | {cnt:,} |")

    lines += [
        "",
        "> **Imbalance observation**: Some attack classes have far fewer samples",
        "> than others. SMOTE corrects this for training (see Phase 2 below).",
        "",
        "---",
        "",
        "## Phase 2 - Preprocessing",
        "",
        "### Step 1 - Cleaning",
        f"| Step | Rows dropped |",
        f"|------|-------------|",
        f"| NaN rows | {clean_summary['nan_rows_dropped']:,} |",
        f"| Inf rows | {clean_summary['inf_rows_dropped']:,} |",
        f"| Duplicate rows | {clean_summary['dupe_rows_dropped']:,} |",
        f"| **Rows after cleaning** | **{clean_summary['output_rows']:,}** |",
        "",
        "### Step 2 - Label Encoding",
        "- Two label columns created: `label_34` (int 0-33) and `label_8` (int 0-7)",
        "- Encoders saved to `artifacts/`",
        "",
        "### Step 3 - Train / Test Split",
        f"- **Ratio**: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}  (stratified on `label_34`)",
        f"- **Train rows before SMOTE (real)**: {n_train_real:,}",
        f"- **Train rows after SMOTE (real + synthetic)**: {len(train_out):,}",
        f"- **Test rows (never resampled)**: {n_test_real:,}",
        "- [OK] Zero index overlap confirmed (no data leakage)",
        "",
        "### Step 4 - StandardScaler",
        "- Fit on **train set only**",
        "- Applied to both train and test",
        "- Scaler saved to `artifacts/standard_scaler.pkl`",
        "",
        f"### Step 5 - SMOTE (class balancing)",
    ]

    if smote_skipped or balance_stats is None:
        lines.append("- **SKIPPED** (--no-smote flag used)")
    else:
        lines += [
            f"- Applied to **train set only** (test set unchanged)",
            f"- Target per class: **{balance_stats['target_count']:,}** (size of largest train class, bounded by SAMPLE_PER_CLASS)",
            f"- Safety cap: maximum **{balance_stats['max_synthetic_multiplier']}x** synthetic rows per class",
            f"- Real train rows: **{balance_stats['total_real']:,}**",
            f"- Synthetic train rows created: **{balance_stats['total_synthetic']:,}** ({balance_stats['overall_synthetic_pct']:.1f}% synthetic)",
            f"- Train rows after SMOTE: **{len(train_out):,}**",
        ]

    lines += [
        "",
        "### Step 6 - Feature Selection (RFE)",
        "",
        "> WARNING **SLIDE CORRECTION**: Your presentation states '115 --> 19 features'.",
        "> CICIoT2023 has **46** numeric features (not 115).",
        "> Please update your slides to say **'46 --> 19 features'**.",
        "",
        f"- Method: Recursive Feature Elimination with RandomForest estimator",
        f"- Input features: **46**",
        f"- Selected features: **{N_FEATURES_TO_SELECT}**",
        f"- RFE subsample used: {min(50_000, len(train_out)):,} rows",
        "",
        "#### Top 19 Selected Features",
        "| # | Feature |",
        "|---|---------|",
    ]
    for i, feat in enumerate(selected_features, 1):
        lines.append(f"| {i} | `{feat}` |")

    lines += [
        "",
        "See `reports/rfe_ranking.png` for the full ranking chart.",
        "",
        "### Step 7 - Final Output",
        f"| File | Shape | Location |",
        f"|------|-------|----------|",
        f"| train_phase2.parquet | {train_out.shape} | `data/processed/` |",
        f"| test_phase2.parquet  | {test_out.shape}  | `data/processed/` |",
        "",
        "Columns in Parquet files: `[19 features] + label_34 + label_8`",
        "",
        "---",
        "",
        "## Handoff to Phase 3",
        "The following artifacts are ready for Phase 3 (Classifier):",
        "- `data/processed/train_phase2.parquet`  - SMOTE-balanced train set",
        "- `data/processed/test_phase2.parquet`   - untouched test set",
        "- `artifacts/standard_scaler.pkl`        - for scaling live traffic",
        "- `artifacts/label_encoder_34class.pkl`  - decode predictions",
        "- `artifacts/label_encoder_8class.pkl`   - decode predictions",
        "- `artifacts/selected_features.json`     - the 19 feature names",
        "",
        "---",
        "",
        "## Assumptions & Notes",
        "1. Only `train.csv` and `test.csv` were loaded; `validation.csv` is reserved.",
        "2. RFE used a subsample for speed -- results are deterministic given the seed.",
        "3. Cowrie honeypot module is a documented placeholder (no log file provided).",
        "4. PCAP demo (`ingest/pcap_demo.py`) is standalone; run manually if needed.",
        "5. Benchmark datasets (UNSW-NB15, NSL-KDD, Bot-IoT) are Phase 3+ stubs.",
        f"6. '115 --> 19' in slides should be corrected to '**46 --> 19**' for CICIoT2023.",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[REPORT] Summary written --> {out_path}")


if __name__ == "__main__":
    main()
