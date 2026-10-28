# ============================================================
# config.py  -  Central configuration for Phase 1 & 2
# ============================================================
# Edit the paths and knobs here; nothing else should need
# to change for the pipeline to run end-to-end.

import os
from pathlib import Path

# ------------------------------------------------------------------
# Root directories
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()

# Where the raw CICIoT2023 CSVs live.
# Structure expected:
#   DATA_RAW_DIR/train/train.csv
#   DATA_RAW_DIR/test/test.csv
#   DATA_RAW_DIR/validation/validation.csv   (used only in Phase 3+)
DATA_RAW_DIR = Path(r"D:\IROP MODEL\CICIOT23")

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR      = PROJECT_ROOT / "artifacts"
REPORTS_DIR        = PROJECT_ROOT / "reports"

# Ensure output directories exist at import time
for _d in [DATA_PROCESSED_DIR, ARTIFACTS_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Sampling knob  (set to None to use ALL rows - needs ~32 GB RAM)
# ------------------------------------------------------------------
# Number of rows to sample PER FINE-GRAINED LABEL (34 classes).
# On a laptop: 10_000 gives ~340 k rows total, runs in ~3 minutes.
# On Colab Pro: 50_000 gives ~1.7 M rows, still manageable.
SAMPLE_PER_CLASS = 10_000          # <-- change this one number to scale up

# CSV chunk size for the chunked reader (rows per read call)
CHUNK_SIZE = 200_000

# ------------------------------------------------------------------
# Data files
# ------------------------------------------------------------------
TRAIN_CSV      = DATA_RAW_DIR / "train" / "train.csv"
TEST_CSV       = DATA_RAW_DIR / "test"  / "test.csv"
VALIDATION_CSV = DATA_RAW_DIR / "validation" / "validation.csv"

# ------------------------------------------------------------------
# Processed outputs
# ------------------------------------------------------------------
TRAIN_PARQUET = DATA_PROCESSED_DIR / "train_phase2.parquet"
TEST_PARQUET  = DATA_PROCESSED_DIR / "test_phase2.parquet"

# ------------------------------------------------------------------
# Artifact paths
# ------------------------------------------------------------------
SCALER_PATH           = ARTIFACTS_DIR / "standard_scaler.pkl"
ENCODER_34_PATH       = ARTIFACTS_DIR / "label_encoder_34class.pkl"
ENCODER_8_PATH        = ARTIFACTS_DIR / "label_encoder_8class.pkl"
SELECTED_FEATURES_PATH = ARTIFACTS_DIR / "selected_features.json"
RFE_RANKING_PATH      = ARTIFACTS_DIR / "rfe_feature_ranking.csv"

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
RANDOM_SEED = 42

# ------------------------------------------------------------------
# Split
# ------------------------------------------------------------------
TEST_SIZE = 0.20   # 80 / 20 train-test split

# ------------------------------------------------------------------
# SMOTE  - oversample minorities up to this cap
# (avoids memory explosion on very large majority classes)
# ------------------------------------------------------------------
SMOTE_TARGET_COUNT = None   # Dynamically computed in Phase 2 from largest train class

# ------------------------------------------------------------------
# RFE
# ------------------------------------------------------------------
N_FEATURES_TO_SELECT = 19              # paper target
RFE_SUBSAMPLE        = 50_000          # rows used to fit RFE (speed)
RFE_ESTIMATOR_TREES  = 50             # RF trees inside RFE

# ------------------------------------------------------------------
# Smoke-test classifier
# ------------------------------------------------------------------
SMOKE_TEST_TREES = 100

# ------------------------------------------------------------------
# 34-label  -->  8-class mapping
# Built from the REAL label scan of CICIoT2023 (not literature).
# ------------------------------------------------------------------ 
LABEL_TO_CLASS = {
    # ---------- DDoS (9 sub-types) ----------
    "DDoS-ACK_Fragmentation":    "DDoS",
    "DDoS-HTTP_Flood":           "DDoS",
    "DDoS-ICMP_Flood":           "DDoS",
    "DDoS-ICMP_Fragmentation":   "DDoS",
    "DDoS-PSHACK_Flood":         "DDoS",
    "DDoS-RSTFINFlood":          "DDoS",
    "DDoS-SYN_Flood":            "DDoS",
    "DDoS-SynonymousIP_Flood":   "DDoS",
    "DDoS-TCP_Flood":            "DDoS",
    "DDoS-UDP_Flood":            "DDoS",
    "DDoS-UDP_Fragmentation":    "DDoS",
    "DDoS-SlowLoris":            "DDoS",
    # ---------- DoS (4 sub-types) ----------
    "DoS-HTTP_Flood":            "DoS",
    "DoS-SYN_Flood":             "DoS",
    "DoS-TCP_Flood":             "DoS",
    "DoS-UDP_Flood":             "DoS",
    # ---------- Recon (4 sub-types) ----------
    "Recon-HostDiscovery":       "Recon",
    "Recon-OSScan":              "Recon",
    "Recon-PingSweep":           "Recon",
    "Recon-PortScan":            "Recon",
    "VulnerabilityScan":         "Recon",
    # ---------- Web-based (5 sub-types) ----------
    "SqlInjection":              "Web-based",
    "CommandInjection":          "Web-based",
    "XSS":                       "Web-based",
    "Uploading_Attack":          "Web-based",
    "BrowserHijacking":          "Web-based",
    "Backdoor_Malware":          "Web-based",
    # ---------- Brute Force (1 sub-type) ----------
    "DictionaryBruteForce":      "BruteForce",
    # ---------- Spoofing (2 sub-types) ----------
    "DNS_Spoofing":              "Spoofing",
    "MITM-ArpSpoofing":         "Spoofing",
    # ---------- Mirai (3 sub-types) ----------
    "Mirai-greeth_flood":        "Mirai",
    "Mirai-greip_flood":         "Mirai",
    "Mirai-udpplain":            "Mirai",
    # ---------- Benign ----------
    "BenignTraffic":             "Benign",
}

# The 8 class names (order matters for encoding)
CLASS_NAMES = ["Benign", "DDoS", "DoS", "Recon",
               "Web-based", "BruteForce", "Spoofing", "Mirai"]

# Label column names
LABEL_COL    = "label"
LABEL_34_COL = "label_34"
LABEL_8_COL  = "label_8"
