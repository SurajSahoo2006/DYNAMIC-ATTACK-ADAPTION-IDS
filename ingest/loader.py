"""
ingest/loader.py
================
Phase 1 - Data Ingestion
Loads CICIoT2023 CSV files in chunks, samples SAMPLE_PER_CLASS rows
per fine-grained label, and returns a combined DataFrame.

Why chunked + sampled?
  The full dataset is ~47 M rows / ~2 GB of CSVs.  Loading it all at
  once on a laptop needs ~16-32 GB RAM.  We read in CHUNK_SIZE-row
  windows, keep a rolling reservoir per label, and stop early once we
  have enough.  This keeps peak RAM below ~2 GB.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_CSV,
    TEST_CSV,
    CHUNK_SIZE,
    SAMPLE_PER_CLASS,
    RANDOM_SEED,
)

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# The 46 numeric feature columns (everything except 'label')          #
# ------------------------------------------------------------------ #
FEATURE_COLS = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration",
    "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
    "IAT", "Number", "Magnitue", "Radius", "Covariance",
    "Variance", "Weight",
]
LABEL_COL = "label"
ALL_COLS   = FEATURE_COLS + [LABEL_COL]


def _reservoir_sample_chunk(
    buckets: dict[str, list[pd.DataFrame]],
    chunk: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> None:
    """
    Reservoir-sample up to `n` rows per label from `chunk` into
    `buckets`.  Once a bucket reaches `n` rows we stop adding.
    """
    for label, grp in chunk.groupby(LABEL_COL, sort=False):
        if label not in buckets:
            buckets[label] = []
        current_count = sum(len(df) for df in buckets[label])
        needed = n - current_count
        if needed <= 0:
            continue
        if len(grp) <= needed:
            buckets[label].append(grp)
        else:
            idx = rng.choice(len(grp), size=needed, replace=False)
            buckets[label].append(grp.iloc[idx])


def load_ciciot2023(
    csv_paths: list[Path] | None = None,
    sample_per_class: int = SAMPLE_PER_CLASS,
    chunk_size: int = CHUNK_SIZE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Read CICIoT2023 CSVs, sample `sample_per_class` rows per
    fine-grained label, and return a single DataFrame.

    Parameters
    ----------
    csv_paths      : list of CSV Path objects.  Defaults to train + test.
    sample_per_class : rows per label.  None --> use ALL rows (needs ~32 GB).
    chunk_size     : rows per pandas read_csv chunk.
    seed           : random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with 46 float32 features + 'label' (str) column.
    """
    if csv_paths is None:
        csv_paths = [p for p in [TRAIN_CSV, TEST_CSV] if p.exists()]
    if not csv_paths:
        raise FileNotFoundError(
            "No CSV files found.  Check DATA_RAW_DIR in config.py.\n"
            f"Expected: {TRAIN_CSV}"
        )

    rng     = np.random.default_rng(seed)
    buckets: dict[str, list[pd.DataFrame]] = {}
    total_rows_read = 0

    dtype_map = {c: "float32" for c in FEATURE_COLS}

    for csv_path in csv_paths:
        log.info("Reading %s ...", csv_path.name)
        reader = pd.read_csv(
            csv_path,
            usecols=ALL_COLS,
            dtype=dtype_map,
            chunksize=chunk_size,
            low_memory=False,
        )
        for chunk in tqdm(reader, desc=f"  {csv_path.name}", unit="chunk"):
            total_rows_read += len(chunk)
            if sample_per_class is None:
                for label, grp in chunk.groupby(LABEL_COL, sort=False):
                    if label not in buckets:
                        buckets[label] = []
                    buckets[label].append(grp)
            else:
                _reservoir_sample_chunk(buckets, chunk, sample_per_class, rng)
                # Early exit if all buckets are full
                all_full = all(
                    sum(len(d) for d in v) >= sample_per_class
                    for v in buckets.values()
                )
                if all_full and len(buckets) >= 34:
                    log.info("All buckets full - stopping early.")
                    break

    log.info("Total rows scanned: %s", f"{total_rows_read:,}")

    frames = []
    for label, dfs in buckets.items():
        df = pd.concat(dfs, ignore_index=True)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # Shuffle so classes are interleaved
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    log.info("Combined shape: %s", combined.shape)
    return combined
