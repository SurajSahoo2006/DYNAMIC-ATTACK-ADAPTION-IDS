"""
ingest/cowrie_loader.py
=======================
Phase 1 - Honeypot Enrichment (Cowrie)
=========================================

PURPOSE
-------
Cowrie is an SSH/Telnet honeypot that writes attacker sessions to a
JSON log file (cowrie.json / cowrie.json.YYYY-MM-DD).

This module parses those logs and converts each session into a row
with the SAME column schema as the CICIoT2023 features so it can be
concatenated with the main DataFrame.

CURRENT STATUS:  PLACEHOLDER / DOCUMENTED STUB
-----------------------------------------------
No Cowrie log is available for this project.  If you capture one,
provide its path in config.py (COWRIE_LOG_PATH) and this module
will process it automatically.  If no log exists, the function
returns an empty DataFrame and the pipeline continues cleanly.

HOW TO GET A COWRIE LOG
------------------------
1. Deploy Cowrie: https://github.com/cowrie/cowrie
2. Let it run for a few hours / days on an exposed SSH port.
3. Copy the output from  cowrie/var/log/cowrie/cowrie.json
   into  data/raw/cowrie/cowrie.json  (git-ignored).
4. Set COWRIE_LOG_PATH in config.py.

SCHEMA MAPPING (documented for Phase 3)
-----------------------------------------
Cowrie fields --> CICIoT2023 approximate mappings:
  src_ip / dst_ip / dst_port  -->  no direct feature (metadata only)
  duration                    -->  flow_duration, Duration
  protocol (ssh=22, tel=23)   -->  SSH, Telnet, Protocol Type
  num_commands                -->  Number (proxy)
  bytes_in / bytes_out        -->  Tot size, AVG (proxy)
  All other 40 features       -->  filled with 0.0  (unknown from log)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---- import config safely (works even when called standalone) ----
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import ARTIFACTS_DIR                       # noqa: F401
    _COWRIE_LOG_PATH: Optional[Path] = None               # override below
    try:
        from config import COWRIE_LOG_PATH                # type: ignore
        _COWRIE_LOG_PATH = Path(COWRIE_LOG_PATH)
    except ImportError:
        pass
except Exception:
    _COWRIE_LOG_PATH = None


# All 46 feature columns (same order as CICIoT2023)
_ALL_FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration",
    "Rate", "Srate", "Drate", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count",
    "fin_count", "urg_count", "rst_count", "HTTP", "HTTPS", "DNS",
    "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP",
    "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std",
    "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance",
    "Variance", "Weight",
]


def load_cowrie_log(
    log_path: Optional[Path | str] = None,
    label: str = "Cowrie-Honeypot",
) -> pd.DataFrame:
    """
    Parse a Cowrie JSON log and return a DataFrame matching the
    CICIoT2023 feature schema.

    Parameters
    ----------
    log_path : path to cowrie.json.  If None, uses COWRIE_LOG_PATH
               from config.py.  If that is also None / file does not
               exist, returns an empty DataFrame.
    label    : the label string to assign to all honeypot rows.

    Returns
    -------
    pd.DataFrame  (may be empty if no log is found)
    """
    if log_path is None:
        log_path = _COWRIE_LOG_PATH
    if log_path is None:
        log.info("Cowrie: no log path configured - skipping.")
        return _empty_df(label)

    log_path = Path(log_path)
    if not log_path.exists():
        log.warning("Cowrie log not found at %s - skipping.", log_path)
        return _empty_df(label)

    log.info("Cowrie: parsing %s ...", log_path)
    sessions: dict = {}

    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            sid = event.get("session", "unknown")
            if sid not in sessions:
                sessions[sid] = {
                    "start_time": None, "end_time": None,
                    "ssh": 0, "telnet": 0, "num_commands": 0,
                    "bytes_in": 0, "bytes_out": 0,
                }
            s = sessions[sid]

            ts = event.get("timestamp")
            if s["start_time"] is None:
                s["start_time"] = ts
            s["end_time"] = ts

            etype = event.get("eventid", "")
            if "ssh" in etype.lower():
                s["ssh"] = 1
            if "telnet" in etype.lower():
                s["telnet"] = 1
            if "command" in etype.lower():
                s["num_commands"] += 1
            s["bytes_in"]  += int(event.get("bytes_read", 0) or 0)
            s["bytes_out"] += int(event.get("bytes_written", 0) or 0)

    rows = []
    for sid, s in sessions.items():
        try:
            from datetime import datetime
            fmt = "%Y-%m-%dT%H:%M:%S.%f"
            t0  = datetime.strptime(s["start_time"][:26], fmt) if s["start_time"] else None
            t1  = datetime.strptime(s["end_time"][:26],   fmt) if s["end_time"]   else None
            dur = (t1 - t0).total_seconds() if (t0 and t1) else 0.0
        except Exception:
            dur = 0.0

        row = {f: 0.0 for f in _ALL_FEATURES}
        row["flow_duration"] = dur
        row["Duration"]      = dur
        row["SSH"]           = float(s["ssh"])
        row["Telnet"]        = float(s["telnet"])
        row["Protocol Type"] = 22.0 if s["ssh"] else (23.0 if s["telnet"] else 0.0)
        row["Number"]        = float(s["num_commands"])
        total_bytes          = float(s["bytes_in"] + s["bytes_out"])
        row["Tot size"]      = total_bytes
        row["AVG"]           = total_bytes / max(dur, 1.0)
        row["label"]         = label
        rows.append(row)

    if not rows:
        log.info("Cowrie: log parsed but no sessions found.")
        return _empty_df(label)

    df = pd.DataFrame(rows)
    for col in _ALL_FEATURES:
        df[col] = df[col].astype("float32")
    log.info("Cowrie: produced %d rows from %d sessions.", len(df), len(sessions))
    return df


def _empty_df(label: str) -> pd.DataFrame:
    cols = _ALL_FEATURES + ["label"]
    return pd.DataFrame(columns=cols)


# ------------------------------------------------------------------ #
# Benchmark dataset stubs                                             #
# ------------------------------------------------------------------ #

def load_benchmark_stub(dataset_name: str) -> pd.DataFrame:
    """
    STUB for loading alternative benchmark datasets.
    These are NOT implemented yet (Phase 3+).

    Supported future datasets:
      - IDSAI
      - UNSW-NB15
      - NSL-KDD
      - Bot-IoT

    Returns an empty DataFrame with the CICIoT2023 schema.
    """
    supported = ["IDSAI", "UNSW-NB15", "NSL-KDD", "Bot-IoT"]
    if dataset_name not in supported:
        raise ValueError(
            f"Unknown benchmark dataset: {dataset_name!r}.  "
            f"Supported: {supported}"
        )
    log.warning(
        "Benchmark dataset '%s' is a stub - not implemented until Phase 3.",
        dataset_name,
    )
    return pd.DataFrame(columns=_ALL_FEATURES + ["label"])
