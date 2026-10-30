"""
ingest/pcap_demo.py
===================
Phase 1 - PCAP Demo  (clearly marked as DEMO, not part of main pipeline)
===========================================================================

PURPOSE
-------
This file demonstrates the "Network Traffic Capture" input box from
the architecture diagram.  It shows that raw .pcap files CAN be read
and that the same 46 features used by the CSV pipeline can in principle
be derived from packets.

This is a STANDALONE DEMO.  It is NOT called by run_phase1_2.py by
default.  Run it manually:

    python ingest/pcap_demo.py path/to/small.pcap

IMPORTANT CAVEATS
-----------------
1. Full flow-feature extraction (IAT, covariance, radius ...) requires a
   flow-aggregation layer (e.g. CICFlowMeter).  This demo only extracts
   the SIMPLE per-packet fields as a proof of concept.
2. Scapy is required:  pip install scapy
3. The output schema is APPROXIMATE - it will not match the CICIoT2023
   feature set exactly without CICFlowMeter running on the same PCAP.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def extract_pcap_features(pcap_path: str | Path) -> "pd.DataFrame":   # noqa: F821
    """
    Extract basic per-packet features from a PCAP file using Scapy.
    Returns a DataFrame with a subset of CICIoT2023-compatible columns.
    """
    try:
        from scapy.all import rdpcap, IP, TCP, UDP, ICMP  # type: ignore
    except ImportError:
        raise ImportError(
            "Scapy is not installed.  Run:  pip install scapy\n"
            "Then try again."
        )

    import pandas as pd

    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    print(f"\n[PCAP DEMO] Reading: {pcap_path}")
    packets = rdpcap(str(pcap_path))
    print(f"[PCAP DEMO] Loaded {len(packets)} packets")

    rows = []
    for pkt in packets:
        row: dict = {col: 0.0 for col in _PCAP_COLS}

        if pkt.haslayer("IP"):
            ip = pkt["IP"]
            row["IPv"]           = 1.0
            row["Protocol Type"] = float(ip.proto)
            row["Tot size"]      = float(len(pkt))
            row["Header_Length"] = float(ip.ihl * 4) if hasattr(ip, "ihl") else 20.0

        if pkt.haslayer("TCP"):
            tcp = pkt["TCP"]
            row["TCP"]              = 1.0
            row["fin_flag_number"]  = float(bool(tcp.flags & 0x01))
            row["syn_flag_number"]  = float(bool(tcp.flags & 0x02))
            row["rst_flag_number"]  = float(bool(tcp.flags & 0x04))
            row["psh_flag_number"]  = float(bool(tcp.flags & 0x08))
            row["ack_flag_number"]  = float(bool(tcp.flags & 0x10))
            row["urg_count"]        = float(bool(tcp.flags & 0x20))
            row["ece_flag_number"]  = float(bool(tcp.flags & 0x40))
            row["cwr_flag_number"]  = float(bool(tcp.flags & 0x80))

        if pkt.haslayer("UDP"):
            row["UDP"] = 1.0
        if pkt.haslayer("ICMP"):
            row["ICMP"] = 1.0
        if pkt.haslayer("DNS"):
            row["DNS"] = 1.0

        rows.append(row)

    df = pd.DataFrame(rows, columns=_PCAP_COLS)
    print(f"[PCAP DEMO] Extracted {len(df)} rows x {len(df.columns)} cols")
    print(df.head())
    print("\n[PCAP DEMO] NOTE: Flow-level features (IAT, Covariance, Rate ...)")
    print("           require CICFlowMeter.  These are packet-level only.")
    return df


# Columns we CAN fill from raw packets (subset of 46)
_PCAP_COLS = [
    "Header_Length", "Protocol Type", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "urg_count", "DNS", "TCP", "UDP", "ICMP", "IPv",
    "Tot size",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python ingest/pcap_demo.py path/to/sample.pcap")
        sys.exit(1)
    extract_pcap_features(sys.argv[1])
