"""
adversarial/constraints.py
==========================
Phase 4 - Adversarial Perturbation Domain Constraints

Defines realistic domain constraints for the 19 CICIoT2023 flow features:
  - Feature type: binary_flag, protocol_code, count, continuous
  - Empirical training range: [min_val, max_val]
  - Modifiable by attacker: True / False
  - Domain justification for why each feature can or cannot be perturbed.

Enforces:
  1. Fixed flag/protocol features cannot be modified by the attacker.
  2. Modifiable features (timing and packet size padding) must be clipped to [min, max].
  3. Physical integer features are rounded to whole numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConstraint:
    name: str
    feature_type: str        # 'binary_flag', 'protocol_code', 'count', 'continuous'
    min_val: float
    max_val: float
    modifiable: bool
    rationale: str


FEATURE_CONSTRAINTS: Dict[str, FeatureConstraint] = {
    "flow_duration": FeatureConstraint(
        name="flow_duration",
        feature_type="continuous",
        min_val=0.0,
        max_val=99435.77,
        modifiable=True,
        rationale="Attacker can intentionally delay packets, hold connections open, or introduce idle intervals."
    ),
    "Header_Length": FeatureConstraint(
        name="Header_Length",
        feature_type="continuous",
        min_val=0.0,
        max_val=9865998.0,
        modifiable=True,
        rationale="Attacker can insert IP/TCP options, pad headers, or manipulate fragmentation headers."
    ),
    "Protocol Type": FeatureConstraint(
        name="Protocol Type",
        feature_type="protocol_code",
        min_val=0.0,
        max_val=47.0,
        modifiable=False,
        rationale="Transport protocol (TCP=6, UDP=17, ICMP=1) is dictated by the attack; cannot be forged without breaking transport semantics."
    ),
    "fin_flag_number": FeatureConstraint(
        name="fin_flag_number",
        feature_type="binary_flag",
        min_val=0.0,
        max_val=1.0,
        modifiable=False,
        rationale="Setting FIN prematurely terminates active TCP connection state; cannot be arbitrarily flipped."
    ),
    "syn_count": FeatureConstraint(
        name="syn_count",
        feature_type="count",
        min_val=0.0,
        max_val=10.4,
        modifiable=False,
        rationale="Handshake rate counter; in IoT attacks (e.g., SYN floods), altering SYN count changes the attack identity itself."
    ),
    "ack_count": FeatureConstraint(
        name="ack_count",
        feature_type="count",
        min_val=0.0,
        max_val=5.0,
        modifiable=False,
        rationale="Governed strictly by TCP acknowledgment handshake mechanics."
    ),
    "syn_flag_number": FeatureConstraint(
        name="syn_flag_number",
        feature_type="binary_flag",
        min_val=0.0,
        max_val=1.0,
        modifiable=False,
        rationale="Binary connection initiation flag; fixed by protocol standard."
    ),
    "urg_count": FeatureConstraint(
        name="urg_count",
        feature_type="count",
        min_val=0.0,
        max_val=3600.5,
        modifiable=False,
        rationale="Urgent pointer flag counter; restricted by OS TCP/IP stack behavior."
    ),
    "rst_count": FeatureConstraint(
        name="rst_count",
        feature_type="count",
        min_val=0.0,
        max_val=9126.5,
        modifiable=False,
        rationale="Reset flag counter abruptly closes sockets; cannot be injected arbitrarily."
    ),
    "ICMP": FeatureConstraint(
        name="ICMP",
        feature_type="binary_flag",
        min_val=0.0,
        max_val=1.0,
        modifiable=False,
        rationale="Indicates ICMP protocol encapsulation; cannot be flipped for TCP/UDP flows."
    ),
    "Tot size": FeatureConstraint(
        name="Tot size",
        feature_type="continuous",
        min_val=42.0,
        max_val=5091.6,
        modifiable=True,
        rationale="Attacker can easily append arbitrary padding bytes or chunk data into custom buffer sizes."
    ),
    "IAT": FeatureConstraint(
        name="IAT",
        feature_type="continuous",
        min_val=0.0,
        max_val=167639424.0,
        modifiable=True,
        rationale="Inter-Arrival Time is directly controllable via transmission pacing, sleep intervals, or packet jitter."
    ),
    "AVG": FeatureConstraint(
        name="AVG",
        feature_type="continuous",
        min_val=42.0,
        max_val=8490.74,
        modifiable=True,
        rationale="Average packet size is shifted directly by inserting dummy payload padding."
    ),
    "Max": FeatureConstraint(
        name="Max",
        feature_type="continuous",
        min_val=42.0,
        max_val=21786.0,
        modifiable=True,
        rationale="Maximum packet size is controllable by sending a larger padded packet within MTU limits."
    ),
    "Min": FeatureConstraint(
        name="Min",
        feature_type="continuous",
        min_val=42.0,
        max_val=4410.0,
        modifiable=True,
        rationale="Minimum packet size can be increased by padding all smaller packets up to a target size."
    ),
    "Tot sum": FeatureConstraint(
        name="Tot sum",
        feature_type="continuous",
        min_val=81.0,
        max_val=61882.6,
        modifiable=True,
        rationale="Total volume transmitted over the flow window is altered by supplementary padding."
    ),
    "Magnitue": FeatureConstraint(
        name="Magnitue",
        feature_type="continuous",
        min_val=9.165,
        max_val=124.95,
        modifiable=True,
        rationale="Geometric magnitude of flow packet dimensions scales directly with payload padding."
    ),
    "Covariance": FeatureConstraint(
        name="Covariance",
        feature_type="continuous",
        min_val=0.0,
        max_val=43075336.0,
        modifiable=True,
        rationale="Packet size and timing covariance is shaped by varying padding and packet inter-arrival bursts."
    ),
    "Variance": FeatureConstraint(
        name="Variance",
        feature_type="continuous",
        min_val=0.0,
        max_val=1.0,
        modifiable=True,
        rationale="Packet size variance ratio is tuned by mixing uniform and variable-length padded frames."
    ),
}


class AdversarialConstraintEnforcer:
    """
    Applies physical and domain constraints to adversarial perturbations.
    """

    def __init__(self, feature_names: List[str], unscaler, scaler):
        self.feature_names = feature_names
        self.unscaler = unscaler
        self.scaler = scaler

        self.modifiable_indices = [
            i for i, f in enumerate(feature_names) if FEATURE_CONSTRAINTS[f].modifiable
        ]
        self.fixed_indices = [
            i for i, f in enumerate(feature_names) if not FEATURE_CONSTRAINTS[f].modifiable
        ]

    def enforce_constraints(
        self,
        X_adv_scaled: np.ndarray,
        X_clean_scaled: np.ndarray,
    ) -> np.ndarray:
        """
        Enforces constraints on perturbed scaled features:
        1. Resets fixed features back to clean values.
        2. Unscales to physical units, clips to [min_val, max_val], rounds counts.
        3. Re-scales back to standardized space.

        Args:
            X_adv_scaled: array of shape (N, 19) in standardized feature space
            X_clean_scaled: array of shape (N, 19) original unperturbed scaled samples

        Returns:
            np.ndarray of shape (N, 19) strictly compliant with realistic domain constraints.
        """
        X_adv_mod = X_adv_scaled.copy()

        # 1. Reset non-modifiable features strictly to original clean values
        X_adv_mod[:, self.fixed_indices] = X_clean_scaled[:, self.fixed_indices]

        # 2. Unscale to physical domain
        df_scaled = pd.DataFrame(X_adv_mod, columns=self.feature_names)
        df_unscaled = self.unscaler.unscale(df_scaled)

        # 3. Clip modifiable features to empirical min/max and enforce types
        for feat in self.feature_names:
            c = FEATURE_CONSTRAINTS[feat]
            if c.modifiable:
                df_unscaled[feat] = np.clip(df_unscaled[feat], c.min_val, c.max_val)
                if c.feature_type == "count":
                    df_unscaled[feat] = np.round(df_unscaled[feat])
            else:
                # Ensure clean value is untouched
                pass

        # 4. Transform back to standardized feature space using scaler
        # We need the full 46-feature array representation or transform the 19 features directly
        # Since standard scaling is coordinate-wise: x_scaled = (x_unscaled - mean) / scale
        means = self.unscaler.means
        scales = self.unscaler.scales
        arr_unscaled = df_unscaled[self.feature_names].to_numpy(dtype=np.float32)
        X_compliant_scaled = (arr_unscaled - means) / scales

        # Final check: fixed columns must strictly match clean scaled
        X_compliant_scaled[:, self.fixed_indices] = X_clean_scaled[:, self.fixed_indices]

        return X_compliant_scaled


def get_constraint_summary_table() -> pd.DataFrame:
    """Returns a formatted DataFrame of all 19 feature constraints."""
    rows = []
    for f_name, c in FEATURE_CONSTRAINTS.items():
        rows.append({
            "Feature Name": f_name,
            "Type": c.feature_type,
            "Allowed Min": f"{c.min_val:.2f}",
            "Allowed Max": f"{c.max_val:.2f}",
            "Attacker Modifiable": "YES (Plausible)" if c.modifiable else "NO (Fixed)",
            "Justification / Defense Insight": c.rationale,
        })
    return pd.DataFrame(rows)
