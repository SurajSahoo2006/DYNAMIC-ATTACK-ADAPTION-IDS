"""
orchestrator/llm_orchestrator.py
================================
Phase 3 - LLM Semantic Orchestrator

Sits between Feature Selection and Classifier triage in the architecture diagram:
  - Consumes flow telemetry (top unscaled features), predicted class from Ensemble,
    confidence score, and deterministic severity.
  - Queries local Ollama (Llama-3.2-3B-Instruct / Llama-3-8B-Instruct).
  - Falls back gracefully to deterministic rule-based triage if Ollama is offline.
  - STRICT DESIGN RULE: The ensemble determines attack_class and config determines severity.
    The LLM never hallucinates or overrides class or severity.
  - STRICT JSON output:
    {
      "attack_class": str,
      "severity": str,
      "explanation": str,
      "recommended_action": str
    }
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SCALER_PATH,
    SELECTED_FEATURES_PATH,
    get_flow_severity,
)
from ingest.loader import FEATURE_COLS

log = logging.getLogger("llm_orchestrator")


# ----------------------------------------------------------------------
# Deterministic Fallback Explanations & Actions
# ----------------------------------------------------------------------
RULE_BASED_KNOWLEDGE: Dict[str, Dict[str, str]] = {
    "Benign": {
        "explanation": (
            "Normal routine network communication exhibiting standard protocol handshakes, "
            "baseline packet sizes, and expected inter-arrival timing without anomalous flood surges."
        ),
        "recommended_action": (
            "Maintain passive network monitoring. No host isolation required; "
            "archive flow telemetry for ongoing baseline traffic profiling."
        ),
    },
    "DDoS": {
        "explanation": (
            "Distributed Denial of Service (DDoS) flood detected. Volumetric packet influx "
            "and abnormal flag fragmentation indicating multi-source bandwidth exhaustion attack."
        ),
        "recommended_action": (
            "Engage upstream BGP Anycast traffic scrubbing, apply edge router ACL rate-limiting "
            "on suspicious UDP/SYN floods, and block offending external subnet ranges."
        ),
    },
    "DoS": {
        "explanation": (
            "Denial of Service (DoS) assault originating from localized sources, targeting "
            "server socket table exhaustion or connection starvation via persistent high-rate requests."
        ),
        "recommended_action": (
            "Activate TCP SYN cookies on edge firewalls, throttle per-IP concurrent connections, "
            "and monitor targeted host resource and memory consumption."
        ),
    },
    "Mirai": {
        "explanation": (
            "Mirai IoT botnet signature detected. Typical telemetry includes GRE/UDP flood surges, "
            "bursty payload patterns, and automated scanning for default credential vulnerabilities."
        ),
        "recommended_action": (
            "Immediately isolate suspected IoT device into a quarantined management VLAN, "
            "terminate unauthorized Telnet/SSH sessions on ports 23/2323, and force firmware credential reset."
        ),
    },
    "Recon": {
        "explanation": (
            "Reconnaissance and network mapping activity detected (port scanning, host discovery, "
            "or ping sweeps) probing for active endpoints, open services, and OS fingerprints."
        ),
        "recommended_action": (
            "Enforce perimeter firewall drop rules for the scanning IP, implement dynamic port knocking "
            "for administrative services, and verify host firewall configuration."
        ),
    },
    "Spoofing": {
        "explanation": (
            "Network spoofing detected (ARP cache poisoning, DNS forgery, or IP impersonation) "
            "attempting traffic redirection, credential interception, or identity evasion."
        ),
        "recommended_action": (
            "Enable Dynamic ARP Inspection (DAI) and DHCP Snooping on local access switches, "
            "deploy unicast Reverse Path Forwarding (uRPF), and flush compromised ARP caches."
        ),
    },
    "Web-based": {
        "explanation": (
            "Web application attack detected (SQL Injection, XSS, Command Injection, or malicious upload) "
            "attempting unauthorized command execution, database extraction, or web server privilege escalation."
        ),
        "recommended_action": (
            "Enforce Web Application Firewall (WAF) inspect-and-drop rules, sanitize input parameters, "
            "invalidate active application sessions, and review web application server access logs."
        ),
    },
    "BruteForce": {
        "explanation": (
            "Authentication brute-force attack detected (systematic dictionary guessing or credential "
            "stuffing) targeting authentication endpoints in rapid sequential attempts."
        ),
        "recommended_action": (
            "Temporarily lock targeted account identities, enforce progressive exponential backoff and CAPTCHA "
            "rate limiting on login endpoints, and mandate Multi-Factor Authentication (MFA)."
        ),
    },
}


class FeatureUnscaler:
    """Unscales the 19 selected features back to original physical measurements."""

    def __init__(self):
        if not SCALER_PATH.exists() or not SELECTED_FEATURES_PATH.exists():
            raise FileNotFoundError("Scaler or selected features artifact missing.")
        self.scaler = joblib.load(SCALER_PATH)
        with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
            self.selected_features: List[str] = json.load(f)

        self.feature_indices = [FEATURE_COLS.index(f) for f in self.selected_features]
        self.means = np.array([self.scaler.mean_[i] for i in self.feature_indices], dtype=np.float32)
        self.scales = np.array([self.scaler.scale_[i] for i in self.feature_indices], dtype=np.float32)

    def unscale(self, df_19: pd.DataFrame) -> pd.DataFrame:
        """Unscales standardized 19 features back to original scale."""
        arr = df_19[self.selected_features].to_numpy(dtype=np.float32)
        unscaled = arr * self.scales + self.means
        return pd.DataFrame(unscaled, columns=self.selected_features, index=df_19.index)


class LLMSemanticOrchestrator:
    """
    Manages semantic triage, natural language explanation, and actionable mitigation
    for classified network flows.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model_name: str = OLLAMA_MODEL,
        request_timeout: float = 8.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = request_timeout
        self.unscaler = FeatureUnscaler()
        self.ollama_online = self._check_ollama_status()

    def _check_ollama_status(self) -> bool:
        """Tests whether local Ollama server is active and accessible."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                models = [m.get("name") for m in resp.json().get("models", [])]
                log.info(f"Ollama server connected at {self.base_url}. Available models: {models}")
                return True
        except Exception:
            pass
        log.warning(
            f"Ollama server offline at {self.base_url}. Operating in deterministic rule-based fallback mode."
        )
        return False

    def generate_flow_summary(
        self,
        flow_features_unscaled: Dict[str, float],
        attack_class: str,
        confidence: float,
        severity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates semantic triage and mitigation recommendation for a single flow.

        Returns strict JSON schema:
          {
            "attack_class": str,
            "severity": str,
            "explanation": str,
            "recommended_action": str,
            "_source": "ollama" | "rule_based_fallback",
            "_latency_sec": float
          }
        """
        t0 = time.time()
        det_severity = severity or get_flow_severity(attack_class, confidence)

        # Attempt Ollama inference if online
        if self.ollama_online:
            try:
                res = self._call_ollama(
                    flow_features_unscaled, attack_class, confidence, det_severity
                )
                if res is not None:
                    res["_source"] = "ollama"
                    res["_latency_sec"] = round(time.time() - t0, 3)
                    return res
            except Exception as e:
                log.warning(f"Ollama request failed: {e}. Falling back to rule-based triage.")

        # Deterministic rule-based fallback
        res = self._fallback_triage(attack_class, det_severity)
        res["_source"] = "rule_based_fallback"
        res["_latency_sec"] = round(time.time() - t0, 4)
        return res

    def _fallback_triage(self, attack_class: str, severity: str) -> Dict[str, str]:
        """Generates deterministic expert triage from internal knowledge base."""
        knowledge = RULE_BASED_KNOWLEDGE.get(
            attack_class,
            {
                "explanation": f"Suspicious anomalous network pattern matching {attack_class} profile.",
                "recommended_action": "Inspect packet headers and isolate source host if malicious behavior persists.",
            },
        )
        return {
            "attack_class": attack_class,
            "severity": severity,
            "explanation": knowledge["explanation"],
            "recommended_action": knowledge["recommended_action"],
        }

    def _call_ollama(
        self,
        flow_features: Dict[str, float],
        attack_class: str,
        confidence: float,
        severity: str,
    ) -> Optional[Dict[str, str]]:
        """Invokes Ollama generate API with strict JSON formatting."""
        # Top distinctive telemetry keys to provide context without overloading token length
        context_features = {
            k: round(v, 2)
            for k, v in flow_features.items()
            if k in [
                "flow_duration",
                "Header_Length",
                "Protocol Type",
                "Tot size",
                "AVG",
                "syn_count",
                "ack_count",
                "ICMP",
                "IAT",
            ]
        }

        prompt = (
            f"You are an expert Network Intrusion Detection Analyst.\n"
            f"An ensemble machine learning IDS classified this network flow:\n"
            f"- Classified Attack Type: {attack_class}\n"
            f"- Model Confidence: {confidence:.2%}\n"
            f"- Severity: {severity}\n"
            f"- Key Network Telemetry: {json.dumps(context_features)}\n\n"
            f"TASK: Provide a concise executive explanation (2 sentences max) of why this attack occurs and "
            f"a specific technical mitigation action (1-2 sentences) for network engineers.\n"
            f"CRITICAL RULES:\n"
            f"1. You MUST set 'attack_class' to exactly '{attack_class}'.\n"
            f"2. You MUST set 'severity' to exactly '{severity}'.\n"
            f"3. Respond ONLY in valid JSON matching this schema:\n"
            f'{{"attack_class": "{attack_class}", "severity": "{severity}", "explanation": "...", "recommended_action": "..."}}'
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2, "num_predict": 150},
        }

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code == 200:
            raw_text = resp.json().get("response", "{}")
            parsed = json.loads(raw_text)
            # Enforce non-overridable ensemble labels
            parsed["attack_class"] = attack_class
            parsed["severity"] = severity
            if "explanation" in parsed and "recommended_action" in parsed:
                return parsed
        return None
