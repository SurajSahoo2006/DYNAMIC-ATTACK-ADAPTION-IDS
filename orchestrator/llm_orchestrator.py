"""
orchestrator/llm_orchestrator.py
================================
Phase 3 - LLM Semantic Orchestrator

Sits AFTER the Ensemble Classifier in the architecture flow:
  - Consumes flow telemetry (top unscaled features), predicted class from Ensemble,
    confidence score, and deterministic severity.
  - Queries local Ollama (Llama-3.2-3B-Instruct).
  - Enforces temperature 0, strict JSON format, and retry on invalid JSON.
  - Falls back to deterministic rule-based expert knowledge if Ollama is offline or fails.
  - Invariant: Ensemble determines attack_class and config table determines severity.
    If the LLM outputs different values, they are overwritten and a 'model_overrode_fields' event is recorded.
  - Output JSON:
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
    for classified network flows. Positioned AFTER the Ensemble Classifier.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model_name: str = OLLAMA_MODEL,
        request_timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = request_timeout
        self.unscaler = FeatureUnscaler()
        self.ollama_online = self._check_ollama_status()
        self.model_overrode_fields_count = 0

    def _check_ollama_status(self) -> bool:
        """Tests whether local Ollama server is active and accessible."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
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
        is_last_call: bool = False,
        shap_top_features: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generates semantic triage and mitigation recommendation for a single flow.

        Returns strict JSON schema:
          {
            "attack_class": str,
            "severity": str,
            "confidence_band": str,
            "explanation": str,
            "recommended_action": str,
            "source": "ollama" | "rule_based_fallback",
            "fallback": bool,
            "overrode_fields": bool,
            "latency_sec": float,
            "used_shap": bool,
            "telemetry_provided": dict
          }
        """
        t0 = time.time()
        det_severity = severity or get_flow_severity(attack_class, confidence)
        conf_band = "low" if confidence < 0.60 else ("medium" if confidence < 0.85 else "high")

        # Attempt Ollama inference if online
        if self.ollama_online:
            res, raw_source = self._query_ollama_with_retry(
                flow_features_unscaled=flow_features_unscaled,
                attack_class=attack_class,
                confidence=confidence,
                severity=det_severity,
                is_last_call=is_last_call,
                shap_top_features=shap_top_features,
            )
            if res is not None:
                overrode = False
                # Verify non-overridable ensemble values
                if res.get("attack_class") != attack_class or res.get("severity") != det_severity:
                    overrode = True
                    self.model_overrode_fields_count += 1
                    res["attack_class"] = attack_class
                    res["severity"] = det_severity

                res["confidence_band"] = conf_band
                res["source"] = "ollama"
                res["fallback"] = False
                res["overrode_fields"] = overrode
                res["used_shap"] = bool(shap_top_features)
                res["latency_sec"] = round(time.time() - t0, 3)
                return res

        # Deterministic rule-based fallback
        res = self._fallback_triage(attack_class, det_severity, confidence)
        res["confidence_band"] = conf_band
        res["source"] = "rule_based_fallback"
        res["fallback"] = True
        res["overrode_fields"] = False
        res["used_shap"] = bool(shap_top_features)
        res["latency_sec"] = round(time.time() - t0, 4)
        return res

    def _fallback_triage(self, attack_class: str, severity: str, confidence: float = 1.0) -> Dict[str, str]:
        """Generates deterministic expert triage from internal knowledge base."""
        knowledge = RULE_BASED_KNOWLEDGE.get(
            attack_class,
            {
                "explanation": f"Suspicious anomalous network pattern matching {attack_class} profile.",
                "recommended_action": "Inspect packet headers and isolate source host if malicious behavior persists.",
            },
        )
        explanation = knowledge["explanation"]
        if confidence < 0.60:
            explanation = f"Result uncertain (classifier confidence is low; potential false positive). {explanation}"
        return {
            "attack_class": attack_class,
            "severity": severity,
            "explanation": explanation,
            "recommended_action": knowledge["recommended_action"],
        }

    def _query_ollama_with_retry(
        self,
        flow_features_unscaled: Dict[str, float],
        attack_class: str,
        confidence: float,
        severity: str,
        is_last_call: bool,
        shap_top_features: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[Dict[str, str]], str]:
        """
        Tries to call Ollama. If response JSON is invalid, retries once.
        Returns (parsed_dict, "ollama") or (None, "fallback").
        """
        # Attempt 1
        res = self._call_ollama(
            flow_features=flow_features_unscaled,
            attack_class=attack_class,
            confidence=confidence,
            severity=severity,
            is_last_call=is_last_call,
            shap_top_features=shap_top_features,
        )
        if res is not None:
            return res, "ollama"

        # Attempt 2 (Retry once on invalid JSON / error)
        log.warning("Ollama first attempt failed for flow. Retrying once...")
        res = self._call_ollama(
            flow_features=flow_features_unscaled,
            attack_class=attack_class,
            confidence=confidence,
            severity=severity,
            is_last_call=is_last_call,
            shap_top_features=shap_top_features,
        )
        if res is not None:
            return res, "ollama"

        return None, "fallback"

    def _call_ollama(
        self,
        flow_features: Dict[str, float],
        attack_class: str,
        confidence: float,
        severity: str,
        is_last_call: bool,
        shap_top_features: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, str]]:
        """Invokes Ollama generate API with temperature 0 and strict JSON formatting."""
        if shap_top_features:
            # Use real top SHAP features and their unscaled physical values
            context_features = {
                item["feature"]: round(float(item["unscaled_value"]), 2)
                for item in shap_top_features
            }
        else:
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

        # Determine confidence band
        if confidence < 0.60:
            confidence_band = "low"
        elif confidence < 0.85:
            confidence_band = "medium"
        else:
            confidence_band = "high"

        # Low confidence uncertainty instruction
        uncertainty_instruction = ""
        if confidence < 0.60:
            uncertainty_instruction = (
                "\nCRITICAL: The classifier is NOT confident. "
                "Say the result is uncertain and may be a false positive. "
                "Do not describe the attack as certain.\n"
            )

        prompt = (
            f"You are an expert Network Intrusion Detection Analyst.\n"
            f"The ensemble machine learning IDS classified this network flow:\n"
            f"- Classified Attack Type: {attack_class}\n"
            f"- Model Confidence: {confidence:.2%}\n"
            f"- Confidence Band: {confidence_band}\n"
            f"- Severity: {severity}\n"
            f"- Key Network Telemetry: {json.dumps(context_features)}\n\n"
            f"TASK: Using ONLY the facts given above:\n"
            f"1. Write an executive explanation (maximum 2 sentences) of what this classification means. "
            f"Do not name or invent any feature values or attributes that were not provided in Key Network Telemetry.\n"
            f"2. Write a recommended mitigation action (maximum 1 sentence) for network operators.\n"
            f"{uncertainty_instruction}\n"
            f"OUTPUT FORMAT: Return a valid JSON object with EXACTLY these four keys:\n"
            f'{{\n'
            f'  "attack_class": "{attack_class}",\n'
            f'  "severity": "{severity}",\n'
            f'  "explanation": "...",\n'
            f'  "recommended_action": "..."\n'
            f'}}'
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_predict": 120,
            },
        }
        if is_last_call:
            payload["keep_alive"] = 0

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                raw_text = resp.json().get("response", "{}").strip()
                parsed = json.loads(raw_text)
                if isinstance(parsed, dict) and "explanation" in parsed and "recommended_action" in parsed:
                    return parsed
        except Exception as e:
            log.warning(f"Ollama call exception: {e}")

        return None
