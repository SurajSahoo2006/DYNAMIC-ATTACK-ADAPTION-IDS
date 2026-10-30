"""
ids_system/pipeline.py
======================
Unified Intrusion Detection & Tactical Response Pipeline (Phase 13).

Consolidates all subsystems from Phases 1 through 12 into a single, cohesive
production architecture:
  - Feature Selection & Scaling (Phase 1 & 2)
  - Raw B2 Ensemble Inference (Phase 3 & 11)
  - Isotonic Probability Calibration (Phase 12)
  - Autoencoder Anomaly Scoring (Phase 10)
  - Unified Fusion Decision Rule (Phase 10 & 12)
  - SHAP Explanations & LLM Incident Briefing (Phase 3 & 4)
  - Greedy Multi-Agent Tactical Response (Phase 5)
  - Cryptographic Forensic Alert Logging & Verification (Phase 4 & 5)
"""

from __future__ import annotations

import collections
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import psutil
import torch

# Enforce single-threaded deterministic CPU execution across libraries
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration and utility imports
from config import (
    ENCODER_8_PATH,
    MODELS_DIR,
    SCALER_PATH,
    SELECTED_FEATURES_PATH,
    get_flow_severity,
)
from config_marl import (
    CONTAINMENT_ACTIONS,
    CONTAINMENT_ACTION_TO_IDX,
    ENV_FEATURE_DIM,
    EPISODE_STEPS,
    ESCALATION_ACTIONS,
    ESCALATION_ACTION_TO_IDX,
    ESCALATION_BUDGET,
    MARL_MODELS_DIR,
    NUM_AGENTS,
    POSTURE_DURATION,
    TRIAGE_ACTIONS,
    TRIAGE_ACTION_TO_IDX,
)

from alerts.alert_logger import ForensicAlertLogger, GENESIS_HASH
from alerts.verify_log import verify_alert_log
from calibration.fit import IsotonicCalibrator, TemperatureScaler
from marl.mappo import MAPPOTeam
from models.ensemble import EnsembleClassifier, PyTorchDNN
from orchestrator.llm_orchestrator import FeatureUnscaler, LLMSemanticOrchestrator
from predict import get_or_load_model, get_selected_features
from xai.shap_explain import SHAPExplainer
from zeroday.train_autoencoder import BenignAutoencoder

# Ensure unpickling finds IsotonicCalibrator even if serialized under __main__
sys.modules["__main__"].IsotonicCalibrator = IsotonicCalibrator
sys.modules["__main__"].TemperatureScaler = TemperatureScaler

log = logging.getLogger("ids_pipeline")


class IDSPipeline:
    """
    Consolidated IDS Pipeline executing end-to-end telemetry ingestion,
    classification, calibration, zero-day anomaly flagging, explainability,
    tactical response, and tamper-evident audit logging.
    """

    # Static component metadata for health auditing
    COMPONENT_METADATA = {
        "scaler": {
            "file": "artifacts/standard_scaler.pkl",
            "phase_script": "models/train_phase2.py",
        },
        "selected_features": {
            "file": "artifacts/selected_features.json",
            "phase_script": "models/feature_selection.py",
        },
        "b2_ensemble": {
            "file": "models/rf_8class_b2.joblib",
            "phase_script": "models/wire_b2_models.py",
        },
        "calibrator": {
            "file": "calibration/models/chosen_calibrator.joblib",
            "phase_script": "calibration/fit.py",
        },
        "autoencoder": {
            "file": "zeroday/models/autoencoder.pt",
            "phase_script": "zeroday/train_autoencoder.py",
        },
        "shap_explainer": {
            "file": "models/xgb_8class_b2.joblib",
            "phase_script": "xai/shap_explain.py",
        },
        "llm_orchestrator": {
            "file": "orchestrator/llm_orchestrator.py",
            "phase_script": "orchestrator/llm_orchestrator.py",
        },
        "marl_policy": {
            "file": "models/marl/mappo_balanced_seed42_best.pt",
            "phase_script": "marl/train.py",
        },
        "marl_cache": {
            "file": "data/processed/marl_cache_test.parquet",
            "phase_script": "marl/build_cache.py",
        },
        "alert_logger": {
            "file": "logs/alerts_live.jsonl",
            "phase_script": "alerts/alert_logger.py",
        },
    }

    def __init__(self, target: str = "8class", model_type: str = "b2"):
        self.target = target
        self.model_type = model_type

        # Lazy component references
        self._scaler = None
        self._features = None
        self._class_names = None
        self._benign_idx = None
        self._ensemble = None
        self._calibrator = None
        self._calibrated_cutoff = None
        self._autoencoder = None
        self._ae_threshold = None
        self._shap_explainer = None
        self._orchestrator = None
        self._marl_team = None
        self._alert_logger = None

        # Live alert paths
        self.live_log_path = PROJECT_ROOT / "logs" / "alerts_live.jsonl"
        self.live_tail_hash_path = PROJECT_ROOT / "logs" / "alerts_live_tail_hash.txt"

    # ------------------------------------------------------------------
    # Lazy Loading Properties
    # ------------------------------------------------------------------
    @property
    def features(self) -> List[str]:
        if self._features is None:
            self._features = get_selected_features()
        return self._features

    @property
    def class_names(self) -> List[str]:
        if self._class_names is None:
            le_8 = joblib.load(ENCODER_8_PATH)
            self._class_names = list(le_8.classes_)
        return self._class_names

    @property
    def benign_idx(self) -> int:
        if self._benign_idx is None:
            self._benign_idx = int(np.where(np.array(self.class_names) == "Benign")[0][0])
        return self._benign_idx

    @property
    def scaler(self):
        if self._scaler is None:
            if not SCALER_PATH.exists():
                raise FileNotFoundError(f"Scaler missing at {SCALER_PATH}")
            self._scaler = joblib.load(SCALER_PATH)
        return self._scaler

    @property
    def scaler_params_19(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts (mean, scale) for the 19 features from the 46-feature StandardScaler."""
        if not hasattr(self, "_scaler_means_19") or self._scaler_means_19 is None:
            feature_names_in = list(getattr(self.scaler, "feature_names_in_", []))
            if feature_names_in:
                feat_to_idx = {name: i for i, name in enumerate(feature_names_in)}
                indices = [feat_to_idx[f] for f in self.features if f in feat_to_idx]
                self._scaler_means_19 = self.scaler.mean_[indices].astype(np.float64)
                self._scaler_scales_19 = self.scaler.scale_[indices].astype(np.float64)
            else:
                self._scaler_means_19 = self.scaler.mean_[:len(self.features)].astype(np.float64)
                self._scaler_scales_19 = self.scaler.scale_[:len(self.features)].astype(np.float64)
        return self._scaler_means_19, self._scaler_scales_19

    def scale_19(self, df_19: pd.DataFrame) -> np.ndarray:
        """Scales the 19 selected features using the saved scaler's exact parameters."""
        means, scales = self.scaler_params_19
        raw_arr = df_19[self.features].to_numpy(dtype=np.float64)
        scaled = (raw_arr - means) / scales
        return scaled.astype(np.float32)

    @property
    def ensemble(self) -> EnsembleClassifier:
        if self._ensemble is None:
            self._ensemble = get_or_load_model(target=self.target, model_type=self.model_type)
        return self._ensemble

    @property
    def calibrator(self) -> IsotonicCalibrator:
        if self._calibrator is None:
            calib_path = PROJECT_ROOT / "calibration" / "models" / "chosen_calibrator.joblib"
            if not calib_path.exists():
                raise FileNotFoundError(f"Calibrator missing at {calib_path}")
            self._calibrator = joblib.load(calib_path)
        return self._calibrator

    @property
    def calibrated_cutoff(self) -> float:
        """
        Validation-chosen confidence cutoff corresponding to the 10.0% Benign FPR budget (Phase 12).
        Defaults to 0.521507 derived from validation data.
        """
        if self._calibrated_cutoff is None:
            summary_path = PROJECT_ROOT / "calibration" / "results" / "openset_calibrated_summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    # Check known_control or budget if stored
                    self._calibrated_cutoff = 0.521507
                except Exception:
                    self._calibrated_cutoff = 0.521507
            else:
                self._calibrated_cutoff = 0.521507
        return self._calibrated_cutoff

    @property
    def autoencoder(self) -> Tuple[BenignAutoencoder, float]:
        """Loads Benign Autoencoder and the 97.5th percentile reconstruction threshold."""
        if self._autoencoder is None:
            ae_pt_path = PROJECT_ROOT / "zeroday" / "models" / "autoencoder.pt"
            ae_json_path = PROJECT_ROOT / "zeroday" / "results" / "ae_training.json"
            if not ae_pt_path.exists():
                raise FileNotFoundError(f"Autoencoder weights missing at {ae_pt_pt_path}")

            checkpoint = torch.load(ae_pt_path, map_location="cpu", weights_only=False)
            in_dim = checkpoint.get("in_features", 19)
            ae_model = BenignAutoencoder(in_features=in_dim)
            ae_model.load_state_dict(checkpoint["state_dict"])
            ae_model.eval()

            # 97.5th percentile threshold from Phase 10 FP-budget operating point
            if "thresholds_by_percentile" in checkpoint and "97.5" in checkpoint["thresholds_by_percentile"]:
                thresh = float(checkpoint["thresholds_by_percentile"]["97.5"])
            elif ae_json_path.exists():
                with open(ae_json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                thresh = float(meta["thresholds_by_percentile"]["97.5"])
            else:
                thresh = float(checkpoint.get("threshold", 0.534869))

            self._autoencoder = (ae_model, thresh)
        return self._autoencoder

    @property
    def shap_explainer(self) -> SHAPExplainer:
        if self._shap_explainer is None:
            self._shap_explainer = SHAPExplainer(target=self.target, model_type=self.model_type)
        return self._shap_explainer

    @property
    def orchestrator(self) -> LLMSemanticOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = LLMSemanticOrchestrator()
        return self._orchestrator

    @property
    def marl_team(self) -> MAPPOTeam:
        if self._marl_team is None:
            ckpt_path = MARL_MODELS_DIR / "mappo_balanced_seed42_best.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"MARL policy checkpoint missing at {ckpt_path}")
            self._marl_team = MAPPOTeam()
            self._marl_team.load_checkpoint(ckpt_path)
        return self._marl_team

    @property
    def alert_logger(self) -> ForensicAlertLogger:
        if self._alert_logger is None:
            self._alert_logger = ForensicAlertLogger(
                log_path=self.live_log_path,
                tail_hash_path=self.live_tail_hash_path,
            )
        return self._alert_logger

    # ------------------------------------------------------------------
    # Health Audit
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """
        Audits every system component without raising exceptions.
        Returns component status, required files, phase creating scripts,
        and current process memory (RSS MB).
        """
        proc = psutil.Process()
        rss_mb = round(proc.memory_info().rss / (1024.0 * 1024.0), 2)
        components = {}

        for comp_name, comp_info in self.COMPONENT_METADATA.items():
            req_rel = comp_info["file"]
            req_abs = PROJECT_ROOT / req_rel
            phase_script = comp_info["phase_script"]

            status = "missing"
            error_msg = None

            if req_abs.exists():
                try:
                    if comp_name == "scaler":
                        _ = self.scaler
                    elif comp_name == "selected_features":
                        _ = self.features
                    elif comp_name == "b2_ensemble":
                        _ = self.ensemble
                    elif comp_name == "calibrator":
                        _ = self.calibrator
                    elif comp_name == "autoencoder":
                        _ = self.autoencoder
                    elif comp_name == "shap_explainer":
                        _ = self.shap_explainer
                    elif comp_name == "marl_policy":
                        _ = self.marl_team
                    elif comp_name == "alert_logger":
                        _ = self.alert_logger
                    status = "ok"
                except Exception as ex:
                    status = "error"
                    error_msg = str(ex)
            else:
                if comp_name == "alert_logger":
                    # alerts_live.jsonl is created on first log entry
                    status = "ok"

            entry = {
                "status": status,
                "file": req_rel,
                "phase_script": phase_script,
            }
            if error_msg:
                entry["error"] = error_msg
            components[comp_name] = entry

        # Check LLM Orchestrator status and Ollama availability
        ollama_reachable = False
        try:
            orch = self.orchestrator
            ollama_reachable = bool(orch.ollama_online)
            components["llm_orchestrator"]["status"] = "ok"
            components["llm_orchestrator"]["ollama_reachable"] = ollama_reachable
        except Exception as ex:
            components["llm_orchestrator"]["status"] = "error"
            components["llm_orchestrator"]["error"] = str(ex)
            components["llm_orchestrator"]["ollama_reachable"] = False

        overall_ok = all(
            c.get("status") == "ok" for c in components.values()
        )

        return {
            "overall_status": "ok" if overall_ok else "degraded",
            "process_rss_mb": rss_mb,
            "components": components,
        }

    # ------------------------------------------------------------------
    # Flow Analysis & Decision Rule
    # ------------------------------------------------------------------
    def analyze_flow(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes a single raw (unscaled) flow dictionary.
        Returns single prediction dictionary.
        """
        df_single = pd.DataFrame([raw])
        res_df = self.analyze_batch(df_single)
        return res_df.iloc[0].to_dict()

    def analyze_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch inference and zero-day anomaly detection across network flows.
        Expects unscaled (raw physical) values for the 19 selected features.
        """
        req_features = self.features
        missing = [c for c in req_features if c not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required feature(s): {missing}")

        X_raw = df[req_features].copy()
        for c in req_features:
            if not np.issubdtype(X_raw[c].dtype, np.number):
                raise ValueError(f"Column '{c}' contains non-numeric values.")

        # 1. Scale with saved StandardScaler
        X_scaled = self.scale_19(X_raw)
        X_scaled_df = pd.DataFrame(X_scaled, columns=req_features, index=df.index)

        # 2. Raw B2 Ensemble Probabilities
        probs_dict = self.ensemble.predict_probabilities(X_scaled_df)
        P_raw = probs_dict["weighted"].astype(np.float32)
        raw_confs = np.max(P_raw, axis=1)
        raw_preds_idx = np.argmax(P_raw, axis=1)
        pred_classes = [self.class_names[idx] for idx in raw_preds_idx]

        # 3. Calibrated Probabilities & Calibrated Confidence
        P_cal = self.calibrator.predict_proba(P_raw)
        cal_confs = np.max(P_cal, axis=1)

        # 4. Autoencoder Reconstruction Scoring (Phase 10)
        ae_model, ae_thresh = self.autoencoder
        with torch.no_grad():
            x_t = torch.from_numpy(X_scaled)
            rec = ae_model(x_t)
            ae_scores = ((x_t - rec) ** 2).mean(dim=1).cpu().numpy()
        ae_flags = (ae_scores > ae_thresh)

        # 5. Fusion Decision Rule
        # predicted_class = argmax of RAW B2 probabilities (preserves Phase 3-6 consistency)
        # unknown_flag = ae_flag OR calibrated_confidence < validation 10% budget cutoff
        tau_cal = self.calibrated_cutoff
        unknown_flags = ae_flags | (cal_confs < tau_cal)

        # Deterministic severity
        severities = [get_flow_severity(cls, c) for cls, c in zip(pred_classes, raw_confs)]

        # Generate decision notes
        decision_notes_list = []
        for i in range(len(df)):
            notes = []
            if ae_flags[i]:
                notes.append(f"Autoencoder anomaly (score {ae_scores[i]:.4f} > threshold {ae_thresh:.4f})")
            if cal_confs[i] < tau_cal:
                notes.append(f"Low calibrated confidence ({cal_confs[i]:.4f} < cutoff {tau_cal:.4f})")
            if unknown_flags[i]:
                notes.append("Flagged as UNKNOWN / Zero-Day attack")
            else:
                notes.append(f"Classified as known class '{pred_classes[i]}'")
            decision_notes_list.append("; ".join(notes))

        # Assemble output DataFrame
        out_dict = {
            "predicted_class_idx": raw_preds_idx,
            "predicted_class": pred_classes,
            "raw_confidence": raw_confs.astype(np.float32),
            "calibrated_confidence": cal_confs.astype(np.float32),
            "ae_score": ae_scores.astype(np.float32),
            "ae_flag": ae_flags,
            "unknown_flag": unknown_flags,
            "severity": severities,
            "decision_notes": decision_notes_list,
        }

        # Add 8-class probability columns
        for c_idx, c_name in enumerate(self.class_names):
            out_dict[f"prob_{c_name}"] = P_raw[:, c_idx].astype(np.float32)

        return pd.DataFrame(out_dict, index=df.index)

    # ------------------------------------------------------------------
    # Explainability & Incident Briefing
    # ------------------------------------------------------------------
    def explain(
        self,
        row: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        generate_brief: bool = True,
    ) -> Dict[str, Any]:
        """
        Generates SHAP top-5 signed feature contributions for the predicted class
        and an incident brief from the LLM orchestrator (or template fallback).
        """
        if isinstance(row, pd.DataFrame):
            df_row = row.head(1).copy()
        elif isinstance(row, pd.Series):
            df_row = pd.DataFrame([row.to_dict()])
        else:
            df_row = pd.DataFrame([row])

        req_features = self.features
        raw_dict = {col: float(df_row[col].iloc[0]) for col in req_features}

        # Scale for SHAP TreeExplainer
        X_scaled = self.scale_19(df_row)
        X_scaled_df = pd.DataFrame(X_scaled, columns=req_features)

        # Run analysis to get predicted class
        analyzed = self.analyze_batch(df_row).iloc[0]
        pred_idx = int(analyzed["predicted_class_idx"])
        pred_cls = str(analyzed["predicted_class"])
        raw_conf = float(analyzed["raw_confidence"])
        severity = str(analyzed["severity"])

        # SHAP attribution
        shap_explainer = self.shap_explainer
        shap_vals_matrix = shap_explainer.explain_xgb(X_scaled_df)
        shap_single = shap_vals_matrix[0]  # shape (19, 8)
        top5_features = shap_explainer.get_local_explanation(
            shap_values_single=shap_single,
            unscaled_features_dict=raw_dict,
            target_class_idx=pred_idx,
            top_k=5,
        )

        # LLM Incident Briefing
        if generate_brief:
            summary = self.orchestrator.generate_flow_summary(
                flow_features_unscaled=raw_dict,
                attack_class=pred_cls,
                confidence=raw_conf,
                severity=severity,
                shap_top_features=top5_features,
            )
            brief_source = "llm" if summary.get("source") == "ollama" else "template_fallback"
            brief_text = summary.get("explanation", "")
            if "recommended_action" in summary:
                brief_text += " Recommended Action: " + summary["recommended_action"]
            rec_act = summary.get("recommended_action", "")
        else:
            brief_source = "template_fallback"
            fb = self.orchestrator._fallback_triage(pred_cls, severity, raw_conf)
            brief_text = fb["explanation"]
            rec_act = fb["recommended_action"]
            if rec_act:
                brief_text += " Recommended Action: " + rec_act

        return {
            "predicted_class_idx": pred_idx,
            "predicted_class": pred_cls,
            "raw_confidence": raw_conf,
            "calibrated_confidence": float(analyzed["calibrated_confidence"]),
            "severity": severity,
            "unknown_flag": bool(analyzed["unknown_flag"]),
            "top_features": top5_features,
            "brief": brief_text,
            "brief_source": brief_source,
            "recommended_action": rec_act,
        }

    # ------------------------------------------------------------------
    # MARL Tactical Defense Response
    # ------------------------------------------------------------------
    def respond(self, batch_results: pd.DataFrame) -> pd.DataFrame:
        """
        Executes trained balanced MAPPO policy greedily over analyzed flows in temporal order.
        Returns per-flow Triage, Containment, and Escalation actions.
        """
        prob_cols = [f"prob_{c}" for c in self.class_names]
        missing_prob = [c for c in prob_cols if c not in batch_results.columns]
        if missing_prob or "predicted_class_idx" not in batch_results.columns or "severity" not in batch_results.columns:
            out_df = batch_results.copy()
            out_df["marl_triage"] = "none"
            out_df["marl_containment"] = "none"
            out_df["marl_escalation"] = "none"
            out_df["response_available"] = False
            return out_df

        n_flows = len(batch_results)
        triage_acts = []
        contain_acts = []
        esc_acts = []
        resp_avail = []

        # Episode state variables matching TacticalDefenseEnv dynamics
        active_posture = "none"
        posture_steps_remaining = 0
        escalations_remaining = ESCALATION_BUDGET
        recent_pred_classes: collections.deque = collections.deque(maxlen=5)
        agent_ids = np.eye(NUM_AGENTS, dtype=np.float32)

        sev_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        team = self.marl_team

        probs_matrix = batch_results[prob_cols].to_numpy(dtype=np.float32)
        confs = batch_results["raw_confidence"].to_numpy(dtype=np.float32)
        sev_strs = batch_results["severity"].to_numpy()
        pred_indices = batch_results["predicted_class_idx"].to_numpy(dtype=np.int32)

        for t in range(n_flows):
            # 1. 8 probabilities
            p_flow = probs_matrix[t]
            # 2. Confidence
            c_flow = confs[t]
            # 3. One-hot severity
            sev_onehot = np.zeros(4, dtype=np.float32)
            sev_onehot[sev_map.get(sev_strs[t], 1)] = 1.0
            # 4. Normalized step fraction
            step_frac = float(t) / float(max(n_flows, EPISODE_STEPS))
            # 5. Normalized escalations left
            esc_left_norm = max(0.0, float(escalations_remaining)) / float(ESCALATION_BUDGET)
            # 6. Containment active posture
            posture_active = 1.0 if active_posture != "none" else 0.0
            # 7. Posture steps remaining normalized
            posture_rem_norm = float(posture_steps_remaining) / float(POSTURE_DURATION)
            # 8. Rolling history counts (last 5 flows)
            history_counts = np.zeros(8, dtype=np.float32)
            for p_idx in recent_pred_classes:
                history_counts[p_idx] += 1.0
            history_counts /= 5.0

            # 25-dim environment observation
            env_obs = np.concatenate([
                p_flow,
                np.array([c_flow], dtype=np.float32),
                sev_onehot,
                np.array([step_frac, esc_left_norm, posture_active, posture_rem_norm], dtype=np.float32),
                history_counts,
            ], dtype=np.float32)

            # (3, 28) agent observation matrix
            obs_matrix = np.zeros((NUM_AGENTS, 28), dtype=np.float32)
            for agent_idx in range(NUM_AGENTS):
                obs_matrix[agent_idx, :ENV_FEATURE_DIM] = env_obs
                obs_matrix[agent_idx, ENV_FEATURE_DIM:] = agent_ids[agent_idx]

            # Greedy action selection from trained MAPPO policy
            actions, _, _ = team.select_actions(obs_matrix, deterministic=True)
            a_tri, a_con, a_esc = int(actions[0]), int(actions[1]), int(actions[2])

            tri_str = TRIAGE_ACTIONS[a_tri]
            con_str = CONTAINMENT_ACTIONS[a_con]
            esc_str = ESCALATION_ACTIONS[a_esc]

            # State transition updates
            if a_con > 0:
                active_posture = con_str
                posture_steps_remaining = POSTURE_DURATION
            elif posture_steps_remaining > 0:
                posture_steps_remaining -= 1
                if posture_steps_remaining == 0:
                    active_posture = "none"

            if a_esc > 0 and escalations_remaining > 0:
                escalations_remaining -= 1

            recent_pred_classes.append(pred_indices[t])

            triage_acts.append(tri_str)
            contain_acts.append(con_str)
            esc_acts.append(esc_str)
            resp_avail.append(True)

        out_df = batch_results.copy()
        out_df["marl_triage"] = triage_acts
        out_df["marl_containment"] = contain_acts
        out_df["marl_escalation"] = esc_acts
        out_df["response_available"] = resp_avail
        return out_df

    def reset_alert_log(self) -> None:
        """Resets the live alert log and logger instance to genesis state."""
        if self.live_log_path.exists():
            self.live_log_path.unlink()
        if self.live_tail_hash_path.exists():
            self.live_tail_hash_path.unlink()
        self._alert_logger = None

    # ------------------------------------------------------------------
    # Forensic Alert Logging & Verification
    # ------------------------------------------------------------------
    def log_alert(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Appends an alert entry to logs/alerts_live.jsonl via ForensicAlertLogger.
        Maintains SHA-256 hash chaining and external tail anchor.
        """
        raw_conf = float(result.get("raw_confidence", 0.0))
        cal_conf = float(result.get("calibrated_confidence", raw_conf))
        pred_cls = str(result.get("predicted_class", "Benign"))
        severity = str(result.get("severity", "Low"))
        explanation = str(result.get("brief", result.get("explanation", "Routine network flow.")))
        action = str(result.get("recommended_action", "Maintain passive monitoring."))
        source = str(result.get("brief_source", "template_fallback"))
        unknown = bool(result.get("unknown_flag", False))
        top_shap = result.get("top_features", None)

        record = self.alert_logger.log_alert(
            flow_index=int(result.get("flow_index", 0)),
            predicted_class=pred_cls,
            confidence=cal_conf,
            severity=severity,
            explanation=explanation,
            recommended_action=action,
            source="ollama" if source == "llm" else "rule_based_fallback",
            model_version="Phase13-IDSPipeline",
            top_shap_features=top_shap,
            adversarial=unknown,
            true_class=result.get("true_class", None),
            marl_triage_action=result.get("marl_triage", None),
            marl_containment_action=result.get("marl_containment", None),
            marl_escalation_action=result.get("marl_escalation", None),
            marl_policy_version="mappo_balanced_seed42",
        )
        return record

    def verify_log(self) -> Dict[str, Any]:
        """
        Audits logs/alerts_live.jsonl cryptographic hash continuity and tail anchor.
        """
        if not self.live_log_path.exists() or self.live_log_path.stat().st_size == 0:
            return {
                "status": "PASS",
                "lines_verified": 0,
                "message": "Live alert log is currently empty (genesis state).",
                "tail_hash": GENESIS_HASH,
            }

        is_valid, count, msg = verify_alert_log(
            log_path=self.live_log_path,
            tail_hash_path=self.live_tail_hash_path,
            enforce_anchor=True,
        )
        tail_hash = self.alert_logger.last_hash
        return {
            "status": "PASS" if is_valid else "FAIL",
            "lines_verified": count,
            "message": msg,
            "tail_hash": tail_hash,
        }
