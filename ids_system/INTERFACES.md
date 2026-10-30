# IDS Component Public Interfaces and Mappings (Phase 13)

This document records the exact signatures and operational parameters of all components consolidated into `ids_system.pipeline.IDSPipeline`.

---

## 1. `predict.py`
- `get_selected_features() -> list[str]`: Reads and returns the 19 ordered feature names from `artifacts/selected_features.json`.
- `get_or_load_model(target: str = "8class", model_type: str = "b2") -> EnsembleClassifier`: Loads and caches the trained ensemble model (Random Forest 0.60 + XGBoost 0.40) from `models/rf_8class_b2.joblib` and `models/xgb_8class_b2.joblib`.
- `predict_flow(df_19_scaled: pd.DataFrame, target: str = "8class", mode: str = "weighted", model_type: str = "b2") -> pd.DataFrame`: Validates input features against the 19 selected columns, executes ensemble inference, and returns DataFrame with `predicted_class_idx`, `predicted_class`, `confidence`, and deterministic `severity`.

---

## 2. `models/ensemble.py`
- `EnsembleClassifier`:
  - `predict_probabilities(X: pd.DataFrame | np.ndarray) -> Dict[str, np.ndarray]`: Computes prediction matrices for `rf`, `xgb`, `dnn`, uniform `soft`, and validation-weighted `weighted` soft voting.
  - `predict(X: pd.DataFrame | np.ndarray, mode: str = "weighted") -> Tuple[np.ndarray, np.ndarray]`: Returns `(predicted_class_indices, max_confidence_scores)`.
- `PyTorchDNN(nn.Module)`: PyTorch feed-forward architecture (19 -> 128 -> 64 -> 8).

---

## 3. `calibration/fit.py` and `calibration/results/fit.json`
- **Chosen Calibrator**: `isotonic_regression` (One-vs-Rest `IsotonicCalibrator` with 8 isotonic regressors and row normalization).
- **Saved Model Path**: `calibration/models/chosen_calibrator.joblib`.
- **Application**:
  - `calibrator.predict_proba(probs: np.ndarray) -> np.ndarray`: Takes raw 8-class probabilities of shape `(N, 8)` and outputs monotonically calibrated probabilities of shape `(N, 8)`.
  - `calibrated_confidence = np.max(calibrated_probs, axis=1)`.

---

## 4. `zeroday/train_autoencoder.py` and `zeroday/results/ae_training.json`
- **Model Architecture**: `BenignAutoencoder(in_features=19)` (19 -> 12 -> 6 -> 12 -> 19 with ReLU).
- **Model Binary**: `zeroday/models/autoencoder.pt`.
- **Reconstruction Score**: Mean squared error across the 19 standardized features:
  $$\text{ae\_score} = \frac{1}{19} \sum_{i=1}^{19} (x_i - \hat{x}_i)^2$$
- **Thresholds**: From `zeroday/results/ae_training.json`:
  - 95.0th percentile: `0.374170`
  - 97.5th percentile: `0.534869`
  - 99.0th percentile: `0.926735`
- **Phase 10 Operating Point**: 97.5th percentile threshold (`0.534869`). If `ae_score > 0.534869`, `ae_flag = True`.

---

## 5. `calibration/results/*.json` (10% Benign False-Positive Budget Cutoff)
- **Validation Cutoffs** (derived from 10.0% validation Benign FPR in Phase 12):
  - Calibrated confidence cutoff $\tau_{\text{cal}} = 0.521507$ (`0.5215`).
  - Raw confidence cutoff $\tau_{\text{raw}} = 0.444221$ (`0.4442`).
- **Fusion Unknown Attack Rule**:
  $$\text{unknown\_flag} = \text{ae\_flag} \lor (\text{calibrated\_confidence} < 0.521507)$$

---

## 6. `xai/shap_explain.py`
- `SHAPExplainer(target="8class", model_type="b2")`: TreeExplainer on XGBoost and Random Forest.
- `explain_xgb(X: pd.DataFrame) -> np.ndarray`: Generates local SHAP values array `(N, 19, 8)`.
- `get_local_explanation(shap_values_single, unscaled_features_dict, target_class_idx, top_k=5) -> List[Dict[str, Any]]`: Returns the top 5 signed feature contributions for the predicted class.

---

## 7. `orchestrator/llm_orchestrator.py`
- `LLMSemanticOrchestrator(base_url, model_name)`: Connects to local Ollama (`http://localhost:11434`) and falls back to deterministic rule-based expert knowledge when Ollama is unreachable.
- `FeatureUnscaler()`: Inverts standardization using `mean_` and `scale_` from `artifacts/standard_scaler.pkl`.
- `get_flow_severity(label, confidence) -> str`: Deterministic mapping from `(predicted_class, confidence)` to severity rating (`Low`, `Medium`, `High`, `Critical`).

---

## 8. `alerts/alert_logger.py` and `alerts/verify_log.py`
- `ForensicAlertLogger(log_path, tail_hash_path)`: Appends tamper-evident audit records linked with SHA-256 hash chains (`prev_hash -> entry_hash`).
- `verify_alert_log(log_path, tail_hash_path, enforce_anchor=True) -> Tuple[bool, int, str]`: Cryptographically verifies the integrity of the audit log and tail anchor.

---

## 9. `marl/env.py`, `marl/build_cache.py`, `marl/evaluate.py`
- **Policy Checkpoint**: `models/marl/mappo_balanced_seed42_best.pt` (Phase 5 best balanced MAPPO team).
- **Policy Loading**: Loaded via `MAPPOTeam.load_checkpoint(checkpoint_path)`. Evaluator wrapped in `MAPPOEvaluatorPolicy`.
- **Cache Schema**: `prob_<class>` (8 columns), `pred_class_idx`, `pred_class`, `confidence`, `severity`, `true_label_idx`, `true_label`.
- **Observation Construction**: 28-dimensional vector per agent (25 environment features: 8 probabilities, 1 confidence, 4 one-hot severity, 4 normalized state variables, 8 rolling predicted class history counts + 3 agent-ID one-hot).
- **Operational Interface Alignment**:
  - In offline evaluation, `TacticalDefenseEnv` computes rewards using `true_label`.
  - In live inference (`analyze_flow` / `respond`), ground-truth labels are absent. As specified, the policy can greedily produce tactical actions (`triage_action`, `containment_action`, `escalation_action`) from live observation features, while ground-truth dependent reward calculations are marked `response_available: False` if `true_label` is omitted.
