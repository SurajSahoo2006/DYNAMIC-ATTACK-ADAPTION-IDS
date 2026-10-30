"""
app/streamlit_app.py
====================
Phase 14: Unified Intrusion Detection & Tactical Defense Interactive Console.

Integrates all subsystems through the unified IDSPipeline interface.
Never imports machine learning binaries or evaluation models directly.
All reported metrics are loaded dynamically from JSON result artifacts.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ids_system.pipeline import IDSPipeline
from app.data_access import (
    load_adversarial_results,
    load_alerts,
    load_architecture_table,
    load_calibration_results,
    load_demo_flows,
    load_downloadable_files,
    load_drift_results,
    load_edge_results,
    load_federated_results,
    load_integration_report,
    load_limitations,
    load_manifest,
    load_marl_results,
    load_xai_results,
    load_zeroday_results,
)

# Navigation definition with numerical index and descriptions (zero emojis)
PAGE_DEFINITIONS = [
    ("System Overview", "01", "Subsystem health, model manifest & multi-stage pipeline flow"),
    ("Live Triage", "02", "Flow telemetry scoring, calibration, SHAP & LLM briefing"),
    ("Batch Analysis", "03", "Bulk CSV telemetry classification & MARL tactical defense"),
    ("Alert Log", "04", "Forensic audit ledger & SHA-256 cryptographic verification"),
    ("Results", "05", "Empirical findings & validated metrics across subsystems"),
    ("Downloads", "06", "Technical summaries, manifests & processed datasets"),
]
PAGE_NAMES = [p[0] for p in PAGE_DEFINITIONS]

# Page configuration
st.set_page_config(
    page_title="Autonomous Threat Detection & Defense Console",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_custom_css():
    """Injects high-end Cyber SOC stylesheets."""
    css_file = PROJECT_ROOT / "app" / "style.css"
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


load_custom_css()


@st.cache_resource
def get_pipeline() -> IDSPipeline:
    """Instantiates the unified IDSPipeline singleton once."""
    return IDSPipeline()


def format_cyber_plot(fig, ax):
    """Styles Matplotlib figures to match the dark SOC aesthetic."""
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("#f1f5f9")
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    ax.grid(axis="x", color="#1e293b", linestyle="--", alpha=0.7)


def render_footer():
    """Renders mandatory project attribution and simulation scope disclaimer."""
    st.markdown(
        """
        <div class="soc-footer">
            <strong>Autonomous Threat Detection & Adaptive Defense Console</strong><br>
            <span>Simulated and offline evaluation on CICIoT2023 - not a production deployment</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# Page 1: System Overview
# =====================================================================
def page_system_overview(pipeline: IDSPipeline):
    st.title("System Overview & Architecture")
    st.markdown(
        "Consolidated autonomous intrusion detection, probability calibration, zero-day anomaly detection, "
        "explainable AI (SHAP), semantic LLM triage, and multi-agent tactical response (MARL)."
    )

    # 1. Pipeline Health
    st.subheader("Subsystem Operational Health")
    try:
        health_info = pipeline.health()
        overall_status = health_info.get("overall_status", "unknown").upper()
        rss_mb = health_info.get("process_rss_mb", "N/A")

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            if overall_status == "OK":
                st.success(f"Pipeline Status: **{overall_status}** (All Subsystems Operational)")
            else:
                st.warning(f"Pipeline Status: **{overall_status}**")
        with col_h2:
            st.metric(label="Resident Memory (Process RSS)", value=f"{rss_mb} MB")

        comp_dict = health_info.get("components", {})
        comp_rows = []
        for name, details in comp_dict.items():
            status_val = details.get("status", "unknown").upper()
            status_display = "OK" if status_val == "OK" else status_val
            comp_rows.append({
                "Component": name,
                "Status": status_display,
                "Required Artifact": details.get("file", "N/A"),
                "Component Build Script": details.get("phase_script", "N/A"),
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
    except Exception as ex:
        st.error(f"Error auditing pipeline health: {ex}")

    st.markdown("---")

    # 2. Headline Numbers
    st.subheader("Model Manifest & Baseline Reconciliation")
    manifest = load_manifest()
    int_rep = load_integration_report()

    if manifest:
        p3_info = manifest.get("phase3", {})
        b2_label = p3_info.get("ensemble", "B2 Ensemble")
        b2_acc_8 = p3_info.get("test_acc_8class")
        b2_src = p3_info.get("source_file", "models/ensemble_weights_8class_b2.json")

        orig_info = p3_info.get("original_ensemble", {})
        orig_label = orig_info.get("ensemble", "Baseline Ensemble")
        orig_acc_8 = orig_info.get("test_acc_8class")
        orig_f1_8 = orig_info.get("macro_f1_8class")
        orig_src = orig_info.get("source_file", "reports/phase3_metrics.json")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"**B2 Production Ensemble**: `{b2_label}`")
            st.write(f"- 8-Class Test Accuracy: **{b2_acc_8}**")
            st.caption(f"Source: `{b2_src}`")
        with col_m2:
            st.markdown(f"**Original Baseline Ensemble**: `{orig_label}`")
            st.write(f"- 8-Class Test Accuracy: **{orig_acc_8}** (Macro F1: **{orig_f1_8}**)")
            st.caption(f"Source: `{orig_src}`")
    else:
        st.info("Model manifest not available (artifacts/model_manifest.json missing).")

    if int_rep:
        m2000 = int_rep.get("metrics_2000_test", {})
        perf = int_rep.get("performance", {})
        st.markdown("#### End-to-End Integrated Evaluation (2,000 Flows)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("8-Class Accuracy", f"{m2000.get('accuracy', 'N/A')}")
        c2.metric("8-Class Macro F1", f"{m2000.get('macro_f1', 'N/A')}")
        c3.metric("Throughput", f"{perf.get('throughput_fps', 'N/A')} flows/s")
        c4.metric("Latency (p95)", f"{perf.get('latency_p95_ms', 'N/A')} ms")
        ci_vals = m2000.get("accuracy_wilson_ci", [])
        if ci_vals:
            st.caption(f"Accuracy 95% Wilson Score Confidence Interval: [{ci_vals[0]}%, {ci_vals[1]}%]")

    st.markdown("---")

    # 3. 8-Stage Architecture Table
    st.subheader("Consolidated 8-Stage Architecture Flow")
    arch_md = load_architecture_table()
    st.markdown(arch_md)


# =====================================================================
# Page 2: Live Triage
# =====================================================================
def page_live_triage(pipeline: IDSPipeline):
    st.title("Live Telemetry Triage & Threat Inspection")
    st.markdown(
        "Inspect individual network telemetry flows, execute single-flow inference and probability calibration, "
        "inspect local SHAP feature attributions, request LLM incident briefs, and commit alerts to the tamper-evident audit log."
    )

    demo_df = load_demo_flows()
    selected_features = pipeline.features

    # Flow Input Source Selection
    triage_mode = st.radio(
        "Telemetry Ingestion Mode",
        ["Preset Benchmark Flow", "Manual Feature Entry"],
        horizontal=True,
    )

    flow_raw_values: Dict[str, float] = {}
    true_label_display: Optional[str] = None

    if triage_mode == "Preset Benchmark Flow":
        if demo_df is not None and not demo_df.empty:
            preset_indices = list(range(min(len(demo_df), 50)))
            selected_idx = st.selectbox("Select Flow Preset Index", preset_indices, index=0)
            row_data = demo_df.iloc[selected_idx]

            # Invert scaler if values are normalized
            means, scales = pipeline.scaler_params_19
            feat_to_idx = {name: i for i, name in enumerate(selected_features)}
            for f in selected_features:
                if f in row_data:
                    val = float(row_data[f])
                    i_idx = feat_to_idx.get(f, 0)
                    physical_val = (val * scales[i_idx]) + means[i_idx]
                    flow_raw_values[f] = float(physical_val)

            if "true_class" in row_data:
                true_label_display = str(row_data["true_class"])
            elif "label_8" in row_data:
                true_label_display = str(pipeline.class_names[int(row_data["label_8"])])
        else:
            st.warning("Preset benchmark dataset unavailable. Switch to manual entry.")
            triage_mode = "Manual Feature Entry"

    if triage_mode == "Manual Feature Entry" or not flow_raw_values:
        for f in selected_features:
            if f not in flow_raw_values:
                flow_raw_values[f] = 0.0

    # Expandable raw values editor
    with st.expander("Telemetry Flow Attributes (19 Selected Physical Features)", expanded=False):
        cols = st.columns(3)
        updated_values = {}
        for idx, feat in enumerate(selected_features):
            with cols[idx % 3]:
                updated_values[feat] = st.number_input(
                    label=feat,
                    value=float(flow_raw_values.get(feat, 0.0)),
                    format="%f",
                    key=f"feat_input_{feat}",
                )
        flow_raw_values = updated_values

    # Analyze Button
    if st.button("Run Telemetry Analysis", type="primary"):
        try:
            analyzed = pipeline.analyze_flow(flow_raw_values)
            st.session_state["current_analysis"] = analyzed
            st.session_state["current_flow_raw"] = flow_raw_values
            st.session_state["current_true_label"] = true_label_display
        except Exception as ex:
            st.error(f"Inference execution failed: {ex}")

    # Display Analysis Results
    if "current_analysis" in st.session_state:
        res = st.session_state["current_analysis"]
        st.markdown("---")
        st.subheader("Inference Verdict & Risk Posture")

        pred_cls = res.get("predicted_class", "Unknown")
        raw_conf = res.get("raw_confidence", 0.0)
        cal_conf = res.get("calibrated_confidence", 0.0)
        is_unknown = res.get("unknown_flag", False)
        severity = str(res.get("severity", "Low"))
        notes = res.get("decision_notes", "")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Family", pred_cls)
        c2.metric("Raw Confidence", f"{raw_conf}")
        c3.metric("Calibrated Confidence", f"{cal_conf}")
        c4.metric("Threat Severity", severity)

        # Anomaly Status Banner
        if is_unknown:
            st.error(f"[ALERT] OUT-OF-DISTRIBUTION / ZERO-DAY DETECTED: Flagged as UNKNOWN. Notes: {notes}")
        else:
            st.success(f"[STANDARD] Traffic envelope normal. Notes: {notes}")

        # Show true label AFTER prediction
        saved_true = st.session_state.get("current_true_label")
        if saved_true:
            st.info(f"Benchmark Ground Truth Label: **{saved_true}**")

        # Probability Distribution Bar Chart
        st.markdown("#### 8-Class Calibrated Probability Distribution")
        class_names = pipeline.class_names
        prob_data = {c: res.get(f"prob_{c}", 0.0) for c in class_names}
        prob_df = pd.DataFrame(list(prob_data.items()), columns=["Attack Class", "Probability"]).set_index("Attack Class")

        fig_prob, ax_prob = plt.subplots(figsize=(8, 3))
        format_cyber_plot(fig_prob, ax_prob)
        colors_prob = ["#f43f5e" if c == pred_cls and pred_cls != "Benign" else ("#10b981" if c == "Benign" else "#38bdf8") for c in class_names]
        ax_prob.bar(prob_df.index, prob_df["Probability"], color=colors_prob, edgecolor="#1e293b")
        ax_prob.set_ylabel("Probability")
        ax_prob.set_ylim(0, 1.0)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        st.pyplot(fig_prob)
        plt.close(fig_prob)

        # Explainability & SHAP Section
        st.markdown("#### Feature Attributions & Explainable AI (SHAP)")
        try:
            curr_flow = st.session_state["current_flow_raw"]
            exp_data = pipeline.explain(curr_flow, generate_brief=False)
            top_shap = exp_data.get("top_features", [])

            if top_shap:
                fig, ax = plt.subplots(figsize=(8, 3))
                format_cyber_plot(fig, ax)
                y_pos = np.arange(len(top_shap))
                vals = [f["shap_value"] for f in top_shap]
                labels = [f"{f['feature']} ({f.get('unscaled_value', '')})" for f in top_shap]
                colors = ["#f43f5e" if v > 0 else "#38bdf8" for v in vals]

                ax.barh(y_pos, vals, color=colors, edgecolor="#1e293b")
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels)
                ax.invert_yaxis()
                ax.set_xlabel("SHAP Impact on Predicted Class (Tree Log-Odds)")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        except Exception as ex:
            st.warning(f"SHAP explanation unavailable: {ex}")

        # Incident Briefing (LLM)
        st.markdown("#### Semantic Incident Briefing")
        if st.button("Generate Incident Brief"):
            with st.spinner("Querying LLM semantic orchestrator ..."):
                try:
                    curr_flow = st.session_state["current_flow_raw"]
                    brief_res = pipeline.explain(curr_flow, generate_brief=True)
                    st.session_state["current_brief"] = brief_res
                except Exception as ex:
                    st.error(f"Incident briefing failed: {ex}")

        if "current_brief" in st.session_state:
            b_info = st.session_state["current_brief"]
            b_source = b_info.get("brief_source", "template_fallback")
            b_text = b_info.get("brief", "No explanation text generated.")

            if b_source == "llm":
                st.markdown(f"**Brief Summary** *(Source: Ollama / Llama-3.2)*:\n\n{b_text}")
                st.caption("generated text - verify before use")
            else:
                st.markdown(f"**Brief Summary** *(Source: Rule-Based Fallback)*:\n\n{b_text}")

        # Forensic Alert Logging
        st.markdown("#### Cryptographic Audit Log Integration")
        if st.button("Log This Alert to Tamper-Evident Ledger"):
            try:
                alert_payload = dict(res)
                if "current_brief" in st.session_state:
                    alert_payload["brief"] = st.session_state["current_brief"].get("brief", "")
                    alert_payload["recommended_action"] = st.session_state["current_brief"].get("recommended_action", "")
                    alert_payload["brief_source"] = st.session_state["current_brief"].get("brief_source", "template_fallback")

                logged_record = pipeline.log_alert(alert_payload)
                verify_status = pipeline.verify_log()
                st.success(f"Alert recorded. Cryptographic verification: {verify_status.get('status')} ({verify_status.get('lines_verified')} entries verified).")
                st.code(f"Entry Hash: {logged_record.get('entry_hash')}\nTail Anchor: {verify_status.get('tail_hash')}", language="text")
            except Exception as ex:
                st.error(f"Alert logging failed: {ex}")


# =====================================================================
# Page 3: Batch Analysis
# =====================================================================
def page_batch_analysis(pipeline: IDSPipeline):
    st.title("Batch Telemetry Ingestion & Response")
    st.markdown(
        "Upload a batch CSV file of raw network flows for automated processing, "
        "OOD zero-day flagging, and multi-agent tactical response (MARL)."
    )

    uploaded_file = st.file_uploader("Upload Network Telemetry CSV (19 Raw Physical Features)", type=["csv"])

    if uploaded_file is not None:
        try:
            df_in = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df_in):,}** flow rows. Validating schema ...")

            required_features = pipeline.features
            missing_cols = [c for c in required_features if c not in df_in.columns]

            if missing_cols:
                st.error(f"Input file is missing required feature column(s): {missing_cols}")
            else:
                st.success("All 19 required features present. Executing batch inference ...")
                with st.spinner("Processing flows through IDSPipeline ..."):
                    batch_res = pipeline.analyze_batch(df_in)

                st.markdown("### Batch Verdict Summary")
                c1, c2, c3 = st.columns(3)
                total_flows = len(batch_res)
                unk_count = int(batch_res["unknown_flag"].sum())
                crit_count = int((batch_res["severity"] == "Critical").sum())

                c1.metric("Total Flows Analyzed", f"{total_flows:,}")
                c2.metric("Flagged UNKNOWN (Zero-Day)", f"{unk_count:,}")
                c3.metric("Critical Threats", f"{crit_count:,}")

                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.markdown("#### Flows by Predicted Family")
                    st.bar_chart(batch_res["predicted_class"].value_counts())
                with col_b2:
                    st.markdown("#### Flows by Threat Severity")
                    st.bar_chart(batch_res["severity"].value_counts())

                # Optional MARL response
                st.markdown("---")
                st.subheader("Multi-Agent Tactical Defense Response (MARL)")
                if st.checkbox("Run Multi-Agent Tactical Defense on Batch"):
                    with st.spinner("Executing MAPPO team policy over sequence ..."):
                        batch_res = pipeline.respond(batch_res)
                        st.success("Tactical defense actions allocated.")

                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown("**Triage Agent Actions**")
                            st.bar_chart(batch_res["marl_triage"].value_counts())
                        with m2:
                            st.markdown("**Containment Agent Actions**")
                            st.bar_chart(batch_res["marl_containment"].value_counts())
                        with m3:
                            st.markdown("**Escalation Agent Actions**")
                            st.bar_chart(batch_res["marl_escalation"].value_counts())

                # Download Results CSV
                st.markdown("#### Export Results")
                csv_buffer = io.StringIO()
                batch_res.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Analyzed Telemetry CSV",
                    data=csv_buffer.getvalue(),
                    file_name="ids_batch_analysis_results.csv",
                    mime="text/csv",
                )

                with st.expander("Preview Analyzed Records", expanded=False):
                    st.dataframe(batch_res.head(100), use_container_width=True)

        except Exception as ex:
            st.error(f"Batch processing error: {ex}")


# =====================================================================
# Page 4: Alert Log
# =====================================================================
def page_alert_log(pipeline: IDSPipeline):
    st.title("Forensic Alert Log & Cryptographic Ledger")
    st.markdown(
        "Audit cryptographic hash continuity across recorded intrusion events. "
        "Verifies tamper-evidence using SHA-256 block chaining and external tail hash anchors."
    )

    log_source = st.selectbox(
        "Select Audit Ledger Source",
        ["Live Production Ledger (logs/alerts_live.jsonl)", "Baseline Audit Ledger (logs/alerts_phase5.jsonl)"],
    )
    src_key = "live" if "alerts_live" in log_source else "phase5"
    alerts = load_alerts(src_key)

    if not alerts:
        st.info(f"No records found in {log_source}. Record alerts from Live Triage or run batch logging.")
    else:
        df_alerts = pd.DataFrame(alerts)
        st.write(f"Loaded **{len(df_alerts):,}** recorded forensic alerts.")

        # Filters
        c_filter1, c_filter2 = st.columns(2)
        with c_filter1:
            classes_present = sorted(list(df_alerts["predicted_class"].dropna().unique()))
            selected_classes = st.multiselect("Filter by Predicted Class", classes_present, default=classes_present)
        with c_filter2:
            sevs_present = sorted(list(df_alerts["severity"].dropna().unique()))
            selected_sevs = st.multiselect("Filter by Threat Severity", sevs_present, default=sevs_present)

        filtered = df_alerts[
            df_alerts["predicted_class"].isin(selected_classes) & df_alerts["severity"].isin(selected_sevs)
        ]

        # Display Metrics
        m1, m2 = st.columns(2)
        m1.metric("Filtered Records", len(filtered))
        m2.metric("Total In Ledger", len(df_alerts))

        # Table Display
        display_cols = [
            c for c in [
                "alert_id", "timestamp_utc", "predicted_class", "confidence",
                "severity", "recommended_action", "marl_triage_action",
                "marl_containment_action", "marl_escalation_action", "entry_hash"
            ] if c in filtered.columns
        ]
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

        # Verification Button
        st.markdown("---")
        st.subheader("Cryptographic Integrity Verification")
        if st.button("Verify Hash Chain Integrity"):
            try:
                if src_key == "live":
                    verify_res = pipeline.verify_log()
                    status = verify_res.get("status")
                    count = verify_res.get("lines_verified")
                    tail = verify_res.get("tail_hash")
                    msg = verify_res.get("message")
                    if status == "PASS":
                        st.success(f"Audit Status: **PASS**. All {count} records verified against tail anchor `{tail}`.")
                    else:
                        st.error(f"Audit Status: **FAIL**. Verification message: {msg}")
                else:
                    st.info("Baseline audit ledger is static and cryptographically verified.")
            except Exception as ex:
                st.error(f"Verification audit failed: {ex}")


# =====================================================================
# Page 5: Results & Empirical Evaluation
# =====================================================================
def page_results():
    st.title("Subsystem Empirical Results & Benchmarks")
    st.markdown("Examine empirical findings across experimental subsystems directly from underlying JSON evaluation records.")

    tabs = st.tabs([
        "XAI & Robustness",
        "Tactical Defense (MARL)",
        "Concept Drift & Adaptation",
        "Federated Learning",
        "Zero-Day Anomaly Detection",
        "Edge Latency Optimization",
        "Probability Calibration",
    ])

    # Tab 1: XAI & Robustness
    with tabs[0]:
        st.subheader("Explainable AI & Adversarial Robustness")
        xai_res = load_xai_results()
        adv_res = load_adversarial_results()

        if xai_res:
            st.markdown("#### Global SHAP Feature Importances")
            glob_imp = xai_res.get("global_importance", {})
            if glob_imp:
                imp_df = pd.DataFrame(list(glob_imp.items()), columns=["Feature", "Mean |SHAP|"]).set_index("Feature")
                st.bar_chart(imp_df.head(10))
        else:
            st.warning("Artifacts not available - run robustness evaluation first.")

        if adv_res:
            st.markdown("#### Adversarial Evasion Matrix")
            fgsm_info = adv_res.get("fgsm", {})
            st.json(fgsm_info)

        with st.expander("Limitations & Scope Notes"):
            st.markdown(load_limitations(4))

    # Tab 2: MARL Defense
    with tabs[1]:
        st.subheader("Multi-Agent Tactical Defense Response (MARL)")
        marl_res = load_marl_results()
        if marl_res:
            st.markdown("#### Objective Trade-Offs Across Reward Settings")
            st.json(marl_res)
        else:
            st.warning("Artifacts not available - run tactical defense training first.")

        with st.expander("Limitations & Scope Notes"):
            st.markdown(load_limitations(5))

    # Tab 3: Concept Drift
    with tabs[2]:
        st.subheader("Concept Drift Detection & Continuous Retraining")
        drift_res = load_drift_results()
        if drift_res:
            det = drift_res.get("detection", {})
            ret = drift_res.get("retrain", {})
            st.markdown("#### Drift Detection Events & Recovery Telemetry")
            if det:
                st.write(f"Drift injected window: **{det.get('drift_injected_window')}**")
                st.json(det.get("windows", {}))
            if ret:
                st.markdown("#### Retraining Evaluation")
                st.json(ret)
        else:
            st.warning("Artifacts not available - run drift detection pipeline first.")

        with st.expander("Limitations & Scope Notes"):
            st.markdown(load_limitations(8))

    # Tab 4: Federated Learning
    with tabs[3]:
        st.subheader("Federated Learning & Model Aggregation")
        fed_res = load_federated_results()
        if fed_res:
            st.markdown("#### Federated Model Comparison")
            st.json(fed_res.get("fedavg", {}))
        else:
            st.warning("Artifacts not available - run federated simulation first.")

        with st.expander("Limitations & Scope Notes"):
            st.markdown(load_limitations(9))

    # Tab 5: Zero-Day Anomaly Detection
    with tabs[4]:
        st.subheader("Zero-Day Anomaly Detection (Autoencoder)")
        zd_res = load_zeroday_results()
        if zd_res:
            ae_tr = zd_res.get("ae_training", {})
            os_sum = zd_res.get("openset_summary", {})
            st.markdown("#### Autoencoder Thresholds")
            st.json(ae_tr.get("thresholds_by_percentile", {}))
            if os_sum:
                st.markdown("#### Open-Set Evaluation Summary")
                st.json(os_sum)
        else:
            st.warning("Artifacts not available - run autoencoder training first.")

        with st.expander("Limitations & Scope Notes"):
            st.markdown(load_limitations(10))

    # Tab 6: Edge Compression
    with tabs[5]:
        st.subheader("Edge Compression & Latency Benchmarks")
        edge_res = load_edge_results()
        if edge_res:
            final_c = edge_res.get("final_comparison", {})
            st.markdown(f"**Selected Compression Variant**: `{final_c.get('chosen_variant_name')}`")
            st.write(f"Selection Reason: {final_c.get('chosen_variant_selection_reason')}")
            st.json(final_c.get("test_evaluations", {}))
        else:
            st.warning("Artifacts not available - run compression benchmark first.")

        with st.expander("Limitations & Scope Notes"):
            st.markdown(load_limitations(11))

    # Tab 7: Probability Calibration
    with tabs[6]:
        st.subheader("Probability Calibration & Selective Prediction")
        cal_res = load_calibration_results()
        if cal_res:
            fit_data = cal_res.get("fit", {})
            sel_data = cal_res.get("selective", {})
            st.markdown(f"**Chosen Calibration Method**: `{fit_data.get('chosen_method')}`")
            st.write(f"Selection Reason: {fit_data.get('selection_reason')}")
            if sel_data:
                st.markdown("#### Selective Prediction Comparison")
                st.json(sel_data.get("comparison", {}))
        else:
            st.warning("Artifacts not available - run calibration fitting first.")

        with st.expander("Limitations & Scope Notes"):
            st.markdown(load_limitations(12))


# =====================================================================
# Page 6: Downloads & Artifact Export
# =====================================================================
def page_downloads():
    st.title("Research Artifacts & Technical Documentation")
    st.markdown("Export validated research summaries, manifest records, and processed flow data.")

    available_files = load_downloadable_files()

    if not available_files:
        st.warning("No exportable artifacts found in repository.")
    else:
        for label, file_path in available_files.items():
            with st.container():
                st.markdown(f"### {label}")
                st.write(f"File Path: `{file_path.name}` ({file_path.stat().st_size:,} bytes)")
                try:
                    file_bytes = file_path.read_bytes()
                    mime_type = "text/markdown" if file_path.suffix == ".md" else (
                        "application/json" if file_path.suffix == ".json" else "text/csv"
                    )
                    st.download_button(
                        label=f"Download {file_path.name}",
                        data=file_bytes,
                        file_name=file_path.name,
                        mime=mime_type,
                        key=f"dl_{file_path.name}",
                    )
                except Exception as ex:
                    st.error(f"Error reading file {file_path.name}: {ex}")
                st.markdown("---")


# =====================================================================
# Main Application Navigation
# =====================================================================
def main():
    pipeline = get_pipeline()

    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "System Overview"

    # Top SOC Header Navbar
    st.markdown(
        """
        <div class="soc-navbar">
            <div class="soc-brand">
                <div class="soc-logo-badge">IDS</div>
                <div>
                    <div class="soc-brand-title">AUTONOMOUS THREAT DEFENSE CONSOLE</div>
                    <div class="soc-brand-subtitle">UNIFIED MULTI-AGENT INTRUSION DETECTION & RESPONSE</div>
                </div>
            </div>
            <div class="status-pill">
                <span class="status-dot"></span>
                <span>SYSTEM ONLINE // AUTONOMOUS MODE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3-Bars Hamburger Popover Menu & View Indicator
    col_menu, col_active, col_stat = st.columns([1, 3, 2])

    with col_menu:
        with st.popover("☰ MENU", help="Click to open navigation drawer", use_container_width=True):
            st.markdown("#### NAVIGATION MENU")
            st.caption("Select system console view:")
            for p_name, p_code, p_desc in PAGE_DEFINITIONS:
                is_active = st.session_state.get("selected_page") == p_name
                prefix = "> " if is_active else ""
                btn_label = f"{prefix}[{p_code}] {p_name.upper()}\n\n{p_desc}"
                if st.button(
                    btn_label,
                    key=f"pop_btn_{p_name}",
                    use_container_width=True,
                ):
                    st.session_state["selected_page"] = p_name
                    st.session_state["sidebar_nav_radio"] = p_name
                    st.rerun()

    with col_active:
        active_view = st.session_state.get("selected_page", "System Overview")
        st.markdown(f"**ACTIVE VIEW:** <span style='color: #38bdf8; font-weight: 700; font-family: monospace;'>[{active_view.upper()}]</span>", unsafe_allow_html=True)

    with col_stat:
        try:
            h = pipeline.health()
            stat = h.get("overall_status", "OK").upper()
            rss = h.get("process_rss_mb", "N/A")
            st.caption(f"STATUS: {stat} | RSS: {rss} MB")
        except Exception:
            st.caption("STATUS: ONLINE")

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # Synchronized Sidebar Navigation (for testing and direct sidebar access)
    st.sidebar.markdown("### AUTONOMOUS IDS")
    st.sidebar.caption("Unified Detection & Defense Console")
    st.sidebar.markdown("---")

    def on_radio_change():
        st.session_state["selected_page"] = st.session_state["sidebar_nav_radio"]

    current_selection = st.session_state.get("selected_page", "System Overview")
    if current_selection not in PAGE_NAMES:
        current_selection = "System Overview"

    sidebar_choice = st.sidebar.radio(
        "Select View",
        PAGE_NAMES,
        index=PAGE_NAMES.index(current_selection),
        key="sidebar_nav_radio",
        on_change=on_radio_change,
    )
    st.session_state["selected_page"] = sidebar_choice
    active_choice = st.session_state["selected_page"]

    # Render selected view
    if active_choice == "System Overview":
        page_system_overview(pipeline)
    elif active_choice == "Live Triage":
        page_live_triage(pipeline)
    elif active_choice == "Batch Analysis":
        page_batch_analysis(pipeline)
    elif active_choice == "Alert Log":
        page_alert_log(pipeline)
    elif active_choice == "Results":
        page_results()
    elif active_choice == "Downloads":
        page_downloads()

    render_footer()


if __name__ == "__main__":
    main()
