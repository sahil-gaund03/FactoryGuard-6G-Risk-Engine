import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.components.styles import load_custom_css, load_predictions_data
from app.components.filters import render_sidebar_filters

st.set_page_config(
    page_title="FactoryGuard 6G — Predictive Maintenance Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_metric_card(label, value, tooltip, color_class="", subtext=""):
    """Render a KPI metric card with a detailed operational tooltip."""
    st.markdown(f"""
        <div class="steel-card" style="border-left-color: {color_class or '#00bcd4'}">
            <span style="font-size: 14px; color: #aaa;" title="{tooltip}">
                {label} ℹ️
            </span>
            <div class="kpi-value" style="color: {color_class or '#00e5ff'}">{value}</div>
            <div style="font-size: 12px; color: #888; margin-top: 5px;">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)

def main():
    load_custom_css()
    
    st.title("🛡️ FactoryGuard 6G")
    st.subheader("Predictive Maintenance and Anomaly Detection in 6G-Integrated Smart Manufacturing")
    
    # Load cached predictions
    df = load_predictions_data()
    if df is None:
        st.error("No predictions dataset found. Please run the scripts to generate the models and prediction Parquet first.")
        st.info("Run: `python scripts/run_data_pipeline.py`, then `python scripts/build_features.py`, then `python scripts/build_proxy_labels.py`, then `python scripts/train_anomaly_models.py`, then `python scripts/train_supervised_models.py`, and finally `python scripts/generate_predictions.py`.")
        return
        
    # Render common sidebar filters and get filtered df
    df_filtered = render_sidebar_filters(df)
    
    # Overview metrics
    st.write("---")
    col1, col2, col3, col4 = st.columns(4)
    
    total_fleet = df_filtered['Machine_ID'].nunique()
    high_alerts = (df_filtered['risk_level'] == 'High').sum()
    med_alerts = (df_filtered['risk_level'] == 'Medium').sum()
    avg_health = 100.0 - float(df_filtered['fused_risk_score'].mean())
    
    with col1:
        render_metric_card(
            label="Monitored Fleet",
            value=f"{total_fleet} Units",
            tooltip="Meaning: Number of unique machine units. Target: Higher is better (represents coverage). Source: Raw Machine_ID data.",
            color_class="#00bcd4",
            subtext="Across all operational blocks"
        )
        
    with col2:
        render_metric_card(
            label="High-Risk Warnings",
            value=str(high_alerts),
            tooltip="Meaning: Observations exceeding 97.5th risk percentile. Target: Lower is better. Source: Fused ML Models.",
            color_class="#f44336",
            subtext="Requires immediate inspection"
        )
        
    with col3:
        render_metric_card(
            label="Medium-Risk Warnings",
            value=str(med_alerts),
            tooltip="Meaning: Observations in 90th-97.5th risk percentile. Target: Lower is better. Source: Fused ML Models.",
            color_class="#ff9800",
            subtext="Schedule window checkups"
        )
        
    with col4:
        render_metric_card(
            label="Fleet Health Index",
            value=f"{avg_health:.1f}%",
            tooltip="Meaning: Inverse of mean risk scores. Range: [0, 100]. Target: Higher is better. Source: Model-Derived.",
            color_class="#4caf50",
            subtext="Overall plant stability"
        )
        
    st.write("---")
    
    # Business context
    st.markdown("""
    <div class="steel-card">
        <h3>Industrial Intelligence System Overview</h3>
        <p><strong>FactoryGuard 6G</strong> integrates multi-sensor physical telemetry with 6G-network reliability metrics to construct proactive risk assessments for modern smart factories. It operates on two independent machine learning tracks:</p>
        <ul>
            <li><strong>Track A (Unsupervised Anomaly Detection):</strong> Models machine-specific baseline behavior using an Isolation Forest to detect rare and anomalous sensor combinations (e.g. abnormal temperature-vibration ratios).</li>
            <li><strong>Track B (Supervised Future-Deterioration Proxy):</strong> Predicts the probability of upcoming operational decay (such as sudden production speed declines or defect rate escalations) in the next 24 hours using an calibrated XGBoost booster.</li>
        </ul>
        <p>Use the navigation panel on the left to explore the fleet risk monitor, machine diagnostic details, alert centers, and network latency impacts.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
