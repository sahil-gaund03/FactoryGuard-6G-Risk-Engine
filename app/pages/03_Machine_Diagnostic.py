import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.components.styles import load_custom_css, load_predictions_data
from app.components.filters import render_sidebar_filters
from src.factoryguard.explainability.recommendation_engine import generate_alert_narrative

st.set_page_config(page_title="Machine Diagnostic — FactoryGuard 6G", layout="wide")

def main():
    load_custom_css()
    st.title("🔧 Machine Diagnostic Center")
    
    df = load_predictions_data()
    if df is None:
        st.warning("Please generate prediction data first.")
        return
        
    # Render common sidebar filters
    df_filtered = render_sidebar_filters(df)
    
    # Machine selector is primary on this page
    all_machines = sorted(df_filtered['Machine_ID'].unique().astype(int))
    selected_machine = st.selectbox("Select Machine for Diagnostic View", options=all_machines)
    
    mach_df = df_filtered[df_filtered['Machine_ID'] == selected_machine].sort_values('datetime')
    
    if mach_df.empty:
        st.info("No records found matching the active global filters for this machine.")
        return
        
    latest_row = mach_df.iloc[-1]
    
    # 1. Alert Narrative Box
    st.write("---")
    st.subheader("Latest System Diagnostic Report")
    
    narrative = generate_alert_narrative(latest_row)
    st.markdown(f"""
        <div class="steel-card" style="border-left-color: {'#f44336' if latest_row['risk_level'] == 'High' else ('#ff9800' if latest_row['risk_level'] == 'Medium' else '#4caf50')}">
            {narrative['html']}
        </div>
    """, unsafe_allow_html=True)
    
    # 2. Telemetry charts
    st.write("---")
    st.subheader("Telemetry Timeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Chart 1: Temperature & Vibration
        fig_temp = px.line(
            mach_df, 
            x='datetime', 
            y=['Temperature_C', 'Vibration_Hz'],
            title="Temperature & Vibration Readings"
        )
        fig_temp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
        st.plotly_chart(fig_temp, width="stretch")
        st.info(
            "**What this shows:** The raw sensor timeline of temperature (°C) and vibration (Hz) for this machine.\n\n"
            "**Operational Interpretation:** Check if temperature and vibration spikes occur concurrently, which typically points to bearing lubrication degradation."
        )
        
    with col2:
        # Chart 2: Fused Risk Score
        fig_risk = px.line(
            mach_df, 
            x='datetime', 
            y='fused_risk_score',
            title="Fused Risk Index Timeline"
        )
        fig_risk.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
        st.plotly_chart(fig_risk, width="stretch")
        st.info(
            "**What this shows:** The fused health risk rating (0-100) trend for this machine unit.\n\n"
            "**Operational Interpretation:** Look for gradual, upward slopes which represent cumulative wear, rather than short transient spikes."
        )

if __name__ == "__main__":
    main()
