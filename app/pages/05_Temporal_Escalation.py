import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.components.styles import load_custom_css, load_predictions_data
from app.components.filters import render_sidebar_filters

st.set_page_config(page_title="Temporal Escalation — FactoryGuard 6G", layout="wide")

def main():
    load_custom_css()
    st.title("📈 Temporal Risk Escalation")
    
    df = load_predictions_data()
    if df is None:
        st.warning("Please generate prediction data first.")
        return
        
    df_filtered = render_sidebar_filters(df)
    
    st.markdown("""
    <div class="steel-card">
        <h3>Escalating and Worsening Fleet Risks</h3>
        <p>This monitor detects machine units experiencing rapid, sequential deterioration. It filters for units with the highest **3-period risk trend slope** (worsening scores over consecutive records), indicating emerging faults.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get latest records per machine to look at current escalation status
    latest_records = df_filtered.sort_values('datetime').groupby('Machine_ID').last().reset_index()
    
    # Filter for active escalation (Rapidly Escalating or Slowly Escalating) or positive slope
    escalating = latest_records[latest_records['risk_trend_slope_3'] > 0.0].sort_values('risk_trend_slope_3', ascending=False)
    
    if escalating.empty:
        st.success("No machines are currently undergoing active risk escalation. Telemetry trends are stable.")
        return
        
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Worsening Telemetry Trend Rankings")
        display_cols = ['Machine_ID', 'escalation_category', 'fused_risk_score', 'risk_trend_slope_3', 'recommended_action']
        esc_display = escalating[display_cols].copy()
        esc_display.columns = ['Machine ID', 'Escalation Category', 'Current Risk Score', '3-Period Score Increase', 'Recommended Action']
        esc_display.reset_index(drop=True, inplace=True)
        esc_display.index += 1
        
        st.dataframe(esc_display, width="stretch")
        
    with col2:
        st.subheader("Worsening Risk Rate (Score Increase)")
        fig_bar = px.bar(
            escalating,
            x='Machine_ID',
            y='risk_trend_slope_3',
            labels={'Machine_ID': 'Machine Unit', 'risk_trend_slope_3': 'Score Increase (3-periods)'},
            color_discrete_sequence=['#ff5252']
        )
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
        st.plotly_chart(fig_bar, width="stretch")
        
    st.info(
        "**What this shows:** Machine units ordered by the rate of their risk score increase.\n\n"
        "**Why it matters:** Units with rapid score increases are actively degrading. These represent emergent issues requiring attention before they become critical.\n\n"
        "**Recommended Action:** Units showing high escalation (slope > 15) should be priority-inspected."
    )

if __name__ == "__main__":
    main()
