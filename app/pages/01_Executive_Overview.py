import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.components.styles import load_custom_css, load_predictions_data
from app.components.filters import render_sidebar_filters

st.set_page_config(page_title="Executive Overview — FactoryGuard 6G", layout="wide")

def main():
    load_custom_css()
    st.title("📊 Executive Overview")
    
    df = load_predictions_data()
    if df is None:
        st.warning("Please generate prediction data first.")
        return
        
    df_filtered = render_sidebar_filters(df)
    
    st.markdown("""
    <div class="steel-card">
        <h3>Operational Summary & Plant Risk Profile</h3>
        <p>This overview summarizes overall plant stability, highlighting the proportion of fleet units experiencing elevated risk thresholds. This assists operations managers in planning maintenance windows and routing production loads to stable units.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Chart 1: Risk Level Pie Chart
        st.subheader("Fleet Risk Level Distribution")
        risk_counts = df_filtered['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Observations']
        
        fig_pie = px.pie(
            risk_counts, 
            values='Observations', 
            names='Risk Level',
            color='Risk Level',
            color_discrete_map={'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'},
            hole=0.4
        )
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
        st.plotly_chart(fig_pie, width="stretch")
        
        st.info(
            "**What this shows:** The relative proportion of all observations classified as Low, Medium, or High risk.\n\n"
            "**Why it matters:** A high proportion of Medium/High risk indicates systemic factory-floor issues (e.g., power line fluctuations or network lag).\n\n"
            "**Recommended Action:** If High risk exceeds 5% of the active fleet, schedule temporary load-balancing redistributions."
        )
        
    with col2:
        # Chart 2: Risk Over Time
        st.subheader("Daily Risk Trend (Mean Risk Score)")
        daily_trend = df_filtered.groupby(df_filtered['datetime'].dt.date)['fused_risk_score'].mean().reset_index()
        daily_trend.columns = ['Date', 'Mean Risk Score']
        
        fig_line = px.line(
            daily_trend, 
            x='Date', 
            y='Mean Risk Score',
            color_discrete_sequence=['#00e5ff']
        )
        fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
        st.plotly_chart(fig_line, width="stretch")
        
        st.info(
            "**What this shows:** The chronological daily average of fused risk scores across the active fleet.\n\n"
            "**Why it matters:** Sudden spikes in the average score indicate plant-wide stress (e.g., factory ambient temperature rise or gateway bottlenecks).\n\n"
            "**Recommended Action:** Compare trend spikes against local weather records or network updates."
        )

if __name__ == "__main__":
    main()
