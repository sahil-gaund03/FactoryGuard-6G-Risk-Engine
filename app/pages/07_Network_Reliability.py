import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.components.styles import load_custom_css, load_predictions_data
from app.components.filters import render_sidebar_filters

st.set_page_config(page_title="Network Reliability — FactoryGuard 6G", layout="wide")

def main():
    load_custom_css()
    st.title("📡 Network Reliability Analysis")
    
    df = load_predictions_data()
    if df is None:
        st.warning("Please generate prediction data first.")
        return
        
    df_filtered = render_sidebar_filters(df)
    
    st.markdown("""
    <div class="steel-card">
        <h3>6G Network Instability & Telemetry Correlation</h3>
        <p>In 6G-integrated smart manufacturing, connection drops or high latency can delay predictive alert routing. This page monitors latency, packet loss, and analyzes if connection degradation correlates with physical sensor deviations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Network Latency vs. Packet Loss")
        fig_scatter = px.scatter(
            df_filtered.sample(n=min(len(df_filtered), 2000), random_state=42), 
            x='Network_Latency_ms', 
            y='Packet_Loss_%',
            color='risk_level',
            color_discrete_map={'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'},
            title="Communication Performance Scatter"
        )
        fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
        st.plotly_chart(fig_scatter, width="stretch")
        st.info(
            "**What this shows:** A scatter plot mapping communication latency (ms) against packet loss (%) for a random sample of telemetry records.\n\n"
            "**Why it matters:** Cluster patterns of High risk at elevated packet loss/latency suggest network bottlenecks are coinciding with (or causing) telemetry instability.\n\n"
            "**Recommended Action:** If latency consistently exceeds 40 ms, consult the local networks team to adjust gateway routes."
        )
        
    with col2:
        st.subheader("Instability Timelines")
        # Compare Mean Anomaly Score and Mean Network Instability daily
        daily_perf = df_filtered.groupby(df_filtered['datetime'].dt.date).agg(
            mean_risk=('fused_risk_score', 'mean'),
            mean_network=('network_instability_index', 'mean')
        ).reset_index()
        daily_perf.columns = ['Date', 'Mean Risk Score', 'Mean Network Instability']
        
        fig_line = px.line(
            daily_perf, 
            x='Date', 
            y=['Mean Risk Score', 'Mean Network Instability'],
            title="Operational vs. Network Instability Over Time"
        )
        fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
        st.plotly_chart(fig_line, width="stretch")
        st.info(
            "**What this shows:** The daily trend of mean machine risk alongside mean network instability.\n\n"
            "**Why it matters:** Coinciding peaks indicate connection drops and machine warnings are linked, pointing to potential power source fluctuations or data transmission drops.\n\n"
            "**Recommended Action:** Use this chart to determine whether alert peaks are physical or net-based."
        )

if __name__ == "__main__":
    main()
