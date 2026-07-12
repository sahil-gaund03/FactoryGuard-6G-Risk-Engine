import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.components.styles import load_custom_css, load_predictions_data
from app.components.filters import render_sidebar_filters

st.set_page_config(page_title="Report Export — FactoryGuard 6G", layout="wide")

def main():
    load_custom_css()
    st.title("🖨️ Report & Data Export")
    
    df = load_predictions_data()
    if df is None:
        st.warning("Please generate prediction data first.")
        return
        
    df_filtered = render_sidebar_filters(df)
    
    st.markdown("""
    <div class="steel-card">
        <h3>Operational Reports and Data Export Center</h3>
        <p>Export localized alert files, fleet summaries, and plain-language maintenance recommendations for offline usage or integration with external computerized maintenance management systems (CMMS).</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Active Alerts Export (CSV)")
        st.write("Download the filtered subset of Medium and High risk alerts.")
        alerts_df = df_filtered[df_filtered['risk_level'].isin(['Medium', 'High'])].sort_values('datetime', ascending=False)
        
        if not alerts_df.empty:
            csv_alerts = alerts_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Active Alerts CSV",
                data=csv_alerts,
                file_name="factoryguard_active_alerts.csv",
                mime="text/csv"
            )
        else:
            st.info("No active alerts to download.")
            
    with col2:
        st.subheader("2. Fleet Rankings Export (CSV)")
        st.write("Download the current prioritized maintenance list for the entire fleet.")
        latest_status = df_filtered.sort_values('datetime').groupby('Machine_ID').last().reset_index()
        rankings = latest_status.sort_values(by='maintenance_priority', ascending=False)
        
        csv_rankings = rankings[['Machine_ID', 'risk_level', 'escalation_category', 'maintenance_priority', 'recommended_action']].to_csv(index=False)
        st.download_button(
            label="📥 Download Fleet Rankings CSV",
            data=csv_rankings,
            file_name="factoryguard_fleet_rankings.csv",
            mime="text/csv"
        )
        
    st.write("---")
    st.subheader("3. Executive Text Summary Generator")
    
    high_count = (latest_status['risk_level'] == 'High').sum()
    med_count = (latest_status['risk_level'] == 'Medium').sum()
    worst_unit = int(rankings.iloc[0]['Machine_ID'])
    worst_priority = float(rankings.iloc[0]['maintenance_priority'])
    
    summary_text = (
        "=========================================\n"
        "FACTORYGUARD 6G — EXECUTIVE RISK SUMMARY\n"
        "=========================================\n"
        f"Fleet Size Evaluated: {latest_status['Machine_ID'].nunique()} units\n"
        f"High Risk Units: {high_count} units (Require immediate inspection)\n"
        f"Medium Risk Units: {med_count} units (Schedule for window inspection)\n\n"
        f"Highest Priority Unit: Machine {worst_unit} (Priority Score: {worst_priority:.1f}/100)\n"
        f"Recommended Action for Machine {worst_unit}: {rankings.iloc[0]['recommended_action']}\n\n"
        "Modeling Disclaimer:\n"
        "All calculations represent statistical deviations and proxy targets. "
        "They do not confirm mechanical component breakdown or exact failure times.\n"
        "=========================================\n"
    )
    
    st.text_area("Generated Executive Summary", value=summary_text, height=250)
    st.download_button(
        label="📥 Download Executive Summary Text",
        data=summary_text,
        file_name="factoryguard_executive_summary.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    main()
