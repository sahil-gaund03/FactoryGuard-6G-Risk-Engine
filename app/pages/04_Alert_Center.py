import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.components.styles import load_custom_css, load_predictions_data
from app.components.filters import render_sidebar_filters

st.set_page_config(page_title="Alert Center — FactoryGuard 6G", layout="wide")

def main():
    load_custom_css()
    st.title("🚨 Alert Center")
    
    df = load_predictions_data()
    if df is None:
        st.warning("Please generate prediction data first.")
        return
        
    df_filtered = render_sidebar_filters(df)
    
    st.markdown("""
    <div class="steel-card">
        <h3>Active Fleet Anomalies and Warning Logs</h3>
        <p>This center acts as the plant floor operations log, showing all observations classified as **Medium** or **High** risk. Review actions and evidence strength for each logged item.</p>
    </div>
    """, unsafe_allow_html=True)
    
    alerts_df = df_filtered[df_filtered['risk_level'].isin(['Medium', 'High'])].sort_values('datetime', ascending=False)
    
    if alerts_df.empty:
        st.success("No active alerts logged for the current filter criteria. Fleet is operating within normal parameters.")
        return
        
    display_cols = [
        'datetime', 'Machine_ID', 'risk_level', 'escalation_category',
        'fused_risk_score', 'evidence_strength', 'recommended_action'
    ]
    
    alerts_display = alerts_df[display_cols].copy()
    alerts_display.columns = [
        'Timestamp', 'Machine ID', 'Risk Level', 'Escalation Category',
        'Fused Risk Score', 'Evidence Strength', 'Recommended Action'
    ]
    
    styler = alerts_display.style
    style_func = lambda val: 'color: #ff5252; font-weight: bold;' if val == 'High' \
                 else ('color: #ffb74d; font-weight: bold;' if val == 'Medium' else '')
    
    if hasattr(styler, 'map'):
        styler = styler.map(style_func, subset=['Risk Level'])
    else:
        styler = styler.applymap(style_func, subset=['Risk Level'])
        
    st.dataframe(
        styler,
        width="stretch"
    )
    
    st.info(
        "**What this shows:** The chronological log of fleet health warnings.\n\n"
        "**Why it matters:** Sustained warnings of high evidence strength represent severe statistical deviations.\n\n"
        "**Recommended Action:** Technical managers should cross-verify the high-evidence alerts first."
    )

if __name__ == "__main__":
    main()
