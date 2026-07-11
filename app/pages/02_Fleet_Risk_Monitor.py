import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.components.styles import load_custom_css, load_predictions_data
from app.components.filters import render_sidebar_filters

st.set_page_config(page_title="Fleet Risk Monitor — FactoryGuard 6G", layout="wide")

def main():
    load_custom_css()
    st.title("🖥️ Fleet Risk Monitor")
    
    df = load_predictions_data()
    if df is None:
        st.warning("Please generate prediction data first.")
        return
        
    df_filtered = render_sidebar_filters(df)
    
    st.markdown("""
    <div class="steel-card">
        <h3>Active Fleet Risk Rankings & Priorities</h3>
        <p>This table lists machines ordered by their **Maintenance Priority Score** (which combines the fused risk score and persistence penalties). This assists supervisors in dispatching maintenance technicians to the units most in need of attention.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Compute latest status per machine
    latest_status = df_filtered.sort_values('datetime').groupby('Machine_ID').last().reset_index()
    # Sort by priority descending
    latest_status = latest_status.sort_values(by='maintenance_priority', ascending=False)
    
    # Select columns to display
    display_cols = [
        'Machine_ID', 'risk_level', 'escalation_category', 
        'maintenance_priority', 'evidence_strength', 'recommended_action'
    ]
    
    latest_display = latest_status[display_cols].copy()
    latest_display.columns = [
        'Machine ID', 'Risk Level', 'Escalation Category', 
        'Maintenance Priority (0-100)', 'Evidence Strength', 'Recommended Action'
    ]
    
    # Reset index and make 1-based index for rankings
    latest_display.reset_index(drop=True, inplace=True)
    latest_display.index += 1
    
    styler = latest_display.style
    style_func = lambda val: 'background-color: #2c0c0e; color: #ff5252; font-weight: bold;' if val == 'High' \
                 else ('background-color: #2c1e0c; color: #ffb74d; font-weight: bold;' if val == 'Medium' \
                       else ('background-color: #0c2c12; color: #81c784; font-weight: bold;' if val == 'Low' else ''))
    
    if hasattr(styler, 'map'):
        styler = styler.map(style_func, subset=['Risk Level'])
    else:
        styler = styler.applymap(style_func, subset=['Risk Level'])
        
    st.dataframe(
        styler,
        width="stretch"
    )
    
    st.info(
        "**What this shows:** The current operational health priority table for the fleet.\n\n"
        "**Why it matters:** Higher scores represent units with overlapping alerts (both anomaly indicators and supervised deterioration proxy risks) and sustained warnings.\n\n"
        "**Recommended Action:** Direct technicians to inspect units ranked #1 and #2 immediately."
    )

if __name__ == "__main__":
    main()
