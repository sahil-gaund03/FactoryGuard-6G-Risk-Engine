import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.components.styles import load_custom_css, load_predictions_data
from app.components.filters import render_sidebar_filters

st.set_page_config(page_title="Maintenance Impact — FactoryGuard 6G", layout="wide")

def main():
    load_custom_css()
    st.title("🛠️ Maintenance Impact Analysis")
    
    df = load_predictions_data()
    if df is None:
        st.warning("Please generate prediction data first.")
        return
        
    df_filtered = render_sidebar_filters(df)
    
    st.markdown("""
    <div class="steel-card">
        <h3>Pre- vs. Post-Maintenance Effectiveness Evaluation</h3>
        <p>This analytics module measures whether maintenance mode events successfully reduce machine health risk. It computes average fused risk scores for the **5 observations prior** to a maintenance window and the **5 observations following** the maintenance window.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Identify maintenance transition indices
    # We loop over machines to find maintenance blocks
    impact_data = []
    
    for machine_id, group in df_filtered.groupby('Machine_ID'):
        group = group.sort_values('datetime').reset_index(drop=True)
        modes = group['Operation_Mode'].values
        scores = group['fused_risk_score'].values
        times = group['datetime'].values
        
        # Look for transitions: mode changes from not-maintenance to maintenance, then back to not-maintenance
        n = len(group)
        i = 5  # start after some history
        while i < n - 10:
            # Transition to maintenance
            if modes[i] == 'Maintenance' and modes[i-1] != 'Maintenance':
                # Find when it exits maintenance
                j = i
                while j < n and modes[j] == 'Maintenance':
                    j += 1
                
                # Exited maintenance
                if j < n - 5:
                    pre_mean = np.mean(scores[i-5:i])
                    post_mean = np.mean(scores[j:j+5])
                    maint_start = times[i]
                    maint_end = times[j]
                    
                    impact_data.append({
                        'Machine_ID': int(machine_id),
                        'Maint_Start': maint_start,
                        'Maint_End': maint_end,
                        'Pre_Maint_Risk': float(pre_mean),
                        'Post_Maint_Risk': float(post_mean),
                        'Risk_Reduction': float(pre_mean - post_mean)
                    })
                    
                i = j  # skip forward
            else:
                i += 1
                
    if not impact_data:
        st.info("No matching maintenance transition events found in the filtered subset. Ensure the filter date range covers maintenance mode operations.")
        return
        
    impact_df = pd.DataFrame(impact_data)
    
    # Overall summary stats
    mean_reduction = impact_df['Risk_Reduction'].mean()
    success_rate = (impact_df['Risk_Reduction'] > 0.0).mean() * 100.0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Effectiveness Metrics")
        st.metric(
            label="Average Risk Reduction",
            value=f"{mean_reduction:.1f} Points",
            delta=f"{mean_reduction:.1f}",
            help="Difference between average risk score before and after maintenance."
        )
        st.metric(
            label="Maintenance Reset Success Rate",
            value=f"{success_rate:.1f}%",
            help="Percentage of maintenance windows that successfully reduced the machine's risk score."
        )
        
    with col2:
        st.subheader("Risk Levels Before & After Intervention")
        plot_df = pd.melt(
            impact_df, 
            id_vars=['Machine_ID', 'Maint_Start'], 
            value_vars=['Pre_Maint_Risk', 'Post_Maint_Risk'],
            var_name='Period', 
            value_name='Risk Score'
        )
        plot_df['Period'] = plot_df['Period'].map({'Pre_Maint_Risk': 'Pre-Maintenance', 'Post_Maint_Risk': 'Post-Maintenance'})
        
        fig_box = px.box(
            plot_df, 
            x='Period', 
            y='Risk Score',
            color='Period',
            color_discrete_map={'Pre-Maintenance': '#ff9800', 'Post-Maintenance': '#4caf50'}
        )
        fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
        st.plotly_chart(fig_box, use_container_width=True)
        
    st.write("---")
    st.subheader("Detailed Maintenance Log")
    log_display = impact_df.copy()
    log_display.columns = ['Machine ID', 'Maintenance Start', 'Maintenance End', 'Pre-Maint Risk Score', 'Post-Maint Risk Score', 'Risk Reduction']
    log_display = log_display.sort_values(by='Maintenance Start', ascending=False)
    st.dataframe(log_display, use_container_width=True)
    
    st.info(
        "**What this shows:** The diagnostic effectiveness of scheduled maintenance periods.\n\n"
        "**Why it matters:** If maintenance events fail to reduce risk scores (negative reduction or success rate < 80%), it typically suggests that either the incorrect repair was performed or the sensors themselves need recalibration.\n\n"
        "**Recommended Action:** Flag maintenance tasks with negative risk reductions for quality assurance review."
    )

if __name__ == "__main__":
    main()
