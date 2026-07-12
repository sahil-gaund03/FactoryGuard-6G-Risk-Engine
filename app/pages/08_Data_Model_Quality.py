import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.components.styles import load_custom_css
from src.factoryguard.paths import DATA_REPORTS, REPORTS_DIR

st.set_page_config(page_title="Data & Model Quality — FactoryGuard 6G", layout="wide")

def main():
    load_custom_css()
    st.title("🛡️ Data & Model Quality Assurance")
    
    st.markdown("""
    <div class="steel-card">
        <h3>System Audits & Model Performance Leaderboards</h3>
        <p>Review pipeline statistics, duplicate handling, schema verification, and metrics of trained bagging, boosting, and ensemble stacking models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Data Ingestion Quality Report", "Machine Learning Leaderboard"])
    
    with tab1:
        st.subheader("Data Pipeline & Validation Summary")
        quality_path = DATA_REPORTS / "data_quality_summary.json"
        
        if quality_path.exists():
            with open(quality_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
                
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Raw CSV Rows Loaded", f"{report['raw_rows']:,}")
                st.metric("Processed Rows Saved", f"{report['processed_rows']:,}")
            with col2:
                st.metric("Exact Duplicates Removed", f"{report['exact_duplicates_count']}")
                st.metric("Deduplicated Timestamps", f"{report['duplicate_machine_timestamps_count']}")
            with col3:
                st.metric("Median Time Gap", f"{report['median_gap_mins']:.1f} mins")
                st.metric("Irregular Gaps (>1.1m)", f"{report['irregular_gaps_count']:,}")
                
            st.write("---")
            st.subheader("Numeric Range Validation Anomalies")
            anoms = report.get('range_anomalies', {})
            if anoms:
                for key, val in anoms.items():
                    st.warning(f"Field `{key}` had {val} observations out of normal boundary ranges during ingestion validation.")
            else:
                st.success("All raw columns matched ranges during ingestion checks.")
        else:
            st.info("No data quality report file found. Run scripts/run_data_pipeline.py to generate it.")
            
    with tab2:
        st.subheader("Calibrated Supervised Classifier Leaderboard")
        leaderboard_path = REPORTS_DIR / "tables" / "model_leaderboard.csv"
        
        if leaderboard_path.exists():
            lead_df = pd.read_csv(leaderboard_path)
            # Sort by PR-AUC descending
            lead_df = lead_df.sort_values(by='pr_auc', ascending=False)
            
            # Format and display
            lead_df.columns = ['PR-AUC (Primary)', 'ROC-AUC', 'Precision', 'Recall', 'F2 Score', 'Brier Score', 'Model Name']
            st.dataframe(lead_df, width="stretch")
            
            st.info(
                "**Selection Rule:** The model with the highest test **PR-AUC** (Precision-Recall Area Under Curve) is designated as the production predictor.\n\n"
                "**Current Status:** Calibrated XGBoost serves as the production model, offering optimal predictive capability on chronological validation folds."
            )
        else:
            st.info("No model leaderboard found. Run scripts/train_supervised_models.py to generate it.")

if __name__ == "__main__":
    main()
