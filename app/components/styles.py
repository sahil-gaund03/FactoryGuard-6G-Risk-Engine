import streamlit as st
import pandas as pd
# Force Python string storage to prevent pyarrow string segmentation faults on Linux
pd.options.mode.string_storage = "python"

from pathlib import Path
from src.factoryguard.paths import DATA_PROCESSED

def load_custom_css():
    """Inject custom CSS for styling surfaces, metric cards, and text hierarchy."""
    st.markdown("""
        <style>
        /* Main background and font sizes */
        .main {
            background-color: #121212;
            color: #e0e0e0;
            font-family: 'Outfit', 'Inter', sans-serif;
        }
        
        /* Premium industrial steel-gray cards */
        .steel-card {
            background-color: #1e1e1e;
            border-left: 5px solid #00bcd4;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .steel-card h3 {
            margin-top: 0px;
            color: #00bcd4;
        }
        
        /* Metric tooltips and KPI styling */
        .kpi-container {
            display: flex;
            justify-content: space-between;
            background-color: #262626;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid #333;
        }
        
        .kpi-value {
            font-size: 24px;
            font-weight: bold;
            color: #00e5ff;
        }
        
        /* Risk labels */
        .risk-low {
            color: #4caf50;
            font-weight: bold;
        }
        .risk-medium {
            color: #ff9800;
            font-weight: bold;
        }
        .risk-high {
            color: #f44336;
            font-weight: bold;
        }
        .risk-gray {
            color: #9e9e9e;
            font-weight: bold;
        }
        
        /* Technical expanders */
        .tech-section {
            background-color: #1a1a1a;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 10px 15px;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_predictions_data():
    """Load and cache the dashboard predictions dataset."""
    # Try slim dashboard file first (for Streamlit Cloud), fallback to full file
    file_path = DATA_PROCESSED / "dashboard_data.parquet"
    if not file_path.exists():
        file_path = DATA_PROCESSED / "final_risk_predictions.parquet"
    if not file_path.exists():
        return None
    df = pd.read_parquet(file_path)
    # Ensure datetime is formatted properly
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df
