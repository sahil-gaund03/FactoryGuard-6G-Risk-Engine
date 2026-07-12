import streamlit as st
from pathlib import Path
import sys

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.components.styles import load_custom_css

st.set_page_config(page_title="Methodology & Limitations — FactoryGuard 6G", layout="wide")

def main():
    load_custom_css()
    st.title("📖 Methodology & Responsible-Use Guide")
    
    st.markdown("""
    <div class="steel-card">
        <h3>Dual-Track Modeling Methodology</h3>
        <p>Because the smart factory telemetry data lacks explicit repair receipts, breakdown times, component tags, or remaining useful life (RUL) labels, 
        <strong>FactoryGuard 6G</strong> uses a dual-track analytical architecture rather than claiming direct failure prediction.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Analytical Track A: Unsupervised Anomaly Detection")
        st.markdown("""
        * **Model:** Isolation Forest fitted on chronological training split.
        * **Objective:** Detect multi-sensor combinations that are statistically rare compared to historical baseline values.
        * **Normalization:** Native isolation scores are converted to a 0–100 percentile rank. A score of 95 means the record is more anomalous than 95% of training samples.
        * **Exclusions:** Excludes `Efficiency_Status` and `Predictive_Maintenance_Score` to prevent data leakage.
        """)
        
        st.subheader("Analytical Track B: Supervised Deterioration Proxy")
        st.markdown(r"""
        * **Model:** Calibrated XGBoost Classifier selected from a temporal cross-validation leaderboard.
        * **Target Definition:** Binary lookahead target (24-hour horizon) indicating if the machine is headed for severe degradation (defined as sustained heat $\ge 75^\circ\text{C}$, sustained vibration $\ge 3.0\text{ Hz}$, or sustained error rates $\ge 9\%$).
        * **Calibration:** Platt Sigmoid scaling mapping raw predictions to actual probabilities.
        """)
        
    with col2:
        st.subheader("⚠️ Important Modeling Limitations & Disclaimers")
        st.warning("""
        * **No Confirmed Failure Predictions:** The system does not predict physical breakdown events, broken components, or downtime duration. The alerts are early warning indicators of *statistical risk*, not deterministic mechanical diagnoses.
        * **Not a Control Loop:** The system is a decision-support dashboard for plant supervisors and technicians. It does not automate machinery shutdown or replacement orders.
        * **Network and Data Dependencies:** Network jitter, latency, or sensor degradation can trigger false alerts. Network quality indicators must be reviewed alongside sensor telemetry.
        * **Causation vs Correlation:** The model establishes statistical correlations. It cannot mathematically prove the physical cause of a machine malfunction.
        """)
        
    st.info(
        "**Guidance for Operators:** Always cross-reference the 6G network latency status prior to dispatching teams for physical inspections. "
        "High latency or packet loss can cause telemetry spikes that resemble physical anomalies."
    )

if __name__ == "__main__":
    main()
