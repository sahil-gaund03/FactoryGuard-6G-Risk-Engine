# FactoryGuard 6G — Predictive Maintenance & Anomaly Detection

FactoryGuard 6G is a condition-monitoring and predictive-maintenance decision-support platform for 6G-integrated smart manufacturing systems. It processes physical machine telemetry alongside 6G network indicators (latency, packet loss) to isolate anomalies and predict operational deterioration.

## Live Application
👉 **Public App URL:** [factoryguard-6g.streamlit.app](https://factoryguard-6g.streamlit.app) *(Placeholder)*

---

## Folder Architecture

```
factoryguard-6g/
├── app/                  # Streamlit Multi-page Dashboard
│   ├── Home.py           # Landing Entry Point
│   ├── pages/            # 10 Subpages
│   └── components/       # Custom CSS & Filters
├── config/               # YAML settings for paths/models/features
├── data/
│   ├── raw/              # Raw telemetry CSV
│   ├── interim/          # Validated and scored Parquet data
│   └── processed/        # Final risk predictions Parquet
├── models/               # Preprocessing, Anomaly, & Supervised models
├── reports/              # Leaderboard tables & PDF summaries
├── scripts/              # Pipeline automation runners
├── src/                  # Core package source code
└── tests/                # Unit & integration tests
```

---

## Dual-Track Architecture

### Track A: Unsupervised Anomaly Engine (Isolation Forest)
* Learns normal machine baseline behavior from the training split.
* Excludes leakage columns (`Efficiency_Status` and `Predictive_Maintenance_Score`).
* Normalizes scores to a `0-100` percentile ranking where `0` is standard and `100` is most abnormal.

### Track B: Supervised Proxy Engine (Calibrated XGBoost)
* Predicts future operational deterioration within a 24-hour lookahead window.
* Deterioration is defined as a sustained low efficiency state combined with high temperature ($>75^\circ\text{C}$), high vibration ($>3.0\text{ Hz}$), or high error rates ($>9\%$).
* Calibrated via Platt sigmoid scaling to return accurate probabilities.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sahilgaund/factoryguard-6g.git
   cd factoryguard-6g
   ```

2. **Create and activate the virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/MacOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```

---

## Running the Data & ML Pipeline

Execute the pipeline scripts sequentially:

1. **Ingest and Validate Raw Data:**
   ```bash
   python scripts/run_data_pipeline.py
   ```
2. **Build Features:**
   ```bash
   python scripts/build_features.py
   ```
3. **Construct Deterioration Proxy Labels:**
   ```bash
   python scripts/build_proxy_labels.py
   ```
4. **Train Anomaly Engine:**
   ```bash
   python scripts/train_anomaly_models.py
   ```
5. **Train Supervised Models & Ensembles:**
   ```bash
   python scripts/train_supervised_models.py
   ```
6. **Execute Risk Fusion Engine:**
   ```bash
   python scripts/generate_predictions.py
   ```

---

## Running the Dashboard

Run the Streamlit application locally:
```bash
streamlit run app/Home.py
```

---

## Running Tests

Verify that all unit and integration tests pass successfully:
```bash
pytest tests/
```

---

## Important Modeling Disclaimers
> [!WARNING]
> **No Confirmed Breakdowns:** The dataset contains no confirmed repair logs or mechanical breakdowns. The system predicts **statistical proxy risk** and **deterioration indicators**, not guaranteed component failures. All outputs must be validated by qualified site engineers prior to physical machine dismantling.
