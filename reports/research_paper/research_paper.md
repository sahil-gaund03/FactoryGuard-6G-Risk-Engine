# Research Paper: FactoryGuard 6G — Predictive Maintenance and Anomaly Detection in 6G-Integrated Smart Manufacturing Systems

## Abstract
Modern smart manufacturing environments depend on low-latency industrial communication links and high-dimensional sensor telemetry to monitor machine health and maintain production throughput. This paper presents **FactoryGuard 6G**, a condition-monitoring and predictive maintenance platform built for 6G-integrated factory floors. Using a dataset of 100,000 observations across 50 machines, we develop a dual-track analytical architecture combining unsupervised anomaly detection (Isolation Forest) and supervised future-deterioration proxy prediction (calibrated XGBoost). The model outputs are fused with time-aware persistence and trend metrics to compute a prioritized maintenance risk score. Our results demonstrate that incorporating network reliability (latency, packet loss) improves the distinction between communication glitches and physical faults.

**Keywords:** Smart Manufacturing, 6G Network, Predictive Maintenance, Anomaly Detection, Isolation Forest, Stacking, Calibration.

---

## 1. Introduction & Context
Industrial IoT (IIoT) systems under the 6G paradigm support real-time telemetry streaming and high-reliability control loops. In these settings, communication latency and packet loss serve not only as network status indicators but also as key factors in condition-monitoring reliability. FactoryGuard 6G addresses the challenge of monitoring multi-sensor machines without ground-truth failure logs, leveraging machine-mode baselines and future window average degradation targets.

---

## 2. Problem Statement & Objectives
Static sensor thresholds fail to adapt to:
1. Differences in normal ranges across distinct machine units (`Machine_ID`).
2. Operational mode changes (`Active` vs. `Idle` vs. `Maintenance`).
3. 6G communication disruptions which mimic sensor anomalies.

Our objectives are:
* Personalize baseline thresholds per machine and operation mode.
* Implement leakage-safe time-series feature engineering.
* Develop a calibrated predictive risk engine.
* Render explanations in non-technical language.

---

## 3. Dataset Ingestion & Data-Quality
We analyze `Thales_Group_Manufacturing.csv` containing 100,000 records. Datetime parsing is executed using day-first parsing. Schema checks verify the presence of 14 raw columns. Outliers are preserved to detect anomalous states. Duplicate Machine-ID and timestamp records (28 occurrences) are handled via median aggregation of numeric fields.

---

## 4. Methodology
We implement a dual-track ML architecture:

```
[Telemetry & Network Raw Data]
       |
       v
[Baseline Personalization & Feature Extractor]
       |
       +-----------------------+-----------------------+
       |                                               |
       v                                               v
[Track A: Isolation Forest]                    [Track B: XGBoost]
(Unsupervised percentile scores)               (Calibrated 24h proxy risk)
       |                                               |
       +-----------------------+-----------------------+
                               |
                               v
                   [Risk Fusion Engine]
                    - Fused Risk Score
                    - Escalation States
                    - Maintenance Priority
```

### 4.1 Feature Engineering
Features are engineered chronologically:
* **Baselines:** Robust z-score mapping `(value - median) / MAD` per machine-mode group.
* **Rolling:** Mean, std, min, max over 3, 5, 10, and 20-row windows.
* **Lags:** Shifted offsets of 1, 2, 3, 5, and 10 periods.
* **Interactions:** Network instability index (`latency * packet_loss`), sensor instability index (`temp_z * vib_z`).

### 4.2 Proxy Target Design
Since actual failure events are absent, we design a 24-hour lookahead target representing operational deterioration (mean temperature $>75^\circ\text{C}$, mean vibration $>3.0\text{ Hz}$, or mean error rate $>9.0\%$).

---

## 5. Results & Discussion
During evaluation on the chronological test split (post-`2025-02-25`), XGBoost outperformed all other models, achieving the highest test **PR-AUC (0.088)** and a well-calibrated Brier Score (**0.085**). 

The final **Risk Fusion Engine** groups risk ratings into Low, Medium, and High, and maps them to operational escalation states (e.g. *Persistent High Risk*, *Data Quality Concern*).

---

## 6. Dashboard Interface & Explainability
A 10-page Streamlit application provides fleet risk ranking, machine diagnostics, alert centers, and network reliability scatter plots. Explanations are rendered using local z-score rankings to identify the top 3 contributing sensors, translating statistical outputs into operational recommendation checklists.

---

## 7. Future Work & Conclusion
Future efforts should focus on collecting actual component breakdown records and validation logs to move from proxy labels to true failure predictions. FactoryGuard 6G establishes a robust framework for condition monitoring in smart, 6G-integrated factories.

## References
1. Thales Group manufacturing analytics dataset (2025).
2. Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons.
3. Breiman, L. (2001). Random Forests. Machine Learning.
