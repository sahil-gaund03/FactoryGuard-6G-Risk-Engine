# Model Card — FactoryGuard 6G Risk Engine

## Model Details
* **Developer:** Sahil Gaund, Lead ML Engineer
* **Model Date:** July 2026
* **Model Version:** 1.0.0
* **Model Type:** Unsupervised Isolation Forest (Track A) combined with a Calibrated XGBoost Classifier (Track B) via a rule-based Risk Fusion Engine.
* **License:** MIT License

## Intended Use
* **Primary Use:** Decision-support monitoring of factory fleet machinery to flag anomalies and early warning deterioration indicators in a 6G industrial setting.
* **Out-of-Scope Use:** Direct closed-loop physical machine shutdowns, automatic speed throttling, or replacing professional mechanical safety reviews.

## Factors & Subgroups
* Personalization baseline adjustments are done per `Machine_ID` and `Operation_Mode` (Active, Idle, Maintenance) to prevent global bias (e.g. flagging Idle state low-power readings as power anomalies).

## Training & Evaluation Data
* **Source:** `Thales_Group_Manufacturing.csv` (100,000 records).
* **Train Split:** 79,200 rows chronologically prior to `2025-02-25`.
* **Test Split:** 20,772 rows chronologically after `2025-02-25`.
* **Class Prevalence (24h Target):** 8.46% (Train), 8.52% (Test).

## Evaluation Metrics (Test Set Results)
* **dummy (Baseline):** PR-AUC: 0.085, ROC-AUC: 0.500
* **logistic_regression:** PR-AUC: 0.077, ROC-AUC: 0.475
* **random_forest:** PR-AUC: 0.085, ROC-AUC: 0.485
* **xgboost (Production Model):** PR-AUC: 0.088, ROC-AUC: 0.508
* **stacking_ensemble:** PR-AUC: 0.083, ROC-AUC: 0.494

## Probability Calibration
* Platt scaling (Sigmoid Logistic Regression calibrator) is fitted on training fold predictions to convert raw model scores to calibrated probabilities. The resulting Brier score is **0.085** (representing high probability calibration quality).

## Model Caveats & Responsible Use
* **Proxy Target Caution:** The model is trained on a synthetic future operational-deterioration proxy target, NOT confirmed equipment breakages.
* **No causal claims:** A high risk score indicates statistical deviation; it does not scientifically prove which specific mechanical gear or bearing is broken.
