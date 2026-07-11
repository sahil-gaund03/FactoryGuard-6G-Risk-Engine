import logging
import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, precision_score, recall_score, fbeta_score, brier_score_loss
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.factoryguard.paths import DATA_INTERIM, DATA_PREDICTIONS, DATA_REPORTS, MODELS_DIR, REPORTS_DIR
from src.factoryguard.logging_config import setup_logging
from src.factoryguard.config_loader import get_base_config, get_models_config
from src.factoryguard.models.stacking import ChronologicalStackingClassifier
from src.factoryguard.models.calibration import ProbabilityCalibrator

def evaluate_metrics(y_true, y_prob, threshold=0.5):
    """Calculate key evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    # Precision-Recall AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_vals, precision_vals)
    
    # ROC AUC
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    
    # Precision, Recall, F2, Brier
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'precision': float(prec),
        'recall': float(rec),
        'f2_score': float(f2),
        'brier_score': float(brier)
    }

def main():
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("factoryguard.scripts.train_supervised_models")
    logger.info("Starting supervised model training and evaluation...")
    
    # 1. Load data and configs
    base_config = get_base_config()
    models_config = get_models_config()
    split_date = pd.to_datetime(base_config['pipeline']['split_date'])
    
    parquet_path = DATA_INTERIM / "anomalies_scored.parquet"
    if not parquet_path.exists():
        logger.error(f"Scored anomalies Parquet not found at {parquet_path}. Run train_anomaly_models.py first!")
        sys.exit(1)
        
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} rows for supervised modeling.")
    
    # 2. Select feature columns
    # We include anomaly scores as inputs!
    exclude = [
        'Efficiency_Status', 'Predictive_Maintenance_Score', 'Machine_ID',
        'Date', 'Timestamp', 'datetime', 'source_row_id'
    ]
    target_col = "target_deterioration_24h"
    
    feature_cols = [
        col for col in df.select_dtypes(include=['number']).columns
        if col not in exclude and col != target_col and not col.startswith('target_')
    ]
    
    logger.info(f"Using {len(feature_cols)} features for supervised learning.")
    
    # 3. Chronological Train/Test Split
    df_train = df[df['datetime'] < split_date].copy()
    df_test = df[df['datetime'] >= split_date].copy()
    
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]
    
    logger.info(f"Train split: {len(X_train)} rows (pre-{split_date})")
    logger.info(f"Test split: {len(X_test)} rows (post-{split_date})")
    
    # Class distribution
    logger.info(f"Train positive prevalence: {y_train.mean():.2%}")
    logger.info(f"Test positive prevalence: {y_test.mean():.2%}")
    
    # Define candidates
    base_candidates = {
        'dummy': DummyClassifier(strategy='prior'),
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(C=1.0, random_state=42, max_iter=1000)),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'extra_trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'xgboost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1),
        'lightgbm': LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1),
        'catboost': CatBoostClassifier(iterations=100, depth=5, learning_rate=0.05, random_seed=42, verbose=0)
    }
    
    leaderboard = []
    trained_models = {}
    
    # 4. Train and evaluate base models
    for name, model in base_candidates.items():
        logger.info(f"Training {name} base model...")
        model.fit(X_train, y_train)
        
        # Predict on test
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)
            
        # Calibration (except dummy)
        if name != 'dummy':
            logger.info(f"Calibrating {name}...")
            # We fit calibrator on train predictions (for simplicity and safety, we can use out-of-fold or simple validation)
            # In a real setup, out-of-fold is preferred. We can do simple Platt calibration
            calibrator = ProbabilityCalibrator(method='sigmoid')
            # Use train predictions to fit calibrator
            if hasattr(model, "predict_proba"):
                train_probs = model.predict_proba(X_train)[:, 1]
            else:
                train_probs = model.predict(X_train)
            calibrator.fit(train_probs, y_train)
            y_prob = calibrator.calibrate(y_prob)
            trained_models[f"{name}_calibrated"] = (model, calibrator)
            
        metrics = evaluate_metrics(y_test, y_prob)
        metrics['model'] = name
        leaderboard.append(metrics)
        logger.info(f"Model {name} test metrics: {metrics}")
        
    # 5. Train Stacking Ensemble
    stacking_base_models = {
        'lightgbm': base_candidates['lightgbm'],
        'catboost': base_candidates['catboost'],
        'random_forest': base_candidates['random_forest'],
        'logistic_regression': base_candidates['logistic_regression']
    }
    
    stacker = ChronologicalStackingClassifier(base_models=stacking_base_models)
    # Stacking handles OOF folds internally
    stacker.train_oof_and_fit_meta(df_train, feature_cols, target_col, n_folds=3)
    
    # Predict and Calibrate Stacking predictions
    stack_probs_test = stacker.predict_proba(X_test)
    stack_probs_train = stacker.predict_proba(X_train)
    
    stack_calibrator = ProbabilityCalibrator(method='sigmoid')
    stack_calibrator.fit(stack_probs_train, y_train)
    
    stack_calibrated_probs_test = stack_calibrator.calibrate(stack_probs_test)
    
    stack_metrics = evaluate_metrics(y_test, stack_calibrated_probs_test)
    stack_metrics['model'] = 'stacking_ensemble'
    leaderboard.append(stack_metrics)
    logger.info(f"Stacking Ensemble test metrics: {stack_metrics}")
    
    # 6. Save leaderboard
    leaderboard_df = pd.DataFrame(leaderboard).sort_values(by='pr_auc', ascending=False)
    leaderboard_df.to_csv(REPORTS_DIR / "tables" / "model_leaderboard.csv", index=False)
    
    with open(DATA_REPORTS / "model_leaderboard.json", 'w', encoding='utf-8') as f:
        json.dump(leaderboard, f, indent=4)
        
    # 7. Select best production model (excluding dummy)
    real_models_df = leaderboard_df[leaderboard_df['model'] != 'dummy']
    best_model_name = real_models_df.iloc[0]['model']
    logger.info(f"Best model based on PR-AUC is: {best_model_name}")
    
    # Save production model
    prod_dir = MODELS_DIR / "supervised"
    prod_dir.mkdir(parents=True, exist_ok=True)
    prod_model_path = prod_dir / "production_model.joblib"
    
    if best_model_name == 'stacking_ensemble':
        joblib.dump((stacker, stack_calibrator, 'stacking'), prod_model_path)
    else:
        best_model_key = f"{best_model_name}_calibrated"
        best_tuple = trained_models[best_model_key]
        joblib.dump((best_tuple[0], best_tuple[1], best_model_name), prod_model_path)
        
    # 8. Compute final predictions for the entire dataset using the best model
    logger.info("Generating predictions for the entire dataset using production model...")
    # Load production model
    prod_model, prod_calibrator, model_type = joblib.load(prod_model_path)
    
    X_all = df[feature_cols]
    if model_type == 'stacking':
        raw_probs = prod_model.predict_proba(X_all)
    else:
        if hasattr(prod_model, "predict_proba"):
            raw_probs = prod_model.predict_proba(X_all)[:, 1]
        else:
            raw_probs = prod_model.predict(X_all)
            
    calibrated_probs = prod_calibrator.calibrate(raw_probs)
    
    # Add outputs
    df['proxy_maintenance_risk_prob'] = calibrated_probs
    # Convert probability to risk rating class
    # High: >= 97.5th percentile of predictions. Medium: 90th to 97.5th. Low: < 90th.
    p_med = np.percentile(calibrated_probs, models_config['risk_thresholds']['percentile_medium'])
    p_high = np.percentile(calibrated_probs, models_config['risk_thresholds']['percentile_high'])
    
    df['risk_level_supervised'] = 'Low'
    df.loc[df['proxy_maintenance_risk_prob'] >= p_med, 'risk_level_supervised'] = 'Medium'
    df.loc[df['proxy_maintenance_risk_prob'] >= p_high, 'risk_level_supervised'] = 'High'
    
    # Save predictions
    DATA_PREDICTIONS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PREDICTIONS / "supervised_predictions.parquet", index=False)
    
    logger.info("Supervised model training and prediction generation completed successfully.")

if __name__ == "__main__":
    main()
