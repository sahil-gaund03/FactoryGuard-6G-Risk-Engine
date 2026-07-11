import logging
import sys
import joblib
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.factoryguard.paths import DATA_PROCESSED, MODELS_DIR
from src.factoryguard.logging_config import setup_logging

def evaluate_fold_pr_auc(model, X_train, y_train, X_val, y_val):
    """Fit model on training split and return PR-AUC on validation split."""
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
    else:
        return 0.0
    precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_prob)
    return auc(recall_vals, precision_vals)

def main():
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("factoryguard.scripts.optimize_hyperparameters")
    logger.info("Starting hyperparameter optimization with Optuna...")
    
    # 1. Load data
    input_path = DATA_PROCESSED / "labeled_features.parquet"
    if not input_path.exists():
        logger.error(f"Labeled features not found at {input_path}. Run build_proxy_labels.py first!")
        sys.exit(1)
        
    df = pd.read_parquet(input_path)
    df = df.sort_values(by=['Machine_ID', 'datetime']).reset_index(drop=True)
    
    # Select feature cols (numeric only, excluding targets and leaky columns)
    exclude_cols = {
        'datetime', 'Machine_ID', 'Operation_Mode', 'Efficiency_Status',
        'Predictive_Maintenance_Score', 'target_deterioration_6h',
        'target_deterioration_24h', 'target_deterioration_72h',
        'Date', 'Timestamp'
    }
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    target_col = 'target_deterioration_24h'
    
    # 2. Chronological split (we only tune on the training split, pre-2025-02-25)
    split_date = pd.to_datetime("2025-02-25")
    df_train = df[df['datetime'] < split_date].copy()
    
    # We do a 3-fold chronological walk-forward split on df_train
    n_samples = len(df_train)
    fold_size = n_samples // 4
    
    def objective(trial):
        # Suggest parameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        
        pr_aucs = []
        # Walk-forward CV
        for i in range(3):
            train_idx = (i + 1) * fold_size
            val_idx = (i + 2) * fold_size
            
            df_tr = df_train.iloc[:train_idx]
            df_va = df_train.iloc[train_idx:val_idx]
            
            X_tr, y_tr = df_tr[feature_cols], df_tr[target_col]
            X_va, y_va = df_va[feature_cols], df_va[target_col]
            
            model = XGBClassifier(**params)
            score = evaluate_fold_pr_auc(model, X_tr, y_tr, X_va, y_va)
            pr_aucs.append(score)
            
        return np.mean(pr_aucs)
        
    # 3. Create study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    logger.info("Running 10 Optuna optimization trials (timed search)...")
    study.optimize(objective, n_trials=10)
    
    logger.info(f"Best Trial Score (Validation PR-AUC): {study.best_value:.4f}")
    logger.info(f"Best Hyperparameters: {study.best_params}")
    
    # 4. Save Study Object to registry
    registry_dir = MODELS_DIR / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    study_path = registry_dir / "optuna_xgboost_study.joblib"
    logger.info(f"Saving Optuna study to {study_path}...")
    joblib.dump(study, study_path)
    
    logger.info("Hyperparameter optimization completed successfully.")

if __name__ == "__main__":
    main()
