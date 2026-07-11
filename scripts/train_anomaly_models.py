import logging
import sys
import joblib
import pandas as pd
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.factoryguard.paths import DATA_PROCESSED, DATA_INTERIM, MODELS_DIR
from src.factoryguard.logging_config import setup_logging
from src.factoryguard.config_loader import get_base_config, get_features_config
from src.factoryguard.models.isolation_forest import FactoryGuardIsolationForest

def main():
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("factoryguard.scripts.train_anomaly_models")
    logger.info("Starting anomaly model training...")
    
    # 1. Load configs and paths
    base_config = get_base_config()
    features_config = get_features_config()
    split_date = pd.to_datetime(base_config['pipeline']['split_date'])
    
    parquet_path = DATA_PROCESSED / "labeled_features.parquet"
    if not parquet_path.exists():
        logger.error(f"Labeled features not found at {parquet_path}. Run build_proxy_labels.py first!")
        sys.exit(1)
        
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} rows for anomaly modeling.")
    
    # 2. Select feature columns
    # We must exclude forbidden features and non-feature columns
    forbidden = features_config['leakage_prevention']['forbidden_inputs']
    exclude = features_config['exclude_from_anomaly']
    
    # Only select numeric columns to avoid passing string categories to sklearn
    numeric_df = df.select_dtypes(include=['number'])
    feature_cols = [
        col for col in numeric_df.columns 
        if col not in forbidden 
        and col not in exclude 
        and not col.startswith('target_')
    ]
    
    logger.info(f"Using {len(feature_cols)} features for unsupervised anomaly detection.")
    
    # 3. Train/test split chronologically
    df_train = df[df['datetime'] < split_date]
    X_train = df_train[feature_cols]
    logger.info(f"Training on split prior to {split_date} ({len(X_train)} rows)")
    
    # 4. Train the Isolation Forest model
    model = FactoryGuardIsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=base_config['pipeline']['random_seed']
    )
    model.fit(X_train)
    
    # 5. Compute scores on the entire dataset
    X_all = df[feature_cols]
    df['anomaly_score_raw'] = model.predict_anomaly_score_raw(X_all)
    df['anomaly_score_normalized'] = model.predict_anomaly_score_normalized(X_all)
    
    # 6. Save model and scored dataset
    models_dir = MODELS_DIR / "anomaly"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "isolation_forest.joblib"
    logger.info(f"Saving trained Isolation Forest to {model_path}...")
    joblib.dump(model, model_path)
    
    output_path = DATA_INTERIM / "anomalies_scored.parquet"
    logger.info(f"Saving scored dataset to {output_path}...")
    df.to_parquet(output_path, index=False)
    
    # Log anomaly rate statistics
    anomalous_count = (df['anomaly_score_normalized'] >= 90.0).sum()
    logger.info(f"Scoring complete. {anomalous_count} rows marked as anomalous (>= 90th percentile).")
    logger.info("Anomaly engine training completed successfully.")

if __name__ == "__main__":
    main()
