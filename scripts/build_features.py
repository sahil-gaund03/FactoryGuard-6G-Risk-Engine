import logging
import sys
import joblib
import pandas as pd
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.factoryguard.paths import DATA_INTERIM, DATA_FEATURES, MODELS_DIR
from src.factoryguard.logging_config import setup_logging
from src.factoryguard.config_loader import get_base_config
from src.factoryguard.features.feature_pipeline import LeakageSafeFeaturePipeline

def main():
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("factoryguard.scripts.build_features")
    logger.info("Starting feature engineering process...")
    
    # 1. Load configs and paths
    config = get_base_config()
    split_date = pd.to_datetime(config['pipeline']['split_date'])
    
    parquet_path = DATA_INTERIM / "validated_data.parquet"
    if not parquet_path.exists():
        logger.error(f"Interim Parquet not found at {parquet_path}. Run run_data_pipeline.py first!")
        sys.exit(1)
        
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} rows for feature engineering.")
    
    # 2. Chronological Train/Test split for baseline fitting
    # All baseline normalizations must be fitted on the train set (prior to split_date)
    df_train = df[df['datetime'] < split_date]
    logger.info(f"Training split for baseline learning: {len(df_train)} rows (< {split_date})")
    
    # 3. Create, fit and transform the pipeline
    pipeline = LeakageSafeFeaturePipeline(epsilon=config['pipeline']['epsilon'])
    pipeline.fit(df_train)
    
    # Transform the entire dataset (leakage-safely using fitted baselines)
    df_features = pipeline.transform(df)
    
    # 4. Save results
    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    features_output_path = DATA_FEATURES / "engineered_features.parquet"
    logger.info(f"Saving engineered features to {features_output_path}...")
    df_features.to_parquet(features_output_path, index=False)
    
    # Save the pipeline joblib for future inference
    pipeline_dir = MODELS_DIR / "preprocessing"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = pipeline_dir / "feature_pipeline.joblib"
    logger.info(f"Saving preprocessing pipeline to {pipeline_path}...")
    joblib.dump(pipeline, pipeline_path)
    
    logger.info("Feature engineering completed successfully.")

if __name__ == "__main__":
    main()
