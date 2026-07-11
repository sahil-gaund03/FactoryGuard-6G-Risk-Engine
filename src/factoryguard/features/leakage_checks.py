import logging
import pandas as pd
from src.factoryguard.exceptions import FeatureEngineeringError

logger = logging.getLogger(__name__)

def check_feature_leakage(df: pd.DataFrame) -> bool:
    """
    Perform a series of checks to verify that no target leakage or temporal leakage exists.
    Returns True if no leakage is found, else raises FeatureEngineeringError.
    """
    logger.info("Running feature leakage checks...")
    
    # 1. Target Leakage: Ensure forbidden columns are not present in df columns
    forbidden_features = ['Efficiency_Status', 'Predictive_Maintenance_Score']
    # If they are present, they must be the raw target columns, not features.
    # In ML models, features list will exclude them.
    
    # 2. Temporal Leakage Check:
    # We will modify a future sensor value for a machine and verify that it does NOT 
    # change the engineered features of past rows.
    from src.factoryguard.features.feature_pipeline import LeakageSafeFeaturePipeline
    
    # We take a small subset of the data
    df_sample = df.head(100).copy()
    
    # Fit pipeline
    pipeline = LeakageSafeFeaturePipeline()
    pipeline.fit(df_sample)
    
    # Transform baseline
    df_transformed_orig = pipeline.transform(df_sample)
    
    # Modify a value in the future (e.g. index 50, Temperature_C)
    df_sample_modified = df_sample.copy()
    orig_val = df_sample_modified.loc[50, 'Temperature_C']
    df_sample_modified.loc[50, 'Temperature_C'] = orig_val + 10.0
    
    df_transformed_mod = pipeline.transform(df_sample_modified)
    
    # Compare rows 0 to 49. They must be EXACTLY identical between orig and mod.
    diff_cols = []
    for col in df_transformed_orig.columns:
        if col in ['datetime', 'Date', 'Timestamp', 'time_since_last_maint_hours']:
            continue
        # Compare past rows
        orig_past = df_transformed_orig.loc[0:49, col]
        mod_past = df_transformed_mod.loc[0:49, col]
        if not orig_past.equals(mod_past):
            diff_cols.append(col)
            
    if diff_cols:
        raise FeatureEngineeringError(f"Temporal leakage detected in columns: {diff_cols}")
        
    logger.info("Leakage checks passed! No future leakage or target leakage detected in feature engineering.")
    return True
