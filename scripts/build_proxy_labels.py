import logging
import sys
import json
import pandas as pd
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.factoryguard.paths import DATA_FEATURES, DATA_PROCESSED, DATA_REPORTS
from src.factoryguard.logging_config import setup_logging
from src.factoryguard.labels.horizon_labeler import construct_horizon_targets

def main():
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("factoryguard.scripts.build_proxy_labels")
    logger.info("Starting proxy target construction...")
    
    parquet_path = DATA_FEATURES / "engineered_features.parquet"
    if not parquet_path.exists():
        logger.error(f"Engineered features not found at {parquet_path}. Run build_features.py first!")
        sys.exit(1)
        
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {df.shape[0]} rows for labeling.")
    
    # Construct targets
    df_labeled, stats = construct_horizon_targets(df, horizons_hours=[6, 24, 72])
    
    # Save the processed dataset
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_REPORTS.mkdir(parents=True, exist_ok=True)
    
    output_path = DATA_PROCESSED / "labeled_features.parquet"
    logger.info(f"Saving final labeled features to {output_path}...")
    df_labeled.to_parquet(output_path, index=False)
    
    # Save target summary statistics
    summary_path = DATA_REPORTS / "proxy_label_summary.json"
    logger.info(f"Saving label definition summary to {summary_path}...")
    
    label_summary = {
        'primary_horizon': '24h',
        'horizons_evaluated': [6, 24, 72],
        'criteria': {
            'efficiency': 'Efficiency_Status == Low',
            'adverse_triggers': [
                'Production_Speed_units_per_hr < 150',
                'Error_Rate_% > 12.0',
                'Quality_Control_Defect_Rate_% > 8.0',
                'Temperature_C > 85.0',
                'Vibration_Hz > 4.5',
                'Operation_Mode == Maintenance'
            ]
        },
        'statistics': stats
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(label_summary, f, indent=4)
        
    logger.info("Proxy label construction completed successfully.")

if __name__ == "__main__":
    main()
