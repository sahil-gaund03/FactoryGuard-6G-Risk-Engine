import logging
import sys
import pandas as pd
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.factoryguard.paths import DATA_PREDICTIONS, DATA_PROCESSED
from src.factoryguard.logging_config import setup_logging
from src.factoryguard.config_loader import get_base_config
from src.factoryguard.scoring.risk_fusion import run_risk_fusion

def main():
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("factoryguard.scripts.generate_predictions")
    logger.info("Starting final risk prediction generation...")
    
    # 1. Load configs and paths
    base_config = get_base_config()
    
    input_path = DATA_PREDICTIONS / "supervised_predictions.parquet"
    if not input_path.exists():
        logger.error(f"Supervised predictions not found at {input_path}. Run train_supervised_models.py first!")
        sys.exit(1)
        
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} rows of predictions.")
    
    # 2. Run risk fusion engine
    df_fused = run_risk_fusion(df, epsilon=base_config['pipeline']['epsilon'])
    
    # 3. Save final output
    output_path = DATA_PROCESSED / "final_risk_predictions.parquet"
    logger.info(f"Saving final risk fused predictions to {output_path}...")
    df_fused.to_parquet(output_path, index=False)
    
    # Print key summary metrics
    high_count = (df_fused['risk_level'] == 'High').sum()
    med_count = (df_fused['risk_level'] == 'Medium').sum()
    logger.info(f"Risk levels compiled. High risk: {high_count} rows, Medium risk: {med_count} rows.")
    logger.info("Predictions generation and fusion completed successfully.")

if __name__ == "__main__":
    main()
