import logging
import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.factoryguard.logging_config import setup_logging
from src.factoryguard.data.loader import run_ingestion_pipeline

def main():
    # Setup logging to console and a local log file
    log_file = Path("logs/data_pipeline.log")
    setup_logging(level=logging.INFO, log_file=log_file)
    
    logger = logging.getLogger("factoryguard.scripts.run_data_pipeline")
    logger.info("Starting ingestion pipeline...")
    
    try:
        df = run_ingestion_pipeline()
        logger.info(f"Pipeline finished. Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
        logger.info("Interim Parquet and quality reports saved successfully.")
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
