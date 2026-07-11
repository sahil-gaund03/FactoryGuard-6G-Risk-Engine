import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from src.factoryguard.paths import RAW_DATA_PATH, DATA_INTERIM, DATA_REPORTS
from src.factoryguard.data.validator import validate_schema, validate_ranges
from src.factoryguard.exceptions import DataValidationError

logger = logging.getLogger(__name__)

def load_raw_dataset(csv_path: Path = None) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    if csv_path is None:
        csv_path = RAW_DATA_PATH
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {csv_path}")
        
    logger.info(f"Ingesting raw CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    return df

def parse_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Combine Date/Timestamp into a single datetime index, sort by Machine_ID and datetime."""
    logger.info("Combining Date and Timestamp, sorting by Machine_ID and datetime...")
    
    # Store the original row index as source_row_id
    df['source_row_id'] = df.index.astype('int64')
    
    # Day-first parsing as specified in the TRD
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], dayfirst=True)
    
    # Sort
    df = df.sort_values(by=['Machine_ID', 'datetime']).reset_index(drop=True)
    return df

def handle_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Handle exact duplicates and duplicate Machine_ID + datetime combinations.
    Policy:
      1. Remove exact duplicate rows (if any).
      2. For duplicate Machine_ID + datetime combinations:
         Aggregate numeric values by median, and retain the first categorical value.
    """
    stats = {}
    
    # 1. Exact duplicates
    exact_dups = df.duplicated()
    stats['exact_duplicates_count'] = int(exact_dups.sum())
    if stats['exact_duplicates_count'] > 0:
        logger.info(f"Removing {stats['exact_duplicates_count']} exact duplicate rows.")
        df = df[~exact_dups].reset_index(drop=True)
        
    # 2. Duplicate Machine_ID + datetime
    dup_keys = df.duplicated(subset=['Machine_ID', 'datetime'], keep=False)
    stats['duplicate_machine_timestamps_count'] = int(df.duplicated(subset=['Machine_ID', 'datetime']).sum())
    
    if stats['duplicate_machine_timestamps_count'] > 0:
        logger.info(f"Handling {stats['duplicate_machine_timestamps_count']} duplicate machine timestamp combinations...")
        
        # Save the audit table for duplicate timestamps before merging
        dup_df = df[dup_keys].copy()
        dup_df.to_csv(DATA_REPORTS / "duplicate_machine_timestamps.csv", index=False)
        
        # Group by and aggregate
        # Numerics -> median. Categoricals -> first.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Keep source_row_id as min or first
        if 'source_row_id' in numeric_cols:
            numeric_cols.remove('source_row_id')
            
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if 'datetime' in non_numeric_cols:
            non_numeric_cols.remove('datetime')
        if 'Machine_ID' in non_numeric_cols:
            non_numeric_cols.remove('Machine_ID')
            
        agg_dict = {}
        for col in numeric_cols:
            agg_dict[col] = 'median'
        for col in non_numeric_cols:
            agg_dict[col] = 'first'
        agg_dict['source_row_id'] = 'first'
        
        df = df.groupby(['Machine_ID', 'datetime'], as_index=False).agg(agg_dict)
        logger.info(f"Deduplicated df shape: {df.shape}")
        
    return df, stats

def analyze_machine_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze observation coverage and date ranges per Machine_ID."""
    coverage = df.groupby('Machine_ID').agg(
        observations=('datetime', 'count'),
        min_date=('datetime', 'min'),
        max_date=('datetime', 'max')
    ).reset_index()
    coverage.to_csv(DATA_REPORTS / "machine_coverage.csv", index=False)
    return coverage

def analyze_time_gaps(df: pd.DataFrame) -> dict:
    """Analyze irregular time gaps between consecutive observations per Machine_ID."""
    gaps = []
    
    # Calculate time diff in minutes
    df['time_gap_mins'] = df.groupby('Machine_ID')['datetime'].diff().dt.total_seconds() / 60.0
    
    # Remove nulls (first row of each machine)
    valid_gaps = df['time_gap_mins'].dropna()
    
    gap_summary = {
        'mean_gap_mins': float(valid_gaps.mean()) if not valid_gaps.empty else 0.0,
        'median_gap_mins': float(valid_gaps.median()) if not valid_gaps.empty else 0.0,
        'min_gap_mins': float(valid_gaps.min()) if not valid_gaps.empty else 0.0,
        'max_gap_mins': float(valid_gaps.max()) if not valid_gaps.empty else 0.0,
        'std_gap_mins': float(valid_gaps.std()) if not valid_gaps.empty else 0.0,
        'irregular_gaps_count': int((valid_gaps > 1.1).sum())  # Gap larger than 1.1 minutes is irregular
    }
    
    # Save a detailed summary of gaps per machine
    machine_gaps = df.groupby('Machine_ID')['time_gap_mins'].agg(
        mean_gap='mean',
        median_gap='median',
        max_gap='max',
        std_gap='std'
    ).reset_index()
    machine_gaps.to_csv(DATA_REPORTS / "time_gap_summary.csv", index=False)
    
    # Clean up time_gap_mins column so it doesn't pollute the saved parquet
    df.drop(columns=['time_gap_mins'], inplace=True)
    
    return gap_summary

def run_ingestion_pipeline(csv_path: Path = None) -> pd.DataFrame:
    """Execute end-to-end data ingestion pipeline and save results."""
    # Create directories just in case
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    DATA_REPORTS.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    df = load_raw_dataset(csv_path)
    raw_shape = df.shape
    
    # 2. Validate schema
    validate_schema(df)
    
    # 3. Check data ranges
    range_anomalies = validate_ranges(df)
    
    # 4. Parse and sort chronologically
    df = parse_and_sort(df)
    
    # 5. Handle duplicates
    df, dup_stats = handle_duplicates(df)
    
    # 6. Analyze coverage
    coverage_df = analyze_machine_coverage(df)
    
    # 7. Analyze time gaps
    gap_stats = analyze_time_gaps(df)
    
    # Compile quality report
    quality_report = {
        'raw_rows': raw_shape[0],
        'raw_cols': raw_shape[1],
        'processed_rows': df.shape[0],
        'processed_cols': df.shape[1],
        'missing_values': df.isnull().sum().to_dict(),
        'range_anomalies': range_anomalies,
        **dup_stats,
        **gap_stats
    }
    
    # Save quality report as JSON
    with open(DATA_REPORTS / "data_quality_summary.json", 'w', encoding='utf-8') as f:
        json.dump(quality_report, f, indent=4, default=str)
        
    # 8. Save validated interim Parquet
    parquet_path = DATA_INTERIM / "validated_data.parquet"
    logger.info(f"Saving validated data to Parquet format: {parquet_path}")
    df.to_parquet(parquet_path, index=False)
    
    logger.info("Data pipeline completed successfully.")
    return df
