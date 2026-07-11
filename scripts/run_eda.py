import logging
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.factoryguard.paths import DATA_INTERIM, REPORTS_DIR
from src.factoryguard.logging_config import setup_logging

def main():
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("factoryguard.scripts.run_eda")
    logger.info("Starting Exploratory Data Analysis...")
    
    parquet_path = DATA_INTERIM / "validated_data.parquet"
    if not parquet_path.exists():
        logger.error(f"Validated interim Parquet file not found at {parquet_path}. Run data pipeline first!")
        sys.exit(1)
        
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {df.shape[0]} rows for EDA.")
    
    # 1. Descriptive stats for numeric columns
    numeric_cols = [
        'Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW', 
        'Network_Latency_ms', 'Packet_Loss_%', 'Quality_Control_Defect_Rate_%', 
        'Production_Speed_units_per_hr', 'Predictive_Maintenance_Score', 'Error_Rate_%'
    ]
    desc_stats = df[numeric_cols].describe().transpose()
    desc_stats.to_csv(REPORTS_DIR / "tables" / "sensor_descriptive_stats.csv")
    
    # 2. Correlation Matrix
    corr_matrix = df[numeric_cols].corr()
    corr_matrix.to_csv(REPORTS_DIR / "tables" / "sensor_correlation_matrix.csv")
    
    # 3. Mode breakdown
    mode_counts = df['Operation_Mode'].value_counts(dropna=False).to_frame().reset_index()
    mode_counts.to_csv(REPORTS_DIR / "tables" / "operation_mode_counts.csv", index=False)
    
    # 4. Efficiency breakdown
    eff_counts = df['Efficiency_Status'].value_counts(dropna=False).to_frame().reset_index()
    eff_counts.to_csv(REPORTS_DIR / "tables" / "efficiency_status_counts.csv", index=False)
    
    # 5. Machine breakdown summary
    machine_summary = df.groupby('Machine_ID').agg(
        record_count=('source_row_id', 'count'),
        mean_temp=('Temperature_C', 'mean'),
        std_temp=('Temperature_C', 'std'),
        mean_vib=('Vibration_Hz', 'mean'),
        std_vib=('Vibration_Hz', 'std'),
        mean_power=('Power_Consumption_kW', 'mean'),
        std_power=('Power_Consumption_kW', 'std'),
        mean_speed=('Production_Speed_units_per_hr', 'mean'),
        std_speed=('Production_Speed_units_per_hr', 'std')
    ).reset_index()
    machine_summary.to_csv(REPORTS_DIR / "tables" / "machine_summary_stats.csv", index=False)
    
    # 6. Sensor readings by operation mode
    mode_sensor_summary = df.groupby('Operation_Mode')[numeric_cols].mean().transpose()
    mode_sensor_summary.to_csv(REPORTS_DIR / "tables" / "mode_sensor_averages.csv")
    
    # Save a JSON file with metadata summaries for simple rendering in the dashboard
    eda_summary = {
        'total_observations': len(df),
        'machines_count': int(df['Machine_ID'].nunique()),
        'date_min': str(df['datetime'].min()),
        'date_max': str(df['datetime'].max()),
        'active_records_pct': float((df['Operation_Mode'] == 'Active').mean() * 100),
        'idle_records_pct': float((df['Operation_Mode'] == 'Idle').mean() * 100),
        'maintenance_records_pct': float((df['Operation_Mode'] == 'Maintenance').mean() * 100),
        'low_efficiency_pct': float((df['Efficiency_Status'] == 'Low').mean() * 100),
        'medium_efficiency_pct': float((df['Efficiency_Status'] == 'Medium').mean() * 100),
        'high_efficiency_pct': float((df['Efficiency_Status'] == 'High').mean() * 100)
    }
    
    with open(REPORTS_DIR / "tables" / "eda_summary.json", 'w', encoding='utf-8') as f:
        json.dump(eda_summary, f, indent=4)
        
    logger.info("Exploratory Data Analysis completed. Output tables generated under reports/tables.")

if __name__ == "__main__":
    main()
