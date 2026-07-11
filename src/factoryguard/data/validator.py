import pandas as pd
import numpy as np
import logging
from src.factoryguard.exceptions import DataValidationError

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    'Date', 'Timestamp', 'Machine_ID', 'Operation_Mode', 
    'Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW', 
    'Network_Latency_ms', 'Packet_Loss_%', 'Quality_Control_Defect_Rate_%', 
    'Production_Speed_units_per_hr', 'Predictive_Maintenance_Score', 
    'Error_Rate_%', 'Efficiency_Status'
]

def validate_schema(df: pd.DataFrame):
    """Validate that the dataset has the expected columns and types."""
    logger.info("Validating dataset columns...")
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing expected columns: {missing_cols}")
    
    # Check numeric types
    numeric_cols = [
        'Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW', 
        'Network_Latency_ms', 'Packet_Loss_%', 'Quality_Control_Defect_Rate_%', 
        'Production_Speed_units_per_hr', 'Predictive_Maintenance_Score', 'Error_Rate_%'
    ]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column {col} is not numeric in raw data. Trying to coerce...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                logger.error(f"Found non-numeric values in numeric column {col}")

def validate_ranges(df: pd.DataFrame) -> dict:
    """Validate data range constraints and return details of any anomalies or failures."""
    anomalies = {}
    
    # Check percentage ranges [0, 100] or [0, 10] etc.
    # Defect rate, error rate, packet loss
    pct_cols = {
        'Packet_Loss_%': 100.0, 
        'Quality_Control_Defect_Rate_%': 100.0, 
        'Error_Rate_%': 100.0
    }
    for col, max_val in pct_cols.items():
        out_of_bounds = df[(df[col] < 0) | (df[col] > max_val)]
        if not out_of_bounds.empty:
            anomalies[f"{col}_range_out"] = len(out_of_bounds)
            logger.warning(f"{col} has {len(out_of_bounds)} rows out of standard percentage bounds [0, {max_val}]")
            
    # Check physical column ranges (no negative values)
    physical_cols = ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW', 'Production_Speed_units_per_hr']
    for col in physical_cols:
        neg_vals = df[df[col] < 0]
        if not neg_vals.empty:
            anomalies[f"{col}_negative"] = len(neg_vals)
            logger.warning(f"{col} has {len(neg_vals)} negative values.")
            
    # Check Predictive Maintenance Score [0, 1]
    pdm_out = df[(df['Predictive_Maintenance_Score'] < 0.0) | (df['Predictive_Maintenance_Score'] > 1.0)]
    if not pdm_out.empty:
        anomalies['Predictive_Maintenance_Score_out'] = len(pdm_out)
        logger.warning(f"Predictive_Maintenance_Score has {len(pdm_out)} rows out of [0, 1] bounds.")
        
    # Check categories
    valid_modes = {'Active', 'Idle', 'Maintenance'}
    invalid_modes = df[~df['Operation_Mode'].isin(valid_modes)]
    if not invalid_modes.empty:
        anomalies['Operation_Mode_invalid'] = len(invalid_modes)
        logger.warning(f"Operation_Mode has {len(invalid_modes)} invalid values: {invalid_modes['Operation_Mode'].unique()}")
        
    valid_effs = {'Low', 'Medium', 'High'}
    invalid_effs = df[~df['Efficiency_Status'].isin(valid_effs)]
    if not invalid_effs.empty:
        anomalies['Efficiency_Status_invalid'] = len(invalid_effs)
        logger.warning(f"Efficiency_Status has {len(invalid_effs)} invalid values: {invalid_effs['Efficiency_Status'].unique()}")

    return anomalies
