import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def construct_horizon_targets(df: pd.DataFrame, horizons_hours: list[int] = [6, 24, 72]) -> tuple[pd.DataFrame, dict]:
    """
    Construct future-deterioration proxy targets for given horizons (in hours).
    Policy:
      A future deterioration event is defined as:
      Within the lookahead window H:
        - Mean Temperature_C > 75.0°C (Sustained heat build-up)
        - OR Mean Vibration_Hz > 3.0 Hz (Sustained abnormal vibration)
        - OR Mean Error_Rate_% > 9.0% (Sustained operational error escalation)
    """
    logger.info("Constructing future-deterioration proxy targets using window averages...")
    df = df.copy()
    
    # Pre-sort to ensure correctness
    df = df.sort_values(by=['Machine_ID', 'datetime']).reset_index(drop=True)
    
    # Store targets and statistics
    label_stats = {}
    
    for H in horizons_hours:
        col_name = f"target_deterioration_{H}h"
        df[col_name] = 0
        
        # Process machine-by-machine
        for machine_id, group in df.groupby('Machine_ID'):
            indices = group.index.values
            times = group['datetime'].values
            temps = group['Temperature_C'].values
            vibs = group['Vibration_Hz'].values
            errs = group['Error_Rate_%'].values
            
            n = len(indices)
            targets = np.zeros(n, dtype=int)
            
            for i in range(n):
                t_current = times[i]
                t_max = t_current + np.timedelta64(H, 'h')
                
                # Find indices in lookahead window (current_time, current_time + H]
                idx_end = np.searchsorted(times, t_max, side='right')
                
                # Lookahead window slice
                if i + 1 < idx_end:
                    mean_t = np.mean(temps[i+1:idx_end])
                    mean_v = np.mean(vibs[i+1:idx_end])
                    mean_e = np.mean(errs[i+1:idx_end])
                    
                    if mean_t > 75.0 or mean_v > 3.0 or mean_e > 9.0:
                        targets[i] = 1
                        
            df.loc[indices, col_name] = targets
            
        # Record prevalence statistics
        pos_count = int(df[col_name].sum())
        total_count = len(df)
        prevalence = pos_count / total_count
        label_stats[f"{H}h_positive_count"] = pos_count
        label_stats[f"{H}h_prevalence"] = prevalence
        logger.info(f"Target {H}h prevalence: {prevalence:.2%} ({pos_count} positive cases)")
        
    return df, label_stats
