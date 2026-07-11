import logging
import pandas as pd
import numpy as np
from pathlib import Path
from src.factoryguard.exceptions import FeatureEngineeringError

logger = logging.getLogger(__name__)

# List of primary raw numeric sensors
SENSOR_COLS = [
    'Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW', 
    'Network_Latency_ms', 'Packet_Loss_%', 'Quality_Control_Defect_Rate_%', 
    'Production_Speed_units_per_hr', 'Error_Rate_%'
]

class LeakageSafeFeaturePipeline:
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = float(epsilon)
        # Dictionary to store baseline statistics learned from training data only
        self.machine_baselines = {}
        self.machine_mode_baselines = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        """Learn robust baselines (median, IQR, MAD) from training data."""
        logger.info("Fitting feature pipeline baselines on training data...")
        
        # 1. Machine-specific baselines
        for machine_id, group in df.groupby('Machine_ID'):
            self.machine_baselines[machine_id] = {}
            for col in SENSOR_COLS:
                values = group[col].dropna()
                if len(values) > 0:
                    med = float(values.median())
                    iqr = float(values.quantile(0.75) - values.quantile(0.25))
                    mad = float((values - med).abs().median())
                    self.machine_baselines[machine_id][col] = {
                        'median': med,
                        'iqr': max(iqr, self.epsilon),
                        'mad': max(mad, self.epsilon)
                    }
        
        # 2. Machine & Operation_Mode baselines
        for (machine_id, op_mode), group in df.groupby(['Machine_ID', 'Operation_Mode']):
            key = (machine_id, op_mode)
            self.machine_mode_baselines[key] = {}
            for col in SENSOR_COLS:
                values = group[col].dropna()
                if len(values) > 0:
                    med = float(values.median())
                    iqr = float(values.quantile(0.75) - values.quantile(0.25))
                    mad = float((values - med).abs().median())
                    self.machine_mode_baselines[key][col] = {
                        'median': med,
                        'iqr': max(iqr, self.epsilon),
                        'mad': max(mad, self.epsilon)
                    }
                    
        self.fitted = True
        logger.info("Baselines fitted successfully.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations chronologically."""
        if not self.fitted:
            raise FeatureEngineeringError("Pipeline must be fitted on training data before calling transform!")
            
        logger.info("Calculating engineered features...")
        df = df.copy()
        
        # Ensure df is sorted chronologically per machine
        df = df.sort_values(by=['Machine_ID', 'datetime']).reset_index(drop=True)
        
        # Dictionary to store new columns
        new_features = {}
        
        # A. Time features
        new_features['hour'] = df['datetime'].dt.hour
        new_features['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Time since previous observation per machine
        time_diff_sec = df.groupby('Machine_ID')['datetime'].diff().dt.total_seconds().fillna(60.0)
        new_features['time_diff_sec'] = time_diff_sec
        new_features['time_diff_hours'] = time_diff_sec / 3600.0
        
        # Time since last Maintenance-mode observation per machine
        is_maint = (df['Operation_Mode'] == 'Maintenance').astype(int)
        df['maint_timestamp'] = df['datetime'].where(is_maint == 1)
        last_maint_timestamp = df.groupby('Machine_ID')['maint_timestamp'].ffill()
        time_since_last_maint_hours = (df['datetime'] - last_maint_timestamp).dt.total_seconds() / 3600.0
        new_features['time_since_last_maint_hours'] = time_since_last_maint_hours.fillna(-1.0)
        df.drop(columns=['maint_timestamp'], inplace=True)
        
        # B. Robust normalizations and baseline features
        # Initialize numpy arrays for baselines
        baseline_arrs = {}
        for col in SENSOR_COLS:
            baseline_arrs[f"{col}_dev_mach"] = np.zeros(len(df))
            baseline_arrs[f"{col}_z_mach"] = np.zeros(len(df))
            baseline_arrs[f"{col}_dev_mach_mode"] = np.zeros(len(df))
            baseline_arrs[f"{col}_z_mach_mode"] = np.zeros(len(df))
            
        # Iterate over records and apply the baselines
        # For performance, we map machine-specific stats
        for machine_id, stats in self.machine_baselines.items():
            mask = (df['Machine_ID'] == machine_id).values
            if not np.any(mask):
                continue
            for col in SENSOR_COLS:
                if col in stats:
                    val = df.loc[mask, col].values
                    baseline_arrs[f"{col}_dev_mach"][mask] = val - stats[col]['median']
                    baseline_arrs[f"{col}_z_mach"][mask] = (val - stats[col]['median']) / stats[col]['mad']
                    
        # Apply machine-mode baselines
        for (machine_id, op_mode), stats in self.machine_mode_baselines.items():
            mask = ((df['Machine_ID'] == machine_id) & (df['Operation_Mode'] == op_mode)).values
            if not np.any(mask):
                continue
            for col in SENSOR_COLS:
                if col in stats:
                    val = df.loc[mask, col].values
                    baseline_arrs[f"{col}_dev_mach_mode"][mask] = val - stats[col]['median']
                    baseline_arrs[f"{col}_z_mach_mode"][mask] = (val - stats[col]['median']) / stats[col]['mad']

        # Add baseline features to dictionary
        for k, v in baseline_arrs.items():
            new_features[k] = v

        # C. Rolling & Lag & Trend features per machine
        windows = [3, 5, 10, 20]
        lags = [1, 2, 3, 5, 10]
        
        grouped = df.groupby('Machine_ID')
        
        for w in windows:
            # Rolling mean, std, min, max for primary sensors
            for col in ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW']:
                new_features[f"{col}_roll_mean_{w}"] = grouped[col].transform(lambda x: x.rolling(w, min_periods=1).mean())
                new_features[f"{col}_roll_std_{w}"] = grouped[col].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0.0))
                new_features[f"{col}_roll_min_{w}"] = grouped[col].transform(lambda x: x.rolling(w, min_periods=1).min())
                new_features[f"{col}_roll_max_{w}"] = grouped[col].transform(lambda x: x.rolling(w, min_periods=1).max())
                new_features[f"{col}_roll_range_{w}"] = new_features[f"{col}_roll_max_{w}"] - new_features[f"{col}_roll_min_{w}"]
                
        for l in lags:
            # Lags for primary sensors
            for col in ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW']:
                new_features[f"{col}_lag_{l}"] = grouped[col].shift(l).bfill()
                
        # Trend features
        for col in ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW']:
            # Absolute change over lag 1
            abs_change = df[col] - new_features[f"{col}_lag_1"]
            new_features[f"{col}_abs_change"] = abs_change
            # Percentage change
            new_features[f"{col}_pct_change"] = abs_change / (new_features[f"{col}_lag_1"].abs() + self.epsilon)
            # Acceleration: change in change
            temp_abs = pd.Series(abs_change, index=df.index)
            new_features[f"{col}_acceleration"] = abs_change - temp_abs.groupby(df['Machine_ID']).shift(1).fillna(0.0)
            # Short-window vs long-window difference
            new_features[f"{col}_short_long_diff"] = new_features[f"{col}_roll_mean_3"] - new_features[f"{col}_roll_mean_20"]
            
        # D. Domain interaction features
        # Vibration to power ratio
        new_features['vibration_to_power_ratio'] = df['Vibration_Hz'] / (df['Power_Consumption_kW'].abs() + self.epsilon)
        
        # Power per production unit
        new_features['power_per_production_unit'] = df['Power_Consumption_kW'] / (df['Production_Speed_units_per_hr'].abs() + self.epsilon)
        
        # Error rate per production unit
        new_features['error_rate_per_production_unit'] = df['Error_Rate_%'] / (df['Production_Speed_units_per_hr'].abs() + self.epsilon)
        
        # Defect rate per production unit
        new_features['defect_rate_per_production_unit'] = df['Quality_Control_Defect_Rate_%'] / (df['Production_Speed_units_per_hr'].abs() + self.epsilon)
        
        # Latency & Packet loss interaction (Network Instability Index)
        new_features['network_instability_index'] = df['Network_Latency_ms'] * df['Packet_Loss_%']
        
        # Vibration & Temperature interaction (Sensor Instability Index)
        # Using machine mode robust z-scores to capture extreme deviation interactions
        new_features['sensor_instability_index'] = np.abs(baseline_arrs['Temperature_C_z_mach_mode']) * np.abs(baseline_arrs['Vibration_Hz_z_mach_mode'])
        
        # Production quality stress index
        new_features['production_quality_stress_index'] = df['Error_Rate_%'] * df['Quality_Control_Defect_Rate_%']
        
        # Multi-signal deviation count: count sensors whose robust z-score exceeds 2.0 (abnormal)
        dev_cols = [f"{col}_z_mach_mode" for col in SENSOR_COLS]
        dev_df = pd.DataFrame({col: baseline_arrs[col] for col in dev_cols})
        new_features['multi_signal_deviation_count'] = (dev_df.abs() > 2.0).sum(axis=1)
        
        # E. Persistence features
        # Ratio of warnings in recent observations (last 5 observations)
        temp_multi = pd.Series(new_features['multi_signal_deviation_count'], index=df.index)
        new_features['recent_deviation_ratio_5'] = temp_multi.rolling(5, min_periods=1).mean()
        
        # Concatenate all new features at once
        new_features_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_features_df], axis=1)
        
        # Fill all final NA values safely
        df = df.fillna(0.0)
        return df
