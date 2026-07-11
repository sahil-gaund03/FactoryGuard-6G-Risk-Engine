import logging
import numpy as np
import pandas as pd
from src.factoryguard.exceptions import ModelingError

logger = logging.getLogger(__name__)

def run_risk_fusion(df: pd.DataFrame, epsilon: float = 1e-6) -> pd.DataFrame:
    """
    Fuse Track A (anomaly score) and Track B (supervised proxy probability) with
    persistence and trend indicators to compute final risk outputs.
    """
    logger.info("Starting risk fusion calculations...")
    df = df.copy()
    
    # Ensure chronological order per machine
    df = df.sort_values(by=['Machine_ID', 'datetime']).reset_index(drop=True)
    
    # 1. Compute Fused Risk Score (0-100)
    # Track A weight: 0.4
    # Track B weight: 0.4
    # Instability trend weight: 0.2
    df['fused_risk_score'] = (
        0.4 * df['anomaly_score_normalized'] +
        0.4 * (df['proxy_maintenance_risk_prob'] * 100.0) +
        0.2 * (df['recent_deviation_ratio_5'] * 100.0)
    )
    df['fused_risk_score'] = np.clip(df['fused_risk_score'], 0.0, 100.0)
    
    # 2. Risk Level Mapping
    # Low: < 40, Medium: 40 to < 75, High: >= 75
    df['risk_level'] = 'Low'
    df.loc[df['fused_risk_score'] >= 40.0, 'risk_level'] = 'Medium'
    df.loc[df['fused_risk_score'] >= 75.0, 'risk_level'] = 'High'
    
    # 3. Persistence count: consecutive Medium or High risk observations
    df['is_warn_or_high'] = (df['risk_level'].isin(['Medium', 'High'])).astype(int)
    # Calculate consecutive counts
    df['risk_persistence_count'] = df.groupby('Machine_ID')['is_warn_or_high'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
    )
    df.drop(columns=['is_warn_or_high'], inplace=True)
    
    # 4. Trend Slope (simple difference over lag 1 and lag 3 of fused score)
    df['fused_score_lag_1'] = df.groupby('Machine_ID')['fused_risk_score'].shift(1).bfill()
    df['fused_score_lag_3'] = df.groupby('Machine_ID')['fused_risk_score'].shift(3).bfill()
    df['risk_trend_slope'] = df['fused_risk_score'] - df['fused_score_lag_1']
    df['risk_trend_slope_3'] = df['fused_risk_score'] - df['fused_score_lag_3']
    df.drop(columns=['fused_score_lag_1', 'fused_score_lag_3'], inplace=True)
    
    # 5. Determine Escalation Category and Evidence Strength
    df['escalation_category'] = 'Stable Normal'
    df['evidence_strength'] = 'Low'
    df['recommended_action'] = 'Continue monitoring'
    df['maintenance_priority'] = df['fused_risk_score']
    
    # Populate row-by-row or using masks
    # High evidence: Anomaly score >= 90 AND Supervised prob >= 0.15 (both agree)
    df.loc[(df['anomaly_score_normalized'] >= 90.0) & (df['proxy_maintenance_risk_prob'] >= 0.15), 'evidence_strength'] = 'High'
    # Medium evidence: Only one is elevated
    df.loc[((df['anomaly_score_normalized'] >= 90.0) | (df['proxy_maintenance_risk_prob'] >= 0.15)) & (df['evidence_strength'] != 'High'), 'evidence_strength'] = 'Medium'
    
    # State assignment logic
    # stable normal
    df.loc[df['risk_level'] == 'Low', 'escalation_category'] = 'Stable Normal'
    # Watch: Low risk but rising trend
    df.loc[(df['risk_level'] == 'Low') & (df['risk_trend_slope'] > 5.0), 'escalation_category'] = 'Watch'
    # Slowly Escalating
    df.loc[(df['risk_level'] == 'Medium') & (df['risk_trend_slope_3'] <= 15.0), 'escalation_category'] = 'Slowly Escalating'
    # Rapidly Escalating
    df.loc[(df['risk_level'] == 'Medium') & (df['risk_trend_slope_3'] > 15.0), 'escalation_category'] = 'Rapidly Escalating'
    df.loc[(df['risk_level'] == 'High') & (df['risk_trend_slope_3'] > 15.0), 'escalation_category'] = 'Rapidly Escalating'
    # Persistent High Risk
    df.loc[(df['risk_level'] == 'High') & (df['risk_persistence_count'] >= 3), 'escalation_category'] = 'Persistent High Risk'
    # Transient High
    df.loc[(df['risk_level'] == 'High') & (df['risk_persistence_count'] < 3) & (df['risk_trend_slope'] > 20.0), 'escalation_category'] = 'Transient High'
    
    # Recovering: risk is falling
    df.loc[(df['risk_level'].isin(['Medium', 'High'])) & (df['risk_trend_slope_3'] < -10.0), 'escalation_category'] = 'Recovering'
    
    # Data Quality Concern
    df.loc[(df['Network_Latency_ms'] > 45.0) | (df['Packet_Loss_%'] > 4.5), 'escalation_category'] = 'Data Quality Concern'
    
    # Recommended Action
    df.loc[df['risk_level'] == 'Low', 'recommended_action'] = 'Maintain standard monitoring schedule.'
    df.loc[df['risk_level'] == 'Medium', 'recommended_action'] = 'Schedule sensor validation and inspect mechanical mounts during next window.'
    df.loc[df['risk_level'] == 'High', 'recommended_action'] = 'Immediate physical inspection: check lubrication, bearings, and communication connections.'
    df.loc[df['escalation_category'] == 'Data Quality Concern', 'recommended_action'] = 'Validate network latency and sensor connections before physical maintenance.'
    
    # Maintenance Priority Score: premium for persistence and defect rates
    df['maintenance_priority'] = df['fused_risk_score'] + (df['risk_persistence_count'] * 5.0)
    # Cap at 100
    df['maintenance_priority'] = np.clip(df['maintenance_priority'], 0.0, 100.0)
    
    logger.info("Risk fusion completed successfully.")
    return df
