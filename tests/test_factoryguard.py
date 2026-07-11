import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import modules to test
from src.factoryguard.data.validator import validate_schema, validate_ranges
from src.factoryguard.features.feature_pipeline import LeakageSafeFeaturePipeline
from src.factoryguard.models.isolation_forest import FactoryGuardIsolationForest
from src.factoryguard.models.calibration import ProbabilityCalibrator
from src.factoryguard.models.stacking import ChronologicalStackingClassifier
from src.factoryguard.scoring.risk_fusion import run_risk_fusion
from src.factoryguard.explainability.recommendation_engine import generate_alert_narrative

@pytest.fixture
def mock_raw_data():
    """Generate a clean mock dataframe representing 5 observations."""
    base_time = datetime(2025, 1, 1, 12, 0, 0)
    data = {
        'Date': [(base_time + timedelta(minutes=i)).strftime('%d/%m/%Y') for i in range(5)],
        'Timestamp': [(base_time + timedelta(minutes=i)).strftime('%H:%M:%S') for i in range(5)],
        'Machine_ID': [1.0] * 5,
        'Operation_Mode': ['Active', 'Active', 'Idle', 'Maintenance', 'Active'],
        'Temperature_C': [55.0, 56.5, 35.0, 40.0, 88.0],
        'Vibration_Hz': [2.5, 2.7, 0.5, 0.8, 4.9],
        'Power_Consumption_kW': [5.0, 5.2, 1.8, 2.0, 9.8],
        'Network_Latency_ms': [15.0, 16.0, 10.0, 12.0, 48.0],
        'Packet_Loss_%': [1.0, 1.2, 0.0, 0.5, 4.8],
        'Quality_Control_Defect_Rate_%': [4.0, 4.2, 0.0, 0.0, 9.8],
        'Production_Speed_units_per_hr': [300.0, 310.0, 0.0, 0.0, 60.0],
        'Predictive_Maintenance_Score': [0.5, 0.51, 0.49, 0.48, 0.05],
        'Error_Rate_%': [5.0, 5.2, 0.0, 0.0, 14.8],
        'Efficiency_Status': ['Low', 'Low', 'Medium', 'Medium', 'Low']
    }
    return pd.DataFrame(data)

def test_data_validation(mock_raw_data):
    """Test schema and range validation logic."""
    df = mock_raw_data.copy()
    validate_schema(df)
    anoms = validate_ranges(df)
    assert len(anoms) == 0

def test_feature_pipeline(mock_raw_data):
    """Test feature extraction and baseline fitting."""
    df = mock_raw_data.copy()
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], dayfirst=True)
    
    pipeline = LeakageSafeFeaturePipeline(epsilon=1e-6)
    pipeline.fit(df)
    df_transformed = pipeline.transform(df)
    
    # Assert features were generated
    assert 'Temperature_C_dev_mach' in df_transformed.columns
    assert 'vibration_to_power_ratio' in df_transformed.columns
    assert 'network_instability_index' in df_transformed.columns

def test_anomaly_forest(mock_raw_data):
    """Test Isolation Forest fit and predict scores."""
    df = mock_raw_data.copy()
    # Fill features
    features = df[['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW']]
    
    model = FactoryGuardIsolationForest(n_estimators=10, random_state=42)
    model.fit(features)
    
    raw = model.predict_anomaly_score_raw(features)
    norm = model.predict_anomaly_score_normalized(features)
    
    assert len(raw) == 5
    assert len(norm) == 5
    assert np.all(norm >= 0.0) and np.all(norm <= 100.0)

def test_probability_calibrator():
    """Test probability calibrator fit and calibrate."""
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1])
    
    calibrator = ProbabilityCalibrator(method='sigmoid')
    calibrator.fit(y_prob, y_true)
    cal_probs = calibrator.calibrate(y_prob)
    
    assert len(cal_probs) == 4
    assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)

def test_risk_fusion(mock_raw_data):
    """Test risk fusion logic."""
    df = mock_raw_data.copy()
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], dayfirst=True)
    df['anomaly_score_normalized'] = [10.0, 12.0, 5.0, 8.0, 95.0]
    df['proxy_maintenance_risk_prob'] = [0.05, 0.06, 0.02, 0.03, 0.98]
    df['recent_deviation_ratio_5'] = [0.0, 0.0, 0.0, 0.0, 0.8]
    
    df_fused = run_risk_fusion(df)
    
    assert 'fused_risk_score' in df_fused.columns
    assert 'risk_level' in df_fused.columns
    assert df_fused.loc[4, 'risk_level'] == 'High'

def test_explainability(mock_raw_data):
    """Test alert narrative generator."""
    df = mock_raw_data.copy()
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], dayfirst=True)
    df['anomaly_score_normalized'] = 95.0
    df['proxy_maintenance_risk_prob'] = 0.98
    df['recent_deviation_ratio_5'] = 0.8
    df['escalation_category'] = 'Persistent High Risk'
    df['evidence_strength'] = 'High'
    df['recommended_action'] = 'Urgent Inspection'
    
    df_fused = run_risk_fusion(df)
    latest_row = df_fused.iloc[-1]
    
    narrative = generate_alert_narrative(latest_row)
    
    assert narrative['machine_id'] == '1'
    assert narrative['risk_level'] == 'High'
    assert 'Validate network latency' in narrative['recommendation']
