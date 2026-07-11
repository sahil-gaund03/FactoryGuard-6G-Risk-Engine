import logging
import pandas as pd
from datetime import timedelta

logger = logging.getLogger(__name__)

# List of sensors to rank for contributions
SENSORS = {
    'Temperature_C': 'Operating Temperature',
    'Vibration_Hz': 'Vibration Frequency',
    'Power_Consumption_kW': 'Power Consumption',
    'Network_Latency_ms': 'Network Latency',
    'Packet_Loss_%': 'Network Packet Loss',
    'Quality_Control_Defect_Rate_%': 'Quality Control Defect Rate',
    'Production_Speed_units_per_hr': 'Production Speed',
    'Error_Rate_%': 'Operational Error Rate'
}

def generate_alert_narrative(row: pd.Series) -> dict:
    """
    Generate a structured, deterministic narrative answering the 11 PRD questions
    for a Medium or High risk observation.
    """
    # 1. Affected machine
    machine = str(int(row['Machine_ID']))
    
    # 2. Begin time
    # Estimate when the anomaly began based on the persistence count
    persistence = int(row['risk_persistence_count'])
    # Assume approx 50 mins per observation gap on average
    mins_back = max(1, persistence) * 50
    begin_time = row['datetime'] - timedelta(minutes=mins_back)
    
    # 3. Escalation category
    escalation = row['escalation_category']
    
    # 4. Anomaly score
    anomaly_score = float(row['anomaly_score_normalized'])
    
    # 5. Proxy probability
    proxy_prob = float(row['proxy_maintenance_risk_prob'] * 100.0)
    
    # 6 & 7. Top contributing signals and deviations
    # We rank sensors by their absolute robust z-score deviation
    deviations = []
    for col, display_name in SENSORS.items():
        z_col = f"{col}_z_mach_mode"
        if z_col in row:
            z_val = float(row[z_col])
            # Determine direction
            direction = "above" if z_val >= 0 else "below"
            deviations.append({
                'sensor': display_name,
                'col': col,
                'z_score': z_val,
                'abs_z_score': abs(z_val),
                'description': f"{display_name} is {abs(z_val):.2f} standard deviations {direction} normal machine-mode behavior"
            })
            
    # Sort by absolute z-score descending
    deviations = sorted(deviations, key=lambda x: x['abs_z_score'], reverse=True)
    top_3 = deviations[:3]
    top_contributors = [d['sensor'] for d in top_3]
    contributor_narratives = [d['description'] for d in top_3]
    
    # 8. Impact areas
    impacts = []
    if abs(float(row.get('Production_Speed_units_per_hr_z_mach_mode', 0))) > 2.0:
        impacts.append("Production Speed Degradation")
    if float(row.get('Quality_Control_Defect_Rate_%_z_mach_mode', 0)) > 2.0:
        impacts.append("Product Quality Defect Escalation")
    if float(row.get('Error_Rate_%_z_mach_mode', 0)) > 2.0:
        impacts.append("Operational Process Instability")
    if float(row.get('Network_Latency_ms_z_mach_mode', 0)) > 2.0 or float(row.get('Packet_Loss_%_z_mach_mode', 0)) > 2.0:
        impacts.append("6G Network Telemetry Instability")
        
    impact_text = ", ".join(impacts) if impacts else "None immediate (baseline variance)"
    
    # 9. Recommendation
    recommendation = row['recommended_action']
    
    # 10. Evidence Strength
    evidence = row['evidence_strength']
    
    # 11. Model Limitations (Disclaimers)
    disclaimer = (
        "The model cannot confirm mechanical component failure, physical breakdown events, "
        "or downtime duration. Recommendations are decision-support guidelines based on statistical "
        "deviations from baseline data. Physical engineering checks are required to diagnose the root cause."
    )
    
    # Construct final plain-language response
    narrative_html = f"""
    <strong>Affected Machine:</strong> Machine {machine}<br/>
    <strong>Alert Level:</strong> {row['risk_level']} Risk ({escalation})<br/>
    <strong>Evidence Strength:</strong> {evidence} (Anomaly score: {anomaly_score:.1f}/100, Future deterioration proxy probability: {proxy_prob:.1f}%)<br/>
    <strong>Anomaly Start Time:</strong> Estimated {begin_time.strftime('%Y-%m-%d %H:%M')}<br/>
    <strong>Top Deviating Signals:</strong>
    <ul>
        {"".join([f"<li>{desc}</li>" for desc in contributor_narratives])}
    </ul>
    <strong>Business & Operational Impact:</strong> {impact_text}<br/>
    <strong>Recommended Action:</strong> {recommendation}<br/>
    <strong>Model Disclaimer:</strong> {disclaimer}
    """
    
    return {
        'machine_id': machine,
        'risk_level': row['risk_level'],
        'escalation': escalation,
        'anomaly_score': anomaly_score,
        'proxy_prob': proxy_prob,
        'begin_time': begin_time,
        'top_contributors': top_contributors,
        'impacts': impacts,
        'recommendation': recommendation,
        'evidence': evidence,
        'disclaimer': disclaimer,
        'html': narrative_html
    }
