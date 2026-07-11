import streamlit as st
import pandas as pd

def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render common sidebar filters and return the filtered DataFrame."""
    st.sidebar.header("Global Operational Filters")
    
    # 1. Reset Filters button
    if st.sidebar.button("Reset All Filters"):
        st.session_state.clear()
        st.rerun()

    # 2. Date and Time Range
    min_date = df['datetime'].min().date()
    max_date = df['datetime'].max().date()
    
    selected_dates = st.sidebar.date_input(
        "Observation Time Window",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # 3. Machine ID Selector
    all_machines = sorted(df['Machine_ID'].unique().astype(int))
    machine_options = ["All Fleet"] + [f"Machine {m}" for m in all_machines]
    selected_machine_str = st.sidebar.selectbox("Select Machine Unit", options=machine_options)
    
    # 4. Operation Mode Selector
    all_modes = df['Operation_Mode'].unique().tolist()
    selected_modes = st.sidebar.multiselect("Machine Operation Modes", options=all_modes, default=all_modes)
    
    # 5. Risk Level Selector
    all_risks = ['Low', 'Medium', 'High']
    selected_risks = st.sidebar.multiselect("Risk Alert Levels", options=all_risks, default=all_risks)
    
    # 6. Risk Threshold Slider
    threshold = st.sidebar.slider("Anomaly Warning Threshold (Percentile)", min_value=70, max_value=99, value=90)
    
    # Filter calculation
    filtered_df = df.copy()
    
    # Apply date filters
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_dt = pd.to_datetime(selected_dates[0])
        end_dt = pd.to_datetime(selected_dates[1]) + pd.Timedelta(days=1)
        filtered_df = filtered_df[(filtered_df['datetime'] >= start_dt) & (filtered_df['datetime'] < end_dt)]
        
    # Apply machine filter
    if selected_machine_str != "All Fleet":
        m_id = int(selected_machine_str.split(" ")[1])
        filtered_df = filtered_df[filtered_df['Machine_ID'] == m_id]
        
    # Apply operation mode filter
    if selected_modes:
        filtered_df = filtered_df[filtered_df['Operation_Mode'].isin(selected_modes)]
        
    # Apply risk level filter
    if selected_risks:
        filtered_df = filtered_df[filtered_df['risk_level'].isin(selected_risks)]
        
    return filtered_df
