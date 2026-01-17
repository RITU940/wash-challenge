import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from digital_twin_sim import PIPES, get_diurnal_pattern, calculate_pressure_drop, DigitalTwinAI, generate_normal_data

# Page Config
st.set_page_config(page_title="WASH AI Digital Twin Dashboard", layout="wide", page_icon="💧")

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #e0e0e0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #00e5ff;
    }
    div[data-testid="stMetricLabel"] {
        color: #b0bec5;
    }
    .stButton>button {
        background-color: #00e5ff;
        color: #000;
        border-radius: 20px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #00b8cc;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. Cache AI Model (Train once per session)
# ==========================================
@st.cache_resource
def get_trained_model():
    ai = DigitalTwinAI()
    train_time_steps = np.arange(3 * 24 * 60)
    normal_data = generate_normal_data(PIPES, train_time_steps)
    ai.train(normal_data)
    return ai

dt_ai = get_trained_model()

# ==========================================
# 2. Simulation Logic (Per Step)
# ==========================================
def simulate_step(t_current, pipes, anomalies):
    """Generate synthetic data for a single timestep."""
    demand_pattern = get_diurnal_pattern(t_current)
    step_data = []
    
    for pipe in pipes:
        noise_flow = np.random.normal(0, pipe['base_flow'] * 0.05)
        expected_flow = pipe['base_flow'] * demand_pattern + noise_flow
        
        source_pressure = 50
        p_drop = calculate_pressure_drop(expected_flow, pipe['len'], pipe['dia'])
        noise_pressure = np.random.normal(0, 0.5)
        expected_pressure = source_pressure - p_drop + noise_pressure
        
        label = 0
        if pipe['id'] == 'P02' and anomalies['leak']:
            expected_pressure *= 0.85
            expected_flow *= 1.1
            label = 1
        if pipe['id'] == 'P05' and anomalies['burst']:
            expected_pressure *= 0.4
            expected_flow *= 2.5
            label = 2
        if pipe['id'] == 'P09' and anomalies['abnormal']:
            expected_flow += 25
            expected_pressure -= 2
            label = 3
            
        step_data.append({
            'time_min': t_current,
            'pipe_id': pipe['id'],
            'flow': expected_flow,
            'pressure': expected_pressure,
            'label': label
        })
        
    df = pd.DataFrame(step_data)
    df['anomaly_score'] = dt_ai.predict(df)
    return df

# ==========================================
# 3. State Initialization
# ==========================================
def create_initial_data():
    """Generate fresh historical data for initialization."""
    all_data = []
    for t in range(120):  # 2 hours of history
        step_df = simulate_step(t, PIPES, {'leak': False, 'burst': False, 'abnormal': False})
        all_data.append(step_df)
    return pd.concat(all_data, ignore_index=True)

if 'simulation_state' not in st.session_state:
    initial_data = create_initial_data()
    st.session_state.simulation_state = {
        'time_min': 120,
        'is_running': False,
        'data_buffer': initial_data,
        'anomalies': {'leak': False, 'burst': False, 'abnormal': False}
    }

# ==========================================
# 4. Sidebar Controls
# ==========================================
with st.sidebar:
    st.title("🎛️ Controls")
    
    st.markdown("### Simulation Status")
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ PLAY", type="primary")
    with col2:
        stop_btn = st.button("⏸ PAUSE")
        
    if start_btn:
        st.session_state.simulation_state['is_running'] = True
    if stop_btn:
        st.session_state.simulation_state['is_running'] = False
    
    st.markdown("### 💾 Data Export")
    current_buffer = st.session_state.simulation_state['data_buffer']
    if not current_buffer.empty:
        csv_data = current_buffer.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="digital_twin_simulation_log.csv",
            mime="text/csv"
        )

    st.markdown("---")
    speed = st.slider("Simulation Speed", 1, 60, 10, help="Minutes per step")
    
    st.markdown("### Anomaly Injection")
    st.info("Toggle to inject anomalies into the simulation.")
    
    leak_toggle = st.toggle("Inject Minor Leak (P02)", value=st.session_state.simulation_state['anomalies']['leak'])
    burst_toggle = st.toggle("Inject MAJOR BURST (P05)", value=st.session_state.simulation_state['anomalies']['burst'])
    usage_toggle = st.toggle("Abnormal Usage (P09)", value=st.session_state.simulation_state['anomalies']['abnormal'])
    
    st.session_state.simulation_state['anomalies']['leak'] = leak_toggle
    st.session_state.simulation_state['anomalies']['burst'] = burst_toggle
    st.session_state.simulation_state['anomalies']['abnormal'] = usage_toggle
    
    reset_btn = st.button("🔄 Reset Simulation")
    if reset_btn:
        del st.session_state['simulation_state']
        st.rerun()

# ==========================================
# 5. Run Simulation Step
# ==========================================
if st.session_state.simulation_state['is_running']:
    new_data = simulate_step(
        st.session_state.simulation_state['time_min'], 
        PIPES, 
        st.session_state.simulation_state['anomalies']
    )
    
    st.session_state.simulation_state['data_buffer'] = pd.concat(
        [st.session_state.simulation_state['data_buffer'], new_data]
    ).tail(500)
    
    st.session_state.simulation_state['time_min'] += speed
    time.sleep(0.5) 
    st.rerun()

# ==========================================
# 6. Dashboard Display
# ==========================================
st.title("💧 Water Network Digital Twin")

current_df = st.session_state.simulation_state['data_buffer']

# KPI Row
if not current_df.empty:
    latest_time = current_df['time_min'].max()
    latest_snapshot = current_df[current_df['time_min'] == latest_time]
    
    avg_pressure = latest_snapshot['pressure'].mean()
    total_flow = latest_snapshot['flow'].sum()
    max_anomaly = latest_snapshot['anomaly_score'].max()
    
    status = "HEALTHY"
    status_color = "green"
    if max_anomaly > 4:
        status = "WARNING"
        status_color = "orange"
    if max_anomaly > 8:
        status = "CRITICAL"
        status_color = "red"
        
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Simulation Time", f"{int(latest_time//60):02d}:{int(latest_time%60):02d}")
    k2.metric("Avg Pressure", f"{avg_pressure:.2f} m")
    k3.metric("Total Flow", f"{total_flow:.1f} L/s")
    k4.markdown(f"Status: <b style='color:{status_color}; font-size:24px'>{status}</b>", unsafe_allow_html=True)

# Main Charts
st.markdown("---")
focus_pipes = ['P02', 'P05', 'P09']

if not current_df.empty:
    chart_df = current_df[current_df['pipe_id'].isin(focus_pipes)].copy()
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### Pressure Dynamics")
        fig_p = px.line(chart_df, x='time_min', y='pressure', color='pipe_id', 
                        title="Live Pressure Head (m)", template="plotly_dark",
                        color_discrete_sequence=['#00e5ff', '#ff00ff', '#ffff00'])
        fig_p.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_p, width='stretch')
        
    with c2:
        st.markdown("### Flow Dynamics")
        fig_f = px.line(chart_df, x='time_min', y='flow', color='pipe_id', 
                        title="Live Water Flow (L/s)", template="plotly_dark",
                        color_discrete_sequence=['#00e5ff', '#ff00ff', '#ffff00'])
        fig_f.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_f, width='stretch')

    st.markdown("### AI Anomaly Detection Score")
    fig_a = px.area(chart_df, x='time_min', y='anomaly_score', color='pipe_id',
                    title="Real-time Anomaly Severity (0-10)", template="plotly_dark",
                     color_discrete_sequence=['#00e5ff', '#ff00ff', '#ffff00'])
    fig_a.add_hline(y=4, line_dash="dash", line_color="orange", annotation_text="Warning")
    fig_a.add_hline(y=9, line_dash="dash", line_color="red", annotation_text="Critical")
    fig_a.update_yaxes(range=[0, 11])
    fig_a.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_a, width='stretch')

else:
    st.info("Click PLAY to start simulation")
