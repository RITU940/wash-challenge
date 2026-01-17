# WASH Challenge Digital Twin Dashboard

This project is an AI-Driven Digital Twin for Intelligent Water Network Monitoring. It simulates a water distribution network, injects anomalies, and uses unsupervised learning (Isolation Forest) to detect them in real-time.

## Features
- **Real-time Simulation**: Synthesizes water flow and pressure data based on daily usage patterns.
- **Anomaly Injection**: Interactive controls to simulate leaks, bursts, and abnormal usage.
- **AI Detection**: Uses a pre-trained Isolation Forest model to score anomaly severity.
- **Interactive Dashboard**: Built with Streamlit and Plotly for deep insights.

## Installation

Ensure you have a Python environment (v3.8+) ready.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Dashboard

Launch the dashboard with the following command:

```bash
streamlit run dashboard_app.py
```

## Structure
- `dashboard_app.py`: Main dashboard application.
- `digital_twin_sim.py`: Core logic for physics simulation and AI modeling.
