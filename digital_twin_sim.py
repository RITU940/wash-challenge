"""
Digital Twin Simulation for WASH Innovation Challenge
Author: AI-Driven Digital Twin Team
Date: 2026-01-16

Objective:
Concept validation of an AI-Driven Digital Twin for Intelligent Water Network Monitoring.
Simulates a small water distribution network, injects anomalies, and detects them using 
physics-based expectations and Unsupervised Learning (Isolation Forest).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. Configuration & Constants
# ==========================================
DURATION_HOURS = 24
MINUTES_PER_HOUR = 60
TOTAL_MINUTES = DURATION_HOURS * MINUTES_PER_HOUR
TIME_STEPS = np.arange(TOTAL_MINUTES)
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# Network Properties (Simplified 10-pipe segment model)
# ID, Length (m), Diameter (mm), Base Flow (L/s)
PIPES = [
    {'id': 'P01', 'len': 100, 'dia': 200, 'base_flow': 50},
    {'id': 'P02', 'len': 150, 'dia': 150, 'base_flow': 40},
    {'id': 'P03', 'len': 120, 'dia': 150, 'base_flow': 35},
    {'id': 'P04', 'len': 80,  'dia': 100, 'base_flow': 20},
    {'id': 'P05', 'len': 200, 'dia': 250, 'base_flow': 60},
    {'id': 'P06', 'len': 130, 'dia': 200, 'base_flow': 45},
    {'id': 'P07', 'len': 90,  'dia': 100, 'base_flow': 15},
    {'id': 'P08', 'len': 110, 'dia': 150, 'base_flow': 30},
    {'id': 'P09', 'len': 160, 'dia': 200, 'base_flow': 55},
    {'id': 'P10', 'len': 140, 'dia': 150, 'base_flow': 25},
]

# ==========================================
# 2. Physics & Data Generation Logic
# ==========================================
def get_diurnal_pattern(t_minutes):
    """
    Returns a multiplier (0.2 to 1.8) representing daily water usage pattern.
    Peaks: Morning (7-10 AM) and Evening (6-9 PM).
    """
    hour = (t_minutes / 60) % 24
    
    # Base load
    pattern = 0.3
    
    # Morning Peak (Gaussian centered at 8 AM)
    pattern += 1.0 * np.exp(-((hour - 8)**2) / (2 * 1.5**2))
    
    # Evening Peak (Gaussian centered at 19 PM)
    pattern += 0.8 * np.exp(-((hour - 19)**2) / (2 * 1.5**2))
    
    # Night low (Gaussian dip logic implicit in lack of peaks, but let's smooth it)
    return pattern

def calculate_pressure_drop(flow, length, diameter):
    """
    Hazen-Williams derived approximation for pressure drop.
    P_drop proportional to (Length * Flow^1.852) / (Diameter^4.87)
    This is conceptual for the Digital Twin physics model.
    """
    # Simplified coefficient for demo
    k = 1000  
    # Avoid zero flow issues
    q = np.abs(flow)
    return k * length * (q**1.852) / (diameter**4.87)

def generate_normal_data(pipes, time_steps):
    """Generates expected baseline data (Physics Model)."""
    data = []
    
    demand_pattern = np.array([get_diurnal_pattern(t) for t in time_steps])
    
    for pipe in pipes:
        # Expected Flow = Base Flow * Diurnal Pattern
        # Add small random fluctuation reflecting normal usage variability
        noise_flow = np.random.normal(0, pipe['base_flow']*0.05, len(time_steps))
        expected_flow = pipe['base_flow'] * demand_pattern + noise_flow
        
        # Expected Pressure (Head) at sensor node (simplified)
        # Assume constant source pressure and subtract drop
        source_pressure = 50 # meters
        p_drop = calculate_pressure_drop(expected_flow, pipe['len'], pipe['dia'])
        # Add small sensor noise
        noise_pressure = np.random.normal(0, 0.5, len(time_steps))
        expected_pressure = source_pressure - p_drop + noise_pressure
        
        # DataFrame for this pipe
        df = pd.DataFrame({
            'time_min': time_steps,
            'pipe_id': pipe['id'],
            'flow': expected_flow,
            'pressure': expected_pressure,
            'label': 0 # 0 = Normal
        })
        data.append(df)
        
    return pd.concat(data, ignore_index=True)

# ==========================================
# 3. Anomaly Injection
# ==========================================
def inject_anomalies(df):
    """
    Injects specific anomalies as per challenge requirements.
    Modifies the dataframe in-place or returns copy.
    """
    df_anom = df.copy()
    
    # A. Minor Leak - Pipe P02 - 10:00 to 12:00 (600 to 720 min)
    # Effect: Slight pressure drop, Slight flow mismatch (flow might increase at source or drop at dest)
    # For this simulation: Measured flow drops (leakage before sensor) or Pressure drops
    mask_a = (df_anom['pipe_id'] == 'P02') & (df_anom['time_min'] >= 600) & (df_anom['time_min'] <= 720)
    df_anom.loc[mask_a, 'pressure'] *= 0.85 # 15% drop
    df_anom.loc[mask_a, 'flow'] *= 1.1      # Flow increases to compensate/leak
    df_anom.loc[mask_a, 'label'] = 1        # ID for Minor Leak
    
    # B. Major Pipe Burst - Pipe P05 - 18:30 to 19:00 (1110 to 1140 min)
    # Effect: Sudden pressure collapse, Sharp flow spike
    mask_b = (df_anom['pipe_id'] == 'P05') & (df_anom['time_min'] >= 1110) & (df_anom['time_min'] <= 1140)
    df_anom.loc[mask_b, 'pressure'] *= 0.4  # 60% collapse
    df_anom.loc[mask_b, 'flow'] *= 2.5      # Huge spike
    df_anom.loc[mask_b, 'label'] = 2        # ID for Burst
    
    # C. Abnormal Usage - Pipe P09 - 02:00 to 03:00 (120 to 180 min)
    # Effect: Unexpected high flow during low-demand
    mask_c = (df_anom['pipe_id'] == 'P09') & (df_anom['time_min'] >= 120) & (df_anom['time_min'] <= 180)
    df_anom.loc[mask_c, 'flow'] += 25       # Add specific flow volume
    df_anom.loc[mask_c, 'pressure'] -= 2    # Slight pressure dip
    df_anom.loc[mask_c, 'label'] = 3        # ID for Abnormal Usage
    
    return df_anom

# ==========================================
# 4. Digital Twin / AI Logic
# ==========================================
class DigitalTwinAI:
    def __init__(self):
        # Isolation Forest: Unsupervised Anomaly Detection
        self.model = IsolationForest(contamination=0.05, random_state=RANDOM_SEED)
        self.scaler = MinMaxScaler()
        
    def train(self, normal_data):
        """Train on normal data (Flow and Pressure features)."""
        X = normal_data[['flow', 'pressure']]
        self.model.fit(X)
        
    def predict(self, current_data):
        """
        Returns anomaly scores (normalized 0-10).
        """
        X = current_data[['flow', 'pressure']]
        # Decision function: average anomaly score of X of the base classifiers.
        # lower = more abnormal. We invert this for a 0-10 severity score.
        raw_scores = self.model.decision_function(X)
        
        # Normalize to 0-10 scale where 10 is most anomalous
        # decision_function output is roughly -0.5 to 0.5 for IF.
        # We want negative values (anomalies) to be high scores.
        
        # Transform: Invert and Scale
        # Heuristic scaling for demo visualization
        scores = -raw_scores 
        scores = (scores - scores.min()) / (scores.max() - scores.min()) # 0 to 1
        scores = scores * 10 
        
        return scores

    def classify_severity(self, score):
        if score >= 9: return "Critical Burst", "RED"
        if score >= 7: return "Major Leak", "ORANGE"
        if score >= 4: return "Minor Leak", "YELLOW"
        return "Normal", "GREEN"

# ==========================================
# 5. Simulation execution & Reporting
# ==========================================
def main_simulation():
    print("--- Digital Twin Simulation Started ---")
    
    # 1. Generate Historical Data (Simulated 'Normal' Days for Training)
    print("Step 1: Generating Historical Normal Data for Training...")
    # Simulate 3 days of normal data to train the AI
    train_time_steps = np.arange(3 * MIN_PER_DAY)
    training_data = generate_normal_data(PIPES, train_time_steps)
    
    # 2. Train AI Model
    print("Step 2: Training Unsupervised AI (Isolation Forest)...")
    dt_ai = DigitalTwinAI()
    dt_ai.train(training_data)
    print("      Model Trained on patterns of Flow & Pressure.")
    
    # 3. Generate Live Data (Today) & Inject Anomalies
    print("Step 3: Simulating 'Live' Day with Anomalies...")
    live_data_clean = generate_normal_data(PIPES, TIME_STEPS)
    live_data_anom = inject_anomalies(live_data_clean)
    
    # 4. Run Detection
    print("Step 4: Running Real-time Detection...")
    live_data_anom['anomaly_score'] = dt_ai.predict(live_data_anom)
    
    # 5. Visualization & Reporting
    print("Step 5: Generating Visualization and Assessment...")
    plot_results(live_data_anom)
    
    # Print Summary of Detected Anomalies
    print("\n" + "="*50)
    print("DIGITAL TWIN ANOMALY REPORT")
    print("="*50)
    print(f"{'Time':<10} | {'Pipe':<5} | {'Severity Score':<14} | {'Status':<15}")
    print("-" * 55)
    
    # Scan for high scores to report (simple thresholding for report)
    anomaly_events = live_data_anom[live_data_anom['anomaly_score'] > 4]
    
    # Group by consecutive events or crucial pipes to avoid console flooding
    # We report the peak score for each injected anomaly type/pipe
    for p_id in ['P02', 'P05', 'P09']: # The ones we injected
        subset = anomaly_events[anomaly_events['pipe_id'] == p_id]
        if not subset.empty:
            peak_idx = subset['anomaly_score'].idxmax()
            row = subset.loc[peak_idx]
            
            # Convert minutes to HH:MM
            h = int(row['time_min'] // 60)
            m = int(row['time_min'] % 60)
            time_str = f"{h:02d}:{m:02d}"
            
            status, _ = dt_ai.classify_severity(row['anomaly_score'])
            
            print(f"{time_str:<10} | {row['pipe_id']:<5} | {row['anomaly_score']:.2f}/10      | {status}")

    print("="*50)
    print("\nPlots generated. Close the figure window to finish.")
    plt.show()

def plot_results(df):
    """
    Plots Flow, Pressure, and Anomaly Score for the affected pipes.
    """
    target_pipes = ['P02', 'P05', 'P09']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    colors = {'P02': 'tab:blue', 'P05': 'tab:red', 'P09': 'tab:green'}
    
    # X-axis ticks (Hours)
    hours = np.arange(0, 25, 2)
    hour_labels = [f"{h:02d}:00" for h in hours]
    
    for ax in axes:
        ax.set_xticks(hours * 60)
        ax.set_xticklabels(hour_labels)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#fafafa')
    
    # Plot 1: Flow
    ax0 = axes[0]
    ax0.set_title("Water Flow Monitoring (L/s)", fontsize=12, fontweight='bold')
    ax0.set_ylabel("Flow (L/s)")
    
    # Plot 2: Pressure
    ax1 = axes[1]
    ax1.set_title("Pressure Monitoring (m)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Pressure Head (m)")
    
    # Plot 3: Anomaly Score
    ax2 = axes[2]
    ax2.set_title("AI Anomaly Score (0-10)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Severity Score")
    ax2.set_xlabel("Time of Day")
    ax2.set_ylim(0, 10.5)
    
    # Critical Threshold Lines
    ax2.axhline(y=4, color='yellow', linestyle='--', alpha=0.5, label='Warning (4)')
    ax2.axhline(y=7, color='orange', linestyle='--', alpha=0.5, label='Major (7)')
    ax2.axhline(y=9, color='red', linestyle='--', alpha=0.5, label='Critical (9)')
    
    handles = []
    labels = []
    
    for pid in target_pipes:
        subset = df[df['pipe_id'] == pid]
        c = colors[pid]
        
        # Flow
        l1, = ax0.plot(subset['time_min'], subset['flow'], label=f"Pipe {pid}", color=c, alpha=0.8)
        
        # Pressure
        ax1.plot(subset['time_min'], subset['pressure'], label=f"Pipe {pid}", color=c, alpha=0.8)
        
        # Score
        ax2.plot(subset['time_min'], subset['anomaly_score'], label=f"Pipe {pid}", color=c, linewidth=2)
        
        if pid == target_pipes[0]:
            handles.append(l1)
            labels.append(f"Pipe {pid}")
    
    # Legend just once
    ax0.legend(loc='upper right')
    ax2.legend(loc='upper left')

if __name__ == "__main__":
    # Helper for training data generation constant
    MIN_PER_DAY = 1440
    main_simulation()
