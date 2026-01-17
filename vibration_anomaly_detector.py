"""
Vibration-Based Structural Stress Detection using Isolation Forest

This module implements an unsupervised anomaly detection model for detecting
abnormal vibration patterns in WASH infrastructure (pipes, pumps, etc.) using
MPU6050 accelerometer data.

Key Features:
- Isolation Forest for unsupervised anomaly detection
- Vibration magnitude computation from 3-axis accelerometer data
- Simple thresholding for binary classification
- Visualization of anomaly scores
- Model persistence with joblib

Author: WASH Infrastructure Monitoring System
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_score, 
    recall_score, 
    f1_score,
    precision_recall_curve,
    roc_curve,
    auc
)
import joblib
from datetime import datetime, timedelta
import os
from typing import Tuple, Optional, Dict


class VibrationAnomalyDetector:
    """
    Anomaly detector for vibration-based structural stress detection.
    
    Uses Isolation Forest algorithm to identify abnormal vibration patterns
    from MPU6050 accelerometer readings (ax, ay, az).
    """
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
        anomaly_threshold: float = 0.0
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies in training data (0-0.5)
            n_estimators: Number of trees in the Isolation Forest
            random_state: Random seed for reproducibility
            anomaly_threshold: Threshold for anomaly score classification
                              (scores below this are anomalies)
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.anomaly_threshold = anomaly_threshold
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def compute_vibration_magnitude(self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
        """
        Compute vibration magnitude from 3-axis accelerometer data.
        
        Formula: magnitude = sqrt(ax^2 + ay^2 + az^2)
        
        Args:
            ax: X-axis acceleration values
            ay: Y-axis acceleration values
            az: Z-axis acceleration values
            
        Returns:
            Array of vibration magnitudes
        """
        return np.sqrt(ax**2 + ay**2 + az**2)
    
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from accelerometer data for anomaly detection.
        
        Features extracted:
        1. Vibration magnitude
        2. Raw ax, ay, az (optional, for richer representation)
        
        Args:
            data: DataFrame with columns ['ax', 'ay', 'az']
            
        Returns:
            Feature array for model input
        """
        # Compute vibration magnitude
        magnitude = self.compute_vibration_magnitude(
            data['ax'].values,
            data['ay'].values,
            data['az'].values
        )
        
        # Create feature matrix with magnitude and raw values
        features = np.column_stack([
            magnitude,
            data['ax'].values,
            data['ay'].values,
            data['az'].values
        ])
        
        return features
    
    def fit(self, normal_data: pd.DataFrame) -> 'VibrationAnomalyDetector':
        """
        Train the anomaly detection model on normal (healthy) vibration data.
        
        Args:
            normal_data: DataFrame with columns ['ax', 'ay', 'az'] containing
                         only normal/healthy vibration patterns
                         
        Returns:
            Self for method chaining
        """
        print("=" * 60)
        print("Training Vibration Anomaly Detector")
        print("=" * 60)
        
        # Extract features
        features = self.extract_features(normal_data)
        print(f"✓ Extracted features from {len(normal_data)} samples")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        print("✓ Scaled features")
        
        # Train Isolation Forest
        import time
        start_time = time.time()
        self.model.fit(features_scaled)
        training_time = time.time() - start_time
        print(f"✓ Trained Isolation Forest in {training_time:.2f} seconds")
        
        self.is_fitted = True
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        return self
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict anomaly scores and classifications for new data.
        
        Args:
            data: DataFrame with columns ['ax', 'ay', 'az']
            
        Returns:
            Tuple of:
                - anomaly_scores: Continuous anomaly scores (lower = more anomalous)
                - predictions: Binary predictions (-1 = anomaly, 1 = normal)
                - magnitudes: Vibration magnitudes
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        # Extract and scale features
        features = self.extract_features(data)
        features_scaled = self.scaler.transform(features)
        
        # Get anomaly scores (decision function)
        anomaly_scores = self.model.decision_function(features_scaled)
        
        # Get predictions (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(features_scaled)
        
        # Compute magnitudes for output
        magnitudes = self.compute_vibration_magnitude(
            data['ax'].values,
            data['ay'].values,
            data['az'].values
        )
        
        return anomaly_scores, predictions, magnitudes
    
    def classify_with_threshold(
        self,
        anomaly_scores: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply simple thresholding for binary classification.
        
        Args:
            anomaly_scores: Anomaly scores from predict()
            threshold: Custom threshold (uses self.anomaly_threshold if None)
            
        Returns:
            Binary labels: 0 = Normal, 1 = Anomaly
        """
        threshold = threshold if threshold is not None else self.anomaly_threshold
        
        # Scores below threshold are anomalies
        binary_labels = (anomaly_scores < threshold).astype(int)
        
        return binary_labels
    
    def get_classification_labels(self, binary_labels: np.ndarray) -> np.ndarray:
        """
        Convert binary labels to human-readable strings.
        
        Args:
            binary_labels: Array of 0s (normal) and 1s (anomaly)
            
        Returns:
            Array of 'Normal' or 'Anomaly' strings
        """
        return np.where(binary_labels == 0, 'Normal', 'Anomaly')
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk using joblib.
        
        Args:
            filepath: Path to save the model (e.g., 'vibration_model.joblib')
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'anomaly_threshold': self.anomaly_threshold,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'VibrationAnomalyDetector':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            VibrationAnomalyDetector instance with loaded model
        """
        model_data = joblib.load(filepath)
        
        detector = cls(
            contamination=model_data['contamination'],
            n_estimators=model_data['n_estimators'],
            anomaly_threshold=model_data['anomaly_threshold']
        )
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.is_fitted = model_data['is_fitted']
        
        print(f"✓ Model loaded from: {filepath}")
        return detector


def generate_synthetic_accelerometer_data(
    n_samples: int = 1000,
    include_anomalies: bool = False,
    anomaly_ratio: float = 0.05,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic MPU6050 accelerometer data for testing.
    
    Normal vibrations simulate healthy pipe operation.
    Anomalies simulate stress events (shocks, excessive vibration).
    
    Args:
        n_samples: Number of samples to generate
        include_anomalies: Whether to include anomalous patterns
        anomaly_ratio: Proportion of anomalies if include_anomalies=True
        random_state: Random seed
        
    Returns:
        DataFrame with columns ['timestamp', 'ax', 'ay', 'az', 'is_anomaly']
    """
    np.random.seed(random_state)
    
    # Generate timestamps
    start_time = datetime.now()
    timestamps = [start_time + timedelta(milliseconds=i*10) for i in range(n_samples)]
    
    # Normal operating vibration (small oscillations around gravity)
    # MPU6050 outputs in g units, 1g ≈ 9.81 m/s²
    normal_ax = np.random.normal(0.0, 0.05, n_samples)  # X-axis: minimal
    normal_ay = np.random.normal(0.0, 0.05, n_samples)  # Y-axis: minimal
    normal_az = np.random.normal(1.0, 0.05, n_samples)  # Z-axis: gravity ~1g
    
    # Add some sinusoidal vibration (simulating pump operation)
    t = np.linspace(0, 4 * np.pi, n_samples)
    normal_ax += 0.02 * np.sin(t * 5)
    normal_ay += 0.02 * np.sin(t * 5 + np.pi/4)
    
    is_anomaly = np.zeros(n_samples, dtype=int)
    
    if include_anomalies:
        n_anomalies = int(n_samples * anomaly_ratio)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            # Simulate various anomaly types
            anomaly_type = np.random.choice(['shock', 'excessive_vibration', 'irregular'])
            
            if anomaly_type == 'shock':
                # Sudden impact
                normal_ax[idx] += np.random.uniform(1.5, 3.0) * np.random.choice([-1, 1])
                normal_ay[idx] += np.random.uniform(1.0, 2.0) * np.random.choice([-1, 1])
                normal_az[idx] += np.random.uniform(0.5, 1.5) * np.random.choice([-1, 1])
            elif anomaly_type == 'excessive_vibration':
                # High amplitude oscillation (affects a window)
                window = range(max(0, idx-5), min(n_samples, idx+5))
                for w in window:
                    normal_ax[w] *= np.random.uniform(3, 6)
                    normal_ay[w] *= np.random.uniform(3, 6)
            else:
                # Irregular pattern
                normal_ax[idx] = np.random.uniform(-2, 2)
                normal_ay[idx] = np.random.uniform(-2, 2)
                normal_az[idx] = np.random.uniform(-1, 3)
            
            is_anomaly[idx] = 1
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'ax': normal_ax,
        'ay': normal_ay,
        'az': normal_az,
        'is_anomaly': is_anomaly
    })
    
    return df


def plot_vibration_analysis(
    magnitudes: np.ndarray,
    anomaly_scores: np.ndarray,
    binary_labels: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create visualization of vibration magnitude vs anomaly score.
    
    Args:
        magnitudes: Vibration magnitude values
        anomaly_scores: Anomaly scores from the model
        binary_labels: Binary classification (0=Normal, 1=Anomaly)
        timestamps: Optional timestamps for x-axis
        save_path: Path to save the figure (displays if None)
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Create sample indices if no timestamps
    x = np.arange(len(magnitudes)) if timestamps is None else timestamps
    
    # Color coding
    colors = np.where(binary_labels == 1, '#e74c3c', '#2ecc71')  # Red for anomaly, green for normal
    
    # Plot 1: Vibration Magnitude
    axes[0].scatter(x, magnitudes, c=colors, alpha=0.6, s=10)
    axes[0].set_ylabel('Vibration Magnitude (g)', fontsize=11)
    axes[0].set_title('Vibration Magnitude Over Time', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=np.mean(magnitudes), color='#3498db', linestyle='--', 
                    label=f'Mean: {np.mean(magnitudes):.3f}', linewidth=2)
    axes[0].legend()
    
    # Plot 2: Anomaly Scores
    axes[1].scatter(x, anomaly_scores, c=colors, alpha=0.6, s=10)
    axes[1].set_ylabel('Anomaly Score', fontsize=11)
    axes[1].set_title('Anomaly Score Over Time', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[1].axhline(y=-0.05, color='#e74c3c', linestyle='--', 
                    label='Anomaly Threshold', linewidth=2)
    axes[1].legend()
    
    # Plot 3: Binary Classification
    axes[2].scatter(x, binary_labels, c=colors, alpha=0.8, s=20)
    axes[2].set_ylabel('Classification', fontsize=11)
    axes[2].set_xlabel('Sample Index' if timestamps is None else 'Time', fontsize=11)
    axes[2].set_title('Binary Classification: Normal (0) / Anomaly (1)', fontsize=12, fontweight='bold')
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['Normal', 'Anomaly'])
    axes[2].grid(True, alpha=0.3)
    
    # Add legend for all plots
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Normal'),
                       Patch(facecolor='#e74c3c', label='Anomaly')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()


def plot_magnitude_vs_anomaly_score(
    magnitudes: np.ndarray,
    anomaly_scores: np.ndarray,
    binary_labels: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Create scatter plot of vibration magnitude vs anomaly score.
    
    Args:
        magnitudes: Vibration magnitude values
        anomaly_scores: Anomaly scores from the model  
        binary_labels: Binary classification (0=Normal, 1=Anomaly)
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate normal and anomaly points
    normal_mask = binary_labels == 0
    anomaly_mask = binary_labels == 1
    
    # Plot normal points
    ax.scatter(magnitudes[normal_mask], anomaly_scores[normal_mask],
               c='#2ecc71', alpha=0.6, s=50, label='Normal', edgecolors='white', linewidth=0.5)
    
    # Plot anomaly points
    ax.scatter(magnitudes[anomaly_mask], anomaly_scores[anomaly_mask],
               c='#e74c3c', alpha=0.8, s=80, label='Anomaly', edgecolors='white', linewidth=0.5,
               marker='X')
    
    # Add threshold line
    ax.axhline(y=-0.05, color='#e74c3c', linestyle='--', linewidth=2, 
               label='Anomaly Threshold', alpha=0.8)
    
    ax.set_xlabel('Vibration Magnitude (g)', fontsize=12)
    ax.set_ylabel('Anomaly Score', fontsize=12)
    ax.set_title('Vibration Magnitude vs Anomaly Score\n(WASH Infrastructure Monitoring)', 
                 fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('Anomaly Zone', xy=(0.95, 0.1), xycoords='axes fraction',
                fontsize=10, color='#e74c3c', fontweight='bold',
                ha='right', va='bottom')
    ax.annotate('Normal Zone', xy=(0.95, 0.6), xycoords='axes fraction',
                fontsize=10, color='#2ecc71', fontweight='bold',
                ha='right', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray
) -> Dict[str, float]:
    """
    Compute evaluation metrics for the anomaly detection model.
    
    Args:
        y_true: Ground truth labels (0=Normal, 1=Anomaly)
        y_pred: Predicted labels (0=Normal, 1=Anomaly)
        anomaly_scores: Continuous anomaly scores from the model
        
    Returns:
        Dictionary containing precision, recall, F1 score, and other metrics
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print a detailed classification report."""
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'], digits=4)
    print(report)
    print("=" * 60)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                annot_kws={'size': 16, 'weight': 'bold'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix\nVibration Anomaly Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    else:
        plt.show()


def plot_metrics_summary(metrics: Dict[str, float], save_path: Optional[str] = None) -> None:
    """Plot a bar chart of evaluation metrics."""
    viz_metrics = {
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1 Score': metrics['f1_score'],
        'Accuracy': metrics['accuracy'],
        'Specificity': metrics['specificity']
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71' if v >= 0.7 else '#f39c12' if v >= 0.5 else '#e74c3c' 
              for v in viz_metrics.values()]
    
    bars = ax.bar(viz_metrics.keys(), viz_metrics.values(), color=colors, 
                  edgecolor='white', linewidth=2)
    
    for bar, value in zip(bars, viz_metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics\nVibration Anomaly Detection', fontsize=14, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.5)')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (0.8)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Metrics summary saved to: {save_path}")
    else:
        plt.show()


def plot_anomaly_score_distribution(
    anomaly_scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float = -0.05,
    save_path: Optional[str] = None
) -> None:
    """Plot the distribution of anomaly scores (alternative to loss visualization)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    normal_scores = anomaly_scores[y_true == 0]
    anomaly_scores_true = anomaly_scores[y_true == 1]
    
    ax1 = axes[0]
    ax1.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='#2ecc71', density=True)
    ax1.hist(anomaly_scores_true, bins=30, alpha=0.7, label='Anomaly', color='#e74c3c', density=True)
    ax1.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax1.set_xlabel('Anomaly Score', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Anomaly Score Distribution by Class', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    bp = ax2.boxplot([normal_scores, anomaly_scores_true], labels=['Normal', 'Anomaly'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax2.axhline(y=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax2.set_ylabel('Anomaly Score', fontsize=11)
    ax2.set_title('Anomaly Score Box Plot by Class', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Anomaly Score Analysis (Alternative to Loss Curve)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Score distribution saved to: {save_path}")
    else:
        plt.show()


def plot_learning_curve(
    detector: 'VibrationAnomalyDetector',
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float = -0.05,
    n_points: int = 10,
    save_path: Optional[str] = None
) -> None:
    """Plot learning curve showing model performance vs training data size."""
    import io
    import sys
    
    train_sizes = np.linspace(0.1, 1.0, n_points)
    n_train = len(train_data)
    
    f1_list, precision_list, recall_list, train_size_list = [], [], [], []
    
    for frac in train_sizes:
        n_samples = int(n_train * frac)
        if n_samples < 10:
            continue
        
        train_subset = train_data.iloc[:n_samples]
        temp_detector = VibrationAnomalyDetector(
            contamination=detector.contamination,
            n_estimators=detector.n_estimators,
            anomaly_threshold=threshold
        )
        
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        temp_detector.fit(train_subset)
        sys.stdout = old_stdout
        
        scores, _, _ = temp_detector.predict(test_data)
        y_pred = temp_detector.classify_with_threshold(scores, threshold)
        
        f1_list.append(f1_score(y_test, y_pred, zero_division=0))
        precision_list.append(precision_score(y_test, y_pred, zero_division=0))
        recall_list.append(recall_score(y_test, y_pred, zero_division=0))
        train_size_list.append(n_samples)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_size_list, f1_list, 'o-', color='#3498db', linewidth=2, markersize=8, label='F1 Score')
    ax.plot(train_size_list, precision_list, 's-', color='#2ecc71', linewidth=2, markersize=8, label='Precision')
    ax.plot(train_size_list, recall_list, '^-', color='#e74c3c', linewidth=2, markersize=8, label='Recall')
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Learning Curve: Performance vs Training Data Size\n(Alternative to Loss Curve)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Learning curve saved to: {save_path}")
    else:
        plt.show()


def plot_precision_recall_curve_analysis(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    save_path: Optional[str] = None
) -> float:
    """Plot Precision-Recall curve for threshold analysis."""
    inverted_scores = -anomaly_scores
    precision, recall, thresholds = precision_recall_curve(y_true, inverted_scores)
    f1_curve = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_curve)
    best_threshold = -thresholds[best_idx] if best_idx < len(thresholds) else 0
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.plot(recall, precision, 'b-', linewidth=2, label='PR Curve')
    ax1.scatter([recall[best_idx]], [precision[best_idx]], color='red', s=100, zorder=5, label=f'Best F1: {f1_curve[best_idx]:.3f}')
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)
    auc_pr = auc(recall, precision)
    ax1.annotate(f'AUC-PR: {auc_pr:.3f}', xy=(0.6, 0.2), fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    ax2 = axes[1]
    ax2.plot(-thresholds, f1_curve[:-1], 'g-', linewidth=2)
    ax2.axvline(x=best_threshold, color='red', linestyle='--', linewidth=2, label=f'Best Threshold: {best_threshold:.3f}')
    ax2.set_xlabel('Anomaly Score Threshold', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score vs Threshold', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Threshold Analysis for Anomaly Detection', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Precision-Recall curve saved to: {save_path}")
    else:
        plt.show()
    
    return best_threshold


def main():
    """
    Main function demonstrating the vibration anomaly detection pipeline.
    """
    print("\n" + "=" * 70)
    print("  VIBRATION-BASED STRUCTURAL STRESS DETECTION")
    print("  WASH Infrastructure Monitoring System")
    print("=" * 70 + "\n")
    
    # ===== Step 1: Generate Training Data (Normal/Healthy only) =====
    print("Step 1: Generating normal (healthy) training data...")
    train_data = generate_synthetic_accelerometer_data(
        n_samples=2000,
        include_anomalies=False,  # Training on NORMAL data only
        random_state=42
    )
    print(f"  → Generated {len(train_data)} normal training samples\n")
    
    # ===== Step 2: Train the Model =====
    print("Step 2: Training Isolation Forest model...")
    detector = VibrationAnomalyDetector(
        contamination=0.05,  # Expect ~5% anomalies in future data
        n_estimators=100,
        anomaly_threshold=-0.05  # Adjusted threshold for better anomaly detection
    )
    detector.fit(train_data)
    print()
    
    # ===== Step 3: Generate Test Data (with anomalies) =====
    print("Step 3: Generating test data with anomalies...")
    test_data = generate_synthetic_accelerometer_data(
        n_samples=500,
        include_anomalies=True,
        anomaly_ratio=0.08,  # 8% anomalies
        random_state=123
    )
    print(f"  → Generated {len(test_data)} test samples")
    print(f"  → Actual anomalies: {test_data['is_anomaly'].sum()}\n")
    
    # ===== Step 4: Predict and Classify =====
    print("Step 4: Running prediction...")
    anomaly_scores, predictions, magnitudes = detector.predict(test_data)
    binary_labels = detector.classify_with_threshold(anomaly_scores)
    labels = detector.get_classification_labels(binary_labels)
    
    # ===== Step 5: Print Results =====
    print("\n" + "=" * 50)
    print("  DETECTION RESULTS")
    print("=" * 50)
    
    n_detected_anomalies = np.sum(binary_labels == 1)
    n_actual_anomalies = test_data['is_anomaly'].sum()
    
    print(f"  Total samples analyzed:    {len(test_data)}")
    print(f"  Detected anomalies:        {n_detected_anomalies}")
    print(f"  Actual anomalies:          {n_actual_anomalies}")
    print(f"  Normal samples:            {len(test_data) - n_detected_anomalies}")
    print()
    print(f"  Mean anomaly score:        {np.mean(anomaly_scores):.4f}")
    print(f"  Min anomaly score:         {np.min(anomaly_scores):.4f}")
    print(f"  Max anomaly score:         {np.max(anomaly_scores):.4f}")
    print(f"  Mean vibration magnitude:  {np.mean(magnitudes):.4f} g")
    print("=" * 50 + "\n")
    
    # ===== Step 6: Create Results DataFrame =====
    results_df = pd.DataFrame({
        'timestamp': test_data['timestamp'],
        'ax': test_data['ax'],
        'ay': test_data['ay'],
        'az': test_data['az'],
        'magnitude': magnitudes,
        'anomaly_score': anomaly_scores,
        'classification': labels,
        'is_anomaly_binary': binary_labels
    })
    
    print("Sample Results (first 10 rows):")
    print(results_df[['magnitude', 'anomaly_score', 'classification']].head(10).to_string())
    print()
    
    # ===== Step 7: Save the Model =====
    print("Step 5: Saving trained model...")
    model_path = "vibration_anomaly_model.joblib"
    detector.save_model(model_path)
    print()
    
    # ===== Step 8: Create Visualizations =====
    print("Step 6: Creating visualizations...")
    
    # Time series plot
    plot_vibration_analysis(
        magnitudes=magnitudes,
        anomaly_scores=anomaly_scores,
        binary_labels=binary_labels,
        save_path="vibration_analysis_timeseries.png"
    )
    
    # Magnitude vs Anomaly Score scatter plot
    plot_magnitude_vs_anomaly_score(
        magnitudes=magnitudes,
        anomaly_scores=anomaly_scores,
        binary_labels=binary_labels,
        save_path="magnitude_vs_anomaly_score.png"
    )
    
    # ===== Step 7: Evaluate Model Performance =====
    print("\nStep 7: Evaluating model performance...")
    
    # Get ground truth labels
    y_true = test_data['is_anomaly'].values
    y_pred = binary_labels
    
    # Calculate metrics
    metrics = evaluate_model(y_true, y_pred, anomaly_scores)
    
    # Print classification report
    print_classification_report(y_true, y_pred)
    
    # Print metrics summary
    print("\n" + "=" * 50)
    print("  PERFORMANCE METRICS")
    print("=" * 50)
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 Score:     {metrics['f1_score']:.4f}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Specificity:  {metrics['specificity']:.4f}")
    print("=" * 50)
    
    # ===== Step 8: Create Evaluation Visualizations =====
    print("\nStep 8: Creating evaluation visualizations...")
    
    # Confusion Matrix
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        save_path="confusion_matrix.png"
    )
    
    # Metrics Summary Bar Chart
    plot_metrics_summary(
        metrics=metrics,
        save_path="metrics_summary.png"
    )
    
    # Anomaly Score Distribution (alternative to loss curve)
    plot_anomaly_score_distribution(
        anomaly_scores=anomaly_scores,
        y_true=y_true,
        threshold=-0.05,
        save_path="score_distribution.png"
    )
    
    # Learning Curve (alternative to loss curve)
    print("\n  Creating learning curve (this may take a moment)...")
    plot_learning_curve(
        detector=detector,
        train_data=train_data,
        test_data=test_data,
        y_test=y_true,
        threshold=-0.05,
        n_points=8,
        save_path="learning_curve.png"
    )
    
    # Precision-Recall Curve
    best_threshold = plot_precision_recall_curve_analysis(
        y_true=y_true,
        anomaly_scores=anomaly_scores,
        save_path="precision_recall_curve.png"
    )
    print(f"  → Best threshold based on F1: {best_threshold:.4f}")
    
    # ===== Step 9: Save the Model =====
    print("\nStep 9: Saving trained model...")
    model_path = "vibration_anomaly_model.joblib"
    detector.save_model(model_path)
    
    # ===== Step 10: Demonstrate Model Loading =====
    print("\nStep 10: Demonstrating model reload...")
    loaded_detector = VibrationAnomalyDetector.load_model(model_path)
    
    # Verify loaded model works
    test_scores, _, _ = loaded_detector.predict(test_data.head(10))
    print(f"  → Loaded model prediction works! First score: {test_scores[0]:.4f}")
    
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nOutputs generated:")
    print(f"  1. Trained model:        {model_path}")
    print("  2. Time series plot:     vibration_analysis_timeseries.png")
    print("  3. Scatter plot:         magnitude_vs_anomaly_score.png")
    print("  4. Confusion matrix:     confusion_matrix.png")
    print("  5. Metrics summary:      metrics_summary.png")
    print("  6. Score distribution:   score_distribution.png")
    print("  7. Learning curve:       learning_curve.png")
    print("  8. PR curve analysis:    precision_recall_curve.png")
    print("\nThis model can now be deployed for real-time pipe stress monitoring.")
    print("=" * 70 + "\n")
    
    return detector, results_df, metrics


if __name__ == "__main__":
    detector, results, metrics = main()

