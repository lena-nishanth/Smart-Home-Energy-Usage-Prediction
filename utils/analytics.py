import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict, List, Tuple, Optional
import json

def detect_anomalies(series: pd.Series, contamination: float = 0.1) -> Dict:
    """
    Detect anomalies in a time series using Isolation Forest.
    Returns a dictionary with anomaly indices, scores, and visualization data.
    """
    # Prepare data for anomaly detection
    X = series.values.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    clf = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    # Predict anomalies
    is_anomaly = clf.fit_predict(X_scaled)
    anomaly_scores = -clf.score_samples(X_scaled)  # Higher score = more anomalous
    
    # Get anomaly indices and values
    anomaly_indices = np.where(is_anomaly == -1)[0]
    anomalies = series.iloc[anomaly_indices]
    
    return {
        'indices': anomaly_indices.tolist(),
        'scores': anomaly_scores.tolist(),
        'anomaly_points': [
            {'x': str(series.index[i]), 'y': float(series.iloc[i])} 
            for i in anomaly_indices
        ]
    }

def analyze_seasonal_patterns(
    series: pd.Series, 
    freq: str = 'D',  # 'D' for daily, 'H' for hourly, etc.
    period: Optional[int] = None
) -> Dict:
    """
    Decompose time series into trend, seasonal, and residual components.
    """
    if period is None:
        # Set default periods based on frequency
        period = {
            'H': 24,      # Daily seasonality
            'D': 7,       # Weekly seasonality
            'M': 12       # Yearly seasonality
        }.get(freq, 1)
    
    # Handle missing values
    series = series.interpolate().fillna(method='bfill').fillna(method='ffill')
    
    try:
        # Decompose the time series
        decomposition = seasonal_decompose(
            series,
            period=period,
            extrapolate_trend='freq'
        )
        
        return {
            'trend': decomposition.trend.dropna().to_dict(),
            'seasonal': decomposition.seasonal.dropna().to_dict(),
            'residual': decomposition.resid.dropna().to_dict(),
            'period': period
        }
    except Exception as e:
        return {'error': str(e)}

def get_energy_insights(series: pd.Series) -> Dict:
    """
    Generate insights from energy consumption data.
    """
    insights = {}
    
    # Basic statistics
    insights['stats'] = {
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max()),
        'total': float(series.sum())
    }
    
    # Time-based patterns
    if len(series) > 24:  # Need sufficient data points
        # Daily patterns
        if isinstance(series.index, pd.DatetimeIndex):
            daily = series.resample('D').mean()
            weekly = series.resample('W').mean()
            
            insights['patterns'] = {
                'peak_hour': series.groupby(series.index.hour).mean().idxmax(),
                'peak_day': series.groupby(series.index.weekday).mean().idxmax(),
                'daily_avg': daily.mean(),
                'weekly_avg': weekly.mean(),
                'is_weekend_higher': (
                    series[series.index.weekday >= 5].mean() > 
                    series[series.index.weekday < 5].mean()
                )
            }
    
    return insights

def detect_consumption_shifts(
    series: pd.Series, 
    window: int = 24
) -> List[Dict]:
    """
    Detect significant shifts in energy consumption patterns.
    """
    if len(series) < window * 2:
        return []
    
    # Calculate rolling statistics
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    # Detect shifts (simple approach using z-score)
    z_scores = (series - rolling_mean) / rolling_std
    shift_indices = np.where(np.abs(z_scores) > 3)[0]
    
    shifts = []
    for idx in shift_indices:
        if idx >= window:
            prev_avg = series.iloc[idx-window:idx].mean()
            current_avg = series.iloc[idx:idx+window].mean()
            if not np.isnan(prev_avg) and not np.isnan(current_avg):
                shift = (current_avg - prev_avg) / prev_avg * 100
                shifts.append({
                    'timestamp': str(series.index[idx]),
                    'percent_change': float(shift),
                    'previous_avg': float(prev_avg),
                    'new_avg': float(current_avg)
                })
    
    return shifts
