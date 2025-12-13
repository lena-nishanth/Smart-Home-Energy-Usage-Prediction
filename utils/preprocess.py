from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

REQ_COLS = [
    "Timestamp", "Appliance", "Status", "Voltage", "Current", "Power", "Energy"
]


def required_columns() -> List[str]:
    return REQ_COLS


def validate_columns(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, "OK"


def filter_dataframe(df: pd.DataFrame, appliance: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    f = df
    if appliance:
        f = f[f["Appliance"] == appliance]
    if start:
        f = f[f["Timestamp"] >= pd.to_datetime(start, errors="coerce")]
    if end:
        f = f[f["Timestamp"] <= pd.to_datetime(end, errors="coerce")]
    return f.copy()


# ---------- Feature engineering for Energy prediction ----------


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Appliance" in d.columns:
        d["Appliance_encoded"] = d["Appliance"].astype("category").cat.codes.replace(-1, np.nan)
    if "Status" in d.columns:
        d["Status_encoded"] = d["Status"].astype("category").cat.codes.replace(-1, np.nan)
    return d

def _infer_dt_hours(frame: pd.DataFrame) -> float:
    try:
        ts = pd.to_datetime(frame["Timestamp"])
        deltas = ts.diff().dropna().dt.total_seconds()
        if len(deltas) == 0:
            return 0.0
        sec = float(deltas.median())
        return max(sec / 3600.0, 0.0)
    except Exception:
        return 0.0


def _looks_cumulative(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3:
        return False
    diffs = s.diff().dropna()
    nonneg_ratio = np.mean(diffs >= -1e-9)
    return nonneg_ratio > 0.95


def compute_interval_energy_kwh(df: pd.DataFrame) -> pd.Series:
    """Return per-interval energy in kWh for each row after the first."""
    df = df.sort_values("Timestamp").copy()
    if "Energy" in df.columns and _looks_cumulative(df["Energy"]):
        y = pd.to_numeric(df["Energy"], errors="coerce").diff()
        y = y.clip(lower=0.0)  # avoid negative wraparounds
        return y.fillna(0.0)
    # derive from Power
    dt_h = _infer_dt_hours(df)
    if dt_h <= 0:
        return pd.Series(0.0, index=df.index)
    p = pd.to_numeric(df.get("Power", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    med = float(np.median(p)) if len(p) else 0.0
    power_kw = p / 1000.0 if med > 10 else p
    return (power_kw * dt_h).fillna(0.0)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ts = pd.to_datetime(d["Timestamp"]) 
    d["hour"] = ts.dt.hour
    d["dow"] = ts.dt.dayofweek
    # cyclic encodings
    d["hour_sin"], d["hour_cos"] = np.sin(2*np.pi*d["hour"]/24), np.cos(2*np.pi*d["hour"]/24)
    d["dow_sin"], d["dow_cos"] = np.sin(2*np.pi*d["dow"]/7), np.cos(2*np.pi*d["dow"]/7)
    return d


def build_energy_dataset(df: pd.DataFrame, lags: List[int] = [1,2,3,6,12,24], rolls: List[int] = [3,6,12,24]) -> Tuple[pd.DataFrame, pd.Series]:
    """Create features and target for Energy per-interval prediction.
    Returns X (features) and y (target interval energy kWh), aligned with NaNs dropped.
    """
    d = df.sort_values("Timestamp").copy()
    d = add_time_features(d)
    d = _encode_categoricals(d)
    # target per interval
    d["y_energy_kwh"] = compute_interval_energy_kwh(d)
    # base numeric drivers
    for col in ["Power", "Voltage", "Current"]:
        if col in d.columns:
            s = pd.to_numeric(d[col], errors="coerce").astype(float)
            d[col] = s
            # lags
            for L in lags:
                d[f"{col}_lag{L}"] = s.shift(L)
            # rolling stats
            for R in rolls:
                d[f"{col}_mean{R}"] = s.rolling(R, min_periods=1).mean()
                d[f"{col}_std{R}"] = s.rolling(R, min_periods=1).std().fillna(0.0)
    # past energy deltas as features
    e = d["y_energy_kwh"].astype(float)
    for L in lags:
        d[f"y_lag{L}"] = e.shift(L)
    for R in rolls:
        # past-only rolling stats to avoid leakage: shift by 1 before rolling
        d[f"y_mean{R}"] = e.shift(1).rolling(R, min_periods=1).mean()

    feature_cols = [c for c in d.columns if c not in REQ_COLS + ["y_energy_kwh"]]
    # Remove non-numeric features and helpers
    X = d[feature_cols].select_dtypes(include=[np.number]).copy()
    y = pd.to_numeric(d["y_energy_kwh"], errors="coerce").astype(float).copy()

    if not X.empty:
        X_imputed = X.copy()
        for col in X_imputed.columns:
            s = pd.to_numeric(X_imputed[col], errors="coerce")
            med = s.median()
            if pd.isna(med):
                med = 0.0
            X_imputed[col] = s.fillna(med)
        for col in X_imputed.columns:
            s = X_imputed[col].astype(float)
            mean = float(s.mean()) if len(s) else 0.0
            std = float(s.std()) if len(s) > 1 else 0.0
            if np.isfinite(std) and std > 0:
                X_imputed[col] = (s - mean) / std
            else:
                X_imputed[col] = s - mean
        X = X_imputed

    valid = np.isfinite(y)
    X = X[valid]
    y = y[valid]
    return X, y
