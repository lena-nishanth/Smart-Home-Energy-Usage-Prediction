# Stubs for classic ML models; to be implemented
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils.preprocess import build_energy_dataset


def _make_model(model_id: str):
    if model_id == "linear":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression())
        ])
    elif model_id == "rf":
        return RandomForestRegressor(
            n_estimators=150,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=2
        )
    elif model_id == "gboost":
        return GradientBoostingRegressor(
            learning_rate=0.05,
            n_estimators=300,
            max_depth=3,
            random_state=42
        )
    elif model_id == "xgb":
        return XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=2,
            reg_lambda=1.0
        )
    elif model_id == "catboost":
        return CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            verbose=False,
            random_state=42
        )
    else:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression())
        ])


def train_and_predict_prepared(X_all: pd.DataFrame, y_all: pd.Series, model_id: str) -> Tuple[np.ndarray, np.ndarray]:
    if len(X_all) < 30:
        return y_all.values, np.full_like(y_all.values, float(np.nanmean(y_all.values)) if len(y_all) else 0.0)

    # time-based split
    split_idx = int(len(X_all) * 0.8)
    X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_train, y_test = y_all.iloc[:split_idx].values, y_all.iloc[split_idx:].values

    model = _make_model(model_id)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred


def train_and_predict_with_model(X_all: pd.DataFrame, y_all: pd.Series, model_id: str) -> Tuple[np.ndarray, np.ndarray, Any, list[str]]:
    """Like train_and_predict_prepared but also returns fitted model and feature names."""
    if len(X_all) < 30:
        dummy_pred = np.full_like(y_all.values, float(np.nanmean(y_all.values)) if len(y_all) else 0.0)
        return y_all.values, dummy_pred, None, list(X_all.columns)

    split_idx = int(len(X_all) * 0.8)
    X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_train, y_test = y_all.iloc[:split_idx].values, y_all.iloc[split_idx:].values

    model = _make_model(model_id)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred, model, list(X_all.columns)


def train_and_predict(df: pd.DataFrame, target: str, model_id: str) -> Tuple[np.ndarray, np.ndarray]:
    # Backwards-compatible entry: build dataset then delegate to prepared path
    X_all, y_all = build_energy_dataset(df)
    return train_and_predict_prepared(X_all, y_all, model_id)


def fit_predict_cached(X_all: pd.DataFrame, y_all: pd.Series, model_id: str, cache_key: str, model_cache: dict) -> Tuple[np.ndarray, np.ndarray, Any, list[str]]:
    """Train once per dataset+model and reuse fitted estimator if available. Returns (y_test, y_pred, model, feature_names)."""
    if len(X_all) < 30:
        dummy_pred = np.full_like(y_all.values, float(np.nanmean(y_all.values)) if len(y_all) else 0.0)
        return y_all.values, dummy_pred, None, list(X_all.columns)

    split_idx = int(len(X_all) * 0.8)
    X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_train, y_test = y_all.iloc[:split_idx].values, y_all.iloc[split_idx:].values

    key = (cache_key, model_id, len(X_all), split_idx)
    model = model_cache.get(key)
    if model is None:
        model = _make_model(model_id)
        model.fit(X_train, y_train)
        model_cache[key] = model
    y_pred = model.predict(X_test)
    return y_test, y_pred, model, list(X_all.columns)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"MAE": None, "RMSE": None, "R2": None, "MAPE": None}

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"MAE": None, "RMSE": None, "R2": None, "MAPE": None}

    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0 or len(yp) == 0:
        return {"MAE": None, "RMSE": None, "R2": None, "MAPE": None}

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))
    nonzero = yt != 0
    mape = float(np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero]) * 100)) if np.any(nonzero) else None
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
