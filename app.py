from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, Response
import pandas as pd
import numpy as np
import uuid
import json
import os
from collections import deque
from datetime import datetime, timedelta
from pywebpush import webpush, WebPushException
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from datetime import datetime, timezone
import re
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from utils.data_store import DataStore
from utils.preprocess import validate_columns, required_columns, filter_dataframe, build_energy_dataset
from utils.analytics import detect_anomalies, analyze_seasonal_patterns, get_energy_insights, detect_consumption_shifts
from statsmodels.tsa.seasonal import seasonal_decompose
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from ml_models.classic import (
    train_and_predict as classic_train_predict,
    train_and_predict_prepared as classic_train_predict_prepared,
    fit_predict_cached,
    metrics as classic_metrics,
)
from ml_models.config import MODELS, CANDIDATE_MODEL_IDS
from sklearn.linear_model import LinearRegression

# Simple in-process caches
# Engineered dataset cache speeds up repeated feature builds
ENGINEERED_CACHE = {}
# Fitted model cache reuses estimators foKOr identical dataset slices
MODEL_CACHE = {}


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    try:
        if isinstance(obj, (np.floating, np.integer)):
            v = float(obj)
            return v if np.isfinite(v) else None
    except Exception:
        pass
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj

app = Flask(__name__)
app.secret_key = "change-this-secret-key"

# Initialize data store
data_store = DataStore()

# Set session lifetime to 1 day (in seconds)
app.permanent_session_lifetime = 86400

ALLOWED_EXTENSIONS = {"csv"}

# Live data storage (global, in-memory). Not session-scoped so ESP can push regardless of UI session.
# Keys by device id, default device "default". Each holds recent samples as a deque.
LIVE_BUFFER = {}
LIVE_MAXLEN = 2000


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_session_id() -> str:
    """Get or create a session ID for the current user."""
    try:
        if "sid" not in session:
            session.permanent = True  # Make the session persistent
            session["sid"] = str(uuid.uuid4())
            logger.info(f"Created new session: {session['sid']}")
        return session["sid"]
    except Exception as e:
        logger.error(f"Error in get_session_id: {str(e)}")
        # Return a fallback session ID if there's an error
        return "default-session-id"


@app.route("/")
def home():
    sid = get_session_id()
    df = data_store.get_dataframe(sid)
    stats = {
        "has_data": df is not None,
        "total_records": int(len(df)) if df is not None else 0,
        "total_appliances": int(df["Appliance"].nunique()) if df is not None and "Appliance" in df.columns else 0,
    }
    return render_template("index.html", stats=stats)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    sid = get_session_id()
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            flash("No file selected", "warning")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Unsupported file type. Please upload a CSV.", "danger")
            return redirect(request.url)
        try:
            df = pd.read_csv(file)
            ok, msg = validate_columns(df)
            if not ok:
                flash(msg, "danger")
                return redirect(request.url)
            # Ensure Timestamp to datetime
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
            data_store.set_dataframe(sid, df)
            preview = df.head(100)
            return render_template("preview.html", columns=list(preview.columns), rows=preview.values.tolist(), n=len(df))
        except Exception as e:
            flash(f"Failed to read CSV: {e}", "danger")
            return redirect(request.url)
    return render_template("upload.html", required_columns=required_columns())


@app.route("/visualize")
def visualize():
    sid = get_session_id()
    df = data_store.get_dataframe(sid)
    if df is None:
        flash("Please upload a dataset first.", "info")
        return redirect(url_for("upload"))
    appliances = sorted(df["Appliance"].dropna().unique().tolist()) if "Appliance" in df.columns else []
    return render_template("visualize.html", appliances=appliances)


# Live page (ESP real-time view)
@app.route("/live")
def live_page():
    return render_template("live.html")


def _get_device_buffer(device: str) -> deque:
    key = device or "esp32"
    if key not in LIVE_BUFFER:
        LIVE_BUFFER[key] = deque(maxlen=LIVE_MAXLEN)
    return LIVE_BUFFER[key]


# Ingest endpoint: accepts JSON or query params from ESP8266/ESP32
@app.route("/api/live/ingest", methods=["POST", "GET"])
def api_live_ingest():
    try:
        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
        else:
            payload = request.args.to_dict()
        device = str(payload.get("device", "esp32"))
        ts = payload.get("timestamp")
        try:
            # normalize timestamp to ISO8601
            if ts:
                ts_iso = pd.to_datetime(ts, errors="coerce")
                if pd.isna(ts_iso):
                    raise ValueError("bad ts")
                ts_str = ts_iso.tz_localize(None).isoformat()
            else:
                ts_str = datetime.utcnow().isoformat()
        except Exception:
            ts_str = datetime.utcnow().isoformat()
        sample = {
            "Timestamp": ts_str,
            "Voltage": _safe_float(payload.get("voltage")),
            "Current": _safe_float(payload.get("current")),
            "Power": _safe_float(payload.get("power")),
            "Energy": _safe_float(payload.get("energy")),
            "Device": device,
        }
        buf = _get_device_buffer(device)
        buf.append(sample)
        return jsonify({"status": "ok", "device": device, "count": len(buf)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


def _safe_float(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except Exception:
        return None


# Latest data for plotting; optional device and max_samples
@app.route("/api/live/latest")
def api_live_latest():
    device = request.args.get("device", "esp32")
    try:
        max_samples = int(request.args.get("max_samples", 500))
    except Exception:
        max_samples = 500
    buf = list(_get_device_buffer(device))
    if len(buf) == 0:
        return jsonify({"series": [], "device": device, "count": 0})
    frame = pd.DataFrame(buf[-max(1, max_samples):])
    def series_payload(col):
        if col not in frame.columns:
            return None
        s = pd.to_numeric(frame[col], errors="coerce")
        if s.notna().sum() == 0:
            return None
        return {
            "x": frame["Timestamp"].astype(str).tolist(),
            "y": s.fillna(np.nan).astype(float).tolist(),
            "name": col
        }
    cols = [c for c in ["Voltage", "Current", "Power", "Energy"] if c in frame.columns]
    series = [sp for c in cols if (sp := series_payload(c)) is not None]
    return jsonify({"series": series, "device": device, "count": int(len(frame))})


# Proxy pull: fetch ESP HTML (no firmware change), parse values, store, and return latest series
@app.route("/api/live/pull")
def api_live_pull():
    esp = request.args.get("esp")
    device = request.args.get("device", "esp32")
    if not esp:
        return jsonify({"error": "missing_esp"}), 400
    try:
        # Basic GET with short UA
        req = Request(esp, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        # Parse simple table values
        def find(pattern):
            m = re.search(pattern, html, re.IGNORECASE)
            return float(m.group(1)) if m else None
        v = find(r"Voltage\s*</th>\s*<td>\s*([0-9.]+)\s*V")
        c = find(r"Current\s*</th>\s*<td>\s*([0-9.]+)\s*mA")
        p = find(r"Power\s*</th>\s*<td>\s*([0-9.]+)\s*W")
        e = find(r"Energy\s*</th>\s*<td>\s*([0-9.]+)\s*kWh")
        sample = {
            "Timestamp": datetime.utcnow().isoformat(),
            "Voltage": v,
            "Current": c,
            "Power": p,
            "Energy": e,
            "Device": device,
        }
        _get_device_buffer(device).append(sample)
        # Return same payload as latest for convenience
        return api_live_latest()
    except (URLError, HTTPError, TimeoutError) as e:
        return jsonify({"error": "fetch_failed", "message": str(e) }), 502
    except Exception as e:
        return jsonify({"error": "parse_failed", "message": str(e)}), 500


@app.route("/api/data")
def api_data():
    sid = get_session_id()
    df = data_store.get_dataframe(sid)
    if df is None:
        return jsonify({"error": "no_data"}), 400

    appliance = request.args.get("appliance")
    start = request.args.get("start")
    end = request.args.get("end")

    fdf = filter_dataframe(df, appliance=appliance, start=start, end=end)
    # Optional: cap to most recent N samples
    ms_param = request.args.get("max_samples")
    try:
        max_samples = int(ms_param) if (ms_param is not None and str(ms_param).strip() != "") else None
    except Exception:
        max_samples = None
    if max_samples and len(fdf) > max_samples:
        fdf = fdf.tail(max_samples)

    def series_payload(col):
        return {
            "x": fdf["Timestamp"].astype(str).tolist(),
            "y": fdf[col].astype(float).tolist(),
            "name": col
        }

    cols = [c for c in ["Voltage", "Current", "Power", "Energy"] if c in fdf.columns]
    payload = {
        "series": [series_payload(c) for c in cols],
        "appliance": appliance,
        "count": int(len(fdf))
    }
    return jsonify(payload)



    def energy_total_kwh(frame: pd.DataFrame) -> float:
        # For per-appliance totals, prefer integrating Power to avoid accidentally using a whole-house cumulative meter
        if "Power" in frame.columns:
            p = pd.to_numeric(frame["Power"], errors="coerce").fillna(0.0)
            dt_h = infer_dt_hours(frame)
            if dt_h > 0:
                med = float(np.median(p)) if len(p) else 0.0
                power_kw = p / 1000.0 if med > 10 else p
                return float((power_kw * dt_h).sum())
        # Fallback: if only Energy is present, treat it as per-interval kWh and sum (do not use last-first within groups)
        if "Energy" in frame.columns:
            return float(pd.to_numeric(frame["Energy"], errors="coerce").fillna(0.0).sum())
        return 0.0

    rows = []
    for ap, g in fdf.groupby("Appliance"):
        e = energy_total_kwh(g)
        rows.append({
            "appliance": str(ap),
            "energy": float(e),
            "cost": float(e * uc)
        })
    rows.sort(key=lambda r: r["energy"], reverse=True)
    return jsonify({"rows": rows, "unit_cost": uc})


@app.route("/analytics")
def analytics_page():
    sid = get_session_id()
    df = data_store.get_dataframe(sid)
    if df is None or len(df) == 0:
        flash("Please upload data first", "warning")
        return redirect(url_for("upload"))
    return render_template("analytics.html")


@app.route("/predict")
def predict_page():
    sid = get_session_id()
    df = data_store.get_dataframe(sid)
    if df is None or len(df) == 0:
        flash("Please upload data first", "warning")
        return redirect(url_for("upload"))
    return render_template("predict.html", models=MODELS)


@app.route("/api/predict", methods=["POST"]) 
def api_predict():
    sid = get_session_id()
    df = data_store.get_dataframe(sid)
    if df is None:
        return jsonify({"error": "no_data"}), 400

    req = request.get_json(force=True)
    model_id = req.get("model", "auto")
    target = "Energy"
    appliance = req.get("appliance")
    start = req.get("start")
    end = req.get("end")
    unit_cost = req.get("unit_cost")  # cost per kWh (float)

    fdf = filter_dataframe(df, appliance=appliance, start=start, end=end)
    if target not in fdf.columns:
        return jsonify({"error": f"Target '{target}' not in columns"}), 400

    # Helper to align timestamps when using a test split (classic models)
    def tail_timestamps_for_length(frame: pd.DataFrame, length: int):
        if length <= 0:
            return []
        return frame.iloc[-length:]["Timestamp"].astype(str).tolist()

    # Energy helpers
    def infer_dt_hours(frame: pd.DataFrame) -> float:
        try:
            ts = pd.to_datetime(frame["Timestamp"])  # already datetime but safe
            deltas = ts.diff().dropna().dt.total_seconds()
            if len(deltas) == 0:
                return 0.0
            sec = float(deltas.median())
            return max(sec / 3600.0, 0.0)
        except Exception:
            return 0.0

    def looks_cumulative(series: pd.Series) -> bool:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) < 3:
            return False
        diffs = s.diff().dropna()
        nonneg_ratio = np.mean(diffs >= -1e-9)
        return nonneg_ratio > 0.95  # mostly non-decreasing

    def energy_total_kwh(frame: pd.DataFrame) -> float:
        # 1) If Energy column exists and looks cumulative, use last-first
        if "Energy" in frame.columns:
            s = pd.to_numeric(frame["Energy"], errors="coerce").dropna()
            if len(s) > 1 and looks_cumulative(s):
                delta = float(s.iloc[-1] - s.iloc[0])
                if delta >= 0:
                    return delta
        # 2) Else compute from Power if available
        if "Power" in frame.columns:
            p = pd.to_numeric(frame["Power"], errors="coerce").fillna(0.0)
            dt_h = infer_dt_hours(frame)
            if dt_h > 0:
                # Heuristic: if median power > 10, assume Watts; else kW
                med = float(np.median(p)) if len(p) else 0.0
                power_kw = p / 1000.0 if med > 10 else p
                return float((power_kw * dt_h).sum())
        # 3) Fallback to sum of Energy if present (assumed already in kWh per row)
        if "Energy" in frame.columns:
            return float(pd.to_numeric(frame["Energy"], errors="coerce").fillna(0.0).sum())
        return 0.0

    # Pre-build engineered dataset once (used by all models for speed), with caching
    last_ts = str(fdf["Timestamp"].iloc[-1]) if len(fdf) else ""
    cache_key = (get_session_id(), appliance or "", start or "", end or "", len(fdf), last_ts)
    cache_entry = ENGINEERED_CACHE.get(cache_key)
    if cache_entry is not None:
        X_all, y_all = cache_entry
    else:
        X_all, y_all = build_energy_dataset(fdf)
        ENGINEERED_CACHE[cache_key] = (X_all, y_all)

    # Run a single model id and return dict with results
    def run_model(mid: str):
        if mid in ("linear", "rf", "gboost", "xgb", "catboost"):
            # Try cached fitted estimator for this dataset slice
            model_ckey = (cache_key, mid, len(X_all))
            try:
                y_true, y_pred, fitted_model, feat_names = fit_predict_cached(X_all, y_all, mid, cache_key, MODEL_CACHE)
            except Exception:
                y_true, y_pred = classic_train_predict_prepared(X_all, y_all, mid)
                fitted_model, feat_names = None, list(X_all.columns)
            x = tail_timestamps_for_length(fdf, len(y_true))
        else:
            # Fallback: naive mean baseline
            y_true = fdf[target].astype(float).values
            y_pred = np.full_like(y_true, float(np.nanmean(y_true)) if len(y_true) else 0.0)
            x = fdf["Timestamp"].astype(str).tolist()
            fitted_model, feat_names = None, []
        m = classic_metrics(y_true.astype(float), y_pred.astype(float))
        # Anomalies via robust z using MAD
        anomalies = []
        try:
            if len(y_true) and len(y_pred):
                y_t = np.asarray(y_true, dtype=float)
                y_p = np.asarray(y_pred, dtype=float)
                resid = y_t - y_p
                med = np.nanmedian(resid)
                mad = np.nanmedian(np.abs(resid - med))
                denom = (mad * 1.4826) if mad > 0 else (np.nanstd(resid) + 1e-12)
                scores = np.abs(resid - med) / (denom if denom > 0 else 1e-12)
                k = int(min(10, len(scores)))
                top_idx = np.argsort(-scores)[:k]
                for i in top_idx:
                    anomalies.append({
                        "t": x[i] if i < len(x) else None,
                        "actual": float(y_t[i]),
                        "pred": float(y_p[i]),
                        "residual": float(resid[i]),
                        "score": float(scores[i]),
                    })
        except Exception:
            anomalies = []
        # Feature importance / coefficients
        importances = None
        try:
            if fitted_model is not None:
                if hasattr(fitted_model, "feature_importances_"):
                    vals = np.asarray(getattr(fitted_model, "feature_importances_"), dtype=float)
                    importances = sorted([
                        {"name": feat_names[i], "value": float(vals[i])}
                        for i in range(min(len(vals), len(feat_names)))
                    ], key=lambda d: abs(d["value"]), reverse=True)[:10]
                elif hasattr(fitted_model, "coef_"):
                    coef = np.asarray(getattr(fitted_model, "coef_"), dtype=float)
                    if coef.ndim > 1:
                        coef = coef.ravel()
                    importances = sorted([
                        {"name": feat_names[i], "value": float(coef[i])}
                        for i in range(min(len(coef), len(feat_names)))
                    ], key=lambda d: abs(d["value"]), reverse=True)[:10]
        except Exception:
            importances = None
        # Summaries will be computed on the evaluation (test) window to match predictions
        # Identify the test window frame based on y_true length
        eval_len = len(y_true)
        eval_frame = fdf.tail(eval_len) if eval_len > 0 else fdf.iloc[0:0]
        eval_start = str(eval_frame["Timestamp"].iloc[0]) if eval_len > 0 else None
        eval_end = str(eval_frame["Timestamp"].iloc[-1]) if eval_len > 0 else None
        # Full series used to plot train vs test
        x_all_full = fdf["Timestamp"].astype(str).tolist()
        y_all_full = fdf[target].astype(float).tolist()
        test_start_index = max(0, len(fdf) - eval_len)
        actual_total_energy = energy_total_kwh(eval_frame)
        predicted_total_energy = None
        # 1) If predicting Energy directly (our engineered target is per-interval kWh)
        if target == "Energy" and len(y_pred):
            predicted_total_energy = float(np.nansum(y_pred))
        # 1b) If predicting Power, convert to energy over the test window
        if predicted_total_energy is None and target == "Power" and len(y_pred):
            test_len = len(y_pred)
            test_frame = fdf.tail(test_len)
            dt_h = infer_dt_hours(test_frame)
            if dt_h > 0:
                # Heuristic units W vs kW based on median of observed Power in the same window
                p_obs = pd.to_numeric(test_frame["Power"], errors="coerce").fillna(0.0) if "Power" in test_frame.columns else pd.Series([0])
                med = float(np.median(p_obs)) if len(p_obs) else 0.0
                power_kw_pred = np.array(y_pred, dtype=float) / 1000.0 if med > 10 else np.array(y_pred, dtype=float)
                predicted_total_energy = float((power_kw_pred * dt_h).sum())
        # 2) Otherwise, learn simple linear mapping from target to Energy on aligned test window
        test_len = len(y_true)
        if test_len > 0:
            test_idx = fdf.index[-test_len:]
            # Predict Energy from target
            if predicted_total_energy is None and "Energy" in fdf.columns:
                try:
                    X_te = fdf.loc[test_idx, target].astype(float).values.reshape(-1, 1)
                    y_te_energy = fdf.loc[test_idx, "Energy"].astype(float).values
                    if np.isfinite(X_te).all() and np.isfinite(y_te_energy).all():
                        reg_e = LinearRegression()
                        reg_e.fit(X_te, y_te_energy)
                        pred_energy_series = reg_e.predict(np.array(y_pred, dtype=float).reshape(-1, 1))
                        predicted_total_energy = float(np.nansum(pred_energy_series))
                except Exception:
                    pass
        # Unit cost: default to 8 if not provided, do not infer from dataset
        try:
            uc = float(unit_cost) if unit_cost is not None and str(unit_cost).strip() != "" else 8.0
        except Exception:
            uc = 8.0
        # Costs computed strictly from energy and unit cost on the same evaluation window
        actual_estimated_cost = float(actual_total_energy * uc) if (actual_total_energy is not None) else None
        predicted_estimated_cost = float(predicted_total_energy * uc) if (predicted_total_energy is not None) else None

        summaries = {
            "actual_total_energy": actual_total_energy,
            "predicted_total_energy": predicted_total_energy,
            "unit_cost": uc,
            "actual_estimated_cost": actual_estimated_cost,
            "predicted_estimated_cost": predicted_estimated_cost,
            "count": int(len(fdf))
        }
        return {"model": mid, "target": target, "x": x, "y_true": y_true.tolist(), "y_pred": y_pred.tolist(), "metrics": m, "summaries": summaries, "anomalies": anomalies, "importances": importances, "eval_start": eval_start, "eval_end": eval_end, "x_all": x_all_full, "y_all": y_all_full, "test_start_index": int(test_start_index)}

    candidate_models = list(CANDIDATE_MODEL_IDS)
    results = {}

    if model_id == "auto":
        # Evaluate all models and pick the best by RMSE
        for mid in candidate_models:
            results[mid] = run_model(mid)
        # pick best by RMSE (lower is better); if RMSE None, treat as inf
        def rmse_or_inf(m):
            v = results[m]["metrics"].get("RMSE")
            return v if isinstance(v, (int, float)) else float("inf")
        best = min(candidate_models, key=rmse_or_inf)
        best_payload = results[best]
        best_payload["best_model"] = best
        best_payload["metrics_all"] = {k: v["metrics"] for k, v in results.items()}
        best_payload["summaries_all"] = {k: v.get("summaries") for k, v in results.items()}
        return jsonify(_sanitize_for_json(best_payload))
    else:
        # Run requested model and also compute best model summary in background for display
        selected_payload = run_model(model_id)
        for mid in candidate_models:
            if mid not in results:
                results[mid] = run_model(mid)
        def rmse_or_inf2(m):
            v = results[m]["metrics"].get("RMSE")
            return v if isinstance(v, (int, float)) else float("inf")
        best = min(candidate_models, key=rmse_or_inf2)
        selected_payload["best_model"] = best
        selected_payload["metrics_all"] = {k: v["metrics"] for k, v in results.items()}
        selected_payload["summaries_all"] = {k: v.get("summaries") for k, v in results.items()}
        return jsonify(_sanitize_for_json(selected_payload))


@app.route("/api/forecast", methods=["POST"]) 
def api_forecast():
    sid = get_session_id()
    df = data_store.get_dataframe(sid)
    if df is None:
        return jsonify({"error": "no_data"}), 400

    req = request.get_json(force=True)
    appliance = req.get("appliance")
    start = req.get("start")
    end = req.get("end")
    try:
        horizon = int(req.get("horizon", 60))
    except Exception:
        horizon = 60
    try:
        max_samples = int(req.get("max_samples", 2000))
    except Exception:
        max_samples = 2000

    fdf = filter_dataframe(df, appliance=appliance, start=start, end=end)
    if max_samples and len(fdf) > max_samples:
        fdf = fdf.tail(max_samples)
    if "Timestamp" not in fdf.columns:
        return jsonify({"error": "no_timestamp"}), 400

    # Build engineered dataset to get per-interval energy target y_all
    X_all, y_all = build_energy_dataset(fdf)
    # infer dt from filtered frame
    try:
        ts = pd.to_datetime(fdf["Timestamp"])  # already datetime
        deltas = ts.diff().dropna().dt.total_seconds()
        sec = float(deltas.median()) if len(deltas) else 60.0
    except Exception:
        sec = 60.0
    if not np.isfinite(sec) or sec <= 0:
        sec = 60.0
    # future timestamps
    last_ts = pd.to_datetime(fdf["Timestamp"].iloc[-1]) if len(fdf) else pd.Timestamp.utcnow()
    future_ts = [ (last_ts + pd.to_timedelta((i+1)*sec, unit="s")).isoformat() for i in range(max(0, horizon)) ]
    # simple baseline forecast: recent rolling mean of y_all
    window = int(min(60, max(1, len(y_all))))
    baseline = float(np.nanmean(y_all.tail(window))) if window > 0 else 0.0
    y_fc = [baseline for _ in range(max(0, horizon))]
    return jsonify({
        "x_future": future_ts,
        "y_forecast": y_fc,
        "dt_seconds": sec,
        "window_used": window
    })



@app.route("/api/analytics/anomalies", methods=["POST"])
def api_analytics_anomalies():
    """Detect anomalies in the time series data."""
    sid = get_session_id()
    df = data_store.get_dataframe(sid)
    
    if df is None or len(df) == 0:
        return jsonify({"error": "No data available"}), 400
    
    try:
        # Get parameters
        data = request.get_json() or {}
        target = data.get("target", "Energy")
        contamination = float(data.get("contamination", 0.1))
        
        if target not in df.columns:
            return jsonify({"error": f"Target column '{target}' not found"}), 400
        
        # Convert index to datetime if it's not already
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Timestamp" in df.columns:
                df = df.set_index("Timestamp")
            else:
                return jsonify({"error": "No timestamp column found"}), 400
        
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort by time
        df = df.sort_index()
        
        # Run anomaly detection
        result = detect_anomalies(df[target], contamination=contamination)
        
        return jsonify({
            "status": "success",
            "anomaly_count": len(result["indices"]),
            "anomaly_points": result["anomaly_points"],
            "scores": result["scores"]
        })
    except Exception as e:
        app.logger.error(f"Error in anomaly detection: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/analytics/insights")
def api_analytics_insights():
    """Get insights from the energy data."""
    try:
        sid = get_session_id()
        df = data_store.get_dataframe(sid)
        
        if df is None or len(df) == 0:
            return jsonify({
                "status": "error",
                "message": "No data available. Please upload data first."
            }), 400

        # Use first numeric column as target if Energy not found
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return jsonify({
                "status": "error",
                "message": "No numeric data columns found"
            }), 400
            
        target_col = 'Energy' if 'Energy' in df.columns else numeric_cols[0]
        
        # Prepare time series
        ts = df[target_col]
        if 'Timestamp' in df.columns or 'timestamp' in [col.lower() for col in df.columns]:
            ts = df.set_index('Timestamp' if 'Timestamp' in df.columns else 'timestamp')[target_col]
        
        # Convert index to datetime if it's not already
        ts.index = pd.to_datetime(ts.index, errors='coerce')
        ts = ts.sort_index().dropna()
        
        if len(ts) == 0:
            return jsonify({
                "status": "error",
                "message": "No valid time series data available after cleaning"
            }), 400

        # Basic stats
        stats = {
            'mean': float(ts.mean()),
            'median': float(ts.median()),
            'std': float(ts.std()) if len(ts) > 1 else 0,
            'min': float(ts.min()),
            'max': float(ts.max()),
            'count': int(len(ts))
        }
        
        # Patterns
        patterns = {}
        try:
            if hasattr(ts.index, 'hour'):
                patterns['peak_hour'] = int(ts.groupby(ts.index.hour).mean().idxmax())
        except:
            pass
            
        # Anomaly detection
        anomaly_count = 0
        try:
            anomaly_result = detect_anomalies(ts, contamination=0.1)
            anomaly_count = len(anomaly_result.get("indices", []))
        except Exception as e:
            app.logger.error(f"Error in anomaly detection: {str(e)}")
            anomaly_count = 0
            
        # Prepare trend data (daily resampled)
        trend_data = {
            'labels': [],
            'values': []
        }
        try:
            ts_daily = ts.resample('D').mean()
            trend_data = {
                'labels': ts_daily.index.strftime('%Y-%m-%d').tolist(),
                'values': ts_daily.tolist()
            }
        except Exception as e:
            app.logger.error(f"Error preparing trend data: {str(e)}")

        # Hourly pattern
        hourly_pattern = None
        try:
            if hasattr(ts.index, 'hour'):
                hourly_avg = ts.groupby(ts.index.hour).mean()
                hourly_pattern = {
                    'hours': hourly_avg.index.tolist(),
                    'values': hourly_avg.tolist()
                }
        except Exception as e:
            app.logger.error(f"Error calculating hourly pattern: {str(e)}")

        # Day of week pattern
        day_of_week = None
        try:
            if hasattr(ts.index, 'dayofweek'):
                day_avg = ts.groupby(ts.index.dayofweek).mean()
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_of_week = {
                    'days': [days[i] for i in day_avg.index],
                    'values': day_avg.tolist()
                }
        except Exception as e:
            app.logger.error(f"Error calculating day of week pattern: {str(e)}")

        # Energy distribution
        distribution = None
        try:
            hist, bin_edges = np.histogram(ts, bins=10)
            distribution = {
                'bins': hist.tolist(),
                'ranges': bin_edges.tolist()
            }
        except Exception as e:
            app.logger.error(f"Error calculating energy distribution: {str(e)}")

        return jsonify({
            "status": "success",
            "insights": {
                "stats": stats,
                "patterns": patterns,
                "anomaly_count": anomaly_count,
                "trend_data": trend_data,
                "hourly_pattern": hourly_pattern,
                "day_of_week": day_of_week,
                "distribution": distribution
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error in insights generation: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Error processing data: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
