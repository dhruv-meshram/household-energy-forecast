"""
⚡ Energy Demand Forecasting Dashboard
Production-grade Streamlit app using pre-trained LSTM model
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import joblib
import warnings
from datetime import timedelta, datetime
from tensorflow import keras

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Energy Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark Glassmorphism Theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import Google Font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* Global */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Dark background */
  .stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    color: #e6edf3;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid rgba(48, 54, 61, 0.8);
  }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: rgba(22, 27, 34, 0.85);
    border: 1px solid rgba(48, 54, 61, 0.8);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
  }

  /* Metric label */
  div[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* Metric value */
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #58a6ff !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
  }

  /* Metric delta */
  div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
  }

  /* Section headers */
  .section-header {
    background: linear-gradient(90deg, rgba(88, 166, 255, 0.15) 0%, rgba(88, 166, 255, 0) 100%);
    border-left: 3px solid #58a6ff;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 20px 0 16px 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: 0.02em;
  }

  /* Glass card */
  .glass-card {
    background: rgba(22, 27, 34, 0.75);
    border: 1px solid rgba(48, 54, 61, 0.8);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(12px);
    margin-bottom: 16px;
  }

  /* Info badge */
  .badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .badge-blue  { background: rgba(88,166,255,0.15); color: #58a6ff; border: 1px solid rgba(88,166,255,0.3); }
  .badge-green { background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }
  .badge-amber { background: rgba(210,153,34,0.15); color: #d2991f; border: 1px solid rgba(210,153,34,0.3); }
  .badge-red   { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid rgba(248,81,73,0.3); }

  /* Selectbox / slider */
  div[data-baseweb="select"] > div {
    background-color: #161b22 !important;
    border-color: rgba(48,54,61,0.8) !important;
    color: #e6edf3 !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(31,111,235,0.3);
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(31,111,235,0.5);
  }

  /* Tabs */
  button[data-baseweb="tab"] {
    color: #8b949e !important;
    font-weight: 500;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
  }

  /* Divider */
  hr { border-color: rgba(48,54,61,0.6); }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }

  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, rgba(31,111,235,0.2) 0%, rgba(88,166,255,0.1) 50%, rgba(63,185,80,0.1) 100%);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(88,166,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #79c0ff, #3fb950);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
  }
  .hero-subtitle {
    color: #8b949e;
    font-size: 0.95rem;
    margin: 0;
  }

  /* Forecast horizon pills */
  .horizon-pill {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 4px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  /* Spinner override */
  .stSpinner > div { border-top-color: #58a6ff !important; }

  /* Expander */
  details summary {
    color: #58a6ff !important;
    font-weight: 500;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
# Prefer the portable .h5 format (cross-TF-version compatible);
# fall back to .keras if .h5 hasn't been generated yet.
import os as _os
_H5_PATH    = "lstm_forecasting_model.h5"
_KERAS_PATH = "lstm_forecasting_model.keras"
MODEL_PATH    = _H5_PATH if _os.path.exists(_H5_PATH) else _KERAS_PATH
SCALER_X_PATH = "scaler_X_forecasting.pkl"
SCALER_Y_PATH = "scaler_y_forecasting.pkl"
FEATURES_PATH = "feature_list_forecasting.json"
DATA_PATH     = "sample_data_2026.csv"
METRICS_PATH  = "forecasting_model_metrics.csv"

TIME_STEPS   = 12
INTERVAL_MIN = 5

HORIZONS = {
    "6 hr":  6  * 60 // INTERVAL_MIN,
    "12 hr": 12 * 60 // INTERVAL_MIN,
    "24 hr": 24 * 60 // INTERVAL_MIN,
}

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,17,23,0.6)",
    font=dict(family="Inter", color="#e6edf3"),
    xaxis=dict(gridcolor="rgba(48,54,61,0.5)", linecolor="rgba(48,54,61,0.8)"),
    yaxis=dict(gridcolor="rgba(48,54,61,0.5)", linecolor="rgba(48,54,61,0.8)"),
)

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_scalers():
    # ── Robust model loader ───────────────────────────────────────────────
    # .h5 (HDF5) stores weights by index → immune to layer-path naming
    # changes between TF versions (Bidirectional LSTM weight name issue).
    # .keras stores weights by path → breaks across TF minor versions.
    # Strategy: try plain load first; on any weight/kwarg error, retry
    # with CompatBiLSTM that strips unknown kwargs from both LSTM and
    # Bidirectional layers.
    def _compat_loader(path):
        """Load with custom objects that tolerate unknown LSTM kwargs."""
        class CompatLSTMCell(keras.layers.LSTMCell):
            def __init__(self, *args, **kwargs):
                kwargs.pop("time_major", None)
                super().__init__(*args, **kwargs)

        class CompatLSTM(keras.layers.LSTM):
            def __init__(self, *args, **kwargs):
                kwargs.pop("time_major", None)
                super().__init__(*args, **kwargs)

        class CompatBiLSTM(keras.layers.Bidirectional):
            def __init__(self, *args, **kwargs):
                kwargs.pop("time_major", None)
                super().__init__(*args, **kwargs)

        return keras.models.load_model(
            path,
            custom_objects={
                "LSTM": CompatLSTM,
                "LSTMCell": CompatLSTMCell,
                "Bidirectional": CompatBiLSTM,
            },
        )

    def _load_model_safe(path):
        # First attempt: plain load (works when TF versions match exactly)
        try:
            return keras.models.load_model(path)
        except Exception as e1:
            err = str(e1).lower()
            # Only retry on known version-mismatch errors
            if not any(k in err for k in [
                "time_major", "unrecognized keyword",
                "expected 3 variables", "expected 0 variables",
                "lstm_cell", "variable",
            ]):
                raise  # unrelated error — surface it

        # Second attempt: compat loader with custom objects
        try:
            return _compat_loader(path)
        except Exception as e2:
            raise RuntimeError(
                f"Could not load model from '{path}'.\n"
                f"First error : {e1}\n"
                f"Second error: {e2}"
            ) from e2

    model    = _load_model_safe(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    with open(FEATURES_PATH) as f:
        feature_cols = json.load(f)
    return model, scaler_X, scaler_y, feature_cols


@st.cache_data(show_spinner=False)
def load_data():
    raw = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    raw.set_index("datetime", inplace=True)
    raw.sort_index(inplace=True)
    raw["Global_active_power"] = raw["Global_active_power"] * 1000  # kW → W
    return raw


@st.cache_data(show_spinner=False)
def load_metrics():
    return pd.read_csv(METRICS_PATH, index_col=0)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    target = "Global_active_power"
    df["hour"]       = df.index.hour
    df["day"]        = df.index.dayofweek
    df["month"]      = df.index.month
    df["is_weekend"] = df["day"].isin([5, 6]).astype(int)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]  / 24)
    df["day_sin"]    = np.sin(2 * np.pi * df["day"]   / 7)
    df["day_cos"]    = np.cos(2 * np.pi * df["day"]   / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["power_lag_1h"]    = df[target].shift(12)
    df["power_lag_3h"]    = df[target].shift(36)
    df["power_lag_6h"]    = df[target].shift(72)
    df["power_lag_12h"]   = df[target].shift(144)
    df["power_lag_1day"]  = df[target].shift(288)
    df["power_lag_1week"] = df[target].shift(2016)
    df["power_rolling_mean_3h"]  = df[target].shift(1).rolling(36).mean()
    df["power_rolling_mean_6h"]  = df[target].shift(1).rolling(72).mean()
    df["power_rolling_mean_24h"] = df[target].shift(1).rolling(288).mean()
    df["power_rolling_std_3h"]   = df[target].shift(1).rolling(36).std()
    df["power_rolling_std_24h"]  = df[target].shift(1).rolling(288).std()
    df["power_rolling_min_24h"]  = df[target].shift(1).rolling(288).min()
    df["power_rolling_max_24h"]  = df[target].shift(1).rolling(288).max()
    df.dropna(inplace=True)
    return df


def build_feature_row(ts, power_history: pd.Series) -> np.ndarray:
    def lag(minutes):
        target_ts = ts - timedelta(minutes=minutes)
        idx = power_history.index.get_indexer([target_ts], method="nearest")[0]
        return float(power_history.iloc[idx])

    def rolling_stat(minutes, stat):
        cutoff = ts - timedelta(minutes=minutes)
        window = power_history[power_history.index >= cutoff]
        if len(window) == 0:
            return float(power_history.mean())
        if stat == "mean": return float(window.mean())
        if stat == "std":  return float(window.std()) if len(window) > 1 else 0.0
        if stat == "min":  return float(window.min())
        if stat == "max":  return float(window.max())

    h, d, m = ts.hour, ts.dayofweek, ts.month
    return np.array([
        np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
        np.sin(2*np.pi*d/7),  np.cos(2*np.pi*d/7),
        np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12),
        1.0 if d >= 5 else 0.0,
        lag(60), lag(180), lag(360), lag(720), lag(1440), lag(10080),
        rolling_stat(180, "mean"), rolling_stat(360, "mean"),
        rolling_stat(1440, "mean"), rolling_stat(180, "std"),
        rolling_stat(1440, "std"), rolling_stat(1440, "min"),
        rolling_stat(1440, "max"),
    ], dtype=np.float32)


def recursive_forecast(model, scaler_X, scaler_y, df_history, feature_cols,
                       n_steps, time_steps=12, interval_min=5):
    seed_X = df_history[feature_cols].values[-time_steps:]
    seed_X_scaled = scaler_X.transform(seed_X)
    window = list(seed_X_scaled)
    power_history = df_history["Global_active_power"].copy()
    last_ts = df_history.index[-1]
    predictions = []

    progress = st.progress(0)
    for step in range(n_steps):
        X_input = np.array(window[-time_steps:]).reshape(1, time_steps, -1)
        y_scaled = model.predict(X_input, verbose=0)
        y_watts  = float(scaler_y.inverse_transform(y_scaled)[0, 0])
        y_watts  = max(0.0, y_watts)
        next_ts  = last_ts + timedelta(minutes=interval_min)
        predictions.append({"datetime": next_ts, "predicted_power_W": y_watts})
        power_history.loc[next_ts] = y_watts
        new_feat = build_feature_row(next_ts, power_history)
        new_feat_scaled = scaler_X.transform(new_feat.reshape(1, -1))[0]
        window.append(new_feat_scaled)
        last_ts = next_ts
        progress.progress((step + 1) / n_steps)

    progress.empty()
    return pd.DataFrame(predictions).set_index("datetime")


def watts_to_kw(series):
    return series / 1000


def fmt_power(w):
    if w >= 1000:
        return f"{w/1000:.2f} kW"
    return f"{w:.0f} W"


def get_trend_color(val, ref):
    return "#3fb950" if val >= ref else "#f85149"


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
      <div style='font-size:2.5rem;'>⚡</div>
      <div style='font-size:1.1rem; font-weight:700; color:#58a6ff;'>EnergyForecast</div>
      <div style='font-size:0.72rem; color:#8b949e; margin-top:4px;'>LSTM · 2026 Edition</div>
    </div>
    <hr style='margin:12px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Forecast Settings")

    horizon_label = st.selectbox(
        "Forecast Horizon",
        options=list(HORIZONS.keys()),
        index=2,
        help="How far ahead to forecast",
    )
    n_steps = HORIZONS[horizon_label]

    show_confidence = st.checkbox("Show Confidence Band", value=True)
    show_history    = st.checkbox("Show Historical Context", value=True)
    history_hours   = st.slider("History Window (hours)", 6, 72, 24, step=6)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📊 Display Options")
    power_unit = st.radio("Power Unit", ["Watts (W)", "Kilowatts (kW)"], index=1)
    use_kw = power_unit == "Kilowatts (kW)"

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#8b949e; text-align:center;'>
      Model: LSTM · Trained 2006–2010<br>
      Forecast: Recursive Autoregressive<br>
      Features: 20 temporal + lag + rolling
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading model and data…"):
    try:
        model, scaler_X, scaler_y, feature_cols = load_model_and_scalers()
        raw = load_data()
        metrics_df = load_metrics()
        df = build_features(raw)
        load_ok = True
    except Exception as e:
        st.error(f"❌ Failed to load resources: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero-banner'>
  <div class='hero-title'>⚡ Energy Demand Forecasting</div>
  <p class='hero-subtitle'>
    LSTM-powered recursive forecasting · {len(feature_cols)} engineered features ·
    Data window: {raw.index[0].strftime('%b %d, %Y')} → {raw.index[-1].strftime('%b %d, %Y')}
    &nbsp;|&nbsp; <span class='badge badge-blue'>{horizon_label} horizon</span>
    &nbsp;<span class='badge badge-green'>{n_steps} steps</span>
    &nbsp;<span class='badge badge-amber'>5-min intervals</span>
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_forecast, tab_eda, tab_model, tab_raw = st.tabs([
    "🔮 Forecast",
    "📈 Data Explorer",
    "🧠 Model Performance",
    "📋 Raw Data",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — FORECAST
# ═══════════════════════════════════════════════════════════════

def parse_uploaded_csv(uploaded_file):
    """
    Parse an uploaded CSV into the same format as sample_data_2026.csv.
    Expected columns: datetime, Global_active_power (in kW — will be converted to W),
    plus optionally the other sensor columns.
    Returns (raw_df, error_string_or_None).
    """
    try:
        up_df = pd.read_csv(uploaded_file, parse_dates=["datetime"])
    except Exception as e:
        return None, f"Could not read CSV: {e}"

    if "datetime" not in up_df.columns:
        return None, "CSV must have a 'datetime' column."
    if "Global_active_power" not in up_df.columns:
        return None, "CSV must have a 'Global_active_power' column."

    up_df.set_index("datetime", inplace=True)
    up_df.sort_index(inplace=True)
    up_df = up_df.dropna(subset=["Global_active_power"])

    # Detect unit: if values look like kW (< 20 typical household max kW) convert to W
    if up_df["Global_active_power"].median() < 50:
        up_df["Global_active_power"] = up_df["Global_active_power"] * 1000

    # Minimum rows = largest lag (1 week = 2016 steps) + LSTM window (12 steps).
    # The 24h rolling window (288) is fully contained within the 1-week lag window,
    # so it does NOT need to be added separately.
    min_rows_needed = 2016 + 12
    if len(up_df) < min_rows_needed:
        return None, (
            f"Need at least {min_rows_needed} rows "
            f"(~{min_rows_needed * 5 // 60} hours of 5-min data) "
            f"so the model has a full 1-week lookback window. "
            f"Uploaded file has only {len(up_df)} rows."
        )
    return up_df, None


with tab_forecast:
    # ── Top action bar ────────────────────────────────────────
    st.markdown("<div class='section-header'>📂 Data Source</div>", unsafe_allow_html=True)

    dl_col, up_col, run_col = st.columns([1, 2, 1])

    # ── Download sample data ──────────────────────────────────
    with dl_col:
        with open(DATA_PATH, "rb") as _f:
            sample_bytes = _f.read()
        st.download_button(
            label="⬇ Download Sample CSV",
            data=sample_bytes,
            file_name="sample_data_2026.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download the default 2026 lookback dataset to use as a template.",
        )
        st.markdown("""
        <div style='font-size:0.72rem; color:#8b949e; margin-top:6px; text-align:center;'>
          Required columns:<br>
          <code style='color:#58a6ff;'>datetime</code>,
          <code style='color:#58a6ff;'>Global_active_power</code><br>
          (5-min intervals, ≥ 1 week of data)
        </div>
        """, unsafe_allow_html=True)

    # ── Upload custom data ────────────────────────────────────
    with up_col:
        uploaded_file = st.file_uploader(
            "Upload your own CSV (optional)",
            type=["csv"],
            help="Upload a CSV with the same format as the sample. Must have 'datetime' and 'Global_active_power' columns with at least 1 week of 5-min data.",
            label_visibility="visible",
        )

        if uploaded_file is not None:
            up_raw, up_err = parse_uploaded_csv(uploaded_file)
            if up_err:
                st.error(f"❌ {up_err}")
                active_raw = raw          # fall back to default
                upload_ok  = False
            else:
                active_raw = up_raw
                upload_ok  = True
                st.success(
                    f"✅ Loaded **{len(active_raw):,}** rows · "
                    f"{active_raw.index[0].strftime('%b %d %H:%M')} → "
                    f"{active_raw.index[-1].strftime('%b %d %H:%M, %Y')}"
                )
        else:
            active_raw = raw
            upload_ok  = True

    # Re-build features from whichever dataset is active
    active_df = build_features(active_raw)

    # ── Run button + info card ────────────────────────────────
    with run_col:
        run_forecast = st.button("▶ Run Forecast", use_container_width=True)

    forecast_start = active_df.index[-1] + timedelta(minutes=INTERVAL_MIN)
    forecast_end   = forecast_start + timedelta(minutes=INTERVAL_MIN * n_steps - INTERVAL_MIN)
    st.markdown(f"""
    <div class='glass-card' style='padding:12px 16px; margin:4px 0 0 0;'>
      <span style='color:#8b949e; font-size:0.8rem;'>Forecast window:</span>
      <strong style='color:#58a6ff;'> {forecast_start.strftime('%b %d %H:%M')} → {forecast_end.strftime('%b %d %H:%M, %Y')}</strong>
      &nbsp;&nbsp;
      <span style='color:#8b949e; font-size:0.8rem;'>Lookback ends:</span>
      <strong style='color:#e6edf3;'> {active_df.index[-1].strftime('%b %d %H:%M, %Y')}</strong>
      &nbsp;&nbsp;
      <span style='color:#8b949e; font-size:0.8rem;'>Source:</span>
      <strong style='color:{'#3fb950' if uploaded_file else '#8b949e'};'>
        {'📤 Uploaded file' if uploaded_file and upload_ok else '📁 Default sample'}
      </strong>
    </div>
    """, unsafe_allow_html=True)

    if run_forecast:
        with st.spinner(f"Running {horizon_label} recursive forecast ({n_steps} steps)…"):
            preds = recursive_forecast(
                model, scaler_X, scaler_y, active_df, feature_cols,
                n_steps=n_steps, time_steps=TIME_STEPS, interval_min=INTERVAL_MIN,
            )

        if use_kw:
            preds["value"] = watts_to_kw(preds["predicted_power_W"])
            unit_label = "kW"
        else:
            preds["value"] = preds["predicted_power_W"]
            unit_label = "W"

        # ── KPI Row ──────────────────────────────────────────
        st.markdown("<div class='section-header'>📊 Forecast Summary</div>", unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)

        mean_val = preds["value"].mean()
        max_val  = preds["value"].max()
        min_val  = preds["value"].min()
        std_val  = preds["value"].std()
        peak_ts  = preds["value"].idxmax()

        # Compare to historical mean
        hist_mean = watts_to_kw(raw["Global_active_power"]).mean() if use_kw else raw["Global_active_power"].mean()
        delta_pct = (mean_val - hist_mean) / hist_mean * 100

        k1.metric("Mean Demand",   f"{mean_val:.2f} {unit_label}", f"{delta_pct:+.1f}% vs hist.")
        k2.metric("Peak Demand",   f"{max_val:.2f} {unit_label}")
        k3.metric("Min Demand",    f"{min_val:.2f} {unit_label}")
        k4.metric("Std Deviation", f"{std_val:.2f} {unit_label}")
        k5.metric("Peak Time",     peak_ts.strftime("%H:%M"), peak_ts.strftime("%b %d"))

        # ── Main Forecast Chart ───────────────────────────────
        st.markdown("<div class='section-header'>🔮 Forecast Visualization</div>", unsafe_allow_html=True)

        fig = go.Figure()

        # Historical context
        if show_history:
            hist_window = history_hours * 60 // INTERVAL_MIN
            hist_slice  = active_df["Global_active_power"].iloc[-hist_window:]
            hist_vals   = watts_to_kw(hist_slice) if use_kw else hist_slice

            fig.add_trace(go.Scatter(
                x=hist_vals.index,
                y=hist_vals.values,
                name="Historical",
                line=dict(color="#8b949e", width=1.5, dash="dot"),
                opacity=0.7,
            ))

        # Confidence band (±10% simple heuristic)
        if show_confidence:
            upper = preds["value"] * 1.10
            lower = preds["value"] * 0.90
            fig.add_trace(go.Scatter(
                x=pd.concat([preds.index.to_series(), preds.index.to_series()[::-1]]),
                y=pd.concat([upper, lower[::-1]]),
                fill="toself",
                fillcolor="rgba(88,166,255,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name="±10% Band",
                showlegend=True,
                hoverinfo="skip",
            ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=preds.index,
            y=preds["value"],
            name=f"Forecast ({horizon_label})",
            line=dict(color="#58a6ff", width=2.5),
            mode="lines",
        ))

        # Peak marker
        fig.add_trace(go.Scatter(
            x=[peak_ts],
            y=[max_val],
            mode="markers+text",
            marker=dict(color="#f85149", size=10, symbol="star"),
            text=[f"Peak: {max_val:.2f} {unit_label}"],
            textposition="top center",
            textfont=dict(color="#f85149", size=11),
            name="Peak",
            showlegend=True,
        ))

        # Vertical separator — use add_shape + add_annotation instead of add_vline
        # to avoid Plotly's internal _mean() crash on mixed datetime/string x-axes
        _vx = active_df.index[-1].isoformat()
        fig.add_shape(
            type="line",
            x0=_vx, x1=_vx,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="rgba(210,153,34,0.6)", width=1.5, dash="dash"),
        )
        fig.add_annotation(
            x=_vx, y=1,
            xref="x", yref="paper",
            text="Forecast Start",
            showarrow=False,
            yanchor="bottom",
            font=dict(color="#d2991f", size=11),
            bgcolor="rgba(13,17,23,0.7)",
            borderpad=3,
        )

        fig.update_layout(
            **PLOTLY_THEME,
            height=420,
            xaxis_title="Time",
            yaxis_title=f"Power ({unit_label})",
            legend=dict(
                bgcolor="rgba(22,27,34,0.8)",
                bordercolor="rgba(48,54,61,0.6)",
                borderwidth=1,
                font=dict(size=11),
            ),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Hourly Aggregation ────────────────────────────────
        st.markdown("<div class='section-header'>📅 Hourly Demand Profile</div>", unsafe_allow_html=True)

        preds_hourly = preds["value"].resample("h").mean()
        colors_bar   = ["#f85149" if v == preds_hourly.max() else "#58a6ff" for v in preds_hourly]

        fig_bar = go.Figure(go.Bar(
            x=preds_hourly.index.strftime("%b %d %H:%M"),
            y=preds_hourly.values,
            marker_color=colors_bar,
            marker_line_width=0,
            name="Hourly Mean",
        ))
        fig_bar.update_layout(
            **PLOTLY_THEME,
            height=280,
            xaxis_title="Hour",
            yaxis_title=f"Mean Power ({unit_label})",
            margin=dict(l=0, r=0, t=10, b=0),
            bargap=0.15,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Download ──────────────────────────────────────────
        st.markdown("<div class='section-header'>💾 Export Forecast</div>", unsafe_allow_html=True)
        export_df = preds[["predicted_power_W"]].copy()
        export_df["predicted_power_kW"] = watts_to_kw(export_df["predicted_power_W"])
        csv_bytes = export_df.to_csv().encode()
        st.download_button(
            label="⬇ Download CSV",
            data=csv_bytes,
            file_name=f"forecast_{horizon_label.replace(' ','_')}_{forecast_start.strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

    else:
        st.markdown("""
        <div class='glass-card' style='text-align:center; padding:40px;'>
          <div style='font-size:3rem; margin-bottom:12px;'>🔮</div>
          <div style='font-size:1.2rem; font-weight:600; color:#58a6ff; margin-bottom:8px;'>
            Ready to Forecast
          </div>
          <div style='color:#8b949e; font-size:0.9rem; margin-bottom:16px;'>
            Optionally upload your own CSV above, then click <strong>▶ Run Forecast</strong>
          </div>
          <div style='font-size:0.78rem; color:#8b949e;'>
            💡 <em>Use <strong style='color:#e6edf3;'>⬇ Download Sample CSV</strong> to get the expected format</em>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("<div class='section-header'>📈 Historical Power Consumption</div>", unsafe_allow_html=True)

    power_series = watts_to_kw(raw["Global_active_power"]) if use_kw else raw["Global_active_power"]
    unit_label   = "kW" if use_kw else "W"

    # KPIs
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Total Records",  f"{len(raw):,}")
    e2.metric("Mean Power",     f"{power_series.mean():.2f} {unit_label}")
    e3.metric("Max Power",      f"{power_series.max():.2f} {unit_label}")
    e4.metric("Data Span",      f"{(raw.index[-1]-raw.index[0]).days} days")

    # Time-series overview
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=power_series.index,
        y=power_series.values,
        mode="lines",
        line=dict(color="#58a6ff", width=1),
        name="Global Active Power",
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.06)",
    ))
    # Rolling mean
    roll_mean = power_series.rolling(window=12*6).mean()
    fig_ts.add_trace(go.Scatter(
        x=roll_mean.index,
        y=roll_mean.values,
        mode="lines",
        line=dict(color="#f0883e", width=2),
        name="6-hr Rolling Mean",
    ))
    fig_ts.update_layout(
        **PLOTLY_THEME,
        height=350,
        xaxis_title="Time",
        yaxis_title=f"Power ({unit_label})",
        legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="rgba(48,54,61,0.6)", borderwidth=1),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # ── Sub-metering breakdown ────────────────────────────────
    st.markdown("<div class='section-header'>🏠 Sub-metering Breakdown</div>", unsafe_allow_html=True)

    sub_cols = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    sub_labels = ["Kitchen (Wh)", "Laundry (Wh)", "HVAC (Wh)"]
    sub_colors = ["#58a6ff", "#3fb950", "#f0883e"]

    fig_sub = go.Figure()
    for col, label, color in zip(sub_cols, sub_labels, sub_colors):
        fig_sub.add_trace(go.Scatter(
            x=raw.index,
            y=raw[col],
            name=label,
            line=dict(color=color, width=1),
            opacity=0.8,
        ))
    fig_sub.update_layout(
        **PLOTLY_THEME,
        height=300,
        xaxis_title="Time",
        yaxis_title="Energy (Wh)",
        legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="rgba(48,54,61,0.6)", borderwidth=1),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_sub, use_container_width=True)

    # ── Hourly & Day-of-week patterns ────────────────────────
    st.markdown("<div class='section-header'>🕐 Temporal Patterns</div>", unsafe_allow_html=True)
    col_h, col_d = st.columns(2)

    with col_h:
        hourly_avg = power_series.groupby(power_series.index.hour).mean()
        fig_h = go.Figure(go.Bar(
            x=hourly_avg.index,
            y=hourly_avg.values,
            marker_color=[
                "#f85149" if v == hourly_avg.max() else
                "#3fb950" if v == hourly_avg.min() else "#58a6ff"
                for v in hourly_avg.values
            ],
            marker_line_width=0,
        ))
        fig_h.update_layout(
            **PLOTLY_THEME,
            height=260,
            title=dict(text="Average by Hour of Day", font=dict(size=13, color="#8b949e")),
            xaxis_title="Hour",
            yaxis_title=f"Mean Power ({unit_label})",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with col_d:
        dow_avg = power_series.groupby(power_series.index.dayofweek).mean()
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig_d = go.Figure(go.Bar(
            x=dow_labels,
            y=dow_avg.values,
            marker_color=["#f0883e" if i >= 5 else "#58a6ff" for i in range(7)],
            marker_line_width=0,
        ))
        fig_d.update_layout(
            **PLOTLY_THEME,
            height=260,
            title=dict(text="Average by Day of Week", font=dict(size=13, color="#8b949e")),
            xaxis_title="Day",
            yaxis_title=f"Mean Power ({unit_label})",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_d, use_container_width=True)

    # ── Distribution ──────────────────────────────────────────
    st.markdown("<div class='section-header'>📊 Power Distribution</div>", unsafe_allow_html=True)
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=power_series.values,
        nbinsx=60,
        marker_color="#58a6ff",
        marker_line_color="rgba(0,0,0,0)",
        opacity=0.8,
        name="Distribution",
    ))
    # Use add_shape + add_annotation instead of add_vline to avoid Plotly _mean() crash
    _mean_val = float(power_series.mean())
    fig_dist.add_shape(
        type="line",
        x0=_mean_val, x1=_mean_val,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#f0883e", width=2, dash="dash"),
    )
    fig_dist.add_annotation(
        x=_mean_val, y=1,
        xref="x", yref="paper",
        text=f"Mean: {_mean_val:.2f} {unit_label}",
        showarrow=False,
        yanchor="bottom",
        font=dict(color="#f0883e", size=11),
        bgcolor="rgba(13,17,23,0.7)",
        borderpad=3,
    )
    fig_dist.update_layout(
        **PLOTLY_THEME,
        height=280,
        xaxis_title=f"Power ({unit_label})",
        yaxis_title="Frequency",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("<div class='section-header'>🧠 Model Architecture & Training</div>", unsafe_allow_html=True)

    col_arch, col_feat = st.columns([1, 1])

    with col_arch:
        st.markdown("""
        <div class='glass-card'>
          <div style='font-size:0.85rem; font-weight:600; color:#58a6ff; margin-bottom:12px;'>
            🏗️ LSTM Architecture
          </div>
          <table style='width:100%; font-size:0.82rem; border-collapse:collapse;'>
            <tr style='border-bottom:1px solid rgba(48,54,61,0.5);'>
              <td style='padding:6px 0; color:#8b949e;'>Model Type</td>
              <td style='padding:6px 0; color:#e6edf3; font-weight:500;'>LSTM (Stacked)</td>
            </tr>
            <tr style='border-bottom:1px solid rgba(48,54,61,0.5);'>
              <td style='padding:6px 0; color:#8b949e;'>Input Shape</td>
              <td style='padding:6px 0; color:#e6edf3; font-weight:500;'>(None, 12, 20)</td>
            </tr>
            <tr style='border-bottom:1px solid rgba(48,54,61,0.5);'>
              <td style='padding:6px 0; color:#8b949e;'>Sequence Length</td>
              <td style='padding:6px 0; color:#e6edf3; font-weight:500;'>12 steps (1 hour)</td>
            </tr>
            <tr style='border-bottom:1px solid rgba(48,54,61,0.5);'>
              <td style='padding:6px 0; color:#8b949e;'>Features</td>
              <td style='padding:6px 0; color:#e6edf3; font-weight:500;'>20</td>
            </tr>
            <tr style='border-bottom:1px solid rgba(48,54,61,0.5);'>
              <td style='padding:6px 0; color:#8b949e;'>Interval</td>
              <td style='padding:6px 0; color:#e6edf3; font-weight:500;'>5 minutes</td>
            </tr>
            <tr>
              <td style='padding:6px 0; color:#8b949e;'>Training Data</td>
              <td style='padding:6px 0; color:#e6edf3; font-weight:500;'>2006 – 2010</td>
            </tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with col_feat:
        st.markdown("""
        <div class='glass-card'>
          <div style='font-size:0.85rem; font-weight:600; color:#58a6ff; margin-bottom:12px;'>
            🔧 Feature Groups
          </div>
          <div style='font-size:0.8rem; margin-bottom:8px;'>
            <span class='badge badge-blue'>Temporal (7)</span>
            <div style='color:#8b949e; margin-top:4px; margin-left:4px;'>
              hour_sin/cos · day_sin/cos · month_sin/cos · is_weekend
            </div>
          </div>
          <div style='font-size:0.8rem; margin-bottom:8px;'>
            <span class='badge badge-green'>Lag Features (6)</span>
            <div style='color:#8b949e; margin-top:4px; margin-left:4px;'>
              1h · 3h · 6h · 12h · 1day · 1week
            </div>
          </div>
          <div style='font-size:0.8rem;'>
            <span class='badge badge-amber'>Rolling Stats (7)</span>
            <div style='color:#8b949e; margin-top:4px; margin-left:4px;'>
              mean(3h/6h/24h) · std(3h/24h) · min(24h) · max(24h)
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Metrics Table ─────────────────────────────────────────
    st.markdown("<div class='section-header'>📉 Performance Metrics</div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    splits = ["Train", "Validation", "Test"]
    split_colors = ["#58a6ff", "#f0883e", "#3fb950"]

    for col, split, color in zip([m1, m2, m3], splits, split_colors):
        with col:
            rmse = metrics_df.loc["RMSE", split]
            mae  = metrics_df.loc["MAE",  split]
            r2   = metrics_df.loc["R2",   split]
            mape = metrics_df.loc["MAPE", split]
            st.markdown(f"""
            <div class='glass-card' style='border-top:3px solid {color};'>
              <div style='font-size:0.9rem; font-weight:700; color:{color}; margin-bottom:12px;'>
                {split} Split
              </div>
              <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                <span style='color:#8b949e; font-size:0.8rem;'>RMSE</span>
                <span style='color:#e6edf3; font-weight:600; font-size:0.85rem;'>{rmse:.1f} W</span>
              </div>
              <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                <span style='color:#8b949e; font-size:0.8rem;'>MAE</span>
                <span style='color:#e6edf3; font-weight:600; font-size:0.85rem;'>{mae:.1f} W</span>
              </div>
              <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                <span style='color:#8b949e; font-size:0.8rem;'>R²</span>
                <span style='color:#e6edf3; font-weight:600; font-size:0.85rem;'>{r2:.4f}</span>
              </div>
              <div style='display:flex; justify-content:space-between;'>
                <span style='color:#8b949e; font-size:0.8rem;'>MAPE</span>
                <span style='color:#e6edf3; font-weight:600; font-size:0.85rem;'>{mape:.1f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Metrics Radar Chart ───────────────────────────────────
    st.markdown("<div class='section-header'>📡 Metrics Comparison (Radar)</div>", unsafe_allow_html=True)

    def hex_to_rgba(hex_color, alpha=0.12):
        """Convert a #rrggbb hex string to a valid rgba(...) string for Plotly."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    metric_names = ["RMSE", "MAE", "MAPE"]
    fig_radar = go.Figure()
    for split, color in zip(splits, split_colors):
        vals = [metrics_df.loc[m, split] for m in metric_names]
        # Normalize for radar (lower is better → invert)
        max_vals = [metrics_df.loc[m].max() for m in metric_names]
        norm_vals = [1 - v/mv for v, mv in zip(vals, max_vals)]
        norm_vals.append(norm_vals[0])
        cats = metric_names + [metric_names[0]]
        # Build a valid rgba fill color from the hex color string
        fill_color = (
            color.replace(")", ",0.12)").replace("rgb", "rgba")
            if color.startswith("rgb")
            else hex_to_rgba(color, alpha=0.12)
        )
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_vals,
            theta=cats,
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=color, width=2),
            name=split,
            opacity=0.8,
        ))
    fig_radar.update_layout(
        **PLOTLY_THEME,
        height=350,
        polar=dict(
            bgcolor="rgba(13,17,23,0.6)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="rgba(48,54,61,0.5)",
                linecolor="rgba(48,54,61,0.5)",
                tickfont=dict(color="#8b949e", size=9),
            ),
            angularaxis=dict(
                gridcolor="rgba(48,54,61,0.5)",
                linecolor="rgba(48,54,61,0.5)",
                tickfont=dict(color="#e6edf3", size=11),
            ),
        ),
        legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="rgba(48,54,61,0.6)", borderwidth=1),
        margin=dict(l=40, r=40, t=20, b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Forecasting Strategy ──────────────────────────────────
    st.markdown("<div class='section-header'>🔄 Recursive Forecasting Strategy</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
      <div style='font-size:0.85rem; color:#8b949e; line-height:1.8;'>
        <strong style='color:#58a6ff;'>How it works:</strong><br>
        1. Load <strong style='color:#e6edf3;'>1 week</strong> of 2026 sample data (full LSTM lookback window)<br>
        2. Compute all <strong style='color:#e6edf3;'>20 forecasting features</strong> (temporal + lag + rolling stats)<br>
        3. Roll the LSTM window forward <strong style='color:#e6edf3;'>step-by-step</strong> to generate predictions<br>
        4. Each predicted value is <strong style='color:#e6edf3;'>fed back</strong> as input for the next step (autoregressive)<br><br>
        <code style='background:rgba(88,166,255,0.1); padding:8px 12px; border-radius:6px; display:block; font-size:0.8rem;'>
          Known history ──► [LSTM window] ──► ŷ₁<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;shift ──► [LSTM window] ──► ŷ₂<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;shift ──► [LSTM window] ──► ŷ₃ …
        </code>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — RAW DATA
# ═══════════════════════════════════════════════════════════════
with tab_raw:
    st.markdown("<div class='section-header'>📋 Sample Data Overview</div>", unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Rows",    f"{len(raw):,}")
    r2.metric("Columns", f"{len(raw.columns)}")
    r3.metric("From",    raw.index[0].strftime("%b %d, %Y"))
    r4.metric("To",      raw.index[-1].strftime("%b %d, %Y"))

    # Column selector
    selected_cols = st.multiselect(
        "Select columns to display",
        options=list(raw.columns),
        default=list(raw.columns),
    )

    n_rows = st.slider("Number of rows to preview", 10, 200, 50, step=10)

    display_df = raw[selected_cols].head(n_rows).copy()
    # Convert power to kW if needed
    if use_kw and "Global_active_power" in display_df.columns:
        display_df["Global_active_power"] = watts_to_kw(display_df["Global_active_power"])

    st.dataframe(
        display_df.style.format("{:.4f}"),
        use_container_width=True,
        height=400,
    )

    # Stats
    st.markdown("<div class='section-header'>📊 Descriptive Statistics</div>", unsafe_allow_html=True)
    stats_df = raw[selected_cols].describe()
    if use_kw and "Global_active_power" in stats_df.columns:
        stats_df["Global_active_power"] = watts_to_kw(stats_df["Global_active_power"])
    st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)

    # Correlation heatmap
    st.markdown("<div class='section-header'>🔗 Correlation Matrix</div>", unsafe_allow_html=True)
    numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
    corr = raw[numeric_cols].corr()

    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=[
            [0.0, "#f85149"],
            [0.5, "#161b22"],
            [1.0, "#3fb950"],
        ],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, color="#e6edf3"),
        colorbar=dict(
            tickfont=dict(color="#8b949e"),
            title=dict(text="r", font=dict(color="#8b949e")),
        ),
    ))
    fig_corr.update_layout(
        **{
            **PLOTLY_THEME,
            # Merge tickfont into the xaxis/yaxis dicts already in PLOTLY_THEME
            # to avoid "multiple values for keyword argument" TypeError
            "xaxis": {**PLOTLY_THEME["xaxis"], "tickfont": dict(size=10)},
            "yaxis": {**PLOTLY_THEME["yaxis"], "tickfont": dict(size=10)},
        },
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<hr style='margin-top:32px;'>
<div style='text-align:center; color:#8b949e; font-size:0.75rem; padding:12px 0;'>
  ⚡ Energy Demand Forecasting Dashboard &nbsp;·&nbsp;
  LSTM Model trained on UCI Household Power Consumption Dataset &nbsp;·&nbsp;
  Recursive Autoregressive Forecasting
</div>
""", unsafe_allow_html=True)
