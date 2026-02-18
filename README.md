# ⚡ Energy Demand Forecasting Dashboard

A production-grade **LSTM-powered energy demand forecasting** dashboard built with Streamlit and Plotly.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 🔮 Features

| Tab | Description |
|-----|-------------|
| **🔮 Forecast** | Run live recursive LSTM forecasts for 6, 12, or 24-hour horizons |
| **📈 Data Explorer** | EDA — time-series, sub-metering breakdown, hourly/daily patterns, distribution |
| **🧠 Model Performance** | Architecture details, Train/Val/Test metrics, radar comparison chart |
| **📋 Raw Data** | Configurable data preview, descriptive stats, correlation heatmap |

### Key capabilities
- **Recursive autoregressive forecasting** — each predicted value feeds back as input for the next step
- **Upload your own CSV** — use any 5-min interval household power data (≥ 1 week)
- **Download sample data** — get the template CSV to understand the expected format
- **Export forecasts** — download predictions as CSV
- **Dark glassmorphism UI** — GitHub-inspired dark theme with Plotly interactive charts

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | Stacked LSTM |
| Input shape | `(None, 12, 20)` |
| Sequence length | 12 steps (1 hour) |
| Interval | 5 minutes |
| Features | 20 (temporal + lag + rolling stats) |
| Training data | UCI Household Power Consumption (2006–2010) |

### Feature Groups
- **Temporal (7):** `hour_sin/cos`, `day_sin/cos`, `month_sin/cos`, `is_weekend`
- **Lag (6):** 1h, 3h, 6h, 12h, 1-day, 1-week
- **Rolling stats (7):** mean/std/min/max over 3h, 6h, 24h windows

---

## 🚀 Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/energy-demand-lstm.git
cd energy-demand-lstm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run dashboard.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 File Structure

```
energy-demand-lstm/
├── dashboard.py                    # Main Streamlit app
├── requirements.txt                # Python dependencies
├── generate_sample_data.py         # Script to regenerate sample CSV
│
├── lstm_forecasting_model.keras    # Pre-trained LSTM model
├── scaler_X_forecasting.pkl        # Feature scaler
├── scaler_y_forecasting.pkl        # Target scaler
├── feature_list_forecasting.json   # Ordered list of 20 feature names
│
├── sample_data_2026.csv            # 2026 lookback data (~1 week, 5-min intervals)
├── forecasting_model_metrics.csv   # Train/Val/Test performance metrics
│
├── predict-2026.ipynb              # Prediction notebook
├── residential-lstm.ipynb          # Exploratory notebook
│
└── .streamlit/
    └── config.toml                 # Streamlit theme config
```

---

## 📊 Model Performance

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| RMSE | 528.7 W | 533.8 W | 480.7 W |
| MAE | 292.9 W | 304.1 W | 285.3 W |
| R² | 0.751 | 0.732 | 0.672 |
| MAPE | 39.8% | 32.8% | 42.0% |

---

## 📄 Data Format (for Upload)

Your CSV must have at minimum:

| Column | Type | Description |
|--------|------|-------------|
| `datetime` | `YYYY-MM-DD HH:MM:SS` | Timestamp at 5-min intervals |
| `Global_active_power` | float | Household power in kW (auto-detected) or W |

Optional columns (not used by model, but included in sample):
`Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1/2/3`

**Minimum rows:** 2,028 rows (~1 week + 1 hour at 5-min intervals)

---

## 📜 License

MIT License — feel free to use, modify, and distribute.
