# CS313 Final Project — Stock Market Prediction with Deep Learning

Time-series forecasting and portfolio construction on Nasdaq and Vietnamese stock data.  
CS313 Deep Learning for AI · Spring 2026 · Fulbright University Vietnam

---

## What this project does

| Task | Description |
|------|-------------|
| 1.1–1.3 | Nasdaq next-day, nth-day, and k-day consecutive price regression (CNN) |
| 2.1–2.3 | Vietnam price regression with 11 features: OHLCV + SMA5/20, RSI14, MACD, Volatility |
| 3.1–3.2 | Buy and sell signal classification (LSTM, percentile-based labels) |
| 4.1–4.3 | Per-company portfolio construction with profitability ranking and risk filtering |
| 5.1–5.2 | FastAPI REST service + browser-based prediction UI |

---

## Repo structure

```
cs313-stock-analysis/
├── data/
│   ├── nasdaq/          # Nasdaq historical CSVs (one per company)
│   └── vn/              # Vietnam historical CSVs (one per company)
├── models/              # Trained .keras model files
├── 240069_project_notebook.ipynb
├── stock_api.py         # FastAPI service with embedded frontend
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Google Colab (recommended)

Open `240069_project_notebook.ipynb` in Colab and run all cells:

```
Runtime → Run all
```

The setup cell clones this repo automatically and sets all paths relative to it. No Drive mounting needed. Pre-trained models load from `models/` so nothing retrains unless you delete the files.

Once the tunnel cell prints a `trycloudflare.com` URL (takes ~10–15s), open it in your browser. If the tunnel cell fails on first try, just rerun it — Cloudflare's free tier is occasionally flaky.

### Local

```bash
git clone https://github.com/tonasthesecond/cs313-stock-analysis
cd cs313-stock-analysis
pip install -r requirements.txt
jupyter notebook 240069_project_notebook.ipynb
```

The setup cell detects it's not in Colab and resolves all paths from the current directory. Run the notebook from the repo root.

---

## Settings

Each section has a dedicated settings cell you can run independently:

| Cell | Variable | Default | Effect |
|------|----------|---------|--------|
| `nasdaq settings` | `NASDAQ_WINDOW_SIZE` | 30 | Lookback window for Nasdaq models |
| `nasdaq settings` | `FORECAST_DAY` | 3 | Which day ahead to predict (task 1.2) |
| `nasdaq settings` | `FORECAST_DAYS` | 3 | How many consecutive days (task 1.3) |
| `vietnam settings` | `VN_WINDOW_SIZE` | 20 | Lookback window for Vietnam models |
| `vietnam settings` | `VN_FORECAST_DAY` | 3 | Which day ahead to predict (task 2.2) |
| `vietnam settings` | `VN_FORECAST_DAYS` | 3 | How many consecutive days (task 2.3) |
| `signal settings` | `SIGNAL_WINDOW_PAST` | 10 | History fed to signal classifier |
| `signal settings` | `SIGNAL_WINDOW_FUTURE` | 10 | Future context used for label generation |
| `task 4 settings` | `MIN_ROWS_RAW` | 160 | Minimum rows for company inclusion |
| `task 4 settings` | `RISK_PERCENTILE` | 70 | Risk threshold percentile for exclusion |
| `task 4 settings` | `TOP_N` | 10 | Companies shown in profitability ranking |

---

## Models

Pre-trained `.keras` files are committed to `models/` — running the notebook loads them directly without retraining. If you want to retrain from scratch, delete the relevant file(s) from `models/` and re-run the cell; `train_or_load_model` will train and save a new one automatically.

```
models/
├── task1_1_<ticker>.keras           # 1.1 next-day
├── task1_2_day3_<ticker>.keras      # 1.2 nth-day
├── task1_3_3days_<ticker>.keras     # 1.3 k-day
├── task2_1_<ticker>.keras           # 2.1
├── task2_2_day3_<ticker>.keras      # 2.2
├── task2_3_3days_<ticker>.keras     # 2.3
├── task3_1_buy_<ticker>.keras       # buy signal
├── task3_2_sell_<ticker>.keras      # sell signal
├── task4_<ticker>.keras             # per-company portfolio models
├── trial_arch_v0.keras              # architecture benchmark trials
└── trial_sweep_v2_w20.keras         # window sweep trials
```

---

## FastAPI service

### Starting the server

**From the notebook (Colab):**

`Runtime → Run all` handles everything. The tunnel cell polls until the URL appears and prints it. Open that URL in your browser — the prediction UI is at `/`.

If the tunnel cell errors on the first run, rerun just that cell.

**Standalone (local or any server):**

```bash
cd cs313-stock-analysis
pip install fastapi uvicorn
uvicorn stock_api:app --host 0.0.0.0 --port 8000
```

Override data/model paths with environment variables if needed:

```bash
MODEL_DIR=./models VN_DATA_DIR=./data/vn uvicorn stock_api:app --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Browser-based prediction UI |
| `GET` | `/health` | Liveness check → `{"status": "ok"}` |
| `GET` | `/tickers` | Lists tickers with deployed models |
| `GET` | `/window/{ticker}` | Latest 20-day input window + last Close |
| `POST` | `/predict` | Runs inference, returns predicted Close (VND) |

### Prediction request

```bash
# Step 1: get the input window
curl http://localhost:8000/window/ADG-VNINDEX-History
```

```json
{
  "window": [[24300.0, 24500.0, 24100.0, 24250.0, 150000.0, 24100.2, 24198.4, 52.3, -80.1, -60.4, 0.012], "..."],
  "last_close": 24250.0
}
```

```bash
# Step 2: predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "ADG-VNINDEX-History", "window": [[...]], "close_idx": 3}'
```

```json
{
  "ticker": "ADG-VNINDEX-History",
  "predicted_close": 24580.0
}
```

### Web interface

Navigate to `http://localhost:8000` (or the Cloudflare tunnel URL) in any browser.

1. Select a company from the dropdown
2. Click **Run prediction**
3. Result shows predicted next-day close in VND and percentage change from the last known price

---

## Requirements

```
tensorflow>=2.15
scikit-learn
pandas
numpy
matplotlib
fastapi
uvicorn
requests
```

Install: `pip install -r requirements.txt`

---

## Report

See `240069_project_report.pdf` for full methodology, architecture decisions, sweep results, and conclusions.
