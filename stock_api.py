"""
stock_api.py — FastAPI service for VN stock price prediction
Serves trained task4_<ticker>.keras models over REST + a browser UI at GET /

Paths are resolved relative to this file by default.
Override with env vars:
    MODEL_DIR   — directory containing task4_*.keras files
    VN_DATA_DIR — directory containing <ticker>.csv files
"""

import os
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List

# ── paths ─────────────────────────────────────────────────────────────────────

_here = Path(__file__).resolve().parent
MODEL_DIR   = os.environ.get("MODEL_DIR",   str(_here / "models"))
VN_DATA_DIR = os.environ.get("VN_DATA_DIR", str(_here / "data" / "vn"))

FEATURES  = ["Open", "High", "Low", "Close", "Volume",
             "SMA5", "SMA20", "RSI14", "MACD", "MACD_signal", "Volatility"]
CLOSE_IDX = FEATURES.index("Close")
WINDOW    = 20

# ── app ───────────────────────────────────────────────────────────────────────

app         = FastAPI(title="VN Stock Predictor", version="1.0.0")
model_cache = {}

# ── utilities ─────────────────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    close = df["Close"]

    df["SMA5"]  = close.rolling(5).mean()
    df["SMA20"] = close.rolling(20).mean()

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, 1e-8)
    df["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["Volatility"] = close.pct_change().rolling(10).std()

    return df.dropna().reset_index(drop=True)


def normalize_window(window: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature min-max normalisation across the window. Returns (normed, mins, maxs)."""
    w    = window.astype(float).copy()
    mins = np.zeros(w.shape[1])
    maxs = np.zeros(w.shape[1])
    for i in range(w.shape[1]):
        mn, mx    = w[:, i].min(), w[:, i].max()
        denom     = mx - mn if mx != mn else 1e-8
        w[:, i]   = (w[:, i] - mn) / denom
        mins[i], maxs[i] = mn, mx
    return w, mins, maxs


def load_model(ticker: str) -> tf.keras.Model:
    if ticker not in model_cache:
        path = os.path.join(MODEL_DIR, f"task4_{ticker}.keras")
        if not os.path.exists(path):
            raise FileNotFoundError(f"no model found for ticker '{ticker}'")
        model_cache[ticker] = tf.keras.models.load_model(path)
    return model_cache[ticker]

# ── request schema ────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    ticker:    str
    window:    List[List[float]]
    close_idx: int = CLOSE_IDX

# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tickers")
def tickers():
    paths = glob.glob(os.path.join(MODEL_DIR, "task4_*.keras"))
    names = [
        re.sub(r"^task4_|\.keras$", "", os.path.basename(p))
        for p in sorted(paths)
    ]
    return {"tickers": names}


@app.get("/window/{ticker}")
def get_window(ticker: str):
    path = os.path.join(VN_DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"data not found for ticker '{ticker}'")

    raw  = pd.read_csv(path, index_col=0).sort_values("TradingDate").reset_index(drop=True)
    data = add_technical_indicators(raw)

    if len(data) < WINDOW:
        raise HTTPException(status_code=422, detail=f"not enough data for '{ticker}' (need {WINDOW} rows)")

    window     = data[FEATURES].values[-WINDOW:].tolist()
    last_close = float(data["Close"].iloc[-1])
    return {"window": window, "last_close": last_close}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        model = load_model(req.ticker)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    window = np.array(req.window)
    if window.shape != (WINDOW, len(FEATURES)):
        raise HTTPException(
            status_code=422,
            detail=f"window must be shape ({WINDOW}, {len(FEATURES)}), got {window.shape}"
        )

    w_norm, mins, maxs = normalize_window(window)
    pred_norm           = float(model.predict(w_norm[np.newaxis], verbose=0)[0, 0])

    mn_c, mx_c     = mins[req.close_idx], maxs[req.close_idx]
    predicted_close = pred_norm * (mx_c - mn_c) + mn_c

    return {"ticker": req.ticker, "predicted_close": round(predicted_close, 2)}


# ── frontend ──────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>VN Stock Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg:        #0a0c0f;
      --surface:   #111418;
      --border:    #1e2328;
      --border-hi: #2a3038;
      --text:      #e2e8f0;
      --muted:     #4a5568;
      --dim:       #718096;
      --accent:    #00d4aa;
      --accent-lo: rgba(0, 212, 170, 0.08);
      --accent-md: rgba(0, 212, 170, 0.15);
      --red:       #f56565;
      --red-lo:    rgba(245, 101, 101, 0.08);
      --mono:      'IBM Plex Mono', monospace;
      --sans:      'IBM Plex Sans', sans-serif;
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: var(--sans);
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }

    /* subtle grid bg */
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(var(--border) 1px, transparent 1px),
        linear-gradient(90deg, var(--border) 1px, transparent 1px);
      background-size: 40px 40px;
      opacity: .4;
      pointer-events: none;
    }

    .shell {
      position: relative;
      width: 100%;
      max-width: 480px;
    }

    /* corner marks */
    .shell::before, .shell::after,
    .inner::before, .inner::after {
      content: '';
      position: absolute;
      width: 10px;
      height: 10px;
      border-color: var(--accent);
      border-style: solid;
      opacity: .6;
    }
    .shell::before { top: -2px; left: -2px;  border-width: 2px 0 0 2px; }
    .shell::after  { top: -2px; right: -2px; border-width: 2px 2px 0 0; }

    .card {
      background: var(--surface);
      border: 1px solid var(--border-hi);
      padding: 36px 40px 40px;
      position: relative;
    }

    .inner::before { bottom: -2px; left: -2px;  border-width: 0 0 2px 2px; }
    .inner::after  { bottom: -2px; right: -2px; border-width: 0 2px 2px 0; }

    /* header */
    .header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      margin-bottom: 32px;
    }
    .title {
      font-family: var(--mono);
      font-size: 1rem;
      font-weight: 600;
      color: var(--accent);
      letter-spacing: .04em;
      text-transform: uppercase;
    }
    .subtitle {
      font-size: 0.75rem;
      color: var(--dim);
      margin-top: 4px;
      font-weight: 300;
    }
    .badge {
      font-family: var(--mono);
      font-size: 0.6rem;
      padding: 3px 7px;
      border: 1px solid var(--accent);
      color: var(--accent);
      letter-spacing: .08em;
      opacity: .7;
    }

    /* form */
    .field-label {
      font-family: var(--mono);
      font-size: 0.65rem;
      font-weight: 500;
      color: var(--muted);
      letter-spacing: .1em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    select {
      width: 100%;
      padding: 10px 14px;
      background: var(--bg);
      border: 1px solid var(--border-hi);
      color: var(--text);
      font-family: var(--mono);
      font-size: 0.85rem;
      appearance: none;
      cursor: pointer;
      outline: none;
      margin-bottom: 20px;
      transition: border-color .15s;
    }
    select:focus { border-color: var(--accent); }

    button {
      width: 100%;
      padding: 12px;
      background: transparent;
      border: 1px solid var(--accent);
      color: var(--accent);
      font-family: var(--mono);
      font-size: 0.8rem;
      font-weight: 500;
      letter-spacing: .1em;
      text-transform: uppercase;
      cursor: pointer;
      transition: background .15s, color .15s;
      position: relative;
      overflow: hidden;
    }
    button::before {
      content: '';
      position: absolute;
      inset: 0;
      background: var(--accent);
      transform: scaleX(0);
      transform-origin: left;
      transition: transform .2s ease;
    }
    button span { position: relative; }
    button:hover::before  { transform: scaleX(1); }
    button:hover          { color: var(--bg); }
    button:disabled       { opacity: .35; cursor: not-allowed; }
    button:disabled::before { transform: scaleX(0) !important; }

    /* divider */
    .divider {
      border: none;
      border-top: 1px solid var(--border);
      margin: 24px 0;
    }

    /* result */
    .result {
      display: none;
      animation: fadeUp .3s ease both;
    }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(6px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .result-label {
      font-family: var(--mono);
      font-size: 0.62rem;
      letter-spacing: .12em;
      text-transform: uppercase;
      color: var(--dim);
      margin-bottom: 10px;
    }

    .price-row {
      display: flex;
      align-items: baseline;
      gap: 14px;
    }
    .price {
      font-family: var(--mono);
      font-size: 2.2rem;
      font-weight: 600;
      color: var(--text);
      letter-spacing: -.01em;
    }
    .currency {
      font-family: var(--mono);
      font-size: 0.75rem;
      color: var(--dim);
    }

    .change-row {
      margin-top: 10px;
      display: flex;
      align-items: center;
      gap: 16px;
      font-family: var(--mono);
      font-size: 0.78rem;
    }
    .last-close { color: var(--dim); }
    .delta { font-weight: 600; }
    .positive { color: var(--accent); }
    .negative { color: var(--red); }

    .meta-row {
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid var(--border);
      display: flex;
      gap: 24px;
    }
    .meta-item { display: flex; flex-direction: column; gap: 3px; }
    .meta-key   { font-family: var(--mono); font-size: .6rem; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; }
    .meta-value { font-family: var(--mono); font-size: .78rem; color: var(--text); }

    /* error */
    .error {
      display: none;
      margin-top: 16px;
      padding: 12px 14px;
      background: var(--red-lo);
      border: 1px solid var(--red);
      font-family: var(--mono);
      font-size: 0.78rem;
      color: var(--red);
      animation: fadeUp .2s ease both;
    }

    /* spinner */
    .spinner {
      display: none;
      margin-top: 16px;
      font-family: var(--mono);
      font-size: 0.72rem;
      color: var(--dim);
      letter-spacing: .06em;
    }
    .spinner::before {
      content: '[ ';
    }
    .spinner::after {
      content: ' ]';
    }
    .dot {
      display: inline-block;
      animation: blink 1s step-start infinite;
    }
    .dot:nth-child(2) { animation-delay: .2s; }
    .dot:nth-child(3) { animation-delay: .4s; }
    @keyframes blink { 50% { opacity: 0; } }
  </style>
</head>
<body>
<div class="shell">
  <div class="card inner">
    <div class="header">
      <div>
        <div class="title">VN Stock Predictor</div>
        <div class="subtitle">CNN-LSTM · next-day close</div>
      </div>
      <div class="badge">LIVE</div>
    </div>

    <div class="field-label">Company</div>
    <select id="ticker"><option>Loading tickers...</option></select>

    <button id="btn" onclick="runPredict()"><span>Run prediction</span></button>

    <div class="spinner" id="spinner">
      <span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
      &nbsp;running inference
    </div>

    <div class="result" id="result">
      <hr class="divider"/>
      <div class="result-label">Predicted next-day close</div>
      <div class="price-row">
        <span class="price" id="price"></span>
        <span class="currency">VND</span>
      </div>
      <div class="change-row">
        <span class="last-close" id="last-close"></span>
        <span class="delta" id="delta"></span>
      </div>
      <div class="meta-row">
        <div class="meta-item">
          <span class="meta-key">Ticker</span>
          <span class="meta-value" id="meta-ticker"></span>
        </div>
        <div class="meta-item">
          <span class="meta-key">Window</span>
          <span class="meta-value">20 days</span>
        </div>
        <div class="meta-item">
          <span class="meta-key">Model</span>
          <span class="meta-value">CNN-LSTM v2</span>
        </div>
      </div>
    </div>

    <div class="error" id="error"></div>
  </div>
</div>

<script>
  async function loadTickers() {
    try {
      const res  = await fetch('/tickers');
      const data = await res.json();
      const sel  = document.getElementById('ticker');
      if (!data.tickers.length) {
        sel.innerHTML = '<option disabled>No models found</option>';
        return;
      }
      sel.innerHTML = data.tickers
        .map(t => `<option value="${t}">${t}</option>`)
        .join('');
    } catch (e) {
      document.getElementById('ticker').innerHTML =
        '<option disabled>Failed to load tickers</option>';
    }
  }

  async function runPredict() {
    const ticker  = document.getElementById('ticker').value;
    const btn     = document.getElementById('btn');
    const spinner = document.getElementById('spinner');
    const result  = document.getElementById('result');
    const errBox  = document.getElementById('error');

    btn.disabled          = true;
    spinner.style.display = 'block';
    result.style.display  = 'none';
    errBox.style.display  = 'none';

    try {
      const winRes  = await fetch(`/window/${encodeURIComponent(ticker)}`);
      if (!winRes.ok) {
        const err = await winRes.json();
        throw new Error(err.detail ?? `HTTP ${winRes.status}`);
      }
      const winData = await winRes.json();

      const predRes = await fetch('/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ ticker, window: winData.window, close_idx: 3 }),
      });
      if (!predRes.ok) {
        const err = await predRes.json();
        throw new Error(err.detail ?? `HTTP ${predRes.status}`);
      }
      const pred = await predRes.json();

      const predicted = pred.predicted_close;
      const last      = winData.last_close;
      const pct       = ((predicted - last) / last * 100).toFixed(2);
      const sign      = pct >= 0 ? '+' : '';
      const cls       = pct >= 0 ? 'positive' : 'negative';

      document.getElementById('price').textContent =
        Math.round(predicted).toLocaleString('vi-VN');
      document.getElementById('last-close').textContent =
        `prev ${Math.round(last).toLocaleString('vi-VN')} VND`;
      document.getElementById('delta').innerHTML =
        `<span class="${cls}">${sign}${pct}%</span>`;
      document.getElementById('meta-ticker').textContent = ticker;

      result.style.display = 'block';

    } catch (e) {
      errBox.textContent   = 'Error: ' + e.message;
      errBox.style.display = 'block';
    } finally {
      btn.disabled          = false;
      spinner.style.display = 'none';
    }
  }

  loadTickers();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def frontend():
    return _HTML
