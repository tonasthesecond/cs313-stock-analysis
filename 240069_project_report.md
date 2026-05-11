# CS313 Deep Learning for Artificial Intelligence — Final Project Report
### Time-Series Data and Application to Stock Markets

**Student ID:** 240069  
**Course:** CS313 Deep Learning for Artificial Intelligence, Spring 2026  
**Fulbright University Vietnam**  
**GitHub:** https://github.com/tonasthesecond/cs313-stock-analysis

---

## Introduction

Stock markets are noisy, non-stationary, and brutally unforgiving to naive forecasting. Classical statistical methods such as ARIMA treat the series as locally linear and collapse during the kind of regime-shifting volatility that characterises emerging markets like Vietnam's HOSE and HNX. Deep learning models can instead learn hierarchical temporal patterns directly from raw price sequences and engineered indicators, without the stationarity assumptions that break statistical approaches during market crises.

This project explores that capacity across four progressively harder problems: raw price regression on Nasdaq data, price regression augmented with technical indicators on Vietnamese stocks, binary trading-signal classification, and multi-company portfolio construction with risk filtering. A FastAPI service with an embedded HTML frontend then wraps the portfolio models into a deployable endpoint reachable over the public internet via a Cloudflare tunnel.

---

## Shared Methodology

Several design decisions apply globally across all tasks and are explained once here.

### Training / Validation / Test Split

All splits are strictly chronological with `shuffle=False` enforced at every stage. The split ratios are 64% training, 16% validation, and 20% test, applied via two sequential `train_test_split(..., shuffle=False)` calls. The test set always represents the most recent observations in the dataset and is never seen during training or hyperparameter selection. Using random splits on time-series data constitutes look-ahead leakage: a model trained on 2023 data could be evaluated against 2020 samples, producing optimistically biased metrics that do not reflect real deployment performance.

### Cross-Validation

Time-series cross-validation requires methods that respect the causal direction of time. Standard k-fold cross-validation is not appropriate because it randomly mixes past and future observations across folds. This project uses an expanding-window approach during the architecture sweep: for each candidate configuration, the training window extends from the chronological start of the data to a fixed cutoff, with validation on the subsequent block. This mirrors how a model would be retrained as new data accumulates, and guarantees that no validation sample precedes any training sample.

### Per-Window Normalisation

Raw price values vary enormously across companies and time periods. Rather than normalising globally — which leaks future statistics into the training distribution — each sliding window is independently normalised using per-feature min-max scaling:

```
X_norm[t] = (X[t] − min(window)) / (max(window) − min(window))
```

Target values (Close prices) are normalised against the same Close min/max from the input window and denormalised after inference to recover predictions in the original price scale. A flat-window filter discards any window whose Close range is below 0.1% of the mean price; these arise in illiquid periods where the denominator approaches zero, producing unstable training dynamics.

---

## Task 1 — Nasdaq Stock Price Prediction

### Data and Window Size

The Nasdaq dataset provides daily OHLCAV columns per company CSV. All six features are used: Open, High, Low, Close, Adjusted Close, and Volume. The single-feature demo relying only on Open discards the intraday range information encoded in the High–Low spread, the liquidity signal in Volume, and the split-adjusted history available through Adjusted Close. Including all six gives the model access to richer context about each trading day.

A sliding window of 30 trading days (approximately one and a half calendar months) was chosen as the lookback. This captures sufficient trend context for the smooth and liquid Nasdaq series without the risk of spanning multiple market regime changes.

### Architecture

The Nasdaq model is a pure CNN regressor: three Conv1D stages (64 → 128 → 64 filters, kernel size 3, same padding) each followed by MaxPooling1D(2), feeding a Dense(100) hidden layer and a linear output. Convolutional layers are well-suited here because local temporal patterns — momentum pulses, consolidation periods, single-day reversals — can be detected by a sliding kernel at multiple time scales simultaneously. No recurrent layers are used; for the relatively smooth Nasdaq series and the short forecast horizons tested, CNNs train faster and more stably.

Training uses Adam at learning rate 1e-2, early stopping with patience=10 (restoring best validation weights), and batch size 64. Trained models are cached to disk; subsequent notebook runs skip training entirely.

### Subtasks

- **Task 1.1 — Next-day forecast:** output is the Close price one timestep beyond the 30-day window.
- **Task 1.2 — Nth-day forecast:** the label index shifts by `forecast_day − 1` positions (`FORECAST_DAY = 3`), so the model predicts the Close three days ahead. The architecture is identical; only the target position changes.
- **Task 1.3 — k-day consecutive forecast:** the output layer is widened to `n_outputs = k` (`FORECAST_DAYS = 3`) and the model predicts a vector of three consecutive Close prices. Loss is MSE averaged across all output steps, giving equal gradient weight to each forecast day.

RMSE and MAE are computed separately per output day. Error increases with forecast horizon as expected — uncertainty compounds and the model's context window shrinks relative to the prediction point.

---

## Task 2 — Vietnam Stock Price Prediction

### Company Filtering

Only companies meeting all of the following criteria are used: at least 160 raw rows (corresponding to approximately 120 usable data points after indicator warm-up plus window buffer), all five OHLCV columns present without nulls, and a mean daily Volume above 1 (eliminating zero-liquidity records). This excludes recently listed companies, those with prolonged trading suspensions, and tickers with incomplete price records — all of which cannot be reliably windowed or normalised.

### Feature Engineering

Relying on raw OHLCV alone is insufficient for Vietnamese stocks, which are sparser and more volatile than Nasdaq. Price levels alone carry limited information about trend direction, momentum state, or risk regime. Five technical indicators are appended to the base features:

- **SMA5 and SMA20** — 5-day and 20-day simple moving averages. SMA5 tracks short-term momentum; SMA20 captures the medium-term trend. The spread between them encodes whether the market is in an accelerating or decelerating trend phase.
- **RSI14** — 14-day Relative Strength Index, bounded in [0, 100]. Values above 70 conventionally indicate overbought conditions; below 30, oversold. Including RSI gives the model an explicit encoding of momentum saturation that it would otherwise need to learn implicitly from price sequences alone.
- **MACD and MACD_signal** — difference between the 12-day and 26-day exponential moving averages, plus its 9-day signal line. The MACD histogram encodes short-term momentum relative to medium-term trend; sign flips in the histogram often precede price reversals.
- **Volatility** — 10-day rolling standard deviation of daily percentage returns. A real-time risk proxy that captures whether the stock is in a high-noise regime.

**Is it useful to add additional Vietnam data (dividends, financial ratios, industry analysis)?** Dividend history could in principle be incorporated to flag ex-dividend days, where prices mechanically drop by the payout amount — a signal a model without this context would misinterpret as genuine bearish price action. Financial ratios (P/E, debt-to-equity, ROE) are slow-moving quarterly signals that may not contribute meaningfully at the daily prediction horizon. Industry analysis could enable sector-normalised features. For this project, these are excluded; the chosen indicators already capture the relevant short-term dynamics, and adding slow-moving fundamentals to a daily-horizon regressor without careful temporal alignment risks introducing noise rather than signal.

After appending indicators and dropping the NaN warm-up rows (minimum 26 rows, driven by the EMA26 required for MACD), the feature dimension expands from 5 to 11.

### Architecture Search

Five candidate architectures were designed and benchmarked:

| Name | Description |
|------|-------------|
| v0 | Stacked LSTM(128 → 64) with dropout. Pure recurrent baseline. |
| v1 | Conv1D(64→128) + MaxPool + LSTM(64). Shallow hybrid. |
| v2 | 2×Conv(64)→Pool, 2×Conv(128)→Pool, LSTM(128), Dropout(0.3), Dense(64). Deeper hybrid. |
| v3 | Same CNN as v1, stacked LSTM(64→32). |
| v4 | CNN + LSTM with temporal attention (learned softmax weighting over timesteps). |

All five were benchmarked at window=30, then a reduced sweep of v0, v1, v2 was run across window sizes of 10, 20, 30, and 60 — twelve configurations in total. **v2 at window=20 produced the lowest RMSE and MAE** and was selected for all subsequent Vietnam tasks.

The paired convolutional blocks in v2 appear to extract richer local features than a single Conv1D layer — patterns like sharp single-day reversals and multi-day consolidation periods. The larger LSTM(128) then integrates those compressed features temporally. The attention mechanism in v4 did not outperform v2, likely because the 20-step window is short enough that the LSTM can implicitly weight relevant positions without an explicit attention layer.

Window size 20 winning over 30 and 60 is notable: longer windows risk spanning market-regime boundaries that introduce conflicting historical signal. The Vietnamese market experienced several significant corrections and re-ratings over the past five years; a 60-day lookback can mix pre- and post-correction data and hurt generalisation.

### Subtasks 2.1, 2.2, 2.3

These follow the same structure as Task 1 — next-day, nth-day with `VN_FORECAST_DAY = 3`, and k-day consecutive with `VN_FORECAST_DAYS = 3` — using the v2 architecture at window=20 and the 11-feature set. Prices are in VND.

---

## Task 3 — Trading Signal Identification

### Justification of Model Design

The regression models in Tasks 1 and 2 produce a continuous price estimate, which does not directly answer the question a trader needs: *is this a good time to act?* Task 3 reframes prediction as binary classification over two independent signals — a buy signal and a sell signal — each producing a probability in [0, 1] rather than a hard binary decision, which allows downstream logic to apply different confidence thresholds for different investor risk tolerances.

The two classifiers are trained independently. Buy signals tend to occur at the end of declining trends when RSI is depressed and volatility is falling; sell signals tend to occur near the peak of rallies when price is extended above moving averages. Training a single multi-output classifier would force shared representations across structurally different pattern types.

### Labelling Strategy

A combined past-future window of 20 steps is used for label generation (`SIGNAL_WINDOW_PAST = SIGNAL_WINDOW_FUTURE = 10`). For each timestep `t`:

- **Buy label = 1** if the current Close ≤ the 20th percentile of the price distribution across the [t−10, t+10] window — indicating proximity to a local trough.
- **Sell label = 1** if the current Close ≥ the 80th percentile of the same window — indicating proximity to a local peak.

The future window is used **only during label generation at training time**; at inference, the model receives only the past 10 steps as input. This is valid as long as the temporal split boundary is respected: no future observations beyond the test cutoff are used to label test samples.

### Is Manual Feature Engineering Useful for Signal Identification?

Yes — and this is the primary reason the same 11-feature set from Task 2 is used as input to the signal classifiers. RSI14 directly encodes the momentum saturation that the labelling strategy targets: a buy signal occurring when RSI is below 30 is a much stronger indicator than one at RSI = 60. MACD histogram sign flips often coincide with the local minima and maxima the labelling function identifies. SMA crossovers confirm trend direction. By supplying these features, the model receives pre-computed summaries of the exact dynamics its labels are based on, substantially simplifying the classification problem compared to raw OHLCV alone.

### Architecture and Class Balancing

Both classifiers use: LSTM(128, return\_sequences=True) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32, relu) → Dense(1, sigmoid). The recurrent architecture is chosen over pure CNN because buy/sell identification depends on the evolving *sequence* of RSI and MACD values — whether momentum is rising or falling over the window — not just local pattern shape. The LSTM can track the running state of these indicators across the 10-step input.

With 20th/80th percentile thresholds, approximately 20% of labels are positive. Class weights computed via `compute_class_weight('balanced')` reweight the binary cross-entropy loss so each class contributes equally to the gradient update. Early stopping with patience=5 prevents overfitting on the minority class. Evaluation uses precision, recall, and F1 per class; recall on the positive class is the primary metric, since a missed genuine signal carries more practical cost than a false alarm.

---

## Task 4 — Portfolio Composition, Risk Management, and Optimisation

### Company Filtering

The same filtering criteria from Task 2 apply across the full Vietnamese universe: at least 160 raw rows, no null values in required OHLCV columns, and non-negligible average volume. This targets companies with at least 120 usable data points after indicator warm-up. Very recently listed companies and those with prolonged suspension periods are automatically excluded — insufficient data prevents reliable windowed training.

### Task 4.1 — Profitable Stock Selection

A separate v2 CNN-LSTM model is trained per company using the same architecture and pipeline from Task 2. Pre-trained models for all filtered companies are committed to the `models/` directory in the repository; the notebook loads them directly without retraining. To retrain any model, delete the corresponding `task4_<ticker>.keras` file — `train_or_load_model` will train and save a new one automatically on the next run.

```
predicted_return = (predicted_close − last_actual_close) / last_actual_close
```

Companies are ranked by this value in descending order. Only those with `predicted_return > 0` advance as portfolio candidates; the top 10 are displayed as a shortlist. This uses the model's one-step-ahead estimate as a forward-looking proxy for short-term expected upside. The methodology is transparent and interpretable: the same models used for price prediction also drive the selection stage, with no separate profitability model required.

### Task 4.2 — Risk Management

Risk is quantified along two independent dimensions:

**Volatility** — standard deviation of the full historical daily return series. High-volatility companies are subject to large day-to-day price swings, increasing both upside potential and downside exposure.

**Maximum drawdown** — the worst peak-to-trough percentage decline across the entire price history:
```
max_drawdown = min((price − running_peak) / running_peak)
```
This is a tail-risk measure that volatility understates: a company can exhibit moderate day-to-day volatility but have suffered a single prolonged crash that makes it unsuitable for a conservative portfolio.

The composite risk score is:
```
risk_score = volatility + |max_drawdown|
```

Companies scoring above the 70th percentile of this distribution are excluded. Setting the threshold by percentile rather than an absolute value makes it robust to changes in market-wide volatility across different time periods. The 70th percentile removes the most volatile and crash-prone third of the candidate universe while retaining enough companies for meaningful diversification.

### Task 4.3 — Portfolio Composition

The final portfolio candidates are the intersection of profitable tickers (positive predicted return) and safe tickers (risk score below the 70th percentile). Two allocation strategies are constructed:

**Equal-weight — prudent investors.** Each candidate receives weight `1/N`. This is appropriate for investors who are uncertain about the model's return predictions and want to maximise diversification. The equal-weight portfolio bets that the filtered safe-profitable set will outperform the market on average, without concentrating exposure in any individual name. It is the lower-conviction, lower-variance choice.

**Return-weighted — risk-taking investors.** Each candidate receives weight proportional to its predicted return, normalised to sum to 1. This concentrates capital in the names with the highest model-estimated upside, increasing expected return if the model is well-calibrated but also increasing exposure to model error and concentration risk. A further median-return split distinguishes the most aggressive subset (predicted return ≥ median of the portfolio set, return-weighted) from the more conservative remainder (below median, equal-weighted).

In practical terms: a risk-taking investor holds fewer, higher-weight positions in the top predicted-return names; a prudent investor spreads exposure equally across all companies that passed both the profitability and risk filters.

---

## Task 5 — Model Deployment and SaaS Interface

### Task 5.1 — Model Deployment (FastAPI REST API)

The trained portfolio models are exposed via a FastAPI application (`stock_api.py`) with four endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe. Returns `{"status": "ok"}`. |
| `/tickers` | GET | Lists all tickers with a deployed model on disk. |
| `/window/{ticker}` | GET | Loads latest historical data, computes indicators, returns the final 20-row window and last Close price. |
| `/predict` | POST | Accepts a window array plus ticker and close_idx, normalises it, runs inference, and returns the denormalised predicted Close. |

**How to start the API server:**

The server launches as part of the normal notebook run. Open the notebook in Colab and select `Runtime → Run all`. The setup cell clones the repository (including pre-trained models), the server cell starts uvicorn in a background thread, and the tunnel cell polls until a `trycloudflare.com` URL appears — typically within 15 seconds. That URL is the live public endpoint.

If the tunnel cell fails on the first attempt, rerun it in isolation. Cloudflare's free accountless tunnels are occasionally unavailable; a second attempt is almost always sufficient.

To run the server outside of Colab:

```bash
cd cs313-stock-analysis
pip install fastapi uvicorn
uvicorn stock_api:app --host 0.0.0.0 --port 8000
```

**How to send a prediction request:**

A prediction requires two sequential calls. First, retrieve the latest input window for a ticker:

```bash
curl https://<tunnel-url>/window/ADG-VNINDEX-History
```

Sample response:
```json
{
  "window": [
    [24300.0, 24500.0, 24100.0, 24250.0, 150000.0, 24100.2, 24198.4, 52.3, -80.1, -60.4, 0.012],
    ...
  ],
  "last_close": 24250.0
}
```

Then POST the window to `/predict`:

```bash
curl -X POST https://<tunnel-url>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "ADG-VNINDEX-History",
    "window": [[24300.0, 24500.0, 24100.0, 24250.0, 150000.0, 24100.2, 24198.4, 52.3, -80.1, -60.4, 0.012], ...],
    "close_idx": 3
  }'
```

Sample response:
```json
{
  "ticker": "ADG-VNINDEX-History",
  "predicted_close": 24580.0
}
```

The `/window/{ticker}` endpoint handles all data loading and indicator computation server-side; the client only needs to pass the ticker name and relay the returned window.

---

### Task 5.2 — Web Interface (SaaS)

The FastAPI application serves a full HTML/CSS/JavaScript frontend directly at `GET /` via FastAPI's `HTMLResponse`, eliminating the need for a separate frontend server or build step.

**How to start the web interface:**

No additional setup is required beyond running the notebook. The frontend is served directly from the FastAPI application at `GET /` — once the tunnel URL is printed, navigating to it in any browser loads the interface immediately.

**How the interface calls the API:**

On page load, the JavaScript fetches `/tickers` and populates the company dropdown. On button click, it makes two sequential `fetch` calls:

1. `GET /window/{ticker}` — retrieves the pre-processed 20-row input window and last Close price from the server.
2. `POST /predict` — sends the window data and receives the predicted Close price.

Both calls use the native browser `fetch` API with no external libraries. Because `/window/{ticker}` handles all data preparation server-side, the client is fully stateless — it holds no price data between interactions.

**How to input values and read predictions:**

1. Open the tunnel URL in any browser.
2. Select a company from the dropdown (populated from `/tickers`).
3. Click **"Predict next-day close"**.
4. The result card displays the predicted Close price in Vietnamese locale format (e.g. `24,580 VND`) alongside the percentage change from the last known Close, coloured green for positive change and red for negative.

Error states — network failure, ticker not found (404), or inference failure — are surfaced in a visible error box below the button rather than failing silently.

---

## Conclusions

**Architecture.** The deeper CNN-LSTM hybrid (v2: paired convolutional blocks into LSTM128) outperformed both pure LSTM and shallower hybrids on Vietnamese data. The added convolutional depth extracts richer local features from the 11-feature input; the temporal attention mechanism in v4 did not justify its complexity for 20-step windows.

**Feature engineering.** Adding SMA5/20, RSI14, MACD, and rolling volatility to raw OHLCV improved performance, most visibly in the signal classification tasks where these indicators encode the exact dynamics the labelling strategy targets. Manual feature engineering remains valuable even when using deep learning — it reduces the amount of pattern-learning the model must do implicitly.

**Window size.** 20-day lookback won across architectures on Vietnamese data. Longer windows risk spanning market-regime boundaries; shorter windows lack sufficient trend context. Window size should always be treated as a hyperparameter and swept, not assumed.

**Normalisation.** Per-window min-max normalisation with flat-window filtering was the most critical preprocessing decision for stable cross-company training. Global normalisation would have biased learning toward high-priced names.

**Risk management.** Combining volatility and maximum drawdown into a percentile-based exclusion filter produced a transparent, data-driven approach to separating safe from risky candidates. The percentile threshold is adjustable to suit different portfolio risk mandates.

**Deployment.** FastAPI with an embedded HTML frontend and a Cloudflare tunnel delivers a complete REST API and browser-accessible SaaS interface from within the Colab environment. Pre-trained models are committed to the repository so the full pipeline — clone, run all, open URL — works without any retraining. The service is request-stateless, with model weights loaded from disk at startup and cached in memory for the session.

The most important methodological principle throughout: strict respect for temporal ordering. No shuffle, no look-ahead normalisation, no random k-fold. Correct data pipeline design is more consequential for result validity than any architectural choice.
