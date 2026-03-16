# XAU/USD & XAUT Institutional ML Strategy

Institutional-grade EMA Pullback strategy for **XAU/USD** (Spot/Futures) and **XAUTUSDT** (Gold token futures). 
Trained on 5 years of historical gold data (Dukascopy) and validated using **Walk-Forward Expanding Window** machine learning.

- **Execution**: [Mudrex Futures API](https://docs.trade.mudrex.com/docs/overview)
- **Prices**: Bybit klines (Live) / Dukascopy (Backtest)
- **ML Model**: RandomForest Classifier with 32 institutional features.

---

## 💎 Institutional Confluence Stack

This bot uses 7 technical filters + 2 macro filters + 1 ML probability filter:

1.  **Filter #1 (Daily EMA200)**: Macro trend bias (only trade in direction of the daily trend).
2.  **Filter #3 (Inverse Volatility Sizing)**: Dynamic position sizing based on ATR regime.
3.  **Filter #8 (Lunar Cycle)**: Avoids trading near Full Moons (systematically validated edge).
4.  **Filter #11 (Institutional ML)**: RandomForest model trained on 11,500+ candidates.
5.  **Multitimeframe Bias**: 1H EMA21 > 1H EMA200 + 1H ADX > 20.
6.  **Momentum**: 5m RSI (50-70 for longs, 30-50 for shorts).
7.  **Microstructure**: Volume > 20-bar MA + Candle Body Ratio checks.

---

## 📂 Project Structure

- `bot_institutional.py`: Main entry point for starting the bot on Mudrex.
- `model_trainer.py`: Trains the ML model on all historical data and saves artifacts.
- `live_scanner.py`: Live monitor for signals using Yahoo Finance data.
- `backtest.py` / `filter_test_*.py`: Research scripts for each strategy iteration.
- `sanity_check.py`: Deep diagnostic script to verify 50+ points of logic integrity.
- `saved_model/`: Directory containing `xauusd_model.joblib` and scaling configuration.

---

## 🚀 How to Run on Mudrex

### 1. Local Setup
```bash
pip3 install -r requirements.txt
python3 model_trainer.py
```

### 2. Configuration
Create a `.env` file with your Mudrex API credentials:
```env
MUDREX_API_SECRET=your_mudrex_secret
```

### 3. Deploy
The project is configured for **Railway** or any Linux server:
```bash
# Start the institutional bot
python3 bot_institutional.py
```

Or for paper-trading:
```bash
python3 bot_institutional.py --paper
```

---

## 📊 Backtest Performance (2021-2026)

| Metric | Baseline (F1+F3+F8) | Institutional ML (Final) |
|---|---|---|
| **Net PnL** | +109.6% | **+115.4%** |
| **Max Drawdown** | 22.7% | **11.4%** |
| **Sharpe Ratio** | 0.80 | **1.21** |
| **Profit Factor** | 1.14x | **1.35x** |
| **Avg Trades/Week** | 7.8 | **7.6** |

---

## 🛠 Advanced Features

- **Forward-Simulation Labeling**: Unlike naive ML bots that only train on past trades, this bot labels *every* candidate bar to learn why some pullbacks fail and others succeed.
- **Walk-Forward Validation**: Model parameters were verified across 5 years using shifting windows to ensure no overfitting to specific market regimes.
- **Micro-Animations (Charts)**: High-quality visualization scripts included in `visualize_institutional.py`.

---

## ⚖️ Disclaimer
This is a research project. Trading involves significant risk. Always test on paper before going live with real capital.
