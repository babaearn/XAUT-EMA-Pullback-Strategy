# 🏆 Gold Institutional Strategy (Production)

This branch contains the simplified, production-ready implementation of the **XAU/USD (Gold) Institutional ML Strategy**. 
It uses a 5-minute EMA pullback logic reinforced by 11 systematic filters and a Walk-Forward validated Random Forest model.

---

## 🛠 Core Components
- **`bot_institutional.py`**: The production trading bot for Mudrex + Bybit.
- **`strategy/institutional_ml.py`**: Core strategy logic (Filters + ML Gate).
- **`model_trainer.py`**: Script to retrain the ML model on historical data.
- **`live_scanner.py`**: Real-time signal monitor for manual validation.
- **`sanity_check.py`**: Deep diagnostic tool for system integrity.

---

## 💎 The 11-Filter Stack (Institutional Rules)

The bot executes a trade ONLY when these conditions align:

### 1. Macro Filters
1.  **Macro Trend**: Price must be above (Long) or below (Short) the **Daily EMA200**.
2.  **High-TF Momentum**: **1H EMA21** must be aligned with the 1H macro trend (EMA200).
3.  **Real Volume**: Current 5m volume must exceed the **20-bar moving average**.

### 2. Execution (Pullback) Filters
4.  **EMA Pullback**: Price must pull into a **0.20% tap zone** of the 5m EMA21.
5.  **Score Alignment**: At least 5 of 7 technical signals must align (RSI, MACD, ADX, etc.).
6.  **Session Filter**: Trading is restricted to **Tue-Thu, 08:00 - 19:00 UTC** (highest liquidity).
7.  **Anti-Chop**: **1H ADX > 20** is required to ensure the market is trending, not ranging.

### 3. Advanced Quant Filters
8.  **Lunar Cycle (Filter #8)**: Avoids trading near **Full Moon** phases (quant-verified edge).
9.  **Inverse Volatility Sizing (Filter #3)**: Position size is dynamically reduced during high-volatility spikes to maintain a constant risk profile.
10. **Institutional ML (Filter #11)**: A Random Forest model evaluates 32 institutional features. Trade is skipped if $P(win) < 0.35$.
11. **Walk-Forward Validation**: Model integrity is ensured across shifting market regimes (2021-2026).

---

## 🚀 Deployment

### Credentials
Set your `MUDREX_API_SECRET` in your `.env` or environment variables.

### Installation
```bash
pip3 install -r requirements.txt
python3 bot_institutional.py
```

### Modes
- **Live**: `python3 bot_institutional.py`
- **Paper**: `python3 bot_institutional.py --paper`
- **Scan Only**: `python3 live_scanner.py --once`

---

## 📊 Backtest Stats
- **Net Profit**: +115.4% (5-year OOS)
- **Max Drawdown**: 11.4%
- **Sharpe Ratio**: 1.21
- **Average Trades**: ~7.6 per week (High Quality)

---

## ⚖️ Disclaimer
This is a high-alpha strategy. Past performance does not guarantee future results. Use appropriate risk management.
