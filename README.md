# CSV-Based Quantitative Backtesting System – Attention + LSTM Factor Model

## **Project Overview**

This project implements a **quantitative trading backtesting system** based on **multi-factor stock selection** and the **Attention-LSTM hybrid modeling framework**.  
It combines **factor-driven stock filtering** with **slope-based market timing**, running entirely on **local CSV data** without any external API dependencies.

Unlike versions that rely on online platforms such as **THS (Tonghuashun)** or **JoinQuant**, this implementation is **fully offline**.  
It uses only `prices.csv` and `factors.csv` as inputs to replicate the model structure and logic, making it ideal for **academic display**, **algorithmic research**, and **educational purposes**.

> **Note:** Due to competition confidentiality, the core dataset, parameter tuning, and feature engineering details have been removed.  
> This version retains the complete framework and running logic to demonstrate the system’s architecture and workflow.

---

## **Model Structure and Principles**

The system integrates **three main components**:

### **1. Multi-Factor Stock Selection (Factor Filtering)**  
Selects stocks based on **profitability**, **growth potential**, and **valuation**.

Uses factors such as:
- `net_profit_growth_ratio` – Net profit growth ratio  
- `his_peg` – PEG ratio  
- `current_market_cap` – Market capitalization  

---

### **2. Slope-Based Timing Signal**  
Applies rolling linear regression between the **high** and **low** prices of a reference index (e.g., `000300.SH`):

```python
high = a + b * low
```
Slope (b) represents trend direction.

R² measures the strength of the trend.

The slope series is standardized (z-score) to produce timing signals:

BUY: z-score > 0.7

SELL: z-score < -0.7

KEEP: market remains neutral

---

### **3. Attention-Like Rebalancing Mechanism**

Adjusts holdings dynamically based on timing signals:

SELL → liquidate all positions

BUY/KEEP → allocate equally among top-ranked stocks

This mechanism is inspired by the Attention mechanism, focusing capital only on high-scoring stocks while filtering out noise.

---

### **Workflow**

The model operates through six key stages:

#### **Data Loading and Initialization**

-Load data from prices.csv and factors.csv.

-Initialize portfolio, benchmark, and regression parameters.

#### **Factor Filtering and Stock Pool Construction**

-Select top 10% by net_profit_growth_ratio.

-Select top 50% by his_peg.

-Sort remaining stocks by current_market_cap.

#### **Slope-Based Market Signal Generation**

-Perform rolling OLS regression to calculate slope and R².

-Convert to z-score for signal classification (BUY/SELL/KEEP).

#### **Trading Decision and Position Adjustment**

-SELL: close all holdings.

-BUY/KEEP: equally allocate among candidate stocks.

#### **Trade Execution and Logging**

-Simulate trades and update cash/positions.

-Save all trade logs to trades.csv.

#### **Equity Calculation and Performance Output**

-Compute total equity:

```Python
Equity = Cash + Σ(Position_i × Price_i)
```
-Save results to equity_curve.csv and plot portfolio growth.

---


### **Example Outputs**

**Equity Curve** `equity_curve.csv`

```Python
date, equity, signal
2023-01-03, 1000000.00, KEEP
2023-01-04, 1003721.56, BUY
...
```

**Trade Log** `trades.csv`

```Python
date, action, symbol, qty, price, gross_value, cash_before, cash_after, position_after_qty, day_signal
2023-01-04, BUY, 600519, 54.2, 1843.5, 99981.7, 1000000.00, 900018.3, 54.2, BUY
2023-02-07, SELL, 600519, 54.2, 1920.3, 104104.3, 900018.3, 1004122.6, 0.0, SELL
```

---

### **Data Confidentiality Statement**

Because this project was originally developed for **a quantitative trading competition and academic research,**
The following materials are intentionally excluded:

-Original datasets

-Feature engineering and preprocessing logic

-Model parameters and optimization details

The current version uses **synthetic CSV data** with the same structure, ensuring the **framework and logic are fully reproducible.**

**`This repository is intended to demonstrate the algorithmic logic and workflow, not to represent actual trading performance.`**

---

### **Dependencies**

-Python ≥ 3.8

-numpy

-pandas

-matplotlib

Install with:

```Python
pip install -r requirements.txt
```
---

### **Future Work**

-Integrate volatility targeting and risk parity weighting

-Add transaction cost and slippage modeling

-Extend to LSTM/Transformer-based timing signals

-Implement walk-forward validation

-Explore reinforcement learning for adaptive portfolio control

---

**License**

MIT License © 2025
For academic and educational use only.










