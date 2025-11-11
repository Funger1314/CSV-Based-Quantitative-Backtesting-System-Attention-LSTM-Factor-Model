# CSV-Based-Quantitative-Backtesting-System-Attention-LSTM-Factor-Model
Project Overview

This project implements a quantitative trading backtesting system based on multi-factor stock selection and the Attention-LSTM hybrid modeling framework.
It combines factor-driven stock filtering with slope-based market timing, running entirely on local CSV data without any external API dependencies.

Unlike versions that rely on online platforms such as THS (Tonghuashun) or JoinQuant, this implementation is fully offline.
It uses only prices.csv and factors.csv as inputs to replicate the model structure and logic, making it ideal for academic display, algorithm research, and educational purposes.

Note: Due to competition confidentiality, the core dataset, parameter tuning, and feature engineering details have been removed.
This version retains the complete framework and running logic to demonstrate the system’s architecture and workflow.

Model Structure and Principles

The system integrates three main components:

Multi-Factor Stock Selection (Factor Filtering)

Selects stocks based on profitability, growth potential, and valuation.

Uses factors such as:

Net profit growth ratio (net_profit_growth_ratio)

PEG ratio (his_peg)

Market capitalization (current_market_cap)

Slope-Based Timing Signal

Applies rolling linear regression between the high and low prices of a reference index (e.g., CSI 300, 000300.SH):

high = a + b * low


The slope (b) represents trend direction, and the R² value measures trend strength.

The slope series is standardized (z-score) to produce timing signals:

BUY: z-score > threshold (e.g., 0.7)

SELL: z-score < -threshold

KEEP: market remains neutral

Attention-Like Rebalancing Mechanism

Adjusts holdings dynamically based on the timing signal:

SELL → liquidate all positions

BUY/KEEP → allocate equally among top-ranked stocks

Inspired by the attention mechanism, the strategy focuses only on high-scoring, important stocks while ignoring noise and low-quality signals.

Workflow Description

The model operates through six main stages:

1. Data Loading and Initialization

Reads price and factor data from prices.csv and factors.csv.

Initializes the backtest environment, including cash, stock universe, benchmark index, and regression parameters.

2. Factor Filtering and Stock Pool Construction

Filters stocks based on multi-stage criteria:

Selects top 10% by net profit growth.

Selects top 50% by PEG ratio (lower is better).

Sorts by market capitalization and retains smaller-cap stocks.

The resulting list forms the investable stock pool for that period.

3. Slope-Based Market Signal

Performs rolling OLS regression on the reference index to calculate slope and R².

Standardizes the slope sequence to obtain a z-score trend signal.

Generates trading signals based on the z-score threshold (BUY / SELL / KEEP).

4. Trading Decision and Position Adjustment

SELL → close all open positions.

BUY/KEEP → equally allocate cash across the candidate stock pool.

This module implements “attention-like” rebalancing: focusing capital only on top-scoring stocks to reduce unnecessary trades.

5. Trade Execution and Logging

Simulates each transaction (buy/sell) and updates cash and position records.

Logs all trades including date, symbol, price, quantity, cash before/after, and signal status.

All records are saved to trades.csv.

6. Equity Calculation and Performance Output

Computes daily total portfolio value:

Equity = Cash + Σ(Position_i × Price_i)


Outputs the equity curve to equity_curve.csv and plots portfolio growth over time.

File Descriptions
File	Description
prices.csv	Historical stock price data (open, high, low, close, volume)
factors.csv	Fundamental factor data (profit growth, PEG, market cap, etc.)
backtest.py	Main script controlling the backtest logic
equity_curve.csv	Daily equity curve output
trades.csv	Detailed transaction log with full trade history
Example Outputs

Equity Curve (equity_curve.csv)

        date        equity  signal
0  2023-01-03  1000000.00    KEEP
1  2023-01-04  1003721.56     BUY
...


Trade Log (trades.csv)

        date  action symbol     qty   price  gross_value  cash_before  cash_after  position_after_qty day_signal
0  2023-01-04     BUY  600519   54.2  1843.5     99981.7   1000000.00   900018.3              54.2        BUY
1  2023-02-07    SELL  600519   54.2  1920.3    104104.3    900018.3   1004122.6               0.0       SELL

Data Confidentiality Statement

Because this project was originally developed for a quantitative investment competition and academic research,
the following materials have been intentionally excluded from the public version:

Original training and testing datasets

Feature engineering and data preprocessing logic

Parameter tuning and optimization settings

The current release contains synthetic example data with identical structure to demonstrate the framework and workflow.
The model logic and computational pipeline remain intact for transparency and reproducibility.

This repository aims to illustrate the design philosophy, architecture, and algorithmic logic, not to reflect real trading performance.

Dependencies

Python ≥ 3.8

numpy

pandas

matplotlib

Install required libraries:

pip install -r requirements.txt

Future Work

Integrate volatility targeting and risk parity weighting

Add transaction cost and slippage modeling

Extend the timing module with LSTM/Transformer-based signal generation

Implement walk-forward validation for robustness testing

Incorporate reinforcement learning for adaptive portfolio control

License

MIT License © 2025
For academic and educational use only. Commercial applications are not permitted.
