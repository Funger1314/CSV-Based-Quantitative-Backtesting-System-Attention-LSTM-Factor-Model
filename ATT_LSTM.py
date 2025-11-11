import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt

# 1. Load local CSV data
prices = pd.read_csv("prices.csv", parse_dates=["date"])
factors = pd.read_csv("factors.csv", parse_dates=["date"])

# 2. Helper functions
def history_df(symbol, fields, n, prices_df, end_date=None):
    """Return the last n rows of selected fields for a symbol."""
    df = prices_df.loc[prices_df["symbol"] == symbol].copy()
    if end_date is None:
        end_date = df["date"].max()
    end_date = pd.Timestamp(end_date)
    df = df[df["date"] <= end_date].sort_values("date")
    return df.tail(n)[["date"] + fields].set_index("date")

def OLS(x, y):
    """Simple linear regression returning intercept, slope, R-like signal."""
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r_signal = 1 - (np.sum(resid**2) / ((len(y) - 1) * np.var(y, ddof=1)))
    return intercept, slope, r_signal

def zscore_last(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2 or np.std(arr) == 0:
        return 0.0
    return (arr[-1] - np.mean(arr)) / np.std(arr)

# 3. Basic backtesting structures
@dataclass
class Position:
    amount: float = 0.0
    cost: float = 0.0

@dataclass
class Portfolio:
    cash: float = 1_000_000.0
    positions: Dict[str, Position] = field(default_factory=dict)

    def total_value(self, price_map):
        return self.cash + sum(
            pos.amount * price_map.get(sym, 0.0) for sym, pos in self.positions.items()
        )

@dataclass
class Context:
    portfolio: Portfolio
    universe: List[str]
    ref_symbol: str
    stock_num: int = 30
    N: int = 22
    M: int = 545
    score_threshold: float = 0.7
    slope_series: List[float] = field(default_factory=list)

# 4. Factor-based stock filtering
def factor_filter(stock_list, column, ascending, proportion, factors_df, as_of_date):
    as_of_date = pd.Timestamp(as_of_date)
    df = factors_df[(factors_df["date"] == as_of_date) & (factors_df["symbol"].isin(stock_list))]
    if df.empty or column not in df.columns:
        cutoff = max(1, int(proportion * len(stock_list)))
        return stock_list[:cutoff]
    df = df[["symbol", column]].dropna().rename(columns={"symbol": "code", column: "score"})
    df = df.sort_values("score", ascending=ascending)
    top_k = max(1, int(proportion * len(stock_list)))
    return df["code"].head(top_k).tolist()

def get_stock_list(context, as_of_date, factors_df):
    step1 = factor_filter(context.universe, "net_profit_growth_ratio", False, 0.10, factors_df, as_of_date)
    step2 = factor_filter(step1, "his_peg", True, 0.50, factors_df, as_of_date)
    df = factors_df[(factors_df["date"] == as_of_date) & (factors_df["symbol"].isin(step2))]
    if "current_market_cap" in df.columns:
        df = df[["symbol", "current_market_cap"]].dropna().sort_values("current_market_cap", ascending=True)
        return df["symbol"].tolist()
    return step2

# 5. Trading helpers WITH LOGGING
def log_trade(trades: List[dict], date, action, symbol, qty, price, cash_before, cash_after, pos_after, signal):
    gross = qty * price
    trades.append({
        "date": pd.Timestamp(date),
        "action": action,               # "BUY" or "SELL"
        "symbol": symbol,
        "qty": float(qty),
        "price": float(price),
        "gross_value": float(gross),
        "cash_before": float(cash_before),
        "cash_after": float(cash_after),
        "position_after_qty": float(pos_after),
        "day_signal": signal            # the portfolio-level timing signal of that day
    })

def open_position(context, symbol, value, price, date, signal, trades):
    """Buy by target value; write trade log."""
    if price is None or price <= 0 or value <= 0:
        return
    qty = value / price
    cash_before = context.portfolio.cash
    context.portfolio.cash -= value
    pos = context.portfolio.positions.get(symbol, Position())
    pos.amount += qty
    pos.cost = price
    context.portfolio.positions[symbol] = pos
    log_trade(trades, date, "BUY", symbol, qty, price, cash_before, context.portfolio.cash, pos.amount, signal)

def close_position(context, symbol, price, date, signal, trades):
    """Sell all; write trade log."""
    pos = context.portfolio.positions.get(symbol)
    if (pos is None) or (pos.amount <= 0) or (price is None) or (price <= 0):
        return
    qty = pos.amount
    cash_before = context.portfolio.cash
    proceeds = qty * price
    context.portfolio.cash += proceeds
    # after selling, position is removed
    del context.portfolio.positions[symbol]
    log_trade(trades, date, "SELL", symbol, qty, price, cash_before, context.portfolio.cash, 0.0, signal)

def ATT_rebalance(context, buy_list, date, prices_df, day_signal, trades):
    """Attention-like rebalancing with trade logs."""
    date = pd.Timestamp(date)

    # 1) Sell names not in buy_list
    for sym in list(context.portfolio.positions.keys()):
        if sym not in buy_list:
            row = prices_df.query("symbol == @sym and date == @date")
            if not row.empty:
                price = float(row["close"].iloc[0])
                close_position(context, sym, price, date, day_signal, trades)

    # 2) Equal-weight buy new slots
    slots = context.stock_num - len(context.portfolio.positions)
    if slots <= 0:
        return
    per_val = context.portfolio.cash / slots if slots > 0 else 0.0
    for sym in buy_list[:slots]:
        # Skip if already held
        if sym in context.portfolio.positions:
            continue
        row = prices_df.query("symbol == @sym and date == @date")
        if not row.empty and per_val > 0:
            price = float(row["close"].iloc[0])
            open_position(context, sym, per_val, price, date, day_signal, trades)

# 6. Slope-based timing signal
def initial_slopes(prices_df, ref_symbol, N, M, end_date):
    end_date = pd.Timestamp(end_date)
    hl = history_df(ref_symbol, ["high", "low"], N + M, prices_df, end_date)
    slopes = []
    for i in range(len(hl) - N + 1):
        _, slope, _ = OLS(hl["low"].iloc[i:i+N], hl["high"].iloc[i:i+N])
        slopes.append(slope)
    return slopes

def trade_signal(context, prices_df, ref_symbol, as_of_date):
    as_of_date = pd.Timestamp(as_of_date)
    hl = history_df(ref_symbol, ["high", "low"], context.N, prices_df, as_of_date)
    if len(hl) < context.N:
        return "KEEP"
    _, slope, r_sig = OLS(hl["low"], hl["high"])
    context.slope_series.append(slope)
    score = zscore_last(context.slope_series[-context.M:]) * r_sig
    if score > context.score_threshold: return "BUY"
    elif score < -context.score_threshold: return "SELL"
    return "KEEP"

# 7. Main backtest loop (now also returns trades DataFrame)
def backtest(prices_df, factors_df, ref_symbol="000300.SH", stock_num=3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_dates = sorted(prices_df["date"].unique())
    universe = [s for s in prices_df["symbol"].unique() if s != ref_symbol]
    context = Context(portfolio=Portfolio(), universe=universe, ref_symbol=ref_symbol, stock_num=stock_num)

    # Initialize slope history
    pre_day = prices_df["date"].min()
    context.slope_series = initial_slopes(prices_df, ref_symbol, context.N, context.M, pre_day)[:-1]

    equity_curve = []
    trades: List[dict] = []

    for d in all_dates:
        price_today = {r.symbol: float(r.close) for r in prices_df[prices_df["date"] == d].itertuples()}
        buy_list = get_stock_list(context, d, factors_df)[:context.stock_num]
        signal = trade_signal(context, prices_df, ref_symbol, d)

        if signal == "SELL":
            # Liquidate all and record trades
            for sym in list(context.portfolio.positions.keys()):
                px = price_today.get(sym)
                if px:
                    close_position(context, sym, px, d, signal, trades)
        else:
            # Rebalance towards attention list (with logs)
            ATT_rebalance(context, buy_list, d, prices_df, signal, trades)

        equity = context.portfolio.total_value(price_today)
        equity_curve.append({"date": d, "equity": equity, "signal": signal})

    result = pd.DataFrame(equity_curve).sort_values("date").reset_index(drop=True)
    result["ret"] = result["equity"].pct_change().fillna(0.0)

    trades_df = pd.DataFrame(trades).sort_values("date").reset_index(drop=True)
    return result, trades_df

# 8. Run the demo backtest
if __name__ == "__main__":
    eq, trades = backtest(prices, factors, "000300.SH", stock_num=3)
    eq.to_csv("equity_curve.csv", index=False)
    trades.to_csv("trades.csv", index=False)

    # Plot equity
    plt.plot(eq["date"], eq["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("Equity tail:")
    print(eq.tail())
    print("\nTrades head:")
    print(trades.head())
    print("\nTrades tail:")
    print(trades.tail())


