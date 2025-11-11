# CSV-Based-Quantitative-Backtesting-System-Attention-LSTM-Factor-Model
# CSV-Based Quantitative Backtesting System – Attention + LSTM Factor Model (Demonstration Version)

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
