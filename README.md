# Financial Models Workspace

A Streamlit multipage application with two separate tools:

- Option Pricer (Black-Scholes + CRR binomial)
- DCF Analyzer for US listed companies (SEC data + CAPM-based WACC)

## Features

### Option Pricer

- European and American option pricing
- Barrier options (Knock-In/Knock-Out)
- Greek calculations (Delta, Gamma, Vega, Rho, Theta)
- Implied volatility solver
- Interactive sensitivity curves
- Heatmaps for spot x volatility
- Dividend yield and discrete dividend support

### DCF Analyzer

- SEC filing data import for US tickers (historical financial statements)
- Historical view for income statement, balance sheet, and cash flow
- Automatic CAPM/WACC estimate with user overrides
- Beta mode selection: 5Y weekly or 2Y daily
- Fallback market data path: yfinance -> Alpha Vantage (optional API key) -> Stooq
- Five-year FCFF forecast with editable growth and margin assumptions
- Terminal value using perpetual growth rate
- Sensitivity table for WACC x terminal growth

## Usage

```bash
pip install -r requirements.txt
streamlit run app.py
```
