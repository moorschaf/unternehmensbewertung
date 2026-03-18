from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup


def _extract_returns(series: pd.Series) -> pd.Series:
    series = series.dropna()
    return series.pct_change().dropna()


def _alpha_get_json(api_key: str, function: str, **params: str) -> dict:
    base_url = "https://www.alphavantage.co/query"
    query = {"function": function, "apikey": api_key}
    query.update(params)
    response = requests.get(base_url, params=query, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and ("Error Message" in payload or "Information" in payload):
        return {}
    return payload


def _load_alpha_close_series(api_key: str, symbol: str, outputsize: str = "full") -> pd.Series:
    payload = _alpha_get_json(
        api_key,
        "TIME_SERIES_DAILY_ADJUSTED",
        symbol=symbol,
        outputsize=outputsize,
    )
    series_node = payload.get("Time Series (Daily)", {}) if isinstance(payload, dict) else {}
    if not isinstance(series_node, dict) or not series_node:
        return pd.Series(dtype=float)

    rows = []
    for dt_str, row in series_node.items():
        if not isinstance(row, dict):
            continue
        close_val = row.get("5. adjusted close") or row.get("4. close")
        if close_val in (None, ""):
            continue
        try:
            rows.append((pd.to_datetime(dt_str), float(close_val)))
        except Exception:
            continue

    if not rows:
        return pd.Series(dtype=float)

    frame = pd.DataFrame(rows, columns=["Date", "Close"]).sort_values("Date")
    return pd.Series(frame["Close"].values, index=frame["Date"], name=symbol)


def _looks_like_rate_limit_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    return "ratelimit" in name or "rate limit" in msg or "too many requests" in msg


def _stooq_symbol(ticker: str) -> str:
    return f"{ticker.lower()}.us"


def _load_stooq_close(symbol: str) -> pd.Series:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = pd.read_csv(io.StringIO(response.text))
    if data.empty or "Date" not in data.columns or "Close" not in data.columns:
        return pd.Series(dtype=float)

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Date", "Close"]).sort_values("Date")
    if data.empty:
        return pd.Series(dtype=float)

    return pd.Series(data["Close"].values, index=data["Date"], name=symbol)


def _beta_from_series(
    stock_close: pd.Series,
    bench_close: pd.Series,
    mode: Literal["5Y Weekly", "2Y Daily"],
) -> float | None:
    years = 5 if mode == "5Y Weekly" else 2
    cutoff = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=365 * years)).tz_localize(None)

    stock = stock_close[stock_close.index >= cutoff]
    bench = bench_close[bench_close.index >= cutoff]
    if stock.empty or bench.empty:
        return None

    if mode == "5Y Weekly":
        stock = stock.resample("W-FRI").last().dropna()
        bench = bench.resample("W-FRI").last().dropna()

    stock_ret = _extract_returns(stock)
    mkt_ret = _extract_returns(bench)
    aligned = pd.concat([stock_ret, mkt_ret], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return None

    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)[0, 1]
    var = np.var(aligned.iloc[:, 1], ddof=1)
    if var <= 0:
        return None
    return float(cov / var)


@st.cache_data(show_spinner=False, ttl=2 * 60 * 60)
def estimate_beta(
    ticker: str,
    mode: Literal["5Y Weekly", "2Y Daily"],
    alpha_vantage_api_key: str | None = None,
) -> float | None:
    period = "5y" if mode == "5Y Weekly" else "2y"
    interval = "1wk" if mode == "5Y Weekly" else "1d"

    try:
        prices = yf.download(
            tickers=[ticker, "^GSPC"],
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if not prices.empty:
            close = prices.get("Close")
            if close is not None and ticker in close and "^GSPC" in close:
                beta = _beta_from_series(close[ticker], close["^GSPC"], mode)
                if beta is not None:
                    return beta
    except Exception as exc:
        if not _looks_like_rate_limit_error(exc):
            return None

    if alpha_vantage_api_key:
        try:
            stock_close = _load_alpha_close_series(alpha_vantage_api_key, ticker.upper())
            bench_close = _load_alpha_close_series(alpha_vantage_api_key, "SPY")
            beta = _beta_from_series(stock_close, bench_close, mode)
            if beta is not None:
                return beta
        except Exception:
            pass

    try:
        stock_close = _load_stooq_close(_stooq_symbol(ticker))
        bench_close = _load_stooq_close("^spx")
        return _beta_from_series(stock_close, bench_close, mode)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=2 * 60 * 60)
def get_market_snapshot(ticker: str, alpha_vantage_api_key: str | None = None) -> dict[str, float | None]:
    try:
        tk = yf.Ticker(ticker)
        info = tk.fast_info
        price = info.get("lastPrice")
        shares = info.get("shares")
        market_cap = info.get("marketCap")

        if market_cap is None and price is not None and shares is not None:
            market_cap = float(price) * float(shares)

        if price is not None or shares is not None or market_cap is not None:
            return {
                "price": float(price) if price is not None else None,
                "shares": float(shares) if shares is not None else None,
                "market_cap": float(market_cap) if market_cap is not None else None,
            }
    except Exception:
        pass

    if alpha_vantage_api_key:
        try:
            overview = _alpha_get_json(alpha_vantage_api_key, "OVERVIEW", symbol=ticker.upper())
            quote = _alpha_get_json(alpha_vantage_api_key, "GLOBAL_QUOTE", symbol=ticker.upper())

            shares_raw = overview.get("SharesOutstanding") if isinstance(overview, dict) else None
            market_cap_raw = overview.get("MarketCapitalization") if isinstance(overview, dict) else None
            quote_node = quote.get("Global Quote", {}) if isinstance(quote, dict) else {}
            price_raw = quote_node.get("05. price") if isinstance(quote_node, dict) else None

            shares = float(shares_raw) if shares_raw not in (None, "") else None
            market_cap = float(market_cap_raw) if market_cap_raw not in (None, "") else None
            price = float(price_raw) if price_raw not in (None, "") else None

            if market_cap is None and price is not None and shares is not None:
                market_cap = price * shares

            if price is not None or shares is not None or market_cap is not None:
                return {
                    "price": price,
                    "shares": shares,
                    "market_cap": market_cap,
                }
        except Exception:
            pass

    try:
        stock_close = _load_stooq_close(_stooq_symbol(ticker))
        if stock_close.empty:
            return {"price": None, "shares": None, "market_cap": None}
        return {
            "price": float(stock_close.iloc[-1]),
            "shares": None,
            "market_cap": None,
        }
    except Exception:
        return {"price": None, "shares": None, "market_cap": None}


@st.cache_data(show_spinner=False, ttl=12 * 60 * 60)
def get_risk_free_rate() -> float:
    csv_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    response = requests.get(csv_url, timeout=30)
    response.raise_for_status()

    lines = response.text.strip().splitlines()[1:]
    values = []
    for line in lines:
        _, value = line.split(",", 1)
        value = value.strip()
        if value:
            values.append(float(value) / 100.0)

    if not values:
        return 0.04
    return float(values[-1])


@st.cache_data(show_spinner=False, ttl=7 * 24 * 60 * 60)
def get_implied_erp() -> float:
    url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/implpr.html"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if table is None:
        return 0.05

    rows = table.find_all("tr")
    if len(rows) < 2:
        return 0.05

    header_cells = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
    target_idx = None
    for i, name in enumerate(header_cells):
        if "Implied Premium (FCFE)" in name or "Implied ERP (FCFE)" in name:
            target_idx = i
            break

    if target_idx is None:
        return 0.05

    latest = None
    for row in rows[1:]:
        cells = [c.get_text(strip=True) for c in row.find_all("td")]
        if len(cells) <= target_idx:
            continue
        value = cells[target_idx].replace("%", "").replace(",", "").strip()
        if not value:
            continue
        try:
            latest = float(value) / 100.0
        except ValueError:
            continue

    return float(latest) if latest is not None else 0.05
