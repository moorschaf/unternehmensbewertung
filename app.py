from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dcf_model import build_sensitivity_table, run_dcf
from financials_mapper import FinancialBundle, map_company_facts_to_financials
from market_data import estimate_beta, get_implied_erp, get_market_snapshot, get_risk_free_rate
from sec_client import get_company_facts, get_company_ticker_map
from wacc import compute_wacc, cost_of_equity_capm, estimate_cost_of_debt, estimate_tax_rate, valid_terminal_setup


st.set_page_config(page_title="DCF Analyzer", layout="wide")


HISTORY_YEARS = 5


def pct_or_default(value: float | None, default: float) -> float:
    if value is None or not np.isfinite(value):
        return default
    return float(value)


def hist_mean(series: pd.Series, fallback: float, tail: int = 3) -> float:
    values = series.dropna().tail(tail)
    if values.empty:
        return fallback
    return float(values.mean())


def hist_cagr(series: pd.Series, fallback: float) -> float:
    values = series.dropna()
    if len(values) < 3:
        return fallback
    start = float(values.iloc[0])
    end = float(values.iloc[-1])
    n = len(values) - 1
    if start <= 0 or end <= 0 or n <= 0:
        return fallback
    return float((end / start) ** (1 / n) - 1)


def _tail_window(series: pd.Series, years: int | None) -> pd.Series:
    clean = series.dropna().sort_index()
    if years is None:
        return clean
    return clean.tail(max(years, 1))


def blended_revenue_growth(revenue_series: pd.Series, yoy_series: pd.Series, years: int | None) -> float:
    revenue_hist = _tail_window(revenue_series, None if years is None else years + 1)
    yoy_hist = _tail_window(yoy_series, years)

    cagr = hist_cagr(revenue_hist, np.nan)
    yoy_mean = hist_mean(yoy_hist, np.nan, tail=max(years or 5, 1))

    candidates = [x for x in [cagr, yoy_mean] if np.isfinite(x)]
    if not candidates:
        return 0.05
    if len(candidates) == 1:
        return float(candidates[0])
    return float(0.7 * cagr + 0.3 * yoy_mean)


def blended_ebit_margin(margin_series: pd.Series, years: int | None) -> float:
    margin_hist = _tail_window(margin_series, years)
    if margin_hist.empty:
        return 0.15

    avg = float(margin_hist.mean())
    med = float(margin_hist.median())
    return float(0.6 * avg + 0.4 * med)


def trend_next_value(series: pd.Series, years: int | None, fallback: float) -> float:
    hist = _tail_window(series, years)
    if len(hist) < 3:
        return fallback

    y = hist.values.astype(float)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    estimate = intercept + slope * float(len(y))
    if not np.isfinite(estimate):
        return fallback
    return float(estimate)


def robust_trend_next_value(series: pd.Series, years: int | None, fallback: float) -> float:
    hist = _tail_window(series, years)
    if len(hist) < 3:
        return fallback

    y = hist.values.astype(float)
    x = np.arange(len(y), dtype=float)

    slopes = []
    for i in range(len(y) - 1):
        for j in range(i + 1, len(y)):
            dx = x[j] - x[i]
            if dx == 0:
                continue
            slopes.append((y[j] - y[i]) / dx)

    if not slopes:
        return fallback

    slope = float(np.median(np.array(slopes, dtype=float)))
    intercepts = y - slope * x
    intercept = float(np.median(intercepts))
    estimate = intercept + slope * float(len(y))

    if not np.isfinite(estimate):
        return fallback
    return float(estimate)


def to_millions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col] / 1_000_000.0
    return out


def format_statement(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = to_millions(df)
    out.index.name = "Fiscal Year"
    return out.round(2)


def to_editable_millions(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    view = df.copy()
    for col in columns:
        if col not in view.columns:
            view[col] = np.nan
    view = view[columns].copy()
    view = (view / 1_000_000.0).round(2)
    view.index.name = "Fiscal Year"
    return view


def from_editable_millions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().apply(pd.to_numeric, errors="coerce")
    out.index = pd.to_numeric(out.index, errors="coerce")
    out = out[~out.index.isna()]
    out.index = out.index.astype(int)
    out = out.sort_index()
    return out * 1_000_000.0


def reset_editor_state(editor_key: str) -> None:
    if editor_key in st.session_state:
        del st.session_state[editor_key]
    st.rerun()


st.title("DCF Analyzer (US Listed Companies)")
st.caption("SEC-Historie -> CAPM-WACC -> 5Y Forecast -> Terminal Value")

with st.sidebar:
    st.header("Company")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    alpha_vantage_api_key = st.text_input(
        "Alpha Vantage API Key (optional)",
        value="",
        type="password",
        help="Optional fallback fuer Marktdaten (Beta, Price, Shares, Market Cap), falls Yahoo rate-limited ist.",
    ).strip()
    user_agent = st.text_input(
        "SEC User-Agent",
        value="DCFResearchTool/1.0 your-email@example.com",
        help="SEC requires identifying User-Agent (Name + contact).",
    )
    load_clicked = st.button("Load SEC + Market Data", type="primary")


if load_clicked:
    if not ticker:
        st.error("Bitte einen Ticker eingeben.")
        st.stop()

    with st.spinner("Loading SEC and market data..."):
        ticker_map = get_company_ticker_map(user_agent)
        if ticker not in ticker_map:
            st.error("Ticker nicht in SEC-Ticker-Liste gefunden.")
            st.stop()

        cik = int(ticker_map[ticker]["cik"])
        company_name = str(ticker_map[ticker]["title"])
        facts = get_company_facts(cik, user_agent)
        bundle = map_company_facts_to_financials(facts)
        market = get_market_snapshot(ticker, alpha_vantage_api_key=alpha_vantage_api_key or None)

        st.session_state["dcf_context"] = {
            "ticker": ticker,
            "cik": cik,
            "company_name": company_name,
            "bundle": bundle,
            "market": market,
            "alpha_vantage_api_key": alpha_vantage_api_key or None,
        }


ctx = st.session_state.get("dcf_context")
if not ctx:
    st.info("Gib links einen Ticker ein und lade die Daten.")
    st.stop()

bundle: FinancialBundle = ctx["bundle"]
market = ctx["market"]

st.subheader(f"{ctx['company_name']} ({ctx['ticker']})")
meta_cols = st.columns(4)
meta_cols[0].metric("CIK", f"{ctx['cik']}")
meta_cols[1].metric("Market Price", "n/a" if market["price"] is None else f"{market['price']:.2f}")
meta_cols[2].metric("Market Cap", "n/a" if market["market_cap"] is None else f"{market['market_cap'] / 1e9:.2f} Bn")
meta_cols[3].metric("Shares (Market)", "n/a" if market["shares"] is None else f"{market['shares'] / 1e6:.2f} M")

if market["market_cap"] is None or market["shares"] is None:
    st.info(
        "Market Cap/Shares konnten nicht automatisch geladen werden (z. B. wegen Rate Limits). "
        "Du kannst E, D, Cash und Shares unten manuell setzen."
    )

tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
income_cols = ["revenue", "ebit", "pretax_income", "tax_expense", "interest_expense", "shares"]
balance_cols = ["cash", "current_assets", "current_liabilities", "short_debt", "long_debt"]
cashflow_cols = ["da", "capex"]

all_statement_years = sorted(
    set(bundle.income_df.index) | set(bundle.balance_df.index) | set(bundle.cashflow_df.index)
)
recent_years = all_statement_years[-HISTORY_YEARS:]

income_source = bundle.income_df[bundle.income_df.index.isin(recent_years)].sort_index()
balance_source = bundle.balance_df[bundle.balance_df.index.isin(recent_years)].sort_index()
cashflow_source = bundle.cashflow_df[bundle.cashflow_df.index.isin(recent_years)].sort_index()

if len(recent_years) < HISTORY_YEARS:
    st.warning(
        f"Nur {len(recent_years)} historische Jahre verfuegbar. "
        f"Die Analyse nutzt alle verfuegbaren Jahre (max. {HISTORY_YEARS})."
    )

st.markdown(f"**Historical Input Window:** Last {HISTORY_YEARS} Fiscal Years")
if recent_years:
    st.caption(f"Verwendete Fiskaljahre: {recent_years[0]} bis {recent_years[-1]}")

with tab1:
    st.caption(f"Editierbar, Werte in USD Mio. (shares in Mio. shares) - letzte {HISTORY_YEARS} Jahre")
    income_editor_key = "income_statement_editor_5y"
    if st.button("Reset Income auf SEC", key="reset_income_btn"):
        reset_editor_state(income_editor_key)
    income_edit = st.data_editor(
        to_editable_millions(income_source, income_cols),
        use_container_width=True,
        key=income_editor_key,
    )
with tab2:
    st.caption(f"Editierbar, Werte in USD Mio. - letzte {HISTORY_YEARS} Jahre")
    balance_editor_key = "balance_sheet_editor_5y"
    if st.button("Reset Balance auf SEC", key="reset_balance_btn"):
        reset_editor_state(balance_editor_key)
    balance_edit = st.data_editor(
        to_editable_millions(balance_source, balance_cols),
        use_container_width=True,
        key=balance_editor_key,
    )
with tab3:
    st.caption(f"Editierbar, Werte in USD Mio. - letzte {HISTORY_YEARS} Jahre")
    cashflow_editor_key = "cashflow_statement_editor_5y"
    if st.button("Reset Cashflow auf SEC", key="reset_cashflow_btn"):
        reset_editor_state(cashflow_editor_key)
    cashflow_edit = st.data_editor(
        to_editable_millions(cashflow_source, cashflow_cols),
        use_container_width=True,
        key=cashflow_editor_key,
    )

income_df = from_editable_millions(income_edit)
balance_df = from_editable_millions(balance_edit)
cashflow_df = from_editable_millions(cashflow_edit)

all_years = sorted(set(income_df.index) | set(balance_df.index) | set(cashflow_df.index))
base = pd.DataFrame(index=all_years)
for col in income_cols:
    if col in income_df.columns:
        base[col] = income_df[col]
for col in balance_cols:
    if col in balance_df.columns:
        base[col] = balance_df[col]
for col in cashflow_cols:
    if col in cashflow_df.columns:
        base[col] = cashflow_df[col]

if base.empty:
    st.error("Keine historischen Daten verfuegbar. Bitte Statements laden oder manuell erfassen.")
    st.stop()

base["capex_abs"] = base["capex"].abs()
base["debt_total"] = base[["short_debt", "long_debt"]].sum(axis=1, min_count=1)
base["nwc"] = (
    base["current_assets"]
    - base["cash"]
    - (base["current_liabilities"] - base["short_debt"])
)

driver = pd.DataFrame(index=base.index)
driver["revenue_growth"] = base["revenue"].pct_change()
driver["ebit_margin"] = base["ebit"] / base["revenue"]
driver["da_pct_rev"] = base["da"] / base["revenue"]
driver["capex_pct_rev"] = base["capex_abs"] / base["revenue"]
driver["nwc_pct_rev"] = base["nwc"] / base["revenue"]

income_hist = base[income_cols].copy()

latest_year = int(base.index.max())
latest = {
    "year": float(latest_year),
    "revenue": float(base.loc[latest_year, "revenue"]) if pd.notna(base.loc[latest_year, "revenue"]) else np.nan,
    "ebit": float(base.loc[latest_year, "ebit"]) if pd.notna(base.loc[latest_year, "ebit"]) else np.nan,
    "pretax_income": float(base.loc[latest_year, "pretax_income"]) if pd.notna(base.loc[latest_year, "pretax_income"]) else np.nan,
    "tax_expense": float(base.loc[latest_year, "tax_expense"]) if pd.notna(base.loc[latest_year, "tax_expense"]) else np.nan,
    "da": float(base.loc[latest_year, "da"]) if pd.notna(base.loc[latest_year, "da"]) else np.nan,
    "capex": float(base.loc[latest_year, "capex_abs"]) if pd.notna(base.loc[latest_year, "capex_abs"]) else np.nan,
    "nwc": float(base.loc[latest_year, "nwc"]) if pd.notna(base.loc[latest_year, "nwc"]) else np.nan,
    "cash": float(base.loc[latest_year, "cash"]) if pd.notna(base.loc[latest_year, "cash"]) else 0.0,
    "debt": float(base.loc[latest_year, "debt_total"]) if pd.notna(base.loc[latest_year, "debt_total"]) else 0.0,
    "interest_expense": float(base.loc[latest_year, "interest_expense"]) if pd.notna(base.loc[latest_year, "interest_expense"]) else np.nan,
    "shares": float(base.loc[latest_year, "shares"]) if pd.notna(base.loc[latest_year, "shares"]) else np.nan,
}

st.subheader("Historische Treiber fuer Schaetzung")
history_years = min(HISTORY_YEARS, len(base.index))
st.caption(f"Schaetzung basiert auf den letzten {history_years} verfuegbaren Jahren.")

estimation_method = st.selectbox(
    "Schaetzmethode",
    ["Historischer Mix", "Trend (lineare Regression)", "Robuster Trend (Median-Slope)"],
    index=0,
    help="Historischer Mix nutzt CAGR/Mittelwerte. Trend nutzt lineare Regression. Robuster Trend nutzt Median-Slope und ist weniger ausreisser-empfindlich.",
)

mix_revenue_growth = blended_revenue_growth(
    income_hist["revenue"],
    driver["revenue_growth"],
    history_years,
)
mix_ebit_margin = blended_ebit_margin(driver["ebit_margin"], history_years)

trend_revenue_growth = trend_next_value(driver["revenue_growth"], history_years, mix_revenue_growth)
trend_ebit_margin = trend_next_value(driver["ebit_margin"], history_years, mix_ebit_margin)
robust_trend_revenue_growth = robust_trend_next_value(driver["revenue_growth"], history_years, mix_revenue_growth)
robust_trend_ebit_margin = robust_trend_next_value(driver["ebit_margin"], history_years, mix_ebit_margin)

if estimation_method == "Trend (lineare Regression)":
    revenue_growth_base = float(np.clip(trend_revenue_growth, -0.25, 0.40))
    ebit_margin_base = float(np.clip(trend_ebit_margin, 0.0, 0.60))
elif estimation_method == "Robuster Trend (Median-Slope)":
    revenue_growth_base = float(np.clip(robust_trend_revenue_growth, -0.25, 0.40))
    ebit_margin_base = float(np.clip(robust_trend_ebit_margin, 0.0, 0.60))
else:
    revenue_growth_base = mix_revenue_growth
    ebit_margin_base = mix_ebit_margin

rev_cagr_3y = hist_cagr(_tail_window(income_hist["revenue"], 4), np.nan)
rev_cagr_5y = hist_cagr(_tail_window(income_hist["revenue"], 6), np.nan)
ebit_avg_3y = hist_mean(_tail_window(driver["ebit_margin"], 3), np.nan, tail=3)
ebit_avg_5y = hist_mean(_tail_window(driver["ebit_margin"], 5), np.nan, tail=5)

hist_metrics = st.columns(4)
hist_metrics[0].metric("Revenue CAGR (3Y)", "n/a" if not np.isfinite(rev_cagr_3y) else f"{rev_cagr_3y*100:.2f}%")
hist_metrics[1].metric("Revenue CAGR (5Y)", "n/a" if not np.isfinite(rev_cagr_5y) else f"{rev_cagr_5y*100:.2f}%")
hist_metrics[2].metric("Avg EBIT Margin (3Y)", "n/a" if not np.isfinite(ebit_avg_3y) else f"{ebit_avg_3y*100:.2f}%")
hist_metrics[3].metric("Avg EBIT Margin (5Y)", "n/a" if not np.isfinite(ebit_avg_5y) else f"{ebit_avg_5y*100:.2f}%")

method_metrics = st.columns(3)
method_metrics[0].metric("Mix Rev Growth", f"{mix_revenue_growth*100:.2f}%")
method_metrics[1].metric("Trend Rev Growth", f"{trend_revenue_growth*100:.2f}%")
method_metrics[2].metric("Robust Trend Rev Growth", f"{robust_trend_revenue_growth*100:.2f}%")

method_metrics_2 = st.columns(3)
method_metrics_2[0].metric("Mix EBIT Margin", f"{mix_ebit_margin*100:.2f}%")
method_metrics_2[1].metric("Trend EBIT Margin", f"{trend_ebit_margin*100:.2f}%")
method_metrics_2[2].metric("Robust Trend EBIT Margin", f"{robust_trend_ebit_margin*100:.2f}%")

hist_driver_view = pd.DataFrame(index=driver.index)
hist_driver_view["Revenue (USD m)"] = income_hist["revenue"] / 1e6
hist_driver_view["Revenue Growth %"] = driver["revenue_growth"] * 100.0
hist_driver_view["EBIT Margin %"] = driver["ebit_margin"] * 100.0
st.dataframe(hist_driver_view.sort_index(ascending=False).round(2), use_container_width=True)

da_pct_rev_base = pct_or_default(hist_mean(driver["da_pct_rev"], 0.03, tail=max(history_years, 1)), 0.03)
capex_pct_rev_base = pct_or_default(hist_mean(driver["capex_pct_rev"], 0.04, tail=max(history_years, 1)), 0.04)
nwc_pct_rev_base = pct_or_default(hist_mean(driver["nwc_pct_rev"], 0.10, tail=max(history_years, 1)), 0.10)

tax_rate_auto = estimate_tax_rate(latest.get("tax_expense"), latest.get("pretax_income"))

beta_mode = st.selectbox("Beta Estimation Mode", ["5Y Weekly", "2Y Daily"], index=0)
beta_auto = estimate_beta(
    ctx["ticker"],
    beta_mode,
    alpha_vantage_api_key=ctx.get("alpha_vantage_api_key"),
)
rf_auto = get_risk_free_rate()
erp_auto = get_implied_erp()

debt_now = latest.get("debt")
debt_prev = None
debt_series = base["debt_total"].dropna().sort_index()
if len(debt_series) > 1:
    debt_prev = float(debt_series.iloc[-2])

rd_auto = estimate_cost_of_debt(latest.get("interest_expense"), debt_now, debt_prev)

shares_auto = market.get("shares") if market.get("shares") else latest.get("shares")
equity_auto = market.get("market_cap") if market.get("market_cap") else 0.0
debt_auto = float(debt_now) if debt_now is not None and np.isfinite(debt_now) else 0.0
cash_auto = float(latest.get("cash", 0.0) or 0.0)

st.subheader("WACC (Auto CAPM + Overrides)")
w1, w2, w3, w4 = st.columns(4)

rf = w1.number_input("Risk-free rate Rf (%)", value=float(rf_auto * 100.0), step=0.1) / 100.0
beta = w2.number_input("Beta", value=float(beta_auto if beta_auto is not None else 1.0), step=0.05)
erp = w3.number_input("Equity Risk Premium ERP (%)", value=float(erp_auto * 100.0), step=0.1) / 100.0
rd = w4.number_input("Cost of Debt Rd (%)", value=float(rd_auto * 100.0), step=0.1) / 100.0

w5, w6, w7, w8 = st.columns(4)
tax_rate = w5.number_input("Tax Rate (%)", value=float(tax_rate_auto * 100.0), step=0.5) / 100.0
equity_value = w6.number_input("Equity Value E (USD m)", value=float((equity_auto or 0.0) / 1e6), step=100.0) * 1e6
debt_value = w7.number_input("Debt Value D (USD m)", value=float((debt_auto or 0.0) / 1e6), step=100.0) * 1e6
cash_value = w8.number_input("Cash (USD m)", value=float(cash_auto / 1e6), step=100.0) * 1e6

shares_used_default = (shares_auto if shares_auto is not None and np.isfinite(shares_auto) else 1_000_000.0)
shares_used = st.number_input("Shares Outstanding", value=float(shares_used_default), step=1_000_000.0)

re = cost_of_equity_capm(rf, beta, erp)
wacc = compute_wacc(re, rd, tax_rate, equity_value, debt_value)

wacc_cols = st.columns(3)
wacc_cols[0].metric("Cost of Equity (CAPM)", f"{re * 100:.2f}%")
wacc_cols[1].metric("Auto WACC", f"{wacc * 100:.2f}%")
wacc_cols[2].metric("Beta Mode", beta_mode)

st.subheader("Forecast Assumptions (5 Years)")
ass_col1, ass_col2, ass_col3 = st.columns(3)
base_growth = ass_col1.number_input("Base Revenue Growth (%)", value=float(revenue_growth_base * 100.0), step=0.5) / 100.0
base_margin = ass_col2.number_input("Base EBIT Margin (%)", value=float(ebit_margin_base * 100.0), step=0.5) / 100.0
terminal_growth = ass_col3.number_input("Terminal Growth g (%)", value=2.5, step=0.25) / 100.0

with st.expander("Year-by-Year Overrides", expanded=True):
    override_rows = []
    for i in range(1, 6):
        c1, c2, c3, c4, c5 = st.columns(5)
        rg = c1.number_input(f"Y{i} Rev Growth %", value=float(base_growth * 100.0), step=0.25, key=f"rg_{i}") / 100.0
        em = c2.number_input(f"Y{i} EBIT Margin %", value=float(base_margin * 100.0), step=0.25, key=f"em_{i}") / 100.0
        da = c3.number_input(f"Y{i} D&A % Rev", value=float(da_pct_rev_base * 100.0), step=0.1, key=f"da_{i}") / 100.0
        cx = c4.number_input(f"Y{i} Capex % Rev", value=float(capex_pct_rev_base * 100.0), step=0.1, key=f"cx_{i}") / 100.0
        nw = c5.number_input(f"Y{i} NWC % Rev", value=float(nwc_pct_rev_base * 100.0), step=0.1, key=f"nw_{i}") / 100.0
        override_rows.append((rg, em, da, cx, nw))

if not valid_terminal_setup(wacc, terminal_growth):
    st.error("Terminal setup ungueltig: Es muss gelten g < WACC.")
    st.stop()

if not np.isfinite(latest.get("revenue", np.nan)):
    st.error("Keine ausreichenden Revenue-Daten aus SEC gefunden.")
    st.stop()

rev_growth = [row[0] for row in override_rows]
ebit_margin = [row[1] for row in override_rows]
da_pct_rev = [row[2] for row in override_rows]
capex_pct_rev = [row[3] for row in override_rows]
nwc_pct_rev = [row[4] for row in override_rows]

start_year = int(latest["year"])

result = run_dcf(
    revenue0=float(latest["revenue"]),
    nwc0=float(latest.get("nwc", 0.0) or 0.0),
    shares=float(shares_used),
    cash=float(cash_value),
    debt=float(debt_value),
    wacc=float(wacc),
    terminal_growth=float(terminal_growth),
    revenue_growth=rev_growth,
    ebit_margin=ebit_margin,
    da_pct_rev=da_pct_rev,
    capex_pct_rev=capex_pct_rev,
    nwc_pct_rev=nwc_pct_rev,
    tax_rate=float(tax_rate),
    start_year=start_year,
)

st.subheader("DCF Ergebnis")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Enterprise Value", f"{result.enterprise_value / 1e9:,.2f} Bn")
r2.metric("Equity Value", f"{result.equity_value / 1e9:,.2f} Bn")
r3.metric("Fair Value / Share", f"{result.fair_value_per_share:,.2f}")
r4.metric("PV FCFF", f"{result.pv_fcff / 1e9:,.2f} Bn")
r5.metric("PV Terminal", f"{result.pv_terminal / 1e9:,.2f} Bn")

forecast_view = result.forecast_df.copy()
forecast_view = forecast_view.set_index("year")
st.dataframe((forecast_view / 1e6).round(2), use_container_width=True)

chart_df = result.forecast_df
fig = go.Figure()
fig.add_trace(go.Bar(x=chart_df["year"], y=chart_df["fcff"] / 1e6, name="FCFF (USD m)"))
fig.update_layout(title="Projected FCFF (5Y)", xaxis_title="Year", yaxis_title="USD m", height=350)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Sensitivity: Fair Value / Share (WACC x g)")
wacc_grid = [wacc - 0.01, wacc - 0.005, wacc, wacc + 0.005, wacc + 0.01]
g_grid = [terminal_growth - 0.005, terminal_growth, terminal_growth + 0.005]
wacc_grid = [max(x, 0.01) for x in wacc_grid]
g_grid = [max(x, 0.0) for x in g_grid]

sense = build_sensitivity_table(
    base_revenue0=float(latest["revenue"]),
    base_nwc0=float(latest.get("nwc", 0.0) or 0.0),
    shares=float(shares_used),
    cash=float(cash_value),
    debt=float(debt_value),
    wacc_values=wacc_grid,
    growth_values=g_grid,
    revenue_growth=rev_growth,
    ebit_margin=ebit_margin,
    da_pct_rev=da_pct_rev,
    capex_pct_rev=capex_pct_rev,
    nwc_pct_rev=nwc_pct_rev,
    tax_rate=float(tax_rate),
    start_year=start_year,
)
st.dataframe(sense.round(2), use_container_width=True)
