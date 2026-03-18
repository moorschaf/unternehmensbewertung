from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DCFResult:
    forecast_df: pd.DataFrame
    enterprise_value: float
    equity_value: float
    fair_value_per_share: float
    pv_fcff: float
    pv_terminal: float


def run_dcf(
    revenue0: float,
    nwc0: float,
    shares: float,
    cash: float,
    debt: float,
    wacc: float,
    terminal_growth: float,
    revenue_growth: list[float],
    ebit_margin: list[float],
    da_pct_rev: list[float],
    capex_pct_rev: list[float],
    nwc_pct_rev: list[float],
    tax_rate: float,
    start_year: int,
) -> DCFResult:
    years = [start_year + i for i in range(1, 6)]
    rows: list[dict[str, float]] = []

    revenue_prev = revenue0
    nwc_prev = nwc0

    for i, year in enumerate(years):
        rev = revenue_prev * (1.0 + revenue_growth[i])
        ebit = rev * ebit_margin[i]
        nopat = ebit * (1.0 - tax_rate)
        da = rev * da_pct_rev[i]
        capex = rev * capex_pct_rev[i]
        nwc = rev * nwc_pct_rev[i]
        delta_nwc = nwc - nwc_prev
        fcff = nopat + da - capex - delta_nwc
        discount_factor = (1.0 + wacc) ** (i + 1)
        pv_fcff = fcff / discount_factor

        rows.append(
            {
                "year": year,
                "revenue": rev,
                "ebit": ebit,
                "nopat": nopat,
                "da": da,
                "capex": capex,
                "nwc": nwc,
                "delta_nwc": delta_nwc,
                "fcff": fcff,
                "pv_fcff": pv_fcff,
            }
        )

        revenue_prev = rev
        nwc_prev = nwc

    forecast_df = pd.DataFrame(rows)

    fcff_5 = float(forecast_df.iloc[-1]["fcff"])
    fcff_6 = fcff_5 * (1.0 + terminal_growth)
    terminal_value = fcff_6 / max(wacc - terminal_growth, 1e-6)
    pv_terminal = terminal_value / ((1.0 + wacc) ** 5)

    pv_fcff = float(forecast_df["pv_fcff"].sum())
    enterprise_value = pv_fcff + pv_terminal
    equity_value = enterprise_value - debt + cash
    fair_value_per_share = equity_value / max(shares, 1e-6)

    return DCFResult(
        forecast_df=forecast_df,
        enterprise_value=float(enterprise_value),
        equity_value=float(equity_value),
        fair_value_per_share=float(fair_value_per_share),
        pv_fcff=float(pv_fcff),
        pv_terminal=float(pv_terminal),
    )


def build_sensitivity_table(
    base_revenue0: float,
    base_nwc0: float,
    shares: float,
    cash: float,
    debt: float,
    wacc_values: list[float],
    growth_values: list[float],
    revenue_growth: list[float],
    ebit_margin: list[float],
    da_pct_rev: list[float],
    capex_pct_rev: list[float],
    nwc_pct_rev: list[float],
    tax_rate: float,
    start_year: int,
) -> pd.DataFrame:
    table = pd.DataFrame(index=[f"{w*100:.2f}%" for w in wacc_values], columns=[f"{g*100:.2f}%" for g in growth_values])

    for w in wacc_values:
        for g in growth_values:
            if w <= g:
                table.loc[f"{w*100:.2f}%", f"{g*100:.2f}%"] = np.nan
                continue
            result = run_dcf(
                revenue0=base_revenue0,
                nwc0=base_nwc0,
                shares=shares,
                cash=cash,
                debt=debt,
                wacc=w,
                terminal_growth=g,
                revenue_growth=revenue_growth,
                ebit_margin=ebit_margin,
                da_pct_rev=da_pct_rev,
                capex_pct_rev=capex_pct_rev,
                nwc_pct_rev=nwc_pct_rev,
                tax_rate=tax_rate,
                start_year=start_year,
            )
            table.loc[f"{w*100:.2f}%", f"{g*100:.2f}%"] = result.fair_value_per_share

    return table
