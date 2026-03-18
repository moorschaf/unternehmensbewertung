from __future__ import annotations

import math


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def estimate_tax_rate(tax_expense: float | None, pretax_income: float | None) -> float:
    if tax_expense is None or pretax_income is None or pretax_income <= 0:
        return 0.21
    rate = tax_expense / pretax_income
    return clamp(rate, 0.0, 0.4)


def estimate_cost_of_debt(interest_expense: float | None, debt_now: float | None, debt_prev: float | None = None) -> float:
    if debt_now is None or debt_now <= 0 or interest_expense is None:
        return 0.045
    avg_debt = debt_now
    if debt_prev is not None and debt_prev > 0:
        avg_debt = 0.5 * (debt_now + debt_prev)
    rd = abs(interest_expense) / max(avg_debt, 1.0)
    return clamp(rd, 0.01, 0.15)


def cost_of_equity_capm(rf: float, beta: float, erp: float) -> float:
    return rf + beta * erp


def compute_wacc(
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float,
    equity_value: float,
    debt_value: float,
) -> float:
    d_plus_e = max(equity_value + debt_value, 0.0)
    if d_plus_e <= 0:
        return max(cost_of_equity, 0.01)

    w_e = equity_value / d_plus_e
    w_d = debt_value / d_plus_e
    wacc = w_e * cost_of_equity + w_d * cost_of_debt * (1.0 - tax_rate)
    return clamp(wacc, 0.01, 0.30)


def valid_terminal_setup(wacc: float, terminal_growth: float) -> bool:
    return math.isfinite(wacc) and math.isfinite(terminal_growth) and wacc > terminal_growth
