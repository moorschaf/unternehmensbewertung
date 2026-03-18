from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


ANNUAL_FORMS = {"10-K", "20-F", "40-F", "10-K/A"}


TAG_CANDIDATES = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "ebit": ["OperatingIncomeLoss"],
    "pretax_income": ["IncomeBeforeTax", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "interest_expense": ["InterestExpense", "InterestAndDebtExpense"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet", "Depreciation"],
    "capex": ["CapitalExpenditureIncurredButNotYetPaid", "PaymentsToAcquirePropertyPlantAndEquipment", "CapitalExpenditures"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "short_debt": ["DebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"],
    "long_debt": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"],
    "shares": ["WeightedAverageNumberOfDilutedSharesOutstanding", "CommonStockSharesOutstanding"],
}


def _latest_per_year(entries: list[dict[str, Any]]) -> dict[int, float]:
    selected: dict[int, dict[str, Any]] = {}
    for row in entries:
        form = str(row.get("form", ""))
        fp = str(row.get("fp", "")).upper()
        fy = row.get("fy")
        if form not in ANNUAL_FORMS or fp != "FY" or not isinstance(fy, int):
            continue

        prev = selected.get(fy)
        if prev is None:
            selected[fy] = row
            continue

        prev_filed = str(prev.get("filed", ""))
        curr_filed = str(row.get("filed", ""))
        if curr_filed >= prev_filed:
            selected[fy] = row

    return {year: float(item["val"]) for year, item in selected.items() if "val" in item}


def _extract_metric(facts: dict[str, Any], tags: list[str], units: tuple[str, ...]) -> dict[int, float]:
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        node = us_gaap.get(tag)
        if not node:
            continue
        unit_map = node.get("units", {})
        for unit in units:
            rows = unit_map.get(unit)
            if rows:
                series = _latest_per_year(rows)
                if series:
                    return series
    return {}


@dataclass
class FinancialBundle:
    income_df: pd.DataFrame
    balance_df: pd.DataFrame
    cashflow_df: pd.DataFrame
    driver_df: pd.DataFrame
    latest: dict[str, float]


def map_company_facts_to_financials(facts: dict[str, Any]) -> FinancialBundle:
    data = {
        key: _extract_metric(facts, tags, ("USD",))
        for key, tags in TAG_CANDIDATES.items()
        if key != "shares"
    }
    data["shares"] = _extract_metric(facts, TAG_CANDIDATES["shares"], ("shares",))

    years = sorted({year for metric in data.values() for year in metric.keys()})
    if not years:
        empty = pd.DataFrame()
        return FinancialBundle(empty, empty, empty, empty, {})

    base = pd.DataFrame(index=years)
    for key, series in data.items():
        base[key] = pd.Series(series)

    base["capex_abs"] = base["capex"].abs()
    base["debt_total"] = base[["short_debt", "long_debt"]].sum(axis=1, min_count=1)
    base["nwc"] = (
        base["current_assets"]
        - base["cash"]
        - (base["current_liabilities"] - base["short_debt"])
    )

    income_df = base[["revenue", "ebit", "pretax_income", "tax_expense", "interest_expense", "shares"]].copy()
    balance_df = base[["cash", "current_assets", "current_liabilities", "short_debt", "long_debt", "debt_total", "nwc"]].copy()
    cashflow_df = base[["da", "capex", "capex_abs"]].copy()

    driver_df = pd.DataFrame(index=base.index)
    driver_df["revenue_growth"] = base["revenue"].pct_change()
    driver_df["ebit_margin"] = base["ebit"] / base["revenue"]
    driver_df["da_pct_rev"] = base["da"] / base["revenue"]
    driver_df["capex_pct_rev"] = base["capex_abs"] / base["revenue"]
    driver_df["nwc_pct_rev"] = base["nwc"] / base["revenue"]
    driver_df["tax_rate"] = np.where(base["pretax_income"] > 0, base["tax_expense"] / base["pretax_income"], np.nan)

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

    return FinancialBundle(
        income_df=income_df.sort_index(ascending=False),
        balance_df=balance_df.sort_index(ascending=False),
        cashflow_df=cashflow_df.sort_index(ascending=False),
        driver_df=driver_df.sort_index(ascending=False),
        latest=latest,
    )
