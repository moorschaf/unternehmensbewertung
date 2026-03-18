from __future__ import annotations

from typing import Any

import requests
import streamlit as st


SEC_BASE = "https://data.sec.gov/api"
SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"


def _headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def get_company_ticker_map(user_agent: str) -> dict[str, dict[str, Any]]:
    response = requests.get(SEC_TICKER_URL, headers=_headers(user_agent), timeout=30)
    response.raise_for_status()
    payload = response.json()

    out: dict[str, dict[str, Any]] = {}
    for row in payload.values():
        ticker = str(row.get("ticker", "")).upper().strip()
        cik = int(row.get("cik_str", 0))
        title = str(row.get("title", "")).strip()
        if ticker and cik > 0:
            out[ticker] = {"cik": cik, "title": title}
    return out


def cik_to_padded(cik: int) -> str:
    return f"{cik:010d}"


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def get_company_facts(cik: int, user_agent: str) -> dict[str, Any]:
    url = f"{SEC_BASE}/xbrl/companyfacts/CIK{cik_to_padded(cik)}.json"
    response = requests.get(url, headers=_headers(user_agent), timeout=40)
    response.raise_for_status()
    return response.json()
