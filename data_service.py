from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd
import streamlit as st


STANDARD_COLUMNS = ["Close", "Volume"]


def empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def standardize_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return empty_price_frame()

    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)

    if "Close" not in data:
        return empty_price_frame()

    output = pd.DataFrame(index=pd.to_datetime(data.index))
    output["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    if "Volume" in data:
        output["Volume"] = pd.to_numeric(data["Volume"], errors="coerce")
    else:
        output["Volume"] = pd.NA

    output = output.sort_index()
    return output.dropna(subset=["Close"])


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_price_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    try:
        import yfinance as yf

        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return standardize_price_frame(raw)
    except Exception:
        return empty_price_frame()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_many_price_data(tickers: tuple[str, ...], start: date, end: date) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        result[ticker] = fetch_price_data(ticker, start, end)
    return result


def fetch_taiwan_stock(ticker: str, start: date, end: date) -> tuple[str, pd.DataFrame]:
    normalized = ticker.upper().replace(".TW", "")
    tw_ticker = f"{normalized}.TW" if normalized.isdigit() and len(normalized) == 4 else ticker
    return tw_ticker, fetch_price_data(tw_ticker, start, end)
