from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd
import streamlit as st


STANDARD_COLUMNS = ["Close", "Volume"]


def empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def normalize_yfinance_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    print(f"{ticker} 原始 columns:", None if df is None else df.columns)
    if df is None or df.empty:
        print(f"{ticker} 標準化後 columns:", STANDARD_COLUMNS)
        print(f"{ticker} Close 有效筆數:", 0)
        return empty_price_frame()

    data = df.copy()
    if "Close" not in data.columns:
        print(f"{ticker} 標準化後 columns:", STANDARD_COLUMNS)
        print(f"{ticker} Close 有效筆數:", 0)
        return empty_price_frame()

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce")

    if "Volume" in data.columns:
        volume = data["Volume"]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
        volume = pd.to_numeric(volume, errors="coerce").fillna(0)
    else:
        volume = pd.Series(0, index=data.index)

    output = pd.DataFrame(
        {
            "Close": close,
            "Volume": volume,
        },
        index=pd.to_datetime(data.index),
    ).sort_index()
    output = output.dropna(subset=["Close"])
    print(f"{ticker} 標準化後 columns:", output.columns)
    print(f"{ticker} Close 有效筆數:", output["Close"].dropna().shape[0])
    return output


def standardize_price_frame(raw: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    return normalize_yfinance_df(raw, ticker)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_price_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    periods = ("2y", "1y", "6mo", "3mo")
    try:
        import yfinance as yf
    except Exception:
        print(f"{ticker} 資料長度:", 0)
        return empty_price_frame()

    for period in periods:
        try:
            raw = yf.download(
                ticker,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            df = normalize_yfinance_df(raw, ticker)
            print(f"{ticker} 資料長度:", len(df))
            if not df.empty:
                return df
        except Exception:
            continue

    print(f"{ticker} 資料長度:", 0)
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
