from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import pandas as pd
import streamlit as st


TICKER_0050 = "0050.TW"
BENCHMARK_TICKER = "^TWII"
VIX_TICKER = "^VIX"
PEER_TICKERS = {
    "0050 元大台灣50": "0050.TW",
    "006208 富邦台50": "006208.TW",
    "加權股價指數": "^TWII",
}


@dataclass(frozen=True)
class Snapshot:
    last_price: float
    daily_change_pct: float
    ytd_return_pct: float
    one_year_return_pct: float
    annual_volatility_pct: float
    max_drawdown_pct: float


@dataclass(frozen=True)
class DecisionEngineResult:
    market_score: int
    recommendation: str
    stock_ratio: int
    cash_ratio: int
    risk_level: str
    market_temperature: str
    status_label: str
    status_color: str
    vix_value: float | None
    vix_analysis: str
    rsi_value: float | None
    trend_score: float
    vix_score: float
    rsi_score: float
    drawdown_score: float
    momentum_score: float
    overheating_penalty: float
    factor_reasons: tuple[str, ...]
    drawdown_pct: float
    drawdown_analysis: str
    overheating_warnings: tuple[str, ...]
    strategy_summary: str
    reasoning_summary: str


@dataclass(frozen=True)
class IntradayReference:
    manual_price: float
    latest_close: float
    recent_high: float
    drawdown_from_high_pct: float
    difference_from_close_pct: float
    add_on_zones: dict[str, bool]
    action_suggestion: str


def require_optional_packages() -> tuple[object | None, object | None]:
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        yf = None

    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError:
        go = None

    return yf, go


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_prices(tickers: Iterable[str], start: date, end: date) -> pd.DataFrame:
    yf, _ = require_optional_packages()
    if yf is None:
        return pd.DataFrame()

    raw = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame()