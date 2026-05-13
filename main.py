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
TAIWAN_STOCK_ALIASES = {
    "台積電": "2330.TW",
    "光寶科": "2301.TW",
    "欣興": "3037.TW",
    "鴻海": "2317.TW",
    "聯發科": "2454.TW",
    "廣達": "2382.TW",
    "緯創": "3231.TW",
    "元大台灣50": "0050.TW",
    "0050": "0050.TW",
}
BACKGROUND_TICKERS = {
    "QQQ": "QQQ",
    "SOXX": "SOXX",
    "SPY": "SPY",
    "VTI": "VTI",
    "006208.TW": "006208.TW",
    "0050.TW": "0050.TW",
    "^TWII": "^TWII",
    "^VIX": "^VIX",
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


@dataclass(frozen=True)
class TaiwanStockAnalysis:
    label: str
    ticker: str
    latest_price: float
    ytd_return_pct: float
    one_year_return_pct: float
    annual_volatility_pct: float
    max_drawdown_pct: float
    rsi_value: float | None
    ma20: float | None
    ma60: float | None
    ma120: float | None
    drawdown_from_high_pct: float
    distance_ma20_pct: float | None
    distance_ma60_pct: float | None
    distance_ma120_pct: float | None
    ai_score: int
    recommendation: str
    risk_level: str
    market_temperature: str
    stock_ratio: int
    cash_ratio: int
    reason_summary: str
    background_summary: dict[str, str]
    missing_background: bool


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

    closes: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                closes[ticker] = raw[ticker]["Close"]
            else:
                closes[ticker] = raw["Close"]
        except KeyError:
            continue

    prices = pd.DataFrame(closes).dropna(how="all")
    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def pct_change(first: float, last: float) -> float:
    if pd.isna(first) or pd.isna(last) or first == 0:
        return 0.0
    return (last / first - 1.0) * 100.0


def max_drawdown(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return 0.0
    running_high = clean.cummax()
    drawdown = clean / running_high - 1.0
    return float(drawdown.min() * 100.0)


def annualized_volatility(series: pd.Series) -> float:
    returns = series.pct_change().dropna()
    if returns.empty:
        return 0.0
    return float(returns.std() * (252**0.5) * 100.0)


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    changes = prices.diff()
    gains = changes.clip(lower=0)
    losses = -changes.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.mask(avg_loss == 0)
    return 100 - (100 / (1 + relative_strength))


def build_snapshot(prices: pd.Series) -> Snapshot:
    clean = prices.dropna()
    if clean.empty:
        return Snapshot(0, 0, 0, 0, 0, 0)

    last_price = float(clean.iloc[-1])
    previous_price = float(clean.iloc[-2]) if len(clean) > 1 else last_price
    current_year = clean[clean.index.year == clean.index[-1].year]
    one_year = clean[clean.index >= clean.index[-1] - pd.DateOffset(years=1)]

    return Snapshot(
        last_price=last_price,
        daily_change_pct=pct_change(previous_price, last_price),
        ytd_return_pct=pct_change(float(current_year.iloc[0]), last_price)
        if not current_year.empty
        else 0.0,
        one_year_return_pct=pct_change(float(one_year.iloc[0]), last_price)
        if not one_year.empty
        else 0.0,
        annual_volatility_pct=annualized_volatility(clean.tail(252)),
        max_drawdown_pct=max_drawdown(clean),
    )


def moving_average(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window, min_periods=max(2, window // 3)).mean()


def normalize_to_100(prices: pd.DataFrame) -> pd.DataFrame:
    clean = prices.dropna(how="all").copy()

    for column in clean.columns:
        series = clean[column].dropna()

        if series.empty:
            continue  # 防止空資料 crash

        first = series.iloc[0]

        if first == 0:
            continue  # 防止除以0

        clean[column] = clean[column] / first * 100.0

    return clean


def format_pct(value: float) -> str:
    return f"{value:+.2f}%"


def format_price(value: float) -> str:
    return f"新台幣{value:,.2f}"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def render_metric(label: str, value: str, delta: str | None = None) -> None:
    st.metric(label=label, value=value, delta=delta)


def render_package_help() -> None:
    st.error("缺少即時行情與圖表所需的套件。")
    st.code("pip install streamlit yfinance plotly pandas", language="bash")
    st.info("安裝後執行：streamlit run main.py")


def render_price_chart(go: object, prices: pd.Series, show_ma20: bool, show_ma60: bool) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices,
            mode="lines",
            name="0050 收盤價",
            line={"width": 2.5, "color": "#2563eb"},
        )
    )

    if show_ma20:
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=moving_average(prices, 20),
                mode="lines",
                name="20日 MA",
                line={"width": 1.5, "color": "#f59e0b"},
            )
        )

    if show_ma60:
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=moving_average(prices, 60),
                mode="lines",
                name="60日 MA",
                line={"width": 1.5, "color": "#10b981"},
            )
        )

    fig.update_layout(
        height=430,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        yaxis_title="還原權值收盤價",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_drawdown_chart(go: object, prices: pd.Series) -> None:
    drawdown = prices / prices.cummax() - 1.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown * 100.0,
            mode="lines",
            name="回撤",
            fill="tozeroy",
            line={"width": 1.5, "color": "#dc2626"},
        )
    )
    fig.update_layout(
        height=280,
        margin={"l": 20, "r": 20, "t": 10, "b": 20},
        yaxis_title="回撤 %",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def current_drawdown(prices: pd.Series) -> float:
    clean = prices.dropna()
    if clean.empty:
        return 0.0
    return float((clean.iloc[-1] / clean.cummax().iloc[-1] - 1.0) * 100.0)


def latest_value(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


def build_intraday_reference(price_0050: pd.Series, manual_price: float) -> IntradayReference:
    clean = price_0050.dropna()
    latest_close = float(clean.iloc[-1])
    recent_high = float(clean.tail(60).max())
    drawdown_from_high_pct = pct_change(recent_high, manual_price)
    difference_from_close_pct = pct_change(latest_close, manual_price)
    zones = {
        "-5%": drawdown_from_high_pct <= -5,
        "-10%": drawdown_from_high_pct <= -10,
        "-15%": drawdown_from_high_pct <= -15,
        "-20%": drawdown_from_high_pct <= -20,
        "-30%": drawdown_from_high_pct <= -30,
    }

    if drawdown_from_high_pct <= -20:
        action_suggestion = "大幅加碼"
    elif drawdown_from_high_pct <= -10:
        action_suggestion = "中度加碼"
    elif drawdown_from_high_pct <= -5:
        action_suggestion = "小量加碼"
    else:
        action_suggestion = "不買"

    return IntradayReference(
        manual_price=manual_price,
        latest_close=latest_close,
        recent_high=recent_high,
        drawdown_from_high_pct=drawdown_from_high_pct,
        difference_from_close_pct=difference_from_close_pct,
        add_on_zones=zones,
        action_suggestion=action_suggestion,
    )


def resolve_taiwan_stock_query(query: str) -> tuple[str, str | None]:
    text = query.strip()
    if not text:
        return "", None

    if text in TAIWAN_STOCK_ALIASES:
        return text, TAIWAN_STOCK_ALIASES[text]

    normalized = text.upper().replace(".TW", "").replace(".TWO", "")
    if normalized.isdigit() and len(normalized) == 4:
        return normalized, f"{normalized}.TW"

    return text, None


def ticker_display_name(ticker: str, fallback: str) -> str:
    for name, mapped_ticker in TAIWAN_STOCK_ALIASES.items():
        if mapped_ticker == ticker and not name.isdigit():
            return f"{name}（{ticker}）"
    return f"{fallback}（{ticker}）"


def load_taiwan_stock_series(query: str, start: date, end: date) -> tuple[str, str, pd.Series]:
    label, ticker = resolve_taiwan_stock_query(query)
    if ticker is None:
        return label, "", pd.Series(dtype="float64")

    prices = load_prices([ticker], start, end)
    if ticker in prices and not prices[ticker].dropna().empty:
        return ticker_display_name(ticker, label), ticker, prices[ticker].dropna()

    if ticker.endswith(".TW"):
        fallback_ticker = ticker.replace(".TW", ".TWO")
        fallback_prices = load_prices([fallback_ticker], start, end)
        if fallback_ticker in fallback_prices and not fallback_prices[fallback_ticker].dropna().empty:
            return ticker_display_name(fallback_ticker, label), fallback_ticker, fallback_prices[fallback_ticker].dropna()

    return ticker_display_name(ticker, label), ticker, pd.Series(dtype="float64")


def safe_latest(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


def distance_pct(reference: float | None, value: float) -> float | None:
    if reference is None or pd.isna(reference) or reference == 0:
        return None
    return pct_change(float(reference), value)


def classify_trend(series: pd.Series) -> str | None:
    clean = series.dropna()
    if clean.empty:
        return None

    last = float(clean.iloc[-1])
    ma20 = safe_latest(moving_average(clean, 20))
    ma60 = safe_latest(moving_average(clean, 60))
    latest_rsi = safe_latest(rsi(clean))
    if latest_rsi is not None and latest_rsi > 72 and ma20 is not None and last > ma20:
        return "過熱"
    if ma20 is not None and ma60 is not None and last > ma20 and last > ma60:
        return "偏多"
    if ma60 is not None and last < ma60:
        return "偏弱"
    return "中性"


def build_background_market_summary(start: date, end: date) -> tuple[dict[str, str], int, int, bool]:
    prices = load_prices(list(BACKGROUND_TICKERS.values()), start, end)
    missing = prices.empty

    def get_series(ticker: str) -> pd.Series:
        if ticker not in prices:
            return pd.Series(dtype="float64")
        return prices[ticker].dropna()

    qqq_status = classify_trend(get_series("QQQ"))
    soxx_status = classify_trend(get_series("SOXX"))
    spy_status = classify_trend(get_series("SPY"))
    vti_status = classify_trend(get_series("VTI"))
    tw50_status = classify_trend(get_series("0050.TW"))
    peer_status = classify_trend(get_series("006208.TW"))
    vix_value = safe_latest(get_series("^VIX"))

    if qqq_status is None:
        missing = True
        qqq_status = "中性"
    if soxx_status is None:
        missing = True
        soxx_status = "中性"
    if spy_status is None or vti_status is None:
        missing = True
        risk_status = "中性"
    elif "過熱" in {spy_status, vti_status}:
        risk_status = "偏高"
    elif "偏弱" in {spy_status, vti_status}:
        risk_status = "偏低"
    else:
        risk_status = "中性"

    if tw50_status is None and peer_status is None:
        missing = True
        large_cap_status = "正常"
    elif "過熱" in {tw50_status, peer_status}:
        large_cap_status = "偏熱"
    elif "偏弱" in {tw50_status, peer_status}:
        large_cap_status = "偏弱"
    else:
        large_cap_status = "正常"

    if vix_value is None:
        missing = True
        vix_status = "正常"
    elif vix_value < 12:
        vix_status = "過度樂觀"
    elif vix_value >= 25:
        vix_status = "恐慌升溫"
    else:
        vix_status = "正常"

    penalty = 0
    bonus = 0
    if qqq_status == "過熱" or soxx_status == "過熱" or large_cap_status == "偏熱" or vix_status == "過度樂觀":
        penalty += 8
    if vix_status == "恐慌升溫" or risk_status == "偏低":
        bonus += 5

    return (
        {
            "美股科技趨勢": qqq_status,
            "半導體風向": soxx_status,
            "大盤風險偏好": risk_status,
            "台股大型股比較": large_cap_status,
            "VIX風險訊號": vix_status,
        },
        penalty,
        bonus,
        missing,
    )


def analyze_taiwan_stock(
    label: str,
    ticker: str,
    prices: pd.Series,
    background_summary: dict[str, str],
    background_penalty: int,
    background_bonus: int,
    missing_background: bool,
) -> TaiwanStockAnalysis | None:
    clean = prices.dropna()
    if clean.empty:
        return None

    latest_price = float(clean.iloc[-1])
    current_year = clean[clean.index.year == clean.index[-1].year]
    one_year = clean[clean.index >= clean.index[-1] - pd.DateOffset(years=1)]
    ma20 = safe_latest(moving_average(clean, 20))
    ma60 = safe_latest(moving_average(clean, 60))
    ma120 = safe_latest(moving_average(clean, 120))
    latest_rsi = safe_latest(rsi(clean))
    recent_high = float(clean.tail(120).max())
    drawdown_from_high_pct = pct_change(recent_high, latest_price)
    max_drawdown_pct = max_drawdown(clean)
    annual_volatility_pct = annualized_volatility(clean.tail(252))
    ytd_return_pct = pct_change(float(current_year.iloc[0]), latest_price) if not current_year.empty else 0.0
    one_year_return_pct = pct_change(float(one_year.iloc[0]), latest_price) if not one_year.empty else 0.0

    score = 50
    reasons = []
    if ma20 is not None and latest_price > ma20:
        score += 5
        reasons.append("價格站上 MA20，短線趨勢偏多。")
    if ma60 is not None and latest_price > ma60:
        score += 7
        reasons.append("價格站上 MA60，中期趨勢獲得支撐。")
    if ma120 is not None and latest_price > ma120:
        score += 8
        reasons.append("價格位於 MA120 之上，長線結構仍偏正向。")
    if ma20 is not None and ma60 is not None and ma20 > ma60:
        score += 5
        reasons.append("MA20 高於 MA60，均線排列有利。")

    if drawdown_from_high_pct <= -20:
        score += 16
        reasons.append("近期高點回撤超過 20%，評價修正幅度較深。")
    elif drawdown_from_high_pct <= -10:
        score += 10
        reasons.append("近期高點回撤超過 10%，出現分批觀察機會。")
    if latest_rsi is not None and latest_rsi < 40:
        score += 8
        reasons.append("RSI 低於 40，短線追殺壓力可能降溫。")
    if ma120 is not None and abs(distance_pct(ma120, latest_price) or 999) <= 3:
        score += 6
        reasons.append("股價接近 MA120 支撐區。")
    if background_bonus:
        score += background_bonus
        reasons.append("背景市場恐慌升溫，逆勢布局分數小幅加分。")

    overheating_penalty = 0
    if latest_rsi is not None and latest_rsi > 70:
        overheating_penalty += 10
        reasons.append("RSI 高於 70，過熱風險扣分。")
    if drawdown_from_high_pct > -3:
        overheating_penalty += 10
        reasons.append("股價接近近期高點，避免在高檔過度追價。")
    if drawdown_from_high_pct > -3 and ma20 is not None and ma60 is not None and latest_price > ma20 and latest_price > ma60:
        overheating_penalty += 6
        reasons.append("趨勢強但幾乎沒有回撤，降低積極加碼訊號。")
    if background_penalty:
        overheating_penalty += background_penalty
        reasons.append("背景市場偏熱或過度樂觀，模型提高保守係數。")

    score = int(clamp(score - overheating_penalty, 0, 100))
    if score >= 75:
        recommendation = "建議加碼"
        risk_level = "低風險"
        stock_ratio = 75
    elif score >= 55:
        recommendation = "持有觀察"
        risk_level = "中等風險"
        stock_ratio = 55
    else:
        recommendation = "暫緩進場"
        risk_level = "高風險"
        stock_ratio = 35

    if latest_rsi is not None and latest_rsi > 75:
        stock_ratio = min(stock_ratio, 45)
    if background_penalty >= 8:
        stock_ratio = min(stock_ratio, 50)

    if score >= 75 and drawdown_from_high_pct <= -10:
        market_temperature = "冷卻"
    elif latest_rsi is not None and latest_rsi > 75 and drawdown_from_high_pct > -5:
        market_temperature = "狂熱"
    elif latest_rsi is not None and latest_rsi > 68:
        market_temperature = "偏熱"
    else:
        market_temperature = "中性"

    cash_ratio = 100 - stock_ratio
    stock_rsi_text = "無資料" if latest_rsi is None else f"{latest_rsi:.1f}"
    reason_summary = (
        f"{label} AI評分為 {score}/100，建議為「{recommendation}」。"
        f"目前 RSI 為 {stock_rsi_text}，"
        f"距近期高點回撤 {drawdown_from_high_pct:.2f}%。"
        f"{''.join(reasons[:4])}"
    )

    return TaiwanStockAnalysis(
        label=label,
        ticker=ticker,
        latest_price=latest_price,
        ytd_return_pct=ytd_return_pct,
        one_year_return_pct=one_year_return_pct,
        annual_volatility_pct=annual_volatility_pct,
        max_drawdown_pct=max_drawdown_pct,
        rsi_value=latest_rsi,
        ma20=ma20,
        ma60=ma60,
        ma120=ma120,
        drawdown_from_high_pct=drawdown_from_high_pct,
        distance_ma20_pct=distance_pct(ma20, latest_price),
        distance_ma60_pct=distance_pct(ma60, latest_price),
        distance_ma120_pct=distance_pct(ma120, latest_price),
        ai_score=score,
        recommendation=recommendation,
        risk_level=risk_level,
        market_temperature=market_temperature,
        stock_ratio=stock_ratio,
        cash_ratio=cash_ratio,
        reason_summary=reason_summary,
        background_summary=background_summary,
        missing_background=missing_background,
    )


def score_vix(vix_value: float | None) -> tuple[float, str]:
    if vix_value is None:
        return 55.0, "VIX 資料無法取得，模型採用中性風險假設。"
    if vix_value < 12:
        return 58.0, "VIX 過低，雖然盤面平靜，但市場自滿風險升高。"
    if vix_value < 15:
        return 70.0, "VIX 偏低，低波動環境可能代表資金交易過度擁擠。"
    if vix_value < 20:
        return 78.0, "VIX 位於正常區間，整體風險條件相對均衡。"
    if vix_value < 25:
        return 55.0, "VIX 偏高，部位配置應維持選擇性。"
    if vix_value < 35:
        return 32.0, "VIX 高檔，市場壓力正壓抑風險性資產。"
    return 15.0, "VIX 極端升高，資金保全應優先於進攻。"


def score_rsi(rsi_value: float | None) -> tuple[float, str]:
    if rsi_value is None:
        return 55.0, "RSI 無法取得，模型採用中性假設。"
    if rsi_value < 30:
        return 82.0, "RSI 進入超賣區，均值回歸機會提供支撐。"
    if rsi_value < 45:
        return 72.0, "RSI 偏冷，短線漲幅尚未過度延伸。"
    if rsi_value < 60:
        return 65.0, "RSI 中性，動能狀態相對均衡。"
    if rsi_value < 68:
        return 55.0, "RSI 偏強但尚未過熱。"
    if rsi_value < 75:
        return 38.0, "RSI 偏高，模型因過熱風險扣分。"
    return 22.0, "RSI 明顯過熱，模型大幅降低追價信心。"


def score_drawdown(drawdown_pct: float) -> tuple[float, str]:
    depth = abs(drawdown_pct)
    if depth < 3:
        return 92.0, "0050 接近近期高檔，回撤壓力有限。"
    if depth < 8:
        return 76.0, "0050 處於正常拉回區間。"
    if depth < 15:
        return 56.0, "0050 已進入明顯修正，較適合分批布局。"
    if depth < 25:
        return 36.0, "0050 處於深度回撤，應保留現金並避免追價。"
    return 20.0, "0050 進入嚴重回撤，風險控管比進場速度更重要。"


def build_decision_engine(prices: pd.DataFrame, price_0050: pd.Series) -> DecisionEngineResult:
    clean = price_0050.dropna()
    if clean.empty:
        return DecisionEngineResult(
            market_score=0,
            recommendation="暫緩進場",
            stock_ratio=0,
            cash_ratio=100,
            risk_level="無資料",
            market_temperature="中性",
            status_label="無資料",
            status_color="#64748b",
            vix_value=None,
            vix_analysis="目前沒有可用價格資料。",
            rsi_value=None,
            trend_score=0.0,
            vix_score=0.0,
            rsi_score=0.0,
            drawdown_score=0.0,
            momentum_score=0.0,
            overheating_penalty=0.0,
            factor_reasons=("缺少價格資料，無法計算評分因子。",),
            drawdown_pct=0.0,
            drawdown_analysis="目前沒有可用回撤資料。",
            overheating_warnings=(),
            strategy_summary="取得 0050 價格資料後，才會產生策略摘要。",
            reasoning_summary="模型目前沒有可用輸入資料。",
        )

    ma20 = latest_value(moving_average(clean, 20))
    ma60 = latest_value(moving_average(clean, 60))
    ma120 = latest_value(moving_average(clean, 120))
    last_price = float(clean.iloc[-1])
    one_month = clean[clean.index >= clean.index[-1] - pd.DateOffset(days=30)]
    three_month = clean[clean.index >= clean.index[-1] - pd.DateOffset(days=90)]
    factor_reasons = []

    trend_score = 50.0
    if ma20 is not None and last_price > ma20:
        trend_score += 15
        factor_reasons.append("趨勢 +15：價格站上 20日 MA。")
    elif ma20 is not None:
        factor_reasons.append("趨勢 +0：價格低於 20日 MA。")

    if ma60 is not None and last_price > ma60:
        trend_score += 15
        factor_reasons.append("趨勢 +15：價格站上 60日 MA。")
    elif ma60 is not None:
        factor_reasons.append("趨勢 +0：價格低於 60日 MA。")

    if ma120 is not None and last_price > ma120:
        trend_score += 10
        factor_reasons.append("趨勢 +10：價格站上 120日 MA。")
    elif ma120 is not None:
        factor_reasons.append("趨勢 +0：價格低於 120日 MA。")

    if ma20 is not None and ma60 is not None and ma20 > ma60:
        trend_score += 10
        factor_reasons.append("趨勢 +10：20日 MA 高於 60日 MA。")
    elif ma20 is not None and ma60 is not None:
        factor_reasons.append("趨勢 +0：20日 MA 未高於 60日 MA。")
    trend_score = clamp(trend_score, 0, 100)

    one_month_return = pct_change(float(one_month.iloc[0]), last_price) if not one_month.empty else 0.0
    three_month_return = pct_change(float(three_month.iloc[0]), last_price) if not three_month.empty else 0.0
    momentum_score = clamp(50 + one_month_return * 2 + three_month_return, 0, 100)
    factor_reasons.append(
        f"動能分數 {momentum_score:.0f}：近 1 個月報酬為 {one_month_return:+.2f}%，"
        f"近 3 個月報酬為 {three_month_return:+.2f}%。"
    )

    volatility = annualized_volatility(clean.tail(60))
    volatility_score = clamp(100 - volatility * 2.2, 10, 100)

    vix_value = latest_value(prices[VIX_TICKER]) if VIX_TICKER in prices else None
    vix_score, vix_analysis = score_vix(vix_value)
    drawdown_pct = current_drawdown(clean)
    drawdown_score, drawdown_analysis = score_drawdown(drawdown_pct)
    rsi_value = latest_value(rsi(clean))
    rsi_score, rsi_analysis = score_rsi(rsi_value)
    all_time_high = float(clean.cummax().iloc[-1])
    distance_from_high_pct = pct_change(all_time_high, last_price)
    factor_reasons.append(f"VIX 分數 {vix_score:.0f}：{vix_analysis}")
    factor_reasons.append(f"RSI 分數 {rsi_score:.0f}：{rsi_analysis}")
    factor_reasons.append(f"回撤分數 {drawdown_score:.0f}：{drawdown_analysis}")
    factor_reasons.append(
        f"波動分數 {volatility_score:.0f}：60日年化波動率為 {volatility:.2f}%。"
    )

    overheating_points = 0
    direct_overheating_penalty = 0
    overheating_warnings = []
    if vix_value is not None and vix_value < 12:
        overheating_points += 2
        direct_overheating_penalty += 6
        overheating_warnings.append("VIX 低於 12，顯示市場可能過度自滿。")
    elif vix_value is not None and vix_value < 15:
        overheating_points += 1
        direct_overheating_penalty += 3
        overheating_warnings.append("VIX 異常平靜，低波動不應直接視為低風險。")

    if distance_from_high_pct > -2:
        overheating_points += 2
        direct_overheating_penalty += 8
        overheating_warnings.append("0050 距離載入期間高點不到 2%。")
    elif distance_from_high_pct > -5:
        overheating_points += 1
        direct_overheating_penalty += 4
        overheating_warnings.append("0050 距離載入期間高點不到 5%。")

    if rsi_value is not None and rsi_value >= 75:
        overheating_points += 2
        direct_overheating_penalty += 8
        overheating_warnings.append("RSI 高於 75，短線走勢明顯過熱。")
    elif rsi_value is not None and rsi_value >= 68:
        overheating_points += 1
        direct_overheating_penalty += 4
        overheating_warnings.append("RSI 偏高，新資金宜分批進場而非一次追價。")

    if one_month_return >= 8:
        overheating_points += 1
        direct_overheating_penalty += 3
        overheating_warnings.append("近 1 個月動能偏強，拉回風險升高。")

    if overheating_points >= 5:
        market_temperature = "狂熱"
        overheating_penalty = 22
    elif overheating_points >= 3:
        market_temperature = "偏熱"
        overheating_penalty = 12
    elif drawdown_pct <= -10 or (vix_value is not None and vix_value >= 25):
        market_temperature = "冷卻"
        overheating_penalty = 0
    else:
        market_temperature = "中性"
        overheating_penalty = 0
    total_overheating_penalty = direct_overheating_penalty + overheating_penalty
    factor_reasons.append(
        f"過熱扣分 -{total_overheating_penalty:.0f}："
        f"市場溫度為「{market_temperature}」，共偵測到 {overheating_points} 項過熱訊號。"
    )

    market_score = round(
        trend_score * 0.30
        + momentum_score * 0.20
        + vix_score * 0.20
        + drawdown_score * 0.20
        + rsi_score * 0.10
        - total_overheating_penalty
    )
    market_score = int(clamp(market_score, 0, 100))

    if market_score >= 75:
        recommendation = "建議加碼"
        stock_ratio = 80
        risk_level = "低風險"
        status_label = "偏多"
        status_color = "#16a34a"
    elif market_score >= 55:
        recommendation = "持有觀察"
        stock_ratio = 60
        risk_level = "中度風險"
        status_label = "均衡"
        status_color = "#ca8a04"
    else:
        recommendation = "暫緩進場"
        stock_ratio = 35
        risk_level = "高風險"
        status_label = "防禦"
        status_color = "#dc2626"

    if vix_value is not None and vix_value >= 25:
        stock_ratio = min(stock_ratio, 50)
    if drawdown_pct <= -15:
        stock_ratio = min(stock_ratio, 45)
    if market_temperature == "偏熱":
        stock_ratio = min(stock_ratio, 55)
        if recommendation == "建議加碼":
            recommendation = "持有觀察"
            risk_level = "中度風險"
            status_label = "均衡"
            status_color = "#ca8a04"
    if market_temperature == "狂熱":
        stock_ratio = min(stock_ratio, 40)
        recommendation = "暫緩進場"
        risk_level = "高風險"
        status_label = "防禦"
        status_color = "#dc2626"
    cash_ratio = 100 - stock_ratio
    rsi_text = "無資料" if rsi_value is None else f"{rsi_value:.1f}"

    strategy_summary = (
        f"{recommendation}：市場分數為 {market_score}/100，風險等級為「{risk_level}」，"
        f"市場溫度為「{market_temperature}」。"
        f"建議配置為股票 {stock_ratio}%／現金 {cash_ratio}%。"
        f"趨勢分數 {trend_score:.0f}，VIX 分數 {vix_score:.0f}，"
        f"RSI 為 {rsi_text}，目前回撤為 {drawdown_pct:.2f}%。"
    )
    reasoning_summary = (
        f"最終分數綜合趨勢（{trend_score:.0f}）、動能（{momentum_score:.0f}）、"
        f"VIX（{vix_score:.0f}）、回撤（{drawdown_score:.0f}）與 RSI（{rsi_score:.0f}），"
        f"再扣除 {total_overheating_penalty:.0f} 分過熱懲罰。"
        f"因此產生「{recommendation}」訊號，股票建議比重為 {stock_ratio}%。"
    )

    return DecisionEngineResult(
        market_score=int(market_score),
        recommendation=recommendation,
        stock_ratio=stock_ratio,
        cash_ratio=cash_ratio,
        risk_level=risk_level,
        market_temperature=market_temperature,
        status_label=status_label,
        status_color=status_color,
        vix_value=vix_value,
        vix_analysis=vix_analysis,
        rsi_value=rsi_value,
        trend_score=trend_score,
        vix_score=vix_score,
        rsi_score=rsi_score,
        drawdown_score=drawdown_score,
        momentum_score=momentum_score,
        overheating_penalty=total_overheating_penalty,
        factor_reasons=tuple(factor_reasons),
        drawdown_pct=drawdown_pct,
        drawdown_analysis=drawdown_analysis,
        overheating_warnings=tuple(overheating_warnings),
        strategy_summary=strategy_summary,
        reasoning_summary=reasoning_summary,
    )


def render_status_badge(result: DecisionEngineResult) -> None:
    st.markdown(
        f"""
        <div style="
            background:{result.status_color};
            color:white;
            padding:18px 22px;
            border-radius:8px;
            font-weight:700;
            font-size:22px;
            text-align:center;
        ">
            {result.status_label}盤勢：{result.recommendation}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_gauge(go: object, result: DecisionEngineResult) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=result.market_score,
            number={"suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": result.status_color},
                "steps": [
                    {"range": [0, 55], "color": "#fee2e2"},
                    {"range": [55, 75], "color": "#fef3c7"},
                    {"range": [75, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "thickness": 0.8,
                    "value": result.market_score,
                },
            },
        )
    )
    fig.update_layout(height=280, margin={"l": 20, "r": 20, "t": 20, "b": 10})
    st.plotly_chart(fig, use_container_width=True)


def temperature_color(temperature: str) -> str:
    return {
        "冷卻": "#2563eb",
        "中性": "#16a34a",
        "偏熱": "#ca8a04",
        "狂熱": "#dc2626",
    }.get(temperature, "#64748b")


def render_temperature_badge(result: DecisionEngineResult) -> None:
    color = temperature_color(result.market_temperature)
    st.markdown(
        f"""
        <div style="
            border:1px solid {color};
            color:{color};
            padding:14px 18px;
            border-radius:8px;
            font-weight:700;
            text-align:center;
        ">
            市場溫度：{result.market_temperature}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scoring_breakdown(result: DecisionEngineResult) -> None:
    breakdown = pd.DataFrame(
        [
            {"因子": "趨勢", "分數": round(result.trend_score, 1), "權重": "30%"},
            {"因子": "VIX", "分數": round(result.vix_score, 1), "權重": "20%"},
            {"因子": "回撤", "分數": round(result.drawdown_score, 1), "權重": "20%"},
            {"因子": "動能", "分數": round(result.momentum_score, 1), "權重": "20%"},
            {"因子": "RSI", "分數": round(result.rsi_score, 1), "權重": "10%"},
            {
                "因子": "過熱扣分",
                "分數": -round(result.overheating_penalty, 1),
                "權重": "扣分",
            },
        ]
    )
    st.dataframe(breakdown, hide_index=True, use_container_width=True)


def render_intraday_reference(reference: IntradayReference) -> None:
    st.markdown(
        """
        <div style="
            border:1px solid #2563eb;
            border-left:6px solid #2563eb;
            background:#eff6ff;
            color:#1e3a8a;
            padding:14px 18px;
            border-radius:8px;
            font-weight:700;
            margin:10px 0 14px 0;
        ">
            盤中參考：手動輸入價格
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("此區僅供盤中觀察，正式決策仍以收盤資料為準，不影響正式智慧分數與配置建議。")
    action_color = {
        "不買": "#64748b",
        "小量加碼": "#16a34a",
        "中度加碼": "#ca8a04",
        "大幅加碼": "#dc2626",
    }.get(reference.action_suggestion, "#64748b")
    st.markdown(
        f"""
        <div style="
            background:{action_color};
            color:white;
            padding:16px 18px;
            border-radius:8px;
            font-size:22px;
            font-weight:800;
            text-align:center;
            margin:8px 0 16px 0;
        ">
            盤中動作建議：{reference.action_suggestion}
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    with cols[0]:
        render_metric("手動盤中價格", format_price(reference.manual_price))
    with cols[1]:
        render_metric("最新收盤價", format_price(reference.latest_close))
    with cols[2]:
        render_metric("距最新收盤", format_pct(reference.difference_from_close_pct))
    with cols[3]:
        render_metric("盤中動作建議", reference.action_suggestion)

    high_cols = st.columns(2)
    with high_cols[0]:
        render_metric("60日近期高點", format_price(reference.recent_high))
    with high_cols[1]:
        render_metric("距近期高點回撤", format_pct(reference.drawdown_from_high_pct))

    st.markdown("**加碼區狀態**")
    zone_cols = st.columns(len(reference.add_on_zones))
    for col, (zone, entered) in zip(zone_cols, reference.add_on_zones.items()):
        color = "#16a34a" if entered else "#64748b"
        background = "#dcfce7" if entered else "#f8fafc"
        label = "已進入" if entered else "未進入"
        with col:
            st.markdown(
                f"""
                <div style="
                    border:1px solid {color};
                    background:{background};
                    color:{color};
                    padding:12px 10px;
                    border-radius:8px;
                    text-align:center;
                    font-weight:700;
                ">
                    <div>{zone}</div>
                    <div style="font-size:13px;">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_comparison_chart(go: object, prices: pd.DataFrame) -> None:
    normalized = normalize_to_100(prices)
    fig = go.Figure()
    palette = ["#2563eb", "#7c3aed", "#059669"]
    for index, column in enumerate(normalized.columns):
        label = next((name for name, ticker in PEER_TICKERS.items() if ticker == column), column)
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[column],
                mode="lines",
                name=label,
                line={"width": 2, "color": palette[index % len(palette)]},
            )
        )
    fig.update_layout(
        height=360,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        hovermode="x unified",
        yaxis_title="指數化表現",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_taiwan_stock_chart(go: object, prices: pd.Series, analysis: TaiwanStockAnalysis) -> None:
    clean = prices.dropna()
    if clean.empty:
        st.warning("此股票目前沒有足夠資料可繪製價格趨勢圖。")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=clean.index,
            y=clean,
            mode="lines",
            name="收盤價",
            line={"width": 2.5, "color": "#2563eb"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=clean.index,
            y=moving_average(clean, 20),
            mode="lines",
            name="MA20",
            line={"width": 1.5, "color": "#f59e0b"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=clean.index,
            y=moving_average(clean, 60),
            mode="lines",
            name="MA60",
            line={"width": 1.5, "color": "#10b981"},
        )
    )
    fig.update_layout(
        height=360,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        hovermode="x unified",
        yaxis_title="還原權值收盤價",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})


def render_taiwan_stock_analysis(go: object, analysis: TaiwanStockAnalysis, prices: pd.Series) -> None:
    st.markdown(f"### {analysis.label}")
    top_cols = st.columns(4)
    with top_cols[0]:
        render_metric("最新價格", format_price(analysis.latest_price))
    with top_cols[1]:
        render_metric("AI評分", f"{analysis.ai_score}/100")
    with top_cols[2]:
        render_metric("建議", analysis.recommendation)
    with top_cols[3]:
        render_metric("風險等級", analysis.risk_level)

    allocation_cols = st.columns(4)
    with allocation_cols[0]:
        render_metric("市場溫度", analysis.market_temperature)
    with allocation_cols[1]:
        render_metric("建議股票比例", f"{analysis.stock_ratio}%")
    with allocation_cols[2]:
        render_metric("建議現金比例", f"{analysis.cash_ratio}%")
    with allocation_cols[3]:
        render_metric("近期高點回撤", format_pct(analysis.drawdown_from_high_pct))

    indicator_cols = st.columns(4)
    with indicator_cols[0]:
        render_metric("RSI", "無資料" if analysis.rsi_value is None else f"{analysis.rsi_value:.1f}")
    with indicator_cols[1]:
        render_metric("MA20", "無資料" if analysis.ma20 is None else format_price(analysis.ma20))
    with indicator_cols[2]:
        render_metric("MA60", "無資料" if analysis.ma60 is None else format_price(analysis.ma60))
    with indicator_cols[3]:
        render_metric("MA120", "無資料" if analysis.ma120 is None else format_price(analysis.ma120))

    distance_cols = st.columns(3)
    with distance_cols[0]:
        render_metric("距 MA20", "無資料" if analysis.distance_ma20_pct is None else format_pct(analysis.distance_ma20_pct))
    with distance_cols[1]:
        render_metric("距 MA60", "無資料" if analysis.distance_ma60_pct is None else format_pct(analysis.distance_ma60_pct))
    with distance_cols[2]:
        render_metric("距 MA120", "無資料" if analysis.distance_ma120_pct is None else format_pct(analysis.distance_ma120_pct))

    st.info(analysis.reason_summary)
    if analysis.missing_background:
        st.caption("部分背景市場資料暫時無法取得，已略過該因子。")

    st.subheader("背景市場摘要")
    background_cols = st.columns(5)
    for col, (label, value) in zip(background_cols, analysis.background_summary.items()):
        with col:
            render_metric(label, value)

    st.subheader("價格趨勢")
    render_taiwan_stock_chart(go, prices, analysis)


def simulate_investment(
    prices: pd.Series,
    initial_investment: float,
    monthly_contribution: float,
) -> pd.DataFrame:
    monthly_prices = prices.dropna().resample("MS").first().dropna()
    if monthly_prices.empty:
        return pd.DataFrame(columns=["price", "shares", "contribution", "value"])

    shares = 0.0
    rows = []
    for i, (month, price) in enumerate(monthly_prices.items()):
        contribution = initial_investment if i == 0 else monthly_contribution
        shares += contribution / float(price)
        rows.append(
            {
                "date": month,
                "price": float(price),
                "shares": shares,
                "contribution": contribution,
                "value": shares * float(price),
            }
        )

    result = pd.DataFrame(rows).set_index("date")
    result["total_contributed"] = result["contribution"].cumsum()
    result["profit"] = result["value"] - result["total_contributed"]
    result["profit_pct"] = result["profit"] / result["total_contributed"] * 100.0
    return result


def render_simulation_chart(go: object, simulation: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=simulation.index,
            y=simulation["value"],
            mode="lines",
            name="投資組合市值",
            line={"width": 2.5, "color": "#2563eb"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=simulation.index,
            y=simulation["total_contributed"],
            mode="lines",
            name="累計投入本金",
            line={"width": 2, "color": "#64748b", "dash": "dash"},
        )
    )
    fig.update_layout(
        height=330,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        hovermode="x unified",
        yaxis_title="新台幣",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="0050 投資儀表板",
        page_icon="0050",
        layout="wide",
    )

    yf, go = require_optional_packages()

    st.title("0050 投資儀表板")
    st.caption("台灣50指數股票型基金行情快照、智慧決策引擎、趨勢檢視、同類比較與定期投入模擬。")

    if yf is None or go is None:
        render_package_help()
        st.stop()

    with st.sidebar:
        st.header("參數設定")
        data_mode = st.radio(
            "資料模式",
            options=["收盤模式", "盤中模式"],
            captions=[
                "自動抓取最新收盤價，作為正式智慧決策依據。",
                "正式智慧分數仍使用收盤資料，另以手動價格提供盤中參考。",
            ],
        )
        start = st.date_input("起始日期", value=date(2021, 1, 1), min_value=date(2008, 1, 1))
        end = st.date_input("結束日期", value=date.today())
        show_ma20 = st.checkbox("顯示 20日 MA", value=True)
        show_ma60 = st.checkbox("顯示 60日 MA", value=True)
        manual_intraday_price = None
        st.divider()
        initial_investment = st.number_input(
            "初始投入金額",
            min_value=0,
            value=100_000,
            step=10_000,
            format="%d",
        )
        monthly_contribution = st.number_input(
            "每月投入金額",
            min_value=0,
            value=10_000,
            step=1_000,
            format="%d",
        )

    if start >= end:
        st.warning("起始日期必須早於結束日期。")
        st.stop()

    tickers = list(dict.fromkeys([TICKER_0050, BENCHMARK_TICKER, VIX_TICKER, *PEER_TICKERS.values()]))

    with st.spinner("正在載入市場資料..."):
        prices = load_prices(tickers, start, end)

    if prices.empty or TICKER_0050 not in prices:
        st.warning("未取得 0050 價格資料。請調整日期區間或確認網路連線。")
        st.stop()

    price_0050 = prices[TICKER_0050].dropna()
    snapshot = build_snapshot(price_0050)
    decision = build_decision_engine(prices, price_0050)
    latest_close = float(price_0050.iloc[-1])
    if data_mode == "盤中模式":
        with st.sidebar:
            manual_intraday_price = st.number_input(
                "手動輸入 0050 盤中價格",
                min_value=0.0,
                value=latest_close,
                step=0.05,
                format="%.2f",
            )
    intraday_reference = None
    if data_mode == "盤中模式":
        intraday_price = latest_close if manual_intraday_price is None or manual_intraday_price <= 0 else float(manual_intraday_price)
        intraday_reference = build_intraday_reference(price_0050, intraday_price)
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    latest_close_date = price_0050.index[-1].strftime("%Y-%m-%d")

    metric_cols = st.columns(6)
    with metric_cols[0]:
        render_metric("最新價格", format_price(snapshot.last_price), format_pct(snapshot.daily_change_pct))
    with metric_cols[1]:
        render_metric("今年以來報酬", format_pct(snapshot.ytd_return_pct))
    with metric_cols[2]:
        render_metric("近一年報酬", format_pct(snapshot.one_year_return_pct))
    with metric_cols[3]:
        render_metric("年化波動率", f"{snapshot.annual_volatility_pct:.2f}%")
    with metric_cols[4]:
        render_metric("最大回撤", f"{snapshot.max_drawdown_pct:.2f}%")
    with metric_cols[5]:
        render_metric("資料筆數", f"{len(price_0050):,}")

    st.caption(f"最後更新時間：{updated_at}｜最新收盤日：{latest_close_date}｜目前模式：{data_mode}")

    tab_ai, tab_stock_ai, tab_trend, tab_simulator, tab_data = st.tabs(
        ["智慧決策", "台股AI分析", "趨勢", "模擬", "資料"]
    )

    with tab_ai:
        st.subheader("智慧決策引擎")
        st.markdown("**正式決策：收盤資料**")
        left, right = st.columns([1, 1])

        with left:
            render_decision_gauge(go, decision)

        with right:
            render_status_badge(decision)
            st.write("")
            render_temperature_badge(decision)
            st.write("")
            ai_cols = st.columns(3)
            with ai_cols[0]:
                render_metric("操作建議", decision.recommendation)
            with ai_cols[1]:
                render_metric("風險等級", decision.risk_level)
            with ai_cols[2]:
                render_metric("市場分數", f"{decision.market_score}/100")

            allocation_cols = st.columns(2)
            with allocation_cols[0]:
                render_metric("建議股票比重", f"{decision.stock_ratio}%")
            with allocation_cols[1]:
                render_metric("建議現金比重", f"{decision.cash_ratio}%")

        if intraday_reference is not None:
            render_intraday_reference(intraday_reference)

        st.subheader("每日策略摘要")
        st.info(decision.strategy_summary)

        st.subheader("評分拆解")
        render_scoring_breakdown(decision)

        st.subheader("加扣分原因")
        for reason in decision.factor_reasons:
            st.write(f"- {reason}")

        st.subheader("最終推論摘要")
        st.success(decision.reasoning_summary)

        if decision.overheating_warnings:
            st.warning(" ".join(decision.overheating_warnings))

        analysis_cols = st.columns(3)
        with analysis_cols[0]:
            st.subheader("VIX風險分析")
            vix_display = "無資料" if decision.vix_value is None else f"{decision.vix_value:.2f}"
            render_metric("最新 VIX", vix_display)
            st.write(decision.vix_analysis)

        with analysis_cols[1]:
            st.subheader("RSI過熱分析")
            rsi_display = "無資料" if decision.rsi_value is None else f"{decision.rsi_value:.1f}"
            render_metric("最新 RSI", rsi_display)
            if decision.rsi_value is None:
                st.write("所選區間無法計算 RSI。")
            elif decision.rsi_value >= 75:
                st.write("RSI 明顯過熱，模型已降低偏多信心。")
            elif decision.rsi_value >= 68:
                st.write("RSI 偏高，分批布局比一次提高曝險更穩健。")
            else:
                st.write("RSI 尚未出現明顯過熱。")

        with analysis_cols[2]:
            st.subheader("回撤分析")
            render_metric("目前回撤", f"{decision.drawdown_pct:.2f}%")
            st.write(decision.drawdown_analysis)

    with tab_stock_ai:
        st.subheader("台股AI分析")
        st.caption("請輸入台股代碼或常見公司名稱；背景市場僅作為風險參考，不作為分析標的。")
        stock_query = st.text_input(
            "台股代碼或公司名稱",
            value="2330",
            placeholder="例如：2330、台積電、2301、光寶科、3037、欣興",
        )

        st.caption("範例：2330 / 台積電、2301 / 光寶科、3037 / 欣興、2317 / 鴻海、2454 / 聯發科、2382 / 廣達、3231 / 緯創")

        query_label, resolved_ticker = resolve_taiwan_stock_query(stock_query)
        if resolved_ticker is None:
            st.warning("請輸入台股代碼，例如 2330，或常見公司名稱，例如 台積電。")
        else:
            with st.spinner("正在分析台股與背景市場資料..."):
                stock_label, stock_ticker, stock_prices = load_taiwan_stock_series(stock_query, start, end)

                if stock_prices.dropna().empty:
                    st.warning("目前無法取得此台股的有效歷史資料，請確認代碼或稍後再試。")
                else:
                    background_summary, background_penalty, background_bonus, missing_background = build_background_market_summary(start, end)
                    stock_analysis = analyze_taiwan_stock(
                        stock_label,
                        stock_ticker,
                        stock_prices,
                        background_summary,
                        background_penalty,
                        background_bonus,
                        missing_background,
                    )
                    if stock_analysis is None:
                        st.warning("此台股資料不足，暫時無法產生 AI 分析。")
                    else:
                        render_taiwan_stock_analysis(go, stock_analysis, stock_prices)

    with tab_trend:
        st.subheader("價格趨勢")
        render_price_chart(go, price_0050, show_ma20, show_ma60)

        st.subheader("回撤")
        render_drawdown_chart(go, price_0050)

    with tab_simulator:
        st.subheader("定期投入模擬")
        simulation = simulate_investment(
            price_0050,
            float(initial_investment),
            float(monthly_contribution),
        )

        if simulation.empty:
            st.info("資料不足，無法執行模擬。")
        else:
            latest = simulation.iloc[-1]
            sim_cols = st.columns(4)
            with sim_cols[0]:
                render_metric("投資組合市值", f"新台幣{latest['value']:,.0f}")
            with sim_cols[1]:
                render_metric("累計投入本金", f"新台幣{latest['total_contributed']:,.0f}")
            with sim_cols[2]:
                render_metric("累計損益", f"新台幣{latest['profit']:,.0f}")
            with sim_cols[3]:
                render_metric("累計報酬率", f"{latest['profit_pct']:+.2f}%")

            render_simulation_chart(go, simulation)

    with tab_data:
        st.subheader("最新資料")
        display = prices.drop(columns=[VIX_TICKER], errors="ignore").copy()
        display.columns = [
            next((name for name, ticker in PEER_TICKERS.items() if ticker == column), column)
            for column in display.columns
        ]
        st.dataframe(display.tail(100), use_container_width=True)

        csv = display.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="下載資料檔",
            data=csv,
            file_name="0050-投資儀表板資料.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
