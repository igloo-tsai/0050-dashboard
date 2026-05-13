import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="0050 投資儀表板", layout="wide")


# =========================
# 工具函數
# =========================
def safe_download(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        return df.dropna(how="all")
    except Exception:
        return None


def get_close_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    close = df["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce").dropna()
    return close


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.dropna()


def safe_latest(series: pd.Series, default: float = 0.0) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return default
    return float(series.iloc[-1])


def max_drawdown(close: pd.Series) -> float:
    close = close.dropna()
    if close.empty:
        return 0.0

    peak = close.cummax()
    drawdown = close / peak - 1
    return float(drawdown.min() * 100)


def pct_change(first: float, last: float) -> float:
    if first == 0 or pd.isna(first) or pd.isna(last):
        return 0.0
    return (last / first - 1) * 100


def format_pct(value: float) -> str:
    return f"{value:+.2f}%"


def format_price(value: float) -> str:
    return f"新台幣 {value:,.2f}"


def safe_price_chart(close: pd.Series, title: str = "價格趨勢"):
    try:
        chart_df = close.reset_index()
        chart_df.columns = ["Date", "Close"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_df["Date"],
                y=chart_df["Close"],
                mode="lines",
                name="收盤價",
            )
        )

        fig.update_layout(
            title=title,
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="日期",
            yaxis_title="價格",
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        st.warning("價格趨勢圖暫時無法顯示，但不影響其他分析功能。")


# =========================
# 台股代碼對照
# =========================
ALIAS_MAP = {
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


def convert_to_ticker(user_input: str) -> str | None:
    text = user_input.strip()

    if text in ALIAS_MAP:
        return ALIAS_MAP[text]

    if text.isdigit() and len(text) == 4:
        return f"{text}.TW"

    return None


def analyze_stock(ticker: str) -> dict | None:
    df = safe_download(ticker, period="2y")

    if df is None or df.empty:
        return None

    close = get_close_series(df)

    if close.empty or len(close) < 60:
        return None

    latest_price = safe_latest(close)

    rsi_series = calculate_rsi(close)
    rsi_value = safe_latest(rsi_series, default=50.0)

    ma20 = safe_latest(close.rolling(20).mean(), default=latest_price)
    ma60 = safe_latest(close.rolling(60).mean(), default=latest_price)
    ma120 = safe_latest(close.rolling(120).mean(), default=latest_price)

    recent_high = float(close.tail(120).max())
    drawdown_from_high = pct_change(recent_high, latest_price)

    one_year_close = close.tail(252)
    one_year_return = (
        pct_change(float(one_year_close.iloc[0]), latest_price)
        if len(one_year_close) > 1
        else 0.0
    )

    volatility = float(close.pct_change().dropna().std() * np.sqrt(252) * 100)
    mdd = max_drawdown(close)

    score = 50
    reasons = []

    if latest_price > ma20:
        score += 5
        reasons.append("價格站上 MA20，短線趨勢偏多。")
    else:
        score -= 5
        reasons.append("價格低於 MA20，短線動能偏弱。")

    if latest_price > ma60:
        score += 8
        reasons.append("價格站上 MA60，中期趨勢仍有支撐。")
    else:
        score -= 8
        reasons.append("價格低於 MA60，中期趨勢偏保守。")

    if latest_price > ma120:
        score += 7
        reasons.append("價格站上 MA120，長線結構仍偏正向。")
    else:
        score -= 7
        reasons.append("價格低於 MA120，長線結構需觀察。")

    if rsi_value < 40:
        score += 10
        reasons.append("RSI 低於 40，短線有修正後反彈機會。")
    elif rsi_value > 70:
        score -= 12
        reasons.append("RSI 高於 70，短線過熱，不宜追價。")
    else:
        reasons.append("RSI 位於中性區間。")

    if drawdown_from_high <= -20:
        score += 15
        reasons.append("距近期高點回撤超過 20%，已進入深度修正區。")
    elif drawdown_from_high <= -10:
        score += 8
        reasons.append("距近期高點回撤超過 10%，可分批觀察。")
    elif drawdown_from_high > -3:
        score -= 10
        reasons.append("價格接近近期高點，追價風險偏高。")

    score = max(0, min(100, int(score)))

    if score >= 70:
        recommendation = "建議加碼"
        risk = "中等風險"
        temperature = "冷卻"
        stock_ratio = 70
    elif score >= 45:
        recommendation = "持有觀察"
        risk = "中等風險"
        temperature = "中性"
        stock_ratio = 55
    else:
        recommendation = "暫緩進場"
        risk = "高風險"
        temperature = "偏熱"
        stock_ratio = 35

    if rsi_value > 75:
        temperature = "狂熱"
        recommendation = "暫緩進場"
        risk = "高風險"
        stock_ratio = min(stock_ratio, 35)

    cash_ratio = 100 - stock_ratio

    return {
        "ticker": ticker,
        "close": close,
        "latest_price": latest_price,
        "rsi": rsi_value,
        "ma20": ma20,
        "ma60": ma60,
        "ma120": ma120,
        "drawdown_from_high": drawdown_from_high,
        "one_year_return": one_year_return,
        "volatility": volatility,
        "max_drawdown": mdd,
        "score": score,
        "recommendation": recommendation,
        "risk": risk,
        "temperature": temperature,
        "stock_ratio": stock_ratio,
        "cash_ratio": cash_ratio,
        "reasons": reasons,
    }


# =========================
# App
# =========================
st.title("0050 投資儀表板")

tab_0050, tab_stock = st.tabs(["0050分析", "台股AI分析"])


# =========================
# 0050 分析
# =========================
with tab_0050:
    st.header("0050 分析")

    result = analyze_stock("0050.TW")

    if result is None:
        st.error("目前無法取得 0050 資料，請稍後再試。")
    else:
        st.subheader("盤中價格資訊")

        intraday_price = st.number_input(
            "請輸入目前盤中價格",
            min_value=0.0,
            value=float(result["latest_price"]),
            step=0.05,
            format="%.2f",
        )

        available_cash = st.number_input(
            "可用現金",
            min_value=0,
            value=1000000,
            step=10000,
            format="%d",
        )

        target_buy_price = st.number_input(
            "預計掛單價格",
            min_value=0.0,
            value=95.50,
            step=0.05,
            format="%.2f",
        )

        estimated_shares = int(available_cash // (target_buy_price * 1000)) if target_buy_price > 0 else 0

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("盤中價格", format_price(intraday_price))
        col_b.metric("可用現金", f"新台幣 {available_cash:,.0f}")
        col_c.metric("可買張數", f"{estimated_shares} 張")

        st.divider()

        col1, col2, col3 = st.columns(3)
        col1.metric("最新收盤價", format_price(result["latest_price"]))
        col2.metric("AI評分", f"{result['score']}/100")
        col3.metric("操作建議", result["recommendation"])

        col4, col5, col6 = st.columns(3)
        col4.metric("RSI", f"{result['rsi']:.1f}")
        col5.metric("市場溫度", result["temperature"])
        col6.metric("風險等級", result["risk"])

        st.subheader("AI 原因摘要")
        for reason in result["reasons"]:
            st.write(f"- {reason}")

        st.subheader("0050 價格趨勢")
        safe_price_chart(result["close"], title="0050 價格趨勢")


# =========================
# 台股 AI 分析
# =========================
with tab_stock:
    st.header("台股 AI 分析")

    user_input = st.text_input(
        "輸入台股代碼或名稱",
        value="2330",
        placeholder="例如：2330、台積電、2301、光寶科、3037、欣興",
    )

    ticker = convert_to_ticker(user_input)

    if ticker is None:
        st.warning("請輸入台股代碼，例如 2330，或常見公司名稱，例如 台積電。")
    else:
        result = analyze_stock(ticker)

        if result is None:
            st.error("目前無法取得有效資料，請確認代碼或稍後再試。")
        else:
            st.subheader(f"{user_input}（{ticker}）")

            col1, col2, col3 = st.columns(3)
            col1.metric("最新價格", format_price(result["latest_price"]))
            col2.metric("AI評分", f"{result['score']}/100")
            col3.metric("操作建議", result["recommendation"])

            col4, col5, col6 = st.columns(3)
            col4.metric("RSI", f"{result['rsi']:.1f}")
            col5.metric("市場溫度", result["temperature"])
            col6.metric("風險等級", result["risk"])

            col7, col8 = st.columns(2)
            col7.metric("建議股票比例", f"{result['stock_ratio']}%")
            col8.metric("建議現金比例", f"{result['cash_ratio']}%")

            st.subheader("技術指標")
            st.write(f"MA20：{result['ma20']:.2f}")
            st.write(f"MA60：{result['ma60']:.2f}")
            st.write(f"MA120：{result['ma120']:.2f}")
            st.write(f"距近期高點回撤：{result['drawdown_from_high']:.2f}%")
            st.write(f"近一年報酬：{result['one_year_return']:.2f}%")
            st.write(f"年化波動率：{result['volatility']:.2f}%")
            st.write(f"最大回撤：{result['max_drawdown']:.2f}%")

            st.subheader("AI 原因摘要")
            for reason in result["reasons"]:
                st.write(f"- {reason}")

            st.subheader("價格趨勢")
            safe_price_chart(result["close"], title=f"{user_input} 價格趨勢")