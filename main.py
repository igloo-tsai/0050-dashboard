import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(
    page_title="0050 投資儀表板",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================
# CSS
# =========================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.big-card {
    padding: 18px;
    border-radius: 16px;
    background: #111827;
    border: 1px solid #374151;
    margin-bottom: 16px;
}

.signal-buy {
    padding: 16px;
    border-radius: 14px;
    background: #064e3b;
    border: 1px solid #10b981;
    color: white;
    font-size: 20px;
    font-weight: 700;
}

.signal-hold {
    padding: 16px;
    border-radius: 14px;
    background: #78350f;
    border: 1px solid #f59e0b;
    color: white;
    font-size: 20px;
    font-weight: 700;
}

.signal-wait {
    padding: 16px;
    border-radius: 14px;
    background: #7f1d1d;
    border: 1px solid #ef4444;
    color: white;
    font-size: 20px;
    font-weight: 700;
}

.small-note {
    color: #9ca3af;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# =========================
# 工具函數
# =========================
@st.cache_data(ttl=1800)
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

    return pd.to_numeric(close, errors="coerce").dropna()


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


def pct_change(first: float, last: float) -> float:
    if first == 0 or pd.isna(first) or pd.isna(last):
        return 0.0
    return (last / first - 1) * 100


def max_drawdown(close: pd.Series) -> float:
    close = close.dropna()
    if close.empty:
        return 0.0

    peak = close.cummax()
    drawdown = close / peak - 1
    return float(drawdown.min() * 100)


def format_price(value: float) -> str:
    return f"新台幣 {value:,.2f}"


def safe_price_chart(close: pd.Series, title: str = "價格趨勢"):
    try:
        chart_df = close.reset_index()
        chart_df.columns = ["Date", "Close"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_df["Date"],
            y=chart_df["Close"],
            mode="lines",
            name="收盤價"
        ))

        fig.update_layout(
            title=title,
            height=330,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="日期",
            yaxis_title="價格",
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        st.warning("價格趨勢圖暫時無法顯示，但不影響分析。")


def signal_class(recommendation: str) -> str:
    if "加碼" in recommendation or "分批" in recommendation:
        return "signal-buy"
    if "觀察" in recommendation:
        return "signal-hold"
    return "signal-wait"


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


def analyze_stock(ticker: str, intraday_price: float | None = None) -> dict | None:
    df = safe_download(ticker, period="2y")

    if df is None or df.empty:
        return None

    close = get_close_series(df)

    if close.empty or len(close) < 60:
        return None

    latest_close = safe_latest(close)
    decision_price = intraday_price if intraday_price and intraday_price > 0 else latest_close

    rsi_series = calculate_rsi(close)
    rsi_value = safe_latest(rsi_series, default=50.0)

    ma20 = safe_latest(close.rolling(20).mean(), default=latest_close)
    ma60 = safe_latest(close.rolling(60).mean(), default=latest_close)
    ma120 = safe_latest(close.rolling(120).mean(), default=latest_close)

    recent_high = float(close.tail(120).max())
    drawdown_from_high = pct_change(recent_high, decision_price)

    one_year_close = close.tail(252)
    one_year_return = (
        pct_change(float(one_year_close.iloc[0]), decision_price)
        if len(one_year_close) > 1
        else 0.0
    )

    volatility = float(close.pct_change().dropna().std() * np.sqrt(252) * 100)
    mdd = max_drawdown(close)
    intraday_gap = pct_change(latest_close, decision_price)

    score = 50
    reasons = []

    if decision_price > ma20:
        score += 5
        reasons.append("決策價格站上 MA20，短線趨勢偏多。")
    else:
        score -= 5
        reasons.append("決策價格低於 MA20，短線動能偏弱。")

    if decision_price > ma60:
        score += 8
        reasons.append("決策價格站上 MA60，中期趨勢仍有支撐。")
    else:
        score -= 8
        reasons.append("決策價格低於 MA60，中期趨勢偏保守。")

    if decision_price > ma120:
        score += 7
        reasons.append("決策價格站上 MA120，長線結構仍偏正向。")
    else:
        score -= 7
        reasons.append("決策價格低於 MA120，長線結構需觀察。")

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

    if intraday_gap >= 2:
        score -= 8
        reasons.append("盤中價格高於最新收盤價超過 2%，追價風險上升。")
    elif intraday_gap <= -2:
        score += 6
        reasons.append("盤中價格低於最新收盤價超過 2%，買點條件改善。")

    if volatility > 35:
        score -= 5
        reasons.append("年化波動率偏高，需降低單次投入比例。")

    score = max(0, min(100, int(score)))

    if score >= 72:
        recommendation = "可分批加碼"
        risk = "中等風險"
        temperature = "冷卻"
        stock_ratio = 70
    elif score >= 55:
        recommendation = "小量試單"
        risk = "中等風險"
        temperature = "中性偏冷"
        stock_ratio = 55
    elif score >= 42:
        recommendation = "持有觀察"
        risk = "中等偏高"
        temperature = "中性"
        stock_ratio = 45
    else:
        recommendation = "暫緩進場"
        risk = "高風險"
        temperature = "偏熱"
        stock_ratio = 30

    if rsi_value > 75:
        temperature = "狂熱"
        recommendation = "暫緩進場"
        risk = "高風險"
        stock_ratio = min(stock_ratio, 30)

    cash_ratio = 100 - stock_ratio

    return {
        "ticker": ticker,
        "close": close,
        "latest_close": latest_close,
        "decision_price": decision_price,
        "intraday_gap": intraday_gap,
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


def render_ai_dashboard(result: dict, title: str):
    st.markdown(f"### {title}")

    css_class = signal_class(result["recommendation"])
    st.markdown(
        f"""
        <div class="{css_class}">
            AI結論：{result["recommendation"]}｜分數 {result["score"]}/100｜市場溫度：{result["temperature"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)
    col1.metric("決策價格", format_price(result["decision_price"]))
    col2.metric("最新收盤價", format_price(result["latest_close"]))
    col3.metric("盤中偏離", f"{result['intraday_gap']:+.2f}%")

    col4, col5, col6 = st.columns(3)
    col4.metric("RSI", f"{result['rsi']:.1f}")
    col5.metric("風險等級", result["risk"])
    col6.metric("距高點回撤", f"{result['drawdown_from_high']:.2f}%")

    col7, col8 = st.columns(2)
    col7.metric("建議股票比例", f"{result['stock_ratio']}%")
    col8.metric("建議現金比例", f"{result['cash_ratio']}%")

    st.subheader("AI 判斷原因")
    for reason in result["reasons"]:
        st.write(f"- {reason}")

    with st.expander("查看技術指標細節"):
        st.write(f"MA20：{result['ma20']:.2f}")
        st.write(f"MA60：{result['ma60']:.2f}")
        st.write(f"MA120：{result['ma120']:.2f}")
        st.write(f"近一年報酬：{result['one_year_return']:.2f}%")
        st.write(f"年化波動率：{result['volatility']:.2f}%")
        st.write(f"最大回撤：{result['max_drawdown']:.2f}%")

    safe_price_chart(result["close"], title=f"{title} 價格趨勢")


# =========================
# App
# =========================
st.title("0050 投資儀表板")
st.caption("進階 AI 決策版：以收盤資料為基礎，並允許手動輸入盤中價格修正 AI 判斷。")

tab_0050, tab_stock = st.tabs(["0050分析", "台股AI分析"])


# =========================
# 0050 分析
# =========================
with tab_0050:
    st.header("0050 分析")

    base_result = analyze_stock("0050.TW")

    if base_result is None:
        st.error("目前無法取得 0050 資料，請稍後再試。")
    else:
        st.subheader("盤中決策輸入")

        col_input1, col_input2, col_input3 = st.columns(3)

        with col_input1:
            intraday_price = st.number_input(
                "0050 目前盤中價格",
                min_value=0.0,
                value=float(base_result["latest_close"]),
                step=0.05,
                format="%.2f",
                key="0050_intraday"
            )

        with col_input2:
            available_cash = st.number_input(
                "可用現金",
                min_value=0,
                value=1000000,
                step=10000,
                format="%d",
                key="0050_cash"
            )

        with col_input3:
            target_buy_price = st.number_input(
                "預計掛單價格",
                min_value=0.0,
                value=float(intraday_price),
                step=0.05,
                format="%.2f",
                key="0050_target_price"
            )

        estimated_lots = int(available_cash // (target_buy_price * 1000)) if target_buy_price > 0 else 0
        estimated_cost = estimated_lots * target_buy_price * 1000

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("預估可買張數", f"{estimated_lots} 張")
        col_b.metric("預估投入金額", f"新台幣 {estimated_cost:,.0f}")
        col_c.metric("剩餘現金", f"新台幣 {available_cash - estimated_cost:,.0f}")

        result = analyze_stock("0050.TW", intraday_price=intraday_price)
        render_ai_dashboard(result, "0050 元大台灣50")


# =========================
# 台股 AI 分析
# =========================
with tab_stock:
    st.header("台股 AI 分析")

    col_search1, col_search2 = st.columns([2, 1])

    with col_search1:
        user_input = st.text_input(
            "輸入台股代碼或名稱",
            value="2330",
            placeholder="例如：2330、台積電、2301、光寶科、3037、欣興",
        )

    ticker = convert_to_ticker(user_input)

    if ticker is None:
        st.warning("請輸入台股代碼，例如 2330，或常見公司名稱，例如 台積電。")
    else:
        base_result = analyze_stock(ticker)

        if base_result is None:
            st.error("目前無法取得有效資料，請確認代碼或稍後再試。")
        else:
            st.subheader("個股盤中決策輸入")

            col_input1, col_input2, col_input3 = st.columns(3)

            with col_input1:
                stock_intraday_price = st.number_input(
                    "目前盤中價格",
                    min_value=0.0,
                    value=float(base_result["latest_close"]),
                    step=0.05,
                    format="%.2f",
                    key="stock_intraday"
                )

            with col_input2:
                stock_cash = st.number_input(
                    "可用資金",
                    min_value=0,
                    value=300000,
                    step=10000,
                    format="%d",
                    key="stock_cash"
                )

            with col_input3:
                stock_target_price = st.number_input(
                    "預計掛單價格",
                    min_value=0.0,
                    value=float(stock_intraday_price),
                    step=0.05,
                    format="%.2f",
                    key="stock_target_price"
                )

            stock_estimated_lots = int(stock_cash // (stock_target_price * 1000)) if stock_target_price > 0 else 0
            stock_estimated_cost = stock_estimated_lots * stock_target_price * 1000

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("預估可買張數", f"{stock_estimated_lots} 張")
            col_b.metric("預估投入金額", f"新台幣 {stock_estimated_cost:,.0f}")
            col_c.metric("剩餘現金", f"新台幣 {stock_cash - stock_estimated_cost:,.0f}")

            result = analyze_stock(ticker, intraday_price=stock_intraday_price)
            render_ai_dashboard(result, f"{user_input}（{ticker}）")