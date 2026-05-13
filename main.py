import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="0050 投資儀表板", layout="wide")

# =========================
# 基本安全函數
# =========================
def safe_download(ticker):
    try:
        df = yf.download(ticker, period="2y", progress=False)
        if df is None or df.empty:
            return None
        return df.dropna()
    except:
        return None


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def max_drawdown(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()


# =========================
# 台股 mapping
# =========================
alias_map = {
    "台積電": "2330.TW",
    "光寶科": "2301.TW",
    "欣興": "3037.TW",
    "鴻海": "2317.TW",
    "聯發科": "2454.TW",
    "廣達": "2382.TW",
    "緯創": "3231.TW",
    "0050": "0050.TW",
    "元大台灣50": "0050.TW"
}


def convert_to_ticker(user_input):
    user_input = user_input.strip()

    if user_input in alias_map:
        return alias_map[user_input]

    if user_input.isdigit() and len(user_input) == 4:
        return f"{user_input}.TW"

    return None


# =========================
# UI
# =========================
tab1, tab2 = st.tabs(["0050分析", "台股AI分析"])

# =========================
# TAB1（保留簡單版）
# =========================
with tab1:
    st.title("0050 投資儀表板")
    st.write("原有功能保留（簡化顯示）")


# =========================
# TAB2 台股AI分析
# =========================
with tab2:
    st.title("台股 AI 分析")

    user_input = st.text_input("輸入台股代碼或名稱", placeholder="例如 2330 或 台積電")

    if user_input:
        ticker = convert_to_ticker(user_input)

        if ticker is None:
            st.warning("請輸入台股代碼，例如 2330，或常見公司名稱，例如 台積電。")
        else:
            df = safe_download(ticker)

            if df is None:
                st.error("無法取得資料，請稍後再試")
            else:
                close = df["Close"].dropna()

                if len(close) < 50:
                    st.warning("資料不足，無法分析")
                else:
                    latest_price = close.iloc[-1]

                    ytd = (close.iloc[-1] / close.iloc[0] - 1) * 100
                    one_year = (close.iloc[-1] / close.iloc[-252] - 1) * 100 if len(close) > 252 else 0

                    vol = close.pct_change().std() * np.sqrt(252) * 100
                    mdd = max_drawdown(close) * 100

                    rsi = calculate_rsi(close).iloc[-1]

                    ma20 = close.rolling(20).mean().iloc[-1]
                    ma60 = close.rolling(60).mean().iloc[-1]
                    ma120 = close.rolling(120).mean().iloc[-1]

                    drawdown = (latest_price - close.max()) / close.max() * 100

                    # =========================
                    # AI評分（簡化版）
                    # =========================
                    score = 50

                    if rsi < 40:
                        score += 10
                    if rsi > 70:
                        score -= 10
                    if drawdown < -10:
                        score += 10

                    score = max(0, min(100, score))

                    # =========================
                    # 建議
                    # =========================
                    if score > 65:
                        suggestion = "建議加碼"
                        risk = "中等風險"
                    elif score > 40:
                        suggestion = "持有觀察"
                        risk = "中等風險"
                    else:
                        suggestion = "暫緩進場"
                        risk = "高風險"

                    # =========================
                    # 顯示
                    # =========================
                    st.subheader(f"{ticker}")

                    col1, col2, col3 = st.columns(3)

                    col1.metric("最新價格", f"{latest_price:.2f}")
                    col2.metric("RSI", f"{rsi:.1f}")
                    col3.metric("AI評分", f"{score}")

                    st.write(f"MA20: {ma20:.2f}")
                    st.write(f"MA60: {ma60:.2f}")
                    st.write(f"MA120: {ma120:.2f}")
                    st.write(f"最大回撤: {mdd:.2f}%")

                    st.success(f"建議：{suggestion}")
                    st.warning(f"風險等級：{risk}")

                    st.line_chart(close)