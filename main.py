from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from data_service import fetch_taiwan_stock
from decision_engine import make_decision
from market_factors import get_market_background
from portfolio_engine import calculate_portfolio
from technical_analysis import build_technical_snapshot
from ui_components import (
    format_pct,
    format_price,
    inject_mobile_css,
    render_final_action_card,
    render_market_factor_card,
    render_position_summary_card,
    render_portfolio_table,
    render_price_trend_chart,
    render_risk_detail,
    render_score_cards,
    render_volume_card,
)
from volume_analysis import analyze_volume


TAIWAN_STOCK_ALIASES = {
    "0050": "0050.TW",
    "元大台灣50": "0050.TW",
    "2330": "2330.TW",
    "台積電": "2330.TW",
    "2317": "2317.TW",
    "鴻海": "2317.TW",
    "2301": "2301.TW",
    "光寶科": "2301.TW",
    "3037": "3037.TW",
    "欣興": "3037.TW",
    "2454": "2454.TW",
    "聯發科": "2454.TW",
    "2382": "2382.TW",
    "廣達": "2382.TW",
    "3231": "3231.TW",
    "緯創": "3231.TW",
}


def convert_to_ticker(query: str) -> tuple[str, str | None]:
    text = query.strip()
    if not text:
        return "", None
    if text in TAIWAN_STOCK_ALIASES:
        return text, TAIWAN_STOCK_ALIASES[text]
    normalized = text.upper().replace(".TW", "")
    if normalized.isdigit() and len(normalized) == 4:
        return normalized, f"{normalized}.TW"
    return text, None


def display_name(label: str, ticker: str) -> str:
    if label and not label.isdigit():
        return f"{label}（{ticker}）"
    return ticker


def apply_intraday_price(data: pd.DataFrame, intraday_price: float | None) -> pd.DataFrame:
    if data.empty or intraday_price is None or intraday_price <= 0 or "Close" not in data:
        return data
    adjusted = data.copy()
    valid_close = adjusted["Close"].dropna()
    if valid_close.empty:
        return adjusted
    adjusted.loc[valid_close.index[-1], "Close"] = intraday_price
    return adjusted


def render_inputs(prefix: str, default_price: float) -> dict[str, float | None]:
    st.subheader("投資參數")
    cols = st.columns(2)
    with cols[0]:
        holding_lots = st.number_input("目前已持有幾張", min_value=0.0, value=0.0, step=1.0, key=f"{prefix}_holding_lots")
        average_cost = st.number_input("目前每股平均成本", min_value=0.0, value=0.0, step=0.1, key=f"{prefix}_average_cost")
        cash = st.number_input("可投入現金", min_value=0.0, value=100_000.0, step=10_000.0, key=f"{prefix}_cash")
        intraday_price = st.number_input("目前市價", min_value=0.0, value=float(default_price), step=0.05, key=f"{prefix}_intraday_price")
    with cols[1]:
        manual_volume = st.number_input("今日成交量，可選填", min_value=0.0, value=0.0, step=1000.0, key=f"{prefix}_manual_volume")
        max_single = st.number_input("這次最多想投入多少", min_value=0.0, value=100_000.0, step=10_000.0, key=f"{prefix}_max_single")
        max_ratio = st.slider("股票部位上限", min_value=0, max_value=100, value=70, step=5, key=f"{prefix}_max_ratio")
    return {
        "holding_lots": holding_lots,
        "average_cost": average_cost,
        "cash": cash,
        "intraday_price": intraday_price,
        "manual_volume": manual_volume if manual_volume > 0 else None,
        "max_single": max_single,
        "max_ratio": float(max_ratio),
    }


def run_analysis_page(label: str, ticker: str, prefix: str, start: date, end: date) -> None:
    with st.spinner("正在取得行情與計算 AI 決策..."):
        resolved_ticker, data = fetch_taiwan_stock(ticker, start, end)

    if data.empty or data["Close"].dropna().empty:
        st.warning("資料不足，使用簡化模型")
        return

    latest_close = float(data["Close"].dropna().iloc[-1])
    inputs = render_inputs(prefix, latest_close)
    current_price = float(inputs["intraday_price"] or latest_close)
    analysis_data = apply_intraday_price(data, current_price)

    tech = build_technical_snapshot(analysis_data)
    if not tech:
        st.warning("資料不足，使用簡化模型")
        return
    if tech.get("is_simplified"):
        st.info("目前使用簡化分析（資料不足）")
    st.caption(f"分析模式：{tech.get('analysis_level', 'minimal')}")

    volume = analyze_volume(analysis_data, inputs["manual_volume"])
    market = get_market_background(start, end)
    portfolio = calculate_portfolio(
        holding_lots=float(inputs["holding_lots"] or 0),
        average_cost=float(inputs["average_cost"] or 0),
        cash=float(inputs["cash"] or 0),
        current_price=current_price,
        max_single_investment=float(inputs["max_single"] or 0),
        max_stock_ratio=float(inputs["max_ratio"] or 0),
    )
    decision = make_decision(
        tech=tech,
        volume=volume,
        market=market,
        portfolio=portfolio,
        max_stock_ratio=float(inputs["max_ratio"] or 0),
        current_price=current_price,
    )

    st.caption(f"分析標的：{display_name(label, resolved_ticker)}｜最新收盤價：{format_price(latest_close)}｜盤中價格：{format_price(current_price)}")
    render_position_summary_card(portfolio)
    render_final_action_card(decision, portfolio)

    with st.expander("進階分析細節", expanded=False):
        render_risk_detail(decision)
        render_score_cards(decision.module_scores, decision.total_score)
        render_volume_card(volume)
        render_market_factor_card(market)
        render_portfolio_table(portfolio)
        st.subheader("技術細節")
        detail_cols = st.columns(4)
        with detail_cols[0]:
            st.metric("RSI", "無資料" if tech.get("rsi") is None else f"{tech['rsi']:.1f}")
            st.metric("簡單漲跌幅", format_pct(tech.get("simple_return")))
            st.metric("近一年報酬", format_pct(tech.get("one_year_return")))
        with detail_cols[1]:
            st.metric("MA20", format_price(tech.get("ma20")))
            st.metric("距 MA20", format_pct(tech.get("distance_ma20")))
        with detail_cols[2]:
            st.metric("MA60", format_price(tech.get("ma60")))
            st.metric("距 MA60", format_pct(tech.get("distance_ma60")))
        with detail_cols[3]:
            st.metric("MA120", format_price(tech.get("ma120")))
            st.metric("距 MA120", format_pct(tech.get("distance_ma120")))
        st.metric("近120日高點回撤", format_pct(tech.get("recent_high_drawdown")))
        st.metric("年化波動率", format_pct(tech.get("annual_volatility")))
        st.metric("最大回撤", format_pct(tech.get("max_drawdown")))
        st.write("AI判斷原因")
        for reason in decision.reasons:
            st.write(f"- {reason}")

    render_price_trend_chart(analysis_data, f"{display_name(label, resolved_ticker)} 價格趨勢")


def main() -> None:
    st.set_page_config(page_title="AI 投資決策資訊系統", page_icon="0050", layout="wide")
    inject_mobile_css()
    st.title("AI 投資決策資訊系統")
    st.caption("以 0050 與台股個股為核心，整合技術面、量能、市場背景與個人持倉風控。")

    start = date(2021, 1, 1)
    end = date.today()
    tab_0050, tab_stock = st.tabs(["0050 AI決策", "台股AI分析"])

    with tab_0050:
        st.subheader("0050 AI決策")
        run_analysis_page("元大台灣50", "0050.TW", "etf0050", start, end)

    with tab_stock:
        st.subheader("台股AI分析")
        query = st.text_input(
            "股票代碼或公司名稱",
            value="2330",
            placeholder="例如：2330、台積電、2317、鴻海",
            key="stock_query",
        )
        st.caption("支援：0050 / 元大台灣50、2330 / 台積電、2317 / 鴻海、2301 / 光寶科、3037 / 欣興、2454 / 聯發科、2382 / 廣達、3231 / 緯創")
        label, ticker = convert_to_ticker(query)
        if ticker is None:
            st.warning("請輸入台股代碼，例如 2330，或常見公司名稱，例如 台積電。")
        else:
            run_analysis_page(label, ticker, "stock_ai", start, end)


if __name__ == "__main__":
    main()
