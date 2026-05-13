from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from technical_analysis import moving_average


def format_price(value: float | None) -> str:
    if value is None:
        return "無資料"
    return f"NT${value:,.2f}"


def format_pct(value: float | None) -> str:
    if value is None:
        return "無資料"
    return f"{value:+.2f}%"


def inject_mobile_css() -> None:
    st.markdown(
        """
        <style>
        * { box-sizing: border-box; }
        @media (max-width: 768px) {
            .block-container { padding: 0.9rem 0.75rem 2rem; max-width: 100%; }
            h1 { font-size: 1.55rem !important; line-height: 1.25 !important; }
            h2, h3 { font-size: 1.12rem !important; line-height: 1.3 !important; }
            [data-testid="stMetric"] {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 0.75rem 0.8rem;
                background: #ffffff;
            }
            [data-testid="stMetric"] label,
            [data-testid="stMetric"] [data-testid="stMetricValue"],
            [data-testid="stMetric"] [data-testid="stMetricDelta"] {
                white-space: normal !important;
                overflow: visible !important;
                text-overflow: clip !important;
                overflow-wrap: anywhere !important;
                line-height: 1.25 !important;
            }
            [data-testid="stMetric"] [data-testid="stMetricValue"] {
                font-size: 1.18rem !important;
            }
            .final-action-card {
                padding: 1rem !important;
                font-size: 1rem !important;
                line-height: 1.35 !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_final_action_card(decision, portfolio: dict[str, object]) -> None:
    color = "#16a34a" if decision.recommendation == "建議加碼" else "#ca8a04"
    if decision.recommendation == "暫緩進場":
        color = "#dc2626"
    table = portfolio.get("scenario_table", pd.DataFrame())
    selected = table[table["加碼張數"] == max(1, decision.suggested_buy_lots)] if not table.empty else pd.DataFrame()
    if decision.suggested_buy_lots == 0 or selected.empty:
        avg_cost = "不適用"
        remaining_cash = "不適用"
        stock_ratio = "不適用"
    else:
        row = selected.iloc[0]
        avg_cost = format_price(float(row["加碼後平均成本"]))
        remaining_cash = format_price(float(row["加碼後剩餘現金"]))
        stock_ratio = format_pct(float(row["加碼後股票資產比例"]))

    st.markdown(
        f"""
        <div class="final-action-card" style="
            border-left: 8px solid {color};
            background: #f8fafc;
            border-radius: 8px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
        ">
            <h3 style="margin:0 0 .6rem 0;">AI最終行動建議：{decision.recommendation}</h3>
            <div>今天買不買：<b>{decision.recommendation}</b></div>
            <div>建議買幾張：<b>{decision.suggested_buy_lots}</b> 張（最大 {decision.max_buy_lots} 張）</div>
            <div>掛單價格區間：積極 {format_price(decision.aggressive_bid)}／合理 {format_price(decision.reasonable_bid)}／保守 {format_price(decision.conservative_bid)}</div>
            <div>是否追價：<b>{decision.chase_today}</b></div>
            <div>買完後平均成本：<b>{avg_cost}</b></div>
            <div>買完後剩餘現金：<b>{remaining_cash}</b></div>
            <div>買完後股票資產比例：<b>{stock_ratio}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_cards(module_scores: dict[str, int], total_score: int) -> None:
    st.subheader("5大模組分數")
    cols = st.columns(5)
    for col, (name, score) in zip(cols, module_scores.items()):
        with col:
            st.metric(name, f"{score}/100")
    st.metric("AI總分", f"{total_score}/100")


def render_volume_card(volume: dict[str, object]) -> None:
    st.subheader("成交量判讀")
    cols = st.columns(3)
    with cols[0]:
        st.metric("量能型態", str(volume.get("volume_signal", "量能資料不足")))
    with cols[1]:
        ratio = volume.get("volume_ratio")
        st.metric("最新量 / 20日均量", "無資料" if ratio is None else f"{float(ratio):.2f}x")
    with cols[2]:
        st.metric("20日平均成交量", "無資料" if volume.get("avg20_volume") is None else f"{float(volume['avg20_volume']):,.0f}")


def render_market_factor_card(market: dict[str, object]) -> None:
    st.subheader("市場背景摘要")
    if market.get("missing"):
        st.caption("部分背景市場資料暫時無法取得，已略過該因子。")
    summary = market.get("summary", {})
    cols = st.columns(5)
    for col, (name, value) in zip(cols, summary.items()):
        with col:
            st.metric(name, str(value))


def render_portfolio_table(portfolio: dict[str, object]) -> None:
    st.subheader("持倉試算表")
    cols = st.columns(3)
    with cols[0]:
        st.metric("目前持倉市值", format_price(float(portfolio.get("market_value", 0.0))))
    with cols[1]:
        st.metric("未實現損益", format_price(float(portfolio.get("unrealized_pnl", 0.0))), format_pct(float(portfolio.get("unrealized_pnl_pct", 0.0))))
    with cols[2]:
        st.metric("目前股票資產比例", format_pct(float(portfolio.get("current_stock_ratio", 0.0))))
    table = portfolio.get("scenario_table", pd.DataFrame())
    if not table.empty:
        display = table.copy()
        for column in ("加碼後平均成本", "加碼後剩餘現金"):
            display[column] = display[column].map(lambda value: f"NT${value:,.0f}")
        display["加碼後股票資產比例"] = display["加碼後股票資產比例"].map(lambda value: f"{value:.2f}%")
        st.dataframe(display, hide_index=True, use_container_width=True)


def render_price_trend_chart(data: pd.DataFrame, title: str) -> None:
    if data is None or data.empty or "Close" not in data:
        st.warning("價格資料不足，暫時無法繪製價格趨勢圖。")
        return
    close = pd.to_numeric(data["Close"], errors="coerce").dropna()
    if close.empty:
        st.warning("價格資料不足，暫時無法繪製價格趨勢圖。")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="收盤價", line={"color": "#2563eb", "width": 2.5}))
    fig.add_trace(go.Scatter(x=close.index, y=moving_average(close, 20), mode="lines", name="MA20", line={"color": "#f59e0b", "width": 1.5}))
    fig.add_trace(go.Scatter(x=close.index, y=moving_average(close, 60), mode="lines", name="MA60", line={"color": "#10b981", "width": 1.5}))
    fig.update_layout(
        title=title,
        height=380,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        hovermode="x unified",
        template="plotly_white",
        yaxis_title="還原權值收盤價",
    )
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
