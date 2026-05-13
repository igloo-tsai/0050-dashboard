from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from technical_analysis import moving_average


def format_price(value: float | None) -> str:
    if value is None:
        return "無資料"
    try:
        return f"NT${float(value):,.2f}"
    except Exception:
        return "無資料"


def format_pct(value: float | None) -> str:
    if value is None:
        return "無資料"
    try:
        return f"{float(value):+.2f}%"
    except Exception:
        return "無資料"


def format_ratio(value: float | None) -> str:
    if value is None:
        return "無資料"
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "無資料"


def inject_mobile_css() -> None:
    st.markdown(
        """
        <style>
        * { box-sizing: border-box; }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        .final-action-card,
        .position-summary-card {
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        .decision-grid,
        .position-grid {
            display: grid;
            gap: 12px;
        }

        .decision-grid {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }

        .position-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .card-stat {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 10px;
            padding: 12px;
            min-height: 74px;
        }

        .card-label {
            color: #d1d5db;
            font-size: 13px;
            margin-bottom: 5px;
            line-height: 1.25;
        }

        .card-value {
            color: #ffffff;
            font-size: 20px;
            font-weight: 800;
            line-height: 1.25;
        }

        .decision-section-title {
            color: #ffffff;
            font-size: 17px;
            font-weight: 800;
            margin: 16px 0 9px;
        }

        .decision-main-text {
            color: #ffffff;
            font-size: 18px;
            font-weight: 800;
            line-height: 1.55;
            margin-top: 4px;
        }

        .decision-detail-text {
            color: #e5e7eb;
            margin-top: 10px;
            line-height: 1.7;
        }

        .risk-track {
            height: 10px;
            background: rgba(255,255,255,.25);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 6px;
        }

        .risk-fill {
            height: 10px;
            background: #ef4444;
            border-radius: 999px;
        }

        @media (max-width: 768px) {
            .block-container {
                padding: 0.9rem 0.75rem 2rem !important;
                max-width: 100% !important;
            }

            h1 {
                font-size: 1.55rem !important;
                line-height: 1.25 !important;
            }

            h2, h3 {
                font-size: 1.12rem !important;
                line-height: 1.3 !important;
            }

            .final-action-card,
            .position-summary-card {
                padding: 16px !important;
                font-size: 1rem !important;
                line-height: 1.45 !important;
            }

            .decision-grid,
            .position-grid {
                grid-template-columns: 1fr !important;
            }

            .card-value {
                font-size: 18px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def decision_colors(action_label: str) -> tuple[str, str]:
    label = str(action_label or "")
    if any(keyword in label for keyword in ("積極", "分批", "可加碼")):
        return "#064e3b", "#10b981"
    if any(keyword in label for keyword in ("觀察", "試單", "小量")):
        return "#78350f", "#f59e0b"
    return "#7f1d1d", "#ef4444"


def render_position_summary_card(portfolio: dict[str, object]) -> None:
    market_value = float(portfolio.get("market_value", 0.0) or 0.0)
    cash = float(portfolio.get("cash", 0.0) or 0.0)
    total_assets = float(portfolio.get("total_assets", 0.0) or 0.0)
    stock_ratio = float(portfolio.get("current_stock_ratio", 0.0) or 0.0)
    target_ratio = float(portfolio.get("max_stock_ratio", 0.0) or 0.0)
    excess = max(0.0, float(portfolio.get("excess_stock_ratio", 0.0) or 0.0))
    over_target = bool(portfolio.get("over_target_ratio", False))

    status = f"已超標 {excess:.1f}%，不建議加碼" if over_target else "未超標，可依訊號控管部位"
    border = "#ef4444" if over_target else "#10b981"

    html = f"""
    <div class="position-summary-card" style="
        border-left: 6px solid {border};
        background-color: #111827;
        color: #ffffff;
        border-radius: 12px;
        padding: 18px;
        margin: 12px 0 16px 0;
    ">
        <div style="font-size:20px;font-weight:800;margin-bottom:12px;">目前部位摘要</div>
        <div class="position-grid">
            <div class="card-stat"><div class="card-label">目前持倉市值</div><div class="card-value">{format_price(market_value)}</div></div>
            <div class="card-stat"><div class="card-label">可用現金</div><div class="card-value">{format_price(cash)}</div></div>
            <div class="card-stat"><div class="card-label">總資產估算</div><div class="card-value">{format_price(total_assets)}</div></div>
            <div class="card-stat"><div class="card-label">股票資產比例</div><div class="card-value">{format_ratio(stock_ratio)}</div></div>
            <div class="card-stat"><div class="card-label">目標上限</div><div class="card-value">{format_ratio(target_ratio)}</div></div>
            <div class="card-stat"><div class="card-label">狀態</div><div class="card-value">{status}</div></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_final_action_card(decision, portfolio: dict[str, object]) -> None:
    """Render final decision card using native Streamlit widgets.

    This intentionally avoids inner HTML for the decision content because some
    Streamlit/Markdown states can display nested HTML as raw text. Native widgets
    are safer for the trading dashboard and mobile viewing.
    """
    action_label = getattr(decision, "action_label", "觀察") or "觀察"
    suggested_lots = int(getattr(decision, "suggested_buy_lots", 0) or 0)
    chase_today = getattr(decision, "chase_today", "否") or "否"
    suggested_bid = getattr(decision, "suggested_bid", None)
    suggested_bid_text = "不掛單" if suggested_lots <= 0 else format_price(suggested_bid)

    after_avg = getattr(decision, "after_buy_average_cost", None)
    after_cash = getattr(decision, "after_buy_remaining_cash", None)
    after_ratio = getattr(decision, "after_buy_stock_ratio", None)

    if suggested_lots <= 0:
        avg_cost_text = f"未加碼，平均成本維持：{format_price(after_avg)}"
        cash_text = f"未加碼，剩餘現金維持：{format_price(after_cash)}"
        ratio_text = f"未加碼，股票資產比例維持：{format_ratio(after_ratio)}"
    else:
        avg_cost_text = f"買完後平均成本：{format_price(after_avg)}"
        cash_text = f"買完後剩餘現金：{format_price(after_cash)}"
        ratio_text = f"買完後股票資產比例：{format_ratio(after_ratio)}"

    over_limit = "是" if bool(getattr(decision, "over_position_limit_after_buy", False)) else "否"
    next_action = getattr(decision, "next_action", "先觀察，不追價。") or "先觀察，不追價。"
    primary_reasons = list(getattr(decision, "primary_reasons", []) or [])[:3]
    if not primary_reasons:
        primary_reasons = ["目前訊號偏中性，先以風控與價格位置為主。"]

    st.subheader("AI最終行動建議")

    # Use native status boxes so the content will never be rendered as raw <div> text.
    summary = f"今日建議：{action_label}｜建議張數：{suggested_lots} 張｜建議掛單價：{suggested_bid_text}｜是否追價：{chase_today}"
    if action_label in ("積極加碼", "分批加碼", "可加碼", "可分批加碼"):
        st.success(summary)
    elif action_label in ("觀察", "試單", "小量試單", "持有觀察"):
        st.warning(summary)
    else:
        st.error(summary)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("今日建議", action_label)
    col2.metric("建議張數", f"{suggested_lots} 張")
    col3.metric("建議掛單價", suggested_bid_text)
    col4.metric("是否追價", chase_today)

    st.markdown("**主要原因**")
    for reason in primary_reasons:
        st.write(f"- {reason}")

    st.markdown("**下一步操作**")
    st.write(next_action)

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.write(avg_cost_text)
        st.write(cash_text)
    with detail_cols[1]:
        st.write(ratio_text)
        st.write(f"是否超過持倉上限：{over_limit}")

def render_score_cards(module_scores: dict[str, int], total_score: int) -> None:
    st.subheader("五大模組分數")
    cols = st.columns(5)
    for col, (name, score) in zip(cols, module_scores.items()):
        with col:
            st.metric(name, f"{score}/100")
    st.metric("AI總分", f"{total_score}/100")


def render_risk_detail(decision) -> None:
    risk_width = max(0, min(100, int(getattr(decision, "risk_score", 0) or 0)))
    st.subheader("進場機率與風險")
    cols = st.columns(3)
    with cols[0]:
        st.metric("AI總分", f"{getattr(decision, 'total_score', 0)}/100")
    with cols[1]:
        st.metric("進場機率", f"{getattr(decision, 'entry_probability', 0)}/100", getattr(decision, "entry_probability_text", ""))
    with cols[2]:
        st.metric("部位模式", getattr(decision, "position_mode_label", ""))
    st.markdown(
        f"""
        <div style="margin:8px 0 18px;">
            <div style="display:flex;justify-content:space-between;color:#e5e7eb;">
                <span>風險條：{getattr(decision, "risk_bar_label", "風險")}</span>
                <span>{getattr(decision, "risk_score", 0)}/100</span>
            </div>
            <div style="height:10px;background:rgba(255,255,255,.25);border-radius:999px;overflow:hidden;margin-top:6px;">
                <div style="height:10px;width:{risk_width}%;background:#ef4444;border-radius:999px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_volume_card(volume: dict[str, object]) -> None:
    st.subheader("成交量判讀")
    cols = st.columns(3)
    with cols[0]:
        st.metric("量能訊號", str(volume.get("volume_signal", "量能資料不足")))
    with cols[1]:
        ratio = volume.get("volume_ratio")
        st.metric("最新量 / 20日均量", "無資料" if ratio is None else f"{float(ratio):.2f}x")
    with cols[2]:
        avg20 = volume.get("avg20_volume")
        st.metric("20日平均成交量", "無資料" if avg20 is None else f"{float(avg20):,.0f}")


def render_market_factor_card(market: dict[str, object]) -> None:
    st.subheader("市場背景因子")
    if market.get("missing"):
        st.caption("部分背景市場資料暫時無法取得，已略過該因子。")
    summary = market.get("summary", {})
    cols = st.columns(5)
    for col, (name, value) in zip(cols, summary.items()):
        with col:
            st.metric(name, str(value))


def render_portfolio_table(portfolio: dict[str, object]) -> None:
    st.subheader("持倉試算表")
    cols = st.columns(4)
    with cols[0]:
        st.metric("目前持倉市值", format_price(float(portfolio.get("market_value", 0.0) or 0.0)))
    with cols[1]:
        st.metric(
            "未實現損益",
            format_price(float(portfolio.get("unrealized_pnl", 0.0) or 0.0)),
            format_pct(float(portfolio.get("unrealized_pnl_pct", 0.0) or 0.0)),
        )
    with cols[2]:
        st.metric("目前股票資產比例", format_ratio(float(portfolio.get("current_stock_ratio", 0.0) or 0.0)))
    with cols[3]:
        st.metric("最大可買張數", f"{int(portfolio.get('max_buy_lots', 0) or 0)} 張")

    table = portfolio.get("scenario_table", pd.DataFrame())
    if isinstance(table, pd.DataFrame) and not table.empty:
        display = table.copy()
        for column in ("加碼後平均成本", "加碼後剩餘現金"):
            if column in display:
                display[column] = display[column].map(lambda value: f"NT${value:,.0f}")
        if "加碼後股票資產比例" in display:
            display["加碼後股票資產比例"] = display["加碼後股票資產比例"].map(lambda value: f"{value:.2f}%")
        st.dataframe(display, hide_index=True, use_container_width=True)


def render_price_trend_chart(data: pd.DataFrame, title: str) -> None:
    if data is None or data.empty or "Close" not in data:
        st.warning("價格資料不足，暫時不顯示價格趨勢圖。")
        return

    close = pd.to_numeric(data["Close"], errors="coerce").dropna()
    if close.empty:
        st.warning("價格資料不足，暫時不顯示價格趨勢圖。")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="收盤價", line={"width": 2.5}))
    fig.add_trace(go.Scatter(x=close.index, y=moving_average(close, 20), mode="lines", name="MA20", line={"width": 1.5}))
    fig.add_trace(go.Scatter(x=close.index, y=moving_average(close, 60), mode="lines", name="MA60", line={"width": 1.5}))
    fig.update_layout(
        title=title,
        height=380,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        hovermode="x unified",
        template="plotly_dark",
        yaxis_title="價格（新台幣）",
    )
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
