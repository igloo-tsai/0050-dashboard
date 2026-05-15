from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from storage import sort_inventory_records
from technical_analysis import moving_average


def format_price(value: float | None) -> str:
    if value is None:
        return "-"
    try:
        return f"NT${float(value):,.2f}"
    except Exception:
        return "-"


def format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):+.2f}%"
    except Exception:
        return "-"


def format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "-"


def render_stock_candidate_selector(candidates: pd.DataFrame, key: str) -> dict[str, str] | None:
    if candidates is None or candidates.empty:
        st.warning("找不到股票，請改用股票代碼、中文名稱或部分關鍵字搜尋。")
        return None

    if len(candidates) == 1:
        row = candidates.iloc[0]
        selected = {
            "ticker_code": str(row.get("ticker_code", "") or ""),
            "stock_name": str(row.get("stock_name", "") or ""),
            "market": str(row.get("market", "") or ""),
        }
        st.success(f"已自動選定：{selected['ticker_code']}｜{selected['stock_name']}｜{selected['market']}")
        return selected

    options = ["請選擇股票"]
    rows: dict[str, dict[str, str]] = {}
    for _, row in candidates.iterrows():
        ticker_code = str(row.get("ticker_code", "") or "")
        stock_name = str(row.get("stock_name", "") or "")
        market = str(row.get("market", "") or "")
        label = f"{ticker_code}｜{stock_name}｜{market}"
        options.append(label)
        rows[label] = {
            "ticker_code": ticker_code,
            "stock_name": stock_name,
            "market": market,
        }

    selected = st.selectbox("搜尋結果候選清單", options=options, key=key)
    if selected == "請選擇股票":
        st.info("請先從候選清單選定股票，再進行 AI 分析。")
        return None
    return rows.get(selected)


def inject_mobile_css() -> None:
    """Small, safe CSS only. No HTML cards are used for rendering content."""
    st.markdown(
        """
        <style>
        * { box-sizing: border-box; }
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        h1 { line-height: 1.15 !important; }
        h2, h3 { line-height: 1.25 !important; }
        h2:has(span), h3:has(span) { font-weight: 900 !important; }
        [data-testid="stMetric"] {
            border: 1px solid rgba(255,255,255,.16);
            border-radius: 12px;
            padding: .8rem .9rem;
            background: rgba(255,255,255,.045);
        }
        [data-testid="stMetric"] label,
        [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stMetric"] [data-testid="stMetricDelta"] {
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            line-height: 1.25 !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.22rem !important;
            font-weight: 800 !important;
        }
        div[data-testid="stAlert"] { border-radius: 12px; }
        @media (max-width: 768px) {
            .block-container { padding: .75rem .7rem 1.6rem; max-width: 100%; }
            h1 { font-size: 1.45rem !important; }
            h2 { font-size: 1.2rem !important; }
            h3 { font-size: 1.08rem !important; }
            [data-testid="stMetric"] { padding: 1rem .9rem; min-height: 92px; }
            [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.18rem !important; }
            div[data-testid="stAlert"] { padding: 1.05rem !important; font-size: 1.05rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _decision_tone(action_label: str) -> str:
    if action_label in ("積極加碼", "分批加碼", "可加碼", "可分批加碼"):
        return "green"
    if action_label in ("觀察", "試單", "小量試單", "持有觀察"):
        return "yellow"
    return "red"


def _short_action_label(action_label: str) -> str:
    mapping = {
        "積極加碼": "可加碼",
        "分批加碼": "分批",
        "試單": "試單",
        "觀察": "觀察",
        "暫緩進場": "暫緩",
    }
    return mapping.get(action_label, action_label or "觀察")


def _alert_decision(tone: str, line: str) -> None:
    if tone == "green":
        st.success(line)
    elif tone == "yellow":
        st.warning(line)
    else:
        st.error(line)


def _trader_status(decision) -> tuple[str, str]:
    today_status = getattr(decision, "today_status", "") or ""
    if today_status == "禁止加碼":
        return "red", "🚫 暫緩"
    if today_status == "等待回檔":
        return "yellow", "⏳ 觀察"
    return "green", "📈 買進"


def _fallback_strategy_levels(
    observation_price: float | None,
    reasonable_price: float | None,
    conservative_price: float | None,
    trade_plan: list[dict[str, object]],
) -> list[dict[str, object]]:
    labels = [
        ("observation", "第一批（觀察）", observation_price),
        ("fair", "第二批（合理）", reasonable_price),
        ("conservative", "第三批（保守）", conservative_price),
    ]
    levels: list[dict[str, object]] = []
    for index, (level_type, title, price) in enumerate(labels):
        plan_row = trade_plan[index] if index < len(trade_plan) else {}
        budget = float(plan_row.get("batch_budget", plan_row.get("budget", 0.0)) or 0.0)
        lots = int(plan_row.get("lots", 0) or 0)
        shares = int(plan_row.get("shares", 0) or 0)
        amount = float(plan_row.get("amount", 0.0) or 0.0)
        levels.append(
            {
                "type": level_type,
                "title": title,
                "price": float(price or plan_row.get("price", 0.0) or 0.0),
                "budget": budget,
                "lots": lots,
                "shares": shares,
                "amount": amount,
                "ratio": float(plan_row.get("batch_ratio", 0.0) or 0.0),
            }
        )
    return levels


def render_trader_decision_card(decision, portfolio: dict[str, object] | None = None) -> None:
    tone, headline = _trader_status(decision)
    immediate_plan = getattr(decision, "immediate_plan", "") or "不立即掛單"
    lots = int(getattr(decision, "immediate_lots", getattr(decision, "suggested_buy_lots", 0)) or 0)
    shares = int(getattr(decision, "immediate_shares", getattr(decision, "suggested_buy_shares", 0)) or 0)
    today_status = getattr(decision, "today_status", "等待回檔") or "等待回檔"
    next_action = getattr(decision, "next_action", "現在不買，等待合理價。") or "現在不買，等待合理價。"
    observation_price = getattr(decision, "observation_price", None)
    reasonable_price = getattr(decision, "reasonable_price", getattr(decision, "suggested_bid", None))
    conservative_price = getattr(decision, "conservative_price", getattr(decision, "conservative_bid", None))
    entry_probability = int(getattr(decision, "entry_probability", 0) or 0)
    position_mode_label = getattr(decision, "position_mode_label", "") or ""
    trade_plan = list(getattr(decision, "trade_plan", []) or [])
    trader_plan_v2 = getattr(decision, "trader_plan_v2", {}) or {}
    v2_levels = list(trader_plan_v2.get("levels", []) or []) if isinstance(trader_plan_v2, dict) else []

    risk_level = getattr(decision, "risk_bar_label", "") or getattr(decision, "risk_level", "風險未定")
    immediate_price = getattr(decision, "immediate_price", None)
    suggested_price = immediate_price if immediate_price is not None else None

    st.subheader("🔥 AI即時決策")
    _alert_decision(tone, headline)

    status_cols = st.columns(5)
    status_cols[0].metric("今日立即建議", immediate_plan)
    status_cols[1].metric("立即張數", f"{lots} 張 {shares} 股")
    status_cols[2].metric("立即價格", format_price(suggested_price) if suggested_price is not None else "不立即掛單")
    status_cols[3].metric("🎯 進場機率", f"{entry_probability}%")
    status_cols[4].metric("⚠️ 風險等級", risk_level)

    st.subheader("今日操作")
    if tone == "green":
        st.success(f"📈 {next_action}")
    elif tone == "yellow":
        st.warning(f"⏳ {next_action}")
    else:
        st.error(f"🚫 {next_action}")

    st.caption("今日建議代表是否現在立即執行；分批策略代表價格到達指定區間時才執行。")
    st.subheader("📊 分批策略（價格到位才執行）")
    strategy_levels = v2_levels or _fallback_strategy_levels(observation_price, reasonable_price, conservative_price, trade_plan)
    titles = {
        "observation": "第一批（觀察）",
        "fair": "第二批（合理）",
        "conservative": "第三批（保守）",
    }
    level_cols = st.columns(3)
    for col, row in zip(level_cols, strategy_levels):
        level_type = str(row.get("type", ""))
        with col:
            st.markdown(f"##### {titles.get(level_type, str(row.get('title', level_type)))}")
            st.metric("價格", format_price(float(row.get("price", 0.0) or 0.0)))
            st.metric("預算", format_price(float(row.get("budget", 0.0) or 0.0)))
            st.metric("張數", f"{int(row.get('lots', 0) or 0)} 張")
            st.metric("零股", f"{int(row.get('shares', 0) or 0)} 股")
            st.metric("投入金額", format_price(float(row.get("amount", 0.0) or 0.0)))

    render_buy_impact(decision)


def render_buy_impact(decision) -> None:
    st.subheader("買完後影響")
    trade_plan = list(getattr(decision, "trade_plan", []) or [])
    if trade_plan:
        last_row = trade_plan[-1]
        total_amount = last_row.get("cumulative_amount")
        total_lots = int(last_row.get("cumulative_lots", 0) or 0)
        total_shares = int(last_row.get("cumulative_shares", 0) or 0)
        average_cost = last_row.get("after_batch_average_cost")
        remaining_cash = last_row.get("after_batch_cash")
        stock_ratio = last_row.get("after_batch_stock_ratio")
        over_limit = bool(last_row.get("over_limit_after_batch", False))
    else:
        total_amount = None
        total_lots = 0
        total_shares = 0
        average_cost = getattr(decision, "after_buy_average_cost", None)
        remaining_cash = getattr(decision, "after_buy_remaining_cash", None)
        stock_ratio = getattr(decision, "after_buy_stock_ratio", None)
        over_limit = bool(getattr(decision, "over_position_limit_after_buy", False))

    cols = st.columns(3)
    cols[0].metric("合計投入", format_price(total_amount))
    cols[1].metric("合計買入", f"{total_lots} 張 {total_shares} 股")
    cols[2].metric("是否超過上限", "是" if over_limit else "否")
    cols = st.columns(3)
    cols[0].metric("最終平均成本", format_price(average_cost))
    cols[1].metric("最終剩餘現金", format_price(remaining_cash))
    cols[2].metric("最終股票比例", format_ratio(stock_ratio))

def render_system_validation(validation_result: dict[str, object]) -> None:
    st.subheader("系統穩定性檢查")
    errors = list(validation_result.get("errors", []) or [])
    warnings = list(validation_result.get("warnings", []) or [])
    fixed = list(validation_result.get("fixed", []) or [])
    if not errors and not warnings:
        st.success("✅ 系統檢查通過")
    elif errors:
        st.error(f"? ?? {len(errors)} ???")
    else:
        st.warning(f"?? ?? {len(warnings)} ???")

    for error in errors:
        st.write(f"- {error}")
    for warning in warnings:
        st.write(f"- {warning}")
    if fixed:
        st.info("已自動修正：")
        for item in fixed:
            st.write(f"- {item}")


def render_valuation_quality_card(result: dict[str, object] | None) -> None:
    st.subheader("估值與標的品質")
    if not result or not isinstance(result, dict):
        st.info("估值與標的品質資料不足，暫以中性分數處理。")
        return

    cols = st.columns(4)
    cols[0].metric("估值分數", f"{int(result.get('valuation_score', 50) or 50)}/100")
    cols[1].metric("品質分數", f"{int(result.get('quality_score', 50) or 50)}/100")
    cols[2].metric("綜合分數", f"{int(result.get('final_score', 50) or 50)}/100")
    cols[3].metric("資料可信度", f"{int(result.get('data_quality_score', 0) or 0)}/100")

    cols = st.columns(3)
    cols[0].metric("估值判斷", str(result.get("valuation_label", "資料不足") or "資料不足"))
    cols[1].metric("品質判斷", str(result.get("quality_label", "資料不足") or "資料不足"))
    cols[2].metric("投資屬性", str(result.get("investability_label", "資料不足") or "資料不足"))

    reasons = [str(item) for item in list(result.get("reasons", []) or [])[:5]]
    if reasons:
        st.caption("主要原因")
        for reason in reasons:
            st.write(f"- {reason}")

    missing_fields = [str(item) for item in list(result.get("missing_fields", []) or [])]
    if missing_fields:
        st.caption("缺失資料欄位")
        st.write("、".join(missing_fields[:12]))

    for warning in list(result.get("warnings", []) or [])[:5]:
        st.warning(str(warning))


def render_today_action(decision) -> None:
    next_action = getattr(decision, "next_action", "先觀察，不追價。") or "先觀察，不追價。"
    st.subheader("今日動作")
    st.markdown(f"### 👉 {next_action}")


def render_key_reasons(decision) -> None:
    st.subheader("關鍵原因")
    reasons = list(getattr(decision, "primary_reasons", []) or [])[:3]
    if not reasons:
        reasons = ["目前訊號尚未完全確認，先以價格與部位風控為主。"]
    for idx, reason in enumerate(reasons, 1):
        st.write(f"{idx}. {reason}")


def _inventory_consistency_warnings(records: list[dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    for idx, record in enumerate(records, 1):
        side = str(record.get("side", "") or "")
        price = float(record.get("price", 0.0) or 0.0)
        shares = int(record.get("shares", 0) or 0)
        gross_amount = float(record.get("gross_amount", 0.0) or 0.0)
        fee = float(record.get("fee", 0.0) or 0.0)
        tax = float(record.get("tax", 0.0) or 0.0)
        transaction_cost = float(record.get("transaction_cost", 0.0) or 0.0)
        net_amount = float(record.get("net_amount", 0.0) or 0.0)
        expected_gross = round(price * shares, 0)
        expected_cost = round(fee + tax, 0)
        expected_net = expected_gross + fee + tax if side == "買入" else expected_gross - fee - tax

        if abs(gross_amount - expected_gross) > 1:
            warnings.append(f"第 {idx} 筆成交金額不一致，應為 {format_price(expected_gross)}。")
        if abs(transaction_cost - expected_cost) > 1:
            warnings.append(f"第 {idx} 筆單筆交易成本不一致，應為 {format_price(expected_cost)}。")
        if abs(net_amount - round(expected_net, 0)) > 1:
            warnings.append(f"第 {idx} 筆淨收付金額不一致，請檢查手續費或交易稅。")
    return warnings


def render_inventory_records(records: list[dict[str, object]], current_price: float | None = None) -> None:
    records = sort_inventory_records(list(records))
    st.caption(f"目前已載入 {len(records)} 筆庫存")
    st.caption("已依日期由舊到新排序")
    if not records:
        st.info("尚未建立逐筆庫存。")
        return
    warnings = _inventory_consistency_warnings(records)
    if warnings:
        st.warning("庫存自我檢查發現資料不一致：")
        for warning in warnings:
            st.write(f"- {warning}")
    display = pd.DataFrame(records)
    if current_price is not None and current_price > 0 and not display.empty:
        display["current_price"] = float(current_price)
        display["unrealized_pnl"] = display.apply(
            lambda row: (float(row.get("shares", 0) or 0) * float(current_price) - float(row.get("net_amount", 0) or 0))
            if str(row.get("side", "") or "") == "買入"
            else 0.0,
            axis=1,
        )
        display["unrealized_pnl_pct"] = display.apply(
            lambda row: float(row["unrealized_pnl"]) / float(row.get("net_amount", 0) or 1) * 100
            if str(row.get("side", "") or "") == "買入" and float(row.get("net_amount", 0) or 0) > 0
            else 0.0,
            axis=1,
        )
    ordered_columns = [
        "ticker",
        "name",
        "date",
        "side",
        "price",
        "current_price",
        "lots",
        "odd_shares",
        "shares",
        "gross_amount",
        "fee",
        "fee_source",
        "tax",
        "transaction_cost",
        "net_amount",
        "unrealized_pnl",
        "unrealized_pnl_pct",
        "note",
    ]
    display = display[[column for column in ordered_columns if column in display.columns]]
    column_map = {
        "date": "日期",
        "ticker": "代碼",
        "name": "名稱",
        "side": "類型",
        "price": "買入價格",
        "current_price": "現價",
        "lots": "張數",
        "odd_shares": "零股",
        "shares": "股數",
        "gross_amount": "成交金額",
        "fee": "手續費",
        "fee_source": "手續費來源",
        "tax": "交易稅",
        "transaction_cost": "單筆交易成本",
        "net_amount": "淨收付金額",
        "unrealized_pnl": "未實現損益",
        "unrealized_pnl_pct": "損益 %",
        "note": "備註",
    }
    display = display.rename(columns=column_map)
    if "損益 %" in display.columns:
        styled = display.style.map(
            lambda value: "color: #22c55e" if float(value or 0) >= 0 else "color: #ef4444",
            subset=["損益 %"],
        )
        st.dataframe(styled, hide_index=True, use_container_width=True)
    else:
        st.dataframe(display, hide_index=True, use_container_width=True)


def render_broker_grade_profit_section(summary: dict[str, object]) -> None:
    st.subheader("券商級損益")
    if not summary:
        st.info("尚無損益資料。")
        return

    cols = st.columns(5)
    cols[0].metric("總股數", f"{int(summary.get('total_shares', 0) or 0):,} 股")
    cols[1].metric("平均成本", format_price(float(summary.get("average_cost", 0.0) or 0.0)))
    cols[2].metric("目前市值", format_price(float(summary.get("market_value", 0.0) or 0.0)))
    cols[3].metric("未實現損益", format_price(float(summary.get("unrealized_pnl", 0.0) or 0.0)))
    cols[4].metric("未實現報酬率", format_pct(float(summary.get("unrealized_pnl_pct", 0.0) or 0.0)))

    cols = st.columns(5)
    cols[0].metric("已實現損益", format_price(float(summary.get("realized_pnl", 0.0) or 0.0)))
    cols[1].metric("股利收入", format_price(float(summary.get("dividend_income", 0.0) or 0.0)))
    cols[2].metric("總報酬", format_price(float(summary.get("total_return", 0.0) or 0.0)))
    cols[3].metric("手續費累計", format_price(float(summary.get("total_fees", 0.0) or 0.0)))
    cols[4].metric("稅金累計", format_price(float(summary.get("total_tax", 0.0) or 0.0)))

    for warning in list(summary.get("broker_summary", {}).get("warnings", []) or []):
        st.warning(str(warning))
    for error in list(summary.get("broker_summary", {}).get("errors", []) or []):
        st.error(str(error))


def render_open_lots_table(open_lots: list[dict[str, object]]) -> None:
    st.subheader("逐筆庫存損益")
    if not open_lots:
        st.info("目前沒有未平倉庫存。")
        return
    rows = []
    for lot in open_lots:
        rows.append(
            {
                "買進日": lot.get("buy_date"),
                "成交價": round(float(lot.get("buy_price", 0.0) or 0.0), 2),
                "剩餘股數": int(lot.get("shares_remaining", 0) or 0),
                "成本": round(float(lot.get("remaining_cost_basis", 0.0) or 0.0), 0),
                "現值": round(float(lot.get("market_value", 0.0) or 0.0), 0),
                "未實現損益": round(float(lot.get("unrealized_pnl", 0.0) or 0.0), 0),
                "報酬率": round(float(lot.get("unrealized_pnl_pct", 0.0) or 0.0), 2),
            }
        )
    display = pd.DataFrame(rows)
    styled = display.style.map(
        lambda value: "color: #22c55e" if float(value or 0) >= 0 else "color: #ef4444",
        subset=["未實現損益", "報酬率"],
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)


def render_realized_pnl_table(realized_details: list[dict[str, object]]) -> None:
    st.subheader("已實現損益")
    if not realized_details:
        st.info("目前沒有已實現損益。")
        return
    rows = []
    for detail in realized_details:
        rows.append(
            {
                "賣出日": detail.get("sell_date"),
                "買入日": detail.get("matched_buy_date"),
                "賣價": round(float(detail.get("sell_price", 0.0) or 0.0), 2),
                "買價": round(float(detail.get("matched_buy_price", 0.0) or 0.0), 2),
                "股數": int(detail.get("matched_shares", 0) or 0),
                "成本": round(float(detail.get("cost_basis", 0.0) or 0.0), 0),
                "賣出收入": round(float(detail.get("sell_proceeds_allocated", 0.0) or 0.0), 0),
                "已實現損益": round(float(detail.get("realized_pnl", 0.0) or 0.0), 0),
                "報酬率": round(float(detail.get("realized_pnl_pct", 0.0) or 0.0), 2),
                "持有天數": int(detail.get("holding_days", 0) or 0),
            }
        )
    display = pd.DataFrame(rows)
    styled = display.style.map(
        lambda value: "color: #22c55e" if float(value or 0) >= 0 else "color: #ef4444",
        subset=["已實現損益", "報酬率"],
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)


def render_transaction_performance(records: list[dict[str, object]]) -> None:
    st.subheader("交易成本與績效")
    if not records:
        st.info("尚無交易紀錄可計算績效。")
        return

    total_fee = sum(float(record.get("fee", 0.0) or 0.0) for record in records)
    total_tax = sum(float(record.get("tax", 0.0) or 0.0) for record in records)
    total_buy_net = sum(float(record.get("net_amount", 0.0) or 0.0) for record in records if record.get("side") == "買入")
    total_sell_net = sum(float(record.get("net_amount", 0.0) or 0.0) for record in records if record.get("side") == "賣出")
    profit = total_sell_net - total_buy_net
    true_return = profit / total_buy_net * 100 if total_buy_net > 0 else 0.0
    total_cost = total_fee + total_tax

    cols = st.columns(4)
    cols[0].metric("交易成本合計", format_price(total_cost))
    cols[1].metric("累積手續費", format_price(total_fee))
    cols[2].metric("累積稅金", format_price(total_tax))
    cols[3].metric("真實報酬率（含費用）", format_pct(true_return))

    detail = pd.DataFrame(
        [
            {"項目": "買入淨額", "金額": format_price(total_buy_net)},
            {"項目": "賣出淨額", "金額": format_price(total_sell_net)},
            {"項目": "已實現損益（賣出淨額 - 買入淨額）", "金額": format_price(profit)},
        ]
    )
    st.dataframe(detail, hide_index=True, use_container_width=True)


def _calculate_trade_log_stats(records: list[dict[str, object]], portfolio: dict[str, object]) -> dict[str, float]:
    total_buy_net = 0.0
    realized_pnl = 0.0
    open_shares = 0
    open_cost = 0.0
    closed_trades = 0
    winning_trades = 0

    for record in records:
        action = str(record.get("action", "") or "")
        shares = int(record.get("shares", 0) or 0)
        net_amount = float(record.get("net_amount", record.get("amount", 0.0)) or 0.0)
        if shares <= 0 or net_amount <= 0:
            continue

        if action == "買入":
            total_buy_net += net_amount
            open_shares += shares
            open_cost += net_amount
            continue

        if action != "賣出":
            continue

        if open_shares > 0 and open_cost > 0:
            matched_shares = min(shares, open_shares)
            average_cost = open_cost / open_shares
            matched_cost = average_cost * matched_shares
            trade_profit = net_amount - matched_cost
            realized_pnl += trade_profit
            closed_trades += 1
            if trade_profit > 0:
                winning_trades += 1
            open_shares -= matched_shares
            open_cost = max(0.0, open_cost - matched_cost)
        else:
            realized_pnl += net_amount

    unrealized_pnl = float(portfolio.get("unrealized_pnl", 0.0) or 0.0)
    total_profit = realized_pnl + unrealized_pnl
    total_return = total_profit / total_buy_net * 100 if total_buy_net > 0 else 0.0
    win_rate = winning_trades / closed_trades * 100 if closed_trades > 0 else 0.0
    return {
        "total_return": total_return,
        "unrealized_pnl": unrealized_pnl,
        "realized_pnl": realized_pnl,
        "win_rate": win_rate,
    }


def render_trade_log_performance(records: list[dict[str, object]], portfolio: dict[str, object]) -> None:
    st.subheader("📊 交易績效")
    if not records:
        st.info("尚無 trade_log 交易紀錄。")
        return

    stats = _calculate_trade_log_stats(records, portfolio)
    cols = st.columns(4)
    cols[0].metric("總報酬率", format_pct(stats["total_return"]))
    cols[1].metric("未實現損益", format_price(stats["unrealized_pnl"]))
    cols[2].metric("已實現損益", format_price(stats["realized_pnl"]))
    cols[3].metric("勝率", f"{stats['win_rate']:.2f}%")


def render_trade_log_records(records: list[dict[str, object]]) -> None:
    st.subheader("交易紀錄")
    st.caption(f"目前已記錄 {len(records)} 筆交易")
    if not records:
        st.info("尚無交易紀錄。")
        return

    display = pd.DataFrame(records)
    column_map = {
        "date": "日期",
        "action": "買賣",
        "price": "成交價",
        "shares": "股數",
        "amount": "成交金額",
        "fee": "手續費",
        "tax": "交易稅",
        "net_amount": "淨收付金額",
    }
    display = display.rename(columns=column_map)
    st.dataframe(display, hide_index=True, use_container_width=True)


def render_holding_pnl_card(portfolio: dict[str, object]) -> None:
    st.subheader("持倉損益")
    holding_lots = float(portfolio.get("holding_lots", 0.0) or 0.0)
    shares = float(portfolio.get("shares", 0.0) or 0.0)
    if holding_lots <= 0 or shares <= 0:
        st.info("尚未持有，目前無未實現損益。")
        return

    cols = st.columns(2)
    cols[0].metric("持有", f"{holding_lots:g} 張 / {shares:,.0f} 股")
    cols[1].metric("平均成本", format_price(float(portfolio.get("average_cost", 0.0) or 0.0)))

    cols = st.columns(2)
    cols[0].metric("目前市價", format_price(float(portfolio.get("current_price", 0.0) or 0.0)))
    cols[1].metric("股票比例", format_ratio(float(portfolio.get("current_stock_ratio", 0.0) or 0.0)))

    cols = st.columns(2)
    cols[0].metric("總成本", format_price(float(portfolio.get("cost_value", 0.0) or 0.0)))
    cols[1].metric("市值", format_price(float(portfolio.get("market_value", 0.0) or 0.0)))

    pnl = float(portfolio.get("unrealized_pnl", 0.0) or 0.0)
    pnl_pct = float(portfolio.get("unrealized_pnl_pct", 0.0) or 0.0)
    cols = st.columns(2)
    cols[0].metric("未實現損益", format_price(pnl))
    cols[1].metric("報酬率", format_pct(pnl_pct))
    st.metric("可用現金", format_price(float(portfolio.get("available_cash", portfolio.get("cash", 0.0)) or 0.0)))


def render_position_summary_card(portfolio: dict[str, object]) -> None:
    """Mobile-first position summary: only the information needed for action."""
    market_value = float(portfolio.get("market_value", 0.0) or 0.0)
    cash = float(portfolio.get("cash", 0.0) or 0.0)
    total_assets = float(portfolio.get("total_assets", 0.0) or 0.0)
    stock_ratio = float(portfolio.get("current_stock_ratio", 0.0) or 0.0)
    target_ratio = float(portfolio.get("max_stock_ratio", 0.0) or 0.0)
    excess = float(portfolio.get("excess_stock_ratio", 0.0) or 0.0)
    over_target = bool(portfolio.get("over_target_ratio", False))

    st.subheader("部位摘要")
    cols = st.columns(4)
    cols[0].metric("持倉市值", format_price(market_value))
    cols[1].metric("股票比例", format_ratio(stock_ratio))
    cols[2].metric("可用現金", format_price(cash))
    cols[3].metric("狀態", "超標" if over_target else "未超標")

    if over_target:
        st.error(f"股票部位已超過上限 {excess:.1f}%，今天不應再加碼。")
    else:
        st.caption(f"目標上限：{format_ratio(target_ratio)}｜總資產估算：{format_price(total_assets)}")


def render_final_action_card(decision, portfolio: dict[str, object]) -> None:
    """Backward-compatible wrapper for old call sites."""
    render_trader_decision_card(decision, portfolio)
    render_today_action(decision)
    render_key_reasons(decision)


def render_self_check(decision) -> None:
    st.subheader("AI模型一致性檢查")
    flags = list(getattr(decision, "conflict_flags", []) or [])
    if not flags:
        st.success("✅ 通過，未發現衝突")
        return
    st.warning("⚠️ 發現以下衝突：")
    for flag in flags:
        st.write(f"- {flag}")


def render_trade_record_summary(decision) -> None:
    st.subheader("本次決策紀錄摘要")
    record = getattr(decision, "trade_record", {}) or {}
    if not record:
        st.caption("目前沒有可顯示的決策紀錄。")
        return
    rows = [{"項目": key, "內容": value} for key, value in record.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_legacy_final_action_card(decision, portfolio: dict[str, object]) -> None:
    """Kept for reference; native Streamlit only; no content HTML."""
    action_label = getattr(decision, "action_label", "觀察") or "觀察"
    short_label = _short_action_label(action_label)
    suggested_lots = int(getattr(decision, "suggested_buy_lots", 0) or 0)
    chase_today = getattr(decision, "chase_today", "否") or "否"
    suggested_bid = getattr(decision, "suggested_bid", None)
    suggested_price_text = "不掛單" if suggested_lots <= 0 else format_price(suggested_bid)
    entry_probability = int(getattr(decision, "entry_probability", 0) or 0)
    risk_label = getattr(decision, "risk_bar_label", "") or getattr(decision, "risk_level", "") or "風險未定"
    next_action = getattr(decision, "next_action", "先觀察，不追價。") or "先觀察，不追價。"
    tone = _decision_tone(action_label)

    st.subheader("AI決策")
    headline = f"今日建議：{short_label}｜{suggested_lots} 張｜{suggested_price_text}"
    _alert_decision(tone, headline)

    cols = st.columns(4)
    cols[0].metric("今日建議", short_label)
    cols[1].metric("建議張數", f"{suggested_lots} 張")
    cols[2].metric("建議價格", suggested_price_text)
    cols[3].metric("是否追價", str(chase_today))

    st.subheader("今日操作")
    if suggested_lots <= 0:
        st.error(f"不要掛單。{next_action}")
    elif tone == "green":
        st.success(next_action)
    else:
        st.warning(next_action)

    st.subheader("關鍵原因")
    reasons = list(getattr(decision, "primary_reasons", []) or [])[:3]
    if not reasons:
        reasons = ["目前訊號尚未完全確認，先以價格與部位風控為主。"]
    for idx, reason in enumerate(reasons, 1):
        st.write(f"{idx}. {reason}")

    with st.expander("買入後試算", expanded=False):
        after_avg = getattr(decision, "after_buy_average_cost", None)
        after_cash = getattr(decision, "after_buy_remaining_cash", None)
        after_ratio = getattr(decision, "after_buy_stock_ratio", None)
        over_limit = bool(getattr(decision, "over_position_limit_after_buy", False))
        if suggested_lots <= 0:
            st.write(f"未加碼，平均成本維持：{format_price(after_avg)}")
            st.write(f"未加碼，剩餘現金維持：{format_price(after_cash)}")
            st.write(f"未加碼，股票比例維持：{format_ratio(after_ratio)}")
        else:
            st.write(f"買完後平均成本：{format_price(after_avg)}")
            st.write(f"買完後剩餘現金：{format_price(after_cash)}")
            st.write(f"買完後股票比例：{format_ratio(after_ratio)}")
        st.write(f"是否超過持倉上限：{'是' if over_limit else '否'}")
        st.write(f"進場機率：{entry_probability}/100")
        st.write(f"風險：{risk_label}")


def render_score_cards(module_scores: dict[str, int], total_score: int) -> None:
    st.subheader("五大模組分數")
    cols = st.columns(5)
    for col, (name, score) in zip(cols, module_scores.items()):
        with col:
            st.metric(name, f"{score}/100")
    st.metric("AI總分", f"{total_score}/100")


def render_risk_detail(decision) -> None:
    risk_score = int(getattr(decision, "risk_score", 0) or 0)
    st.subheader("進場機率與風險")
    cols = st.columns(3)
    cols[0].metric("AI總分", f"{getattr(decision, 'total_score', 0)}/100")
    cols[1].metric("進場機率", f"{getattr(decision, 'entry_probability', 0)}/100", getattr(decision, "entry_probability_text", ""))
    cols[2].metric("部位模式", getattr(decision, "position_mode_label", ""))
    st.progress(max(0, min(100, risk_score)) / 100, text=f"風險條：{getattr(decision, 'risk_bar_label', '')} {risk_score}/100")


def render_volume_card(volume: dict[str, object]) -> None:
    st.subheader("成交量判讀")
    cols = st.columns(3)
    cols[0].metric("量能訊號", str(volume.get("volume_signal", "量能資料不足")))
    ratio = volume.get("volume_ratio")
    cols[1].metric("最新量 / 20日均量", "-" if ratio is None else f"{float(ratio):.2f}x")
    cols[2].metric("20日平均成交量", "-" if volume.get("avg20_volume") is None else f"{float(volume['avg20_volume']):,.0f}")


def render_market_factor_card(market: dict[str, object]) -> None:
    st.subheader("市場背景因子")
    if market.get("missing"):
        st.caption("部分背景市場資料暫時無法取得，已略過該因子。")
    summary = market.get("summary", {})
    if not isinstance(summary, dict) or not summary:
        st.caption("市場背景資料不足。")
        return
    cols = st.columns(min(5, len(summary)))
    for col, (name, value) in zip(cols, summary.items()):
        col.metric(name, str(value))


def render_portfolio_table(portfolio: dict[str, object]) -> None:
    st.subheader("持倉試算表")
    cols = st.columns(4)
    cols[0].metric("目前持倉市值", format_price(float(portfolio.get("market_value", 0.0) or 0.0)))
    cols[1].metric(
        "未實現損益",
        format_price(float(portfolio.get("unrealized_pnl", 0.0) or 0.0)),
        format_pct(float(portfolio.get("unrealized_pnl_pct", 0.0) or 0.0)),
    )
    cols[2].metric("目前股票資產比例", format_ratio(float(portfolio.get("current_stock_ratio", 0.0) or 0.0)))
    cols[3].metric("最大可買張數", f"{int(portfolio.get('max_buy_lots', 0) or 0)} 張")

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
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty:
        st.warning("價格資料不足，暫時不顯示價格趨勢圖。")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="收盤價", line={"color": "#60a5fa", "width": 2.5}))
    fig.add_trace(go.Scatter(x=close.index, y=moving_average(close, 20), mode="lines", name="MA20", line={"color": "#f59e0b", "width": 1.5}))
    fig.add_trace(go.Scatter(x=close.index, y=moving_average(close, 60), mode="lines", name="MA60", line={"color": "#10b981", "width": 1.5}))
    fig.update_layout(
        title=title,
        height=360,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        hovermode="x unified",
        template="plotly_dark",
        yaxis_title="價格（新台幣）",
    )
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
