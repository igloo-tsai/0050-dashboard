from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from data_service import fetch_taiwan_stock
from decision_engine import make_decision
from market_factors import get_market_background
from portfolio_engine import calculate_portfolio
from storage import (
    FEE_RATE,
    calculate_inventory_summary as build_inventory_summary,
    consume_storage_warnings,
    get_inventory_by_ticker,
    load_inventory,
    load_trade_log,
    log_trade,
    save_inventory,
    save_trade_log,
    summarize_inventory_by_ticker,
)
from technical_analysis import build_technical_snapshot
from ui_components import (
    format_pct,
    format_price,
    inject_mobile_css,
    render_holding_pnl_card,
    render_inventory_records,
    render_key_reasons,
    render_market_factor_card,
    render_position_summary_card,
    render_portfolio_table,
    render_price_trend_chart,
    render_risk_detail,
    render_score_cards,
    render_self_check,
    render_system_validation,
    render_trader_decision_card,
    render_trade_record_summary,
    render_trade_log_performance,
    render_trade_log_records,
    render_transaction_performance,
    render_volume_card,
)
from validator import run_system_validation
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


def init_inventory_state() -> None:
    if "inventory_records" not in st.session_state:
        st.session_state.inventory_records = load_inventory()
        st.session_state.inventory_saved = True


def init_trade_log_state() -> None:
    if "trade_log_records" not in st.session_state:
        st.session_state.trade_log_records = load_trade_log()
        st.session_state.trade_log_status_message = "已載入交易紀錄。"


def set_inventory_records(records: list[dict[str, object]], saved_message: str = "庫存已自動儲存。") -> None:
    save_inventory(records)
    st.session_state.inventory_records = load_inventory()
    st.session_state.inventory_saved = True
    st.session_state.inventory_status_message = saved_message


def reload_trade_log(message: str = "交易紀錄已更新。") -> None:
    st.session_state.trade_log_records = load_trade_log()
    st.session_state.trade_log_status_message = message


def render_trade_log_section(prefix: str, decision, portfolio: dict[str, object], current_price: float) -> None:
    records = list(st.session_state.get("trade_log_records", []))
    render_trade_log_performance(records, portfolio)

    st.subheader("交易紀錄")
    st.caption(f"目前已載入 {len(records)} 筆交易紀錄")
    if st.session_state.get("trade_log_status_message"):
        st.caption(str(st.session_state.trade_log_status_message))

    st.subheader("AI建議買入紀錄")
    ai_lots = int(getattr(decision, "suggested_buy_lots", 0) or 0)
    ai_shares = int(getattr(decision, "suggested_buy_shares", 0) or 0)
    ai_total_shares = ai_lots * 1000 + ai_shares
    ai_price = float(
        getattr(decision, "reasonable_price", None)
        or getattr(decision, "suggested_bid", None)
        or current_price
        or 0.0
    )
    st.caption(f"AI建議：{ai_lots} 張 {ai_shares} 股，價格 {format_price(ai_price)}")
    if st.button("將 AI 建議買入寫入交易紀錄", key=f"{prefix}_log_ai_buy"):
        if ai_total_shares <= 0 or ai_price <= 0:
            st.warning("目前 AI 沒有可寫入的買入股數。")
        else:
            log_trade("買入", ai_price, ai_total_shares, ai_price * ai_total_shares)
            reload_trade_log("已將 AI 建議寫入交易紀錄。")
            st.success("已寫入 trade_log.json")
            st.rerun()

    st.divider()
    st.subheader("手動新增交易")
    cols = st.columns(4)
    with cols[0]:
        action = st.selectbox("交易類型", options=["買入", "賣出"], key=f"{prefix}_trade_action")
    with cols[1]:
        trade_price = st.number_input("成交價", min_value=0.0, value=float(current_price), step=0.05, key=f"{prefix}_trade_price")
    with cols[2]:
        trade_shares = st.number_input("股數", min_value=0, value=0, step=1, key=f"{prefix}_trade_shares")
    with cols[3]:
        trade_amount = st.number_input("成交金額，可填 0 自動計算", min_value=0.0, value=0.0, step=1000.0, key=f"{prefix}_trade_amount")

    action_cols = st.columns(3)
    if action_cols[0].button("新增交易紀錄", key=f"{prefix}_add_trade_log"):
        amount = float(trade_amount or 0.0)
        if amount <= 0:
            amount = float(trade_price) * int(trade_shares)
        if trade_price <= 0 or trade_shares <= 0 or amount <= 0:
            st.warning("請輸入有效成交價、股數與金額。")
        else:
            log_trade(action, float(trade_price), int(trade_shares), amount)
            reload_trade_log("已新增一筆交易紀錄。")
            st.success("交易紀錄已寫入 trade_log.json")
            st.rerun()

    if action_cols[1].button("載入交易紀錄", key=f"{prefix}_load_trade_log"):
        reload_trade_log("已從 trade_log.json 載入。")
        st.success("交易紀錄已重新載入。")
        st.rerun()

    if action_cols[2].button("清空交易紀錄", key=f"{prefix}_clear_trade_log"):
        save_trade_log([])
        reload_trade_log("交易紀錄已清空。")
        st.warning("trade_log 已清空。")
        st.rerun()

    render_trade_log_records(list(st.session_state.get("trade_log_records", [])))

def normalize_inventory_editor_rows(edited_rows: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for _, row in edited_rows.iterrows():
        price = float(row.get("price", 0.0) or 0.0)
        lots = int(row.get("lots", 0) or 0)
        odd_shares = int(row.get("odd_shares", 0) or 0)
        fee = float(row.get("fee", 0.0) or 0.0)
        estimated_fee = round(price * (lots * 1000 + odd_shares) * FEE_RATE, 0)
        fee_source = str(row.get("fee_source", "手動輸入") or "手動輸入")
        if abs(fee - estimated_fee) > 1:
            fee_source = "手動輸入"
        records.append(
            {
                "date": str(row.get("date", "")),
                "side": str(row.get("side", "買入") or "買入"),
                "price": price,
                "lots": lots,
                "odd_shares": odd_shares,
                "fee": fee,
                "fee_source": fee_source,
            }
        )
    return records


def calculate_inventory_summary(records: list[dict[str, object]]) -> dict[str, float]:
    total_shares = 0
    total_invested_cost = 0.0
    for record in records:
        side = str(record.get("side", "") or "")
        shares = int(record.get("shares", 0) or 0)
        net_amount = float(record.get("net_amount", 0.0) or 0.0)
        if side == "賣出":
            total_shares -= shares
            total_invested_cost -= net_amount
        else:
            total_shares += shares
            total_invested_cost += net_amount

    total_shares = max(0, total_shares)
    if total_shares <= 0:
        total_invested_cost = 0.0
    average_cost = total_invested_cost / total_shares if total_shares > 0 else 0.0
    return {
        "total_shares": float(total_shares),
        "holding_lots": total_shares / 1000,
        "total_invested_cost": total_invested_cost,
        "average_cost": average_cost,
    }


def render_inventory_readonly_inputs(records: list[dict[str, object]]) -> dict[str, float]:
    if not records:
        st.warning("尚未載入庫存，請先至逐筆庫存儲存區新增或載入庫存。")
    summary = calculate_inventory_summary(records)
    cols = st.columns(4)
    cols[0].metric("目前已持有", f"{summary['holding_lots']:,.3f} 張")
    cols[1].metric("總股數", f"{summary['total_shares']:,.0f} 股")
    cols[2].metric("平均成本", format_price(summary["average_cost"]))
    cols[3].metric("總投入成本", format_price(summary["total_invested_cost"]))
    return summary


def render_inventory_persistence(prefix: str) -> None:
    records = list(st.session_state.get("inventory_records", []))
    st.caption(f"目前已載入 {len(records)} 筆庫存")
    st.caption("儲存狀態：已儲存" if st.session_state.get("inventory_saved", False) else "儲存狀態：尚未儲存")
    if st.session_state.get("inventory_status_message"):
        st.caption(str(st.session_state.inventory_status_message))
    with st.expander("逐筆庫存儲存", expanded=False):
        render_inventory_records(records)
        st.divider()
        st.caption("編輯既有庫存")
        if records:
            editable_columns = ["date", "side", "price", "lots", "odd_shares", "fee", "fee_source"]
            editor_df = pd.DataFrame(records)
            editor_df = editor_df[[column for column in editable_columns if column in editor_df.columns]]
            edited_df = st.data_editor(
                editor_df,
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                key=f"{prefix}_inventory_editor",
                column_config={
                    "date": st.column_config.TextColumn("日期"),
                    "side": st.column_config.SelectboxColumn("類型", options=["買入", "賣出"]),
                    "price": st.column_config.NumberColumn("成交價", min_value=0.0, step=0.05),
                    "lots": st.column_config.NumberColumn("張數", min_value=0, step=1),
                    "odd_shares": st.column_config.NumberColumn("零股", min_value=0, max_value=999, step=1),
                    "fee": st.column_config.NumberColumn("手續費", min_value=0.0, step=1.0),
                    "fee_source": st.column_config.SelectboxColumn("手續費來源", options=["手動輸入", "系統估算"]),
                },
            )
            if st.button("套用修改", key=f"{prefix}_apply_inventory_edits"):
                updated_records = normalize_inventory_editor_rows(edited_df)
                set_inventory_records(updated_records, "既有庫存已套用修改並自動儲存。")
                st.success("已套用修改，股數、成交金額、交易稅、交易成本與淨收付金額已重新計算。")
                st.rerun()
        else:
            st.info("目前沒有可編輯的既有庫存。")

        st.divider()
        st.caption("新增一筆庫存")
        cols = st.columns(3)
        with cols[0]:
            record_date = st.date_input("日期", value=date.today(), key=f"{prefix}_inventory_date")
        with cols[1]:
            record_side = st.selectbox("類型", options=["買入", "賣出"], key=f"{prefix}_inventory_side")
        with cols[2]:
            record_price = st.number_input("成交價", min_value=0.0, value=0.0, step=0.05, key=f"{prefix}_inventory_price")

        cols = st.columns(3)
        with cols[0]:
            record_lots = st.number_input("張數", min_value=0, value=0, step=1, key=f"{prefix}_inventory_lots")
        with cols[1]:
            record_odd_shares = st.number_input("零股", min_value=0, max_value=999, value=0, step=1, key=f"{prefix}_inventory_odd_shares")
        estimated_fee = round(float(record_price) * (int(record_lots) * 1000 + int(record_odd_shares)) * FEE_RATE, 0)
        fee_key = f"{prefix}_inventory_fee"
        estimated_fee_key = f"{prefix}_inventory_estimated_fee"
        previous_estimated_fee = float(st.session_state.get(estimated_fee_key, -1.0) or -1.0)
        if fee_key not in st.session_state or float(st.session_state.get(fee_key, 0.0) or 0.0) == previous_estimated_fee:
            st.session_state[fee_key] = float(estimated_fee)
        st.session_state[estimated_fee_key] = float(estimated_fee)
        with cols[2]:
            record_fee = st.number_input(
                "手續費（以實際券商為準）",
                min_value=0.0,
                step=1.0,
                help="預設只作參考；若有券商折扣，請輸入實際手續費，系統會以此作為真實成本。",
                key=fee_key,
            )
            st.caption(f"參考估算：{format_price(estimated_fee)}，實際成本以手動輸入為準")

        action_cols = st.columns(4)
        if action_cols[0].button("新增庫存", key=f"{prefix}_add_inventory"):
            if record_price <= 0 or (record_lots <= 0 and record_odd_shares <= 0):
                st.warning("請輸入成交價，且張數或零股至少一項大於 0。")
            else:
                records.append(
                    {
                        "date": record_date.isoformat(),
                        "side": record_side,
                        "price": float(record_price),
                        "lots": int(record_lots),
                        "odd_shares": int(record_odd_shares),
                        "fee": float(record_fee),
                    }
                )
                set_inventory_records(records)
                st.success("已新增一筆庫存，並已自動儲存。")
                st.rerun()

        if action_cols[1].button("💾 儲存庫存", key=f"{prefix}_save_inventory"):
            set_inventory_records(records)
            st.success("庫存已儲存。")

        if action_cols[2].button("📂 載入庫存", key=f"{prefix}_load_inventory"):
            st.session_state.inventory_records = load_inventory()
            st.session_state.inventory_saved = True
            st.session_state.inventory_status_message = "已從 inventory.json 載入。"
            st.success("庫存已重新載入。")
            st.rerun()

        if action_cols[3].button("🗑 清空庫存", key=f"{prefix}_clear_inventory"):
            set_inventory_records([], "庫存已清空並自動儲存。")
            st.warning("庫存已清空。")
            st.rerun()

        render_transaction_performance(list(st.session_state.get("inventory_records", [])))


def render_inputs(prefix: str, default_price: float, use_inventory_position: bool = False) -> dict[str, float | None]:
    st.subheader("投資參數")
    inventory_summary = {"holding_lots": 0.0, "average_cost": 0.0}
    if use_inventory_position:
        inventory_summary = render_inventory_readonly_inputs(list(st.session_state.get("inventory_records", [])))

    cols = st.columns(2)
    with cols[0]:
        if use_inventory_position:
            holding_lots = float(inventory_summary["holding_lots"])
            average_cost = float(inventory_summary["average_cost"])
        else:
            holding_lots = st.number_input("目前已持有幾張", min_value=0.0, value=0.0, step=1.0, key=f"{prefix}_holding_lots")
            average_cost = st.number_input("目前每股平均成本", min_value=0.0, value=0.0, step=0.1, key=f"{prefix}_average_cost")
        cash = st.number_input("可投入現金", min_value=0.0, value=100_000.0, step=10_000.0, key=f"{prefix}_cash")
        intraday_price = st.number_input("目前市價", min_value=0.0, value=float(default_price), step=0.05, key=f"{prefix}_intraday_price")
        price_source = "使用者手動輸入" if abs(float(intraday_price) - float(default_price)) > 0.001 else "最新收盤價"
    with cols[1]:
        manual_volume = st.number_input("今日成交量，可選填", min_value=0.0, value=0.0, step=1000.0, key=f"{prefix}_manual_volume")
        today_budget = st.number_input("今日預算投資金額（元）（今天最多願意投入）", min_value=0.0, value=100_000.0, step=10_000.0, key=f"{prefix}_today_budget")
        max_ratio = st.slider("這檔最多佔總資產幾%（避免買太多）", min_value=0, max_value=100, value=70, step=5, key=f"{prefix}_max_ratio")
    render_inventory_persistence(prefix)
    return {
        "holding_lots": holding_lots,
        "average_cost": average_cost,
        "total_shares": float(inventory_summary.get("total_shares", 0.0) or 0.0),
        "total_invested_cost": float(inventory_summary.get("total_invested_cost", 0.0) or 0.0),
        "cash": cash,
        "intraday_price": intraday_price,
        "price_source": price_source,
        "manual_volume": manual_volume if manual_volume > 0 else None,
        "today_budget": today_budget,
        "max_ratio": float(max_ratio),
    }


def run_analysis_page(label: str, ticker: str, prefix: str, start: date, end: date, use_inventory_position: bool = False) -> None:
    with st.spinner("正在取得行情與計算 AI 決策..."):
        resolved_ticker, data = fetch_taiwan_stock(ticker, start, end)

    if data.empty or data["Close"].dropna().empty:
        st.warning("資料不足，使用簡化模型")
        return

    latest_close = float(data["Close"].dropna().iloc[-1])
    inputs = render_inputs(prefix, latest_close, use_inventory_position=use_inventory_position)
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
        max_single_investment=float(inputs["today_budget"] or 0),
        max_stock_ratio=float(inputs["max_ratio"] or 0),
    )
    decision = make_decision(
        tech=tech,
        volume=volume,
        market=market,
        portfolio=portfolio,
        max_stock_ratio=float(inputs["max_ratio"] or 0),
        current_price=current_price,
        today_budget=float(inputs["today_budget"] or 0),
        max_position_ratio=float(inputs["max_ratio"] or 0),
        ticker=resolved_ticker,
        latest_close=latest_close,
    )

    st.caption(f"分析標的：{display_name(label, resolved_ticker)}｜最新收盤價：{format_price(latest_close)}｜盤中價格：{format_price(current_price)}")
    price_cols = st.columns(2)
    price_cols[0].metric("目前決策價格", format_price(current_price))
    price_cols[1].metric("價格來源", str(inputs.get("price_source", "系統行情")))
    if latest_close > 0 and abs(current_price - latest_close) / latest_close >= 0.10:
        st.warning("手動價格與最新收盤價差異較大，請確認是否為即時盤中價格。")
    render_trader_decision_card(decision)
    render_holding_pnl_card(portfolio)
    render_trade_log_section(prefix, decision, portfolio, current_price)
    render_position_summary_card(portfolio)
    render_key_reasons(decision)

    with st.expander("進階分析細節", expanded=False):
        render_self_check(decision)
        render_trade_record_summary(decision)
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
        st.subheader("價格趨勢")
        render_price_trend_chart(analysis_data, f"{display_name(label, resolved_ticker)} 價格趨勢")


def normalize_ticker_editor_rows(edited_rows: pd.DataFrame, ticker: str, name: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for _, row in edited_rows.iterrows():
        if bool(row.get("delete", False)):
            continue
        price = float(row.get("price", 0.0) or 0.0)
        lots = int(row.get("lots", 0) or 0)
        odd_shares = int(row.get("odd_shares", 0) or 0)
        if price <= 0 or (lots <= 0 and odd_shares <= 0):
            continue
        fee = float(row.get("fee", 0.0) or 0.0)
        estimated_fee = round(price * (lots * 1000 + odd_shares) * FEE_RATE, 0)
        fee_source = str(row.get("fee_source", "手動輸入") or "手動輸入")
        if abs(fee - estimated_fee) > 1:
            fee_source = "手動輸入"
        records.append(
            {
                "ticker": ticker,
                "name": name,
                "date": str(row.get("date", "")),
                "side": str(row.get("side", "買入") or "買入"),
                "price": price,
                "lots": lots,
                "odd_shares": odd_shares,
                "fee": fee,
                "fee_source": fee_source,
                "note": str(row.get("note", "") or ""),
            }
        )
    return records


def save_ticker_inventory(ticker: str, ticker_records: list[dict[str, object]], message: str) -> None:
    all_records = list(st.session_state.get("inventory_records", []))
    other_records = [record for record in all_records if str(record.get("ticker", "") or "").upper() != ticker.upper()]
    save_inventory(other_records + ticker_records)
    st.session_state.inventory_records = load_inventory()
    st.session_state.inventory_saved = True
    st.session_state.inventory_status_message = message


def render_target_inventory_manager(ticker: str, name: str, prefix: str, current_price: float) -> list[dict[str, object]]:
    all_records = list(st.session_state.get("inventory_records", []))
    ticker_records = get_inventory_by_ticker(all_records, ticker)
    with st.expander("📦 庫存明細 / 新增紀錄", expanded=False):
        st.caption(f"{ticker} 已載入 {len(ticker_records)} 筆庫存")
        if st.session_state.get("inventory_status_message"):
            st.caption(str(st.session_state.inventory_status_message))
        render_inventory_records(ticker_records, current_price=current_price)

        st.divider()
        st.caption("修改 / 刪除既有紀錄")
        if ticker_records:
            editor_df = pd.DataFrame(ticker_records)
            editor_columns = ["date", "side", "price", "lots", "odd_shares", "fee", "fee_source", "note"]
            editor_df = editor_df[[column for column in editor_columns if column in editor_df.columns]]
            editor_df.insert(0, "delete", False)
            edited_df = st.data_editor(
                editor_df,
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                key=f"{prefix}_ticker_inventory_editor",
                column_config={
                    "delete": st.column_config.CheckboxColumn("刪除"),
                    "date": st.column_config.TextColumn("日期"),
                    "side": st.column_config.SelectboxColumn("類型", options=["買入", "賣出"]),
                    "price": st.column_config.NumberColumn("成交價", min_value=0.0, step=0.05),
                    "lots": st.column_config.NumberColumn("張數", min_value=0, step=1),
                    "odd_shares": st.column_config.NumberColumn("零股", min_value=0, max_value=999, step=1),
                    "fee": st.column_config.NumberColumn("手續費", min_value=0.0, step=1.0),
                    "fee_source": st.column_config.SelectboxColumn("手續費來源", options=["手動輸入", "系統估算"]),
                    "note": st.column_config.TextColumn("備註"),
                },
            )
            if st.button("套用修改並自動儲存", key=f"{prefix}_apply_ticker_inventory"):
                updated = normalize_ticker_editor_rows(edited_df, ticker, name)
                save_ticker_inventory(ticker, updated, f"{ticker} 庫存已自動儲存。")
                st.success("已自動儲存。")
                st.rerun()
        else:
            st.info("尚未持有此標的，可先新增一筆買進紀錄。")

        st.divider()
        st.caption("新增一筆庫存")
        add_cols = st.columns(3)
        with add_cols[0]:
            record_date = st.date_input("日期", value=date.today(), key=f"{prefix}_target_inventory_date")
        with add_cols[1]:
            record_side = st.selectbox("類型", options=["買入", "賣出"], key=f"{prefix}_target_inventory_side")
        with add_cols[2]:
            record_price = st.number_input("成交價", min_value=0.0, value=0.0, step=0.05, key=f"{prefix}_target_inventory_price")
        add_cols = st.columns(4)
        with add_cols[0]:
            record_lots = st.number_input("張數", min_value=0, value=0, step=1, key=f"{prefix}_target_inventory_lots")
        with add_cols[1]:
            record_odd_shares = st.number_input("零股", min_value=0, max_value=999, value=0, step=1, key=f"{prefix}_target_inventory_odd")
        estimated_fee = round(float(record_price) * (int(record_lots) * 1000 + int(record_odd_shares)) * FEE_RATE, 0)
        with add_cols[2]:
            record_fee = st.number_input("手續費（以實際券商為準）", min_value=0.0, value=float(estimated_fee), step=1.0, key=f"{prefix}_target_inventory_fee")
            st.caption(f"參考估算：{format_price(estimated_fee)}")
        with add_cols[3]:
            record_note = st.text_input("備註", value="", key=f"{prefix}_target_inventory_note")

        if st.button("新增紀錄並自動儲存", key=f"{prefix}_add_ticker_inventory"):
            if not ticker:
                st.warning("ticker 缺失，不可寫入庫存。")
            elif record_price <= 0 or (record_lots <= 0 and record_odd_shares <= 0):
                st.warning("請輸入成交價，且張數或零股至少一項大於 0。")
            else:
                new_record = {
                    "ticker": ticker,
                    "name": name,
                    "date": record_date.isoformat(),
                    "side": record_side,
                    "price": float(record_price),
                    "lots": int(record_lots),
                    "odd_shares": int(record_odd_shares),
                    "fee": float(record_fee),
                    "fee_source": "手動輸入",
                    "note": record_note,
                }
                save_ticker_inventory(ticker, ticker_records + [new_record], f"{ticker} 庫存已自動儲存。")
                st.success("已自動儲存。")
                st.rerun()

    return get_inventory_by_ticker(list(st.session_state.get("inventory_records", [])), ticker)


def get_target_params(prefix: str, latest_close: float) -> dict[str, float | str | None]:
    current_price = float(st.session_state.get(f"{prefix}_target_current_price", latest_close) or latest_close or 0.0)
    cash = float(st.session_state.get(f"{prefix}_target_cash", 100_000.0) or 0.0)
    manual_volume_raw = float(st.session_state.get(f"{prefix}_target_manual_volume", 0.0) or 0.0)
    today_budget = float(st.session_state.get(f"{prefix}_target_today_budget", 100_000.0) or 0.0)
    max_ratio = float(st.session_state.get(f"{prefix}_target_max_ratio", 70.0) or 0.0)
    price_source = "使用者手動輸入" if abs(current_price - float(latest_close or 0.0)) > 0.001 else "最新收盤價"
    return {
        "cash": cash,
        "current_price": current_price,
        "manual_volume": manual_volume_raw if manual_volume_raw > 0 else None,
        "today_budget": today_budget,
        "max_ratio": max_ratio,
        "price_source": price_source,
    }


def render_target_inputs(prefix: str, latest_close: float) -> dict[str, float | None]:
    with st.expander("⚙️ 投資參數", expanded=False):
        cols = st.columns(2)
        with cols[0]:
            st.number_input("目前市價", min_value=0.0, value=float(st.session_state.get(f"{prefix}_target_current_price", latest_close) or latest_close), step=0.05, key=f"{prefix}_target_current_price")
            cash = st.number_input("可投入現金", min_value=0.0, value=100_000.0, step=10_000.0, key=f"{prefix}_target_cash")
        with cols[1]:
            manual_volume = st.number_input("今日成交量，可選填", min_value=0.0, value=0.0, step=1000.0, key=f"{prefix}_target_manual_volume")
            today_budget = st.number_input("今日預算投資金額", min_value=0.0, value=100_000.0, step=10_000.0, key=f"{prefix}_target_today_budget")
            max_ratio = st.slider("這檔最多佔整體資金比例", min_value=0, max_value=100, value=70, step=5, key=f"{prefix}_target_max_ratio")
        st.caption("調整後頁面會依新參數重新計算。")
    return {
        "cash": cash,
        "current_price": float(st.session_state.get(f"{prefix}_target_current_price", latest_close) or latest_close or 0.0),
        "manual_volume": manual_volume if manual_volume > 0 else None,
        "today_budget": today_budget,
        "max_ratio": float(max_ratio),
    }


def render_inventory_summary(summary: dict[str, object]) -> None:
    st.subheader("📊 持倉績效")
    if int(summary.get("total_shares", 0) or 0) <= 0:
        st.info("尚未持有，目前無未實現損益。")
    portfolio_summary = summary.get("portfolio_summary", {}) or {}
    performance_cols = st.columns(4)
    performance_cols[0].metric("總損益", format_price(float(portfolio_summary.get("unrealized_pnl", 0.0) or 0.0)))
    performance_cols[1].metric("報酬率", format_pct(float(portfolio_summary.get("pnl_pct", 0.0) or 0.0)))
    performance_cols[2].metric("現金比例", format_pct(float(portfolio_summary.get("cash_ratio", 0.0) or 0.0)))
    performance_cols[3].metric("股票比例", format_pct(float(portfolio_summary.get("stock_ratio", 0.0) or 0.0)))

    st.subheader("持倉損益摘要")
    cols = st.columns(4)
    cols[0].metric("總張數", f"{int(summary.get('total_lots', 0) or 0)} 張 {int(summary.get('total_odd_shares', 0) or 0)} 股")
    cols[1].metric("總股數", f"{int(summary.get('total_shares', 0) or 0):,} 股")
    cols[2].metric("平均成本", format_price(float(summary.get("average_cost", 0.0) or 0.0)))
    cols[3].metric("總成本", format_price(float(summary.get("total_cost", 0.0) or 0.0)))
    cols = st.columns(4)
    cols[0].metric("目前市值", format_price(float(summary.get("market_value", 0.0) or 0.0)))
    cols[1].metric("未實現損益", format_price(float(summary.get("unrealized_pnl", 0.0) or 0.0)))
    cols[2].metric("未實現報酬率", format_pct(float(summary.get("unrealized_pnl_pct", 0.0) or 0.0)))
    cols[3].metric("股票資產比例", format_pct(float(summary.get("current_stock_ratio", 0.0) or 0.0)))
    if summary.get("negative_position"):
        st.warning("同一標的庫存出現負股數，請檢查賣出紀錄。")


def render_target_page(label: str, ticker: str, prefix: str, start: date, end: date) -> None:
    with st.spinner("正在取得行情與計算 AI 決策..."):
        resolved_ticker, data = fetch_taiwan_stock(ticker, start, end)

    if data.empty or data["Close"].dropna().empty:
        st.warning("資料不足，使用簡化模型")
        latest_close = 0.0
    else:
        latest_close = float(data["Close"].dropna().iloc[-1])

    st.subheader(display_name(label, resolved_ticker))
    inputs = get_target_params(prefix, latest_close)
    current_price = float(inputs["current_price"] or latest_close or 0.0)
    st.caption(f"目前決策價格：{format_price(current_price)}｜價格來源：{inputs.get('price_source', '最新收盤價')}")
    if latest_close > 0 and abs(current_price - latest_close) / latest_close >= 0.10:
        st.warning("手動價格與最新收盤價差異較大，請確認是否為即時盤中價格。")

    all_records = list(st.session_state.get("inventory_records", []))
    ticker_records = get_inventory_by_ticker(all_records, resolved_ticker)
    summary = build_inventory_summary(ticker_records, current_price, float(inputs["cash"] or 0.0), float(inputs["max_ratio"] or 0.0))

    if data.empty:
        render_inventory_summary(summary)
        return
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
        holding_lots=float(summary.get("holding_lots", 0.0) or 0.0),
        average_cost=float(summary.get("average_cost", 0.0) or 0.0),
        cash=float(inputs["cash"] or 0.0),
        current_price=current_price,
        max_single_investment=float(inputs["today_budget"] or 0.0),
        max_stock_ratio=float(inputs["max_ratio"] or 0.0),
    )
    if summary.get("negative_position"):
        portfolio["over_target_ratio"] = True
        portfolio["position_room_amount"] = 0.0
        portfolio["negative_position"] = True
    if abs(float(portfolio.get("average_cost", 0.0) or 0.0) - float(summary.get("average_cost", 0.0) or 0.0)) > 0.01:
        st.warning("AI使用的平均成本與庫存計算結果不一致，請檢查資料。")
    decision = make_decision(
        tech=tech,
        volume=volume,
        market=market,
        portfolio=portfolio,
        max_stock_ratio=float(inputs["max_ratio"] or 0.0),
        current_price=current_price,
        today_budget=float(inputs["today_budget"] or 0.0),
        max_position_ratio=float(inputs["max_ratio"] or 0.0),
        ticker=resolved_ticker,
        latest_close=latest_close,
    )
    validation_result = run_system_validation(
        records=ticker_records,
        ticker=resolved_ticker,
        portfolio={**portfolio, "total_shares": summary.get("total_shares", portfolio.get("shares", 0))},
        decision=decision,
        trade_plan=decision.trade_plan,
        available_budget=float(getattr(decision, "available_budget", 0.0) or 0.0),
        today_budget=float(inputs["today_budget"] or 0.0),
        cash=float(inputs["cash"] or 0.0),
        max_stock_ratio=float(inputs["max_ratio"] or 0.0),
        current_price=current_price,
    )
    system_flags = list(validation_result.get("conflict_flags", []) or [])

    render_trader_decision_card(decision)
    render_system_validation(validation_result)
    render_inventory_summary(summary)
    render_target_inputs(prefix, latest_close)
    ticker_records = render_target_inventory_manager(resolved_ticker, label, prefix, current_price)

    with st.expander("進階分析細節", expanded=False):
        render_trade_log_section(prefix, decision, portfolio, current_price)
        render_key_reasons(decision)
        render_self_check(decision)
        render_trade_record_summary(decision)
        render_risk_detail(decision)
        render_score_cards(decision.module_scores, decision.total_score)
        render_volume_card(volume)
        render_market_factor_card(market)
        render_portfolio_table(portfolio)
        render_transaction_performance(ticker_records)
        st.subheader("價格趨勢")
        render_price_trend_chart(analysis_data, f"{display_name(label, resolved_ticker)} 價格趨勢")


def normalize_inventory_ticker(raw_ticker: str) -> str:
    ticker = str(raw_ticker or "").strip().upper()
    if not ticker:
        return ""
    ticker = ticker.replace(".TWO", ".TW")
    if ticker.endswith(".TW"):
        return ticker
    if ticker.isdigit() and len(ticker) == 4:
        return f"{ticker}.TW"
    return ticker


def normalize_all_inventory_editor_rows(edited_rows: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for _, row in edited_rows.iterrows():
        if bool(row.get("delete", False)):
            continue
        ticker = normalize_inventory_ticker(str(row.get("ticker", "") or ""))
        price = float(row.get("price", 0.0) or 0.0)
        lots = int(row.get("lots", 0) or 0)
        odd_shares = int(row.get("odd_shares", 0) or 0)
        if not ticker or price <= 0 or (lots <= 0 and odd_shares <= 0):
            continue
        side = str(row.get("type", row.get("side", "買入")) or "買入")
        if side not in ("買入", "賣出"):
            side = "買入"
        records.append(
            {
                "ticker": ticker,
                "name": str(row.get("name", "") or ticker),
                "date": str(row.get("date", "") or ""),
                "side": side,
                "type": side,
                "price": price,
                "lots": lots,
                "odd_shares": odd_shares,
                "fee": float(row.get("fee", 0.0) or 0.0),
                "fee_source": str(row.get("fee_source", "手動輸入") or "手動輸入"),
                "tax": float(row.get("tax", 0.0) or 0.0),
                "note": str(row.get("note", "") or ""),
            }
        )
    return records


def fetch_inventory_latest_prices(records: list[dict[str, object]], start: date, end: date) -> dict[str, float]:
    prices: dict[str, float] = {}
    tickers = sorted({str(record.get("ticker", "") or "").strip().upper() for record in records if record.get("ticker")})
    for ticker in tickers:
        try:
            _, data = fetch_taiwan_stock(ticker, start, end)
            close = data["Close"].dropna() if not data.empty and "Close" in data else pd.Series(dtype=float)
            if not close.empty:
                prices[ticker] = float(close.iloc[-1])
        except Exception:
            continue
    return prices


def render_inventory_summary_by_ticker(records: list[dict[str, object]], current_prices: dict[str, float]) -> None:
    st.subheader("各標的彙總")
    summary_rows = summarize_inventory_by_ticker(records, current_prices)
    if not summary_rows:
        st.info("目前沒有可彙總的庫存紀錄。")
        return
    display = pd.DataFrame(summary_rows)
    display = display.rename(
        columns={
            "ticker": "ticker",
            "name": "名稱",
            "total_shares": "總股數",
            "total_cost": "總成本",
            "current_price": "估算現價",
            "market_value": "目前市值",
            "unrealized_pnl": "未實現損益",
            "unrealized_pnl_pct": "未實現報酬率%",
        }
    )
    if "未實現報酬率%" in display.columns:
        styled = display.style.map(
            lambda value: "color: #22c55e" if float(value or 0) >= 0 else "color: #ef4444",
            subset=["未實現報酬率%"],
        )
        st.dataframe(styled, hide_index=True, use_container_width=True)
    else:
        st.dataframe(display, hide_index=True, use_container_width=True)


def render_all_inventory_page(start: date, end: date) -> None:
    st.subheader("全部庫存")
    all_records = list(st.session_state.get("inventory_records", []))
    st.caption(f"目前 inventory.json 共 {len(all_records)} 筆庫存紀錄。新增、修改或刪除後會自動儲存。")
    if st.session_state.get("inventory_status_message"):
        st.caption(str(st.session_state.inventory_status_message))

    with st.spinner("正在更新各標的估算現價..."):
        current_prices = fetch_inventory_latest_prices(all_records, start, end) if all_records else {}
    render_inventory_summary_by_ticker(all_records, current_prices)

    tickers = sorted({str(record.get("ticker", "") or "").strip().upper() for record in all_records if record.get("ticker")})
    filter_options = ["全部"] + tickers
    selected_ticker = st.selectbox("依 ticker 篩選", options=filter_options, key="all_inventory_filter")
    filtered_records = all_records if selected_ticker == "全部" else get_inventory_by_ticker(all_records, selected_ticker)

    st.subheader("庫存明細")
    render_inventory_records(filtered_records)

    st.subheader("修改 / 刪除庫存")
    if filtered_records:
        editor_df = pd.DataFrame(filtered_records)
        if "type" not in editor_df.columns:
            editor_df["type"] = editor_df.get("side", "買入")
        editor_columns = ["ticker", "name", "date", "type", "price", "lots", "odd_shares", "fee", "fee_source", "tax", "note"]
        editor_df = editor_df[[column for column in editor_columns if column in editor_df.columns]]
        editor_df.insert(0, "delete", False)
        edited_df = st.data_editor(
            editor_df,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key="all_inventory_editor",
            column_config={
                "delete": st.column_config.CheckboxColumn("刪除"),
                "ticker": st.column_config.TextColumn("ticker"),
                "name": st.column_config.TextColumn("name"),
                "date": st.column_config.TextColumn("date"),
                "type": st.column_config.SelectboxColumn("type", options=["買入", "賣出"]),
                "price": st.column_config.NumberColumn("price", min_value=0.0, step=0.05),
                "lots": st.column_config.NumberColumn("lots", min_value=0, step=1),
                "odd_shares": st.column_config.NumberColumn("odd_shares", min_value=0, max_value=999, step=1),
                "fee": st.column_config.NumberColumn("fee", min_value=0.0, step=1.0),
                "fee_source": st.column_config.SelectboxColumn("fee_source", options=["手動輸入", "系統估算"]),
                "tax": st.column_config.NumberColumn("tax", min_value=0.0, step=1.0),
                "note": st.column_config.TextColumn("note"),
            },
        )
        if st.button("套用修改並自動儲存", key="all_inventory_apply_edits"):
            updated_records = normalize_all_inventory_editor_rows(edited_df)
            if selected_ticker == "全部":
                next_records = updated_records
            else:
                other_records = [
                    record
                    for record in all_records
                    if str(record.get("ticker", "") or "").strip().upper() != selected_ticker
                ]
                next_records = other_records + updated_records
            set_inventory_records(next_records, "全部庫存已自動儲存。")
            st.success("已自動儲存 inventory.json。")
            st.rerun()
    else:
        st.info("目前篩選條件下沒有庫存紀錄。")

    st.divider()
    st.subheader("新增一筆庫存")
    add_cols = st.columns(3)
    with add_cols[0]:
        raw_ticker = st.text_input("ticker", value="0050.TW", key="all_inventory_new_ticker")
    with add_cols[1]:
        record_name = st.text_input("name", value="元大台灣50", key="all_inventory_new_name")
    with add_cols[2]:
        record_date = st.date_input("date", value=date.today(), key="all_inventory_new_date")

    add_cols = st.columns(4)
    with add_cols[0]:
        record_side = st.selectbox("type", options=["買入", "賣出"], key="all_inventory_new_type")
    with add_cols[1]:
        record_price = st.number_input("price", min_value=0.0, value=0.0, step=0.05, key="all_inventory_new_price")
    with add_cols[2]:
        record_lots = st.number_input("lots", min_value=0, value=0, step=1, key="all_inventory_new_lots")
    with add_cols[3]:
        record_odd_shares = st.number_input("odd_shares", min_value=0, max_value=999, value=0, step=1, key="all_inventory_new_odd")

    total_shares = int(record_lots) * 1000 + int(record_odd_shares)
    estimated_fee = round(float(record_price) * total_shares * FEE_RATE, 0)
    add_cols = st.columns(4)
    with add_cols[0]:
        record_fee = st.number_input("fee", min_value=0.0, value=float(estimated_fee), step=1.0, key="all_inventory_new_fee")
        st.caption(f"系統估算：{format_price(estimated_fee)}")
    with add_cols[1]:
        record_fee_source = st.selectbox("fee_source", options=["手動輸入", "系統估算"], key="all_inventory_new_fee_source")
    with add_cols[2]:
        record_tax = st.number_input("tax", min_value=0.0, value=0.0, step=1.0, key="all_inventory_new_tax")
    with add_cols[3]:
        record_note = st.text_input("note", value="", key="all_inventory_new_note")

    if st.button("新增並自動儲存", key="all_inventory_add_record"):
        ticker = normalize_inventory_ticker(raw_ticker)
        if not ticker:
            st.warning("請輸入 ticker，例如 0050.TW 或 2330.TW。")
        elif record_price <= 0 or total_shares <= 0:
            st.warning("請輸入有效成交價與張數或零股。")
        else:
            new_record = {
                "ticker": ticker,
                "name": record_name or ticker,
                "date": record_date.isoformat(),
                "side": record_side,
                "type": record_side,
                "price": float(record_price),
                "lots": int(record_lots),
                "odd_shares": int(record_odd_shares),
                "fee": float(record_fee),
                "fee_source": record_fee_source,
                "tax": float(record_tax),
                "note": record_note,
            }
            set_inventory_records(all_records + [new_record], "已新增庫存並自動儲存。")
            st.success("已新增並自動儲存 inventory.json。")
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="AI 投資決策資訊系統", page_icon="0050", layout="wide")
    init_inventory_state()
    init_trade_log_state()
    inject_mobile_css()
    st.title("AI 投資決策資訊系統")
    st.caption("以 0050 與台股個股為核心，整合技術面、量能、市場背景與個人持倉風控。")
    for warning in consume_storage_warnings():
        st.warning(warning)

    start = date(2021, 1, 1)
    end = date.today()
    tab_0050, tab_stock, tab_inventory = st.tabs(["0050 AI決策", "台股AI分析", "全部庫存"])

    with tab_0050:
        st.subheader("0050 專頁")
        render_target_page("元大台灣50", "0050.TW", "etf0050", start, end)

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
            render_target_page(label, ticker, "stock_ai", start, end)

    with tab_inventory:
        render_all_inventory_page(start, end)


if __name__ == "__main__":
    main()
