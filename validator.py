from __future__ import annotations

from datetime import datetime
from typing import Any


def _result() -> dict[str, Any]:
    return {
        "passed": True,
        "errors": [],
        "warnings": [],
        "fixed": [],
        "conflict_flags": [],
    }


def _merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key in ("errors", "warnings", "fixed", "conflict_flags"):
        target[key].extend(source.get(key, []))
    target["passed"] = not target["errors"]
    for key in ("errors", "warnings", "fixed", "conflict_flags"):
        target[key] = list(dict.fromkeys(target[key]))
    return target


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _is_sell(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"賣出", "sell", "sold", "s"}


def validate_inventory(records: list[dict[str, Any]] | None, ticker: str) -> dict[str, Any]:
    result = _result()
    records = list(records or [])
    normalized_ticker = str(ticker or "").strip().upper()
    if not normalized_ticker:
        result["errors"].append("ticker 不可空白")

    buy_shares = 0
    sell_shares = 0
    for index, record in enumerate(records, 1):
        record_ticker = str(record.get("ticker", "") or "").strip().upper()
        if normalized_ticker and record_ticker and record_ticker != normalized_ticker:
            result["errors"].append(f"第 {index} 筆庫存 ticker 不一致，不得混入目前標的")
        if not record_ticker:
            result["errors"].append(f"第 {index} 筆庫存 ticker 不可空白")

        price = _number(record.get("price"))
        lots = _integer(record.get("lots"))
        odd_shares = _integer(record.get("odd_shares"))
        if price <= 0:
            result["warnings"].append(f"第 {index} 筆庫存 price <= 0，已略過計算")
        if lots < 0 or odd_shares < 0:
            result["errors"].append(f"第 {index} 筆庫存張數或零股不可為負數")

        raw_date = str(record.get("date", "") or "")
        if raw_date:
            try:
                datetime.fromisoformat(raw_date)
            except ValueError:
                result["warnings"].append(f"第 {index} 筆庫存日期格式錯誤")
        else:
            result["warnings"].append(f"第 {index} 筆庫存日期空白")

        shares = max(0, lots) * 1000 + max(0, odd_shares)
        side = record.get("side", record.get("type", "買入"))
        if _is_sell(side):
            sell_shares += shares
        else:
            buy_shares += shares

    if sell_shares > buy_shares:
        result["errors"].append("此標的賣出股數大於買入股數，請檢查庫存")

    result["passed"] = not result["errors"]
    result["conflict_flags"] = list(result["errors"] + result["warnings"])
    return result


def validate_portfolio(portfolio: dict[str, Any]) -> dict[str, Any]:
    result = _result()
    total_assets = _number(portfolio.get("total_assets"))
    market_value = _number(portfolio.get("market_value"))
    current_stock_ratio = _number(portfolio.get("current_stock_ratio"))
    average_cost = _number(portfolio.get("average_cost"))
    total_shares = _number(portfolio.get("total_shares", portfolio.get("shares")))
    cost_value = _number(portfolio.get("cost_value"))
    unrealized_pnl = _number(portfolio.get("unrealized_pnl"))

    if total_assets < 0:
        result["errors"].append("total_assets 不得為負")
    if market_value < 0:
        result["errors"].append("market_value 不得為負")
    if current_stock_ratio < 0 or current_stock_ratio > 100.5:
        result["warnings"].append("current_stock_ratio 超出合理範圍")
    if average_cost == 0 and total_shares > 0:
        result["errors"].append("average_cost 為 0 但 total_shares > 0")
    if abs(unrealized_pnl - (market_value - cost_value)) > 1:
        result["warnings"].append("unrealized_pnl 與 market_value - cost_value 不一致")

    result["passed"] = not result["errors"]
    result["conflict_flags"] = list(result["errors"] + result["warnings"])
    return result


def validate_trade_plan(
    trade_plan: list[dict[str, Any]] | None,
    available_budget: float,
    today_budget: float,
    cash: float,
    max_stock_ratio: float,
    current_price: float | None = None,
) -> dict[str, Any]:
    result = _result()
    if trade_plan is None:
        trade_plan = []
        result["warnings"].append("trade_plan 為 None，UI 需使用三層價格 fallback")
    if not trade_plan:
        result["warnings"].append("trade_plan 為空，UI 需 fallback 顯示三層策略")
        result["passed"] = not result["errors"]
        result["conflict_flags"] = list(result["errors"] + result["warnings"])
        return result

    prices = [_number(row.get("price")) for row in trade_plan[:3]]
    if len(prices) >= 3:
        reference_price = _number(current_price, prices[0])
        if not (reference_price >= prices[0] >= prices[1] >= prices[2]):
            result["warnings"].append("三層價格需符合 current_price >= observation >= reasonable >= conservative")

    total_amount = 0.0
    for index, row in enumerate(trade_plan, 1):
        shares = _integer(row.get("shares"))
        lots = _integer(row.get("lots"))
        if shares >= 1000:
            lots += shares // 1000
            shares = shares % 1000
            row["lots"] = lots
            row["shares"] = shares
            result["fixed"].append(f"第 {index} 批零股 >= 1000，已自動轉換為張數")

        amount = _number(row.get("amount"))
        batch_budget = _number(row.get("batch_budget", row.get("budget")))
        if amount > batch_budget + 1:
            row["amount"] = batch_budget
            amount = batch_budget
            result["fixed"].append(f"第 {index} 批投入金額超過批次預算，已壓回 batch_budget")

        after_ratio = _number(row.get("after_batch_stock_ratio"))
        over_limit = bool(row.get("over_limit_after_batch", False))
        total_buy_shares = _integer(row.get("total_buy_shares", row.get("cumulative_total_shares")))
        total_buy_amount = _number(row.get("total_buy_amount", row.get("cumulative_amount")))
        if total_buy_shares == 0:
            if abs(_number(row.get("after_batch_average_cost")) - _number(row.get("original_average_cost", row.get("after_batch_average_cost")))) > 0.01:
                result["warnings"].append(f"第 {index} 批買入 0 股時，成交後平均成本不應改變")
            if abs(_number(row.get("after_batch_stock_ratio")) - _number(row.get("original_stock_ratio", row.get("after_batch_stock_ratio")))) > 0.01:
                result["warnings"].append(f"第 {index} 批買入 0 股時，成交後股票比例不應改變")
            if abs(_number(row.get("after_batch_cash")) - _number(row.get("original_cash", row.get("after_batch_cash")))) > 1:
                result["warnings"].append(f"第 {index} 批買入 0 股時，成交後現金不應改變")
            if over_limit:
                row["over_limit_after_batch"] = False
                result["fixed"].append(f"第 {index} 批買入 0 股，over_limit_after_batch 已修正為 False")
        if max_stock_ratio > 0 and total_buy_amount > 0:
            expected_over = after_ratio > max_stock_ratio
            if expected_over != over_limit:
                row["over_limit_after_batch"] = expected_over
                result["fixed"].append(f"第 {index} 批 over_limit_after_batch 已依股票比例修正")

        total_amount += amount

    if total_amount > _number(available_budget) + 1:
        result["errors"].append("分批總投入超過 available_budget")
    if total_amount > _number(today_budget) + 1:
        result["errors"].append("分批總投入超過 today_budget")
    if total_amount > _number(cash) + 1:
        result["errors"].append("分批總投入超過 cash")

    result["passed"] = not result["errors"]
    result["conflict_flags"] = list(result["errors"] + result["warnings"])
    return result


def validate_decision(decision: Any, portfolio: dict[str, Any]) -> dict[str, Any]:
    result = _result()
    suggested_lots = _integer(getattr(decision, "suggested_buy_lots", 0))
    suggested_shares = _integer(getattr(decision, "suggested_buy_shares", 0))
    immediate_plan = str(getattr(decision, "immediate_plan", "") or "")
    immediate_lots = _integer(getattr(decision, "immediate_lots", suggested_lots))
    immediate_shares = _integer(getattr(decision, "immediate_shares", suggested_shares))
    today_status = str(getattr(decision, "today_status", "") or "")
    available_budget = _number(getattr(decision, "available_budget", 0.0))
    cash = _number(portfolio.get("cash", portfolio.get("available_cash")))
    over_target = bool(portfolio.get("over_target_ratio", False))
    trade_plan = list(getattr(decision, "trade_plan", []) or [])

    if over_target and (suggested_lots > 0 or suggested_shares > 0 or immediate_lots > 0 or immediate_shares > 0):
        result["errors"].append("部位超標時不得有立即買入建議")
    if available_budget <= 0 and (immediate_lots > 0 or immediate_shares > 0):
        result["errors"].append("今日預算或可用額度為 0 時不得有立即買入建議")
    if cash <= 0 and (immediate_lots > 0 or immediate_shares > 0):
        result["errors"].append("可用現金為 0 時不得有立即買入建議")
    if today_status in ("等待回檔", "禁止加碼", "暫緩進場") and (suggested_lots > 0 or suggested_shares > 0):
        result["errors"].append("today_status 與 suggested_buy_lots/shares 矛盾")
    if immediate_plan in ("WAIT", "AVOID", "不立即掛單") and (immediate_lots > 0 or immediate_shares > 0):
        result["errors"].append("immediate_plan 為等待/避免時，immediate_lots/shares 必須為 0")
    if len(trade_plan) < 3:
        result["warnings"].append("trade_plan 不足三層，UI 必須使用 fallback 顯示三層策略")

    result["passed"] = not result["errors"]
    result["conflict_flags"] = list(result["errors"] + result["warnings"])
    return result


def validate_broker_portfolio(broker_summary: dict[str, Any] | None) -> dict[str, Any]:
    result = _result()
    if not broker_summary:
        result["passed"] = True
        return result

    open_lots = list(broker_summary.get("open_lots", []) or [])
    realized_details = list(broker_summary.get("realized_details", []) or [])
    invalid_records = list(broker_summary.get("invalid_records", []) or [])

    if invalid_records:
        result["errors"].append("存在無效交易紀錄，券商級損益已排除該紀錄")
    for index, lot in enumerate(open_lots, 1):
        if _number(lot.get("fee")) < 0:
            result["errors"].append(f"第 {index} 筆 open lot 手續費不可為負")
        if _number(lot.get("tax")) < 0:
            result["errors"].append(f"第 {index} 筆 open lot 稅金不可為負")
        if _number(lot.get("buy_price")) <= 0:
            result["errors"].append(f"第 {index} 筆 open lot 買入價格不可小於等於 0")
        if _integer(lot.get("shares_remaining")) <= 0:
            result["errors"].append(f"第 {index} 筆 open lot 剩餘股數不可小於等於 0")

    total_shares = sum(_integer(lot.get("shares_remaining")) for lot in open_lots)
    total_cost_basis = sum(_number(lot.get("remaining_cost_basis")) for lot in open_lots)
    unrealized_pnl = sum(_number(lot.get("unrealized_pnl")) for lot in open_lots)
    realized_pnl = sum(_number(detail.get("realized_pnl")) for detail in realized_details)

    if total_shares != _integer(broker_summary.get("total_shares")):
        result["errors"].append("total_shares 必須等於 open_lots shares_remaining 加總")
    if abs(total_cost_basis - _number(broker_summary.get("total_cost_basis"))) > 1:
        result["errors"].append("total_cost_basis 必須等於 open_lots remaining_cost_basis 加總")
    if abs(unrealized_pnl - _number(broker_summary.get("unrealized_pnl"))) > 1:
        result["errors"].append("unrealized_pnl 必須等於 open_lots unrealized_pnl 加總")
    if abs(realized_pnl - _number(broker_summary.get("realized_pnl"))) > 1:
        result["errors"].append("realized_pnl 必須等於 realized_details 加總")

    for warning in list(broker_summary.get("warnings", []) or []):
        result["warnings"].append(str(warning))
    for error in list(broker_summary.get("errors", []) or []):
        result["errors"].append(str(error))

    result["passed"] = not result["errors"]
    result["conflict_flags"] = list(result["errors"] + result["warnings"])
    return result


def validate_valuation_quality(result_data: dict[str, Any] | None) -> dict[str, Any]:
    result = _result()
    if result_data is None:
        result["warnings"].append("估值與標的品質資料不存在，decision_engine 應採中性分數")
        result["passed"] = True
        result["conflict_flags"] = list(result["warnings"])
        return result

    for key in ("final_score", "valuation_score", "quality_score", "data_quality_score"):
        value = result_data.get(key)
        try:
            is_missing_score = value is None or value != value
        except Exception:
            is_missing_score = True
        if is_missing_score:
            result_data[key] = 50
            result["fixed"].append(f"{key} 為 None，已改為 50")
            value = 50
        score = _number(value)
        if score < 0 or score > 100:
            result_data[key] = max(0, min(100, score))
            result["fixed"].append(f"{key} 超出 0~100，已修正")

    for key in ("missing_fields", "reasons", "warnings"):
        if not isinstance(result_data.get(key), list):
            result_data[key] = []
            result["fixed"].append(f"{key} 不是 list，已改為空 list")

    if not bool(result_data.get("is_data_sufficient", False)):
        result["warnings"].append("估值與標的品質資料不足，decision_engine 不可過度依賴此模組")

    result["passed"] = not result["errors"]
    result["conflict_flags"] = list(result["errors"] + result["warnings"])
    return result


def validate_price_source(
    *,
    decision_price: float,
    latest_close: float,
    price_source: str,
    manual_override: bool,
    confirm_extreme_price: bool = False,
    price_warnings: list[str] | None = None,
) -> dict[str, Any]:
    result = _result()
    price = _number(decision_price)
    close = _number(latest_close)
    warnings = list(price_warnings or [])
    source = str(price_source or "")

    if price <= 0:
        result["errors"].append("決策價格必須大於 0")
    if source == "invalid_price":
        result["warnings"].append("目前價格來源標示為 invalid_price，AI 應使用安全 fallback")
    if manual_override and source != "manual_override" and source != "invalid_price":
        result["warnings"].append("手動覆寫狀態與 price_source 不一致")
    if close > 0 and price > 0:
        diff_pct = abs(price - close) / close * 100
        if diff_pct > 20 and source == "manual_override" and not confirm_extreme_price:
            result["errors"].append("決策價格與最近收盤價偏離超過 20%，不得直接進入 AI")
        elif diff_pct > 20 and source == "manual_override":
            result["warnings"].append("決策價格與最近收盤價偏離超過 20%，已由使用者確認")
        elif diff_pct > 10:
            result["warnings"].append("決策價格與最近收盤價偏離超過 10%")
    for warning in warnings:
        result["warnings"].append(str(warning))

    result["passed"] = not result["errors"]
    result["conflict_flags"] = list(result["errors"] + result["warnings"])
    return result


def run_system_validation(
    *,
    records: list[dict[str, Any]] | None = None,
    ticker: str = "",
    portfolio: dict[str, Any] | None = None,
    decision: Any = None,
    trade_plan: list[dict[str, Any]] | None = None,
    available_budget: float = 0.0,
    today_budget: float = 0.0,
    cash: float = 0.0,
    max_stock_ratio: float = 0.0,
    current_price: float | None = None,
    latest_close: float = 0.0,
    price_source: str = "",
    manual_override: bool = False,
    confirm_extreme_price: bool = False,
    price_warnings: list[str] | None = None,
    broker_summary: dict[str, Any] | None = None,
    valuation_quality_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = _result()
    _merge(result, validate_inventory(records or [], ticker))
    if portfolio is not None:
        _merge(result, validate_portfolio(portfolio))
    _merge(result, validate_broker_portfolio(broker_summary))
    _merge(result, validate_valuation_quality(valuation_quality_result))
    if decision is not None:
        _merge(result, validate_decision(decision, portfolio or {}))
    _merge(
        result,
        validate_trade_plan(
            trade_plan,
            available_budget=available_budget,
            today_budget=today_budget,
            cash=cash,
            max_stock_ratio=max_stock_ratio,
            current_price=current_price,
        ),
    )
    _merge(
        result,
        validate_price_source(
            decision_price=_number(current_price),
            latest_close=latest_close,
            price_source=price_source,
            manual_override=manual_override,
            confirm_extreme_price=confirm_extreme_price,
            price_warnings=price_warnings,
        ),
    )
    result["passed"] = not result["errors"]
    return result
