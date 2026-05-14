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
        if max_stock_ratio > 0:
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
) -> dict[str, Any]:
    result = _result()
    _merge(result, validate_inventory(records or [], ticker))
    if portfolio is not None:
        _merge(result, validate_portfolio(portfolio))
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
    result["passed"] = not result["errors"]
    return result
