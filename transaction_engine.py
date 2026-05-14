from __future__ import annotations

from datetime import datetime
from typing import Any


FEE_RATE = 0.001425
ETF_TAX_RATE = 0.001
STOCK_TAX_RATE = 0.003
LOT_SIZE = 1000


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _side(record: dict[str, Any]) -> str:
    raw = str(record.get("type", record.get("side", "")) or "").strip().lower()
    if raw in {"sell", "sold", "s", "賣出", "鞈?"}:
        return "sell"
    if raw in {"dividend", "股利", "配息"}:
        return "dividend"
    if raw in {"fee_adjustment", "fee", "費用調整"}:
        return "fee_adjustment"
    return "buy"


def _security_tax_rate(security_type: str) -> float:
    return ETF_TAX_RATE if str(security_type or "").upper() == "ETF" else STOCK_TAX_RATE


def _parse_date(value: Any) -> datetime:
    try:
        return datetime.fromisoformat(str(value or ""))
    except ValueError:
        return datetime.max


def sort_transaction_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"buy": 0, "sell": 1, "dividend": 2, "fee_adjustment": 3}

    def key(item: tuple[int, dict[str, Any]]) -> tuple[datetime, int, int]:
        index, record = item
        return _parse_date(record.get("date")), order.get(_side(record), 9), index

    return [record for _, record in sorted(enumerate(records or []), key=key)]


def normalize_transaction_record(record: dict[str, Any], security_type: str = "ETF") -> dict[str, Any]:
    side = _side(record)
    price = _number(record.get("price"))
    lots = _integer(record.get("lots"))
    odd_shares = _integer(record.get("odd_shares"))
    shares = _integer(record.get("shares"))
    if shares <= 0:
        shares = lots * LOT_SIZE + odd_shares
    gross_amount = _number(record.get("gross_amount"))
    if gross_amount <= 0 and price > 0 and shares > 0:
        gross_amount = price * shares

    fee_source = str(record.get("fee_source", record.get("fee_mode", "")) or "").strip().lower()
    fee_raw = record.get("fee")
    if fee_source in {"auto", "系統估算"} or fee_raw in (None, ""):
        fee = gross_amount * FEE_RATE if side in {"buy", "sell"} else _number(fee_raw)
        fee_mode = "auto" if side in {"buy", "sell"} else "manual"
    else:
        fee = max(0.0, _number(fee_raw))
        fee_mode = "manual"

    tax_raw = record.get("tax")
    if tax_raw not in (None, ""):
        tax = max(0.0, _number(tax_raw))
    elif side == "sell":
        tax = gross_amount * _security_tax_rate(security_type)
    else:
        tax = 0.0

    if side == "buy":
        net_amount = gross_amount + fee + tax
    elif side == "sell":
        net_amount = gross_amount - fee - tax
    else:
        net_amount = _number(record.get("net_amount"))

    return {
        **record,
        "ticker": str(record.get("ticker", "") or "").strip().upper(),
        "name": str(record.get("name", "") or ""),
        "date": str(record.get("date", "") or ""),
        "type": side,
        "side": side,
        "price": price,
        "lots": lots,
        "odd_shares": odd_shares,
        "shares": shares,
        "gross_amount": gross_amount,
        "fee": fee,
        "fee_mode": fee_mode,
        "fee_source": "manual" if fee_mode == "manual" else "auto",
        "tax": tax,
        "net_amount": net_amount,
        "note": str(record.get("note", "") or ""),
    }


def calculate_broker_grade_portfolio(
    records: list[dict[str, Any]],
    current_price: float,
    ticker: str,
    security_type: str = "ETF",
) -> dict[str, Any]:
    normalized_ticker = str(ticker or "").strip().upper()
    current_price = max(0.0, _number(current_price))
    open_lots: list[dict[str, Any]] = []
    realized_details: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    dividend_income = 0.0
    total_fees = 0.0
    total_tax = 0.0

    for raw in sort_transaction_records(list(records or [])):
        record = normalize_transaction_record(raw, security_type=security_type)
        record_ticker = str(record.get("ticker", "") or "").strip().upper()
        if normalized_ticker and record_ticker and record_ticker != normalized_ticker:
            continue

        side = str(record.get("type", "buy"))
        shares = _integer(record.get("shares"))
        price = _number(record.get("price"))

        if side in {"buy", "sell"}:
            if price <= 0 or shares <= 0:
                invalid = {**record, "reason": "price <= 0 或 shares <= 0"}
                invalid_records.append(invalid)
                errors.append(f"{record.get('date', '')} {side} 價格或股數無效，已排除。")
                continue

        if side == "buy":
            cost_basis = _number(record.get("net_amount"))
            cost_per_share = cost_basis / shares if shares > 0 else 0.0
            total_fees += _number(record.get("fee"))
            total_tax += _number(record.get("tax"))
            open_lots.append(
                {
                    "buy_date": record.get("date", ""),
                    "buy_price": price,
                    "shares_remaining": shares,
                    "original_shares": shares,
                    "cost_basis": cost_basis,
                    "remaining_cost_basis": cost_basis,
                    "cost_per_share": cost_per_share,
                    "fee": _number(record.get("fee")),
                    "tax": _number(record.get("tax")),
                    "source_record": record,
                }
            )
            continue

        if side == "sell":
            available_shares = sum(_integer(lot.get("shares_remaining")) for lot in open_lots)
            if shares > available_shares:
                invalid = {**record, "reason": "賣出股數大於可用庫存"}
                invalid_records.append(invalid)
                errors.append(f"{record.get('date', '')} 賣出 {shares} 股大於庫存 {available_shares} 股，該筆已排除。")
                continue

            total_fees += _number(record.get("fee"))
            total_tax += _number(record.get("tax"))
            shares_to_match = shares
            sell_net_amount = _number(record.get("net_amount"))
            sell_price = price
            for lot in open_lots:
                lot_shares = _integer(lot.get("shares_remaining"))
                if shares_to_match <= 0:
                    break
                if lot_shares <= 0:
                    continue

                matched_shares = min(shares_to_match, lot_shares)
                ratio = matched_shares / shares if shares > 0 else 0.0
                cost_basis = _number(lot.get("cost_per_share")) * matched_shares
                sell_proceeds = sell_net_amount * ratio
                realized_pnl = sell_proceeds - cost_basis
                realized_pnl_pct = realized_pnl / cost_basis * 100 if cost_basis > 0 else 0.0
                holding_days = 0
                try:
                    holding_days = (_parse_date(record.get("date")) - _parse_date(lot.get("buy_date"))).days
                except Exception:
                    holding_days = 0

                realized_details.append(
                    {
                        "sell_date": record.get("date", ""),
                        "sell_price": sell_price,
                        "matched_buy_date": lot.get("buy_date", ""),
                        "matched_buy_price": lot.get("buy_price", 0.0),
                        "matched_shares": matched_shares,
                        "cost_basis": cost_basis,
                        "sell_proceeds_allocated": sell_proceeds,
                        "realized_pnl": realized_pnl,
                        "realized_pnl_pct": realized_pnl_pct,
                        "holding_days": holding_days,
                    }
                )

                lot["shares_remaining"] = lot_shares - matched_shares
                lot["remaining_cost_basis"] = _number(lot.get("cost_per_share")) * _integer(lot.get("shares_remaining"))
                shares_to_match -= matched_shares
            continue

        if side == "dividend":
            current_shares = sum(_integer(lot.get("shares_remaining")) for lot in open_lots)
            if current_shares <= 0:
                warnings.append(f"{record.get('date', '')} 無持股卻有股利紀錄，請確認。")
            cash_dividend_per_share = _number(record.get("cash_dividend_per_share"))
            tax_withheld = _number(record.get("tax_withheld"))
            dividend_received = _number(record.get("dividend_received"))
            if dividend_received <= 0:
                dividend_received = cash_dividend_per_share * current_shares - tax_withheld
            dividend_income += max(0.0, dividend_received)
            total_tax += max(0.0, tax_withheld)
            continue

        if side == "fee_adjustment":
            fee = _number(record.get("fee"))
            tax = _number(record.get("tax"))
            if fee < 0 or tax < 0:
                errors.append(f"{record.get('date', '')} 費用調整不可為負。")
                invalid_records.append({**record, "reason": "費用或稅金為負"})
                continue
            total_fees += fee
            total_tax += tax

    active_lots: list[dict[str, Any]] = []
    for lot in open_lots:
        shares_remaining = _integer(lot.get("shares_remaining"))
        if shares_remaining <= 0:
            continue
        remaining_cost_basis = _number(lot.get("remaining_cost_basis"))
        market_value = current_price * shares_remaining
        unrealized_pnl = market_value - remaining_cost_basis
        unrealized_pnl_pct = unrealized_pnl / remaining_cost_basis * 100 if remaining_cost_basis > 0 else 0.0
        active_lots.append(
            {
                **lot,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
            }
        )

    total_shares = sum(_integer(lot.get("shares_remaining")) for lot in active_lots)
    total_cost_basis = sum(_number(lot.get("remaining_cost_basis")) for lot in active_lots)
    market_value = current_price * total_shares
    unrealized_pnl = sum(_number(lot.get("unrealized_pnl")) for lot in active_lots)
    realized_pnl = sum(_number(detail.get("realized_pnl")) for detail in realized_details)
    total_return = unrealized_pnl + realized_pnl + dividend_income
    invested_basis = total_cost_basis + sum(_number(detail.get("cost_basis")) for detail in realized_details)

    return {
        "ticker": normalized_ticker,
        "total_shares": total_shares,
        "total_lots": total_shares // LOT_SIZE,
        "odd_shares": total_shares % LOT_SIZE,
        "total_cost_basis": total_cost_basis,
        "average_cost": total_cost_basis / total_shares if total_shares > 0 else 0.0,
        "market_value": market_value,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl / total_cost_basis * 100 if total_cost_basis > 0 else 0.0,
        "realized_pnl": realized_pnl,
        "realized_pnl_pct": realized_pnl / sum(_number(detail.get("cost_basis")) for detail in realized_details) * 100 if realized_details else 0.0,
        "dividend_income": dividend_income,
        "total_return": total_return,
        "total_return_pct": total_return / invested_basis * 100 if invested_basis > 0 else 0.0,
        "total_fees": total_fees,
        "total_tax": total_tax,
        "open_lots": active_lots,
        "realized_details": realized_details,
        "invalid_records": invalid_records,
        "warnings": warnings,
        "errors": errors,
    }
