from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


INVENTORY_FILE = Path(__file__).with_name("inventory.json")
INVENTORY_BACKUP_FILE = Path(__file__).with_name("inventory_backup.json")
TRADE_LOG_FILE = Path(__file__).with_name("trade_log.json")
TRADE_LOG_BACKUP_FILE = Path(__file__).with_name("trade_log_backup.json")
FEE_RATE = 0.001425
TAX_RATE = 0.001


def _normalize_record(record: dict[str, Any]) -> dict[str, Any] | None:
    ticker = str(record.get("ticker", "") or "").strip().upper()
    if not ticker and any(record.get(key) not in (None, "", 0, 0.0) for key in ("date", "price", "lots", "odd_shares", "fee")):
        ticker = "0050.TW"
    if not ticker:
        return None
    name = str(record.get("name", "") or ("元大台灣50" if ticker == "0050.TW" else ticker))
    side = str(record.get("side", record.get("type", "買入")) or "買入")
    side = "賣出" if side in ("sell", "SELL", "賣出") else "買入"
    price = float(record.get("price", 0.0) or 0.0)
    lots = int(record.get("lots", 0) or 0)
    odd_shares = int(record.get("odd_shares", 0) or 0)
    if price <= 0 or (lots <= 0 and odd_shares <= 0):
        return None
    shares = lots * 1000 + odd_shares
    gross_amount = price * shares
    fee_input = record.get("fee", None)
    fee_source_input = str(record.get("fee_source", "") or "")
    estimated_fee = round(gross_amount * FEE_RATE, 0)
    if fee_source_input == "系統估算":
        fee = estimated_fee
        fee_source = "系統估算"
    elif fee_input is None or fee_input == "":
        fee = round(gross_amount * FEE_RATE, 0)
        fee_source = "系統估算"
    else:
        fee = max(0.0, float(fee_input or 0.0))
        fee_source = "手動輸入"
    tax = round(gross_amount * TAX_RATE, 0) if side == "賣出" else 0.0
    transaction_cost = fee + tax
    net_amount = gross_amount + transaction_cost if side == "買入" else gross_amount - transaction_cost
    return {
        "ticker": ticker,
        "name": name,
        "date": str(record.get("date", "")),
        "side": side,
        "price": price,
        "lots": lots,
        "odd_shares": odd_shares,
        "shares": shares,
        "gross_amount": round(gross_amount, 0),
        "fee": fee,
        "fee_source": fee_source,
        "tax": tax,
        "transaction_cost": transaction_cost,
        "net_amount": round(net_amount, 0),
        "note": str(record.get("note", "") or ""),
    }


def save_inventory(records: list[dict[str, Any]]) -> None:
    normalized = []
    for record in records:
        if not isinstance(record, dict):
            continue
        normalized_record = _normalize_record(record)
        if normalized_record is not None:
            normalized.append(normalized_record)
    if INVENTORY_FILE.exists():
        shutil.copy2(INVENTORY_FILE, INVENTORY_BACKUP_FILE)
    with INVENTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(normalized, file, ensure_ascii=False, indent=2)


def load_inventory() -> list[dict[str, Any]]:
    if not INVENTORY_FILE.exists():
        return []
    try:
        with INVENTORY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            save_inventory([])
            return []
        normalized = []
        for record in data:
            if not isinstance(record, dict):
                continue
            normalized_record = _normalize_record(record)
            if normalized_record is not None:
                normalized.append(normalized_record)
        return normalized
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        save_inventory([])
        return []


def get_inventory_by_ticker(records: list[dict[str, Any]], ticker: str) -> list[dict[str, Any]]:
    normalized_ticker = str(ticker or "").strip().upper()
    return [record for record in records if str(record.get("ticker", "") or "").strip().upper() == normalized_ticker]


def calculate_inventory_summary(
    records: list[dict[str, Any]],
    current_price: float,
    cash: float,
    max_stock_ratio: float,
) -> dict[str, Any]:
    valid_records = [record for record in records if float(record.get("price", 0.0) or 0.0) > 0]
    ticker = str(valid_records[0].get("ticker", "") or "") if valid_records else ""
    total_shares = 0
    total_cost = 0.0

    current_price = max(0.0, float(current_price or 0.0))
    records_detail: list[dict[str, Any]] = []

    for record in valid_records:
        shares = int(record.get("shares", 0) or 0)
        net_amount = float(record.get("net_amount", 0.0) or 0.0)
        if str(record.get("side", "") or "") == "賣出":
            total_shares -= shares
            total_cost -= net_amount
        else:
            total_shares += shares
            total_cost += net_amount
        detail = dict(record)
        if str(record.get("side", "") or "") == "買入":
            market_value_per_record = shares * current_price
            unrealized_pnl = market_value_per_record - net_amount
            unrealized_pnl_pct = unrealized_pnl / net_amount * 100 if net_amount > 0 else 0.0
        else:
            market_value_per_record = 0.0
            unrealized_pnl = 0.0
            unrealized_pnl_pct = 0.0
        detail["current_price"] = current_price
        detail["unrealized_pnl"] = round(unrealized_pnl, 0)
        detail["unrealized_pnl_pct"] = round(unrealized_pnl_pct, 2)
        detail["market_value"] = round(market_value_per_record, 0)
        records_detail.append(detail)

    negative_position = total_shares < 0
    total_shares = max(0, total_shares)
    if total_shares <= 0:
        total_cost = 0.0

    market_value = total_shares * current_price
    total_assets = market_value + max(0.0, float(cash or 0.0))
    average_cost = total_cost / total_shares if total_shares > 0 else 0.0
    unrealized_pnl = market_value - total_cost if total_shares > 0 else 0.0
    unrealized_pnl_pct = unrealized_pnl / total_cost * 100 if total_cost > 0 else 0.0
    current_stock_ratio = market_value / total_assets * 100 if total_assets > 0 else 0.0
    cash_ratio = max(0.0, float(cash or 0.0)) / total_assets * 100 if total_assets > 0 else 0.0
    portfolio_summary = {
        "total_cost": total_cost,
        "market_value": market_value,
        "unrealized_pnl": unrealized_pnl,
        "pnl_pct": unrealized_pnl_pct,
        "cash_ratio": cash_ratio,
        "stock_ratio": current_stock_ratio,
    }

    return {
        "ticker": ticker,
        "total_shares": total_shares,
        "total_lots": total_shares // 1000,
        "total_odd_shares": total_shares % 1000,
        "holding_lots": total_shares / 1000,
        "total_cost": total_cost,
        "average_cost": average_cost,
        "market_value": market_value,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "current_stock_ratio": current_stock_ratio,
        "total_assets": total_assets,
        "max_stock_ratio": max_stock_ratio,
        "portfolio_summary": portfolio_summary,
        "records_detail": records_detail,
        "negative_position": negative_position,
    }


def _normalize_trade(record: dict[str, Any]) -> dict[str, Any]:
    action = str(record.get("action", "買入") or "買入")
    action = "賣出" if action in ("sell", "SELL", "賣出") else "買入"
    price = float(record.get("price", 0.0) or 0.0)
    shares = int(record.get("shares", 0) or 0)
    gross_amount = float(record.get("amount", 0.0) or 0.0)
    if gross_amount <= 0:
        gross_amount = price * shares
    fee = round(gross_amount * FEE_RATE, 0)
    tax = round(gross_amount * TAX_RATE, 0) if action == "賣出" else 0.0
    net_amount = gross_amount + fee if action == "買入" else gross_amount - fee - tax
    return {
        "date": str(record.get("date", date_today())),
        "action": action,
        "price": price,
        "shares": shares,
        "amount": round(gross_amount, 0),
        "fee": fee,
        "tax": tax,
        "net_amount": round(net_amount, 0),
    }


def date_today() -> str:
    from datetime import date

    return date.today().isoformat()


def save_trade_log(records: list[dict[str, Any]]) -> None:
    normalized = [_normalize_trade(record) for record in records if isinstance(record, dict)]
    if TRADE_LOG_FILE.exists():
        shutil.copy2(TRADE_LOG_FILE, TRADE_LOG_BACKUP_FILE)
    with TRADE_LOG_FILE.open("w", encoding="utf-8") as file:
        json.dump(normalized, file, ensure_ascii=False, indent=2)


def load_trade_log() -> list[dict[str, Any]]:
    if not TRADE_LOG_FILE.exists():
        return []
    try:
        with TRADE_LOG_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            save_trade_log([])
            return []
        return [_normalize_trade(record) for record in data if isinstance(record, dict)]
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        save_trade_log([])
        return []


def log_trade(action: str, price: float, shares: int, amount: float) -> dict[str, Any]:
    records = load_trade_log()
    record = _normalize_trade(
        {
            "date": date_today(),
            "action": action,
            "price": price,
            "shares": shares,
            "amount": amount,
        }
    )
    records.append(record)
    save_trade_log(records)
    return record
