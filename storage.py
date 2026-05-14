from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


INVENTORY_FILE = Path(__file__).with_name("inventory.json")
INVENTORY_BACKUP_FILE = Path(__file__).with_name("inventory_backup.json")
TRADE_LOG_FILE = Path(__file__).with_name("trade_log.json")
TRADE_LOG_BACKUP_FILE = Path(__file__).with_name("trade_log_backup.json")
FEE_RATE = 0.001425
TAX_RATE = 0.001
STOCK_TAX_RATE = 0.003

STORAGE_WARNINGS: list[str] = []


def consume_storage_warnings() -> list[str]:
    warnings = list(STORAGE_WARNINGS)
    STORAGE_WARNINGS.clear()
    return warnings


def _normalize_inventory_list(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("JSON content is not a list")
    normalized = []
    for record in data:
        if not isinstance(record, dict):
            continue
        normalized_record = _normalize_record(record)
        if normalized_record is not None:
            normalized.append(normalized_record)
    return sort_inventory_records(normalized)


def _read_json_list(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path.name} content is not a list")
    return data



def sort_inventory_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, pd.Timestamp, int, int]:
        original_index, record = item
        parsed_date = pd.to_datetime(record.get("date", ""), errors="coerce")
        invalid_date = 1 if pd.isna(parsed_date) else 0
        safe_date = pd.Timestamp.max if invalid_date else parsed_date
        action = str(record.get("side", record.get("type", "")) or "")
        action_order = 0 if action == "買入" else 1 if action == "賣出" else 2
        return invalid_date, safe_date, action_order, original_index

    return [record for _, record in sorted(enumerate(records), key=sort_key)]


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
    sell_tax_rate = TAX_RATE if ticker.startswith(("00", "006", "007", "008")) else STOCK_TAX_RATE
    tax = round(gross_amount * sell_tax_rate, 0) if side == "賣出" else 0.0
    transaction_cost = fee + tax
    net_amount = gross_amount + transaction_cost if side == "買入" else gross_amount - transaction_cost
    return {
        "ticker": ticker,
        "name": name,
        "date": str(record.get("date", "")),
        "side": side,
        "type": side,
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
    normalized = sort_inventory_records(normalized)
    if INVENTORY_FILE.exists():
        shutil.copy2(INVENTORY_FILE, INVENTORY_BACKUP_FILE)
    with INVENTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(normalized, file, ensure_ascii=False, indent=2)


def load_inventory() -> list[dict[str, Any]]:
    if not INVENTORY_FILE.exists():
        return []
    try:
        return _normalize_inventory_list(_read_json_list(INVENTORY_FILE))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        try:
            restored = _normalize_inventory_list(_read_json_list(INVENTORY_BACKUP_FILE))
            shutil.copy2(INVENTORY_BACKUP_FILE, INVENTORY_FILE)
            STORAGE_WARNINGS.append("inventory.json \u8b80\u53d6\u5931\u6557\uff0c\u5df2\u5f9e inventory_backup.json \u9084\u539f\u3002")
            return restored
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            STORAGE_WARNINGS.append("inventory.json \u8b80\u53d6\u5931\u6557\uff0c\u4e14 inventory_backup.json \u4e5f\u7121\u6cd5\u9084\u539f\u3002")
            return []


def get_inventory_by_ticker(records: list[dict[str, Any]], ticker: str) -> list[dict[str, Any]]:
    normalized_ticker = str(ticker or "").strip().upper()
    return sort_inventory_records([record for record in records if str(record.get("ticker", "") or "").strip().upper() == normalized_ticker])


def summarize_inventory_by_ticker(
    records: list[dict[str, Any]],
    current_prices: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    current_prices = {str(key).upper(): float(value or 0.0) for key, value in (current_prices or {}).items()}
    grouped: dict[str, dict[str, Any]] = {}
    for record in sort_inventory_records(records):
        ticker = str(record.get("ticker", "") or "").strip().upper()
        if not ticker:
            continue
        price = float(record.get("price", 0.0) or 0.0)
        shares = int(record.get("shares", 0) or 0)
        net_amount = float(record.get("net_amount", 0.0) or 0.0)
        if price <= 0 or shares <= 0:
            continue
        item = grouped.setdefault(
            ticker,
            {
                "ticker": ticker,
                "name": str(record.get("name", "") or ticker),
                "total_shares": 0,
                "total_cost": 0.0,
                "last_trade_price": 0.0,
            },
        )
        item["name"] = str(record.get("name", "") or item["name"] or ticker)
        item["last_trade_price"] = price
        if str(record.get("side", "") or "") == "賣出":
            item["total_shares"] -= shares
            item["total_cost"] -= net_amount
        else:
            item["total_shares"] += shares
            item["total_cost"] += net_amount

    summaries: list[dict[str, Any]] = []
    for ticker, item in grouped.items():
        total_shares = max(0, int(item["total_shares"]))
        total_cost = float(item["total_cost"]) if total_shares > 0 else 0.0
        current_price = current_prices.get(ticker) or float(item.get("last_trade_price", 0.0) or 0.0)
        market_value = total_shares * current_price
        unrealized_pnl = market_value - total_cost if total_shares > 0 else 0.0
        unrealized_pnl_pct = unrealized_pnl / total_cost * 100 if total_cost > 0 else 0.0
        summaries.append(
            {
                "ticker": ticker,
                "name": item["name"],
                "total_shares": total_shares,
                "total_cost": round(total_cost, 0),
                "current_price": current_price,
                "market_value": round(market_value, 0),
                "unrealized_pnl": round(unrealized_pnl, 0),
                "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
            }
        )
    return sorted(summaries, key=lambda row: str(row.get("ticker", "")))


def calculate_inventory_summary(
    records: list[dict[str, Any]],
    current_price: float,
    cash: float,
    max_stock_ratio: float,
) -> dict[str, Any]:
    valid_records = sort_inventory_records([record for record in records if float(record.get("price", 0.0) or 0.0) > 0])
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
        data = _read_json_list(TRADE_LOG_FILE)
        return [_normalize_trade(record) for record in data if isinstance(record, dict)]
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        try:
            data = _read_json_list(TRADE_LOG_BACKUP_FILE)
            shutil.copy2(TRADE_LOG_BACKUP_FILE, TRADE_LOG_FILE)
            STORAGE_WARNINGS.append("trade_log.json \u8b80\u53d6\u5931\u6557\uff0c\u5df2\u5f9e trade_log_backup.json \u9084\u539f\u3002")
            return [_normalize_trade(record) for record in data if isinstance(record, dict)]
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            STORAGE_WARNINGS.append("trade_log.json \u8b80\u53d6\u5931\u6557\uff0c\u4e14 trade_log_backup.json \u4e5f\u7121\u6cd5\u9084\u539f\u3002")
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
