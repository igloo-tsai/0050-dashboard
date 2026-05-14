from __future__ import annotations

import pandas as pd


LOT_SIZE = 1000


def calculate_portfolio(
    holding_lots: float,
    average_cost: float,
    cash: float,
    current_price: float,
    max_single_investment: float,
    max_stock_ratio: float,
) -> dict[str, object]:
    shares = max(0.0, holding_lots) * LOT_SIZE
    market_value = shares * current_price
    cost_value = shares * average_cost
    unrealized_pnl = market_value - cost_value
    unrealized_pnl_pct = unrealized_pnl / cost_value * 100 if cost_value else 0.0
    total_assets = market_value + max(0.0, cash)
    current_stock_ratio = market_value / total_assets * 100 if total_assets else 0.0
    max_position_value = total_assets * max_stock_ratio / 100 if max_stock_ratio > 0 else 0.0
    position_room_amount = max(0.0, max_position_value - market_value) if max_stock_ratio > 0 else max(0.0, cash)

    lot_cost = current_price * LOT_SIZE
    max_by_cash = int(cash // lot_cost) if lot_cost > 0 else 0
    max_by_single = int(max_single_investment // lot_cost) if lot_cost > 0 else 0
    over_target_ratio = current_stock_ratio > max_stock_ratio if max_stock_ratio > 0 else False
    near_target_ratio = current_stock_ratio >= max_stock_ratio * 0.85 if max_stock_ratio > 0 else False
    excess_stock_ratio = max(0.0, current_stock_ratio - max_stock_ratio) if max_stock_ratio > 0 else 0.0
    price_vs_cost_pct = (current_price / average_cost - 1.0) * 100 if average_cost > 0 else 0.0

    def scenario(add_lots: int) -> dict[str, object]:
        add_cost = add_lots * lot_cost
        new_shares = shares + add_lots * LOT_SIZE
        new_cost = cost_value + add_cost
        new_average_cost = new_cost / new_shares if new_shares else 0.0
        remaining_cash = cash - add_cost
        new_market_value = new_shares * current_price
        new_total_assets = new_market_value + max(0.0, remaining_cash)
        new_stock_ratio = new_market_value / new_total_assets * 100 if new_total_assets else 0.0
        over_limit = remaining_cash < 0 or new_stock_ratio > max_stock_ratio or add_cost > max_single_investment
        return {
            "加碼張數": add_lots,
            "加碼後平均成本": new_average_cost,
            "加碼後剩餘現金": remaining_cash,
            "加碼後股票資產比例": new_stock_ratio,
            "是否超過風控上限": "是" if over_limit else "否",
            "over_limit": over_limit,
        }

    max_by_ratio = 0
    for lots in range(1, min(max_by_cash, max_by_single, 50) + 1):
        if scenario(lots)["over_limit"]:
            break
        max_by_ratio = lots

    max_buy_lots = max(0, min(max_by_cash, max_by_single, max_by_ratio))
    if over_target_ratio:
        max_buy_lots = 0
    elif near_target_ratio:
        max_buy_lots = min(max_buy_lots, 1)

    rows = []
    for add_lots in (1, 2, 3):
        row = scenario(add_lots)
        row.pop("over_limit", None)
        rows.append(row)

    return {
        "holding_lots": holding_lots,
        "shares": shares,
        "average_cost": average_cost,
        "cash": cash,
        "available_cash": cash,
        "current_price": current_price,
        "max_single_investment": max_single_investment,
        "max_stock_ratio": max_stock_ratio,
        "cost_value": cost_value,
        "market_value": market_value,
        "total_assets": total_assets,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "current_stock_ratio": current_stock_ratio,
        "over_target_ratio": over_target_ratio,
        "near_target_ratio": near_target_ratio,
        "excess_stock_ratio": excess_stock_ratio,
        "position_room_amount": position_room_amount,
        "price_vs_cost_pct": price_vs_cost_pct,
        "price_above_cost_10pct": price_vs_cost_pct >= 10,
        "price_below_cost": average_cost > 0 and current_price < average_cost,
        "max_buy_lots": max_buy_lots,
        "scenario": scenario,
        "scenario_table": pd.DataFrame(rows),
    }
