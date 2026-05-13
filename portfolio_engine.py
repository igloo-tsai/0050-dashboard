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

    max_by_cash = int(cash // (current_price * LOT_SIZE)) if current_price > 0 else 0
    max_by_single = int(max_single_investment // (current_price * LOT_SIZE)) if current_price > 0 else 0
    max_buy_lots = max(0, min(max_by_cash, max_by_single))

    rows = []
    for add_lots in (1, 2, 3):
        add_cost = add_lots * LOT_SIZE * current_price
        new_shares = shares + add_lots * LOT_SIZE
        new_cost = cost_value + add_cost
        new_average_cost = new_cost / new_shares if new_shares else 0.0
        remaining_cash = cash - add_cost
        new_market_value = new_shares * current_price
        new_total_assets = new_market_value + max(0.0, remaining_cash)
        new_stock_ratio = new_market_value / new_total_assets * 100 if new_total_assets else 0.0
        over_limit = remaining_cash < 0 or new_stock_ratio > max_stock_ratio or add_cost > max_single_investment
        rows.append(
            {
                "加碼張數": add_lots,
                "加碼後平均成本": new_average_cost,
                "加碼後剩餘現金": remaining_cash,
                "加碼後股票資產比例": new_stock_ratio,
                "是否超過風控上限": "是" if over_limit else "否",
            }
        )

    return {
        "market_value": market_value,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "current_stock_ratio": current_stock_ratio,
        "max_buy_lots": max_buy_lots,
        "scenario_table": pd.DataFrame(rows),
    }
