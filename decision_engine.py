from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DecisionResult:
    total_score: int
    recommendation: str
    action_label: str
    position_mode: str
    position_mode_label: str
    market_temperature: str
    risk_level: str
    chase_today: str
    aggressive_bid: float
    reasonable_bid: float
    conservative_bid: float
    suggested_bid: float
    suggested_buy_lots: int
    suggested_buy_shares: int
    immediate_plan: str
    immediate_lots: int
    immediate_shares: int
    immediate_price: float | None
    max_buy_lots: int
    available_budget: float
    observation_price: float
    reasonable_price: float
    conservative_price: float
    potential_lots_at_reasonable: int
    trade_plan: list[dict[str, object]]
    staged_plan: list[dict[str, object]]
    trader_plan_v2: dict[str, object]
    today_status: str
    conflict_flags: list[str]
    trade_record: dict[str, object]
    entry_probability: int
    entry_probability_text: str
    risk_score: int
    risk_bar_label: str
    after_buy_average_cost: float | None
    after_buy_remaining_cash: float | None
    after_buy_stock_ratio: float | None
    over_position_limit_after_buy: bool
    next_action: str
    primary_reasons: list[str]
    reasons: list[str]
    module_scores: dict[str, int]
    valuation_quality_result: dict[str, object] | None = None


def clamp_score(value: float) -> int:
    return int(max(0, min(100, round(value))))


def score_trend(tech: dict[str, float | None]) -> tuple[int, list[str]]:
    score = 50
    reasons: list[str] = []
    price = tech.get("latest_price")
    if price is None:
        return 50, ["價格資料不足，趨勢分數採中性。"]

    used_indicators = 0
    for key, label, points in (("ma20", "MA20", 8), ("ma60", "MA60", 10), ("ma120", "MA120", 12)):
        ma = tech.get(key)
        if ma is None:
            continue
        used_indicators += 1
        if price > ma:
            score += points
            reasons.append(f"價格站上 {label}，趨勢加分。")
        else:
            score -= 4
            reasons.append(f"價格低於 {label}，趨勢保守。")

    rsi = tech.get("rsi")
    if rsi is not None and rsi > 72:
        score -= 12
        reasons.append("RSI 偏高，避免在過熱區追價。")
    if used_indicators == 0:
        reasons.append("均線資料不足，趨勢分數採中性。")
    return clamp_score(score), reasons


def score_price_position(tech: dict[str, float | None]) -> tuple[int, list[str]]:
    drawdown = tech.get("recent_high_drawdown") or 0.0
    distance_ma120 = tech.get("distance_ma120")
    rsi = tech.get("rsi")
    score = 55
    reasons: list[str] = []

    if drawdown <= -20:
        score += 25
        reasons.append("價格自近期高點回撤超過 20%，具備較明顯分批機會。")
    elif drawdown <= -10:
        score += 15
        reasons.append("價格自近期高點回撤超過 10%，價格位置較有吸引力。")
    elif drawdown > -3:
        score -= 18
        reasons.append("價格接近近期高點，追價風險較高。")

    if rsi is not None and rsi < 40:
        score += 10
        reasons.append("RSI 低於 40，短線偏冷，機會分數提高。")
    if distance_ma120 is not None and abs(distance_ma120) <= 3:
        score += 8
        reasons.append("價格接近 MA120，具備中期支撐參考。")
    return clamp_score(score), reasons


def score_portfolio_risk(portfolio: dict[str, object], max_stock_ratio: float) -> tuple[int, list[str]]:
    max_buy_lots = int(portfolio.get("max_buy_lots", 0))
    price_vs_cost = float(portfolio.get("price_vs_cost_pct", 0.0))
    score = 75
    reasons: list[str] = []

    if portfolio.get("over_target_ratio"):
        score = 5
        reasons.append("目前股票資產比例已高於目標上限，風控大幅扣分。")
    elif portfolio.get("near_target_ratio"):
        score = 45
        reasons.append("目前股票資產比例接近上限，只適合小量試單。")
    else:
        reasons.append("股票資產比例仍在目標範圍內。")

    if max_buy_lots <= 0:
        score = min(score, 35)
        reasons.append("可用現金或今日預算不足，無法安全加碼。")
    if portfolio.get("price_above_cost_10pct"):
        score = min(score, 55)
        reasons.append(f"目前價格高於平均成本 {price_vs_cost:.1f}%，不鼓勵追高加碼。")
    if portfolio.get("price_below_cost") and not portfolio.get("over_target_ratio"):
        score = min(100, score + 10)
        reasons.append("價格低於平均成本且持倉未超標，可提高分批加碼彈性。")
    return clamp_score(score), reasons


def probability_text(probability: int) -> str:
    if probability <= 39:
        return "低，不建議進場"
    if probability <= 59:
        return "中低，僅觀察"
    if probability <= 74:
        return "中高，可小量試單"
    return "高，可分批加碼"


def risk_label(risk_score: int) -> str:
    if risk_score <= 39:
        return "低風險"
    if risk_score <= 69:
        return "中風險"
    return "高風險"


def classify_cost_price_zone(current_price: float, average_cost: float) -> tuple[str, int, str]:
    if average_cost <= 0 or current_price <= 0:
        return "無成本資料", 0, "尚無平均成本，價格區以市場訊號為主。"
    if current_price >= average_cost:
        return "高於成本區", 0, "價格高於平均成本，避免把加碼變成追價。"

    discount = (average_cost - current_price) / average_cost
    if discount <= 0.02:
        return "低於成本承接區", 1, "價格低於平均成本 0%~2%，可小量承接。"
    if discount <= 0.05:
        return "積極分批區", 2, "價格低於平均成本 2%~5%，可積極分批。"
    return "攻擊區", 2, "價格低於平均成本 5%以上，屬攻擊區，但仍需分批。"


def determine_position_mode(
    total: int,
    entry_probability: int,
    risk_score_value: int,
    price_score: int,
    current_stock_ratio: float,
    max_stock_ratio: float,
) -> tuple[str, str]:
    if max_stock_ratio > 0 and current_stock_ratio > max_stock_ratio:
        return "AVOID", "暫緩進場（AVOID）"
    if risk_score_value >= 80:
        return "AVOID", "暫緩進場（AVOID）"

    adjusted = min(total, entry_probability)
    if price_score < 45:
        adjusted -= 5
    if max_stock_ratio > 0 and current_stock_ratio >= max_stock_ratio * 0.85:
        adjusted -= 8

    if adjusted > 75:
        return "AGGRESSIVE", "積極加碼（AGGRESSIVE）"
    if adjusted >= 65:
        return "SCALE_IN", "分批加碼（SCALE_IN）"
    if adjusted >= 50:
        return "PROBE", "試單模式（PROBE）"
    if adjusted >= 40:
        return "HOLD", "觀察（HOLD）"
    return "AVOID", "暫緩進場（AVOID）"


def action_from_mode(position_mode: str) -> str:
    return {
        "AGGRESSIVE": "積極加碼",
        "SCALE_IN": "分批加碼",
        "PROBE": "試單",
        "HOLD": "觀察",
        "AVOID": "暫緩進場",
    }.get(position_mode, "觀察")


def downgrade_position_mode(position_mode: str) -> str:
    order = ["AVOID", "HOLD", "PROBE", "SCALE_IN", "AGGRESSIVE"]
    if position_mode not in order:
        return "HOLD"
    return order[max(0, order.index(position_mode) - 1)]


def build_next_action(
    today_status: str,
    action_label: str,
    lots: int,
    shares: int,
    suggested_bid: float,
    reasonable_price: float,
    trade_plan: list[dict[str, object]],
    over_limit: bool,
) -> str:
    if over_limit or today_status == "禁止加碼":
        return "已達或超過部位上限，不新增部位。"

    observation = next((row for row in trade_plan if row.get("level_name") == "觀察價"), None)
    reasonable = next((row for row in trade_plan if row.get("level_name") == "合理價"), None)
    conservative = next((row for row in trade_plan if row.get("level_name") == "保守價"), None)
    observation_text = f"{float(observation.get('price', suggested_bid)):.2f}" if observation else f"{suggested_bid:.2f}"
    reasonable_text = f"{float(reasonable.get('price', reasonable_price)):.2f}" if reasonable else f"{reasonable_price:.2f}"
    conservative_text = f"{float(conservative.get('price', reasonable_price * 0.95)):.2f}" if conservative else f"{reasonable_price * 0.95:.2f}"

    if today_status == "現在可買":
        return "目前已進入合理區，可依第一批或第二批預算分批掛單。"
    return f"現在不追價；若跌到觀察價 {observation_text}，啟動第一批；若跌到合理價 {reasonable_text}，啟動第二批；若跌到保守價 {conservative_text}，啟動第三批。"


def build_trade_plan(
    total_batch_budget: float,
    prices: list[tuple[str, float]],
    disabled: bool,
    base_shares: float,
    base_cost_value: float,
    cash: float,
    max_position_ratio: float,
    original_price: float | None = None,
) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    batch_ratios = {
        "觀察價": 0.20,
        "合理價": 0.30,
        "保守價": 0.50,
    }
    batch_titles = {
        "觀察價": "第一批｜觀察價",
        "合理價": "第二批｜合理價",
        "保守價": "第三批｜保守價",
    }
    cumulative_amount = 0.0
    cumulative_total_shares = 0

    for level_name, price in prices:
        safe_price = max(0.0, float(price or 0.0))
        batch_ratio = batch_ratios.get(level_name, 0.0)
        batch_budget = max(0.0, total_batch_budget * batch_ratio)

        if disabled or batch_budget <= 0 or safe_price <= 0:
            lots = 0
            shares = 0
            amount = 0.0
            action_text = "預算不足，不建議下單" if not disabled else "已超過部位上限，不加碼"
        else:
            lot_price = safe_price * 1000
            lots = int(batch_budget // lot_price)
            remaining_budget = batch_budget - lots * lot_price
            shares = int(remaining_budget // safe_price)
            shares = max(0, min(999, shares))
            amount = lots * lot_price + shares * safe_price
            if lots == 0 and shares == 0:
                action_text = "預算不足，不建議下單"
                amount = 0.0
            else:
                action_text = f"跌到 {safe_price:.2f} 可買 {lots} 張 {shares} 股"

        cumulative_amount += amount
        cumulative_total_shares += lots * 1000 + shares
        cumulative_lots = cumulative_total_shares // 1000
        cumulative_shares = cumulative_total_shares % 1000
        total_buy_amount = cumulative_amount
        total_buy_shares = cumulative_total_shares

        original_shares = int(max(0.0, base_shares))
        original_avg_cost = base_cost_value / original_shares if original_shares > 0 else 0.0
        original_reference_price = max(0.0, float(original_price if original_price is not None else safe_price))
        original_market_value = original_shares * original_reference_price
        original_total_assets = original_market_value + max(0.0, cash)
        original_stock_ratio = original_market_value / original_total_assets * 100 if original_total_assets > 0 else 0.0

        if total_buy_shares == 0:
            after_batch_total_shares = original_shares
            after_batch_lots = after_batch_total_shares // 1000
            after_batch_odd_shares = after_batch_total_shares % 1000
            after_batch_total_cost = base_cost_value
            after_batch_average_cost = original_avg_cost
            after_batch_market_value = original_market_value
            after_batch_cash = cash
            after_batch_stock_ratio = original_stock_ratio
            after_batch_unrealized_pnl = after_batch_market_value - after_batch_total_cost
            after_batch_unrealized_pnl_pct = after_batch_unrealized_pnl / after_batch_total_cost * 100 if after_batch_total_cost > 0 else 0.0
            over_limit_after_batch = False
        else:
            after_batch_total_shares = int(base_shares + total_buy_shares)
            after_batch_lots = after_batch_total_shares // 1000
            after_batch_odd_shares = after_batch_total_shares % 1000
            after_batch_total_cost = base_cost_value + total_buy_amount
            after_batch_average_cost = after_batch_total_cost / after_batch_total_shares if after_batch_total_shares > 0 else 0.0
            after_batch_market_value = after_batch_total_shares * safe_price
            after_batch_cash = cash - total_buy_amount
            after_batch_total_assets = after_batch_market_value + max(0.0, after_batch_cash)
            after_batch_stock_ratio = after_batch_market_value / after_batch_total_assets * 100 if after_batch_total_assets > 0 else 0.0
            after_batch_unrealized_pnl = after_batch_market_value - after_batch_total_cost
            after_batch_unrealized_pnl_pct = after_batch_unrealized_pnl / after_batch_total_cost * 100 if after_batch_total_cost > 0 else 0.0
            over_limit_after_batch = after_batch_stock_ratio > max_position_ratio if total_buy_amount > 0 and max_position_ratio > 0 else False

        plan.append(
            {
                "level_name": level_name,
                "batch_title": batch_titles.get(level_name, level_name),
                "batch_ratio": batch_ratio,
                "batch_budget": round(batch_budget, 0),
                "price": round(safe_price, 2),
                "lots": lots,
                "shares": shares,
                "amount": round(amount, 0),
                "total_buy_amount": round(total_buy_amount, 0),
                "total_buy_shares": int(total_buy_shares),
                "original_average_cost": round(original_avg_cost, 2),
                "original_stock_ratio": round(original_stock_ratio, 2),
                "original_cash": round(cash, 0),
                "cumulative_amount": round(cumulative_amount, 0),
                "cumulative_lots": int(cumulative_lots),
                "cumulative_shares": int(cumulative_shares),
                "cumulative_total_shares": int(cumulative_total_shares),
                "after_batch_total_shares": after_batch_total_shares,
                "after_batch_lots": int(after_batch_lots),
                "after_batch_odd_shares": int(after_batch_odd_shares),
                "after_batch_total_cost": round(after_batch_total_cost, 0),
                "after_batch_average_cost": round(after_batch_average_cost, 2),
                "after_batch_market_value": round(after_batch_market_value, 0),
                "after_batch_cash": round(after_batch_cash, 0),
                "after_batch_stock_ratio": round(after_batch_stock_ratio, 2),
                "after_batch_unrealized_pnl": round(after_batch_unrealized_pnl, 0),
                "after_batch_unrealized_pnl_pct": round(after_batch_unrealized_pnl_pct, 2),
                "over_limit_after_batch": over_limit_after_batch,
                "action_text": action_text,
            }
        )
    return plan


def build_trader_plan_v2(
    action_label: str,
    total_budget: float,
    observation_price: float,
    fair_price: float,
    conservative_price: float,
    disabled: bool,
) -> dict[str, object]:
    action = "WAIT" if action_label in ("觀察", "暫緩進場") else "BUY"
    summary = "暫緩" if action_label == "暫緩進場" else "觀察" if action_label == "觀察" else "買進"
    level_specs = [
        ("observation", observation_price, 0.20),
        ("fair", fair_price, 0.30),
        ("conservative", conservative_price, 0.50),
    ]
    levels: list[dict[str, object]] = []
    for level_type, price, ratio in level_specs:
        safe_price = max(0.0, float(price or 0.0))
        budget = 0.0 if disabled else max(0.0, float(total_budget or 0.0) * ratio)
        if safe_price <= 0 or budget <= 0:
            lots = 0
            shares = 0
            amount = 0.0
        else:
            lots = int(budget // (safe_price * 1000))
            remaining_budget = budget - lots * safe_price * 1000
            shares = int(remaining_budget // safe_price)
            shares = max(0, min(999, shares))
            amount = lots * safe_price * 1000 + shares * safe_price
        levels.append(
            {
                "type": level_type,
                "price": round(safe_price, 2),
                "budget": round(budget, 0),
                "lots": lots,
                "shares": shares,
                "amount": round(amount, 0),
                "ratio": ratio,
            }
        )
    return {
        "action": action,
        "summary": summary,
        "observation_price": round(float(observation_price or 0.0), 2),
        "fair_price": round(float(fair_price or 0.0), 2),
        "conservative_price": round(float(conservative_price or 0.0), 2),
        "levels": levels,
    }


def normalize_lots_and_shares(lots: int, shares: int) -> tuple[int, int]:
    lots = max(0, int(lots or 0))
    shares = max(0, int(shares or 0))
    if shares >= 1000:
        lots += shares // 1000
        shares = shares % 1000
    return lots, shares


def validate_decision_consistency(
    suggested_buy_lots: int,
    suggested_buy_shares: int,
    max_buy_lots: int,
    action_label: str,
    today_status: str,
    next_action: str,
    available_budget: float,
    cash: float,
    today_budget: float,
    suggested_price: float,
    max_stock_ratio: float,
    over_target_ratio: bool,
    trade_plan: list[dict[str, object]],
) -> tuple[int, int, str, str, str, list[dict[str, object]], list[str]]:
    conflict_flags: list[str] = []
    suggested_buy_lots, suggested_buy_shares = normalize_lots_and_shares(suggested_buy_lots, suggested_buy_shares)

    if over_target_ratio:
        if suggested_buy_lots > 0 or suggested_buy_shares > 0:
            conflict_flags.append("部位已超標但仍產生買進建議")
        suggested_buy_lots = 0
        suggested_buy_shares = 0
        today_status = "禁止加碼"
        action_label = "暫緩進場"
        next_action = "已達或超過部位上限，不新增部位。"

    if available_budget <= 0:
        suggested_buy_lots = 0
        suggested_buy_shares = 0

    if suggested_buy_lots > max_buy_lots:
        suggested_buy_lots = max(0, int(max_buy_lots))

    safe_suggested_price = max(0.0, float(suggested_price or 0.0))
    suggested_amount = suggested_buy_lots * safe_suggested_price * 1000 + suggested_buy_shares * safe_suggested_price
    if suggested_amount > today_budget + 1:
        conflict_flags.append("建議買進金額超過今日預算")
    if suggested_amount > cash + 1:
        conflict_flags.append("建議買進金額超過可用現金")

    total_amount = sum(float(row.get("amount", 0.0) or 0.0) for row in trade_plan)
    if total_amount > available_budget + 1:
        conflict_flags.append("分批策略總投入超過可用分批預算")
        overflow = total_amount - available_budget
        for row in reversed(trade_plan):
            amount = float(row.get("amount", 0.0) or 0.0)
            if overflow <= 0:
                break
            reduction = min(amount, overflow)
            row["amount"] = max(0.0, amount - reduction)
            overflow -= reduction

    for row in trade_plan:
        if float(row.get("amount", 0.0) or 0.0) > float(row.get("batch_budget", 0.0) or 0.0) + 1:
            conflict_flags.append("分批策略單批投入超過該批預算")
            row["amount"] = float(row.get("batch_budget", 0.0) or 0.0)
        if bool(row.get("over_limit_after_batch", False)) and float(row.get("amount", 0.0) or 0.0) > 0 and (suggested_buy_lots > 0 or suggested_buy_shares > 0):
            conflict_flags.append("建議買進後超過部位上限")


    if action_label in ("可加碼", "分批加碼", "積極加碼") and suggested_buy_lots == 0 and suggested_buy_shares == 0:
        conflict_flags.append("AI狀態與建議張數不一致")
        action_label = "觀察"
        next_action = "等待價格回落或提高預算。"

    return suggested_buy_lots, suggested_buy_shares, action_label, today_status, next_action, trade_plan, list(dict.fromkeys(conflict_flags))


def make_decision(
    tech: dict[str, float | None],
    volume: dict[str, object],
    market: dict[str, object],
    portfolio: dict[str, object],
    max_stock_ratio: float,
    current_price: float,
    today_budget: float | None = None,
    max_position_ratio: float | None = None,
    ticker: str = "",
    latest_close: float | None = None,
    valuation_quality_result: dict[str, object] | None = None,
) -> DecisionResult:
    market_score = int(market.get("score", 50))
    trend_score, trend_reasons = score_trend(tech)
    volume_score = int(volume.get("volume_score", 50))
    price_score, price_reasons = score_price_position(tech)
    portfolio_score, portfolio_reasons = score_portfolio_risk(portfolio, max_stock_ratio)
    conflict_flags: list[str] = []
    valuation_quality_result = dict(valuation_quality_result or {})
    valuation_quality_mode = str(valuation_quality_result.get("mode", "ETF") or "ETF")
    valuation_quality_sufficient = bool(valuation_quality_result.get("is_data_sufficient", False))
    valuation_quality_score = int(valuation_quality_result.get("final_score", 50) or 50)
    valuation_quality_score = clamp_score(valuation_quality_score if valuation_quality_sufficient else 50)
    valuation_label = str(valuation_quality_result.get("valuation_label", "") or "")
    quality_score_value = int(valuation_quality_result.get("quality_score", 50) or 50)
    event_risk_score = 50
    if not valuation_quality_sufficient:
        conflict_flags.append("估值與標的品質資料不足，該模組採中性 50 分。")

    max_position_ratio = float(max_position_ratio if max_position_ratio is not None else max_stock_ratio)
    current_stock_ratio = float(portfolio.get("current_stock_ratio", 0.0))
    holding_lots = float(portfolio.get("holding_lots", 0.0))
    average_cost = float(portfolio.get("average_cost", 0.0))
    stock_value = float(portfolio.get("market_value", 0.0))
    cash = float(portfolio.get("cash", 0.0))
    total_asset = stock_value + max(0.0, cash)
    stock_ratio = stock_value / total_asset if total_asset > 0 else 0.0
    max_position_decimal = max(0.0, max_position_ratio) / 100.0
    remaining_ratio = max(0.0, max_position_decimal - stock_ratio)
    max_allow_invest = remaining_ratio * total_asset
    if portfolio.get("negative_position"):
        max_allow_invest = 0.0
    today_budget = float(today_budget if today_budget is not None else portfolio.get("max_single_investment", cash))
    position_room = max_allow_invest
    available_budget = max(0.0, min(today_budget, cash, position_room))
    if portfolio.get("near_target_ratio") and current_price > 0:
        available_budget = min(available_budget, current_price * 1000 * 1.001425)
    profit_ratio = (current_price - average_cost) / average_cost if holding_lots > 0 and average_cost > 0 else 0.0
    cost_price_zone, cost_zone_lot_cap, cost_zone_reason = classify_cost_price_zone(current_price, average_cost)
    close_drop_pct = 0.0
    if latest_close is not None and latest_close > 0:
        close_drop_pct = (current_price - latest_close) / latest_close * 100
        if abs(close_drop_pct) >= 10:
            conflict_flags.append("手動價格與最新收盤價差異較大，請確認是否為即時盤中價格。")
    print("profit_ratio:", profit_ratio)
    print("current_stock_ratio:", current_stock_ratio)

    if valuation_quality_mode == "STOCK":
        total = clamp_score(
            valuation_quality_score * 0.25
            + price_score * 0.20
            + portfolio_score * 0.20
            + market_score * 0.15
            + trend_score * 0.10
            + volume_score * 0.05
            + event_risk_score * 0.05
        )
    else:
        total = clamp_score(
            portfolio_score * 0.30
            + price_score * 0.25
            + market_score * 0.20
            + valuation_quality_score * 0.15
            + trend_score * 0.05
            + volume_score * 0.05
        )

    max_buy_lots = int(portfolio.get("max_buy_lots", 0))
    rsi = tech.get("rsi")
    drawdown = tech.get("recent_high_drawdown") or 0.0
    ma20 = tech.get("ma20")
    ma60 = tech.get("ma60")
    observation_price = max(float(ma20) * 0.995, current_price * 0.97) if ma20 is not None and ma20 > 0 else current_price * 0.97
    reasonable_price = min(float(ma20) * 0.99, current_price * 0.95) if ma20 is not None and ma20 > 0 else current_price * 0.95
    conservative_price = min(float(ma60), current_price * 0.90) if ma60 is not None and ma60 > 0 else current_price * 0.90
    if not (current_price >= observation_price >= reasonable_price >= conservative_price):
        observation_price = current_price * 0.97
        reasonable_price = current_price * 0.95
        conservative_price = current_price * 0.90
        conflict_flags.append("價格區間異常，已改用保守預設折扣")
    aggressive_bid = observation_price
    reasonable_bid = reasonable_price
    conservative_bid = conservative_price
    suggested_bid = reasonable_price

    single_lot_cost = current_price * 1000 * 1.001425
    lot_price = reasonable_price * 1000
    budget_buy_lots = int(available_budget // single_lot_cost) if single_lot_cost > 0 else 0
    budget_remaining_cash = available_budget - budget_buy_lots * single_lot_cost if single_lot_cost > 0 else 0.0
    budget_buy_shares = int(budget_remaining_cash // reasonable_price) if reasonable_price > 0 else 0
    potential_lots_at_reasonable = int(available_budget // (reasonable_price * 1000)) if reasonable_price > 0 else 0

    entry_probability = clamp_score(
        total * 0.35
        + price_score * 0.20
        + volume_score * 0.15
        + market_score * 0.10
        + portfolio_score * 0.20
        - (8 if current_price > reasonable_bid else 0)
    )

    risk_score_value = clamp_score(
        100
        - portfolio_score
        + (10 if rsi is not None and rsi > 70 else 0)
        + (10 if volume_score < 45 else 0)
        + (10 if market_score < 45 else 0)
        + (20 if current_stock_ratio > max_stock_ratio and max_stock_ratio > 0 else 0)
        + (8 if profit_ratio > 0.10 else 0)
        - (20 if holding_lots > 0 and current_price < average_cost else 0)
        - (10 if latest_close is not None and close_drop_pct <= -5 else 0)
    )

    position_mode, position_mode_label = determine_position_mode(
        total,
        entry_probability,
        risk_score_value,
        price_score,
        current_stock_ratio,
        max_stock_ratio,
    )
    if int(valuation_quality_result.get("final_score", 50) or 50) < 40 and position_mode in ("AGGRESSIVE", "SCALE_IN", "PROBE"):
        position_mode = "HOLD"
        position_mode_label = "觀察（估值與品質分數偏低）"
    if valuation_label == "高估" and position_mode in ("AGGRESSIVE", "SCALE_IN", "PROBE"):
        position_mode = downgrade_position_mode(position_mode)
        position_mode_label = f"{position_mode}（估值高估，建議降一級）"
    if valuation_quality_mode == "STOCK" and quality_score_value < 35 and position_mode in ("AGGRESSIVE", "SCALE_IN"):
        position_mode = "HOLD"
        position_mode_label = "觀察（標的品質偏弱，不宜積極）"
    suggested_buy_lots = 0
    primary_reasons: list[str] = []

    over_position_limit = current_stock_ratio > max_position_ratio if max_position_ratio > 0 else False
    price_far_above_cost = holding_lots > 0 and profit_ratio > 0.10
    if holding_lots <= 0 and not over_position_limit and position_mode == "AVOID":
        position_mode = "HOLD"
        position_mode_label = "觀察（HOLD）"

    if over_position_limit:
        position_mode = "AVOID"
        position_mode_label = "暫緩進場（AVOID）"
        suggested_buy_lots = 0
        max_buy_lots = 0
        primary_reasons.append("目前持倉比例已接近或超過這檔設定上限。")
    elif price_far_above_cost:
        primary_reasons.append(f"目前價格高於平均成本 {profit_ratio * 100:.1f}%，不適合追價。")
        if total > 75 and max_buy_lots > 0:
            position_mode = "PROBE"
            position_mode_label = "試單模式（PROBE）"
            suggested_buy_lots = min(1, max_buy_lots)
            primary_reasons.append("AI 分數夠高，但因已有獲利部位，只允許小量試單。")
        else:
            position_mode = "HOLD"
            position_mode_label = "觀察（HOLD）"
            suggested_buy_lots = 0
            primary_reasons.append("分數尚未高到足以支持追高加碼。")
    elif holding_lots > 0 and average_cost > 0 and current_price < average_cost:
        primary_reasons.append(cost_zone_reason)
        if close_drop_pct <= -10:
            primary_reasons.append("目前價格較最新收盤價下跌超過 10%，屬極端回檔區，仍須分批。")
        elif close_drop_pct <= -5:
            primary_reasons.append("目前價格較最新收盤價下跌超過 5%，屬深度回檔區。")

        if cost_price_zone == "低於成本承接區":
            position_mode = "PROBE"
            position_mode_label = "小量承接（PROBE）"
            suggested_buy_lots = min(1, max_buy_lots)
        elif cost_price_zone == "積極分批區":
            position_mode = "SCALE_IN"
            position_mode_label = "積極分批（SCALE_IN）"
            suggested_buy_lots = min(2, max_buy_lots, max(1, cost_zone_lot_cap))
        else:
            position_mode = "AGGRESSIVE"
            position_mode_label = "攻擊區，分批加碼（AGGRESSIVE）"
            suggested_buy_lots = min(2, max_buy_lots)
    else:
        if position_mode == "AGGRESSIVE":
            suggested_buy_lots = min(2, max_buy_lots)
            primary_reasons.append("AI 分數與進場機率偏高，可考慮積極但分批執行。")
        elif position_mode == "SCALE_IN":
            suggested_buy_lots = min(1, max_buy_lots)
            primary_reasons.append("條件偏正向，適合分批加碼而非一次重倉。")
        elif position_mode == "PROBE":
            suggested_buy_lots = min(1, max_buy_lots)
            primary_reasons.append("訊號尚未全面確認，適合小量試單。")
        elif position_mode == "HOLD":
            suggested_buy_lots = 0
            primary_reasons.append("目前分數介於觀察區，等待價格或量能確認。")
        else:
            suggested_buy_lots = 0
            primary_reasons.append("AI 分數偏低，暫不建議進場。")

    if max_buy_lots <= 0 and suggested_buy_lots <= 0 and not over_position_limit:
        primary_reasons.append("可用現金、今日預算或持倉上限不足以支撐整股加碼。")
    if portfolio.get("near_target_ratio") and not over_position_limit:
        primary_reasons.append("目前已接近部位上限，系統已自動降低建議張數。")

    if rsi is not None and rsi > 75 and drawdown > -5 and suggested_buy_lots > 0:
        suggested_buy_lots = min(suggested_buy_lots, 1)
        primary_reasons.append("RSI 偏熱且價格接近高點，避免重倉追價。")

    max_buy_lots = min(max_buy_lots, budget_buy_lots)
    suggested_buy_lots = max(0, min(int(suggested_buy_lots), int(max_buy_lots)))
    action_label = action_from_mode(position_mode)
    if suggested_buy_lots <= 0 and action_label in ("積極加碼", "分批加碼", "試單"):
        action_label = "觀察" if not over_position_limit else "暫緩進場"
    if over_position_limit:
        action_label = "暫緩進場"

    suggested_buy_shares = 0
    if not over_position_limit and not price_far_above_cost and action_label in ("積極加碼", "分批加碼", "試單", "觀察"):
        if suggested_buy_lots <= 0 and budget_buy_shares > 0 and action_label in ("試單", "觀察"):
            suggested_buy_shares = budget_buy_shares
        elif suggested_buy_lots > 0:
            lot_spend = suggested_buy_lots * lot_price
            remaining_for_odd_lot = max(0.0, available_budget - lot_spend)
            suggested_buy_shares = int(remaining_for_odd_lot // reasonable_price) if reasonable_price > 0 else 0

    if available_budget <= 0 and not over_position_limit:
        primary_reasons.append("今日預算、現金或這檔部位上限不足，無法投入。")
    if position_room <= 0 and not any("部位上限" in reason for reason in primary_reasons):
        primary_reasons.append("已達或超過這檔部位上限。")

    if holding_lots > 0 and current_price < average_cost and suggested_buy_lots > 0:
        chase_today = "否"
    else:
        chase_today = "否" if suggested_buy_lots <= 0 or price_far_above_cost or over_position_limit else ("是" if total >= 75 else "否")

    if rsi is not None and rsi > 75 and drawdown > -5:
        market_temperature = "狂熱"
    elif rsi is not None and rsi > 68:
        market_temperature = "偏熱"
    elif drawdown <= -10:
        market_temperature = "冷卻"
    else:
        market_temperature = "中性"

    risk_bar = risk_label(risk_score_value)
    risk_level = risk_bar
    trade_plan = build_trade_plan(
        total_batch_budget=available_budget,
        prices=[
            ("觀察價", observation_price),
            ("合理價", reasonable_price),
            ("保守價", conservative_price),
        ],
        disabled=position_room <= 0,
        base_shares=float(portfolio.get("shares", holding_lots * 1000) or 0.0),
        base_cost_value=float(portfolio.get("cost_value", holding_lots * 1000 * average_cost) or 0.0),
        cash=cash,
        max_position_ratio=max_position_ratio,
        original_price=current_price,
    )
    trader_plan_v2 = build_trader_plan_v2(
        action_label=action_label,
        total_budget=available_budget,
        observation_price=observation_price,
        fair_price=reasonable_price,
        conservative_price=conservative_price,
        disabled=position_room <= 0,
    )
    first_batch = trade_plan[0] if trade_plan else {}
    second_batch = trade_plan[1] if len(trade_plan) > 1 else {}

    if position_room <= 0:
        today_status = "禁止加碼"
    elif holding_lots > 0 and current_price < average_cost and (suggested_buy_lots > 0 or suggested_buy_shares > 0):
        today_status = "現在可買"
    elif price_far_above_cost or action_label in ("觀察", "暫緩進場"):
        today_status = "等待回檔"
    elif current_price <= reasonable_price and (int(first_batch.get("lots", 0) or 0) > 0 or int(first_batch.get("shares", 0) or 0) > 0 or int(second_batch.get("lots", 0) or 0) > 0 or int(second_batch.get("shares", 0) or 0) > 0):
        today_status = "現在可買"
    else:
        today_status = "等待回檔"
    if today_status != "現在可買":
        suggested_buy_lots = 0
        suggested_buy_shares = 0

    next_action = build_next_action(
        today_status,
        action_label,
        suggested_buy_lots,
        suggested_buy_shares,
        suggested_bid,
        reasonable_price,
        trade_plan,
        position_room <= 0,
    )
    (
        suggested_buy_lots,
        suggested_buy_shares,
        action_label,
        today_status,
        next_action,
        trade_plan,
        validation_flags,
    ) = validate_decision_consistency(
        suggested_buy_lots=suggested_buy_lots,
        suggested_buy_shares=suggested_buy_shares,
        max_buy_lots=max_buy_lots,
        action_label=action_label,
        today_status=today_status,
        next_action=next_action,
        available_budget=available_budget,
        cash=cash,
        today_budget=today_budget,
        suggested_price=reasonable_price,
        max_stock_ratio=max_position_ratio,
        over_target_ratio=bool(portfolio.get("over_target_ratio", False)) or position_room <= 0,
        trade_plan=trade_plan,
    )
    conflict_flags = list(dict.fromkeys(conflict_flags + validation_flags))
    if holding_lots > 0 and current_price < average_cost and (action_label == "暫緩進場" or today_status == "禁止加碼" or risk_label(risk_score_value) == "高風險") and not over_position_limit:
        conflict_flags.append("決策衝突：目前價格低於平均成本，但系統判斷為高風險，請檢查模型規則。")

    print("final_suggested_lots:", suggested_buy_lots)

    scenario_fn = portfolio.get("scenario")
    if callable(scenario_fn) and suggested_buy_lots > 0:
        selected = scenario_fn(suggested_buy_lots)
        after_buy_average_cost = float(selected["加碼後平均成本"])
        after_buy_remaining_cash = float(selected["加碼後剩餘現金"])
        after_buy_stock_ratio = float(selected["加碼後股票資產比例"])
        over_position_limit_after_buy = bool(selected["over_limit"])
    elif suggested_buy_shares > 0 and current_price > 0:
        shares = holding_lots * 1000
        cost_value = shares * average_cost
        add_cost = suggested_buy_shares * current_price
        new_shares = shares + suggested_buy_shares
        after_buy_average_cost = (cost_value + add_cost) / new_shares if new_shares > 0 else 0.0
        after_buy_remaining_cash = cash - add_cost
        new_market_value = new_shares * current_price
        new_total_asset = new_market_value + max(0.0, after_buy_remaining_cash)
        after_buy_stock_ratio = new_market_value / new_total_asset * 100 if new_total_asset > 0 else 0.0
        over_position_limit_after_buy = after_buy_stock_ratio > max_position_ratio if max_position_ratio > 0 else False
    else:
        after_buy_average_cost = average_cost if holding_lots > 0 else 0.0
        after_buy_remaining_cash = float(portfolio.get("cash", 0.0))
        after_buy_stock_ratio = current_stock_ratio
        over_position_limit_after_buy = over_position_limit

    module_scores = {
        "市場背景分數": market_score,
        "趨勢技術分數": trend_score,
        "量能確認分數": volume_score,
        "價格位置分數": price_score,
        "個人持倉風控分數": portfolio_score,
        "估值與標的品質分數": valuation_quality_score,
    }
    if valuation_quality_mode == "STOCK":
        module_scores["事件風險分數"] = event_risk_score

    reasons = [f"分析模式：{tech.get('analysis_level', 'minimal')}"]
    if valuation_quality_result:
        reasons.append(f"估值與標的品質：{valuation_quality_result.get('investability_label', '資料不足')}，綜合分數 {valuation_quality_result.get('final_score', 50)}/100。")
        reasons.extend([str(reason) for reason in list(valuation_quality_result.get("warnings", []) or [])[:2]])
    reasons.append(f"價格區：{cost_price_zone}。")
    reasons.extend(primary_reasons)
    reasons.extend(trend_reasons)
    reasons.extend(price_reasons)
    reasons.extend(portfolio_reasons)
    reasons.append(str(market.get("text", "市場背景資料有限。")))
    reasons.append(f"量能判讀：{volume.get('volume_signal', '量能資料不足')}。")
    reasons.append(f"進場機率：{entry_probability}/100（{probability_text(entry_probability)}）。")
    immediate_plan = "立即執行" if today_status == "現在可買" and (suggested_buy_lots > 0 or suggested_buy_shares > 0) else "不立即掛單"
    immediate_lots = suggested_buy_lots if immediate_plan == "立即執行" else 0
    immediate_shares = suggested_buy_shares if immediate_plan == "立即執行" else 0
    immediate_price = reasonable_price if immediate_plan == "立即執行" else None

    trade_record = {
        "date": date.today().isoformat(),
        "ticker": ticker,
        "current_price": current_price,
        "total_score": total,
        "entry_probability": entry_probability,
        "today_status": today_status,
        "suggested_buy_lots": suggested_buy_lots,
        "suggested_buy_shares": suggested_buy_shares,
        "immediate_plan": immediate_plan,
        "immediate_lots": immediate_lots,
        "immediate_shares": immediate_shares,
        "immediate_price": immediate_price,
        "observation_price": observation_price,
        "fair_price": reasonable_price,
        "reasonable_price": reasonable_price,
        "conservative_price": conservative_price,
        "available_budget": available_budget,
        "unrealized_pnl": float(portfolio.get("unrealized_pnl", 0.0) or 0.0),
        "unrealized_pnl_pct": float(portfolio.get("unrealized_pnl_pct", 0.0) or 0.0),
        "current_stock_ratio": current_stock_ratio,
        "next_action": next_action,
    }

    return DecisionResult(
        total_score=total,
        recommendation=action_label,
        action_label=action_label,
        position_mode=position_mode,
        position_mode_label=position_mode_label,
        market_temperature=market_temperature,
        risk_level=risk_level,
        chase_today=chase_today,
        aggressive_bid=aggressive_bid,
        reasonable_bid=reasonable_bid,
        conservative_bid=conservative_bid,
        suggested_bid=suggested_bid,
        suggested_buy_lots=suggested_buy_lots,
        suggested_buy_shares=suggested_buy_shares,
        immediate_plan=immediate_plan,
        immediate_lots=immediate_lots,
        immediate_shares=immediate_shares,
        immediate_price=immediate_price,
        max_buy_lots=max_buy_lots,
        available_budget=available_budget,
        observation_price=observation_price,
        reasonable_price=reasonable_price,
        conservative_price=conservative_price,
        potential_lots_at_reasonable=potential_lots_at_reasonable,
        trade_plan=trade_plan,
        staged_plan=trade_plan,
        trader_plan_v2=trader_plan_v2,
        today_status=today_status,
        conflict_flags=conflict_flags,
        trade_record=trade_record,
        entry_probability=entry_probability,
        entry_probability_text=probability_text(entry_probability),
        risk_score=risk_score_value,
        risk_bar_label=risk_bar,
        after_buy_average_cost=after_buy_average_cost,
        after_buy_remaining_cash=after_buy_remaining_cash,
        after_buy_stock_ratio=after_buy_stock_ratio,
        over_position_limit_after_buy=over_position_limit_after_buy,
        next_action=next_action,
        primary_reasons=primary_reasons[:3],
        reasons=reasons,
        module_scores=module_scores,
        valuation_quality_result=valuation_quality_result,
    )
