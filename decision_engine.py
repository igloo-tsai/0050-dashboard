from __future__ import annotations

from dataclasses import dataclass


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
    max_buy_lots: int
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
        reasons.append("可用現金或單次投入金額不足，無法安全加碼。")
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


def build_next_action(action_label: str, lots: int, suggested_bid: float, conservative_bid: float, over_limit: bool) -> str:
    if over_limit:
        return "今日不掛單，等待價格修正或提高現金部位。"
    if lots <= 0:
        return "今日不掛單，等待更佳價格或量能確認。"
    if lots == 1:
        return f"可掛 {suggested_bid:.2f}，最多 1 張；若跌至 {conservative_bid:.2f}，再評估第二筆。"
    if action_label == "積極加碼":
        return f"可掛 {suggested_bid:.2f}，最多 {lots} 張，仍建議分批成交。"
    return f"可掛 {suggested_bid:.2f}，最多 {lots} 張。"


def make_decision(
    tech: dict[str, float | None],
    volume: dict[str, object],
    market: dict[str, object],
    portfolio: dict[str, object],
    max_stock_ratio: float,
    current_price: float,
) -> DecisionResult:
    market_score = int(market.get("score", 50))
    trend_score, trend_reasons = score_trend(tech)
    volume_score = int(volume.get("volume_score", 50))
    price_score, price_reasons = score_price_position(tech)
    portfolio_score, portfolio_reasons = score_portfolio_risk(portfolio, max_stock_ratio)

    current_stock_ratio = float(portfolio.get("current_stock_ratio", 0.0))
    holding_lots = float(portfolio.get("holding_lots", 0.0))
    average_cost = float(portfolio.get("average_cost", 0.0))
    profit_ratio = (current_price - average_cost) / average_cost if holding_lots > 0 and average_cost > 0 else 0.0
    print("profit_ratio:", profit_ratio)
    print("current_stock_ratio:", current_stock_ratio)

    total = clamp_score(
        market_score * 0.20
        + trend_score * 0.25
        + volume_score * 0.15
        + price_score * 0.20
        + portfolio_score * 0.20
    )

    max_buy_lots = int(portfolio.get("max_buy_lots", 0))
    rsi = tech.get("rsi")
    drawdown = tech.get("recent_high_drawdown") or 0.0
    aggressive_bid = current_price * 0.995
    reasonable_bid = current_price * 0.985
    conservative_bid = current_price * 0.970
    suggested_bid = reasonable_bid

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
    )

    position_mode, position_mode_label = determine_position_mode(
        total,
        entry_probability,
        risk_score_value,
        price_score,
        current_stock_ratio,
        max_stock_ratio,
    )
    suggested_buy_lots = 0
    primary_reasons: list[str] = []

    over_position_limit = current_stock_ratio > max_stock_ratio if max_stock_ratio > 0 else False
    price_far_above_cost = holding_lots > 0 and profit_ratio > 0.10
    if holding_lots <= 0 and not over_position_limit and position_mode == "AVOID":
        position_mode = "HOLD"
        position_mode_label = "觀察（HOLD）"

    if over_position_limit:
        position_mode = "AVOID"
        position_mode_label = "暫緩進場（AVOID）"
        suggested_buy_lots = 0
        max_buy_lots = 0
        primary_reasons.append("目前持倉比例已超過目標上限。")
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
        primary_reasons.append("可用現金、單次投入金額或持倉上限不足以支撐安全加碼。")

    if rsi is not None and rsi > 75 and drawdown > -5 and suggested_buy_lots > 0:
        suggested_buy_lots = min(suggested_buy_lots, 1)
        primary_reasons.append("RSI 偏熱且價格接近高點，避免重倉追價。")

    suggested_buy_lots = max(0, min(int(suggested_buy_lots), int(max_buy_lots)))
    action_label = action_from_mode(position_mode)
    if suggested_buy_lots <= 0 and action_label in ("積極加碼", "分批加碼", "試單"):
        action_label = "觀察" if not over_position_limit else "暫緩進場"
    if over_position_limit:
        action_label = "暫緩進場"

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
    next_action = build_next_action(action_label, suggested_buy_lots, suggested_bid, conservative_bid, over_position_limit)

    print("final_suggested_lots:", suggested_buy_lots)

    scenario_fn = portfolio.get("scenario")
    if callable(scenario_fn) and suggested_buy_lots > 0:
        selected = scenario_fn(suggested_buy_lots)
        after_buy_average_cost = float(selected["加碼後平均成本"])
        after_buy_remaining_cash = float(selected["加碼後剩餘現金"])
        after_buy_stock_ratio = float(selected["加碼後股票資產比例"])
        over_position_limit_after_buy = bool(selected["over_limit"])
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
    }

    reasons = [f"分析模式：{tech.get('analysis_level', 'minimal')}"]
    reasons.extend(primary_reasons)
    reasons.extend(trend_reasons)
    reasons.extend(price_reasons)
    reasons.extend(portfolio_reasons)
    reasons.append(str(market.get("text", "市場背景資料有限。")))
    reasons.append(f"量能判讀：{volume.get('volume_signal', '量能資料不足')}。")
    reasons.append(f"進場機率：{entry_probability}/100（{probability_text(entry_probability)}）。")

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
        max_buy_lots=max_buy_lots,
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
    )
