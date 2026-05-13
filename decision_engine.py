from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionResult:
    total_score: int
    recommendation: str
    market_temperature: str
    risk_level: str
    chase_today: str
    aggressive_bid: float
    reasonable_bid: float
    conservative_bid: float
    suggested_buy_lots: int
    max_buy_lots: int
    reasons: list[str]
    module_scores: dict[str, int]


def clamp_score(value: float) -> int:
    return int(max(0, min(100, round(value))))


def score_trend(tech: dict[str, float | None]) -> tuple[int, list[str]]:
    score = 50
    reasons = []
    price = tech.get("latest_price")
    if price is None:
        return 50, ["價格資料不足，趨勢分數採中性。"]

    used_indicators = 0
    for key, label, points in (("ma20", "MA20", 8), ("ma60", "MA60", 10), ("ma120", "MA120", 12)):
        ma = tech.get(key)
        if price is not None and ma is not None and price > ma:
            used_indicators += 1
            score += points
            reasons.append(f"價格站上 {label}。")
        elif price is not None and ma is not None:
            used_indicators += 1
            score -= 4
            reasons.append(f"價格低於 {label}。")
    rsi = tech.get("rsi")
    if rsi is not None and rsi > 72:
        score -= 12
        reasons.append("RSI 偏高，追價風險升高。")
    if used_indicators == 0:
        reasons.append("均線資料不足，趨勢分數維持基準值。")
    return clamp_score(score), reasons


def score_price_position(tech: dict[str, float | None]) -> tuple[int, list[str]]:
    drawdown = tech.get("recent_high_drawdown") or 0.0
    distance_ma120 = tech.get("distance_ma120")
    rsi = tech.get("rsi")
    score = 55
    reasons = []
    if drawdown <= -20:
        score += 25
        reasons.append("距近120日高點回撤超過20%。")
    elif drawdown <= -10:
        score += 15
        reasons.append("距近120日高點回撤超過10%。")
    elif drawdown > -3:
        score -= 18
        reasons.append("價格接近近期高點，降低追價分數。")
    if rsi is not None and rsi < 40:
        score += 10
        reasons.append("RSI 低於40，短線偏冷。")
    if distance_ma120 is not None and abs(distance_ma120) <= 3:
        score += 8
        reasons.append("價格接近 MA120 支撐區。")
    return clamp_score(score), reasons


def score_portfolio_risk(portfolio: dict[str, object], max_stock_ratio: float) -> tuple[int, list[str]]:
    ratio = float(portfolio.get("current_stock_ratio", 0.0))
    max_buy_lots = int(portfolio.get("max_buy_lots", 0))
    score = 75
    reasons = []
    if ratio >= max_stock_ratio:
        score = 25
        reasons.append("目前股票資產比例已達風控上限。")
    elif ratio >= max_stock_ratio * 0.85:
        score = 45
        reasons.append("股票資產比例接近風控上限。")
    else:
        reasons.append("目前持倉仍有風控空間。")
    if max_buy_lots <= 0:
        score = min(score, 35)
        reasons.append("可用現金或單次投入上限不足以加碼一張。")
    return clamp_score(score), reasons


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
    risk_score, risk_reasons = score_portfolio_risk(portfolio, max_stock_ratio)

    total = clamp_score(
        market_score * 0.20
        + trend_score * 0.25
        + volume_score * 0.15
        + price_score * 0.20
        + risk_score * 0.20
    )
    rsi = tech.get("rsi")
    drawdown = tech.get("recent_high_drawdown") or 0.0
    max_buy_lots = int(portfolio.get("max_buy_lots", 0))
    if total >= 75 and max_buy_lots > 0:
        recommendation = "建議加碼"
        suggested_buy_lots = min(3, max_buy_lots)
        risk_level = "低風險"
    elif total >= 55 and max_buy_lots > 0:
        recommendation = "持有觀察"
        suggested_buy_lots = min(1, max_buy_lots)
        risk_level = "中等風險"
    else:
        recommendation = "暫緩進場"
        suggested_buy_lots = 0
        risk_level = "高風險"

    if rsi is not None and rsi > 75 and drawdown > -5:
        market_temperature = "狂熱"
        chase_today = "否"
        suggested_buy_lots = 0
    elif rsi is not None and rsi > 68:
        market_temperature = "偏熱"
        chase_today = "否"
        suggested_buy_lots = min(suggested_buy_lots, 1)
    elif drawdown <= -10:
        market_temperature = "冷卻"
        chase_today = "可分批，不建議追高"
    else:
        market_temperature = "中性"
        chase_today = "否" if total < 70 else "僅限合理價"

    aggressive_bid = current_price * 0.995
    reasonable_bid = current_price * 0.985
    conservative_bid = current_price * 0.970

    module_scores = {
        "市場背景分數": market_score,
        "趨勢技術分數": trend_score,
        "量能確認分數": volume_score,
        "價格位置分數": price_score,
        "個人持倉風控分數": risk_score,
    }
    reasons = trend_reasons + price_reasons + risk_reasons
    reasons.insert(0, f"analysis mode: {tech.get('analysis_level', 'minimal')}")
    reasons.append(str(market.get("text", "市場背景資料有限。")))
    reasons.append(f"成交量判讀：{volume.get('volume_signal', '量能資料不足')}。")

    return DecisionResult(
        total_score=total,
        recommendation=recommendation,
        market_temperature=market_temperature,
        risk_level=risk_level,
        chase_today=chase_today,
        aggressive_bid=aggressive_bid,
        reasonable_bid=reasonable_bid,
        conservative_bid=conservative_bid,
        suggested_buy_lots=suggested_buy_lots,
        max_buy_lots=max_buy_lots,
        reasons=reasons,
        module_scores=module_scores,
    )
