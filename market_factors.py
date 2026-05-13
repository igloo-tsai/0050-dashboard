from __future__ import annotations

from datetime import date

import pandas as pd

from data_service import fetch_many_price_data
from technical_analysis import build_technical_snapshot, clean_close


BACKGROUND_TICKERS = ("QQQ", "SOXX", "SPY", "^VIX", "^TWII", "006208.TW")


def classify_trend(data: pd.DataFrame) -> str | None:
    snapshot = build_technical_snapshot(data)
    if not snapshot:
        return None
    price = snapshot.get("latest_price")
    ma20 = snapshot.get("ma20")
    ma60 = snapshot.get("ma60")
    rsi = snapshot.get("rsi")
    if price is None:
        return None
    if rsi is not None and rsi >= 72:
        return "過熱"
    if ma20 is not None and ma60 is not None and price > ma20 and price > ma60:
        return "偏多"
    if ma60 is not None and price < ma60:
        return "偏弱"
    return "中性"


def vix_status(vix_data: pd.DataFrame) -> tuple[str, int]:
    close = clean_close(vix_data)
    if close.empty:
        return "正常", 0
    latest = float(close.iloc[-1])
    if latest < 12:
        return "過度樂觀", -8
    if latest >= 25:
        return "恐慌升溫", 6
    return "正常", 0


def get_market_background(start: date, end: date) -> dict[str, object]:
    data = fetch_many_price_data(BACKGROUND_TICKERS, start, end)
    missing = False

    qqq = classify_trend(data.get("QQQ", pd.DataFrame()))
    soxx = classify_trend(data.get("SOXX", pd.DataFrame()))
    spy = classify_trend(data.get("SPY", pd.DataFrame()))
    twii = classify_trend(data.get("^TWII", pd.DataFrame()))
    taiwan_peer = classify_trend(data.get("006208.TW", pd.DataFrame()))
    vix, vix_points = vix_status(data.get("^VIX", pd.DataFrame()))

    if qqq is None:
        qqq, missing = "中性", True
    if soxx is None:
        soxx, missing = "中性", True
    if spy is None:
        spy, missing = "中性", True
    if twii is None:
        twii, missing = "中性", True
    if taiwan_peer is None:
        taiwan_peer, missing = "正常", True

    score = 60
    for status in (qqq, soxx, spy, twii):
        if status == "偏多":
            score += 5
        elif status == "偏弱":
            score -= 8
        elif status == "過熱":
            score -= 6

    if taiwan_peer == "偏多":
        peer_summary = "正常"
        score += 4
    elif taiwan_peer == "過熱":
        peer_summary = "偏熱"
        score -= 8
    elif taiwan_peer == "偏弱":
        peer_summary = "偏弱"
        score -= 8
    else:
        peer_summary = "正常"

    score += vix_points
    score = max(0, min(100, score))

    if spy == "偏多":
        risk_appetite = "偏高"
    elif spy == "偏弱":
        risk_appetite = "偏低"
    else:
        risk_appetite = "中性"

    return {
        "score": score,
        "summary": {
            "美股科技趨勢": qqq,
            "半導體風向": soxx,
            "大盤風險偏好": risk_appetite,
            "台股大型股比較": peer_summary,
            "VIX風險訊號": vix,
        },
        "missing": missing,
        "text": f"市場背景分數 {score}/100，科技趨勢為{qqq}，半導體風向為{soxx}，VIX訊號為{vix}。",
    }
