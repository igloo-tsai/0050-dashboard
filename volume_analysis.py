from __future__ import annotations

import pandas as pd


def analyze_volume(data: pd.DataFrame, manual_volume: float | None = None) -> dict[str, float | str | bool | None]:
    if data is None or data.empty or "Volume" not in data or "Close" not in data:
        return {
            "latest_volume": None,
            "avg20_volume": None,
            "volume_ratio": None,
            "volume_signal": "量能資料不足",
            "volume_score": 50,
            "is_abnormal": False,
        }

    close = pd.to_numeric(data["Close"], errors="coerce").dropna()
    volume = pd.to_numeric(data["Volume"], errors="coerce").dropna()
    if close.empty or volume.empty:
        return {
            "latest_volume": None,
            "avg20_volume": None,
            "volume_ratio": None,
            "volume_signal": "量能資料不足",
            "volume_score": 50,
            "is_abnormal": False,
        }

    latest_volume = float(manual_volume) if manual_volume and manual_volume > 0 else float(volume.iloc[-1])
    avg20_volume = float(volume.tail(20).mean()) if len(volume.tail(20)) else 0.0
    volume_ratio = latest_volume / avg20_volume if avg20_volume else None
    latest_close = float(close.iloc[-1])
    previous_close = float(close.iloc[-2]) if len(close) >= 2 else latest_close
    price_up = latest_close >= previous_close
    volume_up = volume_ratio is not None and volume_ratio >= 1.0
    is_abnormal = volume_ratio is not None and volume_ratio >= 2.0

    if is_abnormal:
        signal = "異常放量"
        score = 65 if price_up else 45
    elif price_up and volume_up:
        signal = "價漲量增"
        score = 75
    elif price_up and not volume_up:
        signal = "價漲量縮"
        score = 55
    elif not price_up and volume_up:
        signal = "價跌量增"
        score = 35
    else:
        signal = "價跌量縮"
        score = 50

    return {
        "latest_volume": latest_volume,
        "avg20_volume": avg20_volume,
        "volume_ratio": volume_ratio,
        "volume_signal": signal,
        "volume_score": score,
        "is_abnormal": is_abnormal,
    }
