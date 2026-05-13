from __future__ import annotations

import pandas as pd


def clean_close(data: pd.DataFrame) -> pd.Series:
    if data is None or data.empty or "Close" not in data:
        return pd.Series(dtype="float64")
    try:
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce")
        return close.dropna()
    except Exception:
        return pd.Series(dtype="float64")


def moving_average(close: pd.Series, window: int) -> pd.Series:
    try:
        clean = close.dropna()
        if clean.empty:
            return pd.Series(dtype="float64")
        min_periods = min(len(clean), max(2, window // 3))
        return clean.rolling(window=window, min_periods=min_periods).mean()
    except Exception:
        return pd.Series(dtype="float64")


def latest_value(series: pd.Series) -> float | None:
    try:
        clean = series.dropna()
        if clean.empty:
            return None
        return float(clean.iloc[-1])
    except Exception:
        return None


def pct_change(first: float | None, last: float | None) -> float:
    if first is None or last is None or pd.isna(first) or pd.isna(last) or first == 0:
        return 0.0
    return (last / first - 1.0) * 100.0


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    try:
        clean = close.dropna()
        if len(clean) < 3:
            return pd.Series(dtype="float64")
        effective_window = min(window, max(3, len(clean) - 1))
        changes = clean.diff()
        gains = changes.clip(lower=0)
        losses = -changes.clip(upper=0)
        avg_gain = gains.ewm(alpha=1 / effective_window, min_periods=max(2, effective_window // 2), adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / effective_window, min_periods=max(2, effective_window // 2), adjust=False).mean()
        rs = avg_gain / avg_loss.mask(avg_loss == 0)
        return 100 - (100 / (1 + rs))
    except Exception:
        return pd.Series(dtype="float64")


def annual_volatility(close: pd.Series) -> float:
    try:
        returns = close.dropna().pct_change().dropna()
        if returns.empty:
            return 0.0
        return float(returns.std() * (252**0.5) * 100.0)
    except Exception:
        return 0.0


def max_drawdown(close: pd.Series) -> float:
    try:
        clean = close.dropna()
        if clean.empty:
            return 0.0
        running_high = clean.cummax()
        drawdown = clean / running_high - 1.0
        return float(drawdown.min() * 100.0)
    except Exception:
        return 0.0


def recent_high_drawdown(close: pd.Series, window: int = 120) -> tuple[float | None, float]:
    try:
        clean = close.dropna()
        if clean.empty:
            return None, 0.0
        recent_high = float(clean.tail(min(window, len(clean))).max())
        latest = float(clean.iloc[-1])
        return recent_high, pct_change(recent_high, latest)
    except Exception:
        return None, 0.0


def one_year_return(close: pd.Series) -> float:
    clean = close.dropna()
    if clean.empty:
        return 0.0
    one_year = clean[clean.index >= clean.index[-1] - pd.DateOffset(years=1)]
    if one_year.empty:
        return 0.0
    return pct_change(float(one_year.iloc[0]), float(clean.iloc[-1]))


def ytd_return(close: pd.Series) -> float:
    clean = close.dropna()
    if clean.empty:
        return 0.0
    current_year = clean[clean.index.year == clean.index[-1].year]
    if current_year.empty:
        return 0.0
    return pct_change(float(current_year.iloc[0]), float(clean.iloc[-1]))


def distance_from(reference: float | None, price: float | None) -> float | None:
    if reference is None or price is None or pd.isna(reference) or pd.isna(price) or reference == 0:
        return None
    return pct_change(reference, price)


def build_technical_snapshot(data: pd.DataFrame) -> dict[str, float | None]:
    close = clean_close(data)
    print("len(close):", len(close))
    if close.empty:
        return {}

    latest = float(close.iloc[-1])
    if len(close) >= 120:
        analysis_level = "full"
    elif len(close) >= 60:
        analysis_level = "mid"
    elif len(close) >= 20:
        analysis_level = "basic"
    else:
        analysis_level = "minimal"
    print("analysis mode:", analysis_level)

    data_points = len(close)
    ma20 = latest_value(moving_average(close, 20))
    ma60 = latest_value(moving_average(close, 60)) if data_points >= 20 else None
    ma120 = latest_value(moving_average(close, 120)) if data_points >= 60 else None
    recent_high, recent_drawdown = recent_high_drawdown(close)
    simple_return = pct_change(float(close.iloc[-2]), latest) if data_points >= 2 else 0.0
    return {
        "data_points": data_points,
        "analysis_level": analysis_level,
        "is_simplified": analysis_level != "full",
        "latest_price": latest,
        "ma20": ma20,
        "ma60": ma60,
        "ma120": ma120,
        "rsi": latest_value(rsi(close)),
        "recent_high": recent_high,
        "recent_high_drawdown": recent_drawdown,
        "one_year_return": one_year_return(close),
        "ytd_return": ytd_return(close),
        "annual_volatility": annual_volatility(close),
        "max_drawdown": max_drawdown(close),
        "simple_return": simple_return,
        "distance_ma20": distance_from(ma20, latest),
        "distance_ma60": distance_from(ma60, latest),
        "distance_ma120": distance_from(ma120, latest),
    }
