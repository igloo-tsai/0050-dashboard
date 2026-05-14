from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriceResolution:
    decision_price: float
    price_source: str
    price_label: str
    manual_override: bool
    warnings: list[str]
    errors: list[str]


def _positive(value: float | None) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0 else 0.0


def resolve_decision_price(
    *,
    latest_close: float | None,
    manual_price: float | None = None,
    use_manual_price: bool = False,
    confirm_extreme_price: bool = False,
    realtime_price: float | None = None,
) -> PriceResolution:
    warnings: list[str] = []
    errors: list[str] = []
    close = _positive(latest_close)
    realtime = _positive(realtime_price)
    manual = _positive(manual_price)
    auto_price = realtime or close
    auto_source = "latest_close" if close > 0 else "fallback_close"
    auto_label = "最新收盤價" if close > 0 else "備援收盤價"

    if not use_manual_price:
        if auto_price <= 0:
            return PriceResolution(0.0, "invalid_price", "異常價格已忽略", False, warnings, ["決策價格必須大於 0"])
        return PriceResolution(auto_price, auto_source, auto_label, False, warnings, errors)

    if manual <= 0:
        warnings.append("手動價格無效，已忽略並改用最新收盤價。")
        if auto_price <= 0:
            return PriceResolution(0.0, "invalid_price", "異常價格已忽略", True, warnings, ["決策價格必須大於 0"])
        return PriceResolution(auto_price, "invalid_price", "異常價格已忽略", True, warnings, errors)

    if close > 0:
        diff_pct = abs(manual - close) / close * 100
        if diff_pct > 20 and not confirm_extreme_price:
            warnings.append("價格與歷史價格差異過大，已暫停使用該價格，請確認資料來源。")
            return PriceResolution(close, "invalid_price", "異常價格已忽略", True, warnings, errors)
        if diff_pct > 10:
            warnings.append("手動價格與最新收盤價差異超過 10%，請確認是否為即時盤中價格。")

    return PriceResolution(manual, "manual_override", "盤中手動覆寫", True, warnings, errors)
