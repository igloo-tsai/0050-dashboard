from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Any

import pandas as pd


ETF_PREFIXES = ("00", "006", "007", "008")


@dataclass
class ValuationQualityResult:
    mode: str
    valuation_score: int
    quality_score: int
    final_score: int
    valuation_label: str
    quality_label: str
    investability_label: str
    reasons: list[str]
    warnings: list[str]
    missing_fields: list[str]
    data_quality_score: int
    is_data_sufficient: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _score(value: float) -> int:
    return int(max(0, min(100, round(value))))


def _safe_close(price_data: pd.DataFrame | None) -> pd.Series:
    if price_data is None or price_data.empty or "Close" not in price_data:
        return pd.Series(dtype=float)
    close = price_data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return pd.to_numeric(close, errors="coerce").dropna()


def _percentile_score(close: pd.Series, window: int) -> tuple[float | None, list[str]]:
    missing: list[str] = []
    if close.empty:
        return None, ["price_history"]
    sample = close.tail(window).dropna()
    if sample.empty:
        return None, ["price_history"]
    latest = float(sample.iloc[-1])
    percentile = (sample <= latest).mean() * 100
    return float(percentile), missing


def _label_by_valuation(score: int, missing: bool = False) -> str:
    if missing:
        return "資料不足"
    if score >= 75:
        return "低估"
    if score >= 55:
        return "合理"
    if score >= 40:
        return "偏貴"
    return "高估"


def _label_by_quality(score: int, missing: bool = False) -> str:
    if missing:
        return "資料不足"
    if score >= 75:
        return "優良"
    if score >= 55:
        return "普通"
    if score >= 35:
        return "偏弱"
    return "危險"


def _investability(final_score: int, quality_score: int, data_quality_score: int) -> str:
    if data_quality_score < 50:
        return "資料不足，僅供參考"
    if quality_score < 35:
        return "不建議"
    if final_score >= 75:
        return "值得長期持有"
    if final_score >= 55:
        return "可觀察"
    if final_score >= 40:
        return "僅短線"
    return "不建議"


@lru_cache(maxsize=128)
def fetch_fundamental_data(ticker: str) -> dict[str, Any]:
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        info = stock.info or {}
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        dividends = stock.dividends
        return {
            "info": info if isinstance(info, dict) else {},
            "financials": financials if isinstance(financials, pd.DataFrame) else pd.DataFrame(),
            "balance_sheet": balance_sheet if isinstance(balance_sheet, pd.DataFrame) else pd.DataFrame(),
            "cashflow": cashflow if isinstance(cashflow, pd.DataFrame) else pd.DataFrame(),
            "dividends": dividends if isinstance(dividends, pd.Series) else pd.Series(dtype=float),
        }
    except Exception as exc:
        return {"info": {}, "financials": pd.DataFrame(), "balance_sheet": pd.DataFrame(), "cashflow": pd.DataFrame(), "dividends": pd.Series(dtype=float), "error": str(exc)}


def is_etf_ticker(ticker: str) -> bool:
    code = str(ticker or "").upper().replace(".TW", "")
    return code.startswith(ETF_PREFIXES)


def evaluate_etf_valuation_quality(
    ticker: str,
    price_data: pd.DataFrame | None,
    market_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    close = _safe_close(price_data)
    reasons: list[str] = []
    warnings: list[str] = []
    missing_fields: list[str] = []

    percentile_1y, missing = _percentile_score(close, 252)
    missing_fields.extend(missing)
    percentile_2y, missing = _percentile_score(close, 504)
    missing_fields.extend([field for field in missing if field not in missing_fields])

    if close.empty:
        valuation_score = 50
        reasons.append("價格資料不足，估值位置採中性分數。")
    else:
        latest = float(close.iloc[-1])
        one_year = close.tail(252)
        high_1y = float(one_year.max()) if not one_year.empty else latest
        low_1y = float(one_year.min()) if not one_year.empty else latest
        drawdown_high = (latest / high_1y - 1) * 100 if high_1y > 0 else 0.0
        distance_low = (latest / low_1y - 1) * 100 if low_1y > 0 else 0.0
        percentile = percentile_2y if percentile_2y is not None else percentile_1y
        if percentile is None:
            valuation_score = 50
        elif percentile >= 85:
            valuation_score = 32
            reasons.append("價格位於歷史高分位，估值偏貴。")
        elif percentile >= 65:
            valuation_score = 48
            reasons.append("價格位於偏高區間，分批需保守。")
        elif percentile >= 35:
            valuation_score = 62
            reasons.append("價格位於中性區間，適合觀察分批。")
        else:
            valuation_score = 78
            reasons.append("價格位於相對低分位，長期分批吸引力提高。")
        reasons.append(f"近一年高點回撤約 {drawdown_high:.1f}%，距離低點約 {distance_low:.1f}%。")

    market_score = _number((market_data or {}).get("score"), 50) or 50
    market_component = 70 if market_score >= 60 else 50 if market_score >= 45 else 38
    if market_score >= 70:
        warnings.append("市場背景偏熱，ETF 分批仍需控制節奏。")

    dividend_score = 50
    missing_fields.append("dividend_yield")
    warnings.append("殖利率資料不足，殖利率吸引力採中性分數。")

    concentration_score = 58
    warnings.append("0050/台股大型 ETF 可能受台積電與半導體權重影響，請留意集中度風險。")

    valuation_score = _score(valuation_score * 0.70 + market_component * 0.30)
    quality_score = _score(market_component * 0.30 + dividend_score * 0.25 + concentration_score * 0.45)
    final_score = _score(valuation_score * 0.65 + quality_score * 0.35)
    data_quality_score = _score(70 if not close.empty else 25)
    if missing_fields:
        data_quality_score = _score(data_quality_score - min(25, len(set(missing_fields)) * 8))

    return ValuationQualityResult(
        mode="ETF",
        valuation_score=valuation_score,
        quality_score=quality_score,
        final_score=final_score,
        valuation_label=_label_by_valuation(valuation_score, close.empty),
        quality_label=_label_by_quality(quality_score, False),
        investability_label=_investability(final_score, quality_score, data_quality_score),
        reasons=reasons[:6],
        warnings=warnings,
        missing_fields=sorted(set(missing_fields)),
        data_quality_score=data_quality_score,
        is_data_sufficient=data_quality_score >= 50,
    ).to_dict()


def _latest_row_value(frame: pd.DataFrame, candidates: list[str]) -> float | None:
    if frame is None or frame.empty:
        return None
    for name in candidates:
        if name in frame.index:
            series = pd.to_numeric(frame.loc[name], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[0])
    return None


def evaluate_stock_valuation_quality(
    ticker: str,
    price_data: pd.DataFrame | None,
    fundamental_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = fundamental_data if fundamental_data is not None else fetch_fundamental_data(ticker)
    info = data.get("info", {}) if isinstance(data, dict) else {}
    financials = data.get("financials", pd.DataFrame()) if isinstance(data, dict) else pd.DataFrame()
    balance_sheet = data.get("balance_sheet", pd.DataFrame()) if isinstance(data, dict) else pd.DataFrame()
    cashflow = data.get("cashflow", pd.DataFrame()) if isinstance(data, dict) else pd.DataFrame()
    dividends = data.get("dividends", pd.Series(dtype=float)) if isinstance(data, dict) else pd.Series(dtype=float)

    reasons: list[str] = []
    warnings: list[str] = []
    missing_fields: list[str] = []

    per = _number(info.get("trailingPE"))
    pbr = _number(info.get("priceToBook"))
    dividend_yield = _number(info.get("dividendYield"))
    roe = _number(info.get("returnOnEquity"))
    gross_margin = _number(info.get("grossMargins"))
    operating_margin = _number(info.get("operatingMargins"))
    net_margin = _number(info.get("profitMargins"))
    eps = _number(info.get("trailingEps"))
    revenue_yoy = _number(info.get("revenueGrowth"))
    earnings_yoy = _number(info.get("earningsGrowth"))
    debt_to_equity = _number(info.get("debtToEquity"))
    current_ratio = _number(info.get("currentRatio"))
    free_cash_flow = _number(info.get("freeCashflow"))
    operating_cash_flow = _number(info.get("operatingCashflow"))
    payout_ratio = _number(info.get("payoutRatio"))

    field_values = {
        "PER": per,
        "PBR": pbr,
        "dividend_yield": dividend_yield,
        "ROE": roe,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "EPS": eps,
        "revenue_yoy": revenue_yoy,
        "EPS_yoy": earnings_yoy,
        "debt_ratio": debt_to_equity,
        "current_ratio": current_ratio,
        "free_cash_flow": free_cash_flow,
        "operating_cash_flow": operating_cash_flow,
        "dividend_payout_ratio": payout_ratio,
    }
    missing_fields.extend([key for key, value in field_values.items() if value is None])

    if operating_cash_flow is None:
        operating_cash_flow = _latest_row_value(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    if free_cash_flow is None:
        free_cash_flow = _latest_row_value(cashflow, ["Free Cash Flow"])

    valuation = 55
    if per is None or per <= 0:
        valuation -= 5
        warnings.append("PER 缺失或為負，估值分數採保守中性。")
    elif per <= 12:
        valuation += 15
        reasons.append("PER 偏低，但仍需搭配成長與現金流確認。")
    elif per <= 22:
        valuation += 8
        reasons.append("PER 位於相對合理區間。")
    elif per <= 35:
        valuation -= 8
        reasons.append("PER 偏高，需要成長支撐。")
    else:
        valuation -= 20
        reasons.append("PER 明顯偏高，估值風險提高。")

    if pbr is not None:
        if pbr <= 1.5:
            valuation += 8
        elif pbr > 5:
            valuation -= 10
            reasons.append("PBR 偏高，需留意淨值評價風險。")

    if dividend_yield is not None:
        dy = dividend_yield * 100 if dividend_yield < 1 else dividend_yield
        if dy >= 4:
            valuation += 6
        elif dy < 1:
            valuation -= 4
    else:
        warnings.append("殖利率資料缺失，股東回報採中性。")

    profitability = 50
    for metric, good, weak in ((roe, 0.15, 0.08), (gross_margin, 0.30, 0.15), (operating_margin, 0.15, 0.06), (net_margin, 0.10, 0.03)):
        if metric is None:
            continue
        if metric >= good:
            profitability += 8
        elif metric < weak:
            profitability -= 8
    if eps is not None and eps <= 0:
        profitability -= 18
        reasons.append("EPS 為負，基本面品質需保守看待。")

    growth = 50
    for metric in (revenue_yoy, earnings_yoy):
        if metric is None:
            continue
        if metric >= 0.10:
            growth += 10
        elif metric < 0:
            growth -= 12
    if (revenue_yoy is not None and revenue_yoy < 0) or (earnings_yoy is not None and earnings_yoy < 0):
        warnings.append("營收或 EPS 成長為負，避免把低 PER 誤判為便宜。")
        if per is not None and per <= 12:
            valuation -= 12
            reasons.append("低 PER 但成長衰退，存在價值陷阱風險。")

    safety = 55
    if debt_to_equity is not None and debt_to_equity > 150:
        safety -= 18
        reasons.append("負債水位偏高，財務安全性扣分。")
    if current_ratio is not None and current_ratio < 1:
        safety -= 10
    if free_cash_flow is not None and free_cash_flow < 0:
        safety -= 12
    if operating_cash_flow is not None and operating_cash_flow < 0:
        safety -= 18
        warnings.append("營業現金流為負，品質分數需保守。")
    if (operating_cash_flow is not None and operating_cash_flow < 0) and (debt_to_equity is not None and debt_to_equity > 100):
        safety -= 18
        warnings.append("現金流為負且負債偏高，財務風險升高。")

    shareholder_return = 50
    if dividend_yield is not None:
        dy = dividend_yield * 100 if dividend_yield < 1 else dividend_yield
        shareholder_return += 8 if dy >= 3 else -4 if dy < 1 else 2
    if payout_ratio is not None and payout_ratio > 0.9:
        shareholder_return -= 8
        warnings.append("配息率偏高，高殖利率不一定代表品質佳。")
    if dividends is not None and not dividends.empty:
        shareholder_return += 5

    if per is not None and per > 30 and (earnings_yoy is None or earnings_yoy < 0.05):
        valuation -= 12
        warnings.append("PER 偏高且 EPS 成長不足，估值分數下修。")

    valuation_score = _score(valuation)
    quality_score = _score(profitability * 0.35 + growth * 0.20 + safety * 0.30 + shareholder_return * 0.15)
    final_score = _score(quality_score * 0.55 + valuation_score * 0.45)
    available_fields = len(field_values) - len(set(missing_fields))
    data_quality_score = _score(available_fields / max(1, len(field_values)) * 100)
    if data.get("error"):
        warnings.append(f"基本面資料取得失敗：{data.get('error')}")
        data_quality_score = min(data_quality_score, 30)

    if data_quality_score < 50:
        final_score = _score(final_score * 0.40 + 50 * 0.60)
        warnings.append("基本面資料不足，估值品質模組僅供參考。")

    if not reasons:
        reasons.append("可用基本面資料有限，採中性保守評估。")

    return ValuationQualityResult(
        mode="STOCK",
        valuation_score=valuation_score,
        quality_score=quality_score,
        final_score=final_score,
        valuation_label=_label_by_valuation(valuation_score, data_quality_score < 25),
        quality_label=_label_by_quality(quality_score, data_quality_score < 25),
        investability_label=_investability(final_score, quality_score, data_quality_score),
        reasons=reasons[:6],
        warnings=warnings,
        missing_fields=sorted(set(missing_fields)),
        data_quality_score=data_quality_score,
        is_data_sufficient=data_quality_score >= 50,
    ).to_dict()


def evaluate_valuation_quality(
    ticker: str,
    price_data: pd.DataFrame | None,
    market_data: dict[str, Any] | None = None,
    fundamental_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if is_etf_ticker(ticker):
        return evaluate_etf_valuation_quality(ticker, price_data, market_data)
    return evaluate_stock_valuation_quality(ticker, price_data, fundamental_data)
