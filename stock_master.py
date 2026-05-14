from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_PATH = DATA_DIR / "stocks_cache.csv"

TWSE_ENDPOINT = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_ENDPOINT = "https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv"

MASTER_COLUMNS = ["ticker_code", "stock_code", "stock_name", "market"]


@dataclass(frozen=True)
class StockMasterResult:
    data: pd.DataFrame
    source: str
    warnings: list[str]
    errors: list[str]


def empty_stock_master() -> pd.DataFrame:
    return pd.DataFrame(columns=MASTER_COLUMNS)


def _read_url(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=8) as response:
        return response.read()


def _normalise_rows(rows: list[dict[str, object]], market: str) -> pd.DataFrame:
    normalised: list[dict[str, str]] = []
    for row in rows:
        code = str(
            row.get("公司代號")
            or row.get("股票代號")
            or row.get("證券代號")
            or row.get("Code")
            or ""
        ).strip()
        name = str(
            row.get("公司簡稱")
            or row.get("公司名稱")
            or row.get("證券名稱")
            or row.get("Name")
            or ""
        ).strip()
        if not code.isdigit() or len(code) != 4 or not name:
            continue
        normalised.append(
            {
                "ticker_code": f"{code}.TW",
                "stock_code": code,
                "stock_name": name,
                "market": market,
            }
        )
    if not normalised:
        return empty_stock_master()
    return pd.DataFrame(normalised, columns=MASTER_COLUMNS)


def _fetch_twse() -> pd.DataFrame:
    payload = _read_url(TWSE_ENDPOINT)
    rows = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(rows, list):
        return empty_stock_master()
    return _normalise_rows(rows, "TWSE")


def _fetch_tpex() -> pd.DataFrame:
    payload = _read_url(TPEX_ENDPOINT)
    frame = pd.read_csv(BytesIO(payload), dtype=str)
    return _normalise_rows(frame.fillna("").to_dict("records"), "TPEx")


def _write_cache(frame: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")


def _read_cache() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return empty_stock_master()
    frame = pd.read_csv(CACHE_PATH, dtype=str).fillna("")
    missing = [column for column in MASTER_COLUMNS if column not in frame.columns]
    if missing:
        return empty_stock_master()
    return frame[MASTER_COLUMNS].drop_duplicates(subset=["ticker_code"]).reset_index(drop=True)


@lru_cache(maxsize=2)
def load_stock_master(force_refresh: bool = False) -> StockMasterResult:
    warnings: list[str] = []
    errors: list[str] = []

    frames: list[pd.DataFrame] = []
    try:
        twse = _fetch_twse()
        if twse.empty:
            warnings.append("TWSE 股票主檔 API 回傳空資料。")
        else:
            frames.append(twse)
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        warnings.append(f"TWSE 股票主檔 API 失敗：{exc}")
    except Exception as exc:
        warnings.append(f"TWSE 股票主檔 API 失敗：{exc}")

    try:
        tpex = _fetch_tpex()
        if tpex.empty:
            warnings.append("TPEx 股票主檔 API 回傳空資料。")
        else:
            frames.append(tpex)
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        warnings.append(f"TPEx 股票主檔 API 失敗：{exc}")
    except Exception as exc:
        warnings.append(f"TPEx 股票主檔 API 失敗：{exc}")

    if frames:
        data = pd.concat(frames, ignore_index=True)
        data = data.drop_duplicates(subset=["ticker_code"]).sort_values(["stock_code", "market"]).reset_index(drop=True)
        _write_cache(data)
        return StockMasterResult(data, "api", warnings, errors)

    cached = _read_cache()
    if not cached.empty:
        warnings.append("股票主檔 API 失敗，已改用本機快取。")
        return StockMasterResult(cached, "cache", warnings, errors)

    errors.append("股票主檔 API 失敗")
    errors.append("快取不存在或格式不正確")
    errors.append("目前無法取得股票清單")
    return StockMasterResult(empty_stock_master(), "unavailable", warnings, errors)


def search_stocks(master: pd.DataFrame, query: str, limit: int = 30) -> pd.DataFrame:
    if master is None or master.empty:
        return empty_stock_master()
    text = str(query or "").strip()
    if not text:
        return empty_stock_master()

    data = master[MASTER_COLUMNS].fillna("").copy()
    upper = text.upper()
    code_match = data["stock_code"].str.contains(upper, case=False, na=False)
    ticker_match = data["ticker_code"].str.contains(upper, case=False, na=False)
    name_match = data["stock_name"].str.contains(text, case=False, na=False)
    result = data[code_match | ticker_match | name_match].copy()
    if result.empty:
        return empty_stock_master()

    result["rank"] = 5
    result.loc[result["stock_code"] == upper, "rank"] = 0
    result.loc[result["ticker_code"].str.upper() == upper, "rank"] = 0
    result.loc[result["stock_name"] == text, "rank"] = 1
    result.loc[result["stock_code"].str.startswith(upper), "rank"] = result["rank"].clip(upper=2)
    result.loc[result["stock_name"].str.startswith(text), "rank"] = result["rank"].clip(upper=3)
    return result.sort_values(["rank", "stock_code", "market"]).drop(columns=["rank"]).head(limit).reset_index(drop=True)
