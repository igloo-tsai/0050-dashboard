from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
from time import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_PATH = DATA_DIR / "stock_master_cache.json"
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

TWSE_ENDPOINT = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_ENDPOINT = "https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv"

MASTER_COLUMNS = ["ticker_code", "stock_code", "stock_name", "market"]

COMMON_STOCKS = [
    {"ticker_code": "0050.TW", "stock_code": "0050", "stock_name": "元大台灣50", "market": "TWSE"},
    {"ticker_code": "2330.TW", "stock_code": "2330", "stock_name": "台積電", "market": "TWSE"},
    {"ticker_code": "2317.TW", "stock_code": "2317", "stock_name": "鴻海", "market": "TWSE"},
    {"ticker_code": "2301.TW", "stock_code": "2301", "stock_name": "光寶科", "market": "TWSE"},
    {"ticker_code": "3037.TW", "stock_code": "3037", "stock_name": "欣興", "market": "TWSE"},
    {"ticker_code": "2454.TW", "stock_code": "2454", "stock_name": "聯發科", "market": "TWSE"},
    {"ticker_code": "2382.TW", "stock_code": "2382", "stock_name": "廣達", "market": "TWSE"},
    {"ticker_code": "3231.TW", "stock_code": "3231", "stock_name": "緯創", "market": "TWSE"},
]


@dataclass(frozen=True)
class StockMasterResult:
    data: pd.DataFrame
    source: str
    warnings: list[str]
    errors: list[str]
    debug_errors: list[str]


def empty_stock_master() -> pd.DataFrame:
    return pd.DataFrame(columns=MASTER_COLUMNS)


def common_stock_master() -> pd.DataFrame:
    return pd.DataFrame(COMMON_STOCKS, columns=MASTER_COLUMNS)


def _frame_from_records(records: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return empty_stock_master()
    for column in MASTER_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[MASTER_COLUMNS].fillna("").astype(str)
    return frame.drop_duplicates(subset=["ticker_code"]).reset_index(drop=True)


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
    return _frame_from_records(normalised)


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
    payload = {
        "created_at": time(),
        "records": frame[MASTER_COLUMNS].fillna("").to_dict("records"),
    }
    CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_cache(allow_expired: bool = False) -> tuple[pd.DataFrame, bool, str]:
    if not CACHE_PATH.exists():
        return empty_stock_master(), False, "cache missing"
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        created_at = float(payload.get("created_at", 0.0) or 0.0)
        expired = time() - created_at > CACHE_TTL_SECONDS
        if expired and not allow_expired:
            return empty_stock_master(), True, "cache expired"
        records = payload.get("records", [])
        if not isinstance(records, list):
            return empty_stock_master(), expired, "cache format invalid"
        frame = _frame_from_records(records)
        if frame.empty:
            return empty_stock_master(), expired, "cache empty"
        return frame, expired, ""
    except Exception as exc:
        return empty_stock_master(), False, f"cache read failed: {exc}"


def _load_from_api(debug_errors: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    try:
        twse = _fetch_twse()
        if not twse.empty:
            frames.append(twse)
        else:
            debug_errors.append("TWSE API returned empty data")
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        debug_errors.append(f"TWSE API failed: {exc}")
    except Exception as exc:
        debug_errors.append(f"TWSE API failed: {exc}")

    try:
        tpex = _fetch_tpex()
        if not tpex.empty:
            frames.append(tpex)
        else:
            debug_errors.append("TPEx API returned empty data")
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        debug_errors.append(f"TPEx API failed: {exc}")
    except Exception as exc:
        debug_errors.append(f"TPEx API failed: {exc}")

    if not frames:
        return empty_stock_master()
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["ticker_code"]).sort_values(["stock_code", "market"]).reset_index(drop=True)
    _write_cache(data)
    return data


@lru_cache(maxsize=2)
def load_stock_master(force_refresh: bool = False) -> StockMasterResult:
    warnings: list[str] = []
    errors: list[str] = []
    debug_errors: list[str] = []

    if not force_refresh:
        cached, expired, cache_error = _read_cache(allow_expired=False)
        if not cached.empty:
            return StockMasterResult(cached, "cache", warnings, errors, debug_errors)
        if cache_error:
            debug_errors.append(cache_error)

    api_data = _load_from_api(debug_errors)
    if not api_data.empty:
        return StockMasterResult(api_data, "api", warnings, errors, debug_errors)

    cached, expired, cache_error = _read_cache(allow_expired=True)
    if not cached.empty:
        warnings.append("股票清單連線失敗，已使用本機快取。")
        if expired:
            debug_errors.append("cache expired but used as fallback")
        if cache_error:
            debug_errors.append(cache_error)
        return StockMasterResult(cached, "cache_fallback", warnings, errors, debug_errors)

    fallback = common_stock_master()
    warnings.append("股票清單連線失敗，已使用內建常用股票清單。")
    if cache_error:
        debug_errors.append(cache_error)
    return StockMasterResult(fallback, "builtin_fallback", warnings, errors, debug_errors)


def search_stocks(master: pd.DataFrame, query: str, limit: int = 30) -> pd.DataFrame:
    text = str(query or "").strip()
    if not text:
        return empty_stock_master()
    if master is None or master.empty:
        master = common_stock_master()

    data = master[MASTER_COLUMNS].fillna("").copy()
    upper = text.upper().replace(".TW", "")
    code_match = data["stock_code"].str.contains(upper, case=False, na=False, regex=False)
    ticker_match = data["ticker_code"].str.contains(upper, case=False, na=False, regex=False)
    name_match = data["stock_name"].str.contains(text, case=False, na=False, regex=False)
    result = data[code_match | ticker_match | name_match].copy()
    if result.empty and upper.isdigit() and len(upper) == 4:
        result = _frame_from_records([{"ticker_code": f"{upper}.TW", "stock_code": upper, "stock_name": upper, "market": "TWSE"}])
    if result.empty:
        return empty_stock_master()

    result["rank"] = 5
    result.loc[result["stock_code"] == upper, "rank"] = 0
    result.loc[result["ticker_code"].str.upper() == f"{upper}.TW", "rank"] = 0
    result.loc[result["stock_name"] == text, "rank"] = 1
    result.loc[result["stock_code"].str.startswith(upper), "rank"] = result["rank"].clip(upper=2)
    result.loc[result["stock_name"].str.startswith(text), "rank"] = result["rank"].clip(upper=3)
    return result.sort_values(["rank", "stock_code", "market"]).drop(columns=["rank"]).head(limit).reset_index(drop=True)
