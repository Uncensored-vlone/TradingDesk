import base64
import hashlib
import hmac
import json
import os
import random
import secrets
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_TIMEOUT = 12
DB_PATH = os.getenv("FRED_CACHE_DB", "output/fred_cache.db")
PBKDF2_ITERATIONS = 260000
FRED_MAX_RETRIES = 3
FRED_BACKOFF_BASE = 1.0
ADMIN_PASSWORD = "HEQ2024"

SERIES_MAP: Dict[str, str] = {
    "UNRATE": "US Unemployment Rate",
    "CPIAUCSL": "US CPI",
    "CPILFESL": "US Core CPI",
    "GDP": "US GDP",
    "FEDFUNDS": "US Fed Funds",
    "DGS10": "US 10Y Yield",
    "LRUN64TTJPM156S": "Japan Unemployment",
    "JPNCPIALLMINMEI": "Japan CPI",
    "JPNRGDPEXP": "Japan GDP",
    "IRSTCI01JPM156N": "Japan Policy Rate",
    "LRHUTTTTGBM156S": "UK Unemployment",
    "GBRCPIALLMINMEI": "UK CPI",
    "UKRGDPEXP": "UK GDP",
    "IRLTLT01GBM156N": "UK Policy Rate",
    "LRHUTTTTEZM156S": "Euro Area Unemployment",
    "CPALTT01EZM659N": "Euro Area CPI",
    "CLVMNACSCAB1GQEA19": "Euro Area GDP",
    "IRSTCI01EZM156N": "Euro Area Policy Rate",
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "UNRATE": 0.25,
    "FEDFUNDS": 0.20,
    "DGS10": 0.20,
    "GDP": 0.15,
    "CPIAUCSL": 0.10,
    "CPILFESL": 0.10,
}

CURRENCY_BASKETS: Dict[str, List[str]] = {
    "USD": ["UNRATE", "CPIAUCSL", "CPILFESL", "GDP", "FEDFUNDS", "DGS10"],
    "JPY": ["LRUN64TTJPM156S", "JPNCPIALLMINMEI", "JPNRGDPEXP", "IRSTCI01JPM156N"],
    "GBP": ["LRHUTTTTGBM156S", "GBRCPIALLMINMEI", "UKRGDPEXP", "IRLTLT01GBM156N"],
    "EUR": ["LRHUTTTTEZM156S", "CPALTT01EZM659N", "CLVMNACSCAB1GQEA19", "IRSTCI01EZM156N"],
}


@dataclass(frozen=True)
class SeriesReading:
    series_id: str
    name: str
    current: Optional[float]
    previous: Optional[float]
    change: Optional[float]
    pct_change: Optional[float]
    trend: str
    last_update: str
    stale: bool = False


@dataclass(frozen=True)
class MacroScoreResult:
    score: float
    bias: str
    confidence: str
    components: Dict[str, float]
    stale_count: int = 0
    missing_count: int = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def init_state() -> None:
    for k, v in {"trade_logs": [], "alerts": [], "historical_scores": [], "last_snapshot_date": None, "admin_unlocked": False}.items():
        if k not in st.session_state:
            st.session_state[k] = v


def hash_password(password: str, salt: Optional[str] = None, iterations: int = PBKDF2_ITERATIONS) -> str:
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${base64.b64encode(dk).decode('ascii').strip()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        if (stored_hash or "").count("$") != 3:
            return False
        algorithm, iterations, salt, _ = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        return hmac.compare_digest(hash_password(password, salt=salt, iterations=int(iterations)), stored_hash)
    except Exception:
        return False


def authenticate_admin(entered_password: str) -> bool:
    return entered_password == ADMIN_PASSWORD


def ensure_output_dir() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


@st.cache_resource
def get_http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "TradingDesk/1.0"})
    return s


def get_db() -> sqlite3.Connection:
    ensure_output_dir()
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db() -> None:
    conn = get_db()
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS fred_cache (series_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS snapshots (snapshot_date TEXT PRIMARY KEY, usd REAL, jpy REAL, gbp REAL, eur REAL, created_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS request_log (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, series_id TEXT NOT NULL, status_code INTEGER, source TEXT NOT NULL, retries INTEGER NOT NULL, success INTEGER NOT NULL)")
        conn.commit()
    finally:
        conn.close()


def log_request(series_id: str, status_code: Optional[int], source: str, retries: int, success: bool) -> None:
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO request_log (ts, series_id, status_code, source, retries, success) VALUES (?, ?, ?, ?, ?, ?)",
            (utc_now(), series_id, status_code, source, retries, 1 if success else 0),
        )
        conn.commit()
    finally:
        conn.close()


def save_cache(series_id: str, payload: Dict[str, Any]) -> None:
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO fred_cache (series_id, payload, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(series_id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
            (series_id, json.dumps(payload), utc_now()),
        )
        conn.commit()
    finally:
        conn.close()


def load_cache(series_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    try:
        row = conn.execute("SELECT payload FROM fred_cache WHERE series_id=?", (series_id,)).fetchone()
        return None if not row else json.loads(row["payload"])
    finally:
        conn.close()


def save_snapshot(scores: Dict[str, MacroScoreResult]) -> None:
    conn = get_db()
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        conn.execute(
            "INSERT INTO snapshots (snapshot_date, usd, jpy, gbp, eur, created_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(snapshot_date) DO UPDATE SET usd=excluded.usd, jpy=excluded.jpy, gbp=excluded.gbp, eur=excluded.eur, created_at=excluded.created_at",
            (today, scores["USD"].score, scores["JPY"].score, scores["GBP"].score, scores["EUR"].score, utc_now()),
        )
        conn.commit()
    finally:
        conn.close()


def load_snapshots() -> pd.DataFrame:
    conn = get_db()
    try:
        rows = conn.execute("SELECT snapshot_date, usd, jpy, gbp, eur FROM snapshots ORDER BY snapshot_date").fetchall()
        if not rows:
            return pd.DataFrame(columns=["date", "USD", "JPY", "GBP", "EUR"])
        df = pd.DataFrame([dict(r) for r in rows])
        return df.rename(columns={"snapshot_date": "date", "usd": "USD", "jpy": "JPY", "gbp": "GBP", "eur": "EUR"})
    finally:
        conn.close()


def fetch_fred_series_live(series_id: str, api_key: str) -> Dict[str, Any]:
    if not api_key:
        raise RuntimeError("FRED_API_KEY is missing.")
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json", "sort_order": "desc", "limit": 2}
    session = get_http_session()
    last_error = None
    for attempt in range(FRED_MAX_RETRIES):
        try:
            resp = session.get(FRED_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                raise RuntimeError("429")
            resp.raise_for_status()
            data = resp.json()
            save_cache(series_id, data)
            log_request(series_id, resp.status_code, "live", attempt, True)
            return data
        except Exception as e:
            last_error = e
            if attempt < FRED_MAX_RETRIES - 1:
                time.sleep(FRED_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.25))
    cached = load_cache(series_id)
    if cached:
        log_request(series_id, 0, "stale_cache", FRED_MAX_RETRIES, True)
        return cached
    log_request(series_id, None, "miss", FRED_MAX_RETRIES, False)
    raise RuntimeError(f"FRED fetch failed for {series_id}: {last_error}")


def observations_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(payload.get("observations", []))
    if df.empty:
        return df
    df = df.loc[:, ["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = df["value"].replace(".", pd.NA)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date", "value"]).sort_values("date", ascending=False)


def trend_label(change: Optional[float]) -> str:
    if change is None:
        return "▬ Flat"
    if change > 0:
        return "▲ Rising"
    if change < 0:
        return "▼ Falling"
    return "▬ Flat"


def latest_reading(series_id: str, api_key: str) -> Optional[SeriesReading]:
    try:
        payload = fetch_fred_series_live(series_id, api_key)
        stale = False
    except Exception:
        payload = load_cache(series_id)
        stale = True
        if not payload:
            return None
    df = observations_to_df(payload)
    if df.empty:
        return None
    current_row = df.iloc[0]
    previous_row = df.iloc[1] if len(df) > 1 else None
    current = float(current_row["value"]) if pd.notna(current_row["value"]) else None
    previous = float(previous_row["value"]) if previous_row is not None and pd.notna(previous_row["value"]) else None
    change = None if current is None or previous is None else current - previous
    pct_change = None if current is None or previous is None or previous == 0 else (change / abs(previous)) * 100.0
    return SeriesReading(series_id, SERIES_MAP[series_id], current, previous, change, pct_change, trend_label(change), str(current_row["date"].date()), stale)


def fetch_basket(basket: List[str], api_key: str) -> List[SeriesReading]:
    out: List[SeriesReading] = []
    for sid in basket:
        try:
            reading = latest_reading(sid, api_key)
            if reading is not None:
                out.append(reading)
        except Exception as e:
            st.warning(f"{SERIES_MAP.get(sid, sid)} skipped: {e}")
    return out


def score_series(r: SeriesReading) -> float:
    if r.series_id in {"UNRATE", "LRHUTTTTGBM156S", "LRHUTTTTEZM156S", "LRUN64TTJPM156S"}:
        return 50.0 if r.change is None else (90.0 if r.change < 0 else 10.0 if r.change > 0 else 50.0)
    if r.series_id in {"FEDFUNDS", "DGS10", "IRSTCI01JPM156N", "IRLTLT01GBM156N", "IRSTCI01EZM156N"}:
        return 50.0 if r.change is None else (80.0 if r.change > 0 else 20.0 if r.change < 0 else 50.0)
    if r.series_id in {"GDP", "JPNRGDPEXP", "UKRGDPEXP", "CLVMNACSCAB1GQEA19"}:
        return 50.0 if r.change is None else (75.0 if r.change > 0 else 25.0 if r.change < 0 else 50.0)
    if r.current is None:
        return 50.0
    return 80.0 if 2.0 <= r.current <= 3.5 else 45.0 if r.current < 2.0 else max(20.0, 75.0 - (r.current - 3.5) * 10.0)


def currency_score(currency: str, readings: List[SeriesReading]) -> MacroScoreResult:
    weights = DEFAULT_WEIGHTS if currency == "USD" else {sid: 1.0 / len(CURRENCY_BASKETS[currency]) for sid in CURRENCY_BASKETS[currency]}
    weighted_sum = 0.0
    weight_sum = 0.0
    stale_count = 0
    components: Dict[str, float] = {}
    for r in readings:
        w = float(weights.get(r.series_id, 0.0))
        if w > 0:
            s = score_series(r)
            components[r.series_id] = s
            weighted_sum += s * w
            weight_sum += w
            if r.stale:
                stale_count += 1
    missing_count = sum(1 for sid in CURRENCY_BASKETS[currency] if sid not in components)
    score = round(weighted_sum / weight_sum, 1) if weight_sum > 0 else 50.0
    if score >= 80:
        bias, conf = "Strong Bullish", "High"
    elif score >= 60:
        bias, conf = "Bullish", "Medium"
    elif score >= 40:
        bias, conf = "Neutral", "Low"
    elif score >= 20:
        bias, conf = "Bearish", "Medium"
    else:
        bias, conf = "Strong Bearish", "High"
    return MacroScoreResult(score, f"{currency} {bias}", conf, components, stale_count, missing_count)


def render_divergence_chart(hist_df: pd.DataFrame) -> None:
    if hist_df.empty:
        st.info("No historical snapshots yet.")
        return
    chart_df = hist_df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"])
    chart_df = chart_df.sort_values("date").set_index("date")
    divergence = pd.DataFrame(index=chart_df.index)
    divergence["USD-EUR"] = chart_df["USD"] - chart_df["EUR"]
    divergence["USD-GBP"] = chart_df["USD"] - chart_df["GBP"]
    divergence["USD-JPY"] = chart_df["USD"] - chart_df["JPY"]
    st.line_chart(divergence, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Multi-Currency Macro Dashboard", layout="wide")
    init_state()
    init_db()

    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        st.error("FRED_API_KEY is missing from environment variables.")
        return

    st.title("Multi-Currency Macro Dashboard")

    with st.sidebar:
        st.header("Weights")
        for sid, default in DEFAULT_WEIGHTS.items():
            st.slider(SERIES_MAP[sid], 0.0, 0.5, float(default), 0.01)

    with st.expander("Admin Panel", expanded=False):
        if st.session_state.admin_unlocked:
            st.success("Admin unlocked.")
            if st.button("Lock Admin"):
                st.session_state.admin_unlocked = False
                st.rerun()
        else:
            pwd = st.text_input("Admin password", type="password")
            if st.button("Unlock Admin"):
                if authenticate_admin(pwd):
                    st.session_state.admin_unlocked = True
                    st.success("Admin unlocked.")
                    st.rerun()
                else:
                    st.error("Invalid admin password.")

    baskets = {k: fetch_basket(v, api_key) for k, v in CURRENCY_BASKETS.items()}
    scores = {k: currency_score(k, baskets[k]) for k in CURRENCY_BASKETS}
    save_snapshot(scores)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("USD Score", f"{scores['USD'].score:.1f}")
    c2.metric("JPY Score", f"{scores['JPY'].score:.1f}")
    c3.metric("GBP Score", f"{scores['GBP'].score:.1f}")
    c4.metric("EUR Score", f"{scores['EUR'].score:.1f}")

    st.subheader("Historical Macro Scores")
    hist_df = load_snapshots()
    if hist_df.empty:
        st.info("No historical snapshots yet.")
    else:
        chart_df = hist_df.copy()
        chart_df["date"] = pd.to_datetime(chart_df["date"])
        chart_df = chart_df.sort_values("date").set_index("date")
        st.line_chart(chart_df[["USD", "JPY", "GBP", "EUR"]], width="stretch")
        st.subheader("Divergence")
        render_divergence_chart(hist_df)

    st.subheader("Basket Status")
    status_rows = []
    for currency, readings in baskets.items():
        for r in readings:
            status_rows.append(
                {
                    "Currency": currency,
                    "Series": r.name,
                    "ID": r.series_id,
                    "Current": r.current,
                    "Previous": r.previous,
                    "Change": r.change,
                    "Percent Change": r.pct_change,
                    "Trend": r.trend,
                    "Last Update": r.last_update,
                    "Stale": r.stale,
                }
            )
    if status_rows:
        st.dataframe(pd.DataFrame(status_rows), width="stretch", hide_index=True)
    else:
        st.info("No series loaded.")

if __name__ == "__main__":
    main()
