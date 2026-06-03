import base64
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_TIMEOUT = 15
CACHE_TTL = 3600
PBKDF2_ITERATIONS = 260000

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


@dataclass(frozen=True)
class MacroScoreResult:
    score: float
    bias: str
    confidence: str
    components: Dict[str, float]


def init_state() -> None:
    defaults = {
        "trade_logs": [],
        "alerts": [],
        "historical_scores": [],
        "last_snapshot_date": None,
        "admin_unlocked": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def pbkdf2_hash_password(password: str, salt: Optional[str] = None, iterations: int = PBKDF2_ITERATIONS) -> str:
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    )
    b64 = base64.b64encode(dk).decode("ascii").strip()
    return f"pbkdf2_sha256${iterations}${salt}${b64}"


def pbkdf2_verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations, salt, _ = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        candidate = pbkdf2_hash_password(password, salt=salt, iterations=int(iterations))
        return hmac.compare_digest(candidate, password_hash)
    except Exception:
        return False


def get_admin_password_hash() -> str:
    return os.getenv("ADMIN_PASSWORD_HASH", "")


def authenticate_admin(entered_password: str) -> bool:
    password_hash = get_admin_password_hash()
    if not password_hash:
        return False
    return pbkdf2_verify_password(entered_password, password_hash)


@st.cache_resource
def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "TradingDesk/1.0"})
    return s


@st.cache_data(ttl=CACHE_TTL)
def get_fred_series(series_id: str, api_key: str) -> Dict[str, Any]:
    if not api_key:
        raise RuntimeError("FRED_API_KEY is missing from environment variables.")
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 2,
    }
    session = get_session()
    response = session.get(FRED_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
    if response.status_code == 429:
        raise RuntimeError("FRED rate limit reached. Please try again later.")
    response.raise_for_status()
    return response.json()


def observations_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(payload.get("observations", []))
    if df.empty:
        return df
    df = df.loc[:, ["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = df["value"].replace(".", pd.NA)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date", ascending=False)
    return df


def trend_label(change: Optional[float]) -> str:
    if change is None:
        return "▬ Flat"
    if change > 0:
        return "▲ Rising"
    if change < 0:
        return "▼ Falling"
    return "▬ Flat"


def latest_reading(series_id: str, api_key: str) -> Optional[SeriesReading]:
    payload = get_fred_series(series_id, api_key)
    df = observations_to_df(payload)
    if df.empty:
        return None
    current_row = df.iloc[0]
    previous_row = df.iloc[1] if len(df) > 1 else None
    current = float(current_row["value"]) if pd.notna(current_row["value"]) else None
    previous = float(previous_row["value"]) if previous_row is not None and pd.notna(previous_row["value"]) else None
    change = None if current is None or previous is None else current - previous
    pct_change = None if current is None or previous is None or previous == 0 else (change / abs(previous)) * 100.0
    return SeriesReading(
        series_id=series_id,
        name=SERIES_MAP[series_id],
        current=current,
        previous=previous,
        change=change,
        pct_change=pct_change,
        trend=trend_label(change),
        last_update=str(current_row["date"].date()),
    )


def fetch_basket(basket: List[str], api_key: str) -> List[SeriesReading]:
    out: List[SeriesReading] = []
    for sid in basket:
        try:
            r = latest_reading(sid, api_key)
            if r is not None:
                out.append(r)
        except Exception as e:
            st.warning(f"{SERIES_MAP.get(sid, sid)} could not be loaded: {e}")
    return out


def score_series(r: SeriesReading) -> float:
    if r.series_id in {"UNRATE", "LRHUTTTTGBM156S", "LRHUTTTTEZM156S", "LRUN64TTJPM156S"}:
        if r.change is None:
            return 50.0
        return 90.0 if r.change < 0 else 10.0 if r.change > 0 else 50.0
    if r.series_id in {"FEDFUNDS", "DGS10", "IRSTCI01JPM156N", "IRLTLT01GBM156N", "IRSTCI01EZM156N"}:
        if r.change is None:
            return 50.0
        return 80.0 if r.change > 0 else 20.0 if r.change < 0 else 50.0
    if r.series_id in {"GDP", "JPNRGDPEXP", "UKRGDPEXP", "CLVMNACSCAB1GQEA19"}:
        if r.change is None:
            return 50.0
        return 75.0 if r.change > 0 else 25.0 if r.change < 0 else 50.0
    if r.current is None:
        return 50.0
    if 2.0 <= r.current <= 3.5:
        return 80.0
    if r.current < 2.0:
        return 45.0
    return max(20.0, 75.0 - (r.current - 3.5) * 10.0)


def currency_score(currency: str, readings: List[SeriesReading]) -> MacroScoreResult:
    if currency == "USD":
        weights = DEFAULT_WEIGHTS
    else:
        weights = {sid: 1.0 / max(1, len(CURRENCY_BASKETS[currency])) for sid in CURRENCY_BASKETS[currency]}
    components: Dict[str, float] = {}
    weighted_sum = 0.0
    weight_sum = 0.0
    for r in readings:
        s = score_series(r)
        w = float(weights.get(r.series_id, 0.0))
        if w > 0:
            components[r.series_id] = s
            weighted_sum += s * w
            weight_sum += w
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
    return MacroScoreResult(score, f"{currency} {bias}", conf, components)


def session_bias(score: float, session_name: str) -> str:
    if session_name == "Asia":
        return "Continuation only" if score >= 80 or score <= 20 else "Range / wait for London"
    if session_name == "London":
        return "Prefer currency strength" if score >= 60 else "Prefer currency weakness" if score <= 40 else "Two-way / structure only"
    if session_name == "New York":
        return "Best continuation window" if score >= 60 else "Best selloff window" if score <= 40 else "Wait for data catalyst"
    return "Neutral"


def pair_bias(scores: Dict[str, MacroScoreResult]) -> List[Dict[str, str]]:
    usd = scores["USD"].score
    eur = scores["EUR"].score
    gbp = scores["GBP"].score
    jpy = scores["JPY"].score

    def bias_from_diff(a: float, b: float) -> str:
        d = a - b
        if d >= 10:
            return "Bullish"
        if d <= -10:
            return "Bearish"
        return "Neutral"

    return [
        {"Instrument": "EUR/USD", "Bias": bias_from_diff(eur, usd)},
        {"Instrument": "GBP/USD", "Bias": bias_from_diff(gbp, usd)},
        {"Instrument": "USD/JPY", "Bias": "Bullish" if usd >= 60 and jpy <= 60 else "Bearish" if usd <= 40 or jpy >= 60 else "Neutral"},
        {"Instrument": "XAU/USD", "Bias": "Bearish" if usd >= 60 else "Bullish" if usd <= 40 else "Neutral"},
        {"Instrument": "Nas100", "Bias": "Bearish" if usd >= 60 else "Bullish" if usd <= 40 else "Neutral"},
        {"Instrument": "US30", "Bias": "Bearish" if usd >= 60 else "Bullish" if usd <= 40 else "Neutral"},
        {"Instrument": "VIX", "Bias": "Bullish" if usd >= 60 else "Bearish" if usd <= 40 else "Neutral"},
    ]


def add_alerts(scores: Dict[str, MacroScoreResult]) -> None:
    alerts: List[str] = []
    if scores["USD"].score >= 80:
        alerts.append("Strong bullish USD regime detected.")
    elif scores["USD"].score <= 20:
        alerts.append("Strong bearish USD regime detected.")
    if scores["EUR"].score >= 60:
        alerts.append("EUR macro regime is firm.")
    if scores["GBP"].score >= 60:
        alerts.append("GBP macro regime is firm.")
    if scores["JPY"].score >= 60:
        alerts.append("JPY macro regime is firm.")
    st.session_state.alerts = alerts


def add_snapshot(scores: Dict[str, MacroScoreResult]) -> None:
    today = datetime.utcnow().date().isoformat()
    if st.session_state.last_snapshot_date != today:
        st.session_state.historical_scores.append(
            {
                "date": today,
                "USD": scores["USD"].score,
                "JPY": scores["JPY"].score,
                "GBP": scores["GBP"].score,
                "EUR": scores["EUR"].score,
            }
        )
        st.session_state.last_snapshot_date = today


def build_historical_score_frame() -> pd.DataFrame:
    data = st.session_state.get("historical_scores", [])
    return pd.DataFrame(data) if data else pd.DataFrame(columns=["date", "USD", "JPY", "GBP", "EUR"])


def is_admin() -> bool:
    return bool(st.session_state.admin_unlocked)


def render_admin_panel() -> None:
    with st.expander("Admin Panel", expanded=False):
        if is_admin():
            st.success("Admin unlocked.")
            if st.button("Lock Admin"):
                st.session_state.admin_unlocked = False
                st.rerun()
            if st.button("Clear trade logs"):
                st.session_state.trade_logs = []
                st.success("Trade logs cleared.")
            if st.button("Reset historical snapshots"):
                st.session_state.historical_scores = []
                st.session_state.last_snapshot_date = None
                st.success("Snapshots cleared.")
            return

        pwd = st.text_input("Admin password", type="password")
        if st.button("Unlock Admin"):
            if authenticate_admin(pwd):
                st.session_state.admin_unlocked = True
                st.success("Admin unlocked.")
            else:
                st.error("Invalid admin password.")


def log_trade() -> None:
    with st.form("trade_log_form"):
        c1, c2 = st.columns(2)
        with c1:
            instrument = st.text_input("Instrument", value="EUR/USD")
            direction = st.selectbox("Direction", ["Long", "Short"])
            entry = st.text_input("Entry", value="")
        with c2:
            stop = st.text_input("Stop", value="")
            target = st.text_input("Target", value="")
            notes = st.text_input("Notes", value="")
        if st.form_submit_button("Add Log"):
            st.session_state.trade_logs.append(
                {
                    "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "instrument": instrument,
                    "direction": direction,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "notes": notes,
                }
            )
            st.success("Trade log added.")


def weight_controls() -> Dict[str, float]:
    st.sidebar.header("Weights")
    weights: Dict[str, float] = {}
    for sid, default in DEFAULT_WEIGHTS.items():
        weights[sid] = st.sidebar.slider(SERIES_MAP[sid], 0.0, 0.5, float(default), 0.01)
    return weights


def render_divergence_chart(hist_df: pd.DataFrame) -> None:
    if hist_df.empty:
        st.info("No historical snapshots yet.")
        return
    chart_df = hist_df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"])
    chart_df = chart_df.sort_values("date").set_index("date")
    divergence = pd.DataFrame(index=chart_df.index)
    divergence["USD-JPY"] = chart_df["USD"] - chart_df["JPY"]
    divergence["USD-EUR"] = chart_df["USD"] - chart_df["EUR"]
    divergence["USD-GBP"] = chart_df["USD"] - chart_df["GBP"]
    st.line_chart(divergence, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="USD / JPY / GBP / EUR Macro Dashboard", layout="wide")
    init_state()
    st.title("Multi-Currency Macro Dashboard")

    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        st.error("FRED_API_KEY is missing from environment variables.")
        return

    weights = weight_controls()
    render_admin_panel()

    baskets = {k: fetch_basket(v, api_key) for k, v in CURRENCY_BASKETS.items()}
    scores = {
        "USD": currency_score("USD", baskets["USD"]),
        "JPY": currency_score("JPY", baskets["JPY"]),
        "GBP": currency_score("GBP", baskets["GBP"]),
        "EUR": currency_score("EUR", baskets["EUR"]),
    }

    add_snapshot(scores)
    add_alerts(scores)

    tab_overview, tab_table, tab_sessions, tab_trend, tab_logs = st.tabs(
        ["Overview", "Macro Table", "Session Bias", "Trend", "Trade Logs"]
    )

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("USD Score", f"{scores['USD'].score:.1f}")
        c2.metric("JPY Score", f"{scores['JPY'].score:.1f}")
        c3.metric("GBP Score", f"{scores['GBP'].score:.1f}")
        c4.metric("EUR Score", f"{scores['EUR'].score:.1f}")
        st.subheader("Pair View")
        st.dataframe(pd.DataFrame(pair_bias(scores)), use_container_width=True, hide_index=True)
        if st.session_state.alerts:
            st.subheader("Alerts")
            for a in st.session_state.alerts:
                st.info(a)

    with tab_table:
        all_readings = baskets["USD"] + baskets["JPY"] + baskets["GBP"] + baskets["EUR"]
        rows = []
        for r in all_readings:
            rows.append(
                {
                    "Series": r.name,
                    "ID": r.series_id,
                    "Current": None if r.current is None else round(r.current, 4),
                    "Previous": None if r.previous is None else round(r.previous, 4),
                    "Change": None if r.change is None else round(r.change, 4),
                    "Percent Change": None if r.pct_change is None else round(r.pct_change, 2),
                    "Trend": r.trend,
                    "Last Update": r.last_update,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_sessions:
        st.info(f"Asia: {session_bias(scores['USD'].score, 'Asia')}")
        st.info(f"London: {session_bias(scores['USD'].score, 'London')}")
        st.info(f"New York: {session_bias(scores['USD'].score, 'New York')}")
        st.subheader("Currency Regime")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Currency": k, "Score": v.score, "Bias": v.bias, "Confidence": v.confidence}
                    for k, v in scores.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tab_trend:
        hist_df = build_historical_score_frame()
        st.subheader("Historical Macro Scores")
        if hist_df.empty:
            st.info("No historical snapshots yet.")
        else:
            chart_df = hist_df.copy()
            chart_df["date"] = pd.to_datetime(chart_df["date"])
            chart_df = chart_df.sort_values("date").set_index("date")
            st.line_chart(chart_df[["USD", "JPY", "GBP", "EUR"]], use_container_width=True)
            st.subheader("Divergence")
            render_divergence_chart(chart_df.reset_index())
        st.caption("Historical snapshots are stored once per day in session state.")

    with tab_logs:
        st.subheader("Trade Management Log")
        log_trade()
        if st.session_state.trade_logs:
            st.dataframe(pd.DataFrame(st.session_state.trade_logs), use_container_width=True, hide_index=True)
        else:
            st.info("No trade logs yet.")


if __name__ == "__main__":
    main()
