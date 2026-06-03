import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_TIMEOUT = 15
CACHE_TTL = 3600

SERIES_MAP: Dict[str, str] = {
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI",
    "CPILFESL": "Core CPI",
    "GDP": "GDP",
    "FEDFUNDS": "Fed Funds",
    "DGS10": "10Y Yield",
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "UNRATE": 0.25,
    "FEDFUNDS": 0.20,
    "DGS10": 0.20,
    "GDP": 0.15,
    "CPIAUCSL": 0.10,
    "CPILFESL": 0.10,
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
    if "trade_logs" not in st.session_state:
        st.session_state.trade_logs = []
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    if "historical_scores" not in st.session_state:
        st.session_state.historical_scores = []
    if "last_snapshot_date" not in st.session_state:
        st.session_state.last_snapshot_date = None


@st.cache_resource
def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "TradingDesk/1.0"})
    return session


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
    try:
        response = session.get(FRED_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 429:
            raise RuntimeError("FRED rate limit reached. Please try again later.")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout as e:
        raise RuntimeError(f"Timeout while fetching {series_id}.") from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"FRED request failed for {series_id}: {e}") from e


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
    previous = (
        float(previous_row["value"])
        if previous_row is not None and pd.notna(previous_row["value"])
        else None
    )
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


def fetch_all_readings(api_key: str) -> List[SeriesReading]:
    readings: List[SeriesReading] = []
    for sid in SERIES_MAP:
        try:
            reading = latest_reading(sid, api_key)
            if reading is not None:
                readings.append(reading)
        except Exception as e:
            st.warning(f"{SERIES_MAP[sid]} could not be loaded: {e}")
    return readings


def score_series(r: SeriesReading) -> float:
    if r.series_id == "UNRATE":
        if r.change is None:
            return 50.0
        return 90.0 if r.change < 0 else 10.0 if r.change > 0 else 50.0

    if r.series_id in {"FEDFUNDS", "DGS10"}:
        if r.change is None:
            return 50.0
        return 80.0 if r.change > 0 else 20.0 if r.change < 0 else 50.0

    if r.series_id == "GDP":
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


def build_macro_score(readings: List[SeriesReading], weights: Dict[str, float]) -> MacroScoreResult:
    components: Dict[str, float] = {}
    weighted_sum = 0.0
    weight_sum = 0.0

    for r in readings:
        score = score_series(r)
        w = float(weights.get(r.series_id, 0.0))
        if w > 0:
            components[r.series_id] = score
            weighted_sum += score * w
            weight_sum += w

    final = round(weighted_sum / weight_sum, 1) if weight_sum > 0 else 50.0

    if final >= 80:
        bias, confidence = "Strong Bullish USD", "High"
    elif final >= 60:
        bias, confidence = "Bullish USD", "Medium"
    elif final >= 40:
        bias, confidence = "Neutral USD", "Low"
    elif final >= 20:
        bias, confidence = "Bearish USD", "Medium"
    else:
        bias, confidence = "Strong Bearish USD", "High"

    return MacroScoreResult(score=final, bias=bias, confidence=confidence, components=components)


def session_bias(score: float, session_name: str) -> str:
    if session_name == "Asia":
        return "Continuation only" if score >= 80 or score <= 20 else "Range / wait for London"
    if session_name == "London":
        if score >= 60:
            return "Prefer USD strength"
        if score <= 40:
            return "Prefer USD weakness"
        return "Two-way / structure only"
    if session_name == "New York":
        if score >= 60:
            return "Best USD continuation window"
        if score <= 40:
            return "Best USD selloff window"
        return "Wait for data catalyst"
    return "Neutral"


def pair_bias(score: float) -> List[Dict[str, str]]:
    def b(long_if_usd_strong: str, long_if_usd_weak: str) -> str:
        if score >= 60:
            return long_if_usd_strong
        if score <= 40:
            return long_if_usd_weak
        return "Neutral"

    return [
        {"Instrument": "EUR/USD", "Bias": b("Bearish", "Bullish")},
        {"Instrument": "GBP/USD", "Bias": b("Bearish", "Bullish")},
        {"Instrument": "AUD/USD", "Bias": b("Bearish", "Bullish")},
        {"Instrument": "NZD/USD", "Bias": b("Bearish", "Bullish")},
        {"Instrument": "USD/JPY", "Bias": b("Bullish", "Bearish")},
        {"Instrument": "USD/CHF", "Bias": b("Bullish", "Bearish")},
        {"Instrument": "USD/CAD", "Bias": b("Bullish", "Bearish")},
        {"Instrument": "XAU/USD", "Bias": b("Bearish", "Bullish")},
        {"Instrument": "Nas100", "Bias": b("Bearish", "Bullish")},
        {"Instrument": "US30", "Bias": b("Bearish", "Bullish")},
        {"Instrument": "VIX", "Bias": b("Bullish", "Bearish")},
    ]


def xau_usdjpy_logic(score: float) -> str:
    if score >= 60:
        return "Prefer USD/JPY longs and XAU/USD shorts unless risk-off overwhelms."
    if score <= 40:
        return "Prefer XAU/USD longs and USD/JPY shorts unless JPY safe-haven demand dominates."
    return "Treat both as range/mean-reversion until a stronger macro regime appears."


def add_alerts(macro: MacroScoreResult) -> None:
    alerts: List[str] = []
    if macro.score >= 80:
        alerts.append("Strong bullish USD regime detected.")
    elif macro.score <= 20:
        alerts.append("Strong bearish USD regime detected.")
    if macro.score >= 60:
        alerts.append("Watch for USD strength continuation in London/NY.")
    if macro.score <= 40:
        alerts.append("Watch for USD weakness continuation in London/NY.")
    st.session_state.alerts = alerts


def add_snapshot(macro: MacroScoreResult) -> None:
    today = datetime.utcnow().date().isoformat()
    if st.session_state.last_snapshot_date != today:
        st.session_state.historical_scores.append(
            {
                "date": today,
                "score": macro.score,
                "bias": macro.bias,
                "confidence": macro.confidence,
            }
        )
        st.session_state.last_snapshot_date = today


def build_historical_score_frame() -> pd.DataFrame:
    data = st.session_state.get("historical_scores", [])
    if not data:
        return pd.DataFrame(columns=["date", "score", "bias", "confidence"])
    return pd.DataFrame(data)


def log_trade() -> None:
    with st.form("trade_log_form"):
        col1, col2 = st.columns(2)
        with col1:
            instrument = st.text_input("Instrument", value="EUR/USD")
            direction = st.selectbox("Direction", ["Long", "Short"])
            entry = st.text_input("Entry", value="")
        with col2:
            stop = st.text_input("Stop", value="")
            target = st.text_input("Target", value="")
            notes = st.text_input("Notes", value="")
        submitted = st.form_submit_button("Add Log")
        if submitted:
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


def sidebar_controls() -> Dict[str, float]:
    st.sidebar.header("Weights")
    weights: Dict[str, float] = {}
    for sid, default in DEFAULT_WEIGHTS.items():
        weights[sid] = st.sidebar.slider(SERIES_MAP[sid], 0.0, 0.5, float(default), 0.01)
    if st.sidebar.button("Reset Weights"):
        st.rerun()
    return weights


def main() -> None:
    st.set_page_config(page_title="USD Macro Dashboard", layout="wide")
    init_state()
    st.title("USD Macro Dashboard")

    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        st.error("FRED_API_KEY is missing from Streamlit environment variables.")
        return

    weights = sidebar_controls()

    try:
        readings = fetch_all_readings(api_key)
    except Exception as e:
        st.error(f"Unable to load macro data: {e}")
        return

    if not readings:
        st.error("No valid macro readings were returned from FRED.")
        return

    macro = build_macro_score(readings, weights)
    add_snapshot(macro)
    add_alerts(macro)

    tab_overview, tab_table, tab_sessions, tab_trend, tab_logs = st.tabs(
        ["Overview", "Macro Table", "Session Bias", "Trend", "Trade Logs"]
    )

    with tab_overview:
        c1, c2, c3 = st.columns(3)
        c1.metric("USD Macro Score", f"{macro.score:.1f}")
        c2.metric("Macro Bias", macro.bias)
        c3.metric("Confidence", macro.confidence)
        st.write(xau_usdjpy_logic(macro.score))
        if st.session_state.alerts:
            st.subheader("Alerts")
            for a in st.session_state.alerts:
                st.info(a)

    with tab_table:
        rows = []
        for r in readings:
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
        s1, s2, s3 = st.columns(3)
        s1.info(f"Asia: {session_bias(macro.score, 'Asia')}")
        s2.info(f"London: {session_bias(macro.score, 'London')}")
        s3.info(f"New York: {session_bias(macro.score, 'New York')}")
        st.subheader("Instrument Bias")
        st.dataframe(pd.DataFrame(pair_bias(macro.score)), use_container_width=True, hide_index=True)

    with tab_trend:
        hist_df = build_historical_score_frame()
        if hist_df.empty:
            st.info("No historical snapshots yet. The chart will populate after app reruns across days.")
        else:
            chart_df = hist_df.copy()
            chart_df["date"] = pd.to_datetime(chart_df["date"])
            chart_df = chart_df.sort_values("date").set_index("date")[["score"]]
            st.line_chart(chart_df, use_container_width=True)
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
