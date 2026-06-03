import os

import requests
import pandas as pd
import streamlit as st

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = os.getenv("FRED_API_KEY")

SERIES_MAP = {
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI",
    "CPILFESL": "Core CPI",
    "GDP": "GDP",
    "FEDFUNDS": "Fed Funds",
    "DGS10": "10Y Yield",
}


@st.cache_data(ttl=3600)
def get_fred_series(series_id: str):
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is missing")

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }

    response = requests.get(
        FRED_BASE_URL,
        params=params,
        headers={"User-Agent": "TradingDesk/1.0"},
        timeout=20,
    )

    if response.status_code == 429:
        raise RuntimeError(
            "FRED rate limit reached. Please try again later."
        )

    response.raise_for_status()
    return response.json()


def observations_to_df(payload):
    obs = payload.get("observations", [])

    df = pd.DataFrame(obs)

    if df.empty:
        return df

    df = df[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df.dropna().sort_values("date")


def latest_value(series_id: str):
    payload = get_fred_series(series_id)

    df = observations_to_df(payload)

    if df.empty:
        return None

    row = df.iloc[-1]

    return {
        "date": row["date"],
        "value": float(row["value"]),
        "series_id": series_id,
    }


def test_fred_connection():
    try:
        data = latest_value("UNRATE")

        if data is None:
            return False, "No data returned"

        return (
            True,
            f"Connected: UNRATE {data['value']} on {data['date'].date()}"
        )

    except Exception as e:
        return False, f"Connection failed: {e}"


def main():
    st.set_page_config(
        page_title="FRED Test",
        layout="wide"
    )

    st.title("FRED API Test")

    if not FRED_API_KEY:
        st.error("FRED_API_KEY is missing from environment variables")
        return

    ok, msg = test_fred_connection()

    if ok:
        st.success(msg)
    else:
        st.error(msg)

    st.subheader("Latest Macro Readings")

    rows = []

    for sid, name in SERIES_MAP.items():
        try:
            info = latest_value(sid)

            rows.append({
                "Series": name,
                "ID": sid,
                "Date": str(info["date"].date()) if info else None,
                "Value": float(info["value"]) if info else None,
            })

        except Exception:
            rows.append({
                "Series": name,
                "ID": sid,
                "Date": None,
                "Value": None,
            })

    df_display = pd.DataFrame(rows)

    st.dataframe(
        df_display,
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    main()