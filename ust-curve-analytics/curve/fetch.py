"""Download daily US Treasury yields from FRED (no API key needed)."""

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# FRED series id -> tenor label used everywhere else in the project
TENORS = {
    "DGS3MO": "3M",
    "DGS2": "2Y",
    "DGS5": "5Y",
    "DGS10": "10Y",
    "DGS30": "30Y",
}


def fetch_series(series_id: str, start: str, timeout: int = 30) -> pd.Series:
    """Download one FRED series as a float Series indexed by date."""
    resp = requests.get(
        FRED_CSV_URL,
        params={"id": series_id, "cosd": start},
        headers={"User-Agent": "ust-curve-analytics"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return parse_fred_csv(resp.text, series_id)


def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """Parse FRED's CSV format. Missing days are written as '.' and become NaN."""
    df = pd.read_csv(StringIO(text))
    date_col = df.columns[0]  # 'observation_date' (older files used 'DATE')
    values = pd.to_numeric(df[series_id], errors="coerce")
    return pd.Series(values.values, index=pd.to_datetime(df[date_col]), name=series_id)


def fetch_curve(start: str = "2015-01-01", raw_dir: Path | None = None) -> pd.DataFrame:
    """Download every tenor and return one DataFrame (columns = tenors, yields in %)."""
    columns = {}
    for series_id, tenor in TENORS.items():
        print(f"  downloading {series_id} ({tenor})...")
        s = fetch_series(series_id, start)
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            s.to_csv(raw_dir / f"{series_id}.csv")
        columns[tenor] = s
    return pd.DataFrame(columns)
