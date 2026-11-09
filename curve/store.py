"""Store yields in a SQLite database and query them back with SQL."""

import pandas as pd
from sqlalchemy import create_engine, text

TABLE = "yields"


def get_engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}")


def save_yields(engine, curve: pd.DataFrame) -> int:
    """Write the curve in long format: one row per (date, tenor, yield)."""
    long = (
        curve.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="tenor", value_name="yield_pct")
        .dropna()
    )
    long["date"] = long["date"].dt.strftime("%Y-%m-%d")
    long.to_sql(TABLE, engine, if_exists="replace", index=False)
    return len(long)


def load_yields(engine) -> pd.DataFrame:
    """Read the table back and pivot to wide format (columns = tenors)."""
    long = pd.read_sql(text(f"SELECT date, tenor, yield_pct FROM {TABLE}"), engine)
    wide = long.pivot(index="date", columns="tenor", values="yield_pct")
    wide.index = pd.to_datetime(wide.index)
    order = [t for t in ["3M", "2Y", "5Y", "10Y", "30Y"] if t in wide.columns]
    return wide[order].sort_index()


def monthly_averages(engine) -> pd.DataFrame:
    """Example of doing the aggregation in SQL instead of pandas."""
    query = text(
        f"""
        SELECT substr(date, 1, 7) AS month,
               tenor,
               ROUND(AVG(yield_pct), 3) AS avg_yield,
               COUNT(*) AS n_days
        FROM {TABLE}
        GROUP BY month, tenor
        ORDER BY month, tenor
        """
    )
    return pd.read_sql(query, engine)
