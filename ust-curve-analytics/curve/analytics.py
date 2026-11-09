"""Curve analytics with pandas and NumPy."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def clean_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """Keep business days where every tenor has a value (drops holidays / gaps)."""
    return curve.sort_index().dropna(how="any")


def spreads_bp(curve: pd.DataFrame) -> pd.DataFrame:
    """Common curve spreads in basis points."""
    return pd.DataFrame(
        {
            "2s10s": (curve["10Y"] - curve["2Y"]) * 100,
            "5s30s": (curve["30Y"] - curve["5Y"]) * 100,
            "3m10y": (curve["10Y"] - curve["3M"]) * 100,
        }
    )


def daily_changes_bp(curve: pd.DataFrame) -> pd.DataFrame:
    """Day-over-day yield changes in basis points."""
    return curve.diff().dropna() * 100


def rolling_vol_bp(changes: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Annualised rolling volatility of daily changes, in bp (63 days ~ 3 months)."""
    return changes.rolling(window).std() * np.sqrt(TRADING_DAYS)


def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """How many standard deviations today's value is from its rolling mean."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def pca(changes: pd.DataFrame, n_components: int = 3):
    """Principal component analysis of yield changes using NumPy only.

    Returns (explained_variance_ratio, loadings) where loadings is a DataFrame
    with tenors as rows and PC1..PCn as columns.
    """
    x = changes.to_numpy()
    x = x - x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)          # ascending order
    order = np.argsort(eigvals)[::-1]               # largest first
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # Fix sign so PC1 (level) has positive loadings - easier to read
    for i in range(eigvecs.shape[1]):
        if eigvecs[:, i].sum() < 0:
            eigvecs[:, i] *= -1

    explained = eigvals / eigvals.sum()
    cols = [f"PC{i + 1}" for i in range(n_components)]
    loadings = pd.DataFrame(eigvecs[:, :n_components], index=changes.columns, columns=cols)
    return explained[:n_components], loadings


def summary_table(curve: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """Latest level, 1-year range and realised vol for each tenor."""
    last_year = curve.loc[curve.index >= curve.index[-1] - pd.DateOffset(years=1)]
    return pd.DataFrame(
        {
            "latest_%": curve.iloc[-1],
            "1y_low_%": last_year.min(),
            "1y_high_%": last_year.max(),
            "ann_vol_bp": changes.std() * np.sqrt(TRADING_DAYS),
        }
    ).round(2)
