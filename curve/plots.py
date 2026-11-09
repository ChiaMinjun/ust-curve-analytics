"""Chart pack with matplotlib."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_yields(curve: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    curve.plot(ax=ax, linewidth=1)
    ax.set_title("US Treasury yields")
    ax.set_ylabel("Yield (%)")
    ax.set_xlabel("")
    return _save(fig, out_dir, "01_yields.png")


def plot_curve_snapshots(curve: pd.DataFrame, out_dir: Path) -> Path:
    """Shape of the curve today vs 1 and 3 years ago."""
    tenor_years = {"3M": 0.25, "2Y": 2, "5Y": 5, "10Y": 10, "30Y": 30}
    x = [tenor_years[t] for t in curve.columns]
    fig, ax = plt.subplots(figsize=(8, 5))
    last = curve.index[-1]
    for label, offset in [("Today", 0), ("1y ago", 1), ("3y ago", 3)]:
        target = last - pd.DateOffset(years=offset)
        row = curve.loc[:target].iloc[-1] if offset else curve.iloc[-1]
        ax.plot(x, row.values, marker="o", label=f"{label} ({row.name:%d %b %Y})")
    ax.set_xscale("log")
    ax.set_xticks(x, curve.columns)
    ax.set_title("Yield curve snapshots")
    ax.set_ylabel("Yield (%)")
    ax.legend()
    return _save(fig, out_dir, "02_curve_snapshots.png")


def plot_spreads(spreads: pd.DataFrame, zscores: pd.Series, out_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    spreads.plot(ax=ax1, linewidth=1)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("Curve spreads")
    ax1.set_ylabel("bp")
    zscores.plot(ax=ax2, color="tab:purple", linewidth=1)
    for level in (-2, 2):
        ax2.axhline(level, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_title("2s10s z-score (1-year rolling)")
    ax2.set_xlabel("")
    return _save(fig, out_dir, "03_spreads_zscore.png")


def plot_rolling_vol(vol: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    vol.plot(ax=ax, linewidth=1)
    ax.set_title("Realised volatility of daily yield changes (3-month rolling, annualised)")
    ax.set_ylabel("bp per year")
    ax.set_xlabel("")
    return _save(fig, out_dir, "04_rolling_vol.png")


def plot_pca(explained, loadings: pd.DataFrame, out_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    names = ["Level", "Slope", "Curvature"]
    labels = [f"{pc}\n{names[i]}" if i < 3 else pc for i, pc in enumerate(loadings.columns)]
    ax1.bar(labels, explained * 100, color="tab:blue")
    ax1.set_title("Variance explained")
    ax1.set_ylabel("%")
    for col in loadings.columns:
        ax2.plot(loadings.index, loadings[col], marker="o", label=col)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("PCA loadings by tenor")
    ax2.legend()
    return _save(fig, out_dir, "05_pca.png")


def plot_change_histogram(changes: pd.Series, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(changes.dropna(), bins=80, color="tab:blue", alpha=0.8)
    ax.set_title(f"Distribution of daily {changes.name} yield changes")
    ax.set_xlabel("bp")
    ax.set_ylabel("Days")
    return _save(fig, out_dir, "06_change_histogram.png")
