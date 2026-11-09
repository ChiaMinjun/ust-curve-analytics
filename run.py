"""Run the whole pipeline: download -> store in SQLite -> analyse -> charts -> report.

Usage:
    python run.py                      # download from FRED, then analyse
    python run.py --start 2010-01-01   # longer history
    python run.py --skip-download      # re-run analysis on the existing database
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from curve import analytics, fetch, plots, stats, store

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
DB_PATH = DATA_DIR / "yields.db"


def fmt(d: dict) -> str:
    """Turn {'beta': 0.78, 'r_squared': 0.62} into 'beta 0.78, r_squared 0.62'."""
    return ", ".join(f"{k} {float(v):g}" for k, v in d.items())


def build_report(curve, spreads, changes, results) -> str:
    """Markdown summary of the key numbers."""
    lines = [
        "# US Treasury curve report",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Data: FRED, "
        f"{curve.index[0]:%d %b %Y} to {curve.index[-1]:%d %b %Y} ({len(curve):,} trading days).",
        "",
        "## Levels and volatility",
        "",
        results["summary"].to_markdown(),
        "",
        "## Curve spreads (bp)",
        "",
        f"- 2s10s latest: {spreads['2s10s'].iloc[-1]:.0f} bp, "
        f"z-score vs last year: {results['z_2s10s'].iloc[-1]:.2f}",
        f"- 5s30s latest: {spreads['5s30s'].iloc[-1]:.0f} bp",
        f"- 3m10y latest: {spreads['3m10y'].iloc[-1]:.0f} bp",
        f"- Days 2s10s was inverted (< 0): {(spreads['2s10s'] < 0).sum():,} "
        f"of {len(spreads):,}",
        "",
        "## Statistical tests",
        "",
        f"- ADF test on 2s10s: {fmt(results['adf'])}",
        f"- Mean-reversion half-life of 2s10s: {results['half_life']:.0f} trading days",
        f"- Regression of 10Y on 2Y daily changes: {fmt(results['beta'])}",
        f"- 10Y daily change distribution: {fmt(results['dist'])}",
        "",
        "## Principal components of daily yield changes",
        "",
        "Variance explained: "
        + ", ".join(f"PC{i + 1} {v:.1%}" for i, v in enumerate(results["pca_explained"])),
        "",
        results["pca_loadings"].round(3).to_markdown(),
        "",
        "## Charts",
        "",
    ]
    lines += [f"![{p.stem}]({p.name})" for p in results["charts"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--start", default="2015-01-01", help="first date to download")
    parser.add_argument("--skip-download", action="store_true", help="use the existing database")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    engine = store.get_engine(DB_PATH)

    if not args.skip_download:
        print("1. Downloading yields from FRED")
        raw = fetch.fetch_curve(args.start, raw_dir=DATA_DIR / "raw")
        n = store.save_yields(engine, raw)
        print(f"2. Saved {n:,} rows to {DB_PATH.relative_to(ROOT)}")
    else:
        print("1-2. Skipping download, using existing database")

    curve = analytics.clean_curve(store.load_yields(engine))
    monthly = store.monthly_averages(engine)
    monthly.to_csv(OUT_DIR / "monthly_averages.csv", index=False)
    print(f"3. Loaded {len(curve):,} clean trading days; monthly averages via SQL: {len(monthly):,} rows")

    spreads = analytics.spreads_bp(curve)
    changes = analytics.daily_changes_bp(curve)
    vol = analytics.rolling_vol_bp(changes)
    z = analytics.zscore(spreads["2s10s"])
    explained, loadings = analytics.pca(changes)

    results = {
        "summary": analytics.summary_table(curve, changes),
        "z_2s10s": z,
        "adf": stats.adf_test(spreads["2s10s"]),
        "half_life": stats.half_life(spreads["2s10s"]),
        "beta": stats.beta_regression(changes["10Y"], changes["2Y"]),
        "dist": stats.distribution_stats(changes["10Y"]),
        "pca_explained": explained,
        "pca_loadings": loadings,
    }
    print("4. Analytics and statistical tests done")

    results["charts"] = [
        plots.plot_yields(curve, OUT_DIR),
        plots.plot_curve_snapshots(curve, OUT_DIR),
        plots.plot_spreads(spreads, z, OUT_DIR),
        plots.plot_rolling_vol(vol, OUT_DIR),
        plots.plot_pca(explained, loadings, OUT_DIR),
        plots.plot_change_histogram(changes["10Y"], OUT_DIR),
    ]
    print(f"5. Saved {len(results['charts'])} charts to output/")

    report = build_report(curve, spreads, changes, results)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print("6. Report written to output/report.md\n")
    print(results["summary"].to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    main()
