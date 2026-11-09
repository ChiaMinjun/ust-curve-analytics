# ust-curve-analytics

![tests](https://github.com/ChiaMinjun/ust-curve-analytics/actions/workflows/tests.yml/badge.svg)

A small data pipeline and analysis of the US Treasury yield curve built on free public data from
[FRED](https://fred.stlouisfed.org/). It downloads daily yields, stores them in a SQLite database,
computes curve analytics and statistical tests, and produces a chart pack and a written report.

## What it does

| Step | What happens | Packages |
|---|---|---|
| 1. Download | Daily 3M, 2Y, 5Y, 10Y and 30Y Treasury yields from FRED | `requests` |
| 2. Store | Saves the data to a SQLite database and runs SQL aggregations (monthly averages) | `SQLAlchemy`, `sqlite3` |
| 3. Clean and analyse | Aligns dates and handles missing days; computes 2s10s, 5s30s and 3m10y spreads, daily changes in bp, rolling volatility and z-scores | `pandas`, `NumPy` |
| 4. PCA | Principal components of daily yield changes (level, slope, curvature) using NumPy linear algebra | `NumPy` |
| 5. Statistics | ADF stationarity test and mean-reversion half-life of 2s10s; OLS regression of 10Y on 2Y changes; skew, kurtosis and Jarque-Bera test of daily changes | `statsmodels`, `SciPy` |
| 6. Charts and report | Six charts and `output/report.md` with the key numbers | `matplotlib` |
| Tests | Unit tests run locally and on every push through GitHub Actions | `pytest` |

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run.py                      # download from 2015 and run everything
python run.py --start 2005-01-01   # longer history
python run.py --skip-download      # re-run the analysis on the saved database

pytest -v                          # run the tests
```

The output goes to `output/` (charts, `report.md`, `monthly_averages.csv`). The database goes to `data/yields.db`.

## Project layout

```
curve/
  fetch.py      download and parse FRED data
  store.py      SQLite storage and SQL queries
  analytics.py  spreads, changes, volatility, z-scores, PCA
  stats.py      ADF test, half-life, regression, distribution tests
  plots.py      matplotlib chart pack
run.py          runs the whole pipeline
tests/          pytest unit tests
```

## Sample output

<!-- After your first run, commit the output/ folder and these will show on GitHub -->
![Spreads](output/03_spreads_zscore.png)
![PCA](output/05_pca.png)

## Data source

Board of Governors of the Federal Reserve System, retrieved from FRED, Federal Reserve Bank of St. Louis
(series DGS3MO, DGS2, DGS5, DGS10, DGS30).
