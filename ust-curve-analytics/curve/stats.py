"""Statistical tests with SciPy and statsmodels."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test. p-value < 0.05 suggests the series mean-reverts."""
    with warnings.catch_warnings():  # newer statsmodels warns about its return type
        warnings.simplefilter("ignore", FutureWarning)
        result = adfuller(series.dropna(), autolag="AIC")
    stat, pvalue = result[0], result[1]
    return {"adf_stat": round(stat, 3), "p_value": round(pvalue, 4)}


def half_life(series: pd.Series) -> float:
    """Mean-reversion half-life in days, from an AR(1) regression:
    change_t = a + b * level_{t-1};  half-life = -ln(2) / ln(1 + b).
    Returns inf if the series does not mean-revert (b >= 0).
    """
    s = series.dropna()
    lagged = s.shift(1).iloc[1:]
    delta = s.diff().iloc[1:]
    model = sm.OLS(delta, sm.add_constant(lagged)).fit()
    b = model.params.iloc[1]
    if b >= 0:
        return float("inf")
    return float(-np.log(2) / np.log(1 + b))


def beta_regression(y: pd.Series, x: pd.Series) -> dict:
    """OLS of y on x (e.g. 10Y changes on 2Y changes): beta, t-stat and R-squared."""
    data = pd.concat([y, x], axis=1).dropna()
    model = sm.OLS(data.iloc[:, 0], sm.add_constant(data.iloc[:, 1])).fit()
    return {
        "beta": round(model.params.iloc[1], 3),
        "t_stat": round(model.tvalues.iloc[1], 1),
        "r_squared": round(model.rsquared, 3),
    }


def distribution_stats(changes: pd.Series) -> dict:
    """Shape of daily changes: fat tails show up as excess kurtosis > 0."""
    x = changes.dropna()
    jb = stats.jarque_bera(x)
    return {
        "mean_bp": round(x.mean(), 3),
        "std_bp": round(x.std(), 2),
        "skew": round(stats.skew(x), 2),
        "excess_kurtosis": round(stats.kurtosis(x), 2),
        "jarque_bera_p": round(jb.pvalue, 4),
        "pct_99_bp": round(np.percentile(x, 99), 2),
        "pct_1_bp": round(np.percentile(x, 1), 2),
    }
