import numpy as np
import pandas as pd
import pytest

from curve import analytics, fetch, stats, store


@pytest.fixture
def curve():
    """Simple made-up curve: a random-walk level plus a mean-reverting slope."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=800)
    level = 3 + np.cumsum(rng.normal(0, 0.05, len(dates)))
    slope = np.zeros(len(dates))
    for i in range(1, len(dates)):
        slope[i] = 0.95 * slope[i - 1] + rng.normal(0, 0.05)
    tenors = {"3M": -1.0, "2Y": -0.5, "5Y": 0.0, "10Y": 0.5, "30Y": 1.0}
    return pd.DataFrame({t: level + w * slope for t, w in tenors.items()}, index=dates)


def test_parse_fred_csv_handles_missing_values():
    text = "observation_date,DGS10\n2024-01-01,.\n2024-01-02,3.95\n"
    s = fetch.parse_fred_csv(text, "DGS10")
    assert np.isnan(s.iloc[0])
    assert s.iloc[1] == 3.95
    assert isinstance(s.index, pd.DatetimeIndex)


def test_spreads_are_in_basis_points(curve):
    sp = analytics.spreads_bp(curve)
    expected = (curve["10Y"] - curve["2Y"]) * 100
    pd.testing.assert_series_equal(sp["2s10s"], expected, check_names=False)


def test_daily_changes_drop_first_row(curve):
    ch = analytics.daily_changes_bp(curve)
    assert len(ch) == len(curve) - 1
    assert ch.iloc[0, 0] == pytest.approx((curve.iloc[1, 0] - curve.iloc[0, 0]) * 100)


def test_zscore_of_constant_plus_jump():
    s = pd.Series([1.0] * 251 + [2.0] * 1 + [1.0] * 10)
    z = analytics.zscore(s, window=252)
    assert z.iloc[251] > 10  # a one-day jump stands out


def test_pca_finds_level_factor_first(curve):
    explained, loadings = analytics.pca(analytics.daily_changes_bp(curve))
    assert explained[0] > 0.5
    assert explained.sum() <= 1.0 + 1e-9
    assert (loadings["PC1"] > 0).all()  # level: all tenors move together


def test_half_life_of_known_ar1():
    # AR(1) with phi = 0.95 -> half-life = ln(0.5)/ln(0.95) ~ 13.5 days
    rng = np.random.default_rng(2)
    x = np.zeros(10_000)
    for i in range(1, len(x)):
        x[i] = 0.95 * x[i - 1] + rng.normal()
    assert stats.half_life(pd.Series(x)) == pytest.approx(13.5, rel=0.15)


def test_half_life_is_infinite_for_trending_series():
    s = pd.Series(np.exp(np.linspace(0, 3, 300)))
    assert stats.half_life(s) == float("inf")


def test_beta_regression_recovers_slope():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=500))
    y = 0.8 * x + rng.normal(scale=0.1, size=500)
    res = stats.beta_regression(y, x)
    assert res["beta"] == pytest.approx(0.8, abs=0.05)
    assert res["r_squared"] > 0.9


def test_sqlite_round_trip(curve, tmp_path):
    engine = store.get_engine(tmp_path / "test.db")
    n = store.save_yields(engine, curve)
    assert n == curve.size
    back = store.load_yields(engine)
    pd.testing.assert_frame_equal(back, curve, check_freq=False, check_names=False)
    monthly = store.monthly_averages(engine)
    assert set(monthly["tenor"]) == set(curve.columns)
