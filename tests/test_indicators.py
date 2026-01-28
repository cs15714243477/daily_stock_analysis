import numpy as np
import pandas as pd


def test_compute_bollinger_constant_prices_bandwidth_zero():
    from src.indicators import compute_bollinger

    df = pd.DataFrame({"close": [10.0] * 25})
    out = compute_bollinger(df, window=20, num_std=2, close_col="close")

    assert out["boll_bandwidth"].iloc[-1] == 0.0
    assert out["boll_upper"].iloc[-1] == 10.0
    assert out["boll_lower"].iloc[-1] == 10.0


def test_compute_atr_constant_prices_zero():
    from src.indicators import compute_atr

    df = pd.DataFrame({"high": [10.0] * 25, "low": [10.0] * 25, "close": [10.0] * 25})
    atr = compute_atr(df, period=14, high_col="high", low_col="low", close_col="close")

    assert atr.iloc[-1] == 0.0


def test_compute_adx_trending_between_0_100():
    from src.indicators import compute_adx

    n = 60
    high = np.array([10.0 + i for i in range(n)], dtype=float)
    low = np.array([9.0 + i * 0.2 for i in range(n)], dtype=float)
    close = (high + low) / 2.0
    df = pd.DataFrame({"high": high, "low": low, "close": close})

    adx = compute_adx(df, period=14, high_col="high", low_col="low", close_col="close")
    last = float(adx.iloc[-1])

    assert not np.isnan(last)
    assert 0.0 <= last <= 100.0

