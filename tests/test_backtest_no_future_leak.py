import numpy as np
import pandas as pd


def test_entry_signals_do_not_use_future_rows():
    from src.backtest.calibrate import BacktestParams, compute_entry_signals

    n = 80
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    close = np.linspace(10.0, 50.0, num=n)
    high = close + 1.0
    low = close - 1.0
    open_ = close

    df1 = pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})
    params = BacktestParams(adx_trend_threshold=20.0, atr_stop_k=2.0)

    sig1 = compute_entry_signals(df1, params)

    df2 = df1.copy()
    df2.loc[df2.index[-1], "close"] = 9999.0
    df2.loc[df2.index[-1], "high"] = 10000.0
    df2.loc[df2.index[-1], "low"] = 9998.0

    sig2 = compute_entry_signals(df2, params)

    # Changing the future row must not change past signals.
    assert sig1.iloc[:-1].reset_index(drop=True).equals(sig2.iloc[:-1].reset_index(drop=True))

