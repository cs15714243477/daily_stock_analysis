# -*- coding: utf-8 -*-
"""
Indicator helpers used by the swing (1–4 weeks) analysis stack.

These functions are designed to be:
- deterministic (pure functions over a DataFrame)
- lightweight (no extra dependencies beyond pandas/numpy)
- safe on short series (return NaN where insufficient history)
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


def _wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing (EMA with alpha = 1/period)."""
    if period <= 0:
        raise ValueError("period must be > 0")
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def compute_atr(
    df: pd.DataFrame,
    period: int = 14,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """
    Compute Average True Range (ATR) using Wilder smoothing.

    Returns a Series aligned to df.index. Values may be NaN until enough history exists.
    """
    high = pd.to_numeric(df[high_col], errors="coerce")
    low = pd.to_numeric(df[low_col], errors="coerce")
    close = pd.to_numeric(df[close_col], errors="coerce")

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1, skipna=True)

    return _wilder_ema(true_range, period)


def compute_bollinger(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    *,
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Compute Bollinger Bands and bandwidth.

    Returns a DataFrame with columns:
    - boll_mid
    - boll_upper
    - boll_lower
    - boll_bandwidth  ( (upper-lower) / mid )
    """
    close = pd.to_numeric(df[close_col], errors="coerce")

    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)

    upper = mid + num_std * std
    lower = mid - num_std * std

    with np.errstate(divide="ignore", invalid="ignore"):
        bandwidth = (upper - lower) / mid.replace(0, np.nan)

    return pd.DataFrame(
        {
            "boll_mid": mid,
            "boll_upper": upper,
            "boll_lower": lower,
            "boll_bandwidth": bandwidth.fillna(0.0),
        }
    )


def compute_adx(
    df: pd.DataFrame,
    period: int = 14,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """
    Compute Average Directional Index (ADX) using Wilder smoothing.

    Returns a Series aligned to df.index. Values may be NaN until enough history exists.
    """
    high = pd.to_numeric(df[high_col], errors="coerce")
    low = pd.to_numeric(df[low_col], errors="coerce")
    close = pd.to_numeric(df[close_col], errors="coerce")

    up_move = high.diff()
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1, skipna=True)

    atr = _wilder_ema(true_range, period)
    plus_dm_smoothed = _wilder_ema(plus_dm, period)
    minus_dm_smoothed = _wilder_ema(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * plus_dm_smoothed / atr.replace(0, np.nan)
        minus_di = 100.0 * minus_dm_smoothed / atr.replace(0, np.nan)

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    adx = _wilder_ema(dx.fillna(0.0), period)

    # ADX is bounded in [0, 100] by definition; numerical noise may exceed slightly.
    return adx.clip(lower=0.0, upper=100.0)


def compute_relative_strength(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (20, 60),
    date_col: str = "date",
    close_col: str = "close",
    benchmark_close_col: str | None = None,
) -> dict[str, float | None]:
    """
    Compute relative strength (out/under-performance) vs a benchmark.

    For each window w, computes:
      RS_w = ( (S_t / S_{t-w}) / (B_t / B_{t-w}) ) - 1

    Returns latest RS values as a dict, e.g. {"rs_20": 0.12, "rs_60": -0.03}.
    If insufficient history exists, value is None.
    """
    if benchmark_close_col is None:
        benchmark_close_col = close_col

    s = stock_df[[date_col, close_col]].copy()
    b = benchmark_df[[date_col, benchmark_close_col]].copy()

    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    b[date_col] = pd.to_datetime(b[date_col], errors="coerce")

    merged = pd.merge(s, b, on=date_col, how="inner", suffixes=("_s", "_b")).sort_values(date_col)
    if merged.empty:
        return {f"rs_{w}": None for w in windows}

    s_close = pd.to_numeric(merged[f"{close_col}_s"], errors="coerce")
    b_close = pd.to_numeric(merged[f"{benchmark_close_col}_b"], errors="coerce")

    out: dict[str, float | None] = {}
    for w in windows:
        if w <= 0:
            raise ValueError("windows must be > 0")

        s_ratio = s_close / s_close.shift(w)
        b_ratio = b_close / b_close.shift(w)

        with np.errstate(divide="ignore", invalid="ignore"):
            rs = (s_ratio / b_ratio.replace(0, np.nan)) - 1.0

        last = rs.iloc[-1]
        out[f"rs_{w}"] = float(last) if pd.notna(last) else None

    return out
