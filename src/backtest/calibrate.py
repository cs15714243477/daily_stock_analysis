# -*- coding: utf-8 -*-
"""
Backtest / calibration helpers (starter).

This module is intentionally lightweight: it provides a causal (no-lookahead)
signal generator plus a minimal backtest loop for quick threshold tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.indicators import compute_adx, compute_atr


@dataclass(frozen=True)
class BacktestParams:
    adx_trend_threshold: float = 25.0
    atr_stop_k: float = 2.0


def compute_entry_signals(df: pd.DataFrame, params: BacktestParams) -> pd.Series:
    """
    Compute entry signals for a simple swing template (trend + pullback).

    Important: This function MUST be causal. All features are computed using
    rolling/ewm operations that depend only on current and past rows.
    """
    if df is None or df.empty:
        return pd.Series(dtype=bool)

    data = df.copy()
    if "date" in data.columns:
        data = data.sort_values("date").reset_index(drop=True)

    close = pd.to_numeric(data["close"], errors="coerce")
    ma10 = close.rolling(window=10, min_periods=10).mean()
    ma20 = close.rolling(window=20, min_periods=20).mean()

    atr14 = compute_atr(data, period=14, high_col="high", low_col="low", close_col="close")
    adx14 = compute_adx(data, period=14, high_col="high", low_col="low", close_col="close")

    # Pullback definition (starter):
    # - ADX above threshold (trend)
    # - price above MA20 (up-trend bias)
    # - close near MA10 within 0.5*ATR
    pullback_dist = (close - ma10).abs()
    near_ma10 = pullback_dist <= (0.5 * atr14)

    signal = (adx14 >= params.adx_trend_threshold) & (close > ma20) & near_ma10
    return signal.fillna(False)


def run_simple_stop_backtest(
    df: pd.DataFrame,
    params: BacktestParams,
    *,
    hold_days: int = 20,
) -> list[dict]:
    """
    Minimal backtest:
    - Signal on day t
    - Enter on t+1 open
    - Stop loss = entry - atr_stop_k * ATR14(t)
    - Exit at stop if breached, else time-exit at close on t+hold_days
    """
    if df is None or df.empty:
        return []

    data = df.copy()
    if "date" in data.columns:
        data = data.sort_values("date").reset_index(drop=True)

    signals = compute_entry_signals(data, params)
    atr14 = compute_atr(data, period=14, high_col="high", low_col="low", close_col="close")

    trades: list[dict] = []
    n = len(data)

    for t in range(n - 1):
        if not bool(signals.iloc[t]):
            continue

        entry_idx = t + 1
        entry_price = float(data.loc[entry_idx, "open"]) if "open" in data.columns else float(data.loc[entry_idx, "close"])
        stop_atr = atr14.iloc[t]
        if pd.isna(stop_atr) or stop_atr <= 0:
            continue

        stop_price = entry_price - float(params.atr_stop_k) * float(stop_atr)
        exit_price = None
        exit_idx = min(entry_idx + hold_days, n - 1)

        # Scan forward for stop hit
        for i in range(entry_idx, exit_idx + 1):
            low = data.loc[i, "low"] if "low" in data.columns else data.loc[i, "close"]
            if pd.notna(low) and float(low) <= stop_price:
                exit_price = stop_price
                exit_idx = i
                break

        if exit_price is None:
            exit_price = float(data.loc[exit_idx, "close"])

        ret = (exit_price - entry_price) / entry_price if entry_price else 0.0
        trades.append(
            {
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return": ret,
                "stop_price": stop_price,
            }
        )

    return trades


def summarize_trades(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "avg_return": 0.0}

    returns = [float(t["return"]) for t in trades]
    wins = sum(1 for r in returns if r > 0)
    return {
        "trades": len(trades),
        "win_rate": wins / len(trades),
        "avg_return": sum(returns) / len(returns),
    }


def grid_search(
    df: pd.DataFrame,
    *,
    adx_thresholds: Iterable[float] = (20.0, 25.0, 30.0),
    atr_stop_ks: Iterable[float] = (1.5, 2.0, 2.5),
    hold_days: int = 20,
) -> list[dict]:
    results: list[dict] = []
    for adx_th in adx_thresholds:
        for k in atr_stop_ks:
            params = BacktestParams(adx_trend_threshold=float(adx_th), atr_stop_k=float(k))
            trades = run_simple_stop_backtest(df, params, hold_days=hold_days)
            summary = summarize_trades(trades)
            summary.update({"adx_trend_threshold": adx_th, "atr_stop_k": k})
            results.append(summary)
    return results

