from datetime import date, timedelta

import pandas as pd

from src.storage import DatabaseManager


def test_get_analysis_context_includes_raw_data_sorted():
    DatabaseManager.reset_instance()
    db = DatabaseManager(db_url="sqlite:///:memory:")

    start = date(2026, 1, 1)
    df = pd.DataFrame(
        {
            "date": [start + timedelta(days=i) for i in range(3)],
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.0, 2.0, 3.0],
            "volume": [100, 110, 120],
            "amount": [1000, 1100, 1200],
            "pct_chg": [0.0, 0.0, 0.0],
            "ma5": [1.0, 1.5, 2.0],
            "ma10": [1.0, 1.5, 2.0],
            "ma20": [1.0, 1.5, 2.0],
            "volume_ratio": [1.0, 1.0, 1.0],
        }
    )

    db.save_daily_data(df, code="TEST", data_source="TestSource")

    ctx = db.get_analysis_context("TEST")
    assert ctx is not None
    assert "raw_data" in ctx
    raw = ctx["raw_data"]
    assert isinstance(raw, list)
    assert len(raw) == 3
    assert raw[0]["date"] <= raw[1]["date"] <= raw[2]["date"]

