from datetime import date

import pandas as pd

import src.core.pipeline as pipeline_mod
from src.config import Config


class _StubDB:
    def has_today_data(self, code: str, target_date: date) -> bool:
        return False

    def save_daily_data(self, df: pd.DataFrame, code: str, data_source: str = "Unknown") -> int:
        return len(df)


class _StubFetcherManager:
    def __init__(self) -> None:
        self.calls = []

    def get_daily_data(self, code: str, days: int = 30):
        self.calls.append((code, days))
        df = pd.DataFrame(
            {
                "date": [date(2026, 1, 1)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }
        )
        return df, "StubSource"


class _StubSearchService:
    def __init__(self, *args, **kwargs) -> None:
        pass

    @property
    def is_available(self) -> bool:
        return False


def test_fetch_and_save_stock_data_uses_hist_days(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "get_db", lambda: _StubDB())
    monkeypatch.setattr(pipeline_mod, "SearchService", _StubSearchService)
    monkeypatch.setattr(pipeline_mod, "NotificationService", lambda *a, **k: object())
    monkeypatch.setattr(pipeline_mod, "GeminiAnalyzer", lambda *a, **k: object())
    monkeypatch.setattr(pipeline_mod, "StockTrendAnalyzer", lambda *a, **k: object())

    cfg = Config(hist_days=123)
    pipeline = pipeline_mod.StockAnalysisPipeline(config=cfg, max_workers=1)
    pipeline.fetcher_manager = _StubFetcherManager()

    ok, err = pipeline.fetch_and_save_stock_data("600519", force_refresh=True)
    assert ok is True
    assert err is None
    assert pipeline.fetcher_manager.calls[-1] == ("600519", 123)

