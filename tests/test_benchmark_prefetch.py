from datetime import date, timedelta

import pandas as pd

import src.core.pipeline as pipeline_mod
from src.config import Config


class _StubDB:
    def get_analysis_context(self, code: str):
        start = date(2026, 1, 1)
        raw_data = []
        for i in range(5):
            d = start + timedelta(days=i)
            raw_data.append({"date": d, "close": 100.0 + i})
        return {"code": code, "date": "2026-01-05", "today": {}, "raw_data": raw_data}


class _StubSearchService:
    def __init__(self, *args, **kwargs) -> None:
        pass

    @property
    def is_available(self) -> bool:
        return False


def test_prefetch_benchmark_history_sets_cache(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "get_db", lambda: _StubDB())
    monkeypatch.setattr(pipeline_mod, "SearchService", _StubSearchService)
    monkeypatch.setattr(pipeline_mod, "NotificationService", lambda *a, **k: object())
    monkeypatch.setattr(pipeline_mod, "GeminiAnalyzer", lambda *a, **k: object())
    monkeypatch.setattr(pipeline_mod, "StockTrendAnalyzer", lambda *a, **k: object())

    cfg = Config(benchmark_code="BENCH")
    pipeline = pipeline_mod.StockAnalysisPipeline(config=cfg, max_workers=1)

    called = {"code": None}

    def _fake_fetch(code: str, force_refresh: bool = False):
        called["code"] = code
        return True, None

    pipeline.fetch_and_save_stock_data = _fake_fetch  # type: ignore[assignment]

    pipeline._prefetch_benchmark_history()

    assert called["code"] == "BENCH"
    assert isinstance(pipeline._benchmark_history_df, pd.DataFrame)
    assert len(pipeline._benchmark_history_df) == 5
    assert list(pipeline._benchmark_history_df.columns) == ["date", "close"]

