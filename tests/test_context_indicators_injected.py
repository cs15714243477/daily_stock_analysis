from datetime import date, timedelta

import pandas as pd

import src.core.pipeline as pipeline_mod
from src.config import Config


class _StubSearchService:
    def __init__(self, *args, **kwargs) -> None:
        pass

    @property
    def is_available(self) -> bool:
        return False


def test_enhance_context_injects_indicators(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "get_db", lambda: object())
    monkeypatch.setattr(pipeline_mod, "SearchService", _StubSearchService)
    monkeypatch.setattr(pipeline_mod, "NotificationService", lambda *a, **k: object())
    monkeypatch.setattr(pipeline_mod, "GeminiAnalyzer", lambda *a, **k: object())
    monkeypatch.setattr(pipeline_mod, "StockTrendAnalyzer", lambda *a, **k: object())

    cfg = Config(hist_days=260, benchmark_code="000300")
    pipeline = pipeline_mod.StockAnalysisPipeline(config=cfg, max_workers=1)

    start = date(2026, 1, 1)
    raw_data = []
    for i in range(60):
        d = start + timedelta(days=i)
        close = 10.0 + i
        raw_data.append(
            {
                "date": d,
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000.0,
                "amount": 1000.0,
                "pct_chg": 0.0,
                "ma5": close,
                "ma10": close,
                "ma20": close,
                "volume_ratio": 1.0,
            }
        )

    benchmark_df = pd.DataFrame(
        {
            "date": [start + timedelta(days=i) for i in range(60)],
            "close": [100.0 + i * 0.5 for i in range(60)],
        }
    )
    pipeline._benchmark_history_df = benchmark_df

    ctx = {"code": "TEST", "date": "2026-01-28", "today": {}, "raw_data": raw_data}
    enhanced = pipeline._enhance_context(ctx, realtime_quote=None, chip_data=None, trend_result=None, stock_name="")

    assert "indicators" in enhanced
    ind = enhanced["indicators"]
    assert ind["atr_14"] is not None
    assert ind["adx_14"] is not None
    assert "boll" in ind and ind["boll"]["bandwidth"] is not None
    assert ind["rs_20"] is not None

