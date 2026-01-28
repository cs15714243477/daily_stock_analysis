from src.analyzer import GeminiAnalyzer


def test_format_prompt_includes_indicators_section():
    analyzer = GeminiAnalyzer()

    ctx = {
        "code": "TEST",
        "date": "2026-01-01",
        "today": {
            "close": 1.0,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "pct_chg": 0.0,
            "volume": 1.0,
            "amount": 1.0,
            "ma5": 1.0,
            "ma10": 1.0,
            "ma20": 1.0,
        },
        "ma_status": "震荡整理 ↔️",
        "indicators": {
            "atr_14": 1.23,
            "adx_14": 25.6,
            "boll": {"bandwidth": 0.1},
            "rs_20": 0.05,
            "rs_60": 0.1,
        },
    }

    prompt = analyzer._format_prompt(ctx, name="TEST", news_context=None)
    assert "ATR(14)" in prompt
    assert "ADX(14)" in prompt

