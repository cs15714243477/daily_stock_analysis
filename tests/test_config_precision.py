from src.config import Config, get_config


def test_hist_days_and_benchmark_code_defaults(monkeypatch):
    Config.reset_instance()
    monkeypatch.delenv("HIST_DAYS", raising=False)
    monkeypatch.delenv("BENCHMARK_CODE", raising=False)

    cfg = get_config()
    assert cfg.hist_days == 260
    assert cfg.benchmark_code == "000300"


def test_hist_days_and_benchmark_code_override(monkeypatch):
    Config.reset_instance()
    monkeypatch.setenv("HIST_DAYS", "520")
    monkeypatch.setenv("BENCHMARK_CODE", "510300")

    cfg = get_config()
    assert cfg.hist_days == 520
    assert cfg.benchmark_code == "510300"

