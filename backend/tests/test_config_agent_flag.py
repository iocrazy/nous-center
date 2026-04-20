import os
import importlib

from src import config


def test_agent_injection_flag_defaults_false(monkeypatch):
    monkeypatch.delenv("NOUS_ENABLE_AGENT_INJECTION", raising=False)
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert s.NOUS_ENABLE_AGENT_INJECTION is False


def test_agent_injection_flag_reads_env(monkeypatch):
    monkeypatch.setenv("NOUS_ENABLE_AGENT_INJECTION", "true")
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert s.NOUS_ENABLE_AGENT_INJECTION is True
