"""Tests for preflight doctor checks."""

from __future__ import annotations

from scripts.doctor import run_doctor


def test_doctor_fails_when_prod_missing_required_env() -> None:
    result = run_doctor(
        {
            "MODE": "prod",
            "PORT": "8000",
            "HOST": "127.0.0.1",
            "FERNET_KEY": "x",
            "CORS_ALLOW_ORIGINS": "https://app.example.com",
        }
    )
    assert result.ok is False
    joined = "\n".join(result.messages)
    assert "DATABASE_URL" in joined


def test_doctor_fails_when_test_mode_limits_missing() -> None:
    result = run_doctor(
        {
            "MODE": "test",
            "PORT": "8000",
            "HOST": "127.0.0.1",
            "TEST_DAILY_TOKEN_LIMIT": "1000000",
            "TEST_MONTHLY_COST_LIMIT": "1000",
        }
    )
    assert result.ok is False
    joined = "\n".join(result.messages)
    assert "TEST_RATE_LIMIT_RPM" in joined


def test_doctor_passes_in_local_stub_mode_defaults() -> None:
    result = run_doctor(
        {
            "MODE": "local",
            "USE_STUB_ADAPTERS": "true",
            "PORT": "8000",
            "HOST": "127.0.0.1",
        }
    )
    assert result.ok is True
    assert result.messages[0] == "Doctor checks passed."
