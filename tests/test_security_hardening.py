"""Security hardening regression tests."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from src.auth.config import AuthMode, get_auth_mode, validate_security_config
from src.db.services import ProviderConfigService
from src.observability.logging import JSONFormatter
from src.security.encryption import EncryptionError, OpenSSLEncryptor


class TestModeGuardrails:
    def test_default_mode_is_prod(self):
        with patch.dict("os.environ", {}, clear=True):
            assert get_auth_mode() == AuthMode.PROD

    def test_invalid_mode_raises(self):
        with patch.dict("os.environ", {"MODE": "dev"}, clear=True):
            with pytest.raises(ValueError):
                get_auth_mode()

    def test_prod_requires_database_and_fernet_and_cors(self):
        with patch.dict("os.environ", {"MODE": "prod"}, clear=True):
            with pytest.raises(RuntimeError, match="DATABASE_URL"):
                validate_security_config()

    def test_prod_rejects_wildcard_cors(self):
        env = {
            "MODE": "prod",
            "DATABASE_URL": "postgresql://example",
            "FERNET_KEY": "test-master-key",
            "CORS_ALLOW_ORIGINS": "*",
        }
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(RuntimeError, match=r"cannot include '\*'"):
                validate_security_config()

    def test_local_does_not_require_prod_guards(self):
        with patch.dict("os.environ", {"MODE": "local"}, clear=True):
            validate_security_config()


class TestProviderKeyEncryption:
    def test_fernet_roundtrip(self):
        enc = OpenSSLEncryptor("test-master-key")
        plaintext = "sk-live-test-secret"
        ciphertext = enc.encrypt(plaintext)

        assert ciphertext != plaintext
        assert enc.decrypt(ciphertext) == plaintext

    def test_fernet_wrong_key_fails(self):
        enc_a = OpenSSLEncryptor("key-a")
        enc_b = OpenSSLEncryptor("key-b")

        ciphertext = enc_a.encrypt("sk-live-test-secret")
        with pytest.raises(EncryptionError):
            enc_b.decrypt(ciphertext)

    @pytest.mark.asyncio
    async def test_provider_config_service_stores_ciphertext_not_plaintext(self):
        fake_db = MagicMock()
        fake_db.fetchrow = AsyncMock(return_value={
            "id": uuid4(),
            "tenant_id": uuid4(),
            "provider": "openai",
            "api_key_encrypted": "x",
            "settings": {},
            "is_active": True,
            "created_at": None,
            "updated_at": None,
        })

        service = ProviderConfigService(
            fake_db,
            encryption_key="test-master-key",
        )

        plain = "sk-super-secret"
        await service.set_provider_key(uuid4(), "openai", plain)

        stored_ciphertext = fake_db.fetchrow.await_args.args[2]
        assert stored_ciphertext != plain


class TestSensitiveLoggingRedaction:
    def test_json_formatter_redacts_provider_key_fields(self):
        formatter = JSONFormatter(redact_sensitive=True)
        record = logging.LogRecord(
            name="security",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="security event",
            args=(),
            exc_info=None,
        )
        record.api_key = "sk-super-secret"
        record.fernet_key = "not-for-logs"
        record.key = "also-secret"

        data = json.loads(formatter.format(record))
        assert data["api_key"] == "[REDACTED]"
        assert data["fernet_key"] == "[REDACTED]"
        assert data["key"] == "[REDACTED]"

class TestGoogleApiKeyHandling:
    @pytest.mark.asyncio
    async def test_google_adapter_does_not_put_api_key_in_query_string(self):
        from src.adapters.base import AdapterConfig
        from src.adapters.google_adapter import GoogleAdapter
        from src.core.models import ChatCompletionRequest, Message

        adapter = GoogleAdapter(AdapterConfig(api_key="GOOGLE_SUPER_SECRET"))

        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
        }

        adapter.client.post = AsyncMock(return_value=fake_response)

        req = ChatCompletionRequest(
            model="google/gemini-1.5-flash",
            messages=[Message(role="user", content="hi")],
        )
        await adapter.chat_completion(req, request_id="req1")

        called_url = adapter.client.post.await_args.args[0]
        called_headers = adapter.client.post.await_args.kwargs["headers"]
        assert "?key=" not in called_url
        assert called_headers["x-goog-api-key"] == "GOOGLE_SUPER_SECRET"

        await adapter.close()
