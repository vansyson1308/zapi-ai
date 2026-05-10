"""
Tests for src/security/encryption.py — keyring rotation + backward compat.
"""

from __future__ import annotations

import os
import shutil

import pytest

from src.security.encryption import (
    EncryptionError,
    KeyringEncryptor,
    KeyringEntry,
    OpenSSLEncryptor,
    _UnconfiguredEncryptor,
    default_encryptor_from_env,
    parse_keyring_env,
)


# We require openssl on PATH for these tests. CI image has it; if it's not
# available we skip the live encryption tests rather than fail.
HAS_OPENSSL = shutil.which("openssl") is not None
needs_openssl = pytest.mark.skipif(not HAS_OPENSSL, reason="openssl binary not on PATH")


# ============================================================
# parse_keyring_env
# ============================================================


def test_parse_keyring_env_empty_returns_empty_list():
    assert parse_keyring_env("") == []
    assert parse_keyring_env("   ") == []


def test_parse_keyring_env_single_entry():
    entries = parse_keyring_env("v1=secret1")
    assert len(entries) == 1
    assert entries[0].id == "v1"
    assert entries[0].secret == "secret1"


def test_parse_keyring_env_multiple_entries():
    entries = parse_keyring_env("v2=new, v1=old")
    assert [e.id for e in entries] == ["v2", "v1"]


def test_parse_keyring_env_rejects_bad_format():
    with pytest.raises(EncryptionError):
        parse_keyring_env("just-a-secret")


def test_parse_keyring_env_rejects_empty_id():
    with pytest.raises(EncryptionError):
        parse_keyring_env("=secret")


def test_parse_keyring_env_rejects_empty_secret():
    with pytest.raises(EncryptionError):
        parse_keyring_env("v1=")


def test_keyring_entry_rejects_invalid_id():
    with pytest.raises(EncryptionError):
        KeyringEntry(id="has=equals", secret="x")
    with pytest.raises(EncryptionError):
        KeyringEntry(id="has:colon", secret="x")
    with pytest.raises(EncryptionError):
        KeyringEntry(id="", secret="x")


# ============================================================
# KeyringEncryptor
# ============================================================


def test_keyring_requires_at_least_one_key():
    with pytest.raises(EncryptionError):
        KeyringEncryptor([])


def test_keyring_rejects_duplicate_ids():
    with pytest.raises(EncryptionError):
        KeyringEncryptor([
            KeyringEntry(id="v1", secret="a"),
            KeyringEntry(id="v1", secret="b"),
        ])


def test_keyring_current_is_first_entry():
    enc = KeyringEncryptor([
        KeyringEntry(id="new", secret="N"),
        KeyringEntry(id="old", secret="O"),
    ])
    assert enc.current.id == "new"
    assert enc.current_key_id() == "new"


@needs_openssl
def test_keyring_roundtrip_v2():
    enc = KeyringEncryptor([KeyringEntry(id="2026q2", secret="hunter2-new")])
    ciphertext = enc.encrypt("api-key-here")
    assert ciphertext.startswith("enc:v2:k=2026q2:")
    plaintext = enc.decrypt(ciphertext)
    assert plaintext == "api-key-here"


@needs_openssl
def test_keyring_decrypts_with_correct_key_only():
    enc_old = KeyringEncryptor([KeyringEntry(id="v1", secret="old-secret")])
    ciphertext_old = enc_old.encrypt("payload")

    # New encryptor has different key — must NOT decrypt
    enc_unrelated = KeyringEncryptor([KeyringEntry(id="v2", secret="other-secret")])
    with pytest.raises(EncryptionError):
        enc_unrelated.decrypt(ciphertext_old)

    # Multi-key encryptor that includes the old key — should decrypt
    enc_multi = KeyringEncryptor([
        KeyringEntry(id="v2", secret="other-secret"),
        KeyringEntry(id="v1", secret="old-secret"),
    ])
    assert enc_multi.decrypt(ciphertext_old) == "payload"


@needs_openssl
def test_keyring_can_decrypt_legacy_v1():
    legacy = OpenSSLEncryptor("master")
    ciphertext_v1 = legacy.encrypt("hello-legacy")
    assert ciphertext_v1.startswith("enc:v1:")

    enc = KeyringEncryptor([KeyringEntry(id="anything", secret="master")])
    assert enc.decrypt(ciphertext_v1) == "hello-legacy"


@needs_openssl
def test_reencrypt_upgrades_v1_to_v2():
    legacy = OpenSSLEncryptor("master")
    ciphertext_v1 = legacy.encrypt("rotate-me")

    enc = KeyringEncryptor([KeyringEntry(id="2026q2", secret="master")])
    new_ciphertext = enc.reencrypt(ciphertext_v1)

    assert new_ciphertext.startswith("enc:v2:k=2026q2:")
    assert enc.decrypt(new_ciphertext) == "rotate-me"


def test_keyring_rejects_malformed_v2_header():
    enc = KeyringEncryptor([KeyringEntry(id="v1", secret="x")])
    with pytest.raises(EncryptionError):
        enc.decrypt("enc:v2:")
    with pytest.raises(EncryptionError):
        enc.decrypt("enc:v2:k=v1")  # missing colon
    with pytest.raises(EncryptionError):
        enc.decrypt("enc:unknown:body")


def test_keyring_unknown_id_raises():
    enc = KeyringEncryptor([KeyringEntry(id="v1", secret="x")])
    with pytest.raises(EncryptionError):
        enc.decrypt("enc:v2:k=missing:Zm9v")


# ============================================================
# default_encryptor_from_env
# ============================================================


def test_default_uses_keyring_when_FERNET_KEYS_set(monkeypatch):
    monkeypatch.setenv("FERNET_KEYS", "v1=abc")
    monkeypatch.delenv("FERNET_KEY", raising=False)
    enc = default_encryptor_from_env()
    assert isinstance(enc, KeyringEncryptor)
    assert enc.current.id == "v1"


def test_default_falls_back_to_legacy_FERNET_KEY(monkeypatch):
    monkeypatch.delenv("FERNET_KEYS", raising=False)
    monkeypatch.setenv("FERNET_KEY", "legacy-master")
    enc = default_encryptor_from_env()
    assert isinstance(enc, KeyringEncryptor)
    assert enc.current.id == "legacy"
    assert enc.current.secret == "legacy-master"


def test_default_returns_unconfigured_when_nothing_set(monkeypatch):
    monkeypatch.delenv("FERNET_KEYS", raising=False)
    monkeypatch.delenv("FERNET_KEY", raising=False)
    enc = default_encryptor_from_env()
    assert isinstance(enc, _UnconfiguredEncryptor)
    with pytest.raises(EncryptionError):
        enc.encrypt("anything")
    with pytest.raises(EncryptionError):
        enc.decrypt("anything")


def test_default_prefers_FERNET_KEYS_over_legacy(monkeypatch):
    monkeypatch.setenv("FERNET_KEYS", "newkey=newsecret")
    monkeypatch.setenv("FERNET_KEY", "old-fallback")
    enc = default_encryptor_from_env()
    assert isinstance(enc, KeyringEncryptor)
    assert enc.current.id == "newkey"
