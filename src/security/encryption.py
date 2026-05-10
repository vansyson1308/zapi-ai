"""
Encryption primitives for sensitive configuration data.

Supports multi-version keyrings so we can rotate the master key without
re-encrypting everything atomically. Ciphertexts produced after this change
embed the version identifier as part of the prefix:

    enc:v1:<base64>      ← legacy single-key format (still decryptable)
    enc:vN:k=<id>:<b64>  ← new multi-key format; id resolves to a keyring entry

A KeyringEncryptor encrypts new payloads with the *current* (highest-priority)
key and decrypts using whichever key the ciphertext names. The rotation flow:

  1. Add new key as `current` (e.g. id="2026q2") via FERNET_KEYS env.
  2. Re-encrypt existing rows lazily or via scripts/rotate_keys.py (offline batch).
  3. Once nothing references the old key, drop it from FERNET_KEYS.

Format details:
- FERNET_KEY (legacy): single secret. Continues to work but new ciphertext
  uses the new format with id="legacy".
- FERNET_KEYS: comma-separated `id=secret` pairs. The first entry is the
  *current* key. Example:
      FERNET_KEYS=2026q2=hunter2-new,2026q1=hunter2-old

Both env vars may coexist; FERNET_KEYS wins for encryption, both can decrypt.
"""

from __future__ import annotations

import base64
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


class EncryptionError(ValueError):
    """Raised when encryption/decryption fails."""


class KeyEncryptor(ABC):
    """Interface for pluggable secret-at-rest encryption."""

    @abstractmethod
    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext and return ciphertext."""

    @abstractmethod
    def decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext and return plaintext."""

    def current_key_id(self) -> str:
        """Return the identifier of the key used for new encryptions."""
        return "default"


def _run_openssl_encrypt(master_key: str, decrypt: bool, payload: bytes) -> bytes:
    """Run openssl enc with consistent options. Module-level so it's reusable."""
    args = [
        "openssl",
        "enc",
        "-aes-256-cbc",
        "-pbkdf2",
        "-iter",
        "200000",
        "-pass",
        f"pass:{master_key}",
    ]
    if decrypt:
        args.insert(3, "-d")

    try:
        proc = subprocess.run(
            args,
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise EncryptionError("OpenSSL binary not found") from exc

    if proc.returncode != 0:
        raise EncryptionError("Failed to encrypt/decrypt with configured key")
    return proc.stdout


class OpenSSLEncryptor(KeyEncryptor):
    """
    Baseline envelope-style encryption using OpenSSL AES-256-CBC + PBKDF2.

    Single-key implementation kept for backward compatibility. New deployments
    should prefer KeyringEncryptor.
    """

    PREFIX = "enc:v1:"

    def __init__(self, master_key: str):
        if not master_key:
            raise EncryptionError("FERNET_KEY must be configured for provider key encryption")
        self._master_key = master_key

    def encrypt(self, plaintext: str) -> str:
        if plaintext is None:
            raise EncryptionError("Cannot encrypt empty secret")
        encrypted = _run_openssl_encrypt(self._master_key, decrypt=False, payload=plaintext.encode("utf-8"))
        return f"{self.PREFIX}{base64.urlsafe_b64encode(encrypted).decode('utf-8')}"

    def decrypt(self, ciphertext: str) -> str:
        if not ciphertext.startswith(self.PREFIX):
            raise EncryptionError("Unsupported ciphertext format")
        data = base64.urlsafe_b64decode(ciphertext[len(self.PREFIX):].encode("utf-8"))
        plaintext = _run_openssl_encrypt(self._master_key, decrypt=True, payload=data)
        return plaintext.decode("utf-8")


@dataclass
class KeyringEntry:
    """One entry in the keyring."""

    id: str
    secret: str

    def __post_init__(self) -> None:
        if not self.id or "=" in self.id or ":" in self.id:
            raise EncryptionError(
                f"invalid keyring id '{self.id}': must be non-empty and contain no '=' or ':'"
            )
        if not self.secret:
            raise EncryptionError(f"empty secret for keyring id '{self.id}'")


class KeyringEncryptor(KeyEncryptor):
    """
    Multi-key encryptor supporting rotation.

    The first entry in `keys` is the current encryption key. Decryption tries
    the key named in the ciphertext header; if missing, falls back to the
    legacy single-key v1 format using the first key (backwards compatibility).
    """

    PREFIX_V2 = "enc:v2:"
    LEGACY_PREFIX_V1 = "enc:v1:"

    def __init__(self, keys: List[KeyringEntry]) -> None:
        if not keys:
            raise EncryptionError("KeyringEncryptor requires at least one key")
        self._keys: List[KeyringEntry] = keys
        self._by_id: Dict[str, KeyringEntry] = {k.id: k for k in keys}
        if len(self._by_id) != len(keys):
            raise EncryptionError("duplicate keyring ids are not allowed")

    @property
    def current(self) -> KeyringEntry:
        return self._keys[0]

    def current_key_id(self) -> str:
        return self.current.id

    def encrypt(self, plaintext: str) -> str:
        if plaintext is None:
            raise EncryptionError("Cannot encrypt empty secret")
        encrypted = _run_openssl_encrypt(self.current.secret, decrypt=False, payload=plaintext.encode("utf-8"))
        b64 = base64.urlsafe_b64encode(encrypted).decode("utf-8")
        return f"{self.PREFIX_V2}k={self.current.id}:{b64}"

    def decrypt(self, ciphertext: str) -> str:
        if ciphertext.startswith(self.PREFIX_V2):
            body = ciphertext[len(self.PREFIX_V2):]
            # Expect form: k=<id>:<b64>
            if not body.startswith("k="):
                raise EncryptionError("Malformed v2 ciphertext header")
            id_end = body.find(":")
            if id_end == -1:
                raise EncryptionError("Malformed v2 ciphertext: missing ':' after id")
            key_id = body[2:id_end]
            payload_b64 = body[id_end + 1:]
            if not payload_b64:
                raise EncryptionError("Malformed v2 ciphertext: empty payload body")
            if not key_id:
                raise EncryptionError("Malformed v2 ciphertext: empty key id")
            entry = self._by_id.get(key_id)
            if entry is None:
                raise EncryptionError(f"Unknown key id '{key_id}'; available={list(self._by_id)}")
            try:
                data = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
            except Exception as exc:
                raise EncryptionError("Failed to decode v2 ciphertext base64") from exc
            if not data:
                raise EncryptionError("Malformed v2 ciphertext: decoded payload is empty")
            plaintext = _run_openssl_encrypt(entry.secret, decrypt=True, payload=data)
            return plaintext.decode("utf-8")

        if ciphertext.startswith(self.LEGACY_PREFIX_V1):
            # Try every key in the ring (oldest single-key deployments)
            try:
                data = base64.urlsafe_b64decode(
                    ciphertext[len(self.LEGACY_PREFIX_V1):].encode("utf-8")
                )
            except Exception as exc:
                raise EncryptionError("Failed to decode v1 ciphertext base64") from exc
            last_error: Optional[Exception] = None
            for entry in self._keys:
                try:
                    plaintext = _run_openssl_encrypt(entry.secret, decrypt=True, payload=data)
                    return plaintext.decode("utf-8")
                except EncryptionError as exc:
                    last_error = exc
                    continue
            raise EncryptionError(
                f"No keyring entry could decrypt v1 ciphertext: {last_error}"
            )

        raise EncryptionError("Unsupported ciphertext format")

    def reencrypt(self, ciphertext: str) -> str:
        """Decrypt with whatever key matches, then re-encrypt with current."""
        plaintext = self.decrypt(ciphertext)
        return self.encrypt(plaintext)


def parse_keyring_env(value: str) -> List[KeyringEntry]:
    """
    Parse `FERNET_KEYS` env var format: `id=secret,id2=secret2,...`.

    Whitespace around commas is tolerated. An empty/whitespace-only string
    returns an empty list (caller should fall back).
    """
    entries: List[KeyringEntry] = []
    if not value or not value.strip():
        return entries
    for raw in value.split(","):
        chunk = raw.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise EncryptionError(
                f"invalid FERNET_KEYS entry '{chunk}': expected 'id=secret'"
            )
        key_id, secret = chunk.split("=", 1)
        entries.append(KeyringEntry(id=key_id.strip(), secret=secret.strip()))
    return entries


def default_encryptor_from_env() -> KeyEncryptor:
    """
    Create encryptor from environment.

    Resolution order:
      1. FERNET_KEYS (multi-key) → KeyringEncryptor
      2. FERNET_KEY (legacy single-key) → KeyringEncryptor with id="legacy"
         so newly-created ciphertexts also use the rotatable v2 format.
      3. Neither → raises EncryptionError on first encryption attempt.
    """
    multi = os.getenv("FERNET_KEYS", "")
    entries = parse_keyring_env(multi)
    if entries:
        return KeyringEncryptor(entries)

    legacy = os.getenv("FERNET_KEY", "")
    if legacy:
        return KeyringEncryptor([KeyringEntry(id="legacy", secret=legacy)])

    # Empty: return a sentinel that fails loudly on encrypt() but lets the
    # process boot in pure stub mode.
    return _UnconfiguredEncryptor()


class _UnconfiguredEncryptor(KeyEncryptor):
    """Encryptor that raises on use; returned when no key env vars are set."""

    def encrypt(self, plaintext: str) -> str:  # noqa: D401
        raise EncryptionError(
            "FERNET_KEY/FERNET_KEYS not configured; cannot encrypt secrets"
        )

    def decrypt(self, ciphertext: str) -> str:  # noqa: D401
        raise EncryptionError(
            "FERNET_KEY/FERNET_KEYS not configured; cannot decrypt secrets"
        )
