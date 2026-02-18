"""Encryption primitives for sensitive configuration data."""

from __future__ import annotations

import base64
import os
import subprocess
from abc import ABC, abstractmethod


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


class OpenSSLEncryptor(KeyEncryptor):
    """
    Baseline envelope-style encryption using OpenSSL AES-256-CBC + PBKDF2.

    This keeps the encryptor pluggable (swap for KMS/Vault later) while avoiding
    toy/base64 storage.
    """

    PREFIX = "enc:v1:"

    def __init__(self, master_key: str):
        if not master_key:
            raise EncryptionError("FERNET_KEY must be configured for provider key encryption")
        self._master_key = master_key

    def _run_openssl(self, decrypt: bool, payload: bytes) -> bytes:
        args = [
            "openssl",
            "enc",
            "-aes-256-cbc",
            "-pbkdf2",
            "-iter",
            "200000",
            "-pass",
            f"pass:{self._master_key}",
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
            raise EncryptionError("Failed to decrypt provider key with configured FERNET_KEY")
        return proc.stdout

    def encrypt(self, plaintext: str) -> str:
        if plaintext is None:
            raise EncryptionError("Cannot encrypt empty secret")
        encrypted = self._run_openssl(decrypt=False, payload=plaintext.encode("utf-8"))
        return f"{self.PREFIX}{base64.urlsafe_b64encode(encrypted).decode('utf-8')}"

    def decrypt(self, ciphertext: str) -> str:
        if not ciphertext.startswith(self.PREFIX):
            raise EncryptionError("Unsupported ciphertext format")
        data = base64.urlsafe_b64decode(ciphertext[len(self.PREFIX):].encode("utf-8"))
        plaintext = self._run_openssl(decrypt=True, payload=data)
        return plaintext.decode("utf-8")


def default_encryptor_from_env() -> KeyEncryptor:
    """Create default encryptor from environment."""
    return OpenSSLEncryptor(os.getenv("FERNET_KEY", ""))
