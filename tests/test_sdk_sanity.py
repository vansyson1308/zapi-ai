"""Offline-friendly SDK sanity checks against deterministic local stub server."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]


def _random_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _wait_health(base_url: str, timeout_s: float = 12.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    return
        except Exception as exc:  # pragma: no cover - startup race
            last_error = exc
        time.sleep(0.2)
    raise AssertionError(f"SDK sanity server failed to become healthy in {timeout_s}s: {last_error}")


def test_python_sdk_import_and_chat_non_stream_and_stream() -> None:
    """Python SDK should import and handle one non-stream + one stream request."""
    # Entrypoint import sanity
    pkg = import_module("src.sdk.python.twoapi")
    assert hasattr(pkg, "TwoAPI")

    client_mod = import_module("src.sdk.python.twoapi.client")
    TwoAPI = getattr(client_mod, "TwoAPI")

    port = _random_free_port()
    base_url = f"http://127.0.0.1:{port}/v1"

    env = os.environ.copy()
    env.update(
        {
            "MODE": "local",
            "USE_STUB_ADAPTERS": "true",
            "OPENAI_API_KEY": "sk-test-do-not-leak",
            "PORT": str(port),
            "LOG_LEVEL": "ERROR",
        }
    )

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "error",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_health(f"http://127.0.0.1:{port}")

        with TwoAPI(api_key="2api_sdk_sanity", base_url=base_url, timeout=10.0) as client:
            non_stream = client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "hello"}],
            )
            assert non_stream.object == "chat.completion"
            assert non_stream.choices
            assert non_stream.choices[0].message.role == "assistant"

            stream = client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "stream please"}],
                stream=True,
            )

            chunks = list(stream)
            assert chunks, "Expected at least one stream chunk before terminal [DONE]"
            assert all(chunk.object == "chat.completion.chunk" for chunk in chunks)
            # Stub stream currently emits one final chunk then [DONE]; iterator termination proves [DONE] handling.
            assert chunks[-1].choices[0].finish_reason in {"stop", None}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
