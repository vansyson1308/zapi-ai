"""Deterministic real-HTTP smoke tests for the gateway."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]


def _random_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def _wait_ready(base_url: str, timeout_s: float = 12.0) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    return
        except Exception as exc:  # pragma: no cover - transient startup race
            last_error = exc
        time.sleep(0.2)
    raise AssertionError(f"Server did not become healthy in {timeout_s}s: {last_error}")


def test_gateway_smoke_real_http() -> None:
    """Start gateway on random port and validate core OpenAI-compatible HTTP behavior."""
    port = _random_free_port()
    base_url = f"http://127.0.0.1:{port}"
    test_api_key = "sk-test-do-not-leak"

    env = os.environ.copy()
    env.update(
        {
            "MODE": "local",
            "PORT": str(port),
            "USE_STUB_ADAPTERS": "true",
            "OPENAI_API_KEY": test_api_key,
            "LOG_LEVEL": "INFO",
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
        "info",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output = ""
    try:
        _wait_ready(base_url)

        headers = {"Authorization": "Bearer 2api_smoke", "Content-Type": "application/json"}

        with httpx.Client(timeout=10.0) as client:
            health = client.get(f"{base_url}/health")
            print(f"health={health.status_code}")
            assert health.status_code == 200
            h = health.json()
            assert "status" in h
            assert "providers" in h

            ready = client.get(f"{base_url}/ready")
            print(f"ready={ready.status_code}")
            assert ready.status_code == 200
            assert ready.json().get("status") == "ready"

            models = client.get(f"{base_url}/v1/models", headers=headers)
            print(f"models={models.status_code}")
            assert models.status_code == 200
            models_json = models.json()
            assert models_json["object"] == "list"
            assert isinstance(models_json["data"], list)

            chat = client.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            )
            print(f"chat_non_stream={chat.status_code}")
            assert chat.status_code == 200
            chat_json = chat.json()
            assert chat_json["object"] == "chat.completion"
            assert "choices" in chat_json and len(chat_json["choices"]) > 0
            assert chat_json["choices"][0]["message"]["role"] == "assistant"

            embeddings = client.post(
                f"{base_url}/v1/embeddings",
                headers=headers,
                json={"model": "openai/text-embedding-3-small", "input": "hello"},
            )
            print(f"embeddings={embeddings.status_code}")
            assert embeddings.status_code == 200
            emb_json = embeddings.json()
            assert emb_json["object"] == "list"
            assert emb_json["data"][0]["object"] == "embedding"

            usage = client.get(f"{base_url}/v1/usage", headers=headers)
            print(f"usage={usage.status_code}")
            assert usage.status_code == 200
            usage_json = usage.json()
            assert usage_json["object"] == "usage"
            assert isinstance(usage_json["tenant_id"], str)

            models_compare = client.get(
                f"{base_url}/v1/models/compare",
                headers=headers,
                params={
                    "models": ["openai/gpt-4o-mini", "openai/text-embedding-3-small"],
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            )
            print(f"models_compare={models_compare.status_code}")
            assert models_compare.status_code == 200

            stream_done = False
            stream_chunks = []
            with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "stream please"}],
                    "stream": True,
                },
            ) as stream_resp:
                print(f"chat_stream={stream_resp.status_code}")
                assert stream_resp.status_code == 200
                assert stream_resp.headers["content-type"].startswith("text/event-stream")
                for line in stream_resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        payload = line[6:]
                        stream_chunks.append(payload)
                        if payload == "[DONE]":
                            stream_done = True
                            break

            assert len(stream_chunks) >= 2
            assert stream_done is True
            print(f"stream_done={stream_done} chunks={len(stream_chunks)}")

    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

        if proc.stdout:
            output = proc.stdout.read()

    # Ensure no raw secret leaked in logs
    assert test_api_key not in output
