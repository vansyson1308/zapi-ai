"""Environment compatibility and preflight checks for local setup."""

from __future__ import annotations

import os
import shutil
import socket
import sys
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 12)


@dataclass
class DoctorResult:
    ok: bool
    messages: List[str]


def _python_version_tuple() -> Tuple[int, int, int]:
    v = sys.version_info
    return (v.major, v.minor, v.micro)


def _check_python_version(errors: List[str]) -> None:
    current = _python_version_tuple()
    if current[:2] < MIN_PYTHON or current[:2] > MAX_PYTHON:
        errors.append(
            f"Python {current[0]}.{current[1]} is unsupported. "
            f"Use Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}–{MAX_PYTHON[0]}.{MAX_PYTHON[1]}."
        )


def _require_non_empty(env: Mapping[str, str], keys: Sequence[str], errors: List[str], mode: str) -> None:
    for key in keys:
        if not env.get(key, "").strip():
            errors.append(f"MODE={mode} requires `{key}` to be set and non-empty.")


def _check_mode_requirements(env: Mapping[str, str], errors: List[str], warnings: List[str]) -> None:
    mode = env.get("MODE", "local").strip().lower()
    if mode not in {"local", "test", "prod", "production"}:
        errors.append("MODE must be one of: local, test, prod, production.")
        return

    if mode in {"prod", "production"}:
        _require_non_empty(env, ["DATABASE_URL", "FERNET_KEY", "CORS_ALLOW_ORIGINS"], errors, mode)

    if mode == "test":
        _require_non_empty(
            env,
            ["TEST_RATE_LIMIT_RPM", "TEST_DAILY_TOKEN_LIMIT", "TEST_MONTHLY_COST_LIMIT"],
            errors,
            mode,
        )

    if mode == "local":
        stub = env.get("USE_STUB_ADAPTERS", "false").strip().lower() in {"1", "true", "yes"}
        has_key = any(env.get(k, "").strip() for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"])
        if not stub and not has_key:
            warnings.append(
                "MODE=local: no provider keys configured. Use `USE_STUB_ADAPTERS=true make dev` "
                "or set at least one provider key."
            )


def _check_port_binding(env: Mapping[str, str], errors: List[str]) -> None:
    host = env.get("HOST", "0.0.0.0").strip() or "0.0.0.0"
    raw_port = env.get("PORT", "8000").strip() or "8000"

    try:
        port = int(raw_port)
    except ValueError:
        errors.append(f"PORT must be an integer, got `{raw_port}`.")
        return

    if not (0 < port < 65536):
        errors.append(f"PORT must be between 1 and 65535, got `{port}`.")
        return

    bind_host = "127.0.0.1" if host in {"0.0.0.0", "localhost", ""} else host
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_host, port))
    except OSError as exc:
        errors.append(
            f"PORT/HOST conflict: cannot bind {bind_host}:{port} ({exc}). "
            "Pick a free port, e.g. `PORT=8010 make dev`."
        )
    finally:
        sock.close()


def _check_js_tooling(env: Mapping[str, str], warnings: List[str], errors: List[str]) -> None:
    require_js = env.get("DOCTOR_REQUIRE_JS", "false").strip().lower() in {"1", "true", "yes"}
    node = shutil.which("node")
    npm = shutil.which("npm")

    if require_js:
        if not node or not npm:
            errors.append("DOCTOR_REQUIRE_JS=true requires both `node` and `npm` in PATH.")
    else:
        if not node or not npm:
            warnings.append("JS SDK checks skipped (node/npm not found).")


def run_doctor(env: Optional[Mapping[str, str]] = None) -> DoctorResult:
    env_map: Mapping[str, str] = env or os.environ
    errors: List[str] = []
    warnings: List[str] = []

    _check_python_version(errors)
    _check_mode_requirements(env_map, errors, warnings)
    _check_port_binding(env_map, errors)
    _check_js_tooling(env_map, warnings, errors)

    messages: List[str] = []
    if errors:
        messages.append("Doctor found configuration issues:")
        for i, msg in enumerate(errors, 1):
            messages.append(f"{i}. {msg}")
        messages.append("Fix the items above and rerun `make doctor`.")
    else:
        messages.append("Doctor checks passed.")

    if warnings:
        messages.append("Warnings:")
        for i, msg in enumerate(warnings, 1):
            messages.append(f"- {msg}")

    return DoctorResult(ok=not errors, messages=messages)


def main() -> int:
    result = run_doctor()
    print("\n".join(result.messages))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
