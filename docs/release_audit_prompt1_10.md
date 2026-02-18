# Release Audit (Prompt 1 → Prompt 10)

Date: 2026-02-18
Repo under audit: `https://github.com/vansyson1308/zapi-ai`
Target branch: `main`

## Source-of-truth check (GitHub main)

### Command
```bash
git remote add origin https://github.com/vansyson1308/zapi-ai.git && git fetch origin main
```

### Output
```text
fatal: unable to access 'https://github.com/vansyson1308/zapi-ai.git/': CONNECT tunnel failed, response 403
```

Result: **Cannot directly verify GitHub `main` from this environment due network/proxy block**.

---

## A) Branch and state verification

### 1) `git status -sb`
```text
## work
```

### 2) `git branch --show-current`
```text
work
```

### 3) `git rev-parse HEAD`
```text
50297971cf9d2e6d9831d3cb67cb8c8d6307a77d
```

### 4) `git log --oneline -n 30`
```text
5029797 chore: remove Pydantic v2 deprecation warning
270d763 Remove Pydantic v2 deprecation in FunctionParameters config
43906f3 fix: add email-validator dependency
64f45d2 docs: update README and quickstart
b0ffcf1 PRD implementation: core routing/auth/usage/observability + SDKs
...
```

---

## B) Hard gates and onboarding

### 1) `make ci`
```text
...
============================= 449 passed in 8.31s ==============================
```

### 2) `make doctor`
```text
Doctor checks passed.
Warnings:
- MODE=local: no provider keys configured. Use `USE_STUB_ADAPTERS=true make dev` or set at least one provider key.
```

### 3) `make smoke-journey`
```text
Starting 2api server in deterministic LOCAL mode on port 8011...
...
Smoke user journey completed successfully.
```

### 4) `make dev` readiness loop proof + stop
```text
Waiting for readiness at http://127.0.0.1:8000/ready
  - retry 1/40 ...
  - retry 2/40 ...
  ...
✅ Server ready: http://127.0.0.1:8000
...
^C
INFO:     Shutting down
...
```

---

## C) Artifact verification

1. **Makefile targets exist (`ci`, `doctor`, `smoke`, `smoke-journey`, `dev/run`)**
   - Confirmed in `Makefile`.

2. **Exactly one quality gate workflow job running `make ci`, no `continue-on-error`**
   - Only `.github/workflows/ci.yml` exists.
   - Contains single job `quality-gate` with `run: make ci`.
   - No `continue-on-error` found.

3. **Contract tests enforce CI contract**
   - `tests/test_contracts.py` includes tests for:
     - Makefile exposes `ci` target
     - CI runs `make ci`
     - CI does not use `continue-on-error: true`

4. **API correctness fixes**
   - `/v1/models/compare` declared before `/v1/models/{model_id:path}` in `src/api/routes/models.py`.
   - `/v1/usage` route is in `src/server.py`.
   - Regression tests exist in `tests/test_api_system.py` for compare routing and unique usage route contract.

5. **Streaming orchestration invariant**
   - Chat streaming path uses `router_instance.route_chat_stream(...)` in `src/api/routes/chat.py`.
   - Integration test exists: `test_api_streaming_uses_router_orchestration_not_direct_adapter`.

6. **Rate limit enforcement on costly endpoints + integration tests**
   - `check_rate_limits` dependency calls `check_limits(...)` and raises `TenantRateLimitedError`.
   - Chat/Embeddings/Images endpoints include `Depends(check_rate_limits)`.
   - Integration tests in `tests/test_rate_limit_integration.py` verify 429 + Retry-After.

7. **README onboarding + model examples**
   - Copy/paste section contains `make doctor -> make dev -> make smoke-journey -> make ci` sequence.
   - README includes model examples under MODEL_DOC_CONTRACT markers.

8. **Doc-drift guard**
   - `tests/test_contracts.py` includes `TestReadmeModelDocContract` validating README model examples exist in stub `/v1/models` support.

9. **Pydantic deprecation warning removed + warning count <=1**
   - `CreateAPIKeyRequest` uses `Field(..., pattern=...)` style in `src/api/management.py` (no deprecated `regex`).
   - `make ci` test run finishes with `449 passed` and no pytest warnings summary shown.

---

## D) Prompt checklist (1..10)

> Mapping used: Prompts 1..9 correspond to required artifact categories in section C; Prompt 10 corresponds to GitHub-main proof requirement.

| Prompt | Status | Evidence summary |
|---|---|---|
| Prompt 1 | PASS | Makefile gate targets exist. |
| Prompt 2 | PASS | Single CI workflow job runs `make ci`; no continue-on-error. |
| Prompt 3 | PASS | Contract tests enforce CI contract. |
| Prompt 4 | PASS | Compare route ordering and unique usage route with regression tests. |
| Prompt 5 | PASS | Streaming path uses router orchestration + integration test. |
| Prompt 6 | PASS | Rate limit dependency wired to costly endpoints + integration tests. |
| Prompt 7 | PASS | README includes onboarding chain and model examples. |
| Prompt 8 | PASS | Doc drift guard test exists for README model examples vs stub models. |
| Prompt 9 | PASS | Pydantic deprecation addressed; `make ci` shows passing tests without warnings summary. |
| Prompt 10 | **FAIL (environmental proof gap)** | Unable to fetch GitHub main due `CONNECT tunnel failed, response 403`; cannot prove remote main contains these commits from this environment. |

### GO / NO-GO
- **NO-GO for strict proof-to-GitHub-main sign-off** in this environment because remote verification is blocked.
- **Conditional GO for local branch quality**: all local gates and required artifacts pass.

### Minimal follow-up PR/task if FAIL must be closed
Smallest follow-up task: run this same audit from a network-enabled CI runner or workstation with GitHub access, then append:
1. `git fetch origin main`
2. `git rev-parse origin/main`
3. `git merge-base --is-ancestor <audited_commit> origin/main`
4. Re-run A/B/C checks against `origin/main` checkout and publish output.

