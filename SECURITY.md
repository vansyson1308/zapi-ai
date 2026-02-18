# Security Model & Deployment Checklist

This project is security-aware, not "secure by default in every environment." Use this checklist before sharing or deploying.

## Threat model (baseline)

In-scope risks this code mitigates:
- API key exfiltration through logs (redaction for common secret fields).
- Cross-tenant provider-key disclosure at rest (tenant provider keys encrypted using `FERNET_KEY`).
- Misconfigured permissive CORS in production (`CORS_ALLOW_ORIGINS` required and wildcard rejected in prod).
- Unsafe startup defaults (`MODE` defaults to `prod`; local mode must be explicit).

Out-of-scope / limitations:
- No HSM/KMS by default; master key management is your responsibility.
- In-memory rate limiting is not a distributed abuse control.
- Prompt injection and tool execution safety depend on your tool sandboxing policy.
- This does not replace a full secrets manager, WAF, or network segmentation.

## Safe deployment checklist

- [ ] Set `MODE=prod`.
- [ ] Set strong `FERNET_KEY` (high-entropy secret, e.g. `openssl rand -base64 32`; rotate periodically).
- [ ] Set `DATABASE_URL` and ensure TLS/auth hardening at DB layer.
- [ ] Set explicit `CORS_ALLOW_ORIGINS` allowlist (no `*`).
- [ ] Disable `USE_STUB_ADAPTERS`.
- [ ] Configure structured log shipping and restricted retention.
- [ ] Rotate provider/API keys on schedule and after any suspected leak.
- [ ] Run `make ci` on every change.

## Key management notes

- `FERNET_KEY` encrypts tenant provider secrets at rest.
- Generate key with:

```bash
openssl rand -base64 32
```

- Store key in a secrets manager in production (Vault/KMS/SSM), not in source control.
- For cloud deployments, prefer envelope encryption backed by KMS and wire a custom encryptor implementation.
