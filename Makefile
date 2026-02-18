.PHONY: ci bootstrap quality test-contracts smoke smoke-journey sdk sdk-python sdk-js doctor test compile lint typecheck format-check dev run clean

PYTHON ?= python3
PIP := $(PYTHON) -m pip

# Deterministic repository quality gate (must stay green in CI)
ci: bootstrap quality

bootstrap:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

quality: compile test-contracts sdk smoke test

compile:
	$(PYTHON) -m py_compile src/server.py

# Full contract checks for documented behavior and repository CI contract
test-contracts:
	pytest tests/test_contracts.py -v --tb=short

# SDK validation and packaging checks
sdk: sdk-python sdk-js

sdk-python:
	$(PIP) install -e src/sdk/python --no-build-isolation
	pytest tests/test_sdk_system.py -v --tb=short

sdk-js:
	$(PYTHON) scripts/validate_js_sdk.py

# Deterministic real-HTTP smoke checks (stub adapter, no external network)
smoke:
	pytest tests/test_smoke_api.py -v --tb=short

# End-to-end user journey smoke script (server + 3 endpoint calls)
smoke-journey:
	bash scripts/smoke_user_journey.sh


# Environment compatibility preflight
doctor:
	$(PYTHON) scripts/doctor.py

# Full repository test suite (provider calls are mocked/stubbed)
test:
	pytest tests -v --tb=short

# Local developer run command (minimal deterministic mode, waits until /ready)
run dev:
	PYTHON_BIN=$(PYTHON) HOST=127.0.0.1 BIND_HOST=0.0.0.0 PORT=8000 bash scripts/dev_run.sh

# Optional local developer commands (not part of green gate yet)
format-check:
	black --check src tests

lint:
	ruff check src tests

typecheck:
	mypy src --ignore-missing-imports

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
