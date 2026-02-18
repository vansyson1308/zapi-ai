"""Offline JS SDK validation checks for CI.

Purposefully avoids npm install so CI remains deterministic without external registry access.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SDK = ROOT / "src" / "sdk" / "javascript"
PKG = SDK / "package.json"


def main() -> int:
    if not PKG.exists():
        print("ERROR: src/sdk/javascript/package.json is missing")
        return 1

    package = json.loads(PKG.read_text(encoding="utf-8"))

    required_scripts = ["build", "typecheck", "test:run"]
    for script in required_scripts:
        if script not in package.get("scripts", {}):
            print(f"ERROR: package.json scripts.{script} is required")
            return 1

    for key in ["main", "module", "types"]:
        if key not in package:
            print(f"ERROR: package.json {key} field is required")
            return 1

    source_entry = SDK / "src" / "index.ts"
    if not source_entry.exists():
        print("ERROR: JS SDK source entry src/index.ts is missing")
        return 1

    # Validate packaging metadata works and package can be packed from current tree.
    proc = subprocess.run(
        ["npm", "pack", "--dry-run", "--json"],
        cwd=str(SDK),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        print("ERROR: npm pack --dry-run failed")
        return 1


    # Run lightweight node-based sanity checks (offline-friendly):
    # - npm pack --dry-run metadata
    # - fixed SSE [DONE] parser sanity
    sanity = ROOT / "scripts" / "js_sdk_sanity.mjs"
    sanity_proc = subprocess.run(
        ["node", str(sanity)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if sanity_proc.returncode != 0:
        print(sanity_proc.stdout)
        print(sanity_proc.stderr)
        print("ERROR: node JS SDK sanity check failed")
        return 1

    print("JS SDK validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
