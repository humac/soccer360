#!/usr/bin/env python3
"""Verify requirements-docker.txt mirrors pyproject.toml dependencies."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements-docker.txt"


def parse_pyproject_deps() -> set[str]:
    """Extract dependencies from pyproject.toml [project.dependencies] array."""
    text = PYPROJECT.read_text()
    match = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.DOTALL | re.MULTILINE)
    if not match:
        print("ERROR: could not find dependencies array in pyproject.toml", file=sys.stderr)
        sys.exit(2)
    block = match.group(1)
    deps: set[str] = set()
    for line in block.splitlines():
        line = line.strip().strip(",").strip('"').strip("'").strip()
        if line:
            deps.add(line)
    return deps


def parse_requirements() -> set[str]:
    lines = REQUIREMENTS.read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}


def main() -> int:
    pyproject_deps = parse_pyproject_deps()
    req_deps = parse_requirements()
    if pyproject_deps == req_deps:
        print("OK: requirements-docker.txt is in sync with pyproject.toml")
        return 0
    only_pyproject = pyproject_deps - req_deps
    only_req = req_deps - pyproject_deps
    print("MISMATCH between pyproject.toml and requirements-docker.txt")
    if only_pyproject:
        print(f"  Only in pyproject.toml: {sorted(only_pyproject)}")
    if only_req:
        print(f"  Only in requirements-docker.txt: {sorted(only_req)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
