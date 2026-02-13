#!/usr/bin/env python3
"""Verify requirements-docker.txt mirrors pyproject.toml dependencies."""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        print(
            "ERROR: needs tomllib (Python 3.11+) or tomli.\n"
            "  Install: pip install tomli",
            file=sys.stderr,
        )
        sys.exit(2)

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements-docker.txt"


def _normalize(dep: str) -> str:
    """Normalize a dependency string for reliable comparison."""
    return " ".join(dep.split())


def parse_pyproject_deps() -> set[str]:
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    return {_normalize(d) for d in data["project"]["dependencies"]}


def parse_requirements() -> set[str]:
    lines = REQUIREMENTS.read_text().splitlines()
    return {
        _normalize(line)
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    }


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
