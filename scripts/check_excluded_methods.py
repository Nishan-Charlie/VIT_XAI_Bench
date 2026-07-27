"""Fail if a separate paper's method reappears in this repository.

HiLRP is the novel contribution of a different paper. Its implementation,
configs, figures and result rows were removed from this benchmark (see
``docs/scientific_integrity.md``). This check is the guard that keeps them out:
it scans code, configuration and result data for references and exits non-zero
if any are found.

A small set of files legitimately name the method *in order to document its
removal* — this record, the README's integrity section, the quarantine notice,
and the exclusion logic itself. Those are allow-listed by path.

Usage::

    python scripts/check_excluded_methods.py
    python scripts/check_excluded_methods.py --verbose
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Case-insensitive patterns that indicate a separate paper's method.
FORBIDDEN = re.compile(r"hi[-_]?lrp", re.IGNORECASE)

#: Directories never scanned.
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache",
    ".ruff_cache", ".mypy_cache", "build", "dist", ".claude",
}

#: Files that document the removal and must therefore be allowed to name it.
ALLOWED = {
    Path("README.md"),
    Path("docs/scientific_integrity.md"),
    Path("figures/_unverified/README.md"),
    Path(".github/workflows/ci.yml"),
    Path("scripts/check_excluded_methods.py"),
    Path("scripts/build_results.py"),
    Path("scripts/validate_results.py"),
    Path("scripts/smoke_test.py"),
    Path("tests/test_taxonomy.py"),
    Path("tests/test_excluded_methods.py"),
}

#: Only text-like files are scanned; binaries can't be meaningfully checked.
TEXT_SUFFIXES = {
    ".py", ".yaml", ".yml", ".json", ".csv", ".md", ".txt", ".toml", ".cfg",
    ".ini", ".sh", ".bat", ".html", ".css", ".js", ".ipynb", ".tex", ".xml",
    "", ".drawio",
}


def _candidate_files(root: Path) -> list[Path]:
    """Files the repository actually publishes.

    Uses ``git ls-files`` so the check matches what a clone would contain:
    untracked scratch files and gitignored artefacts are not published and are
    therefore out of scope. Falls back to a filesystem walk outside a checkout.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=root, capture_output=True, text=True, timeout=30, check=False,
        )
        if out.returncode == 0:
            return [root / line for line in out.stdout.splitlines() if line]
    except (OSError, subprocess.SubprocessError):
        pass
    return [p for p in root.rglob("*") if p.is_file()]


def scan(root: Path = REPO_ROOT) -> list[tuple[Path, int, str]]:
    """Return ``(relative_path, line_number, line)`` for every violation."""
    hits: list[tuple[Path, int, str]] = []
    for path in _candidate_files(root):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(root)
        if rel in ALLOWED:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not FORBIDDEN.search(text):
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN.search(line):
                hits.append((rel, i, line.strip()[:160]))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true", help="list scanned allow-list")
    args = ap.parse_args()

    if args.verbose:
        print("Allow-listed (documents the removal):")
        for p in sorted(ALLOWED):
            print(f"  {p}")
        print()

    hits = scan()
    if hits:
        print(f"FAIL: {len(hits)} reference(s) to a separate paper's method:\n")
        for rel, line_no, line in hits:
            print(f"  {rel}:{line_no}: {line}")
        print(
            "\nHiLRP belongs to a different paper and must not appear in this "
            "benchmark.\nSee docs/scientific_integrity.md."
        )
        return 1

    print("OK: no separate-paper references outside the documented-removal record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
