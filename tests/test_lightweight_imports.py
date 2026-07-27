"""Result tooling must import without the deep-learning stack.

The GitHub Pages deploy and the CI results-integrity job install no torch,
numpy or timm. If an import chain ever pulls them back in, those jobs break —
so this pins the boundary explicitly rather than discovering it in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Packages absent from the lightweight CI jobs.
HEAVY = [
    "torch", "timm", "quantus", "captum", "numpy",
    "pandas", "matplotlib", "scipy", "PIL",
]

_BLOCKER = f"""
import sys
BLOCK = {HEAVY!r}
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCK:
            raise ImportError('blocked: ' + name)
        return None
sys.meta_path.insert(0, Blocker())
"""


def _run_without_heavy_deps(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + body],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )


def test_blocker_actually_blocks():
    """Guard against a no-op test: the blocker must really prevent imports."""
    out = _run_without_heavy_deps("import numpy")
    assert out.returncode != 0


@pytest.mark.parametrize("module", [
    "xai_bench.taxonomy",
    "xai_bench.results_schema",
    "xai_bench.utils.provenance",
])
def test_module_imports_without_heavy_deps(module):
    out = _run_without_heavy_deps(f"import {module}")
    assert out.returncode == 0, f"{module} pulled in a heavy dependency:\n{out.stderr}"


def test_utils_package_import_is_light():
    """`from xai_bench.utils import provenance` must not import image/seed."""
    out = _run_without_heavy_deps("from xai_bench.utils import provenance")
    assert out.returncode == 0, out.stderr


@pytest.mark.parametrize("script", ["validate_results.py", "export_web_data.py"])
def test_pages_scripts_run_without_heavy_deps(script):
    """The scripts the Pages workflow runs, in the environment it runs them in."""
    out = _run_without_heavy_deps(
        "import runpy, sys;"
        f"sys.argv=['{script}'];"
        f"runpy.run_path('scripts/{script}', run_name='__main__')"
    )
    assert out.returncode == 0, f"scripts/{script} failed:\n{out.stderr[-2000:]}"


def test_heavy_imports_still_work_normally():
    """The lazy boundary must not break ordinary use."""
    from xai_bench.utils import set_seed

    set_seed(0)
