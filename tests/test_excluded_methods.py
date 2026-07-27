"""The benchmark must not ship another paper's method.

These tests are the regression guard for the removal recorded in
docs/scientific_integrity.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_excluded_methods import FORBIDDEN, scan  # noqa: E402


def test_no_excluded_method_references_in_repository():
    hits = scan()
    assert not hits, "separate-paper references found:\n" + "\n".join(
        f"  {rel}:{line_no}: {line}" for rel, line_no, line in hits
    )


@pytest.mark.parametrize("text", ["hilrp", "HiLRP", "Hi-LRP", "hi_lrp", "HI-LRP"])
def test_pattern_catches_every_spelling(text):
    assert FORBIDDEN.search(text)


@pytest.mark.parametrize("text", ["attnlrp", "AttnLRP", "lrp", "CP-LRP", "zennit"])
def test_pattern_does_not_catch_legitimate_lrp_baselines(text):
    """AttnLRP and CP-LRP are published baselines and must survive the guard."""
    assert not FORBIDDEN.search(text)


def test_excluded_method_not_in_taxonomy():
    from xai_bench import taxonomy

    assert "hilrp" not in taxonomy.METHODS


def test_published_lrp_baseline_is_retained():
    """Removing HiLRP must not have removed the AttnLRP baseline with it."""
    from xai_bench import taxonomy

    assert "attnlrp" in taxonomy.METHODS
    assert (REPO_ROOT / "xai_bench" / "methods" / "vit_lrp_backend.py").exists()
