"""Seeding is what makes stochastic methods (SmoothGrad, RISE, LIME, ...) and
perturbation metrics reproducible. These tests pin the properties the runner
relies on."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xai_bench.runner import derive_seed  # noqa: E402
from xai_bench.utils import set_seed  # noqa: E402


def test_derive_seed_is_deterministic():
    assert derive_seed(42, "vit", "smoothgrad", 7) == derive_seed(42, "vit", "smoothgrad", 7)


def test_derive_seed_varies_with_every_component():
    base = derive_seed(42, "vit", "smoothgrad", 7)
    assert derive_seed(43, "vit", "smoothgrad", 7) != base
    assert derive_seed(42, "resnet50", "smoothgrad", 7) != base
    assert derive_seed(42, "vit", "rise", 7) != base
    assert derive_seed(42, "vit", "smoothgrad", 8) != base


def test_derive_seed_in_valid_numpy_range():
    """numpy rejects seeds outside [0, 2**32-1]."""
    for i in range(200):
        seed = derive_seed(i, "model", "method", i * 13)
        assert 0 <= seed < 2**32


def test_derive_seed_is_stable_across_processes():
    """Python's str hash is randomised per process; blake2b must not be.

    A per-process-varying seed would make results depend on the interpreter
    instance, silently breaking reproducibility across machines and restarts.
    """
    code = (
        "from xai_bench.runner import derive_seed;"
        "print(derive_seed(42, 'vit_base_patch16_224', 'smoothgrad', 3))"
    )
    repo_root = Path(__file__).resolve().parents[1]
    runs = set()
    for hashseed in ("0", "1", "random"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONPATH": str(repo_root)}
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True, env=env, cwd=repo_root,
        )
        runs.add(out.stdout.strip().splitlines()[-1])
    assert len(runs) == 1, f"seed changed across PYTHONHASHSEED settings: {runs}"


def test_set_seed_makes_torch_reproducible():
    set_seed(123)
    a = torch.randn(64)
    set_seed(123)
    b = torch.randn(64)
    assert torch.equal(a, b)


def test_set_seed_makes_numpy_reproducible():
    set_seed(123)
    a = np.random.rand(64)
    set_seed(123)
    b = np.random.rand(64)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_draws():
    set_seed(1)
    a = torch.randn(64)
    set_seed(2)
    b = torch.randn(64)
    assert not torch.equal(a, b)


def test_per_item_seeding_is_order_independent():
    """The property that makes a resumed run match a clean run.

    Drawing item 5's value after items 0-4 must equal drawing it first.
    """
    def draw(idx: int) -> torch.Tensor:
        set_seed(derive_seed(42, "m", "meth", idx))
        return torch.randn(8)

    sequential = [draw(i) for i in range(6)]
    assert torch.equal(draw(5), sequential[5])
