"""xai_bench — a controlled benchmark of attribution methods on ViTs and CNNs.

The package is organised as four registries — models, methods, datasets,
metrics — plus a runner that sweeps the (model x method x image) matrix and
writes tidy, provenance-carrying result rows.

Importing :mod:`xai_bench` is deliberately cheap. The registries pull in torch,
timm and quantus, so they are populated lazily on first attribute access. That
keeps result analysis and schema validation usable in a lightweight environment
(and in CI) without installing the full deep-learning stack::

    from xai_bench import taxonomy, results_schema   # no torch needed
    from xai_bench import METHODS                    # imports torch on demand
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.2.0"

__all__ = [
    "MODELS",
    "METHODS",
    "DATASETS",
    "METRICS",
    "Registry",
    "taxonomy",
    "results_schema",
]

#: Attributes whose retrieval requires the registries to be populated.
_REGISTRY_ATTRS = frozenset({"MODELS", "METHODS", "DATASETS", "METRICS", "Registry"})

#: Submodules exposed lazily as plain attributes.
_LAZY_MODULES = frozenset({"taxonomy", "results_schema", "config", "runner", "registry"})

_REGISTRIES_LOADED = False


def _load_registries() -> None:
    """Import the sub-packages so their built-in entries self-register."""
    global _REGISTRIES_LOADED
    if _REGISTRIES_LOADED:
        return
    # Set before importing: the sub-modules import xai_bench themselves, so a
    # re-entrant call would otherwise recurse.
    _REGISTRIES_LOADED = True
    for name in ("models", "methods", "datasets", "metrics"):
        importlib.import_module(f"xai_bench.{name}")


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access."""
    if name in _REGISTRY_ATTRS:
        registry_mod = importlib.import_module("xai_bench.registry")
        _load_registries()
        return getattr(registry_mod, name)
    if name in _LAZY_MODULES:
        return importlib.import_module(f"xai_bench.{name}")
    raise AttributeError(f"module 'xai_bench' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
