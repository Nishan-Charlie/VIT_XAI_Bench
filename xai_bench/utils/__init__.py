"""Shared utilities.

Imports are lazy (PEP 562). ``provenance`` depends only on the standard library,
so tooling that merely records or validates results — the website export, the CI
results-integrity job, the Pages deploy — can import it without pulling in
numpy, torch and PIL via ``image``/``seed``::

    from xai_bench.utils import provenance   # stdlib only
    from xai_bench.utils import set_seed     # imports numpy + torch on demand
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_transform",
    "denormalize",
    "load_image_tensor",
    "normalize_map",
    "provenance",
    "resize_map",
    "set_seed",
    "to_uint8_image",
]

#: attribute name -> submodule that provides it
_EXPORTS = {
    "set_seed": "seed",
    "IMAGENET_MEAN": "image",
    "IMAGENET_STD": "image",
    "build_transform": "image",
    "denormalize": "image",
    "load_image_tensor": "image",
    "normalize_map": "image",
    "resize_map": "image",
    "to_uint8_image": "image",
}

_SUBMODULES = {"provenance", "seed", "image"}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access."""
    if name in _SUBMODULES:
        return importlib.import_module(f"xai_bench.utils.{name}")
    if name in _EXPORTS:
        module = importlib.import_module(f"xai_bench.utils.{_EXPORTS[name]}")
        return getattr(module, name)
    raise AttributeError(f"module 'xai_bench.utils' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
