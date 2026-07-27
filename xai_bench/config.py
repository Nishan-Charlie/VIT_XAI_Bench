"""Run configuration: a small dataclass plus YAML/dict loading.

The benchmark is fully described by which models, methods, metrics and which
dataset (with how many images and how many seeds) to sweep. Everything else has
a sensible default so a minimal config is just a handful of lines.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RunConfig:
    # What to sweep
    models: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    # Per-metric keyword overrides, e.g. {"max_sensitivity": {"nr_samples": 5}}.
    # Keeps Quantus metric cost feasible without editing the wrappers.
    metric_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Dataset
    dataset: str = "demo"
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    num_images: int | None = None      # None = all available
    seeds: list[int] = field(default_factory=lambda: [0])

    # Execution
    device: str = "auto"                  # "auto" | "cuda" | "cpu"
    # Force deterministic cuDNN kernels. Slower, but bit-reproducible; see
    # docs/reproducibility.md for the accuracy/speed tradeoff.
    deterministic: bool = False
    # Path of the YAML this config came from, recorded for provenance.
    config_path: str | None = None
    input_size: int = 224
    output_dir: str = "results"
    run_name: str | None = None        # defaults to a timestamp
    save_saliency: bool = False           # dump raw saliency arrays (large)
    limit_methods_to_supported: bool = True  # skip e.g. rollout on CNNs silently

    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> RunConfig:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
