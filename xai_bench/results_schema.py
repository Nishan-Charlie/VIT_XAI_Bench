"""Canonical benchmark result record, plus validation.

One record is one ``(model, method)`` cell of the benchmark: the aggregated
score of every metric over the evaluated images, together with the provenance
needed to answer "exactly how was this number produced?".

Validation is deliberately strict and fails loudly (:class:`ValidationError`),
because a silently malformed record propagates into figures and the website.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

from xai_bench import taxonomy

SCHEMA_VERSION = "1.0.0"


class ValidationError(Exception):
    """Raised when a result record or collection violates the schema."""


@dataclass
class MetricValue:
    """Aggregated value of one metric over ``count`` images."""

    mean: float
    std: float | None = None
    count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "count": self.count}


@dataclass
class Provenance:
    """Everything needed to trace a record back to the run that produced it.

    Fields are ``None`` when the information was not recorded by the original
    run. They are never guessed — an absent value means "unknown", and the
    validator reports it rather than inventing a plausible default.
    """

    source_run: str | None = None      # run directory / summary file it came from
    config_file: str | None = None
    git_commit: str | None = None
    seed: int | None = None
    dataset: str | None = None
    num_images: int | None = None
    timestamp: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def missing_fields(self) -> list[str]:
        """Names of provenance fields that were not recorded."""
        return [k for k, v in dataclasses.asdict(self).items() if v is None]


@dataclass
class ResultRecord:
    """One ``(model, method)`` benchmark cell."""

    model: str
    method: str
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)

    @property
    def model_family(self) -> str:
        return taxonomy.model_family(self.model)

    @property
    def method_family(self) -> str:
        return taxonomy.method_family(self.method)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_label": taxonomy.MODELS[self.model].label,
            "model_family": self.model_family,
            "method": self.method,
            "method_label": taxonomy.METHODS[self.method].label,
            "method_family": self.method_family,
            "metrics": {k: v.to_dict() for k, v in sorted(self.metrics.items())},
            "provenance": self.provenance.to_dict(),
        }


def validate_record(rec: ResultRecord, *, strict_range: bool = True) -> list[str]:
    """Validate one record. Returns a list of human-readable problems.

    Checks identity against the taxonomy, and every metric for NaN/Inf, negative
    or zero sample counts, and (when ``strict_range``) violations of the metric's
    theoretical range.
    """
    problems: list[str] = []
    where = f"{rec.model}/{rec.method}"

    if rec.model not in taxonomy.MODELS:
        problems.append(f"{where}: unknown model '{rec.model}'")
    if rec.method not in taxonomy.METHODS:
        problems.append(f"{where}: unknown method '{rec.method}'")
    if not rec.metrics:
        problems.append(f"{where}: record carries no metrics")

    for name, mv in rec.metrics.items():
        spec = taxonomy.METRICS.get(name)
        if spec is None:
            problems.append(f"{where}: unknown metric '{name}'")
            continue
        if mv.mean is None or not isinstance(mv.mean, int | float):
            problems.append(f"{where}.{name}: mean is not a number ({mv.mean!r})")
            continue
        if math.isnan(mv.mean):
            problems.append(f"{where}.{name}: mean is NaN")
            continue
        if math.isinf(mv.mean):
            problems.append(f"{where}.{name}: mean is infinite")
            continue
        if mv.std is not None and (math.isnan(mv.std) or math.isinf(mv.std)):
            problems.append(f"{where}.{name}: std is not finite ({mv.std!r})")
        if mv.count is not None and mv.count <= 0:
            problems.append(f"{where}.{name}: non-positive sample count ({mv.count})")
        if strict_range and spec.value_range is not None:
            lo, hi = spec.value_range
            if lo is not None and mv.mean < lo - 1e-9:
                problems.append(f"{where}.{name}: mean {mv.mean:.4g} below minimum {lo}")
            if hi is not None and mv.mean > hi + 1e-9:
                problems.append(f"{where}.{name}: mean {mv.mean:.4g} above maximum {hi}")

    return problems


def validate_collection(records: list[ResultRecord], *, strict_range: bool = True) -> list[str]:
    """Validate a set of records, including cross-record duplicate detection."""
    problems: list[str] = []
    seen: dict[tuple, int] = {}
    for rec in records:
        problems.extend(validate_record(rec, strict_range=strict_range))
        key = (rec.model, rec.method)
        seen[key] = seen.get(key, 0) + 1
    for key, n in sorted(seen.items()):
        if n > 1:
            problems.append(f"duplicate record for {key[0]}/{key[1]} ({n} copies)")
    return problems


def assert_valid(records: list[ResultRecord], *, strict_range: bool = True) -> None:
    """Validate and raise :class:`ValidationError` listing every problem found."""
    problems = validate_collection(records, strict_range=strict_range)
    if problems:
        raise ValidationError(
            f"{len(problems)} problem(s) in benchmark results:\n  "
            + "\n  ".join(problems)
        )
