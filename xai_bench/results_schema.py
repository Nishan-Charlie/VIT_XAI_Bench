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

SCHEMA_VERSION = "1.1.0"

#: Statistics a cell can report (see :mod:`xai_bench.scoring`).
STATISTICS = ("mean", "median")


class ValidationError(Exception):
    """Raised when a result record or collection violates the schema."""


@dataclass
class MetricValue:
    """Aggregated value of one metric over ``count`` images.

    Every evaluated image is scored (:mod:`xai_bench.scoring`): an image on which
    the metric is undefined, because the method returned a constant map, gets no
    credit, or for Max-Sensitivity an unbounded value. ``n_undefined`` counts
    those images. The cell value is the ``statistic`` of the scored values: the
    median for Max-Sensitivity, whose failures are unbounded, the mean otherwise.

    A median is infinite when at least half of the images are unbounded. JSON has
    no infinity, so such a cell is stored with ``unbounded=True`` and
    ``median=None``; :attr:`value` returns ``math.inf`` for it. The mean of a
    metric with any unbounded image is infinite as well and is stored as ``None``.
    """

    mean: float | None
    std: float | None = None
    count: int | None = None
    median: float | None = None
    statistic: str = "mean"
    unbounded: bool = False
    n_undefined: int | None = None
    source: str | None = None      # where this metric came from, if not the record's run

    @property
    def value(self) -> float | None:
        """The cell value: ``math.inf`` for an unbounded cell."""
        if self.unbounded:
            return math.inf
        return self.median if self.statistic == "median" else self.mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": None if self.unbounded else self.value,
            "statistic": self.statistic,
            "unbounded": self.unbounded,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "count": self.count,
            "n_undefined": self.n_undefined,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MetricValue:
        """Read a serialised metric; schema 1.0.0 records carry only mean/std/count."""
        return cls(
            mean=raw.get("mean"),
            std=raw.get("std"),
            count=raw.get("count"),
            median=raw.get("median"),
            statistic=raw.get("statistic", "mean"),
            unbounded=bool(raw.get("unbounded", False)),
            n_undefined=raw.get("n_undefined"),
            source=raw.get("source"),
        )


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
        problems.extend(_validate_metric(f"{where}.{name}", spec, mv, strict_range))

    return problems


def _number_problem(label: str, what: str, v: Any) -> str | None:
    """Why ``v`` is not a finite number, or None if it is one."""
    if v is None or isinstance(v, bool) or not isinstance(v, int | float):
        return f"{label}: {what} is not a number ({v!r})"
    if math.isnan(v):
        return f"{label}: {what} is NaN"
    if math.isinf(v):
        return f"{label}: {what} is infinite"
    return None


def _validate_metric(label: str, spec: Any, mv: MetricValue, strict_range: bool) -> list[str]:
    if mv.statistic not in STATISTICS:
        return [f"{label}: unknown statistic {mv.statistic!r}"]
    lo, hi = spec.value_range if spec.value_range is not None else (None, None)

    # The reported statistic must be a finite number, unless the cell is unbounded,
    # which only a median of a metric without an upper bound can be.
    checked: list[tuple[str, Any]] = []
    if mv.unbounded:
        if mv.statistic != "median" or hi is not None:
            return [f"{label}: only the median of an unbounded metric can be infinite"]
    else:
        problem = _number_problem(label, mv.statistic, getattr(mv, mv.statistic))
        if problem:
            return [problem]
        checked.append((mv.statistic, getattr(mv, mv.statistic)))
    # The other statistic is optional; the mean is None when an image is unbounded.
    other = "median" if mv.statistic == "mean" else "mean"
    if getattr(mv, other) is not None:
        problem = _number_problem(label, other, getattr(mv, other))
        if problem:
            return [problem]
        checked.append((other, getattr(mv, other)))

    problems = []
    if mv.std is not None and (math.isnan(mv.std) or math.isinf(mv.std)):
        problems.append(f"{label}: std is not finite ({mv.std!r})")
    if mv.count is not None and mv.count <= 0:
        problems.append(f"{label}: non-positive sample count ({mv.count})")
    if mv.n_undefined is not None and not 0 <= mv.n_undefined <= (mv.count or mv.n_undefined):
        problems.append(f"{label}: {mv.n_undefined} undefined images out of {mv.count}")
    if strict_range:
        for what, v in checked:
            if lo is not None and v < lo - 1e-9:
                problems.append(f"{label}: {what} {v:.4g} below minimum {lo}")
            if hi is not None and v > hi + 1e-9:
                problems.append(f"{label}: {what} {v:.4g} above maximum {hi}")
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
