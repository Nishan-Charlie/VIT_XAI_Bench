"""The taxonomy is the single source of truth for names, so it must stay
internally consistent and in sync with the runtime registries."""

from __future__ import annotations

import pytest

from xai_bench import taxonomy


def test_every_model_has_a_known_family():
    for key, spec in taxonomy.MODELS.items():
        assert spec.family in taxonomy.ARCH_FAMILIES, f"{key} has family {spec.family!r}"


def test_every_method_has_a_known_family():
    for key, spec in taxonomy.METHODS.items():
        assert spec.family in taxonomy.METHOD_FAMILIES, f"{key} has family {spec.family!r}"


def test_every_metric_has_a_known_dimension():
    for key, spec in taxonomy.METRICS.items():
        assert spec.dimension in taxonomy.DIMENSIONS, f"{key} -> {spec.dimension!r}"


def test_all_architecture_families_are_represented():
    """The benchmark's claim is cross-architecture; every family needs a model."""
    covered = {s.family for s in taxonomy.MODELS.values()}
    assert covered == set(taxonomy.ARCH_FAMILIES)


def test_all_method_families_are_represented():
    covered = {s.family for s in taxonomy.METHODS.values()}
    assert covered == set(taxonomy.METHOD_FAMILIES)


def test_keys_match_their_spec_key():
    for key, spec in taxonomy.MODELS.items():
        assert key == spec.key
    for key, spec in taxonomy.METHODS.items():
        assert key == spec.key
    for key, spec in taxonomy.METRICS.items():
        assert key == spec.key


def test_labels_are_unique():
    labels = [s.label for s in taxonomy.MODELS.values()]
    assert len(labels) == len(set(labels))
    labels = [s.label for s in taxonomy.METHODS.values()]
    assert len(labels) == len(set(labels))


def test_lookup_helpers_handle_unknown_keys():
    assert taxonomy.model_family("not_a_model") == "unknown"
    assert taxonomy.method_family("not_a_method") == "unknown"


def test_metrics_by_dimension_partitions_all_metrics():
    flat = [m for keys in taxonomy.METRICS_BY_DIMENSION.values() for m in keys]
    assert sorted(flat) == sorted(taxonomy.METRICS)


@pytest.mark.parametrize("method_key", sorted(taxonomy.METHODS))
def test_taxonomy_methods_are_registered(method_key):
    """A method named in the taxonomy must actually exist in the registry."""
    pytest.importorskip("timm")
    import xai_bench.methods  # noqa: F401  (registers on import)
    from xai_bench.registry import METHODS

    assert method_key in METHODS


@pytest.mark.parametrize("model_key", sorted(taxonomy.MODELS))
def test_taxonomy_models_are_registered(model_key):
    pytest.importorskip("timm")
    import xai_bench.models  # noqa: F401
    from xai_bench.registry import MODELS

    assert model_key in MODELS


def test_no_removed_separate_paper_method_is_registered():
    """HiLRP belongs to a different paper and must not be part of this benchmark."""
    pytest.importorskip("timm")
    import xai_bench.methods  # noqa: F401
    from xai_bench.registry import METHODS

    assert "hilrp" not in METHODS
    assert "hilrp" not in taxonomy.METHODS
