"""Validation must fail loudly on malformed results — a silently bad record
propagates into the paper tables, the figures and the website."""

from __future__ import annotations

import pytest

from xai_bench.results_schema import (
    MetricValue,
    Provenance,
    ResultRecord,
    ValidationError,
    assert_valid,
    validate_collection,
    validate_record,
)


def _record(**metric_means) -> ResultRecord:
    return ResultRecord(
        model="resnet50",
        method="grad_cam",
        metrics={k: MetricValue(mean=v, std=0.1, count=50) for k, v in metric_means.items()},
    )


def test_valid_record_has_no_problems():
    assert validate_record(_record(pointing_game=0.85)) == []


def test_nan_metric_is_rejected():
    problems = validate_record(_record(pointing_game=float("nan")))
    assert any("NaN" in p for p in problems)


def test_infinite_metric_is_rejected():
    problems = validate_record(_record(pointing_game=float("inf")))
    assert any("infinite" in p for p in problems)


def test_out_of_range_metric_is_rejected():
    """Pointing game is a fraction; 1.4 signals a broken aggregation."""
    problems = validate_record(_record(pointing_game=1.4))
    assert any("above maximum" in p for p in problems)

    problems = validate_record(_record(pointing_game=-0.2))
    assert any("below minimum" in p for p in problems)


def test_range_check_can_be_relaxed():
    assert validate_record(_record(pointing_game=1.4), strict_range=False) == []


def test_unbounded_metric_accepts_large_values():
    """Max-sensitivity has no upper bound, so a large value is legitimate."""
    assert validate_record(_record(max_sensitivity=42.0)) == []


def test_negative_unbounded_metric_still_respects_lower_bound():
    problems = validate_record(_record(max_sensitivity=-1.0))
    assert any("below minimum" in p for p in problems)


def test_unknown_model_and_method_are_rejected():
    rec = ResultRecord(model="nope", method="also_nope",
                       metrics={"pointing_game": MetricValue(0.5)})
    problems = validate_record(rec)
    assert any("unknown model" in p for p in problems)
    assert any("unknown method" in p for p in problems)


def test_unknown_metric_is_rejected():
    rec = _record()
    rec.metrics["made_up_metric"] = MetricValue(0.5)
    assert any("unknown metric" in p for p in validate_record(rec))


def test_record_with_no_metrics_is_rejected():
    assert any("no metrics" in p for p in validate_record(_record()))


def test_non_positive_count_is_rejected():
    rec = ResultRecord(model="resnet50", method="grad_cam",
                       metrics={"pointing_game": MetricValue(0.5, 0.1, 0)})
    assert any("non-positive sample count" in p for p in validate_record(rec))


def test_duplicate_records_are_detected():
    records = [_record(pointing_game=0.8), _record(pointing_game=0.9)]
    assert any("duplicate record" in p for p in validate_collection(records))


def test_assert_valid_raises_with_all_problems():
    with pytest.raises(ValidationError) as exc:
        assert_valid([_record(pointing_game=2.0)])
    assert "above maximum" in str(exc.value)


def test_assert_valid_passes_on_good_data():
    assert_valid([_record(pointing_game=0.8)])


def test_record_serialises_with_family_labels():
    payload = _record(pointing_game=0.8).to_dict()
    assert payload["model_family"] == "cnn"
    assert payload["method_family"] == "cam"
    assert payload["model_label"] == "ResNet-50"
    assert payload["metrics"]["pointing_game"]["mean"] == 0.8


def test_provenance_reports_missing_fields():
    prov = Provenance(source_run="run_a", seed=42)
    missing = prov.missing_fields()
    assert "git_commit" in missing
    assert "source_run" not in missing
    assert "seed" not in missing
