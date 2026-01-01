from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from microstructure_metrics.alignment import align_audio_pair
from microstructure_metrics.testing import (
    DEFAULT_REGRESSION_CASES,
    evaluate_metrics,
    generate_pair_from_case,
)

EXPECTED_PATH = Path(__file__).parent / "expected_values.yaml"
EXPECTED_VALUES: Mapping[str, Mapping[str, object]] = (
    yaml.safe_load(EXPECTED_PATH.read_text()) or {}
)


@pytest.mark.parametrize("case", DEFAULT_REGRESSION_CASES, ids=lambda c: c.key)
def test_regression_against_expected(case) -> None:
    assert case.key in EXPECTED_VALUES, (
        f"expected_values.yaml に {case.key} がありません"
    )
    expected = EXPECTED_VALUES[case.key]

    pair = generate_pair_from_case(case)
    alignment = align_audio_pair(
        reference=pair.reference,
        dut=pair.dut,
        sample_rate=pair.sample_rate,
        pilot_freq=pair.common.pilot_freq,
        threshold=0.35,
        band_width_hz=200.0,
        min_duration_ms=pair.common.pilot_duration_ms * 0.9,
        pilot_duration_ms=pair.common.pilot_duration_ms,
        margin_ms=pair.common.fade_ms,
        max_lag_ms=30.0,
    )

    metrics = evaluate_metrics(
        reference=alignment.aligned_ref,
        dut=alignment.aligned_dut,
        sample_rate=pair.sample_rate,
        metadata=pair.reference_metadata,
        metrics=case.metrics,
    )

    for metric_key, spec in expected.get("metrics", {}).items():
        expected_value = float(spec["expected"])
        tolerance = float(spec["tolerance"])
        actual = float(metrics[metric_key])
        assert abs(actual - expected_value) <= tolerance, (
            f"{case.key}.{metric_key}: expected {expected_value}±{tolerance}, "
            f"got {actual}"
        )
