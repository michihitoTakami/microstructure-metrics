from __future__ import annotations

import numpy as np
import pytest

from microstructure_metrics.alignment import (
    check_drift_threshold,
    estimate_clock_drift,
)


def _pilot_wave(
    *, sample_rate: int, freq: float, duration_ms: int, fade_ms: int = 5
) -> np.ndarray:
    samples = max(int(sample_rate * duration_ms / 1000), 1)
    t = np.arange(samples) / sample_rate
    tone = np.sin(2 * np.pi * freq * t)
    fade_samples = min(max(int(sample_rate * fade_ms / 1000), 1), samples // 2)
    if fade_samples > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_samples))
        tone[:fade_samples] *= ramp
        tone[-fade_samples:] *= ramp[::-1]
    return tone.astype(np.float64)


def _synthetic_signal(
    *,
    sample_rate: int = 48_000,
    pilot_freq: float = 1000.0,
    pilot_ms: int = 100,
    silence_ms: int = 500,
    body_duration: float = 3.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    rng = rng or np.random.default_rng(0)
    pilot = _pilot_wave(sample_rate=sample_rate, freq=pilot_freq, duration_ms=pilot_ms)
    silence = np.zeros(int(sample_rate * silence_ms / 1000), dtype=np.float64)
    body = rng.standard_normal(int(sample_rate * body_duration)) * 0.05

    start_pilot_idx = silence.shape[0]
    end_pilot_idx = start_pilot_idx + pilot.shape[0] + body.shape[0]

    timeline = np.concatenate([silence, pilot, body, pilot, silence])
    return timeline, {
        "start_pilot": start_pilot_idx,
        "end_pilot": end_pilot_idx,
    }


def _insert_samples(
    data: np.ndarray, *, insert_at: int, samples: int, fill: float = 0.0
) -> np.ndarray:
    padding = np.full(samples, fill, dtype=np.float64)
    return np.concatenate([data[:insert_at], padding, data[insert_at:]])


def test_estimate_clock_drift_no_drift() -> None:
    reference, positions = _synthetic_signal(body_duration=2.5)
    result = estimate_clock_drift(reference=reference, dut=reference, sample_rate=48000)

    assert abs(result.drift_ppm) < 0.1
    assert abs(result.delay_start_samples) <= 1
    assert abs(result.delay_end_samples) <= 1
    assert abs(result.pilot_detection.start_sample - positions["start_pilot"]) <= 200
    assert abs(result.pilot_detection.end_sample - positions["end_pilot"]) <= 200


def test_estimate_clock_drift_positive_ppm() -> None:
    body_duration = 5.0
    reference, positions = _synthetic_signal(body_duration=body_duration)
    extra_samples = 24  # 約80 ppm @ duration≈6.2s
    dut = _insert_samples(
        reference, insert_at=positions["end_pilot"], samples=extra_samples
    )

    duration_sec = reference.shape[0] / 48_000
    expected_ppm = extra_samples / duration_sec / 48_000 * 1_000_000

    result = estimate_clock_drift(reference=reference, dut=dut, sample_rate=48_000)
    warning = check_drift_threshold(result.drift_ppm)

    assert pytest.approx(result.delay_end_samples - result.delay_start_samples) == 24
    assert pytest.approx(result.drift_ppm, rel=0.05) == expected_ppm
    assert warning.severity == "high"
    assert "Significant clock drift" in warning.message


def test_check_drift_threshold_levels() -> None:
    assert check_drift_threshold(0).severity == "none"
    assert check_drift_threshold(10).severity == "low"
    assert check_drift_threshold(-15).severity == "low"
    assert check_drift_threshold(50).severity == "high"
    assert check_drift_threshold(-120).severity == "critical"


def test_estimate_clock_drift_accepts_stereo_input() -> None:
    reference, _ = _synthetic_signal(body_duration=2.0)
    stereo = np.stack([reference, reference * 0.8], axis=1)
    result = estimate_clock_drift(reference=stereo, dut=stereo, sample_rate=48_000)
    assert abs(result.drift_ppm) < 0.1
