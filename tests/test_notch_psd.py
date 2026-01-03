from __future__ import annotations

import numpy as np
from scipy import signal

from microstructure_metrics.metrics import calculate_narrowband_notch_depth


def _notched_noise(
    *,
    sample_rate: int,
    duration: float,
    center_hz: float,
    q: float,
    scale: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    samples = int(sample_rate * duration)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(samples)
    b, a = signal.iirnotch(w0=center_hz / (sample_rate / 2), Q=q)
    filtered = signal.lfilter(b, a, white)
    peak = np.max(np.abs(filtered)) or 1.0
    return np.asarray(scale * filtered / peak, dtype=np.float64)


def test_narrowband_notch_depth_is_consistent_for_identical_signals() -> None:
    sr = 48_000
    duration = 0.4
    center = 8000.0
    q = 40.0
    reference = _notched_noise(sample_rate=sr, duration=duration, center_hz=center, q=q)

    result = calculate_narrowband_notch_depth(
        reference=reference,
        dut=reference,
        sample_rate=sr,
        notch_center_hz=center,
        notch_q=q,
    )

    assert abs(result.notch_fill_db) < 0.5
    assert result.ref_notch_depth_db > 6.0


def test_narrowband_notch_depth_detects_high_q_fill() -> None:
    sr = 48_000
    duration = 0.4
    center = 9000.0
    q = 80.0
    reference = _notched_noise(
        sample_rate=sr, duration=duration, center_hz=center, q=q, seed=1
    )

    t = np.arange(reference.size) / sr
    fill_noise = 0.02 * np.sin(2 * np.pi * center * t)
    dut = reference + fill_noise

    result = calculate_narrowband_notch_depth(
        reference=reference,
        dut=dut,
        sample_rate=sr,
        notch_center_hz=center,
        notch_q=q,
    )

    assert result.notch_fill_db > 6.0
    assert result.dut_notch_depth_db < result.ref_notch_depth_db
