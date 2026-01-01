from __future__ import annotations

import numpy as np
from scipy import signal

from microstructure_metrics.metrics import calculate_nps


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


def test_nps_detects_notch_fill() -> None:
    sr = 48_000
    duration = 0.6
    center = 8000.0
    q = 8.6
    ref = _notched_noise(sample_rate=sr, duration=duration, center_hz=center, q=q)

    t = np.arange(ref.size) / sr
    fill = 0.02 * np.sin(2 * np.pi * center * t)
    dut = ref + fill

    result = calculate_nps(
        reference=ref,
        dut=dut,
        sample_rate=sr,
        notch_center_hz=center,
        notch_q=q,
    )

    assert result.nps_db < -2.0
    assert result.dut_notch_depth_db < result.ref_notch_depth_db
    assert not result.is_noise_limited


def test_nps_noise_limited_flag_when_dynamic_range_low() -> None:
    sr = 48_000
    duration = 0.3
    center = 6000.0
    q = 6.0
    ref = _notched_noise(
        sample_rate=sr, duration=duration, center_hz=center, q=q, scale=1e-3
    )
    dut = ref + 1e-4 * np.random.default_rng(1).standard_normal(ref.shape[0])

    result = calculate_nps(
        reference=ref,
        dut=dut,
        sample_rate=sr,
        notch_center_hz=center,
        notch_q=q,
        dynamic_range_threshold_db=25.0,
    )

    assert result.is_noise_limited
    assert result.noise_floor_db < -40.0
