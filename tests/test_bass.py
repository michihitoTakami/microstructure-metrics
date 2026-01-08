from __future__ import annotations

import numpy as np

from microstructure_metrics.metrics import calculate_low_freq_complex_reconstruction


def _complex_bass(
    sample_rate: int, duration: float, *, phase_warp: float
) -> np.ndarray:
    t = np.arange(int(sample_rate * duration)) / sample_rate
    freqs = np.array([45.0, 70.0, 110.0, 170.0])
    rng = np.random.default_rng(1)
    phases = rng.uniform(0, 2 * np.pi, size=freqs.shape)
    signal = np.zeros_like(t)
    for idx, freq in enumerate(freqs):
        warp = phase_warp * np.sin(2 * np.pi * 0.7 * t + 0.4 * idx)
        signal += np.sin(2 * np.pi * freq * t + phases[idx] + warp + 0.3 * idx)
    peak = float(np.max(np.abs(signal), initial=1.0))
    return np.asarray(signal / max(peak, 1e-6), dtype=np.float64)


def test_lfcr_detects_phase_warp() -> None:
    sample_rate = 48_000
    duration = 0.8
    reference = _complex_bass(sample_rate, duration, phase_warp=0.0)
    rng = np.random.default_rng(2)
    dut_clean = reference + 0.0005 * rng.standard_normal(reference.shape[0])
    dut_warped = _complex_bass(sample_rate, duration, phase_warp=0.65)

    clean = calculate_low_freq_complex_reconstruction(
        reference=reference, dut=dut_clean, sample_rate=sample_rate
    )
    warped = calculate_low_freq_complex_reconstruction(
        reference=reference, dut=dut_warped, sample_rate=sample_rate
    )

    assert clean.used_cycles > 5
    assert warped.cycle_shape_corr_mean < clean.cycle_shape_corr_mean - 0.05
    assert warped.harmonic_phase_coherence < clean.harmonic_phase_coherence
    assert warped.envelope_diff_outlier_rate > clean.envelope_diff_outlier_rate
