from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from microstructure_metrics.metrics import calculate_residual_microstructure


def _fractional_shift(x: np.ndarray, shift_samples: float) -> np.ndarray:
    idx = np.arange(x.shape[0], dtype=np.float64)
    return np.interp(idx - shift_samples, idx, x, left=0.0, right=0.0)


def test_residual_microstructure_separates_noise_and_ringing() -> None:
    sample_rate = 48_000
    n = sample_rate
    rng = np.random.default_rng(123)

    reference = rng.standard_normal(n).astype(np.float64) * 0.2
    # Use a delay large enough to avoid ambiguity at sub-sample resolution.
    delay_samples = 12.35
    scale = 1.2

    base = scale * _fractional_shift(reference, delay_samples)

    noise = rng.standard_normal(n).astype(np.float64)
    noise_rms = 0.02
    residual_noise = noise * noise_rms

    h_len = int(0.05 * sample_rate)  # 50ms ringing
    t = np.arange(h_len, dtype=np.float64) / float(sample_rate)
    ring_freq_hz = 3000.0
    tau = 0.008
    impulse_response = np.exp(-t / tau) * np.sin(2.0 * np.pi * ring_freq_hz * t)

    impulses = np.zeros(n, dtype=np.float64)
    burst_times_sec = [0.10, 0.30, 0.50, 0.70, 0.90]
    for sec in burst_times_sec:
        impulses[int(sec * sample_rate)] = 1.0
    residual_ring = sp_signal.fftconvolve(impulses, impulse_response, mode="full")[:n]
    residual_ring = residual_ring / max(
        float(np.sqrt(np.mean(residual_ring**2))), 1e-12
    )
    residual_ring = residual_ring * noise_rms

    dut_noise = base + residual_noise
    dut_ring = base + residual_ring

    res_noise = calculate_residual_microstructure(
        reference=reference, dut=dut_noise, sample_rate=sample_rate
    )
    res_ring = calculate_residual_microstructure(
        reference=reference, dut=dut_ring, sample_rate=sample_rate
    )

    assert abs(res_noise.delay_samples - delay_samples) < 0.5
    assert abs(res_ring.delay_samples - delay_samples) < 0.5
    assert abs(res_noise.scale - scale) < 0.1
    assert abs(res_ring.scale - scale) < 0.1

    # Whiteness: noise should be flatter and less autocorrelated than ringing.
    assert res_noise.spectral_flatness > res_ring.spectral_flatness
    assert res_noise.autocorr_peak_excess < res_ring.autocorr_peak_excess

    # Burstiness: ringing bursts should be more impulsive than wideband noise.
    assert res_ring.crest_factor > res_noise.crest_factor
    assert res_ring.kurtosis > res_noise.kurtosis
