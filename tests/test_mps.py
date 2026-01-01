from __future__ import annotations

import numpy as np

from microstructure_metrics.metrics import (
    calculate_mps,
    calculate_mps_similarity,
)


def _am_tone(
    *,
    sample_rate: int,
    duration: float,
    carrier_hz: float,
    mod_hz: float,
    depth: float = 0.5,
) -> np.ndarray:
    t = np.arange(int(sample_rate * duration)) / sample_rate
    envelope = 1.0 + depth * np.sin(2 * np.pi * mod_hz * t)
    carrier = np.sin(2 * np.pi * carrier_hz * t)
    signal = envelope * carrier
    peak = np.max(np.abs(signal)) or 1.0
    return np.asarray(signal / peak, dtype=np.float64)


def test_mps_detects_am_modulation_peak() -> None:
    sr = 48_000
    carrier = 1000.0
    mod = 4.0
    sig = _am_tone(
        sample_rate=sr, duration=0.8, carrier_hz=carrier, mod_hz=mod, depth=0.6
    )

    result = calculate_mps(
        signal=sig,
        sample_rate=sr,
        audio_freq_range=(300.0, 4000.0),
        mod_freq_range=(0.5, 32.0),
        num_audio_bands=48,
    )

    band_idx = int(np.argmin(np.abs(result.audio_freqs - carrier)))
    band_mps = result.mps_matrix[band_idx]
    peak_mod_freq = float(result.mod_freqs[int(np.argmax(band_mps))])

    assert result.mps_matrix.shape[0] == 48
    assert abs(peak_mod_freq - mod) <= 1.0


def test_mps_similarity_high_for_identical_signals() -> None:
    sr = 48_000
    sig = _am_tone(
        sample_rate=sr, duration=0.6, carrier_hz=1200.0, mod_hz=5.0, depth=0.4
    )

    result = calculate_mps_similarity(
        reference=sig,
        dut=sig,
        sample_rate=sr,
        audio_freq_range=(300.0, 4000.0),
        mod_freq_range=(0.5, 32.0),
    )

    assert result.mps_correlation > 0.99
    assert result.mps_distance < 1e-12
    assert result.band_correlations
    peak_band = max(result.band_correlations.values())
    assert peak_band > 0.99


def test_mps_similarity_degrades_when_modulation_removed() -> None:
    sr = 48_000
    carrier = 1500.0
    mod = 6.0
    reference = _am_tone(
        sample_rate=sr, duration=0.7, carrier_hz=carrier, mod_hz=mod, depth=0.6
    )
    dut = _am_tone(
        sample_rate=sr, duration=0.7, carrier_hz=carrier, mod_hz=0.0, depth=0.0
    )

    result = calculate_mps_similarity(
        reference=reference,
        dut=dut,
        sample_rate=sr,
        audio_freq_range=(300.0, 4000.0),
        mod_freq_range=(0.5, 32.0),
    )

    target_band = min(
        result.band_correlations.items(),
        key=lambda item: abs(item[0] - carrier),
    )[1]

    assert result.mps_correlation < 0.3
    assert result.mps_distance > 0.02
    assert target_band < 0.3
