from __future__ import annotations

import numpy as np

from microstructure_metrics.metrics import (
    calculate_tfs_correlation,
    extract_tfs,
)


def _sine(
    *,
    sample_rate: int,
    freq: float,
    duration: float,
    phase: float = 0.0,
) -> np.ndarray:
    t = np.arange(int(sample_rate * duration)) / sample_rate
    return np.sin(2 * np.pi * freq * t + phase)


def test_extract_tfs_returns_expected_components() -> None:
    sr = 48_000
    freq = 4000.0
    sig = 0.8 * _sine(sample_rate=sr, freq=freq, duration=0.1)

    components = extract_tfs(
        signal=sig, sample_rate=sr, center_freq=freq, bandwidth=800.0
    )

    assert components.fine_structure.shape == sig.shape
    assert components.envelope.shape == sig.shape
    assert np.all(components.envelope > 0)
    assert np.allclose(
        components.fine_structure,
        np.cos(components.instantaneous_phase),
        atol=1e-6,
    )


def test_tfs_correlation_high_for_identical_signals() -> None:
    sr = 48_000
    duration = 0.25
    t = np.arange(int(sr * duration)) / sr
    tones = [2500.0, 3500.0, 5000.0, 7000.0]
    reference = np.sum([np.sin(2 * np.pi * f * t) for f in tones], axis=0) / len(tones)

    result = calculate_tfs_correlation(
        reference=reference,
        dut=reference,
        sample_rate=sr,
    )

    assert result.mean_correlation > 0.9
    assert result.percentile_05_correlation > 0.9
    assert result.correlation_variance < 0.01
    assert result.phase_coherence > 0.95
    assert result.group_delay_std_ms < 0.05
    assert all(value > 0.9 for value in result.band_correlations.values())
    assert result.used_frames >= result.frames_per_band * len(result.band_correlations)


def test_tfs_correlation_degrades_with_phase_modulation() -> None:
    sr = 48_000
    duration = 0.2
    carrier = 5000.0
    t = np.arange(int(sr * duration)) / sr
    reference = np.sin(2 * np.pi * carrier * t)
    phase_mod = 0.8 * np.sin(2 * np.pi * 25.0 * t)
    dut = np.sin(2 * np.pi * carrier * t + phase_mod)

    result = calculate_tfs_correlation(
        reference=reference,
        dut=dut,
        sample_rate=sr,
        freq_bands=[(4000.0, 6000.0)],
    )

    assert result.percentile_05_correlation < 0.95
    assert result.phase_coherence < 0.9
    assert result.correlation_variance > 1e-4


def test_tfs_skips_low_envelope_frames() -> None:
    sr = 48_000
    tone = _sine(sample_rate=sr, freq=5000.0, duration=0.05)
    silence = np.zeros(int(sr * 0.05))
    reference = np.concatenate([tone, silence, tone])
    dut = reference.copy()

    result = calculate_tfs_correlation(
        reference=reference, dut=dut, sample_rate=sr, freq_bands=[(4000.0, 6000.0)]
    )

    total_slots = result.frames_per_band * len(result.band_correlations)
    assert result.used_frames < total_slots
    assert result.mean_correlation > 0.85


def test_tfs_phase_coherence_is_robust_to_constant_delay() -> None:
    sr = 48_000
    duration = 0.25
    t = np.arange(int(sr * duration)) / sr
    tones = [2500.0, 3500.0, 5000.0, 7000.0]
    base = np.sum([np.sin(2 * np.pi * f * t) for f in tones], axis=0) / len(tones)

    # Use a small delay so the band-wise lag estimate is not ambiguous
    # (periodicity can produce multiple correlation peaks for larger delays).
    delay = 7  # samples (~0.15 ms)
    # Avoid inserting zeros (which can introduce filtfilt boundary artifacts).
    # Instead, take two offset windows from the same underlying waveform.
    dut = base[:-delay]
    reference = base[delay:]

    result = calculate_tfs_correlation(
        reference=reference,
        dut=dut,
        sample_rate=sr,
    )

    assert result.mean_correlation > 0.9
    assert result.phase_coherence > 0.95
    assert result.used_frames > 0
