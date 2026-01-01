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


def _am_fm_tone(
    *,
    sample_rate: int,
    duration: float,
    carrier_hz: float,
    am_hz: float,
    am_depth: float,
    fm_hz: float,
    fm_dev: float,
) -> np.ndarray:
    t = np.arange(int(sample_rate * duration)) / sample_rate
    am = 1.0 + am_depth * np.sin(2 * np.pi * am_hz * t)
    beta = fm_dev / max(fm_hz, 1e-6)
    phase = 2 * np.pi * carrier_hz * t + beta * np.sin(2 * np.pi * fm_hz * t)
    signal = am * np.sin(phase)
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


def test_mps_supports_log_modulation_axis() -> None:
    sr = 48_000
    mod = 6.0
    sig = _am_tone(
        sample_rate=sr, duration=0.9, carrier_hz=1200.0, mod_hz=mod, depth=0.5
    )
    result = calculate_mps(
        signal=sig,
        sample_rate=sr,
        audio_freq_range=(200.0, 5000.0),
        mod_freq_range=(0.5, 64.0),
        num_audio_bands=32,
        mod_scale="log",
        num_mod_bins=24,
    )

    band_idx = int(np.argmin(np.abs(result.audio_freqs - 1200.0)))
    peak_mod = float(result.mod_freqs[int(np.argmax(result.mps_matrix[band_idx]))])
    log_steps = np.diff(np.log(result.mod_freqs))

    assert result.mps_matrix.shape[1] == 24
    assert abs(peak_mod - mod) <= 1.0
    assert np.allclose(log_steps, np.mean(log_steps), atol=1e-2)


def test_mps_mel_filterbank_runs() -> None:
    sr = 48_000
    sig = _am_tone(
        sample_rate=sr, duration=0.6, carrier_hz=1500.0, mod_hz=5.0, depth=0.4
    )
    result = calculate_mps(
        signal=sig,
        sample_rate=sr,
        audio_freq_range=(300.0, 6000.0),
        mod_freq_range=(0.5, 32.0),
        num_audio_bands=32,
        filterbank="mel",
        filterbank_kwargs={"order": 6},
    )

    assert result.mps_matrix.shape[0] == 32
    assert np.all(np.isfinite(result.mps_matrix))
    diffs = np.diff(result.audio_freqs)
    assert np.all(diffs <= 0)


def test_mps_rectify_envelope_detects_modulation() -> None:
    sr = 48_000
    carrier = 900.0
    mod = 3.5
    sig = _am_tone(
        sample_rate=sr, duration=0.8, carrier_hz=carrier, mod_hz=mod, depth=0.7
    )
    result = calculate_mps(
        signal=sig,
        sample_rate=sr,
        audio_freq_range=(200.0, 4000.0),
        mod_freq_range=(0.5, 32.0),
        num_audio_bands=32,
        envelope_method="rectify",
        envelope_lowpass_hz=32.0,
        envelope_lowpass_order=4,
    )

    band_idx = int(np.argmin(np.abs(result.audio_freqs - carrier)))
    peak_mod = float(result.mod_freqs[int(np.argmax(result.mps_matrix[band_idx]))])
    assert abs(peak_mod - mod) <= 1.0


def test_mps_similarity_band_weighting() -> None:
    sr = 48_000
    ref_low = _am_tone(
        sample_rate=sr, duration=0.8, carrier_hz=500.0, mod_hz=5.0, depth=0.6
    )
    ref_high = _am_tone(
        sample_rate=sr, duration=0.8, carrier_hz=3000.0, mod_hz=5.0, depth=0.6
    )
    reference = ref_low + ref_high
    dut = (
        _am_tone(sample_rate=sr, duration=0.8, carrier_hz=500.0, mod_hz=0.0, depth=0.0)
        + ref_high
    )

    base = calculate_mps(
        signal=reference,
        sample_rate=sr,
        audio_freq_range=(200.0, 5000.0),
        mod_freq_range=(0.5, 32.0),
        num_audio_bands=32,
    )
    weights = 1.0 / (1.0 + base.audio_freqs / 800.0)

    unweighted = calculate_mps_similarity(
        reference=reference,
        dut=dut,
        sample_rate=sr,
        audio_freq_range=(200.0, 5000.0),
        mod_freq_range=(0.5, 32.0),
        num_audio_bands=32,
    )
    weighted = calculate_mps_similarity(
        reference=reference,
        dut=dut,
        sample_rate=sr,
        audio_freq_range=(200.0, 5000.0),
        mod_freq_range=(0.5, 32.0),
        num_audio_bands=32,
        band_weights=weights,
        mps_norm="global",
    )

    assert weighted.mps_correlation < unweighted.mps_correlation
    assert weighted.mps_distance > unweighted.mps_distance


def test_mps_handles_am_fm_composite_on_log_grid() -> None:
    sr = 48_000
    carrier = 1100.0
    am = 6.0
    fm = 8.0
    sig = _am_fm_tone(
        sample_rate=sr,
        duration=0.9,
        carrier_hz=carrier,
        am_hz=am,
        am_depth=0.45,
        fm_hz=fm,
        fm_dev=70.0,
    )

    result = calculate_mps(
        signal=sig,
        sample_rate=sr,
        audio_freq_range=(200.0, 5000.0),
        mod_freq_range=(0.5, 64.0),
        num_audio_bands=48,
        mod_scale="log",
    )
    band_idx = int(np.argmin(np.abs(result.audio_freqs - carrier)))
    peak_mod = float(result.mod_freqs[int(np.argmax(result.mps_matrix[band_idx]))])

    assert abs(peak_mod - am) <= 1.0
