from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from microstructure_metrics.metrics import (
    calculate_delta_se,
    calculate_spectral_entropy,
    plot_delta_se,
    plot_spectral_entropy,
)


def test_spectral_entropy_noise_greater_than_tone() -> None:
    sr = 48_000
    duration = 0.5
    t = np.arange(int(sr * duration)) / sr
    tone = 0.4 * np.sin(2 * np.pi * 1000 * t)
    noise = np.random.default_rng(0).standard_normal(tone.shape[0]) * 0.2

    tone_se = calculate_spectral_entropy(
        signal=tone, sample_rate=sr, frame_size=1024, hop_size=256
    )
    noise_se = calculate_spectral_entropy(
        signal=noise, sample_rate=sr, frame_size=1024, hop_size=256
    )

    assert 0.0 <= tone_se.mean_entropy <= 1.0
    assert 0.0 <= noise_se.mean_entropy <= 1.0
    assert noise_se.mean_entropy > tone_se.mean_entropy
    assert noise_se.entropy_over_time.shape[0] == noise_se.frame_times.shape[0]


def test_delta_se_increases_with_added_noise() -> None:
    sr = 48_000
    duration = 0.5
    t = np.arange(int(sr * duration)) / sr
    reference = 0.6 * np.sin(2 * np.pi * 1500 * t)
    dut = reference + 0.2 * np.random.default_rng(1).standard_normal(reference.size)

    result = calculate_delta_se(
        reference=reference,
        dut=dut,
        sample_rate=sr,
        frame_size=1024,
        hop_size=256,
    )

    assert result.delta_se_mean > 0.0
    assert result.delta_se_over_time.shape == result.frame_times.shape
    assert result.dut_se_mean > result.ref_se_mean


def test_delta_se_increases_with_spectral_smoothing() -> None:
    sr = 48_000
    duration = 0.5
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(int(sr * duration))

    sos_narrow = sp_signal.butter(
        4, [3800.0, 4200.0], btype="bandpass", fs=sr, output="sos"
    )
    sos_wide = sp_signal.butter(
        4, [2000.0, 6000.0], btype="bandpass", fs=sr, output="sos"
    )

    reference = sp_signal.sosfiltfilt(sos_narrow, noise)
    dut = sp_signal.sosfiltfilt(sos_wide, noise)

    result = calculate_delta_se(
        reference=reference,
        dut=dut,
        sample_rate=sr,
        frame_size=1024,
        hop_size=256,
    )

    assert result.delta_se_mean > 0.0
    assert result.dut_se_mean > result.ref_se_mean
    assert result.ref_entropy_over_time.shape == result.frame_times.shape
    assert result.dut_entropy_over_time.shape == result.frame_times.shape


def test_freq_range_limits_entropy_growth() -> None:
    sr = 48_000
    duration = 0.5
    t = np.arange(int(sr * duration)) / sr
    tone = 0.6 * np.sin(2 * np.pi * 1000 * t)
    noise = 0.2 * np.random.default_rng(2).standard_normal(t.size)
    mixture = tone + noise

    full = calculate_spectral_entropy(
        signal=mixture,
        sample_rate=sr,
        frame_size=1024,
        hop_size=256,
    )
    narrow = calculate_spectral_entropy(
        signal=mixture,
        sample_rate=sr,
        frame_size=1024,
        hop_size=256,
        freq_range=(900.0, 1100.0),
    )

    assert narrow.freqs.min() >= 900.0
    assert narrow.freqs.max() <= 1100.0
    assert narrow.freqs.shape[0] < full.freqs.shape[0]


def test_noise_floor_mask_reduces_entropy() -> None:
    sr = 48_000
    duration = 0.4
    t = np.arange(int(sr * duration)) / sr
    tone = 0.6 * np.sin(2 * np.pi * 2000 * t)
    low_noise = 0.05 * np.random.default_rng(3).standard_normal(t.size)
    mixture = tone + low_noise

    base = calculate_spectral_entropy(
        signal=mixture,
        sample_rate=sr,
        frame_size=512,
        hop_size=128,
    )
    masked = calculate_spectral_entropy(
        signal=mixture,
        sample_rate=sr,
        frame_size=512,
        hop_size=128,
        noise_floor_db=30.0,
    )

    assert masked.mean_entropy < base.mean_entropy


def test_plot_helpers_return_axes() -> None:
    import matplotlib

    matplotlib.use("Agg")
    sr = 48_000
    duration = 0.25
    t = np.arange(int(sr * duration)) / sr
    tone = 0.5 * np.sin(2 * np.pi * 1200 * t)
    noise = 0.1 * np.random.default_rng(4).standard_normal(t.size)
    reference = tone
    dut = tone + noise

    se = calculate_spectral_entropy(
        signal=reference, sample_rate=sr, frame_size=512, hop_size=128
    )
    se_ax = plot_spectral_entropy(se, title="SE")
    assert se_ax.get_xlabel() == "Time [s]"

    delta = calculate_delta_se(
        reference=reference,
        dut=dut,
        sample_rate=sr,
        frame_size=512,
        hop_size=128,
    )
    ts_ax, hist_ax = plot_delta_se(delta, bins=20, title="ΔSE")
    assert ts_ax.get_ylabel().startswith("Spectral entropy")
    assert hist_ax.get_xlabel() == "ΔSE"
