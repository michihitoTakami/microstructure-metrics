from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from microstructure_metrics.metrics import (
    calculate_microstructure_distribution_divergence,
    calculate_tfs_correlation,
    calculate_transient_metrics,
)
from microstructure_metrics.metrics.divergence import wasserstein_1d


def test_mdi_wasserstein_detects_local_breakdown_even_when_mean_matches() -> None:
    # "局所だけ崩す": 時間的に局所区間だけ値が大きく振れるが、平均は一致するケース
    n = 1000
    ref = np.zeros(n, dtype=np.float64)
    dut = ref.copy()

    # Local breakdown: symmetric spikes (+A then -A) so mean stays ~0.
    dut[100:150] = 1.0
    dut[150:200] = -1.0

    mean_diff = float(abs(np.mean(ref) - np.mean(dut)))
    mdi = wasserstein_1d(ref, dut)

    assert mean_diff == 0.0
    assert mdi > 0.05


def _click_train(
    sample_rate: int, duration: float, times_ms: list[float]
) -> np.ndarray:
    samples = int(sample_rate * duration)
    data = np.zeros(samples, dtype=np.float64)
    for t_ms in times_ms:
        idx = int(round(sample_rate * t_ms / 1000))
        if idx < samples:
            data[idx] = 1.0
    return data


def _tone_signal(
    sample_rate: int, duration: float, phase_mod: np.ndarray | None
) -> np.ndarray:
    t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    tones = [3000.0, 5000.0, 7000.0]
    phase = phase_mod if phase_mod is not None else np.zeros_like(t)
    signal = sum(np.sin(2 * np.pi * freq * t + phase) for freq in tones) / len(tones)
    return np.asarray(signal, dtype=np.float64)


def _local_window_mask(
    n: int, start_idx: int, end_idx: int, fade_samples: int
) -> np.ndarray:
    mask = np.zeros(n, dtype=np.float64)
    start = max(0, min(n, start_idx))
    end = max(0, min(n, end_idx))
    if end <= start:
        return mask
    mask[start:end] = 1.0
    if fade_samples > 0:
        fade = np.hanning(fade_samples * 2)
        left = fade[:fade_samples]
        right = fade[fade_samples:]
        fade_start = max(start - fade_samples, 0)
        fade_end = min(end + fade_samples, n)
        if fade_start < start:
            span = start - fade_start
            mask[fade_start:start] = left[-span:]
        if end < fade_end:
            span = fade_end - end
            mask[end:fade_end] = right[:span]
    return mask


def test_mdi_increases_for_phase_warp_and_transient_smearing() -> None:
    sample_rate = 48_000
    duration = 0.35
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float64) / sample_rate
    tone_ref = _tone_signal(sample_rate, duration, None)
    phase_mod = 1.6 * np.sin(2 * np.pi * 35.0 * t)
    phase_mask = _local_window_mask(
        n, int(0.08 * sample_rate), int(0.24 * sample_rate), int(0.02 * sample_rate)
    )
    tone_warped = _tone_signal(sample_rate, duration, phase_mod * phase_mask)

    click_times = [40.0, 120.0, 200.0, 280.0]
    click_ref = _click_train(sample_rate, duration, click_times)
    sos = sp_signal.butter(4, 3500 / (sample_rate / 2), btype="low", output="sos")
    click_smeared = sp_signal.sosfiltfilt(sos, click_ref)
    smear_mask = np.zeros(n, dtype=np.float64)
    smear_window = int(0.012 * sample_rate)
    for t_ms in [120.0, 200.0]:
        center = int(round(sample_rate * t_ms / 1000))
        start = max(center - smear_window, 0)
        end = min(center + smear_window, n)
        smear_mask[start:end] = 1.0
    click_dut = click_ref + (click_smeared - click_ref) * smear_mask

    reference = 0.4 * tone_ref + 0.6 * click_ref
    reference = reference / max(float(np.max(np.abs(reference))), 1e-6)
    dut = 0.4 * tone_warped + 0.6 * click_dut
    dut = dut / max(float(np.max(np.abs(dut))), 1e-6)

    tfs_clean = calculate_tfs_correlation(
        reference=reference,
        dut=reference,
        sample_rate=sample_rate,
        freq_bands=[(3000.0, 4000.0), (4000.0, 6000.0), (6000.0, 8000.0)],
    )
    tfs_degraded = calculate_tfs_correlation(
        reference=reference,
        dut=dut,
        sample_rate=sample_rate,
        freq_bands=[(3000.0, 4000.0), (4000.0, 6000.0), (6000.0, 8000.0)],
    )
    transient_clean = calculate_transient_metrics(
        reference=reference, dut=reference, sample_rate=sample_rate
    )
    transient_degraded = calculate_transient_metrics(
        reference=reference, dut=dut, sample_rate=sample_rate
    )

    mdi_clean = calculate_microstructure_distribution_divergence(
        tfs_by_channel={"ch0": tfs_clean},
        transient_by_channel={"ch0": transient_clean},
    )
    mdi_degraded = calculate_microstructure_distribution_divergence(
        tfs_by_channel={"ch0": tfs_degraded},
        transient_by_channel={"ch0": transient_degraded},
    )

    assert mdi_degraded.mdi_total > mdi_clean.mdi_total + 0.2
