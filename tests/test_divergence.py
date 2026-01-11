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


def test_mdi_increases_for_phase_warp_and_transient_smearing() -> None:
    sample_rate = 48_000
    duration = 0.35
    t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    phase_mod = 0.8 * np.sin(2 * np.pi * 25.0 * t)
    tone_ref = _tone_signal(sample_rate, duration, None)
    tone_warped = _tone_signal(sample_rate, duration, phase_mod)

    click_ref = _click_train(sample_rate, duration, [40.0, 120.0, 200.0, 280.0])
    sos = sp_signal.butter(4, 6500 / (sample_rate / 2), btype="low", output="sos")
    click_smeared = sp_signal.sosfiltfilt(sos, click_ref)

    tfs_clean = calculate_tfs_correlation(
        reference=tone_ref,
        dut=tone_ref,
        sample_rate=sample_rate,
        freq_bands=[(3000.0, 4000.0), (4000.0, 6000.0), (6000.0, 8000.0)],
    )
    tfs_degraded = calculate_tfs_correlation(
        reference=tone_ref,
        dut=tone_warped,
        sample_rate=sample_rate,
        freq_bands=[(3000.0, 4000.0), (4000.0, 6000.0), (6000.0, 8000.0)],
    )
    transient_clean = calculate_transient_metrics(
        reference=click_ref, dut=click_ref, sample_rate=sample_rate
    )
    transient_degraded = calculate_transient_metrics(
        reference=click_ref, dut=click_smeared, sample_rate=sample_rate
    )

    mdi_clean = calculate_microstructure_distribution_divergence(
        tfs_by_channel={"ch0": tfs_clean},
        transient_by_channel={"ch0": transient_clean},
    )
    mdi_degraded = calculate_microstructure_distribution_divergence(
        tfs_by_channel={"ch0": tfs_degraded},
        transient_by_channel={"ch0": transient_degraded},
    )

    assert mdi_degraded.mdi_total > mdi_clean.mdi_total + 0.5
