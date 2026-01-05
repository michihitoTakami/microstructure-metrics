from __future__ import annotations

import numpy as np
from scipy import signal

from microstructure_metrics.metrics import calculate_transient_metrics


def _click_train(sample_rate: int, times_ms: list[float]) -> np.ndarray:
    samples = int(sample_rate * 0.25)
    data = np.zeros(samples, dtype=np.float64)
    for t_ms in times_ms:
        idx = int(round(sample_rate * t_ms / 1000))
        if idx < samples:
            data[idx] = 1.0
    return data


def test_transient_metrics_detect_smearing_multi_events() -> None:
    sample_rate = 48_000
    ref = _click_train(sample_rate, [40.0, 90.0, 140.0])

    sos = signal.butter(4, 7500 / (sample_rate / 2), btype="low", output="sos")
    dut = signal.sosfiltfilt(sos, ref)

    result = calculate_transient_metrics(
        reference=ref, dut=dut, sample_rate=sample_rate
    )

    assert result.matched_event_pairs == 3
    assert result.transient_smearing_index > 1.05
    assert result.width_ratio_stats.percentile_95 > 1.05
    assert result.edge_sharpness_ratio < 1.0
    assert result.attack_time_delta_ms > 0.0


def test_transient_metrics_identical_signals_multi_events_neutral() -> None:
    sample_rate = 48_000
    ref = _click_train(sample_rate, [20.0, 70.0, 120.0, 170.0])

    result = calculate_transient_metrics(
        reference=ref, dut=ref, sample_rate=sample_rate
    )

    assert result.matched_event_pairs == 4
    assert result.unmatched_ref_events == 0
    assert result.unmatched_dut_events == 0
    assert abs(result.attack_time_delta_ms) < 1e-6
    assert np.isclose(result.edge_sharpness_ratio, 1.0, atol=1e-3)
    assert np.isclose(result.transient_smearing_index, 1.0, atol=1e-3)


def test_transient_metrics_supports_zero_smoothing_and_new_features() -> None:
    sample_rate = 48_000
    ref = _click_train(sample_rate, [80.0])

    sos = signal.butter(4, 7500 / (sample_rate / 2), btype="low", output="sos")
    dut = signal.sosfiltfilt(sos, ref)

    result = calculate_transient_metrics(
        reference=ref,
        dut=dut,
        sample_rate=sample_rate,
        smoothing_ms=0.0,
        asymmetry_window_ms=3.0,
    )

    assert result.matched_event_pairs >= 1
    assert np.isfinite(result.low_level_attack_time_dut_ms)
    assert np.isfinite(result.pre_energy_fraction_dut)
    assert np.isfinite(result.energy_skewness_dut)


def test_transient_metrics_pre_energy_distinguishes_zero_phase_vs_causal() -> None:
    sample_rate = 48_000
    ref = _click_train(sample_rate, [80.0])
    sos = signal.butter(4, 8000 / (sample_rate / 2), btype="low", output="sos")

    # Zero-phase (sosfiltfilt) introduces symmetric pre-ringing around the peak,
    # while causal filtering (sosfilt) keeps energy largely after the peak.
    dut_zero_phase = signal.sosfiltfilt(sos, ref)
    dut_causal = signal.sosfilt(sos, ref)

    res_zero = calculate_transient_metrics(
        reference=ref,
        dut=dut_zero_phase,
        sample_rate=sample_rate,
        smoothing_ms=0.0,
        asymmetry_window_ms=3.0,
    )
    res_causal = calculate_transient_metrics(
        reference=ref,
        dut=dut_causal,
        sample_rate=sample_rate,
        smoothing_ms=0.0,
        asymmetry_window_ms=3.0,
    )

    # Zero-phase should be close to symmetric around the peak (skewness ~= 0 and
    # pre/post energy ratio ~= 1), while causal filtering should be asymmetric.
    zero_ratio = res_zero.dut_events[0].pre_post_energy_ratio
    causal_ratio = res_causal.dut_events[0].pre_post_energy_ratio

    assert abs(res_zero.energy_skewness_dut) < 0.2
    assert res_causal.energy_skewness_dut > 0.5
    assert abs(np.log(zero_ratio)) < 0.05
    assert abs(np.log(causal_ratio)) > 0.1
