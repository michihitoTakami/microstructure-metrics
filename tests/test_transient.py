from __future__ import annotations

import numpy as np
from scipy import signal

from microstructure_metrics.metrics import calculate_transient_metrics


def test_transient_metrics_detect_smearing() -> None:
    sample_rate = 48_000
    samples = 4096
    ref = np.zeros(samples, dtype=np.float64)
    ref[samples // 2] = 1.0

    sos = signal.butter(4, 8000 / (sample_rate / 2), btype="low", output="sos")
    dut = signal.sosfiltfilt(sos, ref)

    result = calculate_transient_metrics(
        reference=ref, dut=dut, sample_rate=sample_rate
    )

    assert result.attack_time_delta_ms > 0.0
    assert result.edge_sharpness_ratio < 1.0
    assert result.transient_smearing_index >= 1.0


def test_transient_metrics_identical_signals_neutral() -> None:
    sample_rate = 48_000
    samples = 4096
    ref = np.zeros(samples, dtype=np.float64)
    ref[samples // 2] = 1.0

    result = calculate_transient_metrics(
        reference=ref, dut=ref, sample_rate=sample_rate
    )

    assert abs(result.attack_time_delta_ms) < 1e-6
    assert np.isclose(result.edge_sharpness_ratio, 1.0, atol=1e-3)
    assert np.isclose(result.transient_smearing_index, 1.0, atol=1e-3)
