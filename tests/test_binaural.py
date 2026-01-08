from __future__ import annotations

import numpy as np
import pytest

from microstructure_metrics.metrics import calculate_binaural_cue_preservation


def test_binaural_cue_preservation_recovers_itd_and_ild() -> None:
    sample_rate = 48_000
    duration = 0.3
    itd_samples = 10  # ≈0.208 ms
    ild_db = 1.5

    t = np.arange(int(duration * sample_rate)) / sample_rate
    base = 0.7 * np.sin(2 * np.pi * 500 * t) + 0.3 * np.sin(2 * np.pi * 1500 * t)
    right_shifted = np.concatenate([np.zeros(itd_samples), base])[: base.size]
    scale = 10 ** (-ild_db / 20)

    reference = np.stack([base, base], axis=1)
    dut = np.stack([base, right_shifted * scale], axis=1)

    result = calculate_binaural_cue_preservation(
        reference_lr=reference,
        dut_lr=dut,
        sample_rate=sample_rate,
        num_audio_bands=8,
        frame_length_ms=30.0,
        frame_hop_ms=15.0,
        max_itd_ms=1.0,
        envelope_threshold_db=-60.0,
        itd_outlier_threshold_ms=0.25,
    )

    expected_itd_ms = itd_samples / sample_rate * 1000.0
    assert result.used_frames > 0
    assert result.median_abs_delta_itd_ms == pytest.approx(
        expected_itd_ms, rel=0.2, abs=0.08
    )
    assert result.median_abs_delta_ild_db == pytest.approx(ild_db, rel=0.2, abs=0.4)
    assert result.itd_outlier_rate < 0.2
    assert result.iacc_p05 > 0.6
    assert result.band_stats
