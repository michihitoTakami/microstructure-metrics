from __future__ import annotations

import numpy as np
import pytest

from microstructure_metrics.metrics import calculate_thd_n


def _tone(
    freq: float, sample_rate: int, duration: float, *, level_dbfs: float
) -> np.ndarray:
    t = np.arange(int(duration * sample_rate)) / sample_rate
    amplitude = 10 ** (level_dbfs / 20)
    return amplitude * np.sin(2 * np.pi * freq * t)


def test_thd_n_detects_harmonic_and_noise_components() -> None:
    sr = 48_000
    duration = 1.0
    fundamental = _tone(1000.0, sr, duration, level_dbfs=-3.0)
    harmonic = _tone(
        2000.0,
        sr,
        duration,
        level_dbfs=-53.0,  # roughly -50 dBc vs -3 dBFS peak
    )
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1e-4, size=fundamental.shape[0])
    signal = fundamental + harmonic + noise

    result = calculate_thd_n(
        signal=signal,
        fundamental_freq=1000.0,
        sample_rate=sr,
        num_harmonics=5,
    )

    assert 995.0 <= result.fundamental_freq <= 1005.0
    assert -55.0 <= result.thd_db <= -45.0
    assert -55.0 <= result.thd_n_db <= -45.0
    assert 2 in result.harmonic_levels
    assert -60.0 <= result.harmonic_levels[2] <= -40.0
    assert result.noise_db < -60.0


def test_thd_n_respects_bandwidth_and_harmonic_count() -> None:
    sr = 48_000
    duration = 0.5
    fundamental = _tone(4000.0, sr, duration, level_dbfs=-6.0)
    second = _tone(8000.0, sr, duration, level_dbfs=-40.0)
    third = _tone(12_000.0, sr, duration, level_dbfs=-45.0)
    fourth = _tone(16_000.0, sr, duration, level_dbfs=-50.0)  # outside bandwidth
    signal = fundamental + second + third + fourth

    result = calculate_thd_n(
        signal=signal,
        fundamental_freq=4000.0,
        sample_rate=sr,
        bandwidth=(20.0, 12_000.0),
        num_harmonics=6,
        expected_level_dbfs=None,
    )

    assert set(result.harmonic_levels.keys()) == {2, 3}
    assert result.thd_db < -30.0  # harmonics present
    assert result.measurement_bandwidth == (20.0, 12_000.0)


def test_gain_warning_when_fundamental_far_from_expected() -> None:
    sr = 48_000
    duration = 0.25
    signal = _tone(1000.0, sr, duration, level_dbfs=-12.0)

    with pytest.warns(UserWarning):
        result = calculate_thd_n(
            signal=signal,
            fundamental_freq=1000.0,
            sample_rate=sr,
            expected_level_dbfs=-3.0,
            gain_tolerance_db=2.0,
        )

    assert result.warnings
    assert "deviates" in result.warnings[0]
