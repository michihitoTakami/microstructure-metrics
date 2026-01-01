from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal.windows import blackmanharris

EPS = 1e-20


@dataclass
class THDNResult:
    thd_n_db: float
    thd_n_percent: float
    thd_db: float
    noise_db: float
    fundamental_freq: float
    fundamental_level_dbfs: float
    harmonic_levels: dict[int, float]
    sinad_db: float
    measurement_bandwidth: tuple[float, float]
    warnings: list[str]


def calculate_thd_n(
    *,
    signal: npt.ArrayLike,
    fundamental_freq: float,
    sample_rate: int,
    bandwidth: tuple[float, float] = (20.0, 20000.0),
    num_harmonics: int = 10,
    bin_padding: int = 6,
    search_hz: float | None = None,
    expected_level_dbfs: float | None = -3.0,
    gain_tolerance_db: float = 2.0,
) -> THDNResult:
    """
    Calculate THD+N using an FFT-based method.

    The function extracts the fundamental component, aggregates harmonics up to
    ``num_harmonics``, estimates the residual noise floor within ``bandwidth``,
    and returns distortion/noise ratios relative to the fundamental.
    """
    data = np.asarray(signal, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("signal must be a 1-D array")
    if data.size == 0:
        raise ValueError("signal must contain at least one sample")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if fundamental_freq <= 0:
        raise ValueError("fundamental_freq must be positive")
    if num_harmonics < 1:
        raise ValueError("num_harmonics must be >= 1")

    band_low, band_high = bandwidth
    if band_low < 0 or band_high <= band_low:
        raise ValueError("bandwidth must be an increasing (low, high) tuple")
    nyquist = sample_rate / 2
    band_high = min(band_high, nyquist)

    n_samples = data.shape[0]
    n_fft = _next_pow_two(n_samples)

    window = blackmanharris(n_samples, sym=False)
    window_rms = float(np.sqrt(np.mean(np.square(window))))
    windowed = data * window

    spectrum = np.fft.rfft(windowed, n=n_fft)
    freqs: npt.NDArray[np.float64] = np.asarray(
        np.fft.rfftfreq(n_fft, 1.0 / sample_rate), dtype=np.float64
    )
    bin_width = freqs[1] - freqs[0] if freqs.shape[0] > 1 else 0.0
    power_spectrum: npt.NDArray[np.float64] = _weighted_power(spectrum)
    if not np.isfinite(power_spectrum).all():
        raise ValueError("spectrum contains non-finite values")

    band_mask = (freqs >= band_low) & (freqs <= band_high)
    if not np.any(band_mask):
        raise ValueError("bandwidth does not overlap with the FFT frequency bins")

    band_power = float(power_spectrum[band_mask].sum() / (n_fft**2))
    if band_power <= 0:
        raise ValueError("signal has no energy in the specified bandwidth")

    fund_idx = _find_peak_bin(
        power_spectrum, freqs, target_freq=fundamental_freq, search_hz=search_hz
    )
    fund_slice = _bin_slice(fund_idx, padding=bin_padding, length=freqs.shape[0])
    fundamental_power = float(power_spectrum[fund_slice].sum() / (n_fft**2))
    if fundamental_power <= 0:
        raise ValueError("failed to detect fundamental component")
    fundamental_freq_est = float(
        np.average(freqs[fund_slice], weights=power_spectrum[fund_slice])
    )

    exclude_mask = np.zeros_like(freqs, dtype=bool)
    exclude_mask[fund_slice] = True
    harmonic_power = 0.0
    harmonic_powers: dict[int, float] = {}
    band_high_limit = band_high + bin_width * bin_padding
    for order in range(2, num_harmonics + 1):
        harmonic_freq = fundamental_freq_est * order
        if harmonic_freq > band_high_limit or harmonic_freq >= nyquist:
            continue
        try:
            harmonic_idx = _find_peak_bin(
                power_spectrum,
                freqs,
                target_freq=harmonic_freq,
                search_hz=search_hz,
            )
        except ValueError:
            continue
        harmonic_slice = _bin_slice(
            harmonic_idx, padding=bin_padding, length=freqs.shape[0]
        )
        exclude_mask[harmonic_slice] = True
        hp = float(power_spectrum[harmonic_slice].sum() / (n_fft**2))
        harmonic_power += hp
        harmonic_powers[order] = hp

    noise_power = float(power_spectrum[band_mask & ~exclude_mask].sum() / (n_fft**2))
    dist_noise_power = harmonic_power + noise_power

    fundamental_rms = math.sqrt(max(fundamental_power, EPS))
    dist_noise_rms = math.sqrt(dist_noise_power)
    harmonic_rms = math.sqrt(max(harmonic_power, 0.0))
    noise_rms = math.sqrt(noise_power)

    scale = 1.0 / max(window_rms, EPS)
    fundamental_rms *= scale
    dist_noise_rms *= scale
    harmonic_rms *= scale
    noise_rms *= scale

    thd_n_ratio = dist_noise_rms / max(fundamental_rms, EPS)
    thd_ratio = harmonic_rms / max(fundamental_rms, EPS)
    noise_ratio = noise_rms / max(fundamental_rms, EPS)

    fundamental_peak = fundamental_rms * math.sqrt(2.0)
    fundamental_level_dbfs = _amplitude_to_dbfs(fundamental_peak)

    harmonic_levels = {
        order: _amplitude_to_dbfs(math.sqrt(max(power, EPS)) * scale * math.sqrt(2.0))
        for order, power in harmonic_powers.items()
    }

    warnings_list: list[str] = []
    if expected_level_dbfs is not None:
        deviation = fundamental_level_dbfs - expected_level_dbfs
        if abs(deviation) > gain_tolerance_db:
            message = (
                "Fundamental level deviates from expected "
                f"({fundamental_level_dbfs:.2f} dBFS vs expected "
                f"{expected_level_dbfs:+.2f} dBFS)"
            )
            warnings.warn(message, stacklevel=2)
            warnings_list.append(message)

    return THDNResult(
        thd_n_db=_ratio_to_db(thd_n_ratio),
        thd_n_percent=thd_n_ratio * 100,
        thd_db=_ratio_to_db(thd_ratio),
        noise_db=_ratio_to_db(noise_ratio),
        fundamental_freq=fundamental_freq_est,
        fundamental_level_dbfs=fundamental_level_dbfs,
        harmonic_levels=harmonic_levels,
        sinad_db=_ratio_to_db(max(fundamental_rms, EPS) / max(dist_noise_rms, EPS)),
        measurement_bandwidth=(float(band_low), float(band_high)),
        warnings=warnings_list,
    )


def _next_pow_two(n: int) -> int:
    return 1 << (max(n - 1, 0).bit_length())


def _bin_slice(center: int, *, padding: int, length: int) -> slice:
    start = max(center - padding, 0)
    end = min(center + padding + 1, length)
    return slice(start, end)


def _weighted_power(spectrum: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
    power = np.abs(spectrum) ** 2
    weights = np.ones_like(power, dtype=np.float64)
    if power.shape[0] > 1:
        weights[1:-1] = 2.0
    return np.asarray(power * weights, dtype=np.float64)


def _find_peak_bin(
    power_spectrum: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    *,
    target_freq: float,
    search_hz: float | None,
) -> int:
    bin_width = freqs[1] - freqs[0] if freqs.shape[0] > 1 else 0.0
    search_radius = search_hz or max(bin_width * 3, target_freq * 0.02)
    mask = np.abs(freqs - target_freq) <= search_radius
    if not np.any(mask):
        raise ValueError(f"no bins near target frequency {target_freq} Hz")
    local_power = power_spectrum[mask]
    peak_offset = int(np.argmax(local_power))
    return int(np.nonzero(mask)[0][peak_offset])


def _amplitude_to_dbfs(amplitude: float) -> float:
    amp = max(amplitude, EPS)
    return 20.0 * math.log10(amp)


def _ratio_to_db(ratio: float) -> float:
    return 20.0 * math.log10(max(ratio, EPS))
