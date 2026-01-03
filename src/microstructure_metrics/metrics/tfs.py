from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal as sp_signal

EPS = 1e-12


@dataclass(frozen=True)
class TFSComponents:
    """Time fine structure components of a single band."""

    fine_structure: npt.NDArray[np.float64]
    envelope: npt.NDArray[np.float64]
    instantaneous_phase: npt.NDArray[np.float64]


@dataclass(frozen=True)
class TFSCorrelationResult:
    """Result of TFS correlation between reference and DUT."""

    mean_correlation: float
    band_correlations: dict[tuple[float, float], float]
    phase_coherence: float
    group_delay_std_ms: float
    band_group_delays_ms: dict[tuple[float, float], float]


def extract_tfs(
    *,
    signal: npt.ArrayLike,
    sample_rate: int,
    center_freq: float,
    bandwidth: float,
    filter_order: int = 6,
) -> TFSComponents:
    """Extract TFS components around the given center frequency."""

    data = np.asarray(signal, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("signal must be 1-D")
    if data.size == 0:
        raise ValueError("signal must not be empty")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if center_freq <= 0 or bandwidth <= 0:
        raise ValueError("center_freq and bandwidth must be positive")
    if filter_order < 1:
        raise ValueError("filter_order must be >= 1")

    half_bw = bandwidth / 2
    low = center_freq - half_bw
    high = center_freq + half_bw
    nyquist = sample_rate / 2
    if low <= 0 or high >= nyquist:
        raise ValueError("band edges must be within (0, Nyquist)")

    sos = _bandpass(low=low, high=high, sample_rate=sample_rate, order=filter_order)
    filtered = sp_signal.sosfiltfilt(sos, data)
    analytic = sp_signal.hilbert(filtered)
    envelope = np.abs(analytic)
    instantaneous_phase = np.unwrap(np.angle(analytic))
    fine_structure = np.real(analytic) / np.maximum(envelope, EPS)

    return TFSComponents(
        fine_structure=np.asarray(fine_structure, dtype=np.float64),
        envelope=np.asarray(envelope, dtype=np.float64),
        instantaneous_phase=np.asarray(instantaneous_phase, dtype=np.float64),
    )


def calculate_tfs_correlation(
    *,
    reference: npt.ArrayLike,
    dut: npt.ArrayLike,
    sample_rate: int,
    freq_bands: list[tuple[float, float]] | tuple[tuple[float, float], ...] = (
        (2000.0, 3000.0),
        (3000.0, 4000.0),
        (4000.0, 6000.0),
        (6000.0, 8000.0),
    ),
    filter_order: int = 6,
) -> TFSCorrelationResult:
    """Calculate TFS correlation and phase coherence across bands."""

    ref = np.asarray(reference, dtype=np.float64)
    du = np.asarray(dut, dtype=np.float64)
    if ref.ndim != 1 or du.ndim != 1:
        raise ValueError("reference/dut must be 1-D signals")
    if ref.size == 0 or du.size == 0:
        raise ValueError("reference/dut must contain samples")
    if ref.shape[0] != du.shape[0]:
        raise ValueError("reference/dut length mismatch; align signals first")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if filter_order < 1:
        raise ValueError("filter_order must be >= 1")

    nyquist = sample_rate / 2
    bands = list(freq_bands)
    if not bands:
        raise ValueError("freq_bands must not be empty")

    band_correlations: dict[tuple[float, float], float] = {}
    band_group_delays_ms: dict[tuple[float, float], float] = {}
    phase_vector_sum = 0.0 + 0.0j
    phase_count = 0

    for low, high in bands:
        if low <= 0 or high <= low:
            raise ValueError("freq_bands must be increasing (low, high) tuples")
        if high >= nyquist:
            raise ValueError("freq_bands must stay below Nyquist")

        components_ref = _extract_band(
            signal=ref,
            sample_rate=sample_rate,
            low=low,
            high=high,
            filter_order=filter_order,
        )
        components_dut = _extract_band(
            signal=du,
            sample_rate=sample_rate,
            low=low,
            high=high,
            filter_order=filter_order,
        )

        corr, lag = _normalized_correlation(
            components_ref.fine_structure, components_dut.fine_structure
        )
        band_key = (float(low), float(high))
        band_correlations[band_key] = corr
        band_group_delays_ms[band_key] = (lag / sample_rate) * 1000.0

        # Compensate constant time lag (group delay) before phase coherence.
        # Without this, a benign fixed delay between reference/DUT can artificially
        # reduce phase coherence even when fine structure matches.
        phase_ref, phase_dut = _overlap_with_lag(
            components_ref.instantaneous_phase, components_dut.instantaneous_phase, lag
        )
        if phase_ref.size == 0:
            continue
        phase_diff = phase_ref - phase_dut
        wrapped = _wrap_phase(phase_diff)
        phase_vector_sum += np.sum(np.exp(1j * wrapped))
        phase_count += wrapped.size

    mean_correlation = float(np.mean(list(band_correlations.values())))
    phase_coherence = (
        float(np.abs(phase_vector_sum) / phase_count) if phase_count > 0 else 0.0
    )
    group_delay_std_ms = float(np.std(list(band_group_delays_ms.values())))

    return TFSCorrelationResult(
        mean_correlation=mean_correlation,
        band_correlations=band_correlations,
        phase_coherence=phase_coherence,
        group_delay_std_ms=group_delay_std_ms,
        band_group_delays_ms=band_group_delays_ms,
    )


def _extract_band(
    *,
    signal: npt.NDArray[np.float64],
    sample_rate: int,
    low: float,
    high: float,
    filter_order: int,
) -> TFSComponents:
    center = (low + high) / 2.0
    bandwidth = high - low
    return extract_tfs(
        signal=signal,
        sample_rate=sample_rate,
        center_freq=center,
        bandwidth=bandwidth,
        filter_order=filter_order,
    )


def _bandpass(*, low: float, high: float, sample_rate: int, order: int) -> npt.NDArray:
    sos = sp_signal.butter(
        order, [low, high], btype="bandpass", fs=sample_rate, output="sos"
    )
    return np.asarray(sos, dtype=np.float64)


def _normalized_correlation(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> tuple[float, int]:
    if a.shape != b.shape:
        raise ValueError("signals must have the same shape for correlation")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0, 0
    corr = sp_signal.correlate(a, b, mode="full", method="fft") / denom
    lags = sp_signal.correlation_lags(a.size, b.size, mode="full")
    # Prefer the lag that maximizes *positive* correlation.
    # Using |corr| can pick an inverted (anti-correlated) alignment, which then
    # pollutes downstream stats (including phase coherence).
    real_corr = np.real(corr)
    peak_idx = int(np.argmax(real_corr))
    return float(real_corr[peak_idx]), int(lags[peak_idx])


def _wrap_phase(phase_diff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Wrap phase difference to [-pi, pi] range."""

    return np.asarray(np.angle(np.exp(1j * phase_diff)), dtype=np.float64)


def _overlap_with_lag(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], lag: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return overlapping segments after aligning b to a using lag (in samples).

    lag is the shift (in samples) that maximizes correlation between (a, b).
    We drop non-overlapping edges instead of padding/circular shifting.
    """

    if a.shape != b.shape:
        raise ValueError("signals must have the same shape to apply lag overlap")
    if lag == 0:
        return a, b
    if lag > 0:
        # b is delayed relative to a -> drop a's head and b's tail
        if lag >= a.size:
            return a[:0], b[:0]
        return a[lag:], b[:-lag]
    # lag < 0: b is advanced relative to a -> drop a's tail and b's head
    shift = -lag
    if shift >= a.size:
        return a[:0], b[:0]
    return a[:-shift], b[shift:]
