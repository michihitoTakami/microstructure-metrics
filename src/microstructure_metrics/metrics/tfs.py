from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import signal as sp_signal

EPS = 1e-12
DEFAULT_FRAME_LENGTH_MS = 25.0
DEFAULT_FRAME_HOP_MS = 10.0
DEFAULT_MAX_LAG_MS = 1.0
DEFAULT_ENVELOPE_THRESHOLD_DB = -40.0
DEFAULT_WINDOW: Literal["hann"] = "hann"


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
    percentile_05_correlation: float
    correlation_variance: float
    band_correlations: dict[tuple[float, float], float]
    phase_coherence: float
    group_delay_std_ms: float
    band_group_delays_ms: dict[tuple[float, float], float]
    frame_length_ms: float
    frame_hop_ms: float
    max_lag_ms: float
    envelope_threshold_db: float
    frames_per_band: int
    used_frames: int


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
    frame_length_ms: float = DEFAULT_FRAME_LENGTH_MS,
    frame_hop_ms: float = DEFAULT_FRAME_HOP_MS,
    max_lag_ms: float = DEFAULT_MAX_LAG_MS,
    envelope_threshold_db: float = DEFAULT_ENVELOPE_THRESHOLD_DB,
    window: Literal["hann"] = DEFAULT_WINDOW,
) -> TFSCorrelationResult:
    """Calculate short-time TFS correlation and phase coherence across bands."""

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
    if frame_length_ms <= 0 or frame_hop_ms <= 0:
        raise ValueError("frame_length_ms and frame_hop_ms must be positive")
    if max_lag_ms < 0:
        raise ValueError("max_lag_ms must be non-negative")
    if envelope_threshold_db >= 0:
        raise ValueError("envelope_threshold_db must be negative (dB relative)")

    nyquist = sample_rate / 2
    bands = list(freq_bands)
    if not bands:
        raise ValueError("freq_bands must not be empty")

    frame_length_samples = int(round(frame_length_ms * sample_rate / 1000))
    hop_samples = int(round(frame_hop_ms * sample_rate / 1000))
    max_lag_samples = int(round(max_lag_ms * sample_rate / 1000))
    if frame_length_samples < 1 or hop_samples < 1:
        raise ValueError("frame_length_ms/hop_ms too small for given sample_rate")
    if frame_length_samples > ref.size:
        frame_length_samples = ref.size
    if frame_length_samples < hop_samples:
        hop_samples = frame_length_samples
    frames_per_band = _frame_count(ref.size, frame_length_samples, hop_samples)
    window_taper = _window(window, frame_length_samples)

    band_correlations: dict[tuple[float, float], float] = {}
    band_group_delays_ms: dict[tuple[float, float], float] = {}
    phase_vector_sum = 0.0 + 0.0j
    phase_count = 0
    all_correlations: list[float] = []
    all_weights: list[float] = []
    used_frames = 0

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
        band_threshold = _envelope_threshold(
            ref_envelope=components_ref.envelope,
            dut_envelope=components_dut.envelope,
            threshold_db=envelope_threshold_db,
        )
        correlations, lags, weights = _short_time_correlations(
            ref_fine=components_ref.fine_structure,
            dut_fine=components_dut.fine_structure,
            ref_envelope=components_ref.envelope,
            dut_envelope=components_dut.envelope,
            window=window_taper,
            frame_length=frame_length_samples,
            hop=hop_samples,
            max_lag=max_lag_samples,
            envelope_threshold=band_threshold,
        )
        used_frames += len(correlations)
        band_key = (float(low), float(high))
        if correlations:
            band_mean = float(np.average(correlations, weights=weights))
            band_lag_samples = int(_weighted_median(np.asarray(lags), weights))
            band_correlations[band_key] = band_mean
            band_group_delays_ms[band_key] = (band_lag_samples / sample_rate) * 1000.0
            all_correlations.extend(correlations)
            all_weights.extend(weights)
        else:
            band_correlations[band_key] = 0.0
            band_group_delays_ms[band_key] = 0.0
            band_lag_samples = 0

        # Compensate representative time lag (median of short-time lags) before
        # phase coherence. This keeps coherence robust to constant offsets while
        # still reflecting local phase instability.
        phase_ref, phase_dut = _overlap_with_lag(
            components_ref.instantaneous_phase,
            components_dut.instantaneous_phase,
            band_lag_samples,
        )
        if phase_ref.size == 0:
            continue
        phase_diff = phase_ref - phase_dut
        wrapped = _wrap_phase(phase_diff)
        phase_vector_sum += np.sum(np.exp(1j * wrapped))
        phase_count += wrapped.size

    if all_correlations:
        mean_correlation = float(np.average(all_correlations, weights=all_weights))
        variance = float(
            np.average(
                (np.asarray(all_correlations) - mean_correlation) ** 2,
                weights=all_weights,
            )
        )
        percentile_05 = float(np.percentile(all_correlations, 5))
    else:
        mean_correlation = 0.0
        variance = 0.0
        percentile_05 = 0.0
    phase_coherence = (
        float(np.abs(phase_vector_sum) / phase_count) if phase_count > 0 else 0.0
    )
    group_delay_std_ms = float(np.std(list(band_group_delays_ms.values())))

    return TFSCorrelationResult(
        mean_correlation=mean_correlation,
        percentile_05_correlation=percentile_05,
        correlation_variance=variance,
        band_correlations=band_correlations,
        phase_coherence=phase_coherence,
        group_delay_std_ms=group_delay_std_ms,
        band_group_delays_ms=band_group_delays_ms,
        frame_length_ms=float(frame_length_samples * 1000 / sample_rate),
        frame_hop_ms=float(hop_samples * 1000 / sample_rate),
        max_lag_ms=float(max_lag_samples * 1000 / sample_rate),
        envelope_threshold_db=float(envelope_threshold_db),
        frames_per_band=int(frames_per_band),
        used_frames=int(used_frames),
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
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    max_lag: int | None = None,
) -> tuple[float, int]:
    if a.shape != b.shape:
        raise ValueError("signals must have the same shape for correlation")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0, 0
    corr = sp_signal.correlate(a, b, mode="full", method="fft") / denom
    lags = sp_signal.correlation_lags(a.size, b.size, mode="full")
    if max_lag is not None:
        window = np.abs(lags) <= max_lag
        if not np.any(window):
            return 0.0, 0
        corr = corr[window]
        lags = lags[window]
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


def _frame_count(length: int, frame_length: int, hop: int) -> int:
    if length <= 0:
        return 0
    if frame_length <= 0 or hop <= 0:
        raise ValueError("frame_length and hop must be positive")
    if length < frame_length:
        return 1
    return 1 + (length - frame_length) // hop


def _envelope_threshold(
    *,
    ref_envelope: npt.NDArray[np.float64],
    dut_envelope: npt.NDArray[np.float64],
    threshold_db: float,
) -> float:
    peak = max(
        float(np.max(np.asarray(ref_envelope, dtype=np.float64), initial=0.0)),
        float(np.max(np.asarray(dut_envelope, dtype=np.float64), initial=0.0)),
        EPS,
    )
    return peak * 10 ** (threshold_db / 20.0)


def _window(name: Literal["hann"], length: int) -> npt.NDArray[np.float64]:
    if length < 1:
        raise ValueError("window length must be positive")
    if name == "hann":
        return np.asarray(np.hanning(length), dtype=np.float64)
    raise ValueError(f"Unsupported window type: {name}")


def _short_time_correlations(
    *,
    ref_fine: npt.NDArray[np.float64],
    dut_fine: npt.NDArray[np.float64],
    ref_envelope: npt.NDArray[np.float64],
    dut_envelope: npt.NDArray[np.float64],
    window: npt.NDArray[np.float64],
    frame_length: int,
    hop: int,
    max_lag: int,
    envelope_threshold: float,
) -> tuple[list[float], list[int], list[float]]:
    if (
        ref_fine.shape != dut_fine.shape
        or ref_envelope.shape != dut_envelope.shape
        or ref_fine.shape != ref_envelope.shape
    ):
        raise ValueError("ref/dut fine structures and envelopes must share shape")
    correlations: list[float] = []
    lags: list[int] = []
    weights: list[float] = []
    if frame_length < 1 or hop < 1:
        return correlations, lags, weights

    signal_length = ref_fine.size
    if signal_length < frame_length:
        frame_length = signal_length
    if frame_length == 0:
        return correlations, lags, weights

    for start in range(0, signal_length - frame_length + 1, hop):
        stop = start + frame_length
        env_mean = float(
            np.mean(
                0.5 * (ref_envelope[start:stop] + dut_envelope[start:stop]),
                dtype=np.float64,
            )
        )
        if env_mean <= envelope_threshold:
            continue
        ref_frame = ref_fine[start:stop] * window
        dut_frame = dut_fine[start:stop] * window
        corr, lag = _normalized_correlation(ref_frame, dut_frame, max_lag)
        correlations.append(corr)
        lags.append(lag)
        weights.append(env_mean)

    return correlations, lags, weights


def _weighted_median(values: npt.NDArray[np.float64], weights: list[float]) -> float:
    if values.size == 0:
        return 0.0
    weights_array = np.asarray(weights, dtype=np.float64)
    if weights_array.size != values.size:
        raise ValueError("values and weights must have the same length")
    if np.all(weights_array == 0):
        return float(np.median(values))
    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_weights = weights_array[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = 0.5 * cumulative[-1]
    idx = int(np.searchsorted(cumulative, cutoff))
    return float(sorted_vals[min(idx, sorted_vals.size - 1)])
