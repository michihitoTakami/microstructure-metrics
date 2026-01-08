from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy import signal as sp_signal

EPS: Final = 1e-12
TWO_PI: Final = 2 * np.pi


@dataclass(frozen=True)
class BassBandMetrics:
    """Per-band LFCR metrics."""

    band_hz: tuple[float, float]
    cycle_shape_corr_mean: float
    cycle_shape_corr_p05: float
    harmonic_phase_coherence: float
    envelope_diff_outlier_rate: float
    cycles_used: int
    weight: float
    fundamental_hz: float
    harmonic_orders: tuple[int, ...]
    cycle_points: int


@dataclass(frozen=True)
class BassResult:
    """Low-frequency complex reconstruction (LFCR) summary."""

    cycle_shape_corr_mean: float
    cycle_shape_corr_p05: float
    harmonic_phase_coherence: float
    envelope_diff_outlier_rate: float
    band_metrics: tuple[BassBandMetrics, ...]
    bands_hz: tuple[tuple[float, float], ...]
    filter_order: int
    cycle_points: int
    envelope_threshold_db: float
    harmonic_max_order: int
    fundamental_search_hz: tuple[float, float]
    used_cycles: int


def calculate_low_freq_complex_reconstruction(
    *,
    reference: npt.ArrayLike,
    dut: npt.ArrayLike,
    sample_rate: int,
    bands_hz: Sequence[Sequence[float]] = ((20.0, 80.0), (80.0, 200.0)),
    filter_order: int = 4,
    cycle_points: int = 128,
    envelope_threshold_db: float = -50.0,
    harmonic_max_order: int = 5,
    fundamental_search_hz: tuple[float, float] = (30.0, 180.0),
) -> BassResult:
    """Quantify LFCR via cycle shape, harmonic phase, and envelope stability."""

    ref = np.asarray(reference, dtype=np.float64)
    du = np.asarray(dut, dtype=np.float64)
    if ref.ndim != 1 or du.ndim != 1:
        raise ValueError("reference/dut must be 1-D signals")
    if ref.shape[0] != du.shape[0]:
        raise ValueError("reference/dut length mismatch; align signals first")
    if ref.size == 0:
        raise ValueError("reference/dut must not be empty")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if filter_order < 1:
        raise ValueError("filter_order must be >= 1")
    if cycle_points < 8:
        raise ValueError("cycle_points must be >= 8")
    if harmonic_max_order < 2:
        raise ValueError("harmonic_max_order must be >= 2")
    if fundamental_search_hz[0] <= 0 or fundamental_search_hz[1] <= 0:
        raise ValueError("fundamental_search_hz must be positive")

    bands = _sanitize_bands(bands_hz, sample_rate)
    envelope_threshold_db = float(envelope_threshold_db)
    envelope_scale = float(
        np.max(np.abs(np.concatenate([ref, du]))) if ref.size else EPS
    )
    envelope_threshold = envelope_scale * 10 ** (envelope_threshold_db / 20.0)

    band_results: list[BassBandMetrics] = []
    all_cycle_corrs: list[float] = []
    all_cycle_weights: list[float] = []
    band_weights: list[float] = []
    harmonic_scores: list[float] = []
    envelope_outliers: list[float] = []
    total_cycles = 0

    for low, high in bands:
        ref_band = _bandpass(
            data=ref, sample_rate=sample_rate, low=low, high=high, order=filter_order
        )
        dut_band = _bandpass(
            data=du, sample_rate=sample_rate, low=low, high=high, order=filter_order
        )
        weight = float(np.sqrt(np.mean(np.square(ref_band), dtype=np.float64)))
        band_weights.append(weight)

        cycle_corrs, cycle_weights = _cycle_correlations(
            ref_band=ref_band,
            dut_band=dut_band,
            cycle_points=cycle_points,
            envelope_threshold=envelope_threshold,
        )
        total_cycles += len(cycle_corrs)
        all_cycle_corrs.extend(cycle_corrs)
        all_cycle_weights.extend(cycle_weights)
        band_cycle_mean = _weighted_mean(cycle_corrs, cycle_weights)
        band_cycle_p05 = _weighted_percentile(
            np.asarray(cycle_corrs, dtype=np.float64),
            np.asarray(cycle_weights, dtype=np.float64),
            0.05,
        )

        fundamental = _estimate_fundamental(
            ref_band=ref_band,
            sample_rate=sample_rate,
            search_range_hz=fundamental_search_hz,
        )
        harmonic_orders, phase_coherence = _harmonic_phase_alignment(
            ref_band=ref_band,
            dut_band=dut_band,
            sample_rate=sample_rate,
            fundamental_hz=fundamental,
            max_order=harmonic_max_order,
        )
        harmonic_scores.append(phase_coherence)

        env_outlier = _envelope_diff_outlier_rate(ref_band=ref_band, dut_band=dut_band)
        envelope_outliers.append(env_outlier)

        band_results.append(
            BassBandMetrics(
                band_hz=(low, high),
                cycle_shape_corr_mean=band_cycle_mean,
                cycle_shape_corr_p05=band_cycle_p05,
                harmonic_phase_coherence=phase_coherence,
                envelope_diff_outlier_rate=env_outlier,
                cycles_used=len(cycle_corrs),
                weight=weight,
                fundamental_hz=fundamental,
                harmonic_orders=harmonic_orders,
                cycle_points=cycle_points,
            )
        )

    band_weights_arr = np.asarray(band_weights, dtype=np.float64)
    harmonic_scores_arr = np.asarray(harmonic_scores, dtype=np.float64)
    envelope_outliers_arr = np.asarray(envelope_outliers, dtype=np.float64)

    cycle_shape_corr_mean = _weighted_mean(all_cycle_corrs, all_cycle_weights)
    cycle_shape_corr_p05 = _weighted_percentile(
        np.asarray(all_cycle_corrs, dtype=np.float64),
        np.asarray(all_cycle_weights, dtype=np.float64),
        0.05,
    )
    harmonic_phase_coherence = _weighted_mean(harmonic_scores_arr, band_weights_arr)
    envelope_diff_outlier_rate = _weighted_mean(envelope_outliers_arr, band_weights_arr)

    return BassResult(
        cycle_shape_corr_mean=cycle_shape_corr_mean,
        cycle_shape_corr_p05=cycle_shape_corr_p05,
        harmonic_phase_coherence=harmonic_phase_coherence,
        envelope_diff_outlier_rate=envelope_diff_outlier_rate,
        band_metrics=tuple(band_results),
        bands_hz=tuple(bands),
        filter_order=int(filter_order),
        cycle_points=int(cycle_points),
        envelope_threshold_db=float(envelope_threshold_db),
        harmonic_max_order=int(harmonic_max_order),
        fundamental_search_hz=(
            float(fundamental_search_hz[0]),
            float(fundamental_search_hz[1]),
        ),
        used_cycles=int(total_cycles),
    )


def _sanitize_bands(
    bands_hz: Sequence[Sequence[float]], sample_rate: int
) -> tuple[tuple[float, float], ...]:
    nyquist = sample_rate / 2
    sanitized: list[tuple[float, float]] = []
    for band in bands_hz:
        if len(band) != 2:
            raise ValueError("Each band must be a (low, high) tuple")
        low, high = float(band[0]), float(band[1])
        if low <= 0 or high <= low:
            raise ValueError("Each band must satisfy 0 < low < high")
        if high >= nyquist:
            raise ValueError("Band edges must be below Nyquist")
        sanitized.append((low, high))
    if not sanitized:
        raise ValueError("bands_hz must not be empty")
    return tuple(sanitized)


def _bandpass(
    *,
    data: npt.NDArray[np.float64],
    sample_rate: int,
    low: float,
    high: float,
    order: int,
) -> npt.NDArray[np.float64]:
    nyquist = sample_rate / 2
    sos = sp_signal.butter(
        order, [low / nyquist, high / nyquist], btype="band", output="sos"
    )
    return np.asarray(sp_signal.sosfiltfilt(sos, data), dtype=np.float64)


def _cycle_correlations(
    *,
    ref_band: npt.NDArray[np.float64],
    dut_band: npt.NDArray[np.float64],
    cycle_points: int,
    envelope_threshold: float,
) -> tuple[list[float], list[float]]:
    analytic_ref = sp_signal.hilbert(ref_band)
    envelope_ref = np.abs(analytic_ref)
    phase_ref = np.unwrap(np.angle(analytic_ref))

    cycle_ids = np.floor((phase_ref - phase_ref[0]) / TWO_PI).astype(int)
    unique_cycles = np.unique(cycle_ids)
    phase_grid = np.linspace(0.0, TWO_PI, cycle_points, endpoint=False)

    corrs: list[float] = []
    weights: list[float] = []

    for cycle in unique_cycles:
        mask = cycle_ids == cycle
        if np.count_nonzero(mask) < 2:
            continue
        phase_segment = phase_ref[mask]
        span = float(phase_segment[-1] - phase_segment[0])
        if span < TWO_PI * 0.75:
            continue
        env_mean = float(np.mean(envelope_ref[mask]))
        if env_mean <= envelope_threshold:
            continue
        phase_rel = phase_segment - phase_segment[0]
        ref_seg = ref_band[mask]
        dut_seg = dut_band[mask]

        ref_shape = np.interp(
            phase_grid, phase_rel, ref_seg, left=ref_seg[0], right=ref_seg[-1]
        )
        dut_shape = np.interp(
            phase_grid, phase_rel, dut_seg, left=dut_seg[0], right=dut_seg[-1]
        )
        corrs.append(_pearson(ref_shape, dut_shape))
        weights.append(env_mean)
    return corrs, weights


def _estimate_fundamental(
    *,
    ref_band: npt.NDArray[np.float64],
    sample_rate: int,
    search_range_hz: tuple[float, float],
) -> float:
    low, high = search_range_hz
    if low >= high:
        return 0.0
    window = np.hanning(ref_band.size)
    windowed = ref_band * window
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(ref_band.size, 1.0 / sample_rate)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    mag = np.abs(spectrum[mask])
    if not np.any(np.isfinite(mag)):
        return 0.0
    idx = int(np.argmax(mag))
    return float(freqs[mask][idx])


def _harmonic_phase_alignment(
    *,
    ref_band: npt.NDArray[np.float64],
    dut_band: npt.NDArray[np.float64],
    sample_rate: int,
    fundamental_hz: float,
    max_order: int,
) -> tuple[tuple[int, ...], float]:
    if fundamental_hz <= 0:
        return (), 0.0
    window = np.hanning(ref_band.size)
    ref_spec = np.fft.rfft(ref_band * window)
    dut_spec = np.fft.rfft(dut_band * window)
    freqs = np.fft.rfftfreq(ref_band.size, 1.0 / sample_rate)

    def _phase_at(freq: float, spectrum: npt.NDArray[np.complex128]) -> float:
        if spectrum.size == 0:
            return 0.0
        idx = int(np.argmin(np.abs(freqs - freq)))
        return float(np.angle(spectrum[idx]))

    base_ref = _phase_at(fundamental_hz, ref_spec)
    base_dut = _phase_at(fundamental_hz, dut_spec)

    phase_diffs: list[float] = []
    used_orders: list[int] = []
    nyquist = sample_rate / 2

    for order in range(2, max_order + 1):
        target = fundamental_hz * order
        if target >= nyquist:
            break
        used_orders.append(order)
        ref_phase = _phase_at(target, ref_spec)
        dut_phase = _phase_at(target, dut_spec)
        ref_rel = _wrap_phase(ref_phase - order * base_ref)
        dut_rel = _wrap_phase(dut_phase - order * base_dut)
        phase_diffs.append(_wrap_phase(dut_rel - ref_rel))

    if not phase_diffs:
        return tuple(used_orders), 0.0
    vectors = np.exp(1j * np.asarray(phase_diffs, dtype=np.float64))
    coherence = float(np.abs(np.mean(vectors)))
    return tuple(used_orders), coherence


def _envelope_diff_outlier_rate(
    *, ref_band: npt.NDArray[np.float64], dut_band: npt.NDArray[np.float64]
) -> float:
    analytic_ref = sp_signal.hilbert(ref_band)
    analytic_dut = sp_signal.hilbert(dut_band)
    env_ref = np.abs(analytic_ref)
    env_dut = np.abs(analytic_dut)
    scale = float(max(np.max(env_ref, initial=0.0), np.max(env_dut, initial=0.0), EPS))
    if scale <= 0:
        return 0.0
    ref_grad = np.diff(env_ref / scale)
    dut_grad = np.diff(env_dut / scale)
    if ref_grad.size == 0 or dut_grad.size == 0:
        return 0.0
    ref_abs = np.abs(ref_grad)
    baseline = float(np.median(ref_abs))
    threshold = float(np.percentile(ref_abs, 95)) + baseline
    deviation = np.abs(dut_grad - ref_grad)
    if threshold <= 0:
        return 0.0
    return float(np.mean(deviation > threshold))


def _weighted_percentile(
    values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64], percentile: float
) -> float:
    if values.size == 0 or weights.size == 0:
        return 0.0
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return 0.0
    vals = values[valid]
    w = weights[valid]
    order = np.argsort(vals)
    vals_sorted = vals[order]
    w_sorted = w[order]
    cumulative = np.cumsum(w_sorted)
    cutoff = percentile * cumulative[-1]
    idx = int(np.searchsorted(cumulative, cutoff))
    idx = min(idx, vals_sorted.size - 1)
    return float(vals_sorted[idx])


def _weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    vals_arr = np.asarray(list(values), dtype=np.float64)
    weights_arr = np.asarray(list(weights), dtype=np.float64)
    if vals_arr.size == 0:
        return 0.0
    if weights_arr.size != vals_arr.size:
        raise ValueError("values and weights must have the same length")
    total = float(np.sum(weights_arr))
    if total <= 0 or not np.isfinite(total):
        return float(np.mean(vals_arr))
    return float(np.sum(vals_arr * weights_arr) / total)


def _pearson(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    if a.shape != b.shape:
        raise ValueError("arrays must have the same shape for correlation")
    a_dev = a - float(np.mean(a))
    b_dev = b - float(np.mean(b))
    denom = float(np.sqrt(np.sum(a_dev**2) * np.sum(b_dev**2)))
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    return float(np.sum(a_dev * b_dev) / denom)


def _wrap_phase(phase: float | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(np.angle(np.exp(1j * phase)), dtype=np.float64)
