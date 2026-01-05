from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal

EPS = 1e-12


@dataclass(frozen=True)
class TransientParams:
    """Detection parameters used for transient analysis."""

    # Envelope smoothing window. Set to 0.0 to use Hilbert envelope (no smoothing).
    smoothing_ms: float = 0.05
    peak_threshold_db: float = -25.0
    refractory_ms: float = 2.5
    match_tolerance_ms: float = 1.5
    max_event_duration_ms: float = 40.0
    width_fraction: float = 0.3
    # Half-window (±ms) around peak used for pre-energy / skewness computation.
    asymmetry_window_ms: float = 3.0


@dataclass(frozen=True)
class DistributionStats:
    """Summary statistics of a distribution."""

    mean: float
    median: float
    percentile_05: float
    percentile_95: float
    std: float


@dataclass(frozen=True)
class TransientEvent:
    """Per-event features derived from the envelope."""

    peak_index: int
    peak_time_ms: float
    peak_value: float
    low_level_attack_time_ms: float
    attack_time_ms: float
    edge_sharpness: float
    width_ms: float
    pre_energy_fraction: float
    pre_post_energy_ratio: float
    energy_skewness: float


@dataclass(frozen=True)
class TransientResult:
    """Transient/edge metrics comparing reference and DUT."""

    low_level_attack_time_ref_ms: float
    low_level_attack_time_dut_ms: float
    low_level_attack_time_delta_ms: float
    attack_time_ref_ms: float
    attack_time_dut_ms: float
    attack_time_delta_ms: float
    edge_sharpness_ref: float
    edge_sharpness_dut: float
    edge_sharpness_ratio: float
    width_ref_ms: float
    width_dut_ms: float
    transient_smearing_index: float
    pre_energy_fraction_ref: float
    pre_energy_fraction_dut: float
    pre_energy_fraction_delta: float
    energy_skewness_ref: float
    energy_skewness_dut: float
    energy_skewness_delta: float
    ref_events: tuple[TransientEvent, ...]
    dut_events: tuple[TransientEvent, ...]
    matched_event_pairs: int
    unmatched_ref_events: int
    unmatched_dut_events: int
    low_level_attack_time_stats_ref: DistributionStats
    low_level_attack_time_stats_dut: DistributionStats
    low_level_attack_time_delta_stats_ms: DistributionStats
    attack_time_stats_ref: DistributionStats
    attack_time_stats_dut: DistributionStats
    attack_time_delta_stats_ms: DistributionStats
    edge_sharpness_stats_ref: DistributionStats
    edge_sharpness_stats_dut: DistributionStats
    edge_sharpness_ratio_stats: DistributionStats
    width_stats_ref: DistributionStats
    width_stats_dut: DistributionStats
    width_ratio_stats: DistributionStats
    pre_energy_fraction_stats_ref: DistributionStats
    pre_energy_fraction_stats_dut: DistributionStats
    pre_energy_fraction_delta_stats: DistributionStats
    energy_skewness_stats_ref: DistributionStats
    energy_skewness_stats_dut: DistributionStats
    energy_skewness_delta_stats: DistributionStats
    params: TransientParams


def calculate_transient_metrics(
    *,
    reference: npt.ArrayLike,
    dut: npt.ArrayLike,
    sample_rate: int,
    smoothing_ms: float = 0.05,
    peak_threshold_db: float = -25.0,
    refractory_ms: float = 2.5,
    match_tolerance_ms: float = 1.5,
    max_event_duration_ms: float = 40.0,
    width_fraction: float = 0.3,
    asymmetry_window_ms: float = 3.0,
) -> TransientResult:
    """Quantify transient sharpness and smearing using multiple events."""

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
    if smoothing_ms < 0:
        raise ValueError("smoothing_ms must be non-negative")
    if refractory_ms <= 0:
        raise ValueError("refractory_ms must be positive")
    if match_tolerance_ms <= 0:
        raise ValueError("match_tolerance_ms must be positive")
    if max_event_duration_ms <= 0:
        raise ValueError("max_event_duration_ms must be positive")
    if not 0.0 < width_fraction < 1.0:
        raise ValueError("width_fraction must be in (0, 1)")
    if asymmetry_window_ms <= 0:
        raise ValueError("asymmetry_window_ms must be positive")

    params = TransientParams(
        smoothing_ms=smoothing_ms,
        peak_threshold_db=peak_threshold_db,
        refractory_ms=refractory_ms,
        match_tolerance_ms=match_tolerance_ms,
        max_event_duration_ms=max_event_duration_ms,
        width_fraction=width_fraction,
        asymmetry_window_ms=asymmetry_window_ms,
    )

    env_ref = _smoothed_envelope(
        ref, sample_rate=sample_rate, smoothing_ms=params.smoothing_ms
    )
    env_dut = _smoothed_envelope(
        du, sample_rate=sample_rate, smoothing_ms=params.smoothing_ms
    )

    ref_events = _detect_events(
        data=ref, envelope=env_ref, sample_rate=sample_rate, params=params
    )
    dut_events = _detect_events(
        data=du, envelope=env_dut, sample_rate=sample_rate, params=params
    )

    tolerance_samples = max(
        1, int(round(sample_rate * params.match_tolerance_ms / 1000))
    )
    pairs = _match_events(ref_events, dut_events, tolerance_samples=tolerance_samples)

    edge_ratios = [
        dut.edge_sharpness / max(ref.edge_sharpness, EPS) for ref, dut in pairs
    ]
    width_ratios = [dut.width_ms / max(ref.width_ms, EPS) for ref, dut in pairs]
    attack_deltas = [dut.attack_time_ms - ref.attack_time_ms for ref, dut in pairs]
    low_level_attack_deltas = [
        dut.low_level_attack_time_ms - ref.low_level_attack_time_ms
        for ref, dut in pairs
    ]
    pre_energy_deltas = [
        dut.pre_energy_fraction - ref.pre_energy_fraction for ref, dut in pairs
    ]
    skewness_deltas = [dut.energy_skewness - ref.energy_skewness for ref, dut in pairs]

    low_level_attack_stats_ref = _describe_distribution(
        [e.low_level_attack_time_ms for e in ref_events]
    )
    low_level_attack_stats_dut = _describe_distribution(
        [e.low_level_attack_time_ms for e in dut_events]
    )
    low_level_attack_delta_stats = _describe_distribution(low_level_attack_deltas)

    attack_stats_ref = _describe_distribution([e.attack_time_ms for e in ref_events])
    attack_stats_dut = _describe_distribution([e.attack_time_ms for e in dut_events])
    attack_delta_stats = _describe_distribution(attack_deltas)

    edge_stats_ref = _describe_distribution([e.edge_sharpness for e in ref_events])
    edge_stats_dut = _describe_distribution([e.edge_sharpness for e in dut_events])
    edge_ratio_stats = _describe_distribution(edge_ratios)

    width_stats_ref = _describe_distribution([e.width_ms for e in ref_events])
    width_stats_dut = _describe_distribution([e.width_ms for e in dut_events])
    width_ratio_stats = _describe_distribution(width_ratios)

    pre_energy_stats_ref = _describe_distribution(
        [e.pre_energy_fraction for e in ref_events]
    )
    pre_energy_stats_dut = _describe_distribution(
        [e.pre_energy_fraction for e in dut_events]
    )
    pre_energy_delta_stats = _describe_distribution(pre_energy_deltas)

    skewness_stats_ref = _describe_distribution([e.energy_skewness for e in ref_events])
    skewness_stats_dut = _describe_distribution([e.energy_skewness for e in dut_events])
    skewness_delta_stats = _describe_distribution(skewness_deltas)

    low_level_attack_time_ref_ms = low_level_attack_stats_ref.median
    low_level_attack_time_dut_ms = low_level_attack_stats_dut.median
    low_level_attack_time_delta_ms = low_level_attack_delta_stats.median

    attack_time_ref_ms = attack_stats_ref.median
    attack_time_dut_ms = attack_stats_dut.median
    attack_time_delta_ms = attack_delta_stats.median

    edge_sharpness_ref = edge_stats_ref.median
    edge_sharpness_dut = edge_stats_dut.median
    edge_sharpness_ratio = edge_ratio_stats.median

    width_ref_ms = width_stats_ref.median
    width_dut_ms = width_stats_dut.median
    transient_smearing_index = width_ratio_stats.median

    pre_energy_fraction_ref = pre_energy_stats_ref.median
    pre_energy_fraction_dut = pre_energy_stats_dut.median
    pre_energy_fraction_delta = pre_energy_delta_stats.median

    energy_skewness_ref = skewness_stats_ref.median
    energy_skewness_dut = skewness_stats_dut.median
    energy_skewness_delta = skewness_delta_stats.median

    matched_pairs = len(pairs)
    unmatched_ref = max(len(ref_events) - matched_pairs, 0)
    unmatched_dut = max(len(dut_events) - matched_pairs, 0)

    return TransientResult(
        low_level_attack_time_ref_ms=low_level_attack_time_ref_ms,
        low_level_attack_time_dut_ms=low_level_attack_time_dut_ms,
        low_level_attack_time_delta_ms=low_level_attack_time_delta_ms,
        attack_time_ref_ms=attack_time_ref_ms,
        attack_time_dut_ms=attack_time_dut_ms,
        attack_time_delta_ms=attack_time_delta_ms,
        edge_sharpness_ref=edge_sharpness_ref,
        edge_sharpness_dut=edge_sharpness_dut,
        edge_sharpness_ratio=edge_sharpness_ratio,
        width_ref_ms=width_ref_ms,
        width_dut_ms=width_dut_ms,
        transient_smearing_index=transient_smearing_index,
        pre_energy_fraction_ref=pre_energy_fraction_ref,
        pre_energy_fraction_dut=pre_energy_fraction_dut,
        pre_energy_fraction_delta=pre_energy_fraction_delta,
        energy_skewness_ref=energy_skewness_ref,
        energy_skewness_dut=energy_skewness_dut,
        energy_skewness_delta=energy_skewness_delta,
        ref_events=tuple(ref_events),
        dut_events=tuple(dut_events),
        matched_event_pairs=matched_pairs,
        unmatched_ref_events=unmatched_ref,
        unmatched_dut_events=unmatched_dut,
        low_level_attack_time_stats_ref=low_level_attack_stats_ref,
        low_level_attack_time_stats_dut=low_level_attack_stats_dut,
        low_level_attack_time_delta_stats_ms=low_level_attack_delta_stats,
        attack_time_stats_ref=attack_stats_ref,
        attack_time_stats_dut=attack_stats_dut,
        attack_time_delta_stats_ms=attack_delta_stats,
        edge_sharpness_stats_ref=edge_stats_ref,
        edge_sharpness_stats_dut=edge_stats_dut,
        edge_sharpness_ratio_stats=edge_ratio_stats,
        width_stats_ref=width_stats_ref,
        width_stats_dut=width_stats_dut,
        width_ratio_stats=width_ratio_stats,
        pre_energy_fraction_stats_ref=pre_energy_stats_ref,
        pre_energy_fraction_stats_dut=pre_energy_stats_dut,
        pre_energy_fraction_delta_stats=pre_energy_delta_stats,
        energy_skewness_stats_ref=skewness_stats_ref,
        energy_skewness_stats_dut=skewness_stats_dut,
        energy_skewness_delta_stats=skewness_delta_stats,
        params=params,
    )


def _smoothed_envelope(
    data: npt.NDArray[np.float64], *, sample_rate: int, smoothing_ms: float
) -> npt.NDArray[np.float64]:
    # Use Hilbert envelope as the base. This preserves low-level pre-ringing
    # better than an energy+window estimate, especially with small windows.
    analytic = signal.hilbert(data)
    envelope = np.asarray(np.abs(analytic), dtype=np.float64)

    if smoothing_ms <= 0:
        return envelope

    window = max(3, int(round(sample_rate * smoothing_ms / 1000)))
    if window % 2 == 0:
        window += 1
    kernel = np.hanning(window)
    kernel_sum = np.sum(kernel)
    if kernel_sum <= 0:
        return envelope
    kernel = kernel / kernel_sum
    smoothed = np.convolve(envelope, kernel, mode="same")
    return np.asarray(np.maximum(smoothed, 0.0), dtype=np.float64)


def _detect_events(
    *,
    data: npt.NDArray[np.float64],
    envelope: npt.NDArray[np.float64],
    sample_rate: int,
    params: TransientParams,
) -> list[TransientEvent]:
    """Detect transient peaks using a sliding window style scan."""

    if envelope.size == 0:
        return []

    peak = float(np.max(envelope))
    if peak <= EPS:
        return []

    ratio = 10.0 ** (params.peak_threshold_db / 20.0)
    rel_threshold = peak * ratio
    noise_floor = float(np.percentile(envelope, 90))
    threshold = max(rel_threshold, noise_floor * 0.5, EPS)

    distance = max(1, int(round(sample_rate * params.refractory_ms / 1000)))
    max_event_samples = max(
        1, int(round(sample_rate * params.max_event_duration_ms / 1000))
    )

    peaks, _ = signal.find_peaks(envelope, height=threshold, distance=distance)
    events: list[TransientEvent] = []
    for peak_idx in peaks:
        events.append(
            _extract_event_features(
                data=data,
                envelope=envelope,
                sample_rate=sample_rate,
                peak_idx=int(peak_idx),
                max_event_samples=max_event_samples,
                width_fraction=params.width_fraction,
                asymmetry_window_ms=params.asymmetry_window_ms,
            )
        )
    return events


def _match_events(
    ref_events: Sequence[TransientEvent],
    dut_events: Sequence[TransientEvent],
    *,
    tolerance_samples: int,
) -> list[tuple[TransientEvent, TransientEvent]]:
    """Greedy nearest-neighbour matching between reference and DUT peaks."""

    matched: list[tuple[TransientEvent, TransientEvent]] = []
    used: set[int] = set()

    for ref_event in ref_events:
        best_idx: int | None = None
        best_dist: int | None = None
        for idx, dut_event in enumerate(dut_events):
            if idx in used:
                continue
            distance = abs(dut_event.peak_index - ref_event.peak_index)
            if distance > tolerance_samples:
                continue
            if best_dist is None or distance < best_dist:
                best_dist = distance
                best_idx = idx
        if best_idx is None:
            continue
        used.add(best_idx)
        matched.append((ref_event, dut_events[best_idx]))
    return matched


def _extract_event_features(
    *,
    data: npt.NDArray[np.float64],
    envelope: npt.NDArray[np.float64],
    sample_rate: int,
    peak_idx: int,
    max_event_samples: int,
    width_fraction: float,
    asymmetry_window_ms: float,
) -> TransientEvent:
    peak = float(envelope[peak_idx])
    if peak <= EPS:
        return TransientEvent(
            peak_index=peak_idx,
            peak_time_ms=(peak_idx / sample_rate) * 1000.0,
            peak_value=0.0,
            low_level_attack_time_ms=0.0,
            attack_time_ms=0.0,
            edge_sharpness=0.0,
            width_ms=0.0,
            pre_energy_fraction=0.0,
            pre_post_energy_ratio=0.0,
            energy_skewness=0.0,
        )

    start_idx = max(0, peak_idx - max_event_samples)
    end_idx = min(envelope.shape[0], peak_idx + max_event_samples + 1)
    segment = envelope[start_idx:end_idx]
    local_peak_idx = peak_idx - start_idx

    threshold_001 = 0.001 * peak
    threshold_10 = 0.1 * peak
    threshold_90 = 0.9 * peak

    pre_peak = segment[: local_peak_idx + 1]
    below_001 = np.where(pre_peak <= threshold_001)[0]
    low_attack_start_local = int(below_001[-1]) if below_001.size else 0
    post_from_low_start = segment[low_attack_start_local : local_peak_idx + 1]
    above_10_from_low = np.where(post_from_low_start >= threshold_10)[0]
    low_attack_10_local = (
        low_attack_start_local + int(above_10_from_low[0])
        if above_10_from_low.size
        else local_peak_idx
    )
    low_attack_samples = max(1, low_attack_10_local - low_attack_start_local)
    low_level_attack_time_ms = (low_attack_samples / sample_rate) * 1000.0

    below_10 = np.where(pre_peak <= threshold_10)[0]
    attack_start_local = int(below_10[-1]) if below_10.size else 0

    post_from_start = segment[attack_start_local : local_peak_idx + 1]
    above_90 = np.where(post_from_start >= threshold_90)[0]
    attack_90_local = (
        attack_start_local + int(above_90[0]) if above_90.size else local_peak_idx
    )

    attack_samples = max(1, attack_90_local - attack_start_local)
    attack_time_ms = (attack_samples / sample_rate) * 1000.0

    norm_pre_peak = pre_peak / peak
    grad = np.gradient(norm_pre_peak)
    lookback = int(round(sample_rate * 0.003))
    region_start = max(0, pre_peak.shape[0] - lookback - 1)
    edge_sharpness = float(np.max(grad[region_start:]) * sample_rate)

    width_level = width_fraction * peak
    left_candidates = np.where(pre_peak <= width_level)[0]
    left_idx = int(left_candidates[-1]) if left_candidates.size else attack_start_local
    post_peak = segment[local_peak_idx:]
    right_candidates = np.where(post_peak < width_level)[0]
    right_local = (
        local_peak_idx + int(right_candidates[0])
        if right_candidates.size
        else segment.shape[0] - 1
    )
    width_samples = max(1, right_local - left_idx)
    width_ms = (width_samples / sample_rate) * 1000.0

    global_peak_idx = start_idx + local_peak_idx

    # Pre-energy / asymmetry: compute using raw-signal energy around the peak to
    # capture low-level pre-ringing that is often below -20 dB.
    half_window = max(1, int(round(sample_rate * asymmetry_window_ms / 1000)))
    win_start = max(0, global_peak_idx - half_window)
    win_end = min(data.shape[0] - 1, global_peak_idx + half_window)
    rel = np.arange(win_start, win_end + 1, dtype=np.float64) - float(global_peak_idx)
    weights = np.square(data[win_start : win_end + 1])
    total = float(np.sum(weights)) + EPS

    pre_mask = rel < 0
    post_mask = rel > 0
    pre_energy = float(np.sum(weights[pre_mask])) / total if np.any(pre_mask) else 0.0
    post_energy = (
        float(np.sum(weights[post_mask])) / total if np.any(post_mask) else 0.0
    )
    pre_post_energy_ratio = pre_energy / max(post_energy, EPS)

    centroid = float(np.sum(rel * weights) / total)
    centered = rel - centroid
    var = float(np.sum(np.square(centered) * weights) / total)
    std = float(np.sqrt(max(var, 0.0)))
    if std <= EPS:
        energy_skewness = 0.0
    else:
        m3 = float(np.sum((centered**3) * weights) / total)
        energy_skewness = m3 / (std**3)

    return TransientEvent(
        peak_index=global_peak_idx,
        peak_time_ms=(global_peak_idx / sample_rate) * 1000.0,
        peak_value=peak,
        low_level_attack_time_ms=low_level_attack_time_ms,
        attack_time_ms=attack_time_ms,
        edge_sharpness=edge_sharpness,
        width_ms=width_ms,
        pre_energy_fraction=pre_energy,
        pre_post_energy_ratio=pre_post_energy_ratio,
        energy_skewness=energy_skewness,
    )


def _describe_distribution(values: Sequence[float]) -> DistributionStats:
    if not values:
        return DistributionStats(
            mean=0.0,
            median=0.0,
            percentile_05=0.0,
            percentile_95=0.0,
            std=0.0,
        )
    arr = np.asarray(values, dtype=np.float64)
    return DistributionStats(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        percentile_05=float(np.percentile(arr, 5)),
        percentile_95=float(np.percentile(arr, 95)),
        std=float(np.std(arr)),
    )
