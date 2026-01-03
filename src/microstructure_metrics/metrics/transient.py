from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

EPS = 1e-12


@dataclass(frozen=True)
class TransientResult:
    """Transient/edge metrics comparing reference and DUT."""

    attack_time_ref_ms: float
    attack_time_dut_ms: float
    attack_time_delta_ms: float
    edge_sharpness_ref: float
    edge_sharpness_dut: float
    edge_sharpness_ratio: float
    width_ref_ms: float
    width_dut_ms: float
    transient_smearing_index: float


@dataclass(frozen=True)
class _TransientFeatures:
    attack_time_ms: float
    edge_sharpness: float
    width_ms: float


def calculate_transient_metrics(
    *,
    reference: npt.ArrayLike,
    dut: npt.ArrayLike,
    sample_rate: int,
    smoothing_ms: float = 0.25,
) -> TransientResult:
    """Quantify transient sharpness and smearing using envelope dynamics."""

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
    if smoothing_ms <= 0:
        raise ValueError("smoothing_ms must be positive")

    env_ref = _smoothed_envelope(
        ref, sample_rate=sample_rate, smoothing_ms=smoothing_ms
    )
    env_dut = _smoothed_envelope(du, sample_rate=sample_rate, smoothing_ms=smoothing_ms)

    features_ref = _extract_features(env_ref, sample_rate=sample_rate)
    features_dut = _extract_features(env_dut, sample_rate=sample_rate)

    edge_ratio = features_dut.edge_sharpness / max(features_ref.edge_sharpness, EPS)
    width_ratio = features_dut.width_ms / max(features_ref.width_ms, EPS)
    attack_delta = features_dut.attack_time_ms - features_ref.attack_time_ms

    return TransientResult(
        attack_time_ref_ms=features_ref.attack_time_ms,
        attack_time_dut_ms=features_dut.attack_time_ms,
        attack_time_delta_ms=attack_delta,
        edge_sharpness_ref=features_ref.edge_sharpness,
        edge_sharpness_dut=features_dut.edge_sharpness,
        edge_sharpness_ratio=edge_ratio,
        width_ref_ms=features_ref.width_ms,
        width_dut_ms=features_dut.width_ms,
        transient_smearing_index=width_ratio,
    )


def _smoothed_envelope(
    signal: npt.NDArray[np.float64], *, sample_rate: int, smoothing_ms: float
) -> npt.NDArray[np.float64]:
    energy = np.square(signal)
    window = max(3, int(round(sample_rate * smoothing_ms / 1000)))
    if window % 2 == 0:
        window += 1
    kernel = np.hanning(window)
    kernel_sum = np.sum(kernel)
    if kernel_sum <= 0:
        return np.abs(signal)
    kernel = kernel / kernel_sum
    smoothed = np.convolve(energy, kernel, mode="same")
    return np.sqrt(np.maximum(smoothed, 0.0))


def _extract_features(
    envelope: npt.NDArray[np.float64], *, sample_rate: int
) -> _TransientFeatures:
    peak = float(np.max(envelope))
    if peak <= EPS:
        return _TransientFeatures(attack_time_ms=0.0, edge_sharpness=0.0, width_ms=0.0)

    peak_idx = int(np.argmax(envelope))
    threshold_10 = 0.1 * peak
    threshold_90 = 0.9 * peak

    pre_peak = envelope[: peak_idx + 1]
    below_10 = np.where(pre_peak <= threshold_10)[0]
    attack_start_idx = int(below_10[-1]) if below_10.size else 0

    post_from_start = envelope[attack_start_idx : peak_idx + 1]
    above_90 = np.where(post_from_start >= threshold_90)[0]
    attack_90_idx = attack_start_idx + int(above_90[0]) if above_90.size else peak_idx

    attack_samples = max(1, attack_90_idx - attack_start_idx)
    attack_time_ms = (attack_samples / sample_rate) * 1000.0

    norm_pre_peak = pre_peak / peak
    grad = np.gradient(norm_pre_peak)
    lookback = int(round(sample_rate * 0.003))
    region_start = max(0, pre_peak.shape[0] - lookback - 1)
    edge_sharpness = float(np.max(grad[region_start:]) * sample_rate)

    half = 0.5 * peak
    left_candidates = np.where(pre_peak <= half)[0]
    left_idx = int(left_candidates[-1]) if left_candidates.size else attack_start_idx
    post_peak = envelope[peak_idx:]
    right_candidates = np.where(post_peak < half)[0]
    right_idx = (
        peak_idx + int(right_candidates[0])
        if right_candidates.size
        else envelope.shape[0] - 1
    )
    width_samples = max(1, right_idx - left_idx)
    width_ms = (width_samples / sample_rate) * 1000.0

    return _TransientFeatures(
        attack_time_ms=attack_time_ms,
        edge_sharpness=edge_sharpness,
        width_ms=width_ms,
    )
