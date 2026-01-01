from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from microstructure_metrics.filterbank import GammatoneFilterbank

EPS = 1e-12


@dataclass(frozen=True)
class NPSResult:
    """Notch Preservation Score (NPS)."""

    nps_db: float
    nps_ratio: float
    ref_notch_depth_db: float
    dut_notch_depth_db: float
    notch_center_hz: float
    notch_q: float
    noise_floor_db: float
    is_noise_limited: bool


def calculate_nps(
    *,
    reference: npt.ArrayLike,
    dut: npt.ArrayLike,
    sample_rate: int,
    notch_center_hz: float = 8000.0,
    notch_q: float = 8.6,
    num_filters: int = 64,
    low_freq: float = 20.0,
    high_freq: float | None = None,
    filter_width: float = 1.0,
    adjacent_span: int = 2,
    dynamic_range_threshold_db: float = 12.0,
) -> NPSResult:
    """Measure how well the DUT preserves the spectral notch of the reference."""

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
    if notch_center_hz <= 0 or notch_q <= 0:
        raise ValueError("notch_center_hz and notch_q must be positive")
    nyquist = sample_rate / 2
    if notch_center_hz >= nyquist:
        raise ValueError("notch_center_hz must be below Nyquist")
    if adjacent_span < 1:
        raise ValueError("adjacent_span must be >= 1")

    fb = GammatoneFilterbank(
        sample_rate=sample_rate,
        num_filters=num_filters,
        low_freq=low_freq,
        high_freq=high_freq,
        width=filter_width,
    )
    centers = fb.center_frequencies
    ref_powers = fb.band_powers(ref)
    dut_powers = fb.band_powers(du)

    notch_half_bw = notch_center_hz / (2 * notch_q)
    notch_mask = np.abs(centers - notch_center_hz) <= notch_half_bw
    if not np.any(notch_mask):
        nearest = int(np.argmin(np.abs(centers - notch_center_hz)))
        notch_mask[nearest] = True
    notch_indices = np.nonzero(notch_mask)[0]
    anchor_idx = int(np.argmin(np.abs(centers - notch_center_hz)))

    adj_indices = _collect_adjacent_indices(
        anchor_idx=anchor_idx,
        length=ref_powers.shape[0],
        span=adjacent_span,
        exclude_mask=notch_mask,
    )
    if not adj_indices:
        raise ValueError("notch is too close to band edges for adjacent measurement")

    ref_notch_power = float(np.mean(ref_powers[notch_indices]))
    dut_notch_power = float(np.mean(dut_powers[notch_indices]))
    ref_adjacent_power = float(np.mean(ref_powers[adj_indices]))
    dut_adjacent_power = float(np.mean(dut_powers[adj_indices]))

    ref_depth_ratio = ref_adjacent_power / max(ref_notch_power, EPS)
    dut_depth_ratio = dut_adjacent_power / max(dut_notch_power, EPS)

    ref_depth_db = _ratio_to_db(ref_depth_ratio)
    dut_depth_db = _ratio_to_db(dut_depth_ratio)
    nps_ratio = dut_depth_ratio / max(ref_depth_ratio, EPS)

    combined = np.concatenate([ref_powers, dut_powers])
    noise_floor_power = float(np.percentile(combined, 10))
    noise_floor_db = _power_to_db(noise_floor_power)
    dynamic_range_db = _ratio_to_db(
        max(ref_adjacent_power, dut_adjacent_power) / max(noise_floor_power, EPS)
    )
    is_noise_limited = dynamic_range_db < dynamic_range_threshold_db

    return NPSResult(
        nps_db=_ratio_to_db(nps_ratio),
        nps_ratio=nps_ratio,
        ref_notch_depth_db=ref_depth_db,
        dut_notch_depth_db=dut_depth_db,
        notch_center_hz=float(notch_center_hz),
        notch_q=float(notch_q),
        noise_floor_db=noise_floor_db,
        is_noise_limited=is_noise_limited,
    )


def _collect_adjacent_indices(
    *,
    anchor_idx: int,
    length: int,
    span: int,
    exclude_mask: npt.NDArray[np.bool_],
) -> list[int]:
    indices: list[int] = []
    offset = 1
    target_count = span * 2
    while len(indices) < target_count and (
        anchor_idx - offset >= 0 or anchor_idx + offset < length
    ):
        for candidate in (anchor_idx - offset, anchor_idx + offset):
            if 0 <= candidate < length and not bool(exclude_mask[candidate]):
                indices.append(candidate)
                if len(indices) >= target_count:
                    break
        offset += 1
    return indices


def _ratio_to_db(ratio: float) -> float:
    return 10.0 * math.log10(max(ratio, EPS))


def _power_to_db(power: float) -> float:
    return 10.0 * math.log10(max(power, EPS))
