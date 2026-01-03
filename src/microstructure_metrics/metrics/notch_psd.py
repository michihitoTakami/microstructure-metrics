from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal as sp_signal

EPS = 1e-20


@dataclass(frozen=True)
class NarrowbandNotchDepthResult:
    """PSDベースのノッチ深さ評価結果。"""

    ref_notch_depth_db: float
    dut_notch_depth_db: float
    notch_fill_db: float
    ref_notch_power_db: float
    dut_notch_power_db: float
    ref_ring_power_db: float
    dut_ring_power_db: float
    notch_center_hz: float
    notch_bandwidth_hz: float
    ring_bandwidth_hz: float


def calculate_narrowband_notch_depth(
    *,
    reference: npt.ArrayLike,
    dut: npt.ArrayLike,
    sample_rate: int,
    notch_center_hz: float = 8000.0,
    notch_q: float = 20.0,
    notch_bandwidth_hz: float | None = None,
    ring_bandwidth_hz: float | None = None,
    window: str = "hann",
    nperseg: int = 4096,
    noverlap: int | None = None,
) -> NarrowbandNotchDepthResult:
    """高Qノッチの埋まりをPSDで定量化する。

    Welch PSD からノッチ周辺の狭帯域パワーを測り、周辺帯域との比を深さ[dB]
    として算出する。DUTでノッチが埋まると dut_notch_depth_db が小さくなり、
    notch_fill_db が正の方向に増える。
    """

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
    if notch_center_hz <= 0:
        raise ValueError("notch_center_hz must be positive")
    if notch_q <= 0:
        raise ValueError("notch_q must be positive")
    if nperseg <= 0:
        raise ValueError("nperseg must be positive")
    nyquist = sample_rate / 2
    if notch_center_hz >= nyquist:
        raise ValueError("notch_center_hz must be below Nyquist")

    base_bandwidth = notch_bandwidth_hz or (notch_center_hz / notch_q)
    fft_resolution_hz = sample_rate / max(nperseg, 1)
    notch_bw = float(max(base_bandwidth, fft_resolution_hz))
    ring_bw = float(max(ring_bandwidth_hz or notch_bw, fft_resolution_hz))
    notch_half = notch_bw / 2
    ring_inner = notch_half
    ring_outer = notch_half + ring_bw
    if ring_outer <= ring_inner:
        raise ValueError("ring_bandwidth_hz must be positive")

    noverlap = int(noverlap) if noverlap is not None else nperseg // 2
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must be in [0, nperseg)")

    freqs, ref_psd = sp_signal.welch(
        ref,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )
    _, dut_psd = sp_signal.welch(
        du,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )

    notch_mask = (freqs >= notch_center_hz - notch_half) & (
        freqs <= notch_center_hz + notch_half
    )
    ring_mask = (
        (freqs >= notch_center_hz - ring_outer) & (freqs < notch_center_hz - ring_inner)
    ) | (
        (freqs > notch_center_hz + ring_inner) & (freqs <= notch_center_hz + ring_outer)
    )
    if not np.any(notch_mask):
        raise ValueError("notch window is empty; broaden bandwidth or lower Q")
    if not np.any(ring_mask):
        raise ValueError("ring window is empty; increase ring_bandwidth_hz")

    ref_notch_power = float(np.mean(ref_psd[notch_mask]))
    dut_notch_power = float(np.mean(dut_psd[notch_mask]))
    ref_ring_power = float(np.mean(ref_psd[ring_mask]))
    dut_ring_power = float(np.mean(dut_psd[ring_mask]))

    ref_depth_db = _ratio_to_db(ref_ring_power / max(ref_notch_power, EPS))
    dut_depth_db = _ratio_to_db(dut_ring_power / max(dut_notch_power, EPS))
    notch_fill_db = ref_depth_db - dut_depth_db

    return NarrowbandNotchDepthResult(
        ref_notch_depth_db=ref_depth_db,
        dut_notch_depth_db=dut_depth_db,
        notch_fill_db=notch_fill_db,
        ref_notch_power_db=_power_to_db(ref_notch_power),
        dut_notch_power_db=_power_to_db(dut_notch_power),
        ref_ring_power_db=_power_to_db(ref_ring_power),
        dut_ring_power_db=_power_to_db(dut_ring_power),
        notch_center_hz=float(notch_center_hz),
        notch_bandwidth_hz=notch_bw,
        ring_bandwidth_hz=ring_bw,
    )


def _power_to_db(power: float) -> float:
    return 10.0 * np.log10(max(power, EPS))


def _ratio_to_db(ratio: float) -> float:
    return 10.0 * np.log10(max(ratio, EPS))
