from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
from scipy import signal as sp_signal

from microstructure_metrics.filterbank import GammatoneFilterbank, MelFilterbank

EPS = 1e-12


@dataclass(frozen=True)
class BinauralBandStats:
    """帯域ごとの集約値。"""

    center_freq_hz: float
    median_abs_delta_itd_ms: float
    median_abs_delta_ild_db: float
    median_iacc: float


@dataclass(frozen=True)
class BinauralResult:
    """BCP（Binaural Cue Preservation）計算結果。"""

    median_abs_delta_itd_ms: float
    p95_abs_delta_itd_ms: float
    itd_outlier_rate: float
    median_abs_delta_ild_db: float
    p95_abs_delta_ild_db: float
    iacc_p05: float
    delta_iacc_median: float
    band_stats: tuple[BinauralBandStats, ...]
    frame_times_ms: npt.NDArray[np.float64]
    itd_ref_ms: npt.NDArray[np.float64]
    itd_dut_ms: npt.NDArray[np.float64]
    ild_ref_db: npt.NDArray[np.float64]
    ild_dut_db: npt.NDArray[np.float64]
    iacc_ref: npt.NDArray[np.float64]
    iacc_dut: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    frame_length_ms: float
    frame_hop_ms: float
    max_itd_ms: float
    envelope_threshold_db: float
    itd_outlier_threshold_ms: float
    freq_centers_hz: npt.NDArray[np.float64]
    frames_per_band: int
    used_frames: int


@dataclass(frozen=True)
class BinauralSummary:
    """Binaural集約値。"""

    median_abs_delta_itd_ms: float
    p95_abs_delta_itd_ms: float
    itd_outlier_rate: float
    median_abs_delta_ild_db: float
    p95_abs_delta_ild_db: float
    iacc_p05: float
    delta_iacc_median: float
    band_stats: tuple[BinauralBandStats, ...]


def calculate_binaural_cue_preservation(
    *,
    reference_lr: npt.ArrayLike,
    dut_lr: npt.ArrayLike,
    sample_rate: int,
    audio_freq_range: tuple[float, float] = (125.0, 8000.0),
    num_audio_bands: int = 16,
    frame_length_ms: float = 25.0,
    frame_hop_ms: float = 10.0,
    max_itd_ms: float = 1.0,
    envelope_threshold_db: float = -50.0,
    itd_outlier_threshold_ms: float = 0.2,
    filterbank: Literal["gammatone", "mel"] = "gammatone",
    filterbank_kwargs: Mapping[str, float | int | None] | None = None,
) -> BinauralResult:
    """左右手がかり(ITD/ILD/IACC)の保存度を計算する。

    Args:
        reference_lr: 参照ステレオ信号 (samples, 2)。
        dut_lr: DUTステレオ信号 (samples, 2)。
        sample_rate: サンプルレート(Hz)。
        audio_freq_range: 分析帯域の下限・上限(Hz)。
        num_audio_bands: 聴覚フィルタの本数。
        frame_length_ms: 短時間解析フレーム長(ms)。
        frame_hop_ms: フレームシフト(ms)。
        max_itd_ms: ITD探索の最大ラグ(ms)。
        envelope_threshold_db: フレーム無視の包絡閾値(dBFS相対)。
        itd_outlier_threshold_ms: ITD外れ値判定閾値(ms)。
        filterbank: 聴覚フィルタ種類。
        filterbank_kwargs: フィルタバンク固有パラメータ。
    """
    ref = np.asarray(reference_lr, dtype=np.float64)
    dut = np.asarray(dut_lr, dtype=np.float64)
    if ref.ndim != 2 or dut.ndim != 2:
        raise ValueError("reference_lr/dut_lr must be 2-D (samples, channels).")
    if ref.shape != dut.shape:
        raise ValueError("reference_lr and dut_lr must share the same shape.")
    if ref.shape[1] != 2:
        raise ValueError("Stereo signals with 2 channels are required.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if frame_length_ms <= 0 or frame_hop_ms <= 0:
        raise ValueError("frame_length_ms and frame_hop_ms must be positive.")
    audio_low, audio_high = audio_freq_range
    if audio_low <= 0 or audio_high <= audio_low:
        raise ValueError("audio_freq_range must be an increasing (low, high) tuple.")

    fb_kwargs = dict(filterbank_kwargs or {})
    nyquist = sample_rate / 2
    high_limit = min(audio_high, nyquist * 0.99)
    width_value = fb_kwargs.get("width")
    width = float(width_value) if width_value is not None else 1.0
    order_value = fb_kwargs.get("order")
    bandwidth_value = fb_kwargs.get("bandwidth_scale")
    order = int(order_value) if order_value is not None else 4
    bandwidth_scale = float(bandwidth_value) if bandwidth_value is not None else 1.0
    if filterbank == "gammatone":
        fb: GammatoneFilterbank | MelFilterbank = GammatoneFilterbank(
            sample_rate=sample_rate,
            num_filters=num_audio_bands,
            low_freq=audio_low,
            high_freq=high_limit,
            width=width,
        )
    else:
        fb = MelFilterbank(
            sample_rate=sample_rate,
            num_filters=num_audio_bands,
            low_freq=audio_low,
            high_freq=high_limit,
            order=order,
            bandwidth_scale=bandwidth_scale,
        )

    ref_left = fb.analyze(ref[:, 0])
    ref_right = fb.analyze(ref[:, 1])
    dut_left = fb.analyze(dut[:, 0])
    dut_right = fb.analyze(dut[:, 1])

    frame_length = int(round(frame_length_ms * sample_rate / 1000))
    frame_hop = int(round(frame_hop_ms * sample_rate / 1000))
    max_lag_samples = int(round(max_itd_ms * sample_rate / 1000))
    if frame_length < 2:
        raise ValueError("frame_length_ms is too small for the given sample_rate.")
    if frame_hop < 1:
        frame_hop = 1
    frame_starts = list(range(0, ref.shape[0] - frame_length + 1, frame_hop))
    frames_per_band = len(frame_starts)
    frame_times_ms = np.asarray(
        [(start + frame_length / 2) * 1000 / sample_rate for start in frame_starts],
        dtype=np.float64,
    )

    shape = (num_audio_bands, frames_per_band)
    itd_ref_ms = np.full(shape, np.nan, dtype=np.float64)
    itd_dut_ms = np.full(shape, np.nan, dtype=np.float64)
    ild_ref_db = np.full(shape, np.nan, dtype=np.float64)
    ild_dut_db = np.full(shape, np.nan, dtype=np.float64)
    iacc_ref = np.full(shape, np.nan, dtype=np.float64)
    iacc_dut = np.full(shape, np.nan, dtype=np.float64)
    weights = np.zeros(shape, dtype=np.float64)

    global_peak = float(
        max(np.max(np.abs(ref), initial=0.0), np.max(np.abs(dut), initial=0.0), EPS)
    )
    envelope_threshold = global_peak * 10 ** (envelope_threshold_db / 20.0)

    for band_idx in range(num_audio_bands):
        ref_l_band = ref_left[band_idx]
        ref_r_band = ref_right[band_idx]
        dut_l_band = dut_left[band_idx]
        dut_r_band = dut_right[band_idx]
        for frame_idx, start in enumerate(frame_starts):
            stop = start + frame_length
            ref_l_frame = ref_l_band[start:stop]
            ref_r_frame = ref_r_band[start:stop]
            dut_l_frame = dut_l_band[start:stop]
            dut_r_frame = dut_r_band[start:stop]

            rms_val = _joint_rms(ref_l_frame, ref_r_frame, dut_l_frame, dut_r_frame)
            if rms_val <= envelope_threshold:
                continue

            ild_ref_db[band_idx, frame_idx] = _ild_db(ref_l_frame, ref_r_frame)
            ild_dut_db[band_idx, frame_idx] = _ild_db(dut_l_frame, dut_r_frame)

            itd_ref, iacc_ref_val = _lag_and_iacc(
                ref_l_frame, ref_r_frame, max_lag_samples, sample_rate
            )
            itd_dut_val, iacc_dut_val = _lag_and_iacc(
                dut_l_frame, dut_r_frame, max_lag_samples, sample_rate
            )
            itd_ref_ms[band_idx, frame_idx] = itd_ref
            itd_dut_ms[band_idx, frame_idx] = itd_dut_val
            iacc_ref[band_idx, frame_idx] = iacc_ref_val
            iacc_dut[band_idx, frame_idx] = iacc_dut_val
            weights[band_idx, frame_idx] = rms_val

    summary = _summarize(
        itd_ref_ms=itd_ref_ms,
        itd_dut_ms=itd_dut_ms,
        ild_ref_db=ild_ref_db,
        ild_dut_db=ild_dut_db,
        iacc_ref=iacc_ref,
        iacc_dut=iacc_dut,
        weights=weights,
        freq_centers=np.asarray(fb.center_frequencies, dtype=np.float64),
        itd_outlier_threshold_ms=itd_outlier_threshold_ms,
    )

    return BinauralResult(
        median_abs_delta_itd_ms=summary.median_abs_delta_itd_ms,
        p95_abs_delta_itd_ms=summary.p95_abs_delta_itd_ms,
        itd_outlier_rate=summary.itd_outlier_rate,
        median_abs_delta_ild_db=summary.median_abs_delta_ild_db,
        p95_abs_delta_ild_db=summary.p95_abs_delta_ild_db,
        iacc_p05=summary.iacc_p05,
        delta_iacc_median=summary.delta_iacc_median,
        band_stats=summary.band_stats,
        frame_times_ms=frame_times_ms,
        itd_ref_ms=itd_ref_ms,
        itd_dut_ms=itd_dut_ms,
        ild_ref_db=ild_ref_db,
        ild_dut_db=ild_dut_db,
        iacc_ref=iacc_ref,
        iacc_dut=iacc_dut,
        weights=weights,
        frame_length_ms=float(frame_length_ms),
        frame_hop_ms=float(frame_hop_ms),
        max_itd_ms=float(max_itd_ms),
        envelope_threshold_db=float(envelope_threshold_db),
        itd_outlier_threshold_ms=float(itd_outlier_threshold_ms),
        freq_centers_hz=np.asarray(fb.center_frequencies, dtype=np.float64),
        frames_per_band=int(frames_per_band),
        used_frames=int(np.count_nonzero(weights > 0)),
    )


def _lag_and_iacc(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
    max_lag_samples: int,
    sample_rate: int,
) -> tuple[float, float]:
    if left.shape != right.shape:
        raise ValueError("Left/right frames must share shape.")

    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0:
        return 0.0, 0.0
    corr = sp_signal.correlate(left, right, mode="full", method="fft") / denom
    lags = sp_signal.correlation_lags(left.size, right.size, mode="full")
    if max_lag_samples >= 0:
        window = np.abs(lags) <= max_lag_samples
        if not np.any(window):
            return 0.0, 0.0
        corr = corr[window]
        lags = lags[window]
    idx = int(np.argmax(np.abs(corr)))
    lag_samples = int(lags[idx])
    itd_ms = (lag_samples / sample_rate) * 1000.0
    return float(itd_ms), float(np.abs(np.clip(corr[idx], -1.0, 1.0)))


def _ild_db(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> float:
    rms_left = float(np.sqrt(np.mean(np.square(left), dtype=np.float64)))
    rms_right = float(np.sqrt(np.mean(np.square(right), dtype=np.float64)))
    if rms_right <= 0 and rms_left <= 0:
        return 0.0
    ratio = max(rms_left, EPS) / max(rms_right, EPS)
    return float(20.0 * np.log10(ratio))


def _joint_rms(*frames: npt.NDArray[np.float64]) -> float:
    components = [f.ravel() for f in frames if f.size]
    if not components:
        return 0.0
    stacked = np.hstack(components)
    return float(np.sqrt(np.mean(np.square(stacked), dtype=np.float64)))


def _weighted_percentile(
    values: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    percentile: float,
) -> float:
    if values.size == 0 or weights.size == 0:
        return 0.0
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return 0.0
    vals = cast(npt.NDArray[np.float64], np.asarray(values[valid], dtype=np.float64))
    w = cast(npt.NDArray[np.float64], np.asarray(weights[valid], dtype=np.float64))
    order = np.argsort(vals)
    vals_sorted = vals[order]
    w_sorted = w[order]
    cumulative = np.cumsum(w_sorted)
    cutoff = percentile * cumulative[-1]
    idx = int(np.searchsorted(cumulative, cutoff))
    idx = min(idx, vals_sorted.size - 1)
    value_array = np.asarray(vals_sorted[idx], dtype=np.float64)
    value = float(value_array.item())
    return value


def _summarize(
    *,
    itd_ref_ms: npt.NDArray[np.float64],
    itd_dut_ms: npt.NDArray[np.float64],
    ild_ref_db: npt.NDArray[np.float64],
    ild_dut_db: npt.NDArray[np.float64],
    iacc_ref: npt.NDArray[np.float64],
    iacc_dut: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    freq_centers: npt.NDArray[np.float64],
    itd_outlier_threshold_ms: float,
) -> BinauralSummary:
    delta_itd = itd_dut_ms - itd_ref_ms
    delta_ild = ild_dut_db - ild_ref_db
    delta_iacc = iacc_dut - iacc_ref

    abs_delta_itd = np.abs(delta_itd)
    abs_delta_ild = np.abs(delta_ild)
    valid_weights = np.where(np.isfinite(weights), weights, 0.0)

    median_abs_delta_itd_ms = _weighted_percentile(abs_delta_itd, valid_weights, 0.5)
    p95_abs_delta_itd_ms = _weighted_percentile(abs_delta_itd, valid_weights, 0.95)
    median_abs_delta_ild_db = _weighted_percentile(abs_delta_ild, valid_weights, 0.5)
    p95_abs_delta_ild_db = _weighted_percentile(abs_delta_ild, valid_weights, 0.95)
    delta_iacc_median = _weighted_percentile(delta_iacc, valid_weights, 0.5)
    iacc_p05 = _weighted_percentile(iacc_dut, valid_weights, 0.05)

    outlier_mask = abs_delta_itd > itd_outlier_threshold_ms
    weighted_outliers = float(np.sum(valid_weights[outlier_mask]))
    weight_total = float(np.sum(valid_weights))
    itd_outlier_rate = weighted_outliers / weight_total if weight_total > 0 else 0.0

    band_stats: list[BinauralBandStats] = []
    for idx, center in enumerate(freq_centers):
        band_weights = valid_weights[idx]
        band_median_itd = _weighted_percentile(abs_delta_itd[idx], band_weights, 0.5)
        band_median_ild = _weighted_percentile(abs_delta_ild[idx], band_weights, 0.5)
        band_median_iacc = _weighted_percentile(iacc_dut[idx], band_weights, 0.5)
        band_stats.append(
            BinauralBandStats(
                center_freq_hz=float(center),
                median_abs_delta_itd_ms=band_median_itd,
                median_abs_delta_ild_db=band_median_ild,
                median_iacc=band_median_iacc,
            )
        )

    return BinauralSummary(
        median_abs_delta_itd_ms=float(median_abs_delta_itd_ms),
        p95_abs_delta_itd_ms=float(p95_abs_delta_itd_ms),
        itd_outlier_rate=float(itd_outlier_rate),
        median_abs_delta_ild_db=float(median_abs_delta_ild_db),
        p95_abs_delta_ild_db=float(p95_abs_delta_ild_db),
        iacc_p05=float(iacc_p05),
        delta_iacc_median=float(delta_iacc_median),
        band_stats=tuple(band_stats),
    )
