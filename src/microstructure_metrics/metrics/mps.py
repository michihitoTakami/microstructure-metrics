from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from scipy import signal as sp_signal

from microstructure_metrics.filterbank import GammatoneFilterbank, MelFilterbank

EPS: Final = 1e-12


@dataclass(frozen=True)
class MPSResult:
    """変調パワースペクトラム(MPS)計算結果。"""

    mps_matrix: npt.NDArray[np.float64]
    audio_freqs: npt.NDArray[np.float64]
    mod_freqs: npt.NDArray[np.float64]


@dataclass(frozen=True)
class MPSSimilarityResult:
    """入出力MPSの類似度計算結果。"""

    mps_correlation: float
    mps_distance: float
    band_correlations: dict[float, float]
    ref_mps: npt.NDArray[np.float64]
    dut_mps: npt.NDArray[np.float64]
    audio_freqs: npt.NDArray[np.float64]
    mod_freqs: npt.NDArray[np.float64]


def calculate_mps(
    *,
    signal: npt.ArrayLike,
    sample_rate: int,
    audio_freq_range: tuple[float, float] = (100.0, 8000.0),
    mod_freq_range: tuple[float, float] = (0.5, 64.0),
    num_audio_bands: int = 48,
    envelope_lowpass_hz: float | None = 64.0,
    modulation_fft_size: int | None = None,
    remove_dc: bool = True,
    filterbank: Literal["gammatone", "mel"] = "gammatone",
    filterbank_kwargs: Mapping[str, float | int | None] | None = None,
    envelope_method: Literal["hilbert", "rectify"] = "hilbert",
    envelope_lowpass_order: int = 4,
    mod_scale: Literal["linear", "log"] = "linear",
    num_mod_bins: int | None = None,
    mps_scale: Literal["power", "log"] = "power",
) -> MPSResult:
    """変調パワースペクトラム(MPS)を計算する。

    Args:
        signal: モノラル信号。
        sample_rate: サンプルレート(Hz)。
        audio_freq_range: 聴覚帯域の下限・上限(Hz)。
        mod_freq_range: 変調周波数の下限・上限(Hz)。
        num_audio_bands: 聴覚フィルタの本数（32〜64を推奨）。
        envelope_lowpass_hz: ヒルベルト包絡に適用するLPFのカットオフ(Hz)。
            Noneで無効。
        modulation_fft_size: 変調スペクトルFFTサイズ。Noneで信号長から自動決定。
        remove_dc: 包絡の平均値を除去してDC漏れを抑えるか。
        filterbank: 聴覚フィルタの種類（gammatone/mel）。
        filterbank_kwargs: フィルタバンク固有パラメータ（例: width, order）。
        envelope_method: 包絡検出手法（hilbert or rectify）。
        envelope_lowpass_order: 包絡LPFの次数。
        mod_scale: 変調周波数軸のスケール（linear/log）。
        num_mod_bins: mod_scale=log時のリサンプル先bin数(Noneなら元bin数)。
        mps_scale: パワースペクトラムのスケール（power/log）。

    Returns:
        MPSResult
    """
    data = np.asarray(signal, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("signal must be a 1-D array")
    if data.size == 0:
        raise ValueError("signal must not be empty")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    audio_low, audio_high = audio_freq_range
    if audio_low <= 0 or audio_high <= audio_low:
        raise ValueError("audio_freq_range must be an increasing (low, high) tuple")

    mod_low, mod_high = mod_freq_range
    if mod_low <= 0 or mod_high <= mod_low:
        raise ValueError("mod_freq_range must be an increasing (low, high) tuple")
    nyquist = sample_rate / 2
    if mod_high >= nyquist:
        raise ValueError("mod_freq_range high must be below Nyquist")
    if num_mod_bins is not None and num_mod_bins < 2:
        raise ValueError("num_mod_bins must be >= 2 when provided")
    if envelope_method not in {"hilbert", "rectify"}:
        raise ValueError("envelope_method must be 'hilbert' or 'rectify'")
    if envelope_lowpass_order < 1:
        raise ValueError("envelope_lowpass_order must be >= 1")
    if mod_scale not in {"linear", "log"}:
        raise ValueError("mod_scale must be 'linear' or 'log'")
    if mps_scale not in {"power", "log"}:
        raise ValueError("mps_scale must be 'power' or 'log'")

    high_limit = min(audio_high, nyquist * 0.99)
    fb_kwargs = dict(filterbank_kwargs or {})
    if filterbank == "gammatone":
        fb = GammatoneFilterbank(
            sample_rate=sample_rate,
            num_filters=num_audio_bands,
            low_freq=audio_low,
            high_freq=high_limit,
            **_select_kwargs(fb_kwargs, {"width"}),
        )
    else:
        fb = MelFilterbank(
            sample_rate=sample_rate,
            num_filters=num_audio_bands,
            low_freq=audio_low,
            high_freq=high_limit,
            **_select_kwargs(fb_kwargs, {"order", "bandwidth_scale"}),
        )
    band_signals = fb.analyze(data)

    envelopes = _extract_envelopes(
        band_signals=band_signals,
        method=envelope_method,
        sample_rate=sample_rate,
        lowpass_hz=envelope_lowpass_hz,
        lowpass_order=envelope_lowpass_order,
        remove_dc=remove_dc,
    )

    n_fft = modulation_fft_size or _next_pow_two(envelopes.shape[1])
    mod_spectrum = np.fft.rfft(envelopes, n=n_fft, axis=1)
    mod_freqs: npt.NDArray[np.float64] = np.asarray(
        np.fft.rfftfreq(n_fft, 1.0 / sample_rate), dtype=np.float64
    )

    mod_mask = (mod_freqs >= mod_low) & (mod_freqs <= mod_high)
    if not np.any(mod_mask):
        raise ValueError("mod_freq_range removes all modulation bins")

    mod_axis = mod_freqs[mod_mask]
    mps_matrix = np.abs(mod_spectrum[:, mod_mask]) ** 2
    if mod_scale == "log":
        target_bins = num_mod_bins or mod_axis.size
        log_freqs = np.geomspace(mod_low, mod_high, num=target_bins)
        mps_matrix = _interpolate_mod_axis(
            mps_matrix, source_freqs=mod_axis, target_freqs=log_freqs
        )
        mod_axis = log_freqs

    if mps_scale == "log":
        mps_matrix = _power_to_db(mps_matrix)

    return MPSResult(
        mps_matrix=np.asarray(mps_matrix, dtype=np.float64),
        audio_freqs=np.asarray(fb.center_frequencies, dtype=np.float64),
        mod_freqs=np.asarray(mod_axis, dtype=np.float64),
    )


def calculate_mps_similarity(
    *,
    reference: npt.ArrayLike,
    dut: npt.ArrayLike,
    sample_rate: int,
    audio_freq_range: tuple[float, float] = (100.0, 8000.0),
    mod_freq_range: tuple[float, float] = (0.5, 64.0),
    num_audio_bands: int = 48,
    envelope_lowpass_hz: float | None = 64.0,
    modulation_fft_size: int | None = None,
    filterbank: Literal["gammatone", "mel"] = "gammatone",
    filterbank_kwargs: Mapping[str, float | int | None] | None = None,
    envelope_method: Literal["hilbert", "rectify"] = "hilbert",
    envelope_lowpass_order: int = 4,
    mod_scale: Literal["linear", "log"] = "linear",
    num_mod_bins: int | None = None,
    mps_scale: Literal["power", "log"] = "power",
    mps_norm: Literal["global", "per_band", "none"] = "global",
    band_weights: npt.ArrayLike | None = None,
    band_weighting: Literal["none", "energy"] = "none",
) -> MPSSimilarityResult:
    """入出力のMPS類似度（相関/距離）を計算する。"""

    ref = np.asarray(reference, dtype=np.float64)
    du = np.asarray(dut, dtype=np.float64)
    if ref.ndim != 1 or du.ndim != 1:
        raise ValueError("reference/dut must be 1-D signals")
    if ref.size == 0 or du.size == 0:
        raise ValueError("reference/dut must contain samples")
    if ref.shape[0] != du.shape[0]:
        raise ValueError("reference/dut length mismatch; align signals first")

    ref_result = calculate_mps(
        signal=ref,
        sample_rate=sample_rate,
        audio_freq_range=audio_freq_range,
        mod_freq_range=mod_freq_range,
        num_audio_bands=num_audio_bands,
        envelope_lowpass_hz=envelope_lowpass_hz,
        modulation_fft_size=modulation_fft_size,
        filterbank=filterbank,
        filterbank_kwargs=filterbank_kwargs,
        envelope_method=envelope_method,
        envelope_lowpass_order=envelope_lowpass_order,
        mod_scale=mod_scale,
        num_mod_bins=num_mod_bins,
        mps_scale=mps_scale,
    )
    dut_result = calculate_mps(
        signal=du,
        sample_rate=sample_rate,
        audio_freq_range=audio_freq_range,
        mod_freq_range=mod_freq_range,
        num_audio_bands=num_audio_bands,
        envelope_lowpass_hz=envelope_lowpass_hz,
        modulation_fft_size=modulation_fft_size,
        filterbank=filterbank,
        filterbank_kwargs=filterbank_kwargs,
        envelope_method=envelope_method,
        envelope_lowpass_order=envelope_lowpass_order,
        mod_scale=mod_scale,
        num_mod_bins=num_mod_bins,
        mps_scale=mps_scale,
    )

    weights = _resolve_band_weights(
        weighting=band_weighting,
        explicit_weights=band_weights,
        reference_mps=ref_result.mps_matrix,
    )
    ref_weighted = _apply_band_weights(ref_result.mps_matrix, weights)
    dut_weighted = _apply_band_weights(dut_result.mps_matrix, weights)

    ref_norm = _normalize_mps(ref_weighted, mode=mps_norm)
    dut_norm = _normalize_mps(dut_weighted, mode=mps_norm)
    ref_band_norm = _normalize_rows(ref_weighted)
    dut_band_norm = _normalize_rows(dut_weighted)

    mps_correlation = _pearson(ref_norm.ravel(), dut_norm.ravel())
    mps_distance = float(np.sqrt(np.mean(np.square(ref_norm - dut_norm))))

    band_correlations: dict[float, float] = {}
    for idx, freq in enumerate(ref_result.audio_freqs):
        band_correlations[freq] = _pearson(ref_band_norm[idx], dut_band_norm[idx])

    return MPSSimilarityResult(
        mps_correlation=mps_correlation,
        mps_distance=mps_distance,
        band_correlations=band_correlations,
        ref_mps=ref_result.mps_matrix,
        dut_mps=dut_result.mps_matrix,
        audio_freqs=ref_result.audio_freqs,
        mod_freqs=ref_result.mod_freqs,
    )


def _extract_envelopes(
    *,
    band_signals: npt.NDArray[np.float64],
    method: Literal["hilbert", "rectify"],
    sample_rate: int,
    lowpass_hz: float | None,
    lowpass_order: int,
    remove_dc: bool,
) -> npt.NDArray[np.float64]:
    if method == "hilbert":
        envelopes = np.abs(sp_signal.hilbert(band_signals, axis=1))
    else:
        envelopes = np.abs(band_signals)

    envelopes = _apply_lowpass(
        envelopes,
        cutoff_hz=lowpass_hz,
        sample_rate=sample_rate,
        order=lowpass_order,
    )

    if remove_dc:
        envelopes = envelopes - np.mean(envelopes, axis=1, keepdims=True)
    return envelopes


def _apply_lowpass(
    data: npt.NDArray[np.float64],
    *,
    cutoff_hz: float | None,
    sample_rate: int,
    order: int,
) -> npt.NDArray[np.float64]:
    if cutoff_hz is None:
        return data
    nyquist = sample_rate / 2
    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        raise ValueError("envelope_lowpass_hz must be in (0, Nyquist)")
    sos = sp_signal.butter(order, cutoff_hz, btype="low", fs=sample_rate, output="sos")
    return sp_signal.sosfiltfilt(sos, data, axis=1)


def _interpolate_mod_axis(
    matrix: npt.NDArray[np.float64],
    *,
    source_freqs: npt.NDArray[np.float64],
    target_freqs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if source_freqs.ndim != 1 or target_freqs.ndim != 1:
        raise ValueError("frequency axes must be 1-D")
    if source_freqs.size < 2:
        return matrix
    interpolated = np.zeros((matrix.shape[0], target_freqs.size), dtype=np.float64)
    for idx, row in enumerate(matrix):
        interpolated[idx] = np.interp(target_freqs, source_freqs, row)
    return interpolated


def _power_to_db(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 10.0 * np.log10(np.maximum(matrix, EPS))


def _resolve_band_weights(
    *,
    weighting: Literal["none", "energy"],
    explicit_weights: npt.ArrayLike | None,
    reference_mps: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if weighting not in {"none", "energy"}:
        raise ValueError("band_weighting must be 'none' or 'energy'")

    if explicit_weights is not None:
        weights = np.asarray(explicit_weights, dtype=np.float64).ravel()
    elif weighting == "energy":
        energy = np.sum(np.maximum(reference_mps, 0.0), axis=1)
        max_energy = float(np.max(energy)) if energy.size else 0.0
        weights = energy / max(max_energy, EPS)
    else:
        weights = np.ones(reference_mps.shape[0], dtype=np.float64)

    if weights.size != reference_mps.shape[0]:
        raise ValueError("band_weights length must match number of bands")
    return np.asarray(weights, dtype=np.float64)


def _apply_band_weights(
    matrix: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.asarray(matrix * weights[:, None], dtype=np.float64)


def _normalize_mps(
    matrix: npt.NDArray[np.float64], *, mode: Literal["global", "per_band", "none"]
) -> npt.NDArray[np.float64]:
    if mode == "none":
        return np.asarray(matrix, dtype=np.float64)
    if mode == "global":
        return _normalize_matrix(matrix)
    if mode == "per_band":
        return _normalize_rows(matrix)
    raise ValueError("mps_norm must be 'global', 'per_band', or 'none'")


def _next_pow_two(n: int) -> int:
    return 1 << (max(n - 1, 0).bit_length())


def _normalize_matrix(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    norm = float(np.linalg.norm(matrix))
    if norm <= 0:
        return np.zeros_like(matrix)
    return np.asarray(matrix, dtype=np.float64) / norm


def _normalize_rows(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    normalized = np.zeros_like(matrix, dtype=np.float64)
    for idx, row in enumerate(matrix):
        norm = float(np.linalg.norm(row))
        if norm > 0:
            normalized[idx] = row / norm
    return normalized


def _pearson(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    if a.shape != b.shape:
        raise ValueError("arrays must have the same shape for correlation")
    a_mean = float(np.mean(a))
    b_mean = float(np.mean(b))
    a_dev = a - a_mean
    b_dev = b - b_mean
    denom = float(np.sqrt(np.sum(a_dev**2) * np.sum(b_dev**2)))
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    return float(np.sum(a_dev * b_dev) / denom)


def _select_kwargs(
    source: Mapping[str, float | int | None], allowed: set[str]
) -> dict[str, float | int | None]:
    return {key: value for key, value in source.items() if key in allowed}
