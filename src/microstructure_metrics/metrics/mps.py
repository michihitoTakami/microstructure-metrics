from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal as sp_signal

from microstructure_metrics.filterbank import GammatoneFilterbank

EPS = 1e-12


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

    fb = GammatoneFilterbank(
        sample_rate=sample_rate,
        num_filters=num_audio_bands,
        low_freq=audio_low,
        high_freq=min(audio_high, nyquist * 0.99),
    )
    band_signals = fb.analyze(data)

    envelopes = np.abs(sp_signal.hilbert(band_signals, axis=1))
    if envelope_lowpass_hz is not None:
        if envelope_lowpass_hz <= 0 or envelope_lowpass_hz >= nyquist:
            raise ValueError("envelope_lowpass_hz must be in (0, Nyquist)")
        sos = sp_signal.butter(
            4,
            envelope_lowpass_hz,
            btype="low",
            fs=sample_rate,
            output="sos",
        )
        envelopes = sp_signal.sosfiltfilt(sos, envelopes, axis=1)

    if remove_dc:
        envelopes = envelopes - np.mean(envelopes, axis=1, keepdims=True)

    n_fft = modulation_fft_size or _next_pow_two(envelopes.shape[1])
    mod_spectrum = np.fft.rfft(envelopes, n=n_fft, axis=1)
    mod_freqs: npt.NDArray[np.float64] = np.asarray(
        np.fft.rfftfreq(n_fft, 1.0 / sample_rate), dtype=np.float64
    )

    mod_mask = (mod_freqs >= mod_low) & (mod_freqs <= mod_high)
    if not np.any(mod_mask):
        raise ValueError("mod_freq_range removes all modulation bins")

    mps_matrix = np.abs(mod_spectrum[:, mod_mask]) ** 2
    return MPSResult(
        mps_matrix=np.asarray(mps_matrix, dtype=np.float64),
        audio_freqs=np.asarray(fb.center_frequencies, dtype=np.float64),
        mod_freqs=np.asarray(mod_freqs[mod_mask], dtype=np.float64),
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
    )
    dut_result = calculate_mps(
        signal=du,
        sample_rate=sample_rate,
        audio_freq_range=audio_freq_range,
        mod_freq_range=mod_freq_range,
        num_audio_bands=num_audio_bands,
        envelope_lowpass_hz=envelope_lowpass_hz,
        modulation_fft_size=modulation_fft_size,
    )

    ref_norm = _normalize_matrix(ref_result.mps_matrix)
    dut_norm = _normalize_matrix(dut_result.mps_matrix)
    ref_band_norm = _normalize_rows(ref_result.mps_matrix)
    dut_band_norm = _normalize_rows(dut_result.mps_matrix)

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
