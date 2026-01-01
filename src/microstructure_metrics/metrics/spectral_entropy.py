from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal as sp_signal

EPS = 1e-12


@dataclass(frozen=True)
class SpectralEntropyResult:
    """スペクトルエントロピー計算結果。"""

    mean_entropy: float
    entropy_over_time: npt.NDArray[np.float64]
    frame_times: npt.NDArray[np.float64]
    freqs: npt.NDArray[np.float64]
    normalized: bool


@dataclass(frozen=True)
class DeltaSEResult:
    """入出力間のスペクトルエントロピー差分。"""

    delta_se_mean: float
    delta_se_std: float
    delta_se_max: float
    ref_se_mean: float
    dut_se_mean: float
    delta_se_over_time: npt.NDArray[np.float64]
    frame_times: npt.NDArray[np.float64]


def calculate_spectral_entropy(
    *,
    signal: npt.ArrayLike,
    sample_rate: int,
    frame_size: int = 2048,
    hop_size: int = 512,
    freq_range: tuple[float, float] | None = (20.0, 20000.0),
    normalize: bool = True,
    noise_floor_db: float | None = None,
) -> SpectralEntropyResult:
    """スペクトルエントロピーをフレームごとに計算する。

    Args:
        signal: モノラル信号。
        sample_rate: サンプルレート(Hz)。
        frame_size: STFTウィンドウサイズ。
        hop_size: フレームシフト（サンプル）。
        freq_range: 利用する周波数範囲 (low, high)。Noneで全帯域。
        normalize: log2(N_bins) で正規化して 0-1 に収めるか。
        noise_floor_db: フレームピークからの相対ノイズフロア閾値(dB)。
            閾値より低いビンを除外する。Noneで無効。

    Returns:
        SpectralEntropyResult
    """
    data = np.asarray(signal, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("signal must be a 1-D array")
    if data.size == 0:
        raise ValueError("signal must contain samples")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")
    if hop_size >= frame_size:
        raise ValueError("hop_size must be smaller than frame_size")
    if data.size < frame_size:
        raise ValueError("signal length must be at least frame_size samples")

    low = None
    high = None
    if freq_range is not None:
        low, high = freq_range
        if low < 0 or high <= low:
            raise ValueError("freq_range must be an increasing (low, high) tuple")
        nyquist = sample_rate / 2
        if high > nyquist:
            raise ValueError("freq_range high must be below Nyquist")

    if noise_floor_db is not None and noise_floor_db < 0:
        raise ValueError("noise_floor_db must be non-negative when specified")

    freqs, times, stft_matrix = sp_signal.stft(
        data,
        fs=sample_rate,
        window="hann",
        nperseg=frame_size,
        noverlap=frame_size - hop_size,
        boundary=None,
        padded=False,
    )

    band_mask = np.ones_like(freqs, dtype=bool)
    if low is not None and high is not None:
        band_mask = (freqs >= low) & (freqs <= high)
    if not np.any(band_mask):
        raise ValueError("freq_range removes all frequency bins")

    band_freqs = freqs[band_mask]
    spectrum = stft_matrix[band_mask, :]
    power = np.abs(spectrum) ** 2

    if noise_floor_db is not None:
        peak_power = np.max(power, axis=0, keepdims=True)
        peak_power = np.maximum(peak_power, EPS)
        threshold = peak_power * 10.0 ** (-noise_floor_db / 10.0)
        power = np.where(power >= threshold, power, 0.0)

    power_sum = np.sum(power, axis=0)
    power_sum = np.where(power_sum > 0.0, power_sum, EPS)

    prob = power / power_sum
    prob = np.maximum(prob, EPS)
    entropy = -np.sum(prob * np.log2(prob), axis=0)

    if normalize:
        max_entropy = math.log2(power.shape[0])
        entropy = entropy / max(max_entropy, EPS)
        entropy = np.clip(entropy, 0.0, 1.0)

    mean_entropy = float(np.mean(entropy))
    return SpectralEntropyResult(
        mean_entropy=mean_entropy,
        entropy_over_time=np.asarray(entropy, dtype=np.float64),
        frame_times=np.asarray(times, dtype=np.float64),
        freqs=np.asarray(band_freqs, dtype=np.float64),
        normalized=normalize,
    )


def calculate_delta_se(
    *,
    reference: npt.ArrayLike,
    dut: npt.ArrayLike,
    sample_rate: int,
    frame_size: int = 2048,
    hop_size: int = 512,
    freq_range: tuple[float, float] | None = (20.0, 20000.0),
    normalize: bool = True,
    noise_floor_db: float | None = None,
) -> DeltaSEResult:
    """入出力のスペクトルエントロピー差分(ΔSE)を算出する。"""
    ref_result = calculate_spectral_entropy(
        signal=reference,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
        freq_range=freq_range,
        normalize=normalize,
        noise_floor_db=noise_floor_db,
    )
    dut_result = calculate_spectral_entropy(
        signal=dut,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
        freq_range=freq_range,
        normalize=normalize,
        noise_floor_db=noise_floor_db,
    )

    ref_entropy = ref_result.entropy_over_time
    dut_entropy = dut_result.entropy_over_time
    if ref_entropy.shape != dut_entropy.shape:
        raise ValueError("reference/dut entropy lengths mismatch; align signals first")

    delta = dut_entropy - ref_entropy
    return DeltaSEResult(
        delta_se_mean=float(np.mean(delta)),
        delta_se_std=float(np.std(delta)),
        delta_se_max=float(np.max(delta)),
        ref_se_mean=ref_result.mean_entropy,
        dut_se_mean=dut_result.mean_entropy,
        delta_se_over_time=np.asarray(delta, dtype=np.float64),
        frame_times=ref_result.frame_times,
    )
