from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import signal

NDArray = npt.NDArray[np.float64]


def remove_dc_offset(
    data: NDArray,
    *,
    method: Literal["mean", "highpass"] = "mean",
    sample_rate: int = 48000,
    cutoff_hz: float = 10.0,
) -> NDArray:
    """Remove DC offset via mean subtraction or a simple high-pass filter.

    対応次元: 1D (samples) または 2D (samples, channels)。
    """
    if data.size == 0:
        return data
    if data.ndim not in {1, 2}:
        raise ValueError("data must be 1-D or 2-D")

    arr = np.asarray(data, dtype=np.float64)
    if method == "mean":
        mean = np.mean(arr, axis=0, keepdims=True)
        if np.all(np.abs(mean) < 1e-7):
            return arr
        return np.asarray(arr - mean, dtype=np.float64)
    if method != "highpass":
        raise ValueError("method must be 'mean' or 'highpass'")

    nyquist = sample_rate / 2
    norm_cutoff = max(cutoff_hz / nyquist, 1e-6)
    sos = signal.butter(2, norm_cutoff, btype="highpass", output="sos")
    return np.asarray(signal.sosfilt(sos, arr, axis=0), dtype=np.float64)


def normalize_audio(
    data: NDArray,
    *,
    mode: Literal["peak", "rms"] = "peak",
    target: float = 0.99,
) -> NDArray:
    """Normalize signal by peak or RMS to the target amplitude.

    対応次元: 1D (samples) または 2D (samples, channels)。複数chの場合は ch 毎に
    独立にスケールする。
    """
    if data.size == 0:
        return data
    if data.ndim not in {1, 2}:
        raise ValueError("data must be 1-D or 2-D")
    if mode not in {"peak", "rms"}:
        raise ValueError("mode must be 'peak' or 'rms'")
    if target <= 0:
        raise ValueError("target must be positive")

    arr = np.asarray(data, dtype=np.float64)
    if mode == "peak":
        reference = np.max(np.abs(arr), axis=0, keepdims=True)
    else:
        reference = np.sqrt(
            np.mean(np.square(arr), axis=0, dtype=np.float64, keepdims=True)
        )

    reference = np.where(reference == 0, 1.0, reference)
    scaled = arr * (target / reference)
    return np.asarray(scaled, dtype=np.float64)


def resample_audio(data: NDArray, *, orig_sr: int, target_sr: int) -> NDArray:
    """Resample audio using polyphase filtering (samples axis=0)."""
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive")
    if data.ndim not in {1, 2}:
        raise ValueError("data must be 1-D or 2-D")
    if orig_sr == target_sr:
        return data
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return np.asarray(signal.resample_poly(data, up, down, axis=0), dtype=np.float64)
