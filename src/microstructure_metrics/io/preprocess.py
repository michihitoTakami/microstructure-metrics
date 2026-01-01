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
    """Remove DC offset via mean subtraction or a simple high-pass filter."""
    if data.size == 0:
        return data
    if method == "mean":
        mean = float(np.mean(data))
        if abs(mean) < 1e-7:
            return np.asarray(data, dtype=np.float64)
        return np.asarray(data - mean, dtype=np.float64)
    if method != "highpass":
        raise ValueError("method must be 'mean' or 'highpass'")

    nyquist = sample_rate / 2
    norm_cutoff = max(cutoff_hz / nyquist, 1e-6)
    sos = signal.butter(2, norm_cutoff, btype="highpass", output="sos")
    return np.asarray(signal.sosfilt(sos, data), dtype=np.float64)


def normalize_audio(
    data: NDArray,
    *,
    mode: Literal["peak", "rms"] = "peak",
    target: float = 0.99,
) -> NDArray:
    """Normalize signal by peak or RMS to the target amplitude."""
    if data.size == 0:
        return data
    if mode not in {"peak", "rms"}:
        raise ValueError("mode must be 'peak' or 'rms'")
    if target <= 0:
        raise ValueError("target must be positive")

    if mode == "peak":
        reference = float(np.max(np.abs(data)))
    else:
        reference = float(np.sqrt(np.mean(np.square(data), dtype=np.float64)))

    if reference == 0:
        return data
    scaled = data * (target / reference)
    return np.asarray(scaled, dtype=np.float64)


def resample_audio(data: NDArray, *, orig_sr: int, target_sr: int) -> NDArray:
    """Resample 1D audio using polyphase filtering."""
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive")
    if orig_sr == target_sr:
        return data
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return np.asarray(signal.resample_poly(data, up, down), dtype=np.float64)
