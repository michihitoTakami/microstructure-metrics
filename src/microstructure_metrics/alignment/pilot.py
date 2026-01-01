from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal


class PilotDetectionError(ValueError):
    """Raised when pilot tone detection fails."""


@dataclass(frozen=True)
class PilotDetectionResult:
    first_start: int
    first_end: int
    second_start: int
    second_end: int
    confidence: float

    @property
    def start_sample(self) -> int:
        """Alias to the first pilot start sample (for backward compatibility)."""
        return self.first_start

    @property
    def end_sample(self) -> int:
        """Alias to the last pilot end sample (for backward compatibility)."""
        return self.second_end


def detect_pilot_tones(
    audio: npt.NDArray[np.float64],
    *,
    sample_rate: int,
    pilot_freq: float = 1000.0,
    threshold: float = 0.5,
    band_width_hz: float = 200.0,
    min_duration_ms: float = 90.0,
    pilot_duration_ms: float = 100.0,
) -> PilotDetectionResult:
    """Detect two pilot-tone regions and return their boundaries.

    The function applies a narrow band-pass filter around the pilot frequency,
    extracts the amplitude envelope via the Hilbert transform, and finds
    segments exceeding the given threshold. Two strongest segments are treated
    as the start/end pilots.
    """
    audio_arr = np.asarray(audio, dtype=np.float64)
    if audio_arr.ndim != 1:
        raise ValueError("audio must be a 1-D mono signal.")
    if audio_arr.size == 0:
        raise PilotDetectionError("空の信号です。")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    nyquist = sample_rate / 2.0
    if pilot_freq <= 0 or pilot_freq >= nyquist:
        raise ValueError("pilot_freq must be within (0, Nyquist).")

    low = max(pilot_freq - band_width_hz, 5.0)
    high = min(pilot_freq + band_width_hz, nyquist * 0.98)
    if low >= high:
        raise ValueError("Invalid band-pass bounds for pilot detection.")

    sos = signal.butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
    filtered = signal.sosfilt(sos, audio_arr)
    envelope = np.abs(signal.hilbert(filtered))
    if envelope.size == 0:
        raise PilotDetectionError("信号長が不足しています。")

    normalized = envelope / (float(np.max(envelope)) + 1e-12)
    min_samples = max(int(sample_rate * min_duration_ms / 1000), 1)
    pilot_samples = max(int(sample_rate * pilot_duration_ms / 1000), min_samples)

    mask = normalized >= threshold
    true_indices = np.flatnonzero(mask)
    if true_indices.size == 0:
        raise PilotDetectionError("パイロットトーンを検出できませんでした。")

    first_start = int(true_indices[0])
    second_end = int(true_indices[-1]) + 1

    first_end = min(first_start + pilot_samples, second_end)
    second_start = max(second_end - pilot_samples, first_end + 1)

    if second_start <= first_start:
        raise PilotDetectionError("2つのパイロットトーンを分離できませんでした。")

    first_slice = normalized[first_start:first_end]
    second_slice = normalized[second_start:second_end]
    first_peak = float(first_slice.max())
    second_peak = float(second_slice.max())
    amplitude_score = min(first_peak, second_peak, 1.0)
    if amplitude_score < max(0.1, threshold * 0.5):
        raise PilotDetectionError("パイロットトーンが閾値を下回っています。")
    stability_score = min(float(first_slice.mean()), float(second_slice.mean()), 1.0)
    confidence = float(min(amplitude_score, stability_score))

    return PilotDetectionResult(
        first_start=first_start,
        first_end=first_end,
        second_start=second_start,
        second_end=second_end,
        confidence=confidence,
    )
