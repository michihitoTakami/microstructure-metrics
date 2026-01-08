from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import signal

DEFAULT_PILOT_DURATION_MS = 100
DEFAULT_FADE_MS = 5
DEFAULT_BAND_WIDTH_HZ = 200.0
DEFAULT_THRESHOLD = 0.3


@dataclass(frozen=True)
class PilotDetection:
    """検出したパイロット位置と信頼度."""

    start_sample: int
    end_sample: int
    start_confidence: float
    end_confidence: float


@dataclass(frozen=True)
class DriftEstimate:
    """クロックドリフト推定結果."""

    delay_start_samples: int
    delay_end_samples: int
    drift_samples_per_second: float
    drift_ppm: float
    pilot_detection: PilotDetection


@dataclass(frozen=True)
class DriftWarning:
    """ドリフト警告情報."""

    drift_ppm: float
    severity: Literal["none", "low", "high", "critical"]
    message: str


def drift_to_report(result: DriftEstimate, warning: DriftWarning) -> dict[str, object]:
    """レポート/JSON出力向けの辞書へ変換する."""
    return {
        "drift_ppm": result.drift_ppm,
        "drift_samples_per_second": result.drift_samples_per_second,
        "delay_start_samples": result.delay_start_samples,
        "delay_end_samples": result.delay_end_samples,
        "severity": warning.severity,
        "message": warning.message,
        "pilot_confidence": {
            "start": result.pilot_detection.start_confidence,
            "end": result.pilot_detection.end_confidence,
        },
    }


def estimate_clock_drift(
    *,
    reference: npt.NDArray[np.float64] | npt.NDArray[np.float32],
    dut: npt.NDArray[np.float64] | npt.NDArray[np.float32],
    sample_rate: int,
    pilot_freq: float = 1000.0,
    pilot_duration_ms: int = DEFAULT_PILOT_DURATION_MS,
    band_width_hz: float = DEFAULT_BAND_WIDTH_HZ,
    peak_threshold: float = DEFAULT_THRESHOLD,
) -> DriftEstimate:
    """パイロット位置の差分からクロックドリフトを推定する."""
    ref_arr = np.asarray(reference, dtype=np.float64)
    dut_arr = np.asarray(dut, dtype=np.float64)

    # Stereo/multi-channel: estimate drift on the shared mid (mean across channels).
    if ref_arr.ndim == 2:
        ref_arr = np.mean(ref_arr, axis=1)
    if dut_arr.ndim == 2:
        dut_arr = np.mean(dut_arr, axis=1)

    ref = ref_arr.flatten()
    dut_sig = dut_arr.flatten()
    if ref.ndim != 1 or dut_sig.ndim != 1:
        raise ValueError("reference/dut は1chまたは2ch(s, ch)の波形を期待します。")
    if sample_rate <= 0:
        raise ValueError("sample_rate は正の整数で指定してください。")

    template = _pilot_template(
        sample_rate=sample_rate,
        freq=pilot_freq,
        duration_ms=pilot_duration_ms,
    )

    ref_filtered = _bandpass(ref, sample_rate, pilot_freq, band_width_hz)
    dut_filtered = _bandpass(dut_sig, sample_rate, pilot_freq, band_width_hz)

    ref_pilot = _detect_pilots(
        ref_filtered,
        template=template,
        peak_threshold=peak_threshold,
    )
    dut_pilot = _detect_pilots(
        dut_filtered,
        template=template,
        peak_threshold=peak_threshold,
    )

    delay_start = dut_pilot.start_sample - ref_pilot.start_sample
    delay_end = dut_pilot.end_sample - ref_pilot.end_sample

    duration_sec = min(ref.shape[0], dut_sig.shape[0]) / float(sample_rate)
    if duration_sec <= 0:
        raise ValueError("信号長が0です。入力波形を確認してください。")

    drift_samples_per_second = (delay_end - delay_start) / duration_sec
    drift_ppm = drift_samples_per_second / float(sample_rate) * 1_000_000

    detection = PilotDetection(
        start_sample=ref_pilot.start_sample,
        end_sample=ref_pilot.end_sample,
        start_confidence=min(ref_pilot.start_confidence, dut_pilot.start_confidence),
        end_confidence=min(ref_pilot.end_confidence, dut_pilot.end_confidence),
    )

    return DriftEstimate(
        delay_start_samples=delay_start,
        delay_end_samples=delay_end,
        drift_samples_per_second=drift_samples_per_second,
        drift_ppm=drift_ppm,
        pilot_detection=detection,
    )


def check_drift_threshold(drift_ppm: float) -> DriftWarning:
    """閾値テーブルに基づき重大度を判定する."""
    abs_ppm = abs(drift_ppm)
    if abs_ppm < 5:
        return DriftWarning(drift_ppm=drift_ppm, severity="none", message="")
    if abs_ppm < 20:
        return DriftWarning(
            drift_ppm=drift_ppm,
            severity="low",
            message=f"Minor clock drift detected: {drift_ppm:.1f} ppm",
        )
    if abs_ppm < 100:
        return DriftWarning(
            drift_ppm=drift_ppm,
            severity="high",
            message=(
                f"Significant clock drift: {drift_ppm:.1f} ppm. "
                "TFS results may be unreliable."
            ),
        )
    return DriftWarning(
        drift_ppm=drift_ppm,
        severity="critical",
        message=(
            f"Critical clock drift: {drift_ppm:.1f} ppm. "
            "Use synchronized clocks for reliable measurements."
        ),
    )


def _pilot_template(
    *, sample_rate: int, freq: float, duration_ms: int, fade_ms: int = DEFAULT_FADE_MS
) -> npt.NDArray[np.float64]:
    samples = max(int(sample_rate * duration_ms / 1000), 1)
    t = np.arange(samples) / sample_rate
    tone = np.sin(2 * np.pi * freq * t)
    fade_samples = max(int(sample_rate * fade_ms / 1000), 1)
    return _apply_fade(tone, fade_samples=fade_samples)


def _bandpass(
    data: npt.NDArray[np.float64],
    sample_rate: int,
    center_hz: float,
    band_width_hz: float,
) -> npt.NDArray[np.float64]:
    nyquist = sample_rate / 2
    low = max(center_hz - band_width_hz, 1.0)
    high = min(center_hz + band_width_hz, nyquist * 0.99)
    sos = signal.butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
    return np.asarray(signal.sosfilt(sos, data), dtype=np.float64)


def _detect_pilots(
    data: npt.NDArray[np.float64],
    *,
    template: npt.NDArray[np.float64],
    peak_threshold: float,
) -> PilotDetection:
    ncc = _normalized_correlation(data, template)
    min_distance = max(len(template), int(len(template) * 0.8))
    peaks, props = signal.find_peaks(ncc, distance=min_distance)
    heights = props.get("peak_heights", ncc[peaks])

    if peak_threshold > 0:
        mask = heights >= peak_threshold
        peaks = peaks[mask]
        heights = heights[mask]

    if peaks.size >= 2:
        top = np.argsort(heights)[-2:]
        selected_peaks = peaks[top]
        selected_heights = heights[top]
    else:
        top_two = np.argsort(ncc)[-2:]
        selected_peaks = top_two
        selected_heights = ncc[top_two]

    selected_peaks = np.sort(selected_peaks)
    selected_heights = selected_heights[np.argsort(selected_peaks)]

    if selected_peaks.size < 2:
        raise ValueError("パイロットが2箇所検出できませんでした。")

    start_idx = int(selected_peaks[0])
    end_idx = int(selected_peaks[-1])
    start_conf = float(selected_heights[0])
    end_conf = float(selected_heights[-1])

    return PilotDetection(
        start_sample=start_idx,
        end_sample=end_idx,
        start_confidence=start_conf,
        end_confidence=end_conf,
    )


def _normalized_correlation(
    data: npt.NDArray[np.float64], template: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    template_zero_mean = template - np.mean(template)
    template_energy = np.sum(template_zero_mean**2)

    if template_energy <= 0:
        raise ValueError("テンプレートのエネルギーが0です。")

    corr = signal.correlate(data, template_zero_mean, mode="valid")
    window_energy = signal.correlate(
        data**2, np.ones_like(template_zero_mean), mode="valid"
    )
    window_energy = np.clip(window_energy, 0.0, None)
    denom = np.sqrt(window_energy * template_energy + 1e-12)
    return np.asarray(corr / denom, dtype=np.float64)


def _apply_fade(
    data: npt.NDArray[np.float64], *, fade_samples: int
) -> npt.NDArray[np.float64]:
    if fade_samples <= 0:
        return data
    fade_samples = min(fade_samples, data.shape[0] // 2)
    if fade_samples == 0:
        return data
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_samples))
    out = data.copy()
    out[:fade_samples] *= ramp
    out[-fade_samples:] *= ramp[::-1]
    return out
