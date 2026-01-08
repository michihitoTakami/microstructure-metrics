from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from microstructure_metrics.alignment.correlation import estimate_delay
from microstructure_metrics.alignment.pilot import (
    PilotDetectionResult,
    detect_pilot_tones,
)


@dataclass(frozen=True)
class AlignmentResult:
    aligned_ref: npt.NDArray[np.float64]
    aligned_dut: npt.NDArray[np.float64]
    delay_samples: float
    start_sample: int
    end_sample: int
    confidence: float


def extract_test_segment(
    audio: npt.NDArray[np.float64],
    pilot_result: PilotDetectionResult,
    *,
    sample_rate: int,
    margin_ms: float = 5.0,
) -> npt.NDArray[np.float64]:
    """Cut out the test body between the two detected pilot tones."""
    if margin_ms < 0:
        raise ValueError("margin_ms must be non-negative.")
    if audio.ndim not in {1, 2}:
        raise ValueError("audio must be 1-D or 2-D.")
    margin_samples = int(sample_rate * margin_ms / 1000)
    start = pilot_result.first_end + margin_samples
    end = pilot_result.second_start - margin_samples
    if start >= end:
        raise ValueError(
            "テスト区間の切り出しに失敗しました。margin_ms を確認してください。"
        )
    return np.asarray(audio[start:end], dtype=np.float64)


def align_signals(
    reference: npt.NDArray[np.float64],
    dut: npt.NDArray[np.float64],
    *,
    delay_samples: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Shift DUT by the estimated delay and trim to the common length.

    1D/2D (samples, channels) に対応し、0軸方向にシフトを適用する。
    """
    ref = np.asarray(reference, dtype=np.float64)
    du = np.asarray(dut, dtype=np.float64)
    if ref.ndim not in {1, 2} or du.ndim != ref.ndim:
        raise ValueError("reference/dut must be 1-D or 2-D signals with same rank.")

    shift = int(np.round(delay_samples))
    if shift > 0:
        du_shifted = du[shift:]
        ref_shifted = ref[: du_shifted.shape[0]]
    elif shift < 0:
        ref_shifted = ref[-shift:]
        du_shifted = du[: ref_shifted.shape[0]]
    else:
        length = min(ref.shape[0], du.shape[0])
        ref_shifted = ref[:length]
        du_shifted = du[:length]

    final_length = min(ref_shifted.shape[0], du_shifted.shape[0])
    if final_length <= 0:
        raise ValueError(
            "アライメント後の信号長が0です。遅延推定値を確認してください。"
        )
    return ref_shifted[:final_length], du_shifted[:final_length]


def align_audio_pair(
    *,
    reference: npt.NDArray[np.float64],
    dut: npt.NDArray[np.float64],
    sample_rate: int,
    pilot_freq: float = 1000.0,
    threshold: float = 0.5,
    band_width_hz: float = 200.0,
    min_duration_ms: float = 90.0,
    pilot_duration_ms: float = 100.0,
    margin_ms: float = 5.0,
    max_lag_ms: float = 100.0,
    refine_delay: bool = True,
) -> AlignmentResult:
    """End-to-end alignment: detect pilots, cut body, estimate delay, and align."""
    ref_signal = np.asarray(reference, dtype=np.float64)
    dut_signal = np.asarray(dut, dtype=np.float64)
    if ref_signal.ndim not in {1, 2} or dut_signal.ndim != ref_signal.ndim:
        raise ValueError("reference/dut must be 1-D or 2-D signals.")

    def _alignment_view(signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if signal.ndim == 1:
            return signal
        return np.asarray(np.mean(signal, axis=1), dtype=np.float64)

    ref_view = _alignment_view(ref_signal)
    dut_view = _alignment_view(dut_signal)

    ref_pilot = detect_pilot_tones(
        ref_view,
        sample_rate=sample_rate,
        pilot_freq=pilot_freq,
        threshold=threshold,
        band_width_hz=band_width_hz,
        min_duration_ms=min_duration_ms,
        pilot_duration_ms=pilot_duration_ms,
    )
    dut_pilot = detect_pilot_tones(
        dut_view,
        sample_rate=sample_rate,
        pilot_freq=pilot_freq,
        threshold=threshold,
        band_width_hz=band_width_hz,
        min_duration_ms=min_duration_ms,
        pilot_duration_ms=pilot_duration_ms,
    )

    ref_segment_view = extract_test_segment(
        ref_view, ref_pilot, sample_rate=sample_rate, margin_ms=margin_ms
    )
    dut_segment_view = extract_test_segment(
        dut_view, dut_pilot, sample_rate=sample_rate, margin_ms=margin_ms
    )
    ref_segment = extract_test_segment(
        ref_signal, ref_pilot, sample_rate=sample_rate, margin_ms=margin_ms
    )
    dut_segment = extract_test_segment(
        dut_signal, dut_pilot, sample_rate=sample_rate, margin_ms=margin_ms
    )

    pilot_offset = dut_pilot.first_start - ref_pilot.first_start

    residual_delay = estimate_delay(
        reference=ref_segment_view,
        dut=dut_segment_view,
        sample_rate=sample_rate,
        max_lag_ms=max_lag_ms,
        refine=refine_delay,
    )
    total_delay = pilot_offset + residual_delay
    aligned_ref, aligned_dut = align_signals(
        ref_segment, dut_segment, delay_samples=residual_delay
    )

    return AlignmentResult(
        aligned_ref=aligned_ref,
        aligned_dut=aligned_dut,
        delay_samples=total_delay,
        start_sample=ref_pilot.start_sample,
        end_sample=ref_pilot.end_sample,
        confidence=min(ref_pilot.confidence, dut_pilot.confidence),
    )
