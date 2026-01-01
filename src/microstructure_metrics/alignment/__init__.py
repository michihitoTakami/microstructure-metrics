"""Alignment utilities for pilot-tone based synchronization."""

from microstructure_metrics.alignment.align import (
    AlignmentResult,
    align_audio_pair,
    align_signals,
    extract_test_segment,
)
from microstructure_metrics.alignment.correlation import estimate_delay
from microstructure_metrics.alignment.pilot import (
    PilotDetectionError,
    PilotDetectionResult,
    detect_pilot_tones,
)

__all__ = [
    "AlignmentResult",
    "PilotDetectionError",
    "PilotDetectionResult",
    "align_audio_pair",
    "align_signals",
    "detect_pilot_tones",
    "estimate_delay",
    "extract_test_segment",
]
