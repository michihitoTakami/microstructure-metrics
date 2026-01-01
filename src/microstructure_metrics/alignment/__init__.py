"""アライメント系ユーティリティ (S-04/S-05)."""

from microstructure_metrics.alignment.align import (
    AlignmentResult,
    align_audio_pair,
    align_signals,
    extract_test_segment,
)
from microstructure_metrics.alignment.correlation import estimate_delay
from microstructure_metrics.alignment.drift import (
    DriftEstimate,
    DriftWarning,
    check_drift_threshold,
    drift_to_report,
    estimate_clock_drift,
)
from microstructure_metrics.alignment.pilot import (
    PilotDetectionError,
    PilotDetectionResult,
    detect_pilot_tones,
)

__all__ = [
    "AlignmentResult",
    "PilotDetectionError",
    "PilotDetectionResult",
    "DriftEstimate",
    "DriftWarning",
    "align_audio_pair",
    "align_signals",
    "check_drift_threshold",
    "detect_pilot_tones",
    "drift_to_report",
    "estimate_clock_drift",
<<<<<<< HEAD
    "estimate_delay",
    "extract_test_segment",
=======
    "drift_to_report",
>>>>>>> 7e1ece7 (feat: add drift warning cli and report output)
]
