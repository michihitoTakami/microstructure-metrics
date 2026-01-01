from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class AudioMetadata:
    sample_rate: int
    bit_depth: int
    channels: int
    duration_sec: float
    peak_amplitude: float
    rms_amplitude: float
    median_amplitude: float
    p95_amplitude: float
    dc_offset: float
    has_clipping: bool


@dataclass
class AudioPair:
    reference: npt.NDArray[np.float64]
    dut: npt.NDArray[np.float64]
    sample_rate: int
    metadata_ref: AudioMetadata
    metadata_dut: AudioMetadata


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    metadata_ref: AudioMetadata
    metadata_dut: AudioMetadata


def validate_audio_pair(
    *,
    metadata_ref: AudioMetadata,
    metadata_dut: AudioMetadata,
    pre_warnings: Iterable[str] | None = None,
    pre_errors: Iterable[str] | None = None,
    clip_warn_threshold: float = 0.99,
    dc_offset_threshold: float = 1e-3,
    silence_threshold: float = 1e-4,
    min_duration: float = 0.05,
    max_duration: float = 600.0,
) -> ValidationResult:
    """Validate reference/DUT metadata and collect warnings/errors."""
    errors: list[str] = list(pre_errors or [])
    warnings: list[str] = list(pre_warnings or [])

    if metadata_ref.sample_rate != metadata_dut.sample_rate:
        errors.append(
            f"Sample rate mismatch: ref={metadata_ref.sample_rate} Hz, "
            f"dut={metadata_dut.sample_rate} Hz"
        )
    if metadata_ref.channels != metadata_dut.channels:
        errors.append(
            f"Channel count mismatch: ref={metadata_ref.channels}, "
            f"dut={metadata_dut.channels}"
        )
    if metadata_ref.bit_depth != metadata_dut.bit_depth:
        warnings.append(
            f"Bit depth differs: ref={metadata_ref.bit_depth}bit, "
            f"dut={metadata_dut.bit_depth}bit"
        )

    warnings.extend(
        _validate_metadata(
            metadata_ref,
            clip_warn_threshold,
            dc_offset_threshold,
            silence_threshold,
            min_duration,
            max_duration,
            label="reference",
        )
    )
    warnings.extend(
        _validate_metadata(
            metadata_dut,
            clip_warn_threshold,
            dc_offset_threshold,
            silence_threshold,
            min_duration,
            max_duration,
            label="dut",
        )
    )

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        metadata_ref=metadata_ref,
        metadata_dut=metadata_dut,
    )


def _validate_metadata(
    metadata: AudioMetadata,
    clip_warn_threshold: float,
    dc_offset_threshold: float,
    silence_threshold: float,
    min_duration: float,
    max_duration: float,
    *,
    label: str,
) -> list[str]:
    warnings: list[str] = []
    if metadata.duration_sec < min_duration:
        warnings.append(
            f"{label}: duration {metadata.duration_sec:.3f}s is shorter than {min_duration}s"
        )
    if metadata.duration_sec > max_duration:
        warnings.append(
            f"{label}: duration {metadata.duration_sec:.1f}s exceeds {max_duration}s"
        )

    if metadata.has_clipping or metadata.peak_amplitude >= clip_warn_threshold:
        warnings.append(
            f"{label}: possible clipping (peak {metadata.peak_amplitude:.3f})"
        )

    if abs(metadata.dc_offset) >= dc_offset_threshold:
        warnings.append(f"{label}: DC offset {metadata.dc_offset:.4f}")

    if (
        metadata.rms_amplitude < silence_threshold
        or metadata.p95_amplitude < silence_threshold
    ):
        warnings.append(
            f"{label}: very low level "
            f"(rms {metadata.rms_amplitude:.4e}, p95 {metadata.p95_amplitude:.4e})"
        )

    return warnings
