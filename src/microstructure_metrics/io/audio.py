from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

from microstructure_metrics.io.preprocess import (
    normalize_audio,
    remove_dc_offset,
    resample_audio,
)
from microstructure_metrics.io.validation import (
    AudioMetadata,
    AudioPair,
    ValidationResult,
    validate_audio_pair,
)


def load_audio_pair(
    reference_path: str | Path,
    dut_path: str | Path,
    *,
    validate: bool = True,
    allow_resample: bool = False,
    target_sample_rate: int | None = None,
    channel: int | None = None,
    remove_dc: bool = True,
    dc_method: Literal["mean", "highpass"] = "mean",
    dc_cutoff_hz: float = 10.0,
    normalize: Literal["peak", "rms"] | None = None,
    normalize_target: float = 0.99,
    clip_warn_threshold: float = 0.99,
    dc_offset_threshold: float = 1e-3,
    silence_threshold: float = 1e-4,
    min_duration: float = 0.05,
    max_duration: float = 600.0,
) -> tuple[np.ndarray, np.ndarray, ValidationResult]:
    """Load reference/DUT WAV, optionally validate, and return arrays + validation."""
    ref_path = Path(reference_path)
    dut_path = Path(dut_path)

    ref_data, ref_sr, ref_bit_depth = _read_audio(ref_path)
    dut_data, dut_sr, dut_bit_depth = _read_audio(dut_path)

    pre_warnings: list[str] = []
    pre_errors: list[str] = []

    ref_data, warning = _select_channel(ref_data, channel)
    if warning:
        pre_warnings.append(f"reference: {warning}")

    dut_data, warning = _select_channel(dut_data, channel)
    if warning:
        pre_warnings.append(f"dut: {warning}")

    target_sr = target_sample_rate or ref_sr
    if allow_resample or target_sample_rate is not None:
        if ref_sr != target_sr:
            pre_warnings.append(f"reference: resampled {ref_sr} -> {target_sr} Hz")
            ref_data = resample_audio(ref_data, orig_sr=ref_sr, target_sr=target_sr)
            ref_sr = target_sr
        if dut_sr != target_sr:
            pre_warnings.append(f"dut: resampled {dut_sr} -> {target_sr} Hz")
            dut_data = resample_audio(dut_data, orig_sr=dut_sr, target_sr=target_sr)
            dut_sr = target_sr
    elif ref_sr != dut_sr:
        pre_errors.append(
            f"Sample rate mismatch: ref={ref_sr} Hz, dut={dut_sr} Hz "
            "(enable allow_resample to auto-fix)"
        )

    if remove_dc:
        ref_data = remove_dc_offset(
            ref_data, method=dc_method, sample_rate=ref_sr, cutoff_hz=dc_cutoff_hz
        )
        dut_data = remove_dc_offset(
            dut_data, method=dc_method, sample_rate=dut_sr, cutoff_hz=dc_cutoff_hz
        )

    if normalize is not None:
        ref_data = normalize_audio(ref_data, mode=normalize, target=normalize_target)
        dut_data = normalize_audio(dut_data, mode=normalize, target=normalize_target)

    ref_meta = _compute_metadata(ref_data, sample_rate=ref_sr, bit_depth=ref_bit_depth)
    dut_meta = _compute_metadata(dut_data, sample_rate=dut_sr, bit_depth=dut_bit_depth)

    validation = validate_audio_pair(
        metadata_ref=ref_meta,
        metadata_dut=dut_meta,
        pre_warnings=pre_warnings,
        pre_errors=pre_errors,
        clip_warn_threshold=clip_warn_threshold,
        dc_offset_threshold=dc_offset_threshold,
        silence_threshold=silence_threshold,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    if validate and validation.errors:
        joined = "; ".join(validation.errors)
        raise ValueError(f"Validation failed: {joined}")

    pair = AudioPair(
        reference=ref_data,
        dut=dut_data,
        sample_rate=ref_sr,
        metadata_ref=ref_meta,
        metadata_dut=dut_meta,
    )
    return pair.reference, pair.dut, validation


def _read_audio(path: Path) -> tuple[np.ndarray, int, int]:
    data, sample_rate = sf.read(path, always_2d=True)
    info = sf.info(path)
    bit_depth = _bit_depth_from_subtype(info.subtype)
    return data.astype(np.float64), sample_rate, bit_depth


def _select_channel(
    data: np.ndarray, channel: int | None
) -> tuple[np.ndarray, str | None]:
    if data.ndim == 1:
        return data.astype(np.float64), None
    channels = data.shape[1]
    if channels == 1:
        return data[:, 0].astype(np.float64), None
    if channel is None:
        return data[:, 0].astype(np.float64), "stereo input; using channel 0"
    if not 0 <= channel < channels:
        raise ValueError(f"channel index must be in [0, {channels - 1}]")
    return data[:, channel].astype(np.float64), None


def _compute_metadata(
    data: np.ndarray, *, sample_rate: int, bit_depth: int
) -> AudioMetadata:
    duration = data.shape[0] / sample_rate if sample_rate > 0 else 0.0
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = (
        float(np.sqrt(np.mean(np.square(data), dtype=np.float64))) if data.size else 0.0
    )
    abs_data = np.abs(data) if data.size else np.array([], dtype=np.float64)
    median = float(np.median(abs_data)) if abs_data.size else 0.0
    p95 = float(np.percentile(abs_data, 95)) if abs_data.size else 0.0
    dc = float(np.mean(data)) if data.size else 0.0
    has_clipping = peak >= 0.999
    return AudioMetadata(
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        channels=1,
        duration_sec=duration,
        peak_amplitude=peak,
        rms_amplitude=rms,
        median_amplitude=median,
        p95_amplitude=p95,
        dc_offset=dc,
        has_clipping=has_clipping,
    )


def _bit_depth_from_subtype(subtype: str) -> int:
    normalized = subtype.upper()
    mapping = {
        "PCM_U8": 8,
        "PCM_16": 16,
        "PCM_24": 24,
        "PCM_32": 32,
        "FLOAT": 32,
        "DOUBLE": 64,
    }
    if normalized in mapping:
        return mapping[normalized]
    # fallback: attempt to parse trailing digits
    digits = "".join(ch for ch in normalized if ch.isdigit())
    return int(digits) if digits else 0
