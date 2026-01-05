from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf
from numpy.random import Generator, default_rng
from scipy import signal

from microstructure_metrics.metrics import (
    calculate_mps_similarity,
    calculate_tfs_correlation,
    calculate_thd_n,
    calculate_transient_metrics,
)
from microstructure_metrics.signals import (
    CommonSignalConfig,
    SignalBuildResult,
    build_signal,
    subtype_for_bit_depth,
)

DEGRADATION_TYPES = (
    "harmonic_distortion",
    "soft_clipping",
    "band_limit",
    "noise",
    "phase_distortion",
    "modulation_suppression",
    "time_smoothing",
)
_ALL_METRIC_KEYS = {
    "thd_n_db",
    "sinad_db",
    "mps_correlation",
    "mps_distance",
    "tfs_mean_correlation",
    "tfs_phase_coherence",
    "tfs_percentile_05_correlation",
    "tfs_correlation_variance",
    "attack_time_ms",
    "attack_time_delta_ms",
    "edge_sharpness_ratio",
    "transient_smearing_index",
    "transient_smearing_index_p95",
    "edge_sharpness_ratio_p05",
    "edge_sharpness_ratio_p95",
    "attack_time_delta_p95_ms",
    "transient_events_ref",
    "transient_events_dut",
    "transient_events_matched",
    "transient_unmatched_ref",
    "transient_unmatched_dut",
}


@dataclass(frozen=True)
class RegressionCase:
    key: str
    degradation: str
    signal_type: str
    severity: float = 0.5
    duration: float = 1.0
    sample_rate: int = 48_000
    rng_seed: int = 0
    signal_kwargs: dict[str, float | int | str | None] = field(default_factory=dict)
    degradation_kwargs: dict[str, float | int | str | None] = field(
        default_factory=dict
    )
    description: str = ""
    metrics: tuple[str, ...] | None = None


@dataclass(frozen=True)
class DegradedPair:
    reference: npt.NDArray[np.float64]
    dut: npt.NDArray[np.float64]
    sample_rate: int
    common: CommonSignalConfig
    reference_metadata: dict[str, object]
    dut_metadata: dict[str, object]
    stem: str
    degradation: str
    severity: float


DEFAULT_REGRESSION_CASES: tuple[RegressionCase, ...] = (
    RegressionCase(
        key="harmonic_distortion",
        degradation="harmonic_distortion",
        signal_type="thd",
        severity=0.7,
        duration=1.0,
        description="2nd/3rd harmonic injection around -60 dB.",
        metrics=("thd_n_db", "sinad_db"),
    ),
    RegressionCase(
        key="soft_clipping",
        degradation="soft_clipping",
        signal_type="pink-noise",
        severity=0.65,
        duration=1.2,
        description="tanh soft clipping to introduce broadband distortion.",
        metrics=("thd_n_db", "sinad_db"),
    ),
    RegressionCase(
        key="band_limit",
        degradation="band_limit",
        signal_type="tfs-tones",
        severity=0.8,
        duration=1.2,
        description="10 kHz low-pass to emulate bandwidth limitation.",
        metrics=(
            "mps_correlation",
            "mps_distance",
            "tfs_mean_correlation",
            "tfs_phase_coherence",
            "tfs_percentile_05_correlation",
            "tfs_correlation_variance",
        ),
    ),
    RegressionCase(
        key="noise_floor",
        degradation="noise",
        signal_type="thd",
        severity=0.6,
        duration=1.0,
        description="white noise at roughly -60 dBFS.",
        metrics=("thd_n_db", "sinad_db"),
    ),
    RegressionCase(
        key="phase_distortion",
        degradation="phase_distortion",
        signal_type="tfs-tones",
        severity=0.75,
        duration=1.0,
        description="slow phase warp via all-pass style modulation.",
        metrics=(
            "tfs_mean_correlation",
            "tfs_phase_coherence",
            "tfs_percentile_05_correlation",
            "tfs_correlation_variance",
        ),
    ),
    RegressionCase(
        key="modulation_suppression",
        degradation="modulation_suppression",
        signal_type="modulated",
        severity=0.7,
        duration=1.0,
        description="flatten AM depth with envelope smoothing.",
        metrics=("mps_correlation", "mps_distance"),
    ),
    RegressionCase(
        key="modulation_composite",
        degradation="modulation_suppression",
        signal_type="modulated",
        severity=0.65,
        duration=1.0,
        description="AM+FM composite with envelope flattening.",
        signal_kwargs={
            "carrier_freq": 1200.0,
            "am_freq": 6.0,
            "am_depth": 0.5,
            "fm_dev": 60.0,
            "fm_freq": 6.0,
        },
        metrics=("mps_correlation", "mps_distance"),
    ),
    RegressionCase(
        key="edge_rounding_click",
        degradation="band_limit",
        signal_type="click",
        severity=0.8,
        duration=1.0,
        degradation_kwargs={"cutoff_hz": 8000.0},
        description="band-limited click to emulate rounded transient edges.",
        metrics=(
            "attack_time_delta_ms",
            "edge_sharpness_ratio",
            "transient_smearing_index",
        ),
    ),
    RegressionCase(
        key="time_smoothing_attack",
        degradation="time_smoothing",
        signal_type="am-attack",
        severity=0.7,
        duration=1.0,
        description="envelope smoothing to emulate rounded attacks without heavy LPF.",
        metrics=(
            "attack_time_ms",
            "attack_time_delta_ms",
            "edge_sharpness_ratio",
            "transient_smearing_index",
        ),
    ),
    RegressionCase(
        key="tone_burst_smearing",
        degradation="time_smoothing",
        signal_type="tone-burst",
        severity=0.65,
        duration=1.0,
        description="smooth tone bursts to emulate rounded edges across repeated hits.",
        metrics=(
            "attack_time_ms",
            "attack_time_delta_ms",
            "edge_sharpness_ratio",
            "transient_smearing_index",
        ),
    ),
)


def generate_degraded_pair(
    *,
    degradation: str,
    severity: float = 0.5,
    signal_type: str | None = None,
    duration: float = 1.0,
    sample_rate: int = 48_000,
    rng_seed: int = 0,
    signal_kwargs: Mapping[str, float | int | str | None] | None = None,
    degradation_kwargs: Mapping[str, float | int | str | None] | None = None,
) -> DegradedPair:
    """Generate reference/DUT pair with a known degradation pattern."""
    if degradation not in DEGRADATION_TYPES:
        raise ValueError(f"Unsupported degradation type: {degradation}")
    base_signal = signal_type or _default_signal_for_degradation(degradation)
    common = CommonSignalConfig(sample_rate=sample_rate, duration=duration)
    rng = default_rng(rng_seed)

    base = _build_base_signal(
        signal_type=base_signal,
        common=common,
        rng_seed=rng_seed,
        signal_kwargs=signal_kwargs or {},
    )
    degraded = _apply_degradation(
        data=base.data,
        sample_rate=sample_rate,
        degradation=degradation,
        severity=severity,
        rng=rng,
        metadata=base.metadata,
        degradation_kwargs=degradation_kwargs or {},
    )

    # Stabilize metadata for regression fixtures (avoid noisy diffs due to timestamps).
    fixed_created_at = datetime(2000, 1, 1, tzinfo=UTC).isoformat()
    reference_metadata = dict(base.metadata)
    reference_metadata["created_at"] = fixed_created_at

    dut_metadata = dict(base.metadata)
    dut_metadata.update(
        {
            "degradation": degradation,
            "severity": float(severity),
            "rng_seed": rng_seed,
        }
    )
    dut_metadata["created_at"] = fixed_created_at
    stem = f"{base.suggested_stem}_{degradation}"

    return DegradedPair(
        reference=base.data,
        dut=degraded,
        sample_rate=sample_rate,
        common=common,
        reference_metadata=reference_metadata,
        dut_metadata=dut_metadata,
        stem=stem,
        degradation=degradation,
        severity=severity,
    )


def generate_pair_from_case(case: RegressionCase) -> DegradedPair:
    """Helper to build a deterministic pair based on a RegressionCase."""
    return generate_degraded_pair(
        degradation=case.degradation,
        severity=case.severity,
        signal_type=case.signal_type,
        duration=case.duration,
        sample_rate=case.sample_rate,
        rng_seed=case.rng_seed,
        signal_kwargs=case.signal_kwargs,
        degradation_kwargs=case.degradation_kwargs,
    )


def save_test_pair(
    pair: DegradedPair, output_dir: str | Path, stem: str | None = None
) -> tuple[Path, Path]:
    """Persist the reference/DUT WAVs and metadata JSON sidecars."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_stem = stem or pair.stem
    subtype = subtype_for_bit_depth(pair.common.bit_depth)

    ref_path = out_dir / f"{file_stem}_ref.wav"
    dut_path = out_dir / f"{file_stem}_dut.wav"
    sf.write(ref_path, pair.reference, samplerate=pair.sample_rate, subtype=subtype)
    sf.write(dut_path, pair.dut, samplerate=pair.sample_rate, subtype=subtype)

    ref_meta_path = ref_path.with_suffix(".json")
    dut_meta_path = dut_path.with_suffix(".json")
    ref_meta_path.write_text(
        json.dumps(pair.reference_metadata, ensure_ascii=False, indent=2)
    )
    dut_meta_path.write_text(
        json.dumps(pair.dut_metadata, ensure_ascii=False, indent=2)
    )
    return ref_path, dut_path


def _default_signal_for_degradation(degradation: str) -> str:
    mapping = {
        "harmonic_distortion": "thd",
        "soft_clipping": "pink-noise",
        "band_limit": "tfs-tones",
        "noise": "thd",
        "phase_distortion": "tfs-tones",
        "modulation_suppression": "modulated",
        "time_smoothing": "am-attack",
    }
    return mapping.get(degradation, "pink-noise")


def _coerce_float(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float(default)
    return float(default)


def _build_base_signal(
    *,
    signal_type: str,
    common: CommonSignalConfig,
    rng_seed: int,
    signal_kwargs: Mapping[str, float | int | str | None],
) -> SignalBuildResult:
    clean_kwargs = dict(signal_kwargs)
    return build_signal(
        signal_type,
        common=common,
        rng=default_rng(rng_seed),
        **clean_kwargs,  # type: ignore[arg-type]
    )


def _apply_degradation(
    *,
    data: npt.NDArray[np.float64],
    sample_rate: int,
    degradation: str,
    severity: float,
    rng: Generator,
    metadata: Mapping[str, object],
    degradation_kwargs: Mapping[str, float | int | str | None],
) -> npt.NDArray[np.float64]:
    level = float(np.clip(severity, 0.0, 1.0))
    degrad = degradation.lower()
    if degrad == "harmonic_distortion":
        return _apply_harmonic_distortion(
            data, sample_rate=sample_rate, level=level, metadata=metadata
        )
    if degrad == "soft_clipping":
        return _apply_soft_clipping(data, level=level)
    if degrad == "band_limit":
        cutoff = _coerce_float(degradation_kwargs.get("cutoff_hz", 10_000.0), 10_000.0)
        return _apply_band_limit(data, sample_rate=sample_rate, cutoff_hz=cutoff)
    if degrad == "noise":
        return _apply_noise(data, level=level, rng=rng)
    if degrad == "notch_fill":
        return _apply_notch_fill(
            data,
            sample_rate=sample_rate,
            level=level,
            metadata=metadata,
            rng=rng,
        )
    if degrad == "notch_blur":
        return _apply_notch_blur(
            data,
            level=level,
            blur_bins=degradation_kwargs.get("blur_bins"),
        )
    if degrad == "phase_distortion":
        return _apply_phase_distortion(data, sample_rate=sample_rate, level=level)
    if degrad == "modulation_suppression":
        return _apply_modulation_suppression(data, level=level)
    if degrad == "time_smoothing":
        smoothing_override = degradation_kwargs.get("smoothing_ms")
        smoothing_ms = (
            _coerce_float(smoothing_override, 0.0)
            if smoothing_override is not None
            else None
        )
        return _apply_time_smoothing(
            data,
            sample_rate=sample_rate,
            level=level,
            smoothing_ms=smoothing_ms,
        )
    raise ValueError(f"Unsupported degradation type: {degradation}")


def _apply_harmonic_distortion(
    data: npt.NDArray[np.float64],
    *,
    sample_rate: int,
    level: float,
    metadata: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    base_freq = _coerce_float(metadata.get("tone_freq_hz", 1000.0), 1000.0)
    target_db = -70.0 + 20.0 * level  # ~-70 to -50 dB relative to carrier
    amp = 10 ** (target_db / 20.0)
    t = np.arange(data.shape[0]) / sample_rate
    harmonics = amp * (
        np.sin(2 * np.pi * base_freq * 2 * t) + np.sin(2 * np.pi * base_freq * 3 * t)
    )
    return _match_peak(data + harmonics, reference=data)


def _apply_soft_clipping(
    data: npt.NDArray[np.float64],
    *,
    level: float,
) -> npt.NDArray[np.float64]:
    drive = 1.0 + 3.0 * level
    clipped = np.tanh(data * drive)
    return _match_peak(clipped, reference=data)


def _apply_band_limit(
    data: npt.NDArray[np.float64],
    *,
    sample_rate: int,
    cutoff_hz: float,
) -> npt.NDArray[np.float64]:
    nyquist = sample_rate / 2
    cutoff = min(cutoff_hz, nyquist * 0.95)
    sos = signal.butter(6, cutoff / nyquist, btype="low", output="sos")
    filtered = signal.sosfilt(sos, data)
    return _match_peak(filtered, reference=data)


def _apply_noise(
    data: npt.NDArray[np.float64],
    *,
    level: float,
    rng: Generator,
) -> npt.NDArray[np.float64]:
    ref_rms = _safe_rms(data)
    target_db = -70.0 + 20.0 * level  # ~-70 to -50 dBFS relative to RMS
    noise_rms = ref_rms * (10 ** (target_db / 20.0))
    noise = rng.standard_normal(data.shape[0]) * noise_rms
    return data + noise


def _apply_notch_fill(
    data: npt.NDArray[np.float64],
    *,
    sample_rate: int,
    level: float,
    metadata: Mapping[str, object],
    rng: Generator,
) -> npt.NDArray[np.float64]:
    center = _coerce_float(metadata.get("notch_center_hz"), 8000.0)
    q = _coerce_float(metadata.get("notch_q"), 8.6)
    nyquist = sample_rate / 2
    half_bw = center / max(q * 2, 1e-6)
    low = max(50.0, center - half_bw)
    high = min(nyquist * 0.99, center + half_bw)
    if high <= low:
        return data
    sos = signal.butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
    ref_rms = _safe_rms(data)
    target_db = -30.0 + 8.0 * level  # ~-30 to -22 dB below RMS
    band_noise = signal.sosfilt(sos, rng.standard_normal(data.shape[0]))
    current = _safe_rms(band_noise)
    if current > 0:
        band_noise = band_noise * (ref_rms * 10 ** (target_db / 20.0) / current)
    return _match_peak(data + band_noise, reference=data)


def _apply_notch_blur(
    data: npt.NDArray[np.float64],
    *,
    level: float,
    blur_bins: float | int | str | None,
) -> npt.NDArray[np.float64]:
    spec = np.fft.rfft(data)
    if spec.size == 0:
        return data
    magnitude = np.abs(spec)
    phase = np.angle(spec)
    base_radius = max(1, int(round(4 + 40 * level)))
    radius = int(max(1, _coerce_float(blur_bins, float(base_radius))))
    radius = min(radius, magnitude.shape[0] - 1)
    if radius <= 0:
        return data
    window = signal.windows.gaussian(2 * radius + 1, std=max(radius / 2.5, 1.0))
    window_sum = float(np.sum(window))
    if window_sum <= 0:
        return data
    kernel = window / window_sum
    blurred_mag = np.convolve(magnitude, kernel, mode="same")
    blurred_spec = blurred_mag * np.exp(1j * phase)
    blurred = np.fft.irfft(blurred_spec, n=data.shape[0])
    return _match_peak(np.real(blurred), reference=data)


def _apply_phase_distortion(
    data: npt.NDArray[np.float64],
    *,
    sample_rate: int,
    level: float,
) -> npt.NDArray[np.float64]:
    analytic = signal.hilbert(data)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    t = np.arange(data.shape[0]) / sample_rate
    mod_depth = 0.6 * level  # radians
    phase_mod = mod_depth * np.sin(2 * np.pi * 12.0 * t)
    distorted = envelope * np.exp(1j * (phase + phase_mod))
    return _match_peak(np.real(distorted), reference=data)


def _apply_modulation_suppression(
    data: npt.NDArray[np.float64],
    *,
    level: float,
) -> npt.NDArray[np.float64]:
    analytic = signal.hilbert(data)
    envelope = np.abs(analytic)
    phase = np.angle(analytic)
    suppression = float(np.clip(level, 0.0, 1.0))
    mean_env = float(np.mean(envelope))
    flattened_env = mean_env + (1.0 - suppression) * (envelope - mean_env)
    flattened = flattened_env * np.exp(1j * phase)
    return _match_peak(np.real(flattened), reference=data)


def _apply_time_smoothing(
    data: npt.NDArray[np.float64],
    *,
    sample_rate: int,
    level: float,
    smoothing_ms: float | None,
) -> npt.NDArray[np.float64]:
    target_ms = 1.0 + 6.0 * level
    tau_ms = max(0.1, smoothing_ms) if smoothing_ms is not None else max(0.1, target_ms)
    tau_seconds = tau_ms / 1000.0
    alpha = float(np.exp(-1.0 / max(sample_rate * tau_seconds, 1e-6)))
    b = [1.0 - alpha]
    a = [1.0, -alpha]
    analytic = signal.hilbert(data)
    envelope = np.abs(analytic)
    smoothed_env = signal.lfilter(b, a, envelope)
    smoothed = smoothed_env * np.exp(1j * np.angle(analytic))
    return _match_peak(np.real(smoothed), reference=data)


def _safe_rms(data: npt.NDArray[np.float64]) -> float:
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data), dtype=np.float64)))


def _match_peak(
    candidate: npt.NDArray[np.float64], *, reference: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    ref_peak = float(np.max(np.abs(reference))) if reference.size else 1.0
    cand_peak = float(np.max(np.abs(candidate))) if candidate.size else 1.0
    if cand_peak == 0:
        return candidate
    scaled = candidate * (ref_peak / cand_peak)
    return np.asarray(np.clip(scaled, -0.9999, 0.9999), dtype=np.float64)


def evaluate_metrics(
    *,
    reference: npt.NDArray[np.float64],
    dut: npt.NDArray[np.float64],
    sample_rate: int,
    metadata: Mapping[str, object],
    metrics: Iterable[str] | None = None,
) -> dict[str, float]:
    """Calculate selected metrics for a prepared reference/DUT pair."""
    requested = set(metrics) if metrics is not None else set(_ALL_METRIC_KEYS)
    results: dict[str, float] = {}

    if {"thd_n_db", "sinad_db"} & requested:
        fundamental_value = metadata.get("tone_freq_hz")
        if fundamental_value is not None:
            fundamental = _coerce_float(fundamental_value, 1000.0)
            expected_level_value = metadata.get("tone_level_dbfs", -3.0)
            expected_level = (
                _coerce_float(expected_level_value, -3.0)
                if expected_level_value is not None
                else None
            )
            thd = calculate_thd_n(
                signal=dut,
                fundamental_freq=fundamental,
                sample_rate=sample_rate,
                expected_level_dbfs=expected_level,
            )
            results["thd_n_db"] = thd.thd_n_db
            results["sinad_db"] = thd.sinad_db

    if {"mps_correlation", "mps_distance", "mps_distance_weighted"} & requested:
        mps = calculate_mps_similarity(
            reference=reference, dut=dut, sample_rate=sample_rate
        )
        results["mps_correlation"] = mps.mps_correlation
        results["mps_distance"] = mps.mps_distance
        results["mps_distance_weighted"] = mps.mps_distance_weighted

    if {
        "tfs_mean_correlation",
        "tfs_phase_coherence",
        "tfs_percentile_05_correlation",
        "tfs_correlation_variance",
    } & requested:
        tfs = calculate_tfs_correlation(
            reference=reference, dut=dut, sample_rate=sample_rate
        )
        results["tfs_mean_correlation"] = tfs.mean_correlation
        results["tfs_phase_coherence"] = tfs.phase_coherence
        results["tfs_percentile_05_correlation"] = tfs.percentile_05_correlation
        results["tfs_correlation_variance"] = tfs.correlation_variance

    if {
        "attack_time_ms",
        "attack_time_delta_ms",
        "edge_sharpness_ratio",
        "transient_smearing_index",
        "transient_smearing_index_p95",
        "edge_sharpness_ratio_p05",
        "edge_sharpness_ratio_p95",
        "attack_time_delta_p95_ms",
        "transient_events_ref",
        "transient_events_dut",
        "transient_events_matched",
        "transient_unmatched_ref",
        "transient_unmatched_dut",
    } & requested:
        transient = calculate_transient_metrics(
            reference=reference, dut=dut, sample_rate=sample_rate
        )
        results["attack_time_ms"] = transient.attack_time_dut_ms
        results["attack_time_delta_ms"] = transient.attack_time_delta_ms
        results["edge_sharpness_ratio"] = transient.edge_sharpness_ratio
        results["transient_smearing_index"] = transient.transient_smearing_index
        results["transient_smearing_index_p95"] = (
            transient.width_ratio_stats.percentile_95
        )
        results["edge_sharpness_ratio_p05"] = (
            transient.edge_sharpness_ratio_stats.percentile_05
        )
        results["edge_sharpness_ratio_p95"] = (
            transient.edge_sharpness_ratio_stats.percentile_95
        )
        results["attack_time_delta_p95_ms"] = (
            transient.attack_time_delta_stats_ms.percentile_95
        )
        results["transient_events_ref"] = len(transient.ref_events)
        results["transient_events_dut"] = len(transient.dut_events)
        results["transient_events_matched"] = transient.matched_event_pairs
        results["transient_unmatched_ref"] = transient.unmatched_ref_events
        results["transient_unmatched_dut"] = transient.unmatched_dut_events

    return {k: v for k, v in results.items() if k in requested}
