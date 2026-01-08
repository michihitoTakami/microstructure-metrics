from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from scipy import signal

DEFAULT_VERSION = "1.0.0"
DEFAULT_FILE_VERSION_TAG = "v1"
SUPPORTED_SIGNALS = (
    "thd",
    "notched-noise",
    "pink-noise",
    "modulated",
    "tfs-tones",
    "tone-burst",
    "am-attack",
    "click",
    "complex-bass",
    "binaural-cues",
    "ms-side-texture",
)


def _format_q_for_stem(q: float) -> str:
    # Keep filenames stable/readable: 2.0 -> "2", 8.6 -> "8.6"
    return f"{float(q):g}"


@dataclass(frozen=True)
class CommonSignalConfig:
    sample_rate: int = 48000
    bit_depth: str = "24bit"
    duration: float = 10.0  # test body length (seconds)
    pilot_freq: float = 1000.0
    pilot_duration_ms: int = 100
    silence_duration_ms: int = 500
    pilot_level_dbfs: float = -6.0
    fade_ms: int = 5
    version: str = DEFAULT_VERSION

    def normalized_bit_depth(self) -> str:
        bit = self.bit_depth.lower().replace("bit", "").replace("pcm_", "")
        if bit in {"24", "pcm24", "pcm24bit", "pcm"}:
            return "24bit"
        if bit in {"32f", "float32", "32"}:
            return "32f"
        raise ValueError(f"Unsupported bit depth: {self.bit_depth}")


@dataclass
class SignalBuildResult:
    data: npt.NDArray[np.float64]
    metadata: dict[str, object]
    suggested_stem: str
    channels: int


def default_output_stem(
    signal_type: str, common: CommonSignalConfig, descriptor: str | None = None
) -> str:
    parts = [
        signal_type.replace("-", "_"),
        descriptor if descriptor else None,
        str(common.sample_rate),
        common.normalized_bit_depth(),
        DEFAULT_FILE_VERSION_TAG,
    ]
    compact = [p for p in parts if p]
    return "_".join(compact)


def subtype_for_bit_depth(bit_depth: str) -> str:
    normalized = CommonSignalConfig(bit_depth=bit_depth).normalized_bit_depth()
    return "PCM_24" if normalized == "24bit" else "FLOAT"


def build_signal(
    signal_type: str,
    *,
    common: CommonSignalConfig,
    rng: Generator | None = None,
    tone_freq: float = 1000.0,
    tone_level_dbfs: float = -3.0,
    notch_center: float = 8000.0,
    notch_q: float = 8.6,
    notch_centers_hz: list[float] | None = None,
    notch_cascade_stages: int = 1,
    noise_lowcut: float = 20.0,
    noise_highcut: float | None = 20000.0,
    am_freq: float = 4.0,
    am_depth: float = 0.5,
    fm_dev: float = 50.0,
    fm_freq: float | None = None,
    carrier_freq: float = 1000.0,
    min_tone_freq: float = 4000.0,
    tone_count: int = 5,
    tone_step: float = 2000.0,
    burst_freq: float = 8000.0,
    burst_cycles: int = 10,
    burst_level_dbfs: float = -6.0,
    burst_fade_cycles: int = 2,
    click_level_dbfs: float = -6.0,
    click_band_limit_hz: float = 20000.0,
    attack_ms: float = 2.0,
    release_ms: float = 10.0,
    gate_period_ms: float = 100.0,
    binaural_itd_ms: float = 0.35,
    binaural_ild_db: float = 6.0,
) -> SignalBuildResult:
    """Generate a test signal body + timeline and metadata."""
    normalized_type = signal_type.lower()
    if normalized_type not in SUPPORTED_SIGNALS:
        raise ValueError(f"Unsupported signal type: {signal_type}")

    sample_rate = common.sample_rate
    body_duration = max(common.duration, 0.01)
    samples = int(body_duration * sample_rate)
    rng = rng or np.random.default_rng()

    if normalized_type == "thd":
        body = _generate_sine(freq=tone_freq, sample_rate=sample_rate, samples=samples)
        body = _scale_to_dbfs(body, tone_level_dbfs, mode="peak")
        descriptor = f"{int(tone_freq)}hz"
        extra_meta: dict[str, object] = {
            "tone_freq_hz": tone_freq,
            "tone_level_dbfs": tone_level_dbfs,
        }
    elif normalized_type == "notched-noise":
        if notch_cascade_stages < 1:
            raise ValueError("notch_cascade_stages must be >= 1")
        centers = notch_centers_hz or [notch_center]
        body = _generate_notched_noise(
            sample_rate=sample_rate,
            samples=samples,
            centers_hz=centers,
            q=notch_q,
            cascade_stages=notch_cascade_stages,
            lowcut=noise_lowcut,
            highcut=noise_highcut,
            rng=rng,
        )
        q_text = _format_q_for_stem(notch_q)
        if len(centers) == 1:
            descriptor = f"{int(centers[0])}hz_q{q_text}"
        else:
            descriptor = (
                f"{len(centers)}n_{int(min(centers))}-{int(max(centers))}hz_q{q_text}"
            )
        extra_meta = {
            "notch_center_hz": float(centers[0]),
            "notch_centers_hz": [float(c) for c in centers],
            "notch_q": notch_q,
            "notch_cascade_stages": int(notch_cascade_stages),
            "noise_color": "pink",
            "noise_lowcut_hz": noise_lowcut,
            "noise_highcut_hz": noise_highcut,
            "target_rms_dbfs": -14.0,
        }
    elif normalized_type == "pink-noise":
        body = _generate_pink_noise(
            sample_rate=sample_rate,
            samples=samples,
            lowcut=noise_lowcut,
            highcut=noise_highcut,
            rng=rng,
        )
        descriptor = f"{int(noise_lowcut)}-{int(noise_highcut or sample_rate / 2)}hz"
        extra_meta = {
            "noise_color": "pink",
            "noise_lowcut_hz": noise_lowcut,
            "noise_highcut_hz": noise_highcut,
            "target_rms_dbfs": -14.0,
        }
    elif normalized_type == "modulated":
        body = _generate_modulated(
            sample_rate=sample_rate,
            samples=samples,
            carrier_hz=carrier_freq,
            am_freq_hz=am_freq,
            am_depth_ratio=am_depth,
            fm_dev_hz=fm_dev,
            fm_freq_hz=fm_freq or am_freq,
        )
        descriptor = f"{int(carrier_freq)}hz_am{am_freq}hz{int(am_depth * 100)}"
        extra_meta = {
            "carrier_hz": carrier_freq,
            "am_freq_hz": am_freq,
            "am_depth_ratio": am_depth,
            "fm_dev_hz": fm_dev,
            "fm_freq_hz": fm_freq or am_freq,
            "target_peak_dbfs": -6.0,
        }
    elif normalized_type == "tone-burst":
        body = _generate_tone_burst(
            sample_rate=sample_rate,
            samples=samples,
            freq_hz=burst_freq,
            cycles=int(burst_cycles),
            fade_cycles=int(burst_fade_cycles),
        )
        body = _scale_to_dbfs(body, burst_level_dbfs, mode="peak")
        descriptor = f"{int(burst_freq)}hz_{int(burst_cycles)}cy"
        extra_meta = {
            "burst_freq_hz": float(burst_freq),
            "burst_cycles": int(burst_cycles),
            "burst_fade_cycles": int(burst_fade_cycles),
            "burst_level_dbfs": float(burst_level_dbfs),
        }
    elif normalized_type == "am-attack":
        body = _generate_am_attack(
            sample_rate=sample_rate,
            samples=samples,
            carrier_hz=carrier_freq,
            attack_ms=float(attack_ms),
            release_ms=float(release_ms),
            period_ms=float(gate_period_ms),
        )
        body = _scale_to_dbfs(body, -6.0, mode="peak")
        descriptor = f"{int(carrier_freq)}hz_atk{attack_ms}ms"
        extra_meta = {
            "carrier_hz": float(carrier_freq),
            "attack_ms": float(attack_ms),
            "release_ms": float(release_ms),
            "gate_period_ms": float(gate_period_ms),
            "target_peak_dbfs": -6.0,
        }
    elif normalized_type == "click":
        body = _generate_click(
            sample_rate=sample_rate,
            samples=samples,
            band_limit_hz=float(click_band_limit_hz),
        )
        body = _scale_to_dbfs(body, click_level_dbfs, mode="peak")
        descriptor = f"bl{int(click_band_limit_hz)}hz"
        extra_meta = {
            "click_level_dbfs": float(click_level_dbfs),
            "click_band_limit_hz": float(click_band_limit_hz),
        }
    elif normalized_type == "complex-bass":
        body, extra_meta, descriptor = _generate_complex_bass(
            sample_rate=sample_rate,
            samples=samples,
            lowcut=noise_lowcut,
            highcut=noise_highcut,
            rng=rng,
        )
    elif normalized_type == "binaural-cues":
        body, extra_meta, descriptor = _generate_binaural_cues(
            sample_rate=sample_rate,
            samples=samples,
            lowcut=noise_lowcut,
            highcut=noise_highcut,
            itd_ms=binaural_itd_ms,
            ild_db=binaural_ild_db,
            rng=rng,
        )
    elif normalized_type == "ms-side-texture":
        body, extra_meta, descriptor = _generate_ms_side_texture(
            sample_rate=sample_rate,
            samples=samples,
            min_freq_hz=min_tone_freq,
            tone_count=tone_count,
            tone_step_hz=tone_step,
            rng=rng,
        )
    else:  # tfs-tones
        body, freqs = _generate_tfs_tones(
            sample_rate=sample_rate,
            samples=samples,
            min_freq_hz=min_tone_freq,
            tone_count=tone_count,
            tone_step_hz=tone_step,
        )
        descriptor = f"{int(min_tone_freq)}hz_{tone_count}t"
        extra_meta = {
            "tones_hz": freqs,
            "tone_level_dbfs": -6.0,
        }

    timeline, duration_sec = _compose_timeline(body=body, common=common)
    channels = timeline.shape[1] if timeline.ndim == 2 else 1
    metadata = _build_metadata(
        signal_type=normalized_type,
        common=common,
        duration_sec=duration_sec,
        channels=channels,
        extra=extra_meta,
    )

    return SignalBuildResult(
        data=timeline,
        metadata=metadata,
        suggested_stem=default_output_stem(
            normalized_type, common=common, descriptor=descriptor
        ),
        channels=channels,
    )


def _compose_timeline(
    body: npt.NDArray[np.float64], common: CommonSignalConfig
) -> tuple[npt.NDArray[np.float64], float]:
    sr = common.sample_rate
    body_array = np.asarray(body, dtype=np.float64)
    pilot_mono = _generate_pilot(
        sample_rate=sr,
        freq=common.pilot_freq,
        duration_ms=common.pilot_duration_ms,
        level_dbfs=common.pilot_level_dbfs,
        fade_ms=common.fade_ms,
    )
    silence_samples = int(sr * common.silence_duration_ms / 1000)
    if body_array.ndim == 1:
        silence = np.zeros(silence_samples, dtype=np.float64)
        timeline = np.concatenate(
            [silence, pilot_mono, body_array, pilot_mono, silence]
        )
    elif body_array.ndim == 2:
        channels = body_array.shape[1]
        silence = np.zeros((silence_samples, channels), dtype=np.float64)
        pilot = np.tile(pilot_mono[:, None], (1, channels))
        timeline = np.concatenate([silence, pilot, body_array, pilot, silence], axis=0)
    else:
        raise ValueError("body must be 1D or 2D array")

    duration_sec = timeline.shape[0] / sr
    timeline = np.clip(timeline, -0.9999, 0.9999)
    return timeline.astype(np.float64), duration_sec


def _generate_pilot(
    *,
    sample_rate: int,
    freq: float,
    duration_ms: int,
    level_dbfs: float,
    fade_ms: int,
) -> npt.NDArray[np.float64]:
    samples = max(int(sample_rate * duration_ms / 1000), 1)
    t = np.arange(samples) / sample_rate
    tone = np.sin(2 * np.pi * freq * t)
    tone = _apply_fade(tone, fade_samples=int(sample_rate * fade_ms / 1000))
    return _scale_to_dbfs(tone, level_dbfs, mode="peak")


def _generate_sine(
    *, freq: float, sample_rate: int, samples: int
) -> npt.NDArray[np.float64]:
    t = np.arange(samples) / sample_rate
    return np.sin(2 * np.pi * freq * t)


def _generate_notched_noise(
    *,
    sample_rate: int,
    samples: int,
    centers_hz: list[float],
    q: float,
    cascade_stages: int,
    lowcut: float | None,
    highcut: float | None,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    base = _generate_pink_noise(
        sample_rate=sample_rate,
        samples=samples,
        lowcut=lowcut,
        highcut=highcut,
        rng=rng,
    )
    nyquist = sample_rate / 2
    if q <= 0:
        raise ValueError("Notch Q must be positive.")
    if cascade_stages < 1:
        raise ValueError("cascade_stages must be >= 1")
    if not centers_hz:
        raise ValueError("centers_hz must not be empty")
    filtered = base
    for center_hz in centers_hz:
        if not 0 < center_hz < nyquist:
            raise ValueError("Notch center must be within (0, Nyquist)")
    for _ in range(cascade_stages):
        for center_hz in centers_hz:
            b, a = signal.iirnotch(w0=float(center_hz) / nyquist, Q=float(q))
            filtered = signal.lfilter(b, a, filtered)
    return _scale_to_dbfs(filtered, -14.0, mode="rms")


def _generate_tone_burst(
    *,
    sample_rate: int,
    samples: int,
    freq_hz: float,
    cycles: int,
    fade_cycles: int,
) -> npt.NDArray[np.float64]:
    if cycles < 1:
        raise ValueError("cycles must be >= 1")
    if fade_cycles < 0:
        raise ValueError("fade_cycles must be >= 0")
    burst_samples = int(round((cycles / max(freq_hz, 1e-6)) * sample_rate))
    burst_samples = max(1, min(burst_samples, samples))
    t = np.arange(burst_samples) / sample_rate
    burst = np.sin(2 * np.pi * freq_hz * t)
    fade_samples = int(round((fade_cycles / max(freq_hz, 1e-6)) * sample_rate))
    burst = _apply_fade(burst, fade_samples=fade_samples)
    body = np.zeros(samples, dtype=np.float64)
    start = max(0, (samples - burst_samples) // 2)
    body[start : start + burst_samples] = burst
    return body


def _generate_click(
    *,
    sample_rate: int,
    samples: int,
    band_limit_hz: float,
) -> npt.NDArray[np.float64]:
    if samples <= 0:
        return np.zeros(0, dtype=np.float64)
    body = np.zeros(samples, dtype=np.float64)
    body[samples // 2] = 1.0
    nyquist = sample_rate / 2
    cutoff = min(max(10.0, band_limit_hz), nyquist * 0.95)
    sos = signal.butter(4, cutoff / nyquist, btype="low", output="sos")
    return np.asarray(signal.sosfiltfilt(sos, body), dtype=np.float64)


def _generate_complex_bass(
    *,
    sample_rate: int,
    samples: int,
    lowcut: float | None,
    highcut: float | None,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], dict[str, object], str]:
    min_freq = max(lowcut or 0.0, 25.0)
    max_freq = highcut if highcut not in (None, 0) else 220.0
    max_freq = min(max_freq, 260.0)
    if max_freq <= min_freq:
        raise ValueError("complex-bass requires highcut > lowcut.")

    tone_count = 8
    freqs = np.linspace(min_freq, max_freq, tone_count)
    t = np.arange(samples) / sample_rate
    phases = rng.uniform(0, 2 * np.pi, tone_count)
    fm_rates = rng.uniform(0.3, 1.1, tone_count)
    pm_rates = rng.uniform(0.4, 1.6, tone_count)
    fm_dev_hz = 3.0
    pm_depth_rad = 0.25

    components = []
    for idx, base_freq in enumerate(freqs):
        fm_rate = fm_rates[idx]
        fm_phase = rng.uniform(0, 2 * np.pi)
        pm_phase = rng.uniform(0, 2 * np.pi)
        beta = fm_dev_hz / max(fm_rate, 1e-3)
        phase = (
            2 * np.pi * base_freq * t
            + beta * np.sin(2 * np.pi * fm_rate * t + fm_phase)
            + pm_depth_rad * np.sin(2 * np.pi * pm_rates[idx] * t + pm_phase)
            + phases[idx]
        )
        components.append(np.sin(phase))
    body = np.sum(components, axis=0) / max(len(components), 1)
    body = _band_limit(
        data=body, sample_rate=sample_rate, lowcut=min_freq, highcut=max_freq
    )
    body = _scale_to_dbfs(body, -2.0, mode="peak")

    descriptor = f"{int(min_freq)}to{int(max_freq)}hz"
    extra_meta = {
        "bass_components_hz": [float(f) for f in freqs],
        "bass_fm_dev_hz": fm_dev_hz,
        "bass_fm_rates_hz": [float(f) for f in fm_rates],
        "bass_pm_depth_rad": pm_depth_rad,
        "bass_pm_rates_hz": [float(f) for f in pm_rates],
        "band_lowcut_hz": float(min_freq),
        "band_highcut_hz": float(max_freq),
        "target_peak_dbfs": -2.0,
    }
    return body, extra_meta, descriptor


def _apply_fractional_delay(
    data: npt.NDArray[np.float64], delay_samples: float
) -> npt.NDArray[np.float64]:
    if data.size == 0:
        return data
    idx = np.arange(data.size, dtype=np.float64)
    delayed = np.interp(idx - delay_samples, idx, data, left=0.0, right=0.0)
    return delayed.astype(np.float64)


def _generate_binaural_cues(
    *,
    sample_rate: int,
    samples: int,
    lowcut: float | None,
    highcut: float | None,
    itd_ms: float,
    ild_db: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], dict[str, object], str]:
    nyquist = sample_rate / 2
    base_low = max(lowcut or 0.0, 150.0)
    base_high = highcut if highcut not in (None, 0) else nyquist * 0.95
    base_high = min(base_high, nyquist * 0.95)
    if base_high <= base_low:
        raise ValueError("binaural-cues requires highcut > lowcut.")

    base = _generate_pink_noise(
        sample_rate=sample_rate,
        samples=samples,
        lowcut=base_low,
        highcut=base_high,
        rng=rng,
    )
    itd_samples = itd_ms * sample_rate / 1000.0
    left = base
    right = _apply_fractional_delay(base, itd_samples)

    ild_abs = abs(float(ild_db))
    if ild_db >= 0:
        left_gain = 1.0
        right_gain = _dbfs_to_amplitude(-ild_abs)
    else:
        left_gain = _dbfs_to_amplitude(-ild_abs)
        right_gain = 1.0
    stereo = np.column_stack([left_gain * left, right_gain * right])
    stereo = _scale_to_dbfs(stereo, -3.0, mode="peak")

    descriptor = f"itd{itd_ms:g}ms_ild{ild_abs:g}db"
    extra_meta = {
        "itd_ms": float(itd_ms),
        "ild_db": float(ild_db),
        "base_noise_lowcut_hz": float(base_low),
        "base_noise_highcut_hz": float(base_high),
        "target_peak_dbfs": -3.0,
    }
    return stereo, extra_meta, descriptor


def _generate_ms_side_texture(
    *,
    sample_rate: int,
    samples: int,
    min_freq_hz: float,
    tone_count: int,
    tone_step_hz: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], dict[str, object], str]:
    mid = _generate_pink_noise(
        sample_rate=sample_rate,
        samples=samples,
        lowcut=80.0,
        highcut=3200.0,
        rng=rng,
    )
    mid = _scale_to_dbfs(mid, -10.0, mode="rms")

    side_min = max(min_freq_hz, 4000.0)
    freqs = [side_min + i * tone_step_hz for i in range(max(tone_count, 1))]
    nyquist = sample_rate / 2
    freqs = [f for f in freqs if f < nyquist * 0.95]
    if not freqs:
        raise ValueError("ms-side-texture requires side frequencies below Nyquist.")

    t = np.arange(samples) / sample_rate
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    mod_freq_hz = 5.0
    mod_depth = 0.35
    mod_phase = rng.uniform(0, 2 * np.pi)
    side_components = [
        np.sin(2 * np.pi * f * t + phases[idx]) for idx, f in enumerate(freqs)
    ]
    side_raw = np.sum(side_components, axis=0) / max(len(side_components), 1)
    modulation = 1.0 + mod_depth * np.sin(2 * np.pi * mod_freq_hz * t + mod_phase)
    side = side_raw * modulation
    side = _band_limit(
        data=side,
        sample_rate=sample_rate,
        lowcut=side_min * 0.9,
        highcut=min(freqs[-1] * 1.3, nyquist * 0.95),
    )
    side = _scale_to_dbfs(side, -6.0, mode="peak")

    left = 0.5 * (mid + side)
    right = 0.5 * (mid - side)
    stereo = np.column_stack([left, right])
    stereo = _scale_to_dbfs(stereo, -3.0, mode="peak")

    descriptor = f"side{int(side_min)}hz"
    extra_meta = {
        "mid_band_lowcut_hz": 80.0,
        "mid_band_highcut_hz": 3200.0,
        "side_tones_hz": [float(f) for f in freqs],
        "side_mod_freq_hz": mod_freq_hz,
        "side_mod_depth": mod_depth,
        "side_target_peak_dbfs": -6.0,
        "target_peak_dbfs": -3.0,
    }
    return stereo, extra_meta, descriptor


def _generate_am_attack(
    *,
    sample_rate: int,
    samples: int,
    carrier_hz: float,
    attack_ms: float,
    release_ms: float,
    period_ms: float,
) -> npt.NDArray[np.float64]:
    if attack_ms <= 0 or release_ms <= 0 or period_ms <= 0:
        raise ValueError("attack_ms/release_ms/period_ms must be positive")
    t = np.arange(samples) / sample_rate
    carrier = np.sin(2 * np.pi * carrier_hz * t)
    period_s = period_ms / 1000.0
    atk_s = attack_ms / 1000.0
    rel_s = release_ms / 1000.0
    phase = np.mod(t, period_s)
    # Gate: off -> attack ramp -> on -> release ramp -> off
    on_s = max(period_s - atk_s - rel_s, 0.0)
    env = np.zeros_like(t, dtype=np.float64)
    atk_mask = phase < atk_s
    env[atk_mask] = phase[atk_mask] / atk_s
    on_mask = (phase >= atk_s) & (phase < atk_s + on_s)
    env[on_mask] = 1.0
    rel_mask = (phase >= atk_s + on_s) & (phase < atk_s + on_s + rel_s)
    env[rel_mask] = 1.0 - (phase[rel_mask] - (atk_s + on_s)) / rel_s
    return env * carrier


def _generate_pink_noise(
    *,
    sample_rate: int,
    samples: int,
    lowcut: float | None,
    highcut: float | None,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    white = rng.standard_normal(samples)
    b = np.array(
        [0.049922035, -0.095993537, 0.050612699, -0.004408786], dtype=np.float64
    )
    a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400], dtype=np.float64)
    pink = signal.lfilter(b, a, white)
    pink = _band_limit(
        data=pink,
        sample_rate=sample_rate,
        lowcut=lowcut,
        highcut=highcut,
    )
    return _scale_to_dbfs(pink, -14.0, mode="rms")


def _generate_modulated(
    *,
    sample_rate: int,
    samples: int,
    carrier_hz: float,
    am_freq_hz: float,
    am_depth_ratio: float,
    fm_dev_hz: float,
    fm_freq_hz: float,
) -> npt.NDArray[np.float64]:
    t = np.arange(samples) / sample_rate
    am = 1.0 + am_depth_ratio * np.sin(2 * np.pi * am_freq_hz * t)
    beta = fm_dev_hz / max(fm_freq_hz, 1e-6)
    phase = 2 * np.pi * carrier_hz * t + beta * np.sin(2 * np.pi * fm_freq_hz * t)
    carrier = np.sin(phase)
    modulated = am * carrier
    return _scale_to_dbfs(modulated, -6.0, mode="peak")


def _generate_tfs_tones(
    *,
    sample_rate: int,
    samples: int,
    min_freq_hz: float,
    tone_count: int,
    tone_step_hz: float,
) -> tuple[npt.NDArray[np.float64], list[float]]:
    freqs = [min_freq_hz + i * tone_step_hz for i in range(tone_count)]
    nyquist = sample_rate / 2
    for f in freqs:
        if f >= nyquist:
            raise ValueError("Tone frequency must be below Nyquist frequency.")
    t = np.arange(samples) / sample_rate
    tones = [np.sin(2 * np.pi * f * t) for f in freqs]
    summed = np.sum(tones, axis=0) / max(len(tones), 1)
    return _scale_to_dbfs(summed, -6.0, mode="peak"), freqs


def _band_limit(
    *,
    data: npt.NDArray[np.float64],
    sample_rate: int,
    lowcut: float | None,
    highcut: float | None,
) -> npt.NDArray[np.float64]:
    nyquist = sample_rate / 2
    low = lowcut if lowcut not in (None, 0) else None
    high = highcut if highcut not in (None, 0) else None

    if high is not None and high >= nyquist:
        high = nyquist * 0.99
    if low is None and high is None:
        return data
    if low is None:
        assert high is not None
        sos = signal.butter(4, high / nyquist, btype="low", output="sos")
    elif high is None:
        sos = signal.butter(4, low / nyquist, btype="high", output="sos")
    else:
        sos = signal.butter(
            4, [low / nyquist, high / nyquist], btype="band", output="sos"
        )
    return np.asarray(signal.sosfilt(sos, data), dtype=np.float64)


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


def _scale_to_dbfs(
    data: npt.NDArray[np.float64], target_dbfs: float, *, mode: str
) -> npt.NDArray[np.float64]:
    if data.size == 0:
        return data
    if mode not in {"peak", "rms"}:
        raise ValueError("mode must be 'peak' or 'rms'")
    if mode == "peak":
        reference = np.max(np.abs(data))
    else:
        reference = float(np.sqrt(np.mean(np.square(data))))
    if reference == 0:
        return data
    target = _dbfs_to_amplitude(target_dbfs)
    scaled = data * (target / reference)
    return np.asarray(scaled, dtype=np.float64)


def _dbfs_to_amplitude(dbfs: float) -> float:
    return 10 ** (dbfs / 20)


def _build_metadata(
    *,
    signal_type: str,
    common: CommonSignalConfig,
    duration_sec: float,
    channels: int,
    extra: dict[str, object],
) -> dict[str, object]:
    return {
        "signal_type": signal_type.replace("-", "_"),
        "sample_rate": common.sample_rate,
        "bit_depth": common.normalized_bit_depth(),
        "channels": channels,
        "duration_sec": duration_sec,
        "pilot_tone_freq_hz": common.pilot_freq,
        "pilot_duration_ms": common.pilot_duration_ms,
        "pilot_level_dbfs": common.pilot_level_dbfs,
        "lead_silence_ms": common.silence_duration_ms,
        "tail_silence_ms": common.silence_duration_ms,
        "created_at": datetime.now(UTC).isoformat(),
        "version": common.version,
        **extra,
    }
