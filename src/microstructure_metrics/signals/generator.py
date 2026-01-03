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
    metadata = _build_metadata(
        signal_type=normalized_type,
        common=common,
        duration_sec=duration_sec,
        extra=extra_meta,
    )

    return SignalBuildResult(
        data=timeline,
        metadata=metadata,
        suggested_stem=default_output_stem(
            normalized_type, common=common, descriptor=descriptor
        ),
    )


def _compose_timeline(
    body: npt.NDArray[np.float64], common: CommonSignalConfig
) -> tuple[npt.NDArray[np.float64], float]:
    sr = common.sample_rate
    pilot = _generate_pilot(
        sample_rate=sr,
        freq=common.pilot_freq,
        duration_ms=common.pilot_duration_ms,
        level_dbfs=common.pilot_level_dbfs,
        fade_ms=common.fade_ms,
    )
    silence_samples = int(sr * common.silence_duration_ms / 1000)
    silence = np.zeros(silence_samples, dtype=np.float64)

    timeline = np.concatenate([silence, pilot, body, pilot, silence])
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
    extra: dict[str, object],
) -> dict[str, object]:
    return {
        "signal_type": signal_type.replace("-", "_"),
        "sample_rate": common.sample_rate,
        "bit_depth": common.normalized_bit_depth(),
        "channels": 1,
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
