from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from microstructure_metrics.io import load_audio_pair


def _sine(duration: float, sample_rate: int, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return amplitude * np.sin(2 * np.pi * 1000 * t)


def test_load_audio_pair_returns_validation_and_metadata(tmp_path: Path) -> None:
    sr = 48_000
    ref = _sine(0.2, sr, amplitude=0.5)
    dut = _sine(0.2, sr, amplitude=0.4)
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, ref, sr, subtype="PCM_24")
    sf.write(dut_path, dut, sr, subtype="PCM_24")

    ref_out, dut_out, validation = load_audio_pair(ref_path, dut_path)

    assert ref_out.shape == dut_out.shape
    assert validation.is_valid
    assert validation.errors == []
    assert validation.metadata_ref.sample_rate == sr
    assert validation.metadata_ref.bit_depth == 24
    assert validation.metadata_ref.channels == 1
    assert np.isclose(validation.metadata_ref.peak_amplitude, 0.5, atol=1e-3)
    assert np.isclose(validation.metadata_dut.peak_amplitude, 0.4, atol=1e-3)
    assert not validation.warnings


def test_sample_rate_mismatch_requires_resample_option(tmp_path: Path) -> None:
    ref = _sine(0.1, 48_000, amplitude=0.5)
    dut = _sine(0.1, 44_100, amplitude=0.4)
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, ref, 48_000, subtype="PCM_24")
    sf.write(dut_path, dut, 44_100, subtype="PCM_24")

    with pytest.raises(ValueError):
        load_audio_pair(ref_path, dut_path)

    _, _, validation = load_audio_pair(
        ref_path, dut_path, allow_resample=True, target_sample_rate=48_000
    )
    assert validation.is_valid
    assert any("resampled" in w for w in validation.warnings)
    assert validation.metadata_ref.sample_rate == 48_000
    assert validation.metadata_dut.sample_rate == 48_000


def test_bit_depth_mismatch_is_warning(tmp_path: Path) -> None:
    sr = 48_000
    ref = _sine(0.1, sr, amplitude=0.5)
    dut = _sine(0.1, sr, amplitude=0.5)
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, ref, sr, subtype="PCM_24")
    sf.write(dut_path, dut, sr, subtype="PCM_16")

    _, _, validation = load_audio_pair(ref_path, dut_path)

    assert validation.is_valid
    assert any("Bit depth differs" in w for w in validation.warnings)


def test_quality_checks_detect_clipping_dc_and_silence(tmp_path: Path) -> None:
    sr = 48_000
    duration = 0.05
    ref = 0.02 + _sine(duration, sr, amplitude=0.5)  # DC offset present
    dut = np.zeros(int(duration * sr), dtype=np.float64)
    dut[:10] = 1.0  # clipping present on a few samples
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, ref, sr, subtype="PCM_24")
    sf.write(dut_path, dut, sr, subtype="PCM_24")

    _, _, validation = load_audio_pair(
        ref_path,
        dut_path,
        remove_dc=False,  # keep DC to ensure detection
        normalize=None,
        silence_threshold=0.05,  # force low-level warning for mostly silent DUT
    )

    warnings = "\n".join(validation.warnings)
    assert "DC offset" in warnings
    assert "clipping" in warnings
    assert "very low level" in warnings  # dut is mostly silent


def test_channel_selection_defaults_to_left_with_warning(tmp_path: Path) -> None:
    sr = 48_000
    duration = 0.1
    left = _sine(duration, sr, amplitude=0.3)
    right = np.full_like(left, 0.1)
    stereo = np.stack([left, right], axis=1)
    dut = _sine(duration, sr, amplitude=0.25)
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, stereo, sr, subtype="PCM_24")
    sf.write(dut_path, dut, sr, subtype="PCM_24")

    ref_out, _, validation = load_audio_pair(ref_path, dut_path)

    assert np.allclose(ref_out, left, atol=1e-9)
    assert any("stereo input" in w for w in validation.warnings)


def test_channel_selection_with_index_and_bounds(tmp_path: Path) -> None:
    sr = 48_000
    duration = 0.05
    ch0 = _sine(duration, sr, amplitude=0.1)
    ch1 = np.full_like(ch0, 0.2)
    ch2 = np.full_like(ch0, -0.05)
    multi = np.stack([ch0, ch1, ch2], axis=1)
    dut = _sine(duration, sr, amplitude=0.25)
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, multi, sr, subtype="PCM_24")
    sf.write(dut_path, dut, sr, subtype="PCM_24")

    ref_out, _, validation = load_audio_pair(
        ref_path, dut_path, channel=1, remove_dc=False
    )
    assert np.allclose(ref_out, ch1, atol=1e-9)
    assert not any("stereo input" in w for w in validation.warnings)

    with pytest.raises(ValueError):
        load_audio_pair(ref_path, dut_path, channel=5)


def test_channels_option_stereo_mid_side(tmp_path: Path) -> None:
    sr = 48_000
    duration = 0.1
    left = _sine(duration, sr, amplitude=0.3)
    right = _sine(duration, sr, amplitude=0.15)
    stereo = np.stack([left, right], axis=1)
    dut = stereo * 0.9
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, stereo, sr, subtype="PCM_24")
    sf.write(dut_path, dut, sr, subtype="PCM_24")

    ref_stereo, dut_stereo, validation = load_audio_pair(
        ref_path, dut_path, channels="stereo"
    )
    assert ref_stereo.shape[1] == 2
    assert dut_stereo.shape == ref_stereo.shape
    assert validation.metadata_ref.channels == 2

    ref_mid, _, validation_mid = load_audio_pair(ref_path, dut_path, channels="mid")
    expected_mid = 0.5 * (left + right)
    assert ref_mid.ndim == 1
    assert np.allclose(ref_mid, expected_mid, atol=1e-12)
    assert validation_mid.metadata_ref.channels == 1

    ref_side, _, _ = load_audio_pair(ref_path, dut_path, channels="side")
    expected_side = 0.5 * (left - right)
    assert np.allclose(ref_side, expected_side, atol=1e-12)


def test_channels_option_conflict_with_channel(tmp_path: Path) -> None:
    sr = 48_000
    data = _sine(0.05, sr, amplitude=0.2)
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, data, sr, subtype="PCM_24")
    sf.write(dut_path, data, sr, subtype="PCM_24")

    with pytest.raises(ValueError):
        load_audio_pair(ref_path, dut_path, channel=0, channels="stereo")


def test_normalize_peak_and_rms(tmp_path: Path) -> None:
    sr = 48_000
    dur = 0.1
    ref = _sine(dur, sr, amplitude=0.2)
    dut = _sine(dur, sr, amplitude=0.4)
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, ref, sr, subtype="PCM_24")
    sf.write(dut_path, dut, sr, subtype="PCM_24")

    ref_out, dut_out, _ = load_audio_pair(
        ref_path, dut_path, normalize="peak", normalize_target=0.8
    )
    assert np.isclose(np.max(np.abs(ref_out)), 0.8, atol=1e-3)
    assert np.isclose(np.max(np.abs(dut_out)), 0.8, atol=1e-3)

    ref_out, dut_out, _ = load_audio_pair(
        ref_path, dut_path, normalize="rms", normalize_target=0.1
    )
    rms_ref = float(np.sqrt(np.mean(np.square(ref_out))))
    rms_dut = float(np.sqrt(np.mean(np.square(dut_out))))
    assert np.isclose(rms_ref, 0.1, atol=1e-3)
    assert np.isclose(rms_dut, 0.1, atol=1e-3)


def test_dc_removal_highpass(tmp_path: Path) -> None:
    sr = 48_000
    dur = 0.1
    t = np.arange(int(dur * sr)) / sr
    ref = 0.05 + 0.2 * np.sin(2 * np.pi * 5 * t)  # DC + very low freq
    dut = _sine(dur, sr, amplitude=0.2)
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, ref, sr, subtype="PCM_24")
    sf.write(dut_path, dut, sr, subtype="PCM_24")

    original_mean = float(np.mean(ref))
    ref_out, _, _ = load_audio_pair(
        ref_path, dut_path, dc_method="highpass", dc_cutoff_hz=20.0
    )
    filtered_mean = abs(float(np.mean(ref_out)))
    assert filtered_mean < original_mean * 0.1


def test_resample_preserves_length_and_warns(tmp_path: Path) -> None:
    ref_sr = 44_100
    target_sr = 48_000
    dur = 0.1
    ref = _sine(dur, ref_sr, amplitude=0.3)
    dut = _sine(dur, target_sr, amplitude=0.3)
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, ref, ref_sr, subtype="PCM_24")
    sf.write(dut_path, dut, target_sr, subtype="PCM_24")

    _, _, validation = load_audio_pair(
        ref_path, dut_path, allow_resample=True, target_sample_rate=target_sr
    )
    assert validation.metadata_ref.sample_rate == target_sr
    assert validation.metadata_dut.sample_rate == target_sr
    assert any("resampled" in w for w in validation.warnings)

    # confirm length matches expected target samples
    expected = int(dur * target_sr)
    assert validation.metadata_ref.duration_sec == pytest.approx(dur, rel=1e-6)
    assert validation.metadata_dut.duration_sec == pytest.approx(dur, rel=1e-6)
    assert expected == int(validation.metadata_ref.duration_sec * target_sr)


def test_target_sample_rate_without_flag_resamples_and_warns(tmp_path: Path) -> None:
    ref_sr = 44_100
    target_sr = 48_000
    dur = 0.05
    ref = _sine(dur, ref_sr, amplitude=0.2)
    dut = _sine(dur, target_sr, amplitude=0.2)
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, ref, ref_sr, subtype="PCM_24")
    sf.write(dut_path, dut, target_sr, subtype="PCM_24")

    _, _, validation = load_audio_pair(ref_path, dut_path, target_sample_rate=target_sr)
    assert validation.metadata_ref.sample_rate == target_sr
    assert validation.metadata_dut.sample_rate == target_sr
    assert any("resampled" in w for w in validation.warnings)


def test_custom_thresholds_for_warnings(tmp_path: Path) -> None:
    sr = 48_000
    dur = 0.05
    low = _sine(dur, sr, amplitude=0.05)
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, low, sr, subtype="PCM_24")
    sf.write(dut_path, low, sr, subtype="PCM_24")

    _, _, validation = load_audio_pair(
        ref_path,
        dut_path,
        silence_threshold=0.2,
        clip_warn_threshold=0.2,
    )
    warnings = "\n".join(validation.warnings)
    assert "very low level" in warnings
    assert "possible clipping" not in warnings


@pytest.mark.parametrize(
    "subtype,expected_depth",
    [("PCM_16", 16), ("PCM_24", 24), ("FLOAT", 32)],
)
def test_bit_depth_mapping(tmp_path: Path, subtype: str, expected_depth: int) -> None:
    sr = 48_000
    dur = 0.02
    data = _sine(dur, sr, amplitude=0.2)
    ref_path = tmp_path / f"ref_{subtype}.wav"
    dut_path = tmp_path / f"dut_{subtype}.wav"
    sf.write(ref_path, data, sr, subtype=subtype)
    sf.write(dut_path, data, sr, subtype=subtype)

    _, _, validation = load_audio_pair(ref_path, dut_path)
    assert validation.metadata_ref.bit_depth == expected_depth
    assert validation.metadata_dut.bit_depth == expected_depth


def test_duration_bounds_warnings(tmp_path: Path) -> None:
    sr = 48_000
    short = _sine(0.01, sr, amplitude=0.2)
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, short, sr, subtype="PCM_24")
    sf.write(dut_path, short, sr, subtype="PCM_24")

    _, _, validation = load_audio_pair(ref_path, dut_path, min_duration=0.05)
    assert any("shorter than 0.05" in w for w in validation.warnings)
