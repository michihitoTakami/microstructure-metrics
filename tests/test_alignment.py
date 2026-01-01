from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from click.testing import CliRunner
from numpy.testing import assert_allclose

from microstructure_metrics.alignment import (
    align_audio_pair,
    align_signals,
    detect_pilot_tones,
    estimate_delay,
    extract_test_segment,
)
from microstructure_metrics.cli import main
from microstructure_metrics.signals import CommonSignalConfig, build_signal


def test_detect_pilot_tones_finds_two_segments() -> None:
    common = CommonSignalConfig(duration=0.4)
    signal_build = build_signal("thd", common=common)
    result = detect_pilot_tones(
        signal_build.data,
        sample_rate=common.sample_rate,
        pilot_freq=common.pilot_freq,
    )

    assert result.first_end - result.first_start > int(0.08 * common.sample_rate)
    assert result.second_start > result.first_end
    assert 0.3 <= result.confidence <= 1.0


def test_extract_test_segment_matches_body_length() -> None:
    common = CommonSignalConfig(duration=0.5)
    signal_build = build_signal("thd", common=common)
    pilot = detect_pilot_tones(
        signal_build.data,
        sample_rate=common.sample_rate,
        pilot_freq=common.pilot_freq,
    )
    margin_ms = 5.0
    segment = extract_test_segment(
        signal_build.data,
        pilot,
        sample_rate=common.sample_rate,
        margin_ms=margin_ms,
    )

    expected_body = int(common.sample_rate * common.duration)
    margin_samples = int(common.sample_rate * margin_ms / 1000)
    assert abs(segment.shape[0] - (expected_body - 2 * margin_samples)) < 1500


def test_estimate_delay_recovers_integer_shift() -> None:
    sample_rate = 48000
    t = np.arange(sample_rate // 2) / sample_rate
    reference = np.sin(2 * np.pi * 440 * t)
    shift = 120
    dut = np.concatenate([np.zeros(shift), reference])

    delay = estimate_delay(
        reference=reference,
        dut=dut,
        sample_rate=sample_rate,
        max_lag_ms=10.0,
    )
    assert_allclose(delay, shift, atol=1.0)


def test_align_audio_pair_end_to_end() -> None:
    common = CommonSignalConfig(duration=0.3)
    reference_build = build_signal("thd", common=common)
    reference = reference_build.data
    delay_samples = 90
    rng = np.random.default_rng(0)
    dut = np.concatenate([np.zeros(delay_samples), reference])
    dut = dut + 1e-3 * rng.standard_normal(dut.shape[0])

    result = align_audio_pair(
        reference=reference,
        dut=dut,
        sample_rate=common.sample_rate,
        pilot_freq=common.pilot_freq,
        margin_ms=5.0,
        max_lag_ms=20.0,
    )

    assert_allclose(result.delay_samples, delay_samples, atol=2.0)
    assert result.aligned_ref.shape == result.aligned_dut.shape
    correlation = np.corrcoef(result.aligned_ref, result.aligned_dut)[0, 1]
    assert correlation > 0.95


def test_align_signals_trims_to_shorter_after_shift() -> None:
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dut = np.concatenate([np.zeros(2), reference])
    aligned_ref, aligned_dut = align_signals(reference, dut, delay_samples=2.0)
    assert aligned_ref.shape == aligned_dut.shape == (5,)
    assert_allclose(aligned_ref, aligned_dut)


def test_cli_align_outputs_files(tmp_path: Path) -> None:
    common = CommonSignalConfig(duration=0.25)
    reference = build_signal("thd", common=common).data
    delay_samples = 60
    rng = np.random.default_rng(1)
    dut = np.concatenate([np.zeros(delay_samples), reference])
    dut = dut + 1e-3 * rng.standard_normal(dut.shape[0])

    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, reference, samplerate=common.sample_rate)
    sf.write(dut_path, dut, samplerate=common.sample_rate)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "align",
            str(ref_path),
            str(dut_path),
            "--max-lag-ms",
            "50",
            "--margin-ms",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output

    out_ref = ref_path.with_suffix(".aligned_ref.wav")
    out_dut = dut_path.with_suffix(".aligned_dut.wav")
    meta_path = ref_path.with_name(f"{ref_path.stem}_alignment.json")
    assert out_ref.exists() and out_dut.exists() and meta_path.exists()

    aligned_ref, sr_ref = sf.read(out_ref)
    aligned_dut, sr_dut = sf.read(out_dut)
    assert sr_ref == sr_dut == common.sample_rate
    assert aligned_ref.shape == aligned_dut.shape
    corr = np.corrcoef(aligned_ref, aligned_dut)[0, 1]
    assert corr > 0.9

    meta = json.loads(meta_path.read_text())
    assert meta["sample_rate"] == common.sample_rate
    assert abs(meta["delay_samples"] - delay_samples) < 5
