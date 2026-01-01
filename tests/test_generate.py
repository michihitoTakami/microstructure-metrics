from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from microstructure_metrics.cli import main


def _expected_samples(
    *,
    sample_rate: int,
    body_duration: float,
    pilot_ms: int,
    silence_ms: int,
) -> int:
    body_samples = int(body_duration * sample_rate)
    pilot_samples = int(sample_rate * pilot_ms / 1000)
    silence_samples = int(sample_rate * silence_ms / 1000)
    return body_samples + 2 * (pilot_samples + silence_samples)


def _pilot_rms_dbfs(
    data, *, sample_rate: int, pilot_ms: int, silence_ms: int, margin_ms: int = 20
) -> float:
    pilot_samples = int(sample_rate * pilot_ms / 1000)
    silence_samples = int(sample_rate * silence_ms / 1000)
    start = silence_samples + int(sample_rate * margin_ms / 1000)
    end = start + max(pilot_samples - 2 * int(sample_rate * margin_ms / 1000), 1)
    pilot_seg = data[start:end]
    rms = (pilot_seg.astype("float64") ** 2).mean() ** 0.5
    rms = max(rms, 1e-12)
    return 20 * float(np.log10(rms))


def test_generate_thd_creates_wav_and_json(tmp_path: Path) -> None:
    runner = CliRunner()
    wav_path = tmp_path / "thd.wav"
    result = runner.invoke(
        main,
        [
            "generate",
            "thd",
            "--duration",
            "0.5",
            "--freq",
            "1000",
            "--output",
            str(wav_path),
            "--with-metadata",
        ],
    )

    assert result.exit_code == 0, result.output
    assert wav_path.exists()
    assert wav_path.with_suffix(".json").exists()

    data, sample_rate = sf.read(wav_path)
    assert sample_rate == 48000
    assert data.ndim == 1
    expected = _expected_samples(
        sample_rate=sample_rate,
        body_duration=0.5,
        pilot_ms=100,
        silence_ms=500,
    )
    assert data.shape[0] == expected
    pilot_dbfs = _pilot_rms_dbfs(
        data, sample_rate=sample_rate, pilot_ms=100, silence_ms=500
    )
    assert -10.0 <= pilot_dbfs <= -8.0


def test_generate_tfs_tones_defaults(tmp_path: Path) -> None:
    runner = CliRunner()
    wav_path = tmp_path / "tfs.wav"
    result = runner.invoke(
        main,
        [
            "generate",
            "tfs-tones",
            "--duration",
            "0.25",
            "--tone-count",
            "3",
            "--output",
            str(wav_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert wav_path.exists()

    data, sample_rate = sf.read(wav_path)
    expected = _expected_samples(
        sample_rate=sample_rate,
        body_duration=0.25,
        pilot_ms=100,
        silence_ms=500,
    )
    assert data.shape[0] == expected
    assert data.ndim == 1


@pytest.mark.parametrize(
    "signal_type,opts",
    [
        ("notched-noise", {"--center": "8000", "--q": "8.6"}),
        ("pink-noise", {"--highcut": "20000"}),
        (
            "modulated",
            {
                "--carrier": "1000",
                "--am-freq": "4",
                "--am-depth": "0.5",
                "--fm-dev": "50",
            },
        ),
    ],
)
def test_generate_other_signals_structure_and_pilot(tmp_path: Path, signal_type, opts):
    runner = CliRunner()
    wav_path = tmp_path / f"{signal_type}.wav"
    args = [
        "generate",
        signal_type,
        "--duration",
        "0.3",
        "--output",
        str(wav_path),
        "--with-metadata",
    ]
    for k, v in opts.items():
        args.extend([k, v])
    result = runner.invoke(main, args)
    assert result.exit_code == 0, result.output

    data, sample_rate = sf.read(wav_path)
    expected = _expected_samples(
        sample_rate=sample_rate, body_duration=0.3, pilot_ms=100, silence_ms=500
    )
    assert data.shape[0] == expected
    assert data.ndim == 1

    pilot_dbfs = _pilot_rms_dbfs(
        data, sample_rate=sample_rate, pilot_ms=100, silence_ms=500
    )
    assert -10.0 <= pilot_dbfs <= -8.0

    metadata = sf.info(wav_path)
    assert metadata.samplerate == 48000
