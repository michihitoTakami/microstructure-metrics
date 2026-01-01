from __future__ import annotations

from pathlib import Path

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
