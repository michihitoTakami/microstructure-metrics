from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from click.testing import CliRunner

from microstructure_metrics.cli import main


def _pilot_wave(*, sample_rate: int, freq: float, duration_ms: int) -> np.ndarray:
    samples = max(int(sample_rate * duration_ms / 1000), 1)
    t = np.arange(samples) / sample_rate
    return np.sin(2 * np.pi * freq * t).astype(np.float64)


def _synthetic(
    *,
    sample_rate: int = 48_000,
    pilot_freq: float = 1000.0,
    pilot_ms: int = 100,
    silence_ms: int = 500,
    body_duration: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(1)
    pilot = _pilot_wave(sample_rate=sample_rate, freq=pilot_freq, duration_ms=pilot_ms)
    silence = np.zeros(int(sample_rate * silence_ms / 1000), dtype=np.float64)
    body = rng.standard_normal(int(sample_rate * body_duration)) * 0.05
    return np.concatenate([silence, pilot, body, pilot, silence])


def _insert_samples(
    data: np.ndarray, *, insert_at: int, samples: int, fill: float = 0.0
) -> np.ndarray:
    padding = np.full(samples, fill, dtype=np.float64)
    return np.concatenate([data[:insert_at], padding, data[insert_at:]])


def test_cli_drift_none(tmp_path: Path) -> None:
    runner = CliRunner()
    ref = _synthetic(body_duration=1.0)
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, ref, 48_000)
    sf.write(dut_path, ref, 48_000)

    result = runner.invoke(
        main,
        ["drift", str(ref_path), str(dut_path)],
    )

    assert result.exit_code == 0, result.output
    assert "severity: none" in result.output


def test_cli_drift_strict_high(tmp_path: Path) -> None:
    runner = CliRunner()
    ref = _synthetic(body_duration=2.0)
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, ref, 48_000)

    # 挿入してドリフトを発生させる（約80ppm超）
    silence_samples = int(48_000 * 0.5)
    pilot_samples = int(48_000 * 0.1)
    insert_at = silence_samples + pilot_samples + int(48_000 * 2.0)
    dut = _insert_samples(ref, insert_at=insert_at, samples=32)
    sf.write(dut_path, dut, 48_000)

    json_path = tmp_path / "drift.json"
    result = runner.invoke(
        main,
        [
            "drift",
            str(ref_path),
            str(dut_path),
            "--strict",
            "--json-output",
            str(json_path),
        ],
    )

    assert result.exit_code != 0
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["severity"] in {"high", "critical"}
    assert payload["drift_ppm"] > 20
