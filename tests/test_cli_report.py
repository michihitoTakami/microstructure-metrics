from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from click.testing import CliRunner

from microstructure_metrics.cli import main
from microstructure_metrics.signals import CommonSignalConfig, build_signal


def _delayed_pair(common: CommonSignalConfig) -> tuple[np.ndarray, np.ndarray, int]:
    reference = build_signal("thd", common=common).data
    delay_samples = int(0.002 * common.sample_rate)
    rng = np.random.default_rng(0)
    dut = np.concatenate([np.zeros(delay_samples), reference])
    dut = dut + 1e-4 * rng.standard_normal(dut.shape[0])
    return reference, dut, delay_samples


def test_cli_report_outputs_json_csv_md(tmp_path: Path) -> None:
    common = CommonSignalConfig(duration=0.4)
    ref, dut, delay_samples = _delayed_pair(common)
    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, ref, samplerate=common.sample_rate)
    sf.write(dut_path, dut, samplerate=common.sample_rate)

    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    md_path = tmp_path / "report.md"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "report",
            str(ref_path),
            str(dut_path),
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
            "--output-md",
            str(md_path),
            "--expected-level-dbfs",
            "-6",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    for key in ["thd_n", "transient", "mps", "tfs"]:
        assert key in payload["metrics"]
    tfs_payload = payload["metrics"]["tfs"]
    for key in [
        "percentile_05_correlation",
        "correlation_variance",
        "frames_per_band",
        "frame_length_ms",
        "frame_hop_ms",
        "max_lag_ms",
        "envelope_threshold_db",
    ]:
        assert key in tfs_payload
    assert tfs_payload["frames_per_band"] > 0
    assert abs(payload["alignment"]["delay_samples"] - delay_samples) < 10

    assert csv_path.exists()
    csv_body = csv_path.read_text()
    assert "thd_n" in csv_body
    assert md_path.exists()
    assert "Microstructure Metrics Report" in md_path.read_text()


def test_cli_report_no_align_with_resample(tmp_path: Path) -> None:
    # ref: 48k, dut: 44.1k -> allow_resample により自動で合わせる
    ref_common = CommonSignalConfig(sample_rate=48_000, duration=0.3)
    dut_common = CommonSignalConfig(sample_rate=44_100, duration=0.3)
    ref = build_signal("thd", common=ref_common).data
    dut = build_signal("thd", common=dut_common).data

    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, ref, samplerate=ref_common.sample_rate)
    sf.write(dut_path, dut, samplerate=dut_common.sample_rate)

    json_path = tmp_path / "report_no_align.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "report",
            str(ref_path),
            str(dut_path),
            "--no-align",
            "--allow-resample",
            "--output-json",
            str(json_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(json_path.read_text())
    # no-align でも metrics が生成され、alignment は delay 0 のダミー
    assert payload["alignment"]["delay_samples"] == 0.0
    assert payload["metrics"]["thd_n"]


def test_cli_report_fails_without_resample_permission(tmp_path: Path) -> None:
    # SR が異なるのに --allow-resample を指定しない場合はエラーになる
    ref_common = CommonSignalConfig(sample_rate=48_000, duration=0.2)
    dut_common = CommonSignalConfig(sample_rate=44_100, duration=0.2)
    ref = build_signal("thd", common=ref_common).data
    dut = build_signal("thd", common=dut_common).data

    ref_path = tmp_path / "ref.wav"
    dut_path = tmp_path / "dut.wav"
    sf.write(ref_path, ref, samplerate=ref_common.sample_rate)
    sf.write(dut_path, dut, samplerate=dut_common.sample_rate)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "report",
            str(ref_path),
            str(dut_path),
        ],
    )

    assert result.exit_code != 0
    assert "Sample rate mismatch" in result.output
