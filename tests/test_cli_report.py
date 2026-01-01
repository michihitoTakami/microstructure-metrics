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
    for key in ["thd_n", "nps", "delta_se", "mps", "tfs"]:
        assert key in payload["metrics"]
    assert abs(payload["alignment"]["delay_samples"] - delay_samples) < 10

    assert csv_path.exists()
    csv_body = csv_path.read_text()
    assert "thd_n" in csv_body
    assert md_path.exists()
    assert "Microstructure Metrics Report" in md_path.read_text()
