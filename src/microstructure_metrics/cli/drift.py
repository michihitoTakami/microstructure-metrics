from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import soundfile as sf

from microstructure_metrics.alignment import (
    check_drift_threshold,
    drift_to_report,
    estimate_clock_drift,
)

DEFAULT_PILOT_FREQ = 1000.0
DEFAULT_PILOT_DURATION_MS = 100
DEFAULT_BAND_WIDTH_HZ = 200.0
DEFAULT_THRESHOLD = 0.3


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path)
    if data.ndim == 2:
        data = data[:, 0]
    return np.asarray(data, dtype=np.float64), sr


@click.command(name="drift")
@click.argument("reference", type=click.Path(exists=True, dir_okay=False))
@click.argument("dut", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--pilot-freq",
    default=DEFAULT_PILOT_FREQ,
    show_default=True,
    help="パイロットトーン周波数 (Hz)",
)
@click.option(
    "--pilot-duration-ms",
    default=DEFAULT_PILOT_DURATION_MS,
    show_default=True,
    help="パイロット長 (ms)",
)
@click.option(
    "--band-width-hz",
    default=DEFAULT_BAND_WIDTH_HZ,
    show_default=True,
    help="パイロット周辺帯域幅 (±Hz)",
)
@click.option(
    "--threshold",
    default=DEFAULT_THRESHOLD,
    show_default=True,
    help="相関ピークの閾値",
)
@click.option(
    "--json-output",
    type=click.Path(dir_okay=False),
    help="ドリフト警告をJSONで出力するパス（未指定なら書き出しなし）",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="重大度 high/critical をエラー扱いにする",
)
def drift(
    reference: str,
    dut: str,
    pilot_freq: float,
    pilot_duration_ms: int,
    band_width_hz: float,
    threshold: float,
    json_output: str | None,
    strict: bool,
) -> None:
    """パイロットトーンからクロックドリフトを推定し警告を表示する."""
    ref_path = Path(reference)
    dut_path = Path(dut)
    ref_data, ref_sr = _load_mono(ref_path)
    dut_data, dut_sr = _load_mono(dut_path)
    if ref_sr != dut_sr:
        raise click.BadParameter("reference と dut のサンプルレートが一致しません。")

    result = estimate_clock_drift(
        reference=ref_data,
        dut=dut_data,
        sample_rate=ref_sr,
        pilot_freq=pilot_freq,
        pilot_duration_ms=pilot_duration_ms,
        band_width_hz=band_width_hz,
        peak_threshold=threshold,
    )
    warning = check_drift_threshold(result.drift_ppm)

    color = {
        "none": "green",
        "low": "yellow",
        "high": "red",
        "critical": "red",
    }[warning.severity]
    summary = (
        f"Drift: {result.drift_ppm:.2f} ppm "
        f"(severity: {warning.severity}, "
        f"delay_start: {result.delay_start_samples} samples, "
        f"delay_end: {result.delay_end_samples} samples)"
    )
    click.secho(summary, fg=color)
    if warning.message:
        click.secho(warning.message, fg=color)

    if json_output:
        payload = drift_to_report(result, warning)
        out_path = Path(json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        click.echo(f"JSONを書き出しました: {out_path}")

    if strict and warning.severity in {"high", "critical"}:
        raise click.ClickException("strictモード: 高重大度のドリフトが検出されました。")
