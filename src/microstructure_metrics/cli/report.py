from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import click

from microstructure_metrics.alignment import (
    AlignmentResult,
    PilotDetectionError,
    align_audio_pair,
    align_signals,
    check_drift_threshold,
    drift_to_report,
    estimate_clock_drift,
)
from microstructure_metrics.io import load_audio_pair
from microstructure_metrics.metrics import (
    DeltaSEResult,
    MPSSimilarityResult,
    NPSResult,
    TFSCorrelationResult,
    THDNResult,
    calculate_delta_se,
    calculate_mps_similarity,
    calculate_nps,
    calculate_tfs_correlation,
    calculate_thd_n,
)

DEFAULT_JSON = "metrics_report.json"


@click.command(name="report")
@click.argument("reference", type=click.Path(exists=True, dir_okay=False))
@click.argument("dut", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output-json",
    type=click.Path(),
    default=DEFAULT_JSON,
    show_default=True,
    help="集計レポート(JSON)の出力パス",
)
@click.option(
    "--output-csv",
    type=click.Path(),
    help="サマリCSVの出力パス（未指定なら出力しない）",
)
@click.option(
    "--output-md",
    type=click.Path(),
    help="Markdownレポートの出力パス（未指定なら出力しない）",
)
@click.option(
    "--allow-resample",
    is_flag=True,
    default=False,
    help="ref/dut のサンプルレートが異なる場合に自動リサンプルする",
)
@click.option(
    "--target-sample-rate",
    type=click.IntRange(min=1),
    help="リサンプル先サンプルレート（未指定なら reference の SR を優先）",
)
@click.option(
    "--channel",
    type=click.IntRange(min=0),
    help="ステレオ入力時に使用するチャンネル（未指定ならch0）",
)
@click.option(
    "--align/--no-align",
    default=True,
    show_default=True,
    help="パイロットを用いた自動アライメントを行うか",
)
@click.option(
    "--pilot-freq",
    default=1000.0,
    show_default=True,
    help="パイロットトーン周波数 (Hz)",
)
@click.option(
    "--pilot-threshold",
    default=0.5,
    show_default=True,
    help="パイロット検出の正規化閾値 (0-1)",
)
@click.option(
    "--pilot-band-width-hz",
    default=200.0,
    show_default=True,
    help="パイロット周辺帯域幅 (±Hz)",
)
@click.option(
    "--pilot-duration-ms",
    default=100.0,
    show_default=True,
    help="パイロット想定長 (ms)",
)
@click.option(
    "--min-duration-ms",
    default=90.0,
    show_default=True,
    help="パイロット検出に用いる最小長 (ms)",
)
@click.option(
    "--margin-ms",
    default=5.0,
    show_default=True,
    help="パイロット端から本体を切り出す際のマージン (ms)",
)
@click.option(
    "--max-lag-ms",
    default=100.0,
    show_default=True,
    help="相互相関で探索する最大ラグ (ms)",
)
@click.option(
    "--fundamental-freq",
    default=1000.0,
    show_default=True,
    help="THD+N の基本周波数 (Hz)",
)
@click.option(
    "--expected-level-dbfs",
    default=-3.0,
    show_default=True,
    help="THD+N の想定ピークレベル (dBFS)",
)
@click.option(
    "--notch-center-hz",
    default=8000.0,
    show_default=True,
    help="NPS のノッチ中心周波数 (Hz)",
)
@click.option(
    "--notch-q",
    default=8.6,
    show_default=True,
    help="NPS のノッチQ",
)
@click.option(
    "--mps-filterbank",
    type=click.Choice(["gammatone", "mel"]),
    default="gammatone",
    show_default=True,
    help="MPS用の聴覚フィルタバンクを選択",
)
@click.option(
    "--mps-filterbank-order",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="melフィルタバンクのIIR次数（gammatoneでは無視）",
)
@click.option(
    "--mps-filterbank-bandwidth-scale",
    type=click.FloatRange(min=0.1),
    default=1.0,
    show_default=True,
    help="mel帯域幅スケール（gammatoneでは無視）",
)
@click.option(
    "--mps-envelope-method",
    type=click.Choice(["hilbert", "rectify"]),
    default="hilbert",
    show_default=True,
    help="MPS包絡抽出手法（ヒルベルト or 整流）",
)
@click.option(
    "--mps-envelope-lpf-hz",
    default=64.0,
    show_default=True,
    help="包絡ローパスのカットオフ(Hz)。0以下で無効化",
)
@click.option(
    "--mps-envelope-lpf-order",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="包絡ローパスの次数",
)
@click.option(
    "--mps-mod-scale",
    type=click.Choice(["linear", "log"]),
    default="linear",
    show_default=True,
    help="変調周波数軸スケール",
)
@click.option(
    "--mps-num-mod-bins",
    type=click.IntRange(min=2),
    help="mod_scale=log時のbin数（省略時は元bin数）",
)
@click.option(
    "--mps-scale",
    type=click.Choice(["power", "log"]),
    default="power",
    show_default=True,
    help="MPSのスケール（powerまたはlog）",
)
@click.option(
    "--mps-norm",
    type=click.Choice(["global", "per_band", "none"]),
    default="global",
    show_default=True,
    help="MPS類似度の正規化モード",
)
@click.option(
    "--mps-band-weighting",
    type=click.Choice(["none", "energy"]),
    default="none",
    show_default=True,
    help="帯域重み付け: none/energy(参照MPSのエネルギーで重み付け)",
)
def report(
    reference: str,
    dut: str,
    output_json: str,
    output_csv: str | None,
    output_md: str | None,
    allow_resample: bool,
    target_sample_rate: int | None,
    channel: int | None,
    align: bool,
    pilot_freq: float,
    pilot_threshold: float,
    pilot_band_width_hz: float,
    pilot_duration_ms: float,
    min_duration_ms: float,
    margin_ms: float,
    max_lag_ms: float,
    fundamental_freq: float,
    expected_level_dbfs: float,
    notch_center_hz: float,
    notch_q: float,
    mps_filterbank: str,
    mps_filterbank_order: int,
    mps_filterbank_bandwidth_scale: float,
    mps_envelope_method: str,
    mps_envelope_lpf_hz: float,
    mps_envelope_lpf_order: int,
    mps_mod_scale: str,
    mps_num_mod_bins: int | None,
    mps_scale: str,
    mps_norm: str,
    mps_band_weighting: str,
) -> None:
    """リファレンス/DUT WAVを整列し、全指標を計算してレポートする。"""
    try:
        ref_data, dut_data, validation = load_audio_pair(
            reference_path=reference,
            dut_path=dut,
            allow_resample=allow_resample,
            target_sample_rate=target_sample_rate,
            channel=channel,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    sample_rate = validation.metadata_ref.sample_rate

    drift_result = estimate_clock_drift(
        reference=ref_data,
        dut=dut_data,
        sample_rate=sample_rate,
        pilot_freq=pilot_freq,
        pilot_duration_ms=int(pilot_duration_ms),
        band_width_hz=pilot_band_width_hz,
        peak_threshold=pilot_threshold,
    )
    drift_warning = check_drift_threshold(drift_result.drift_ppm)
    drift_payload = drift_to_report(drift_result, drift_warning)

    if align:
        try:
            alignment = align_audio_pair(
                reference=ref_data,
                dut=dut_data,
                sample_rate=sample_rate,
                pilot_freq=pilot_freq,
                threshold=pilot_threshold,
                band_width_hz=pilot_band_width_hz,
                min_duration_ms=min_duration_ms,
                pilot_duration_ms=pilot_duration_ms,
                margin_ms=margin_ms,
                max_lag_ms=max_lag_ms,
                refine_delay=True,
            )
            aligned_ref = alignment.aligned_ref
            aligned_dut = alignment.aligned_dut
        except (ValueError, PilotDetectionError) as exc:
            raise click.ClickException(f"アライメントに失敗しました: {exc}") from exc
    else:
        aligned_ref, aligned_dut = align_signals(
            reference=ref_data, dut=dut_data, delay_samples=0.0
        )
        alignment = None

    metrics = _calculate_metrics(
        aligned_ref=aligned_ref,
        aligned_dut=aligned_dut,
        sample_rate=sample_rate,
        fundamental_freq=fundamental_freq,
        expected_level_dbfs=expected_level_dbfs,
        notch_center_hz=notch_center_hz,
        notch_q=notch_q,
        mps_filterbank=mps_filterbank,
        mps_filterbank_kwargs={
            "order": mps_filterbank_order,
            "bandwidth_scale": mps_filterbank_bandwidth_scale,
        },
        mps_envelope_method=mps_envelope_method,
        mps_envelope_lpf_hz=mps_envelope_lpf_hz if mps_envelope_lpf_hz > 0 else None,
        mps_envelope_lpf_order=mps_envelope_lpf_order,
        mps_mod_scale=mps_mod_scale,
        mps_num_mod_bins=mps_num_mod_bins,
        mps_scale=mps_scale,
        mps_norm=mps_norm,
        mps_band_weighting=mps_band_weighting,
    )

    report_payload = {
        "sample_rate": sample_rate,
        "validation": {
            "is_valid": validation.is_valid,
            "warnings": validation.warnings,
            "errors": validation.errors,
        },
        "alignment": _alignment_summary(alignment, aligned_ref.shape[0]),
        "drift": drift_payload,
        "metrics": metrics,
    }

    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2))
    click.echo(f"JSONレポートを書き出しました: {json_path}")

    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(csv_path, metrics)
        click.echo(f"CSVサマリを書き出しました: {csv_path}")

    if output_md:
        md_path = Path(output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(
            md_path,
            report_payload=report_payload,
        )
        click.echo(f"Markdownレポートを書き出しました: {md_path}")


def _calculate_metrics(
    *,
    aligned_ref: Any,
    aligned_dut: Any,
    sample_rate: int,
    fundamental_freq: float,
    expected_level_dbfs: float,
    notch_center_hz: float,
    notch_q: float,
    mps_filterbank: str,
    mps_filterbank_kwargs: dict[str, float | int | None],
    mps_envelope_method: str,
    mps_envelope_lpf_hz: float | None,
    mps_envelope_lpf_order: int,
    mps_mod_scale: str,
    mps_num_mod_bins: int | None,
    mps_scale: str,
    mps_norm: str,
    mps_band_weighting: str,
) -> dict[str, dict[str, object]]:
    """各指標を計算し、JSONに載せやすい辞書へまとめる。"""
    thd = calculate_thd_n(
        signal=aligned_dut,
        fundamental_freq=fundamental_freq,
        sample_rate=sample_rate,
        expected_level_dbfs=expected_level_dbfs,
    )
    nps = calculate_nps(
        reference=aligned_ref,
        dut=aligned_dut,
        sample_rate=sample_rate,
        notch_center_hz=notch_center_hz,
        notch_q=notch_q,
    )
    delta_se = calculate_delta_se(
        reference=aligned_ref,
        dut=aligned_dut,
        sample_rate=sample_rate,
    )
    mps = calculate_mps_similarity(
        reference=aligned_ref,
        dut=aligned_dut,
        sample_rate=sample_rate,
        filterbank=cast(Literal["gammatone", "mel"], mps_filterbank),
        filterbank_kwargs=mps_filterbank_kwargs,
        envelope_method=cast(Literal["hilbert", "rectify"], mps_envelope_method),
        envelope_lowpass_hz=mps_envelope_lpf_hz,
        envelope_lowpass_order=mps_envelope_lpf_order,
        mod_scale=cast(Literal["linear", "log"], mps_mod_scale),
        num_mod_bins=mps_num_mod_bins,
        mps_scale=cast(Literal["power", "log"], mps_scale),
        mps_norm=cast(Literal["global", "per_band", "none"], mps_norm),
        band_weighting=cast(Literal["none", "energy"], mps_band_weighting),
    )
    tfs = calculate_tfs_correlation(
        reference=aligned_ref,
        dut=aligned_dut,
        sample_rate=sample_rate,
    )

    return {
        "thd_n": _thd_summary(thd),
        "nps": _nps_summary(nps),
        "delta_se": _delta_se_summary(delta_se),
        "mps": _mps_summary(mps),
        "tfs": _tfs_summary(tfs),
    }


def _alignment_summary(
    alignment: AlignmentResult | None, aligned_length: int
) -> dict[str, object]:
    if alignment is None:
        return {
            "delay_samples": 0.0,
            "start_sample": 0,
            "end_sample": aligned_length,
            "confidence": 1.0,
            "aligned_length": aligned_length,
        }

    return {
        "delay_samples": float(alignment.delay_samples),
        "start_sample": int(alignment.start_sample),
        "end_sample": int(alignment.end_sample),
        "confidence": float(alignment.confidence),
        "aligned_length": aligned_length,
    }


def _thd_summary(result: THDNResult) -> dict[str, object]:
    return {
        "thd_n_db": float(result.thd_n_db),
        "thd_n_percent": float(result.thd_n_percent),
        "thd_db": float(result.thd_db),
        "noise_db": float(result.noise_db),
        "sinad_db": float(result.sinad_db),
        "fundamental_freq": float(result.fundamental_freq),
        "fundamental_level_dbfs": float(result.fundamental_level_dbfs),
        "measurement_bandwidth": list(result.measurement_bandwidth),
        "harmonic_levels": {
            str(k): float(v) for k, v in result.harmonic_levels.items()
        },
        "warnings": list(result.warnings),
    }


def _nps_summary(result: NPSResult) -> dict[str, object]:
    return {
        "nps_db": float(result.nps_db),
        "nps_ratio": float(result.nps_ratio),
        "ref_notch_depth_db": float(result.ref_notch_depth_db),
        "dut_notch_depth_db": float(result.dut_notch_depth_db),
        "notch_center_hz": float(result.notch_center_hz),
        "notch_q": float(result.notch_q),
        "noise_floor_db": float(result.noise_floor_db),
        "is_noise_limited": bool(result.is_noise_limited),
    }


def _delta_se_summary(result: DeltaSEResult) -> dict[str, object]:
    return {
        "delta_se_mean": float(result.delta_se_mean),
        "delta_se_std": float(result.delta_se_std),
        "delta_se_max": float(result.delta_se_max),
        "ref_se_mean": float(result.ref_se_mean),
        "dut_se_mean": float(result.dut_se_mean),
    }


def _mps_summary(result: MPSSimilarityResult) -> dict[str, object]:
    band_corr = {
        f"{freq:.1f}": float(value) for freq, value in result.band_correlations.items()
    }
    return {
        "mps_correlation": float(result.mps_correlation),
        "mps_distance": float(result.mps_distance),
        "band_correlations": band_corr,
    }


def _tfs_summary(result: TFSCorrelationResult) -> dict[str, object]:
    band_corr = {
        f"{low:.0f}-{high:.0f}": float(val)
        for (low, high), val in result.band_correlations.items()
    }
    group_delays = {
        f"{low:.0f}-{high:.0f}": float(val)
        for (low, high), val in result.band_group_delays_ms.items()
    }
    return {
        "mean_correlation": float(result.mean_correlation),
        "phase_coherence": float(result.phase_coherence),
        "group_delay_std_ms": float(result.group_delay_std_ms),
        "band_correlations": band_corr,
        "band_group_delays_ms": group_delays,
    }


def _write_csv(path: Path, metrics: dict[str, dict[str, object]]) -> None:
    rows: list[tuple[str, str, object]] = []
    for metric_name, metric_values in metrics.items():
        rows.extend(_flatten_metric(metric_name, metric_values))

    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "key", "value"])
        for metric, key, value in rows:
            writer.writerow([metric, key, value])


def _write_markdown(path: Path, *, report_payload: dict[str, object]) -> None:
    lines: list[str] = []
    sample_rate = report_payload.get("sample_rate")
    lines.append("# Microstructure Metrics Report")
    lines.append("")
    lines.append(f"- Sample rate: {sample_rate} Hz")
    validation = report_payload.get("validation", {})
    if isinstance(validation, dict):
        warnings = validation.get("warnings", [])
        errors = validation.get("errors", [])
        lines.append(f"- Validation warnings: {len(warnings)}")
        lines.append(f"- Validation errors: {len(errors)}")
    lines.append("")

    metrics_obj = report_payload.get("metrics", {})
    if not isinstance(metrics_obj, dict):
        metrics_obj = {}
    for metric_name, metric_values in metrics_obj.items():
        if not isinstance(metric_values, dict):
            continue
        lines.append(f"## {metric_name.upper()}")
        lines.append("")
        lines.append("| key | value |")
        lines.append("| --- | --- |")
        for _, key, value in _flatten_metric(metric_name, metric_values):
            lines.append(f"| {key} | {value} |")
        lines.append("")

    path.write_text("\n".join(lines))


def _flatten_metric(
    metric_name: str, values: Mapping[str, object]
) -> list[tuple[str, str, object]]:
    flattened: list[tuple[str, str, object]] = []
    for key, value in values.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flattened.append((metric_name, f"{key}.{sub_key}", sub_value))
        else:
            flattened.append((metric_name, key, value))
    return flattened
