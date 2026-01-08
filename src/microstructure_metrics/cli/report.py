from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
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
    MPSSimilarityResult,
    TFSCorrelationResult,
    THDNResult,
    TransientResult,
    calculate_mps_similarity,
    calculate_tfs_correlation,
    calculate_thd_n,
    calculate_transient_metrics,
)
from microstructure_metrics.visualization import (
    save_mps_delta_heatmap,
    save_tfs_correlation_timeseries,
)

DEFAULT_JSON = "metrics_report.json"


@dataclass(frozen=True)
class CalculatedMetrics:
    thd: THDNResult
    transient: TransientResult
    mps: MPSSimilarityResult
    tfs: TFSCorrelationResult


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
    "--plot",
    is_flag=True,
    default=False,
    help="MPS/TFSの可視化画像を出力する",
)
@click.option(
    "--plot-dir",
    type=click.Path(file_okay=False),
    help="プロット画像の出力ディレクトリ（未指定時は --output-json と同じ場所に作成）",
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
    "--channels",
    type=click.Choice(["ch0", "ch1", "stereo", "mid", "side"]),
    help="ステレオ入力処理モード（--channel との同時指定不可）",
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
    type=float,
    default=None,
    show_default=False,
    help=(
        "THD+N の想定ピークレベル (dBFS)。"
        "指定すると、基本波レベルが許容範囲から外れた場合に警告を出す。"
    ),
)
@click.option(
    "--transient-smoothing-ms",
    type=click.FloatRange(min=0.0),
    default=0.05,
    show_default=True,
    help="Transient包絡の平滑化時間(ms)。0でヒルベルト包絡(無平滑)",
)
@click.option(
    "--transient-asymmetry-window-ms",
    type=click.FloatRange(min=0.1),
    default=3.0,
    show_default=True,
    help="Pre-energy/Skewness計算に使うピーク周辺半窓(±ms)",
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
@click.option(
    "--mps-mod-weighting",
    type=click.Choice(["none", "high_mod"]),
    default="none",
    show_default=True,
    help="変調周波数の重み付け: none/high_mod(4Hz以上を強め、10Hz以上をさらに重視)",
)
def report(
    reference: str,
    dut: str,
    output_json: str,
    output_csv: str | None,
    output_md: str | None,
    plot: bool,
    plot_dir: str | None,
    allow_resample: bool,
    target_sample_rate: int | None,
    channel: int | None,
    channels: str | None,
    align: bool,
    pilot_freq: float,
    pilot_threshold: float,
    pilot_band_width_hz: float,
    pilot_duration_ms: float,
    min_duration_ms: float,
    margin_ms: float,
    max_lag_ms: float,
    fundamental_freq: float,
    expected_level_dbfs: float | None,
    transient_smoothing_ms: float,
    transient_asymmetry_window_ms: float,
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
    mps_mod_weighting: str,
) -> None:
    """リファレンス/DUT WAVを整列し、全指標を計算してレポートする。"""
    if channel is not None and channels is not None:
        raise click.ClickException("--channel と --channels は同時に指定できません。")

    try:
        ref_data, dut_data, validation = load_audio_pair(
            reference_path=reference,
            dut_path=dut,
            allow_resample=allow_resample,
            target_sample_rate=target_sample_rate,
            channel=channel,
            channels=channels,
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

    metrics_by_channel: dict[str, CalculatedMetrics] = {}
    if aligned_ref.ndim == 1:
        metrics_single = _calculate_metrics(
            aligned_ref=aligned_ref,
            aligned_dut=aligned_dut,
            sample_rate=sample_rate,
            fundamental_freq=fundamental_freq,
            expected_level_dbfs=expected_level_dbfs,
            transient_smoothing_ms=transient_smoothing_ms,
            transient_asymmetry_window_ms=transient_asymmetry_window_ms,
            mps_filterbank=mps_filterbank,
            mps_filterbank_kwargs={
                "order": mps_filterbank_order,
                "bandwidth_scale": mps_filterbank_bandwidth_scale,
            },
            mps_envelope_method=mps_envelope_method,
            mps_envelope_lpf_hz=(
                mps_envelope_lpf_hz if mps_envelope_lpf_hz > 0 else None
            ),
            mps_envelope_lpf_order=mps_envelope_lpf_order,
            mps_mod_scale=mps_mod_scale,
            mps_num_mod_bins=mps_num_mod_bins,
            mps_scale=mps_scale,
            mps_norm=mps_norm,
            mps_band_weighting=mps_band_weighting,
            mps_mod_weighting=mps_mod_weighting,
        )
        metrics_by_channel["mono"] = metrics_single
        metrics_payload: dict[str, object] = _metrics_to_payload(metrics_single)
    else:
        for idx in range(aligned_ref.shape[1]):
            key = f"ch{idx}"
            metrics_by_channel[key] = _calculate_metrics(
                aligned_ref=aligned_ref[:, idx],
                aligned_dut=aligned_dut[:, idx],
                sample_rate=sample_rate,
                fundamental_freq=fundamental_freq,
                expected_level_dbfs=expected_level_dbfs,
                transient_smoothing_ms=transient_smoothing_ms,
                transient_asymmetry_window_ms=transient_asymmetry_window_ms,
                mps_filterbank=mps_filterbank,
                mps_filterbank_kwargs={
                    "order": mps_filterbank_order,
                    "bandwidth_scale": mps_filterbank_bandwidth_scale,
                },
                mps_envelope_method=mps_envelope_method,
                mps_envelope_lpf_hz=(
                    mps_envelope_lpf_hz if mps_envelope_lpf_hz > 0 else None
                ),
                mps_envelope_lpf_order=mps_envelope_lpf_order,
                mps_mod_scale=mps_mod_scale,
                mps_num_mod_bins=mps_num_mod_bins,
                mps_scale=mps_scale,
                mps_norm=mps_norm,
                mps_band_weighting=mps_band_weighting,
                mps_mod_weighting=mps_mod_weighting,
            )
        metrics_payload = {
            name: _metrics_to_payload(metric)
            for name, metric in metrics_by_channel.items()
        }

    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    plot_enabled = plot or plot_dir is not None
    plot_payload: dict[str, object] | None = None
    if plot_enabled:
        resolved_plot_dir = (
            Path(plot_dir)
            if plot_dir is not None
            else json_path.parent / f"{json_path.stem}_plots"
        )
        if aligned_ref.ndim == 1:
            plot_payload = _generate_plots(
                metrics=metrics_by_channel["mono"],
                plot_dir=resolved_plot_dir,
            )
        else:
            plot_payload = {"plot_dir": str(resolved_plot_dir.resolve())}
            for name, metric in metrics_by_channel.items():
                plot_payload[name] = _generate_plots(
                    metrics=metric,
                    plot_dir=resolved_plot_dir / name,
                )
        click.echo(f"プロットを書き出しました: {resolved_plot_dir}")

    report_payload: dict[str, object] = {
        "sample_rate": sample_rate,
        "validation": {
            "is_valid": validation.is_valid,
            "warnings": validation.warnings,
            "errors": validation.errors,
        },
        "alignment": _alignment_summary(alignment, aligned_ref.shape[0]),
        "drift": drift_payload,
        "metrics": metrics_payload,
    }
    if plot_payload is not None:
        report_payload["plots"] = plot_payload

    json_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2))
    click.echo(f"JSONレポートを書き出しました: {json_path}")

    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(csv_path, metrics_payload)
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
    expected_level_dbfs: float | None,
    transient_smoothing_ms: float,
    transient_asymmetry_window_ms: float,
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
    mps_mod_weighting: str,
) -> CalculatedMetrics:
    """各指標を計算し、後続処理で利用しやすい形へまとめる。"""
    thd = calculate_thd_n(
        signal=aligned_dut,
        fundamental_freq=fundamental_freq,
        sample_rate=sample_rate,
        expected_level_dbfs=expected_level_dbfs,
    )
    transient = calculate_transient_metrics(
        reference=aligned_ref,
        dut=aligned_dut,
        sample_rate=sample_rate,
        smoothing_ms=transient_smoothing_ms,
        asymmetry_window_ms=transient_asymmetry_window_ms,
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
        mod_weighting=cast(Literal["none", "high_mod"], mps_mod_weighting),
    )
    tfs = calculate_tfs_correlation(
        reference=aligned_ref,
        dut=aligned_dut,
        sample_rate=sample_rate,
    )

    return CalculatedMetrics(thd=thd, transient=transient, mps=mps, tfs=tfs)


def _metrics_to_payload(metrics: CalculatedMetrics) -> dict[str, dict[str, object]]:
    return {
        "thd_n": _thd_summary(metrics.thd),
        "transient": _transient_summary(metrics.transient),
        "mps": _mps_summary(metrics.mps),
        "tfs": _tfs_summary(metrics.tfs),
    }


def _generate_plots(*, metrics: CalculatedMetrics, plot_dir: Path) -> dict[str, str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    mps_path = save_mps_delta_heatmap(
        result=metrics.mps, path=plot_dir / "mps_delta_heatmap.png"
    )
    tfs_path = save_tfs_correlation_timeseries(
        result=metrics.tfs, path=plot_dir / "tfs_correlation_timeseries.png"
    )
    return {
        "plot_dir": str(plot_dir.resolve()),
        "mps_delta_heatmap": str(mps_path.resolve()),
        "tfs_correlation_timeseries": str(tfs_path.resolve()),
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


def _mps_summary(result: MPSSimilarityResult) -> dict[str, object]:
    band_corr = {
        f"{freq:.1f}": float(value) for freq, value in result.band_correlations.items()
    }
    return {
        "mps_correlation": float(result.mps_correlation),
        "mps_distance": float(result.mps_distance),
        "mps_distance_weighted": float(result.mps_distance_weighted),
        "mps_mod_weighting": str(result.mod_weighting),
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
        "percentile_05_correlation": float(result.percentile_05_correlation),
        "correlation_variance": float(result.correlation_variance),
        "phase_coherence": float(result.phase_coherence),
        "group_delay_std_ms": float(result.group_delay_std_ms),
        "band_correlations": band_corr,
        "band_group_delays_ms": group_delays,
        "frames_per_band": int(result.frames_per_band),
        "used_frames": int(result.used_frames),
        "frame_length_ms": float(result.frame_length_ms),
        "frame_hop_ms": float(result.frame_hop_ms),
        "max_lag_ms": float(result.max_lag_ms),
        "envelope_threshold_db": float(result.envelope_threshold_db),
    }


def _transient_summary(result: TransientResult) -> dict[str, object]:
    return {
        "low_level_attack_time_ref_ms": float(result.low_level_attack_time_ref_ms),
        "low_level_attack_time_dut_ms": float(result.low_level_attack_time_dut_ms),
        "low_level_attack_time_delta_ms": float(result.low_level_attack_time_delta_ms),
        "low_level_attack_time_delta_p95_ms": float(
            result.low_level_attack_time_delta_stats_ms.percentile_95
        ),
        "attack_time_ref_ms": float(result.attack_time_ref_ms),
        "attack_time_dut_ms": float(result.attack_time_dut_ms),
        "attack_time_delta_ms": float(result.attack_time_delta_ms),
        "attack_time_delta_p95_ms": float(
            result.attack_time_delta_stats_ms.percentile_95
        ),
        "edge_sharpness_ref": float(result.edge_sharpness_ref),
        "edge_sharpness_dut": float(result.edge_sharpness_dut),
        "edge_sharpness_ratio": float(result.edge_sharpness_ratio),
        "edge_sharpness_ratio_p05": float(
            result.edge_sharpness_ratio_stats.percentile_05
        ),
        "edge_sharpness_ratio_p95": float(
            result.edge_sharpness_ratio_stats.percentile_95
        ),
        "width_ref_ms": float(result.width_ref_ms),
        "width_dut_ms": float(result.width_dut_ms),
        "transient_smearing_index": float(result.transient_smearing_index),
        "transient_smearing_index_p95": float(result.width_ratio_stats.percentile_95),
        "pre_energy_fraction_ref": float(result.pre_energy_fraction_ref),
        "pre_energy_fraction_dut": float(result.pre_energy_fraction_dut),
        "pre_energy_fraction_delta": float(result.pre_energy_fraction_delta),
        "pre_energy_fraction_delta_p95": float(
            result.pre_energy_fraction_delta_stats.percentile_95
        ),
        "energy_skewness_ref": float(result.energy_skewness_ref),
        "energy_skewness_dut": float(result.energy_skewness_dut),
        "energy_skewness_delta": float(result.energy_skewness_delta),
        "energy_skewness_delta_p95": float(
            result.energy_skewness_delta_stats.percentile_95
        ),
        "event_counts": {
            "ref": len(result.ref_events),
            "dut": len(result.dut_events),
            "matched": int(result.matched_event_pairs),
            "unmatched_ref": int(result.unmatched_ref_events),
            "unmatched_dut": int(result.unmatched_dut_events),
        },
        "params": {
            "smoothing_ms": float(result.params.smoothing_ms),
            "peak_threshold_db": float(result.params.peak_threshold_db),
            "refractory_ms": float(result.params.refractory_ms),
            "match_tolerance_ms": float(result.params.match_tolerance_ms),
            "max_event_duration_ms": float(result.params.max_event_duration_ms),
            "width_fraction": float(result.params.width_fraction),
            "asymmetry_window_ms": float(result.params.asymmetry_window_ms),
        },
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

    plots_obj = report_payload.get("plots")
    if isinstance(plots_obj, dict):
        plot_lines = _render_plot_section(plots_obj, path.parent)
        if plot_lines:
            lines.extend(plot_lines)

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


def _render_plot_section(plots: Mapping[str, object], base_dir: Path) -> list[str]:
    lines: list[str] = []
    entries: list[tuple[str, str]] = []
    plot_dir_value = plots.get("plot_dir")
    plot_dir_path = Path(str(plot_dir_value)) if plot_dir_value else None

    base_entries: list[tuple[str, str]] = []
    nested_entries: list[tuple[str, str]] = []

    targets = [
        ("MPS Delta Heatmap", "mps_delta_heatmap"),
        ("TFS Correlation", "tfs_correlation_timeseries"),
    ]

    for title, key in targets:
        plot_path = _extract_plot_path(plots, key)
        if plot_path is None:
            continue
        rendered = _path_for_markdown(Path(plot_path), base_dir)
        base_entries.append((title, rendered))

    if not base_entries:
        for name, value in plots.items():
            if name == "plot_dir" or not isinstance(value, Mapping):
                continue
            for title, key in targets:
                plot_path = _extract_plot_path(value, key)
                if plot_path is None:
                    continue
                rendered = _path_for_markdown(Path(plot_path), base_dir)
                nested_entries.append((f"{name}: {title}", rendered))

    entries.extend(base_entries if base_entries else nested_entries)

    if not entries:
        return lines

    lines.append("## PLOTS")
    lines.append("")
    if plot_dir_path is not None:
        lines.append(f"- Plot directory: {_path_for_markdown(plot_dir_path, base_dir)}")
        lines.append("")
    for title, rendered_path in entries:
        lines.append(f"### {title}")
        lines.append(f"![{title}]({rendered_path})")
        lines.append("")
    return lines


def _extract_plot_path(plots: Mapping[str, object], key: str) -> str | None:
    value = plots.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        nested = value.get("path")
        if isinstance(nested, str):
            return nested
    return None


def _path_for_markdown(target: Path, base_dir: Path) -> str:
    resolved = target.resolve()
    try:
        return resolved.relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


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
