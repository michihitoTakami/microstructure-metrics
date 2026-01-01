from __future__ import annotations

import json
from pathlib import Path

import click
import soundfile as sf

from microstructure_metrics.alignment import align_audio_pair


@click.command(name="align")
@click.argument("reference", type=click.Path(exists=True, dir_okay=False))
@click.argument("dut", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--pilot-freq",
    default=1000.0,
    show_default=True,
    help="パイロットトーン周波数 (Hz)",
)
@click.option(
    "--threshold",
    default=0.5,
    show_default=True,
    help="パイロット検出の正規化閾値 (0-1)",
)
@click.option(
    "--band-width-hz",
    default=200.0,
    show_default=True,
    help="パイロット周辺の帯域幅 (Hz, ±帯域)",
)
@click.option(
    "--min-duration-ms",
    default=90.0,
    show_default=True,
    help="パイロット最小長 (ms)",
)
@click.option(
    "--pilot-duration-ms",
    default=100.0,
    show_default=True,
    help="パイロット想定長 (ms)",
)
@click.option(
    "--margin-ms",
    default=5.0,
    show_default=True,
    help="パイロット端からテスト区間を切り出す際のマージン (ms)",
)
@click.option(
    "--max-lag-ms",
    default=100.0,
    show_default=True,
    help="相互相関で探索する最大ラグ (ms)",
)
@click.option(
    "--no-refine-delay",
    is_flag=True,
    default=False,
    help="ラグ推定のサブサンプル補間を無効化",
)
@click.option(
    "--output-ref",
    type=click.Path(),
    help="アライメント後のリファレンス出力WAVパス（未指定なら自動命名）",
)
@click.option(
    "--output-dut",
    type=click.Path(),
    help="アライメント後のDUT出力WAVパス（未指定なら自動命名）",
)
@click.option(
    "--metadata",
    type=click.Path(),
    help="結果メタデータ(JSON)の出力パス（未指定なら *_alignment.json）",
)
def align(
    reference: str,
    dut: str,
    pilot_freq: float,
    threshold: float,
    band_width_hz: float,
    min_duration_ms: float,
    pilot_duration_ms: float,
    margin_ms: float,
    max_lag_ms: float,
    no_refine_delay: bool,
    output_ref: str | None,
    output_dut: str | None,
    metadata: str | None,
) -> None:
    """リファレンスとDUTのWAVをアライメントし、遅延補正したWAVを出力する。"""
    ref_path = Path(reference)
    dut_path = Path(dut)

    ref_data, ref_sr = sf.read(ref_path)
    dut_data, dut_sr = sf.read(dut_path)
    if ref_data.ndim != 1 or dut_data.ndim != 1:
        raise click.ClickException("mono WAV のみ対応しています。")
    if ref_sr != dut_sr:
        raise click.ClickException(f"サンプルレート不一致: ref={ref_sr}, dut={dut_sr}")

    result = align_audio_pair(
        reference=ref_data,
        dut=dut_data,
        sample_rate=ref_sr,
        pilot_freq=pilot_freq,
        threshold=threshold,
        band_width_hz=band_width_hz,
        min_duration_ms=min_duration_ms,
        pilot_duration_ms=pilot_duration_ms,
        margin_ms=margin_ms,
        max_lag_ms=max_lag_ms,
        refine_delay=not no_refine_delay,
    )

    out_ref = (
        Path(output_ref) if output_ref else ref_path.with_suffix(".aligned_ref.wav")
    )
    out_dut = (
        Path(output_dut) if output_dut else dut_path.with_suffix(".aligned_dut.wav")
    )
    out_ref.parent.mkdir(parents=True, exist_ok=True)
    out_dut.parent.mkdir(parents=True, exist_ok=True)

    sf.write(out_ref, result.aligned_ref, samplerate=ref_sr)
    sf.write(out_dut, result.aligned_dut, samplerate=ref_sr)

    meta_path = (
        Path(metadata)
        if metadata
        else ref_path.with_suffix("").with_name(f"{ref_path.stem}_alignment.json")
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "delay_samples": result.delay_samples,
        "start_sample": result.start_sample,
        "end_sample": result.end_sample,
        "confidence": result.confidence,
        "sample_rate": ref_sr,
        "pilot_freq": pilot_freq,
        "threshold": threshold,
        "band_width_hz": band_width_hz,
        "min_duration_ms": min_duration_ms,
        "pilot_duration_ms": pilot_duration_ms,
        "margin_ms": margin_ms,
        "max_lag_ms": max_lag_ms,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    click.echo(f"Aligned WAVを書き出しました: {out_ref} , {out_dut}")
    click.echo(f"Metadataを書き出しました: {meta_path}")
