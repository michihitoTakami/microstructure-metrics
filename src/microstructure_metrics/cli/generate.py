from __future__ import annotations

import json
from pathlib import Path

import click
import soundfile as sf

from microstructure_metrics.signals import (
    SUPPORTED_SIGNALS,
    CommonSignalConfig,
    build_signal,
    subtype_for_bit_depth,
)


def _parse_float_list(text: str) -> list[float]:
    parts = [p.strip() for p in text.replace(" ", ",").split(",")]
    values: list[float] = []
    for part in parts:
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError as exc:
            raise click.ClickException(
                f"数値リストの解析に失敗しました: {text}"
            ) from exc
    if not values:
        raise click.ClickException("--centers は空にできません")
    return values


@click.command(name="generate")
@click.argument(
    "signal_type", type=click.Choice(SUPPORTED_SIGNALS, case_sensitive=False)
)
@click.option(
    "--sample-rate",
    "-sr",
    default=48000,
    type=click.IntRange(8000, 384000),
    show_default=True,
    help="サンプルレート (Hz)",
)
@click.option(
    "--bit-depth",
    "-bd",
    default="24bit",
    show_default=True,
    help="ビット深度 (24bit または 32f)",
)
@click.option(
    "--duration",
    "-d",
    default=10.0,
    show_default=True,
    type=click.FloatRange(min=0.01),
    help="テスト本体の長さ (秒)",
)
@click.option(
    "--pilot-freq",
    default=1000.0,
    show_default=True,
    help="パイロットトーン周波数 (Hz)",
)
@click.option(
    "--pilot-duration",
    default=100,
    show_default=True,
    help="パイロットトーン長さ (ms)",
)
@click.option(
    "--silence-duration",
    default=500,
    show_default=True,
    help="前後無音長さ (ms)",
)
@click.option(
    "--freq",
    default=1000.0,
    show_default=True,
    help="THD用トーン周波数 (Hz)",
)
@click.option(
    "--level-dbfs",
    default=-3.0,
    show_default=True,
    help="THD用トーンの目標ピークレベル (dBFS)",
)
@click.option(
    "--center",
    default=8000.0,
    show_default=True,
    help="ノッチ中心周波数 (Hz)",
)
@click.option(
    "--centers",
    default=None,
    help='ノッチ中心周波数のリスト(Hz)。例: "3000,5000,7000,9000"',
)
@click.option(
    "--q",
    multiple=True,
    default=(8.6,),
    show_default=True,  # click will show '(8.6,)' but keeps backward-compat in usage
    type=click.FloatRange(min=0.1),
    help="ノッチQ値（複数指定するとQスイープとして複数ファイルを生成）",
)
@click.option(
    "--notch-cascade-stages",
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help="ノッチフィルタのカスケード段数（ノッチ強化）",
)
@click.option(
    "--lowcut",
    default=20.0,
    show_default=True,
    type=click.FloatRange(min=0.0),
    help="ノイズ低域カット (Hz)",
)
@click.option(
    "--highcut",
    default=20000.0,
    show_default=True,
    type=click.FloatRange(min=0.0),
    help="ノイズ高域カット (Hz)",
)
@click.option(
    "--carrier",
    default=1000.0,
    show_default=True,
    help="AM/FM搬送波周波数 (Hz)",
)
@click.option(
    "--am-freq",
    default=4.0,
    show_default=True,
    help="AM変調周波数 (Hz)",
)
@click.option(
    "--am-depth",
    default=0.5,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0),
    help="AM変調深度 (0-1)",
)
@click.option(
    "--fm-dev",
    default=50.0,
    show_default=True,
    help="FM周波数偏移 (Hz)",
)
@click.option(
    "--fm-freq",
    default=None,
    help="FM変調周波数 (Hz, 未指定ならAM周波数を使用)",
)
@click.option(
    "--min-freq",
    default=4000.0,
    show_default=True,
    help="TFSマルチトーンの最小周波数 (Hz)",
)
@click.option(
    "--tone-count",
    default=5,
    show_default=True,
    type=click.IntRange(min=1),
    help="TFSマルチトーンのトーン数",
)
@click.option(
    "--tone-step",
    default=2000.0,
    show_default=True,
    type=click.FloatRange(min=1.0),
    help="TFSマルチトーンの周波数間隔 (Hz)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="出力WAVパス（未指定なら仕様に沿ったファイル名を生成）",
)
@click.option(
    "--with-metadata",
    is_flag=True,
    default=False,
    help="メタデータJSONも出力する",
)
def generate(
    signal_type: str,
    sample_rate: int,
    bit_depth: str,
    duration: float,
    pilot_freq: float,
    pilot_duration: int,
    silence_duration: int,
    freq: float,
    level_dbfs: float,
    center: float,
    centers: str | None,
    q: tuple[float, ...],
    notch_cascade_stages: int,
    lowcut: float,
    highcut: float,
    carrier: float,
    am_freq: float,
    am_depth: float,
    fm_dev: float,
    fm_freq: float | None,
    min_freq: float,
    tone_count: int,
    tone_step: float,
    output: str | None,
    with_metadata: bool,
) -> None:
    """テスト信号を生成してWAV/JSONを書き出す."""
    common = CommonSignalConfig(
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        duration=duration,
        pilot_freq=pilot_freq,
        pilot_duration_ms=pilot_duration,
        silence_duration_ms=silence_duration,
    )

    normalized_type = signal_type.lower()
    q_values = q or (8.6,)
    centers_list = _parse_float_list(centers) if centers else [float(center)]

    # Guard: options below only make sense for notched-noise.
    if normalized_type != "notched-noise" and (
        centers is not None or len(q_values) > 1 or notch_cascade_stages != 1
    ):
        raise click.ClickException(
            "--centers/複数--q/--notch-cascade-stages は notched-noise 専用です"
        )

    def _write_one(*, result, wav_path: Path) -> None:
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            wav_path,
            result.data,
            samplerate=common.sample_rate,
            subtype=subtype_for_bit_depth(common.normalized_bit_depth()),
        )
        click.echo(f"WAVを書き出しました: {wav_path}")
        if with_metadata:
            metadata_path = wav_path.with_suffix(".json")
            metadata_path.write_text(
                json.dumps(result.metadata, ensure_ascii=False, indent=2)
            )
            click.echo(f"メタデータを書き出しました: {metadata_path}")

    # Q sweep: generate multiple files when multiple --q is provided for notched-noise.
    if normalized_type == "notched-noise" and len(q_values) > 1:
        out_base = Path(output) if output else Path(".")
        if out_base.suffix.lower() == ".wav":
            raise click.ClickException(
                "複数Qを生成する場合、--output は .wav ではなくディレクトリを指定してください"
            )
        out_base.mkdir(parents=True, exist_ok=True)
        for q_value in q_values:
            result = build_signal(
                signal_type,
                common=common,
                tone_freq=freq,
                tone_level_dbfs=level_dbfs,
                notch_center=center,
                notch_centers_hz=centers_list,
                notch_q=float(q_value),
                notch_cascade_stages=notch_cascade_stages,
                noise_lowcut=lowcut,
                noise_highcut=highcut,
                am_freq=am_freq,
                am_depth=am_depth,
                fm_dev=fm_dev,
                fm_freq=fm_freq,
                carrier_freq=carrier,
                min_tone_freq=min_freq,
                tone_count=tone_count,
                tone_step=tone_step,
            )
            wav_path = out_base / f"{result.suggested_stem}.wav"
            _write_one(result=result, wav_path=wav_path)
        return

    # Single output
    chosen_q = float(q_values[0])
    result = build_signal(
        signal_type,
        common=common,
        tone_freq=freq,
        tone_level_dbfs=level_dbfs,
        notch_center=center,
        notch_centers_hz=centers_list if centers else None,
        notch_q=chosen_q,
        notch_cascade_stages=notch_cascade_stages,
        noise_lowcut=lowcut,
        noise_highcut=highcut,
        am_freq=am_freq,
        am_depth=am_depth,
        fm_dev=fm_dev,
        fm_freq=fm_freq,
        carrier_freq=carrier,
        min_tone_freq=min_freq,
        tone_count=tone_count,
        tone_step=tone_step,
    )
    wav_path = Path(output) if output else Path(f"{result.suggested_stem}.wav")
    _write_one(result=result, wav_path=wav_path)
