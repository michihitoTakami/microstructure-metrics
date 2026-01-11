from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf
import yaml
from numpy.random import default_rng

from microstructure_metrics.alignment import align_audio_pair
from microstructure_metrics.signals import CommonSignalConfig, build_signal
from microstructure_metrics.testing import (
    DEFAULT_REGRESSION_CASES,
    evaluate_metrics,
    generate_pair_from_case,
    save_test_pair,
)

DATA_DIR = Path(__file__).parent / "data"
EXPECTED_PATH = Path(__file__).parent / "expected_values.yaml"


def _visualization_pairs(
    *, sample_rate: int
) -> list[
    tuple[
        str,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        int,
        str,
        dict[str, object],
        dict[str, object],
    ]
]:
    """可視化用に差分が分かりやすいテストペアを返す（WAVはgit管理しない想定）。"""
    common = CommonSignalConfig(sample_rate=sample_rate, duration=2.0)
    rng_seed = 0

    cases: list[
        tuple[
            str,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            int,
            str,
            dict[str, object],
            dict[str, object],
        ]
    ] = []

    # BCP (binaural) の可視化向け: 同一ノイズから ITD/ILD だけ変える
    ref = build_signal(
        "binaural-cues",
        common=common,
        rng=default_rng(rng_seed),
        binaural_itd_ms=0.20,
        binaural_ild_db=0.0,
    )
    dut = build_signal(
        "binaural-cues",
        common=common,
        rng=default_rng(rng_seed),
        binaural_itd_ms=0.60,
        binaural_ild_db=6.0,
    )
    cases.append(
        (
            "binaural_cues_itd_ild",
            ref.data,
            dut.data,
            common.sample_rate,
            "binaural_cues_itd0.2ms_ild0_to_itd0.6ms_ild6",
            ref.metadata,
            dut.metadata,
        )
    )

    # Mid/Side の可視化向け: 同一テクスチャで side 成分だけ増減
    ref_ms = build_signal(
        "ms-side-texture",
        common=common,
        rng=default_rng(rng_seed),
        min_tone_freq=4000.0,
        tone_count=6,
        tone_step=1200.0,
    )
    # DUT: side を強めてステレオ差分を出す（M/S変換してSideのみゲイン）
    mid = 0.5 * (ref_ms.data[:, 0] + ref_ms.data[:, 1])
    side = 0.5 * (ref_ms.data[:, 0] - ref_ms.data[:, 1])
    side_gain = 1.8
    dut_ms_data = (mid + side_gain * side, mid - side_gain * side)
    dut_ms = ref_ms.data.copy()
    dut_ms[:, 0] = dut_ms_data[0]
    dut_ms[:, 1] = dut_ms_data[1]
    dut_ms = np.clip(dut_ms, -0.9999, 0.9999)
    cases.append(
        (
            "ms_side_texture_side_gain",
            ref_ms.data,
            dut_ms,
            common.sample_rate,
            "ms_side_texture_side_gain",
            ref_ms.metadata,
            {**ref_ms.metadata, "side_gain": float(side_gain)},
        )
    )

    return cases


def _default_tolerance(metric: str, value: float) -> float:
    if metric.endswith("_db"):
        return 2.0
    if metric.endswith("_ms"):
        return max(0.1, abs(value) * 0.1)
    if "correlation" in metric or "coherence" in metric:
        return 0.05
    if "distance" in metric:
        return 0.01
    if metric.endswith("_ratio") or metric.endswith("_index"):
        return max(0.02, abs(value) * 0.05)
    return max(0.01, abs(value) * 0.05)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate degraded test pairs and optional expected YAML."
    )
    parser.add_argument(
        "--profile",
        choices=["regression", "visual"],
        default="regression",
        help="生成プロファイル (default: regression)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="WAV/JSON を出力するディレクトリ (default: tests/data)",
    )
    parser.add_argument(
        "--write-expected",
        action="store_true",
        help="期待値ファイル(expected_values.yaml)も更新する",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48_000,
        help="visual profile 用のサンプルレート (default: 48000)",
    )
    args = parser.parse_args()

    expected_payload: dict[str, dict[str, object]] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.profile == "regression":
        for case in DEFAULT_REGRESSION_CASES:
            pair = generate_pair_from_case(case)
            alignment = align_audio_pair(
                reference=pair.reference,
                dut=pair.dut,
                sample_rate=pair.sample_rate,
                pilot_freq=pair.common.pilot_freq,
                threshold=0.35,
                band_width_hz=200.0,
                min_duration_ms=pair.common.pilot_duration_ms * 0.9,
                pilot_duration_ms=pair.common.pilot_duration_ms,
                margin_ms=pair.common.fade_ms,
                max_lag_ms=30.0,
            )
            metrics = evaluate_metrics(
                reference=alignment.aligned_ref,
                dut=alignment.aligned_dut,
                sample_rate=pair.sample_rate,
                metadata=pair.reference_metadata,
                metrics=case.metrics,
            )

            case_dir = args.output_dir / case.key
            save_test_pair(pair, case_dir)

            expected_payload[case.key] = {
                "description": case.description,
                "degradation": case.degradation,
                "severity": case.severity,
                "signal_type": case.signal_type,
                "metrics": {
                    metric: {
                        "expected": float(value),
                        "tolerance": _default_tolerance(metric, float(value)),
                    }
                    for metric, value in metrics.items()
                },
            }
    else:
        for (
            key,
            reference,
            dut,
            sample_rate,
            stem,
            reference_metadata,
            dut_metadata,
        ) in _visualization_pairs(sample_rate=args.sample_rate):
            out_dir = args.output_dir / key
            out_dir.mkdir(parents=True, exist_ok=True)
            ref_path = out_dir / f"{stem}_ref.wav"
            dut_path = out_dir / f"{stem}_dut.wav"

            # WAV/JSON を書き出す（WAVは.gitignore想定）
            sf.write(ref_path, reference, samplerate=sample_rate)
            sf.write(dut_path, dut, samplerate=sample_rate)
            ref_path.with_suffix(".json").write_text(
                json.dumps(reference_metadata, ensure_ascii=False, indent=2)
            )
            dut_path.with_suffix(".json").write_text(
                json.dumps(dut_metadata, ensure_ascii=False, indent=2)
            )

    if args.write_expected:
        EXPECTED_PATH.write_text(
            yaml.safe_dump(expected_payload, allow_unicode=True, sort_keys=True)
        )
        print(f"updated expected values -> {EXPECTED_PATH}")
    else:
        print("Skip writing expected_values.yaml (use --write-expected to update).")


if __name__ == "__main__":
    main()
