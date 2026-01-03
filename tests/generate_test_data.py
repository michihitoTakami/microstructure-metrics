from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from microstructure_metrics.alignment import align_audio_pair
from microstructure_metrics.testing import (
    DEFAULT_REGRESSION_CASES,
    evaluate_metrics,
    generate_pair_from_case,
    save_test_pair,
)

DATA_DIR = Path(__file__).parent / "data"
EXPECTED_PATH = Path(__file__).parent / "expected_values.yaml"


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
    args = parser.parse_args()

    expected_payload: dict[str, dict[str, object]] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    if args.write_expected:
        EXPECTED_PATH.write_text(
            yaml.safe_dump(expected_payload, allow_unicode=True, sort_keys=True)
        )
        print(f"updated expected values -> {EXPECTED_PATH}")
    else:
        print("Skip writing expected_values.yaml (use --write-expected to update).")


if __name__ == "__main__":
    main()
