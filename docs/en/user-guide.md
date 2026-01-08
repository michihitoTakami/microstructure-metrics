# User Guide (English)

This guide is for end users who measure and compare DAC/AMP devices using the CLI. Development/engineering specs remain in Japanese docs.

## What you get
- Pilot-tone-based alignment and drift estimation
- Test signal generation (THD, pink-noise, AM/FM, TFS tones, transient clicks/bursts)
- Metrics: THD+N, MPS, TFS, Transient
- Reports in JSON/CSV/Markdown, optional plots for MPS/TFS

## Prerequisites
- Python 3.13+, [uv](https://github.com/astral-sh/uv)
- WAV I/O: stereo (2ch), 48 kHz / 24-bit (32f allowed). Mono inputs are duplicated to stereo internally. Keep lead/tail silence; do not trim.

## Install
```bash
git clone https://github.com/michihitoTakami/microstructure-metrics.git
cd microstructure-metrics
uv sync
```

## Typical workflow
1) Generate or download the test signal
   - Example: `uv run microstructure-metrics generate notched-noise --with-metadata`
2) Play and record the DUT path with bit-perfect playback (no EQ/AGC/SRC). Keep the file untrimmed.
3) Align ref/dut using pilots
   - `uv run microstructure-metrics align ref.wav dut.wav`
4) Check drift
   - `uv run microstructure-metrics drift ref.aligned_ref.wav dut.aligned_dut.wav --json-output drift.json`
5) Compute all metrics
   - `uv run microstructure-metrics report ref.aligned_ref.wav dut.aligned_dut.wav --output-json report.json --output-md report.md --plot`

## CLI quick reference
- Generate: `microstructure-metrics generate <signal_type> [options]`
- Align: `microstructure-metrics align ref.wav dut.wav [options]`
- Drift: `microstructure-metrics drift ref.wav dut.wav [options]`
- Report: `microstructure-metrics report ref.wav dut.wav [options]`
See `docs/api-cli-reference.md` for full option lists (Japanese).

## Recording tips
- Sample rate/bit depth: 48 kHz, 24-bit (or 32f). Mono preferred.
- Levels: keep peaks below -1 dBFS; pilot is around -6 dBFS. Avoid clipping and AGC.
- Do not trim silence; pilots (100 ms) and 500 ms silences are required for alignment/drift.

## Inputs/outputs
- Input: WAV for reference and DUT, optional JSON metadata from the generator.
- Outputs:
  - Aligned WAVs: `*.aligned_ref.wav`, `*.aligned_dut.wav`
  - Drift report (optional JSON)
  - Metrics report: JSON (default `metrics_report.json`), optional CSV/Markdown
  - Plots (when `--plot`): MPS delta heatmap, TFS correlation time-series

## Related docs
- Japanese overview: `README_JP.md`
- Measurement setup: `docs/en/measurement-setup.md` (EN), `docs/jp/measurement-setup.md` (JP)
- Signal specs: `docs/en/signal-specifications.md` (EN), `docs/jp/signal-specifications.md` (JP)
- Metrics interpretation: `docs/en/metrics-interpretation.md` (EN), `docs/jp/metrics-interpretation.md` (JP)
- CLI/API reference: `docs/jp/api-cli-reference.md` (JP)
