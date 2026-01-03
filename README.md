# Microstructure Metrics

Human-perception-oriented DAC/AMP microdynamics evaluation suite.
Provides offline alignment, drift estimation, signal generation, and six metrics (THD+N / NPS / PSD notch depth / ΔSE / MPS / TFS) in one pipeline. (EPIC: [issue #1](https://github.com/michihitoTakami/microstructure-metrics/issues/1))

## Why this project

Modern DAC/AMPs often exceed 120 dB SINAD, yet steady-state metrics alone cannot explain perceived differences in “microdynamics” (transients, texture, spatial cues). This toolkit targets micro-structure degradations under these hypotheses:
- Strong NFB may over-smooth transients.
- Spectral notches/peaks can be filled by dynamic IMD, losing information.
- Temporal fine structure (TFS) phase coherence may break at high bands, harming localization.

## Metrics implemented

| Metric | Purpose |
| --- | --- |
| THD+N | Baseline gain/distortion health |
| Notch Preservation Score (NPS) | Notch depth preservation vs noise/IMD pollution |
| PSD Notch Depth (high-Q) | Narrowband Welch PSD depth vs surrounding ring for high-Q notches |
| Spectral Entropy ΔSE | Information loss / spectral flattening |
| Modulation Power Spectrum (MPS) | Texture preservation (correlation/distance) |
| Temporal Fine Structure (TFS) | High-band phase coherence & group-delay stability |

### MPS options (S-09)
- Log-scale modulation grid (`mod_scale="log"`, `num_mod_bins`).
- Filterbank selection: gammatone (default) or mel (`filterbank="mel"`).
- Envelope extraction switch: Hilbert (default) or rectification + LPF.
- Similarity controls: optional band weights/energy weighting, global/per-band normalization, and power/log scale.

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
git clone https://github.com/michihitoTakami/microstructure-metrics.git
cd microstructure-metrics
uv sync
```

## Quickstart

```bash
# Version / help
uv run microstructure-metrics --version
uv run microstructure-metrics --help

# Generate test signal (example: notched noise + metadata)
uv run microstructure-metrics generate notched-noise --with-metadata

# Align ref/dut WAVs using pilot tones
uv run microstructure-metrics align ref.wav dut.wav

# Estimate drift and emit warnings (optionally JSON)
uv run microstructure-metrics drift ref.aligned_ref.wav dut.aligned_dut.wav

# Compute all metrics and export JSON/CSV/Markdown
uv run microstructure-metrics report ref.aligned_ref.wav dut.aligned_dut.wav \
  --output-json report.json --output-md report.md
```

## Documentation

- User Guide (EN): `docs/en/user-guide.md`
- Japanese overview: `README_JP.md`
- Measurement setup: `docs/en/measurement-setup.md` (EN), `docs/jp/measurement-setup.md` (JP)
- Signal specifications: `docs/en/signal-specifications.md` (EN), `docs/jp/signal-specifications.md` (JP)
- Metrics interpretation: `docs/en/metrics-interpretation.md` (EN), `docs/jp/metrics-interpretation.md` (JP)
- CLI/API reference (JP): `docs/jp/api-cli-reference.md`
- Alignment details (JP): `docs/jp/alignment.md`

## Development setup

```bash
git clone https://github.com/michihitoTakami/microstructure-metrics.git
cd microstructure-metrics
uv sync --extra dev
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

### Quality checks

```bash
uv run ruff check src/
uv run ruff format src/
uv run mypy src/
uv run pytest
uv run pre-commit run --all-files
```

## Project layout

```
microstructure-metrics/
├── docs/                          # Specifications & guides
├── src/microstructure_metrics/    # Package
├── tests/                         # Unit/regression tests
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md
```

## License

MIT License
