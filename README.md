# Microstructure Metrics

Human-perception-oriented DAC/AMP microdynamics evaluation suite.

## Overview

This project provides an offline pipeline to compute **microdynamics metrics** that better align with human auditory perception, alongside conventional steady-state metrics such as SINAD and THD+N.

### Background

Modern DAC/AMP devices easily achieve SINAD > 120 dB, yet steady-state measurements alone cannot explain perceived differences in “musicality” or microdynamics—fine transient behavior, spatial reproduction, and timbral texture. We implement new metrics based on these hypotheses:
- Strong negative feedback (NFB) may smooth transient microstructures.
- Spectral notches/peaks may be lost, causing information deficits.
- Temporal fine structure (TFS) phase coherence may degrade at high frequencies.

### Metrics to be implemented

| Metric | Abbr. | Target |
|--------|-------|--------|
| THD+N | THD+N | Baseline check (gain, distortion) |
| Notch Preservation Score | NPS | Spectral pollution from dynamic IMD |
| Spectral Entropy Delta | ΔSE | Flattening / information loss of structure |
| Modulation Power Spectrum | MPS | Preservation of texture / modulation info |
| Temporal Fine Structure Correlation | TFS | High-frequency phase coherence |

### MPS options (S-09)
- Log-scale modulation grid (`mod_scale="log"`, `num_mod_bins`).
- Filterbank selection: gammatone (default) or mel (`filterbank="mel"`).
- Envelope extraction switch: Hilbert (default) or rectification + LPF.
- Similarity controls: optional band weights/energy weighting, global/per-band normalization, and power/log scale.

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation (via uv)

```bash
git clone https://github.com/michihitoTakami/microstructure-metrics.git
cd microstructure-metrics
uv sync
```

## Quickstart

```bash
# Show CLI help
uv run microstructure-metrics --help

# Check version
uv run microstructure-metrics --version

# Align recorded WAVs (ref/dut) using pilot tones
uv run microstructure-metrics align ref.wav dut.wav

# Run integrated report (align + all metrics)
uv run microstructure-metrics report ref.wav dut.wav \
  --output-json metrics_report.json
```

## Development setup

```bash
git clone https://github.com/michihitoTakami/microstructure-metrics.git
cd microstructure-metrics

# Install with dev extras
uv sync --extra dev

# Install hooks
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
├── docs/                          # Documentation
├── src/
│   └── microstructure_metrics/    # Main package
│       ├── __init__.py
│       ├── cli.py                 # CLI entrypoint
│       └── py.typed               # PEP 561 marker
├── tests/                         # Tests
├── .pre-commit-config.yaml        # pre-commit settings
├── pyproject.toml                 # Project config
└── README.md
```

## References

- `docs/DAC_AMP評価指標 再実装指示.pdf`
- `docs/ミクロダイナミクス評価の新指標.pdf`
- `docs/alignment.md`

## License

MIT License
