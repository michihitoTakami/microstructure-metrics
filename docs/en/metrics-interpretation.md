# Metrics Interpretation (EN)

Purpose: explain how to read the metrics produced by `report` (JSON/CSV/Markdown) when comparing DUT vs reference.

## General guidance
- Values come after pilot-based alignment. Large drift or alignment failure lowers confidence (see `validation` / `drift` in the report).
- Some metrics use dB (bigger magnitude often means more degradation), others use correlation (closer to 1 is better). Check each metric.

## Metric-specific notes

### THD+N
- What: harmonic + noise vs fundamental. See `thd_n_db`, `thd_n_percent`, `sinad_db`.
- Heuristic: higher `sinad_db` is better. >90 dB is decent; >110 dB is strong for high-end.
- If `fundamental_level_dbfs` deviates from expected, check level matching.

### Notch Preservation Score (NPS)
- What: how much the reference notch is filled in the DUT. `nps_db` (smaller/negative is better), `nps_ratio`.
- Heuristic: ≥0 dB means the notch is filling; >+3 dB suggests IMD/noise pollution.
- `is_noise_limited` true means noise floor is too high to trust depth.

### PSD Notch Depth (high-Q)
- What: Welch PSD around the notch center vs surrounding ring; narrowband depth for high-Q notches. Keys under `notch_psd`: `notch_fill_db`, `ref_notch_depth_db`, `dut_notch_depth_db`, `notch_bandwidth_hz`, `ring_bandwidth_hz`.
- Heuristic: `notch_fill_db` near 0 means preserved; +6 dB or more indicates notable fill. Negative depth implies the notch is effectively gone.
- Note: More sensitive to PSD resolution; low noise floor or multiple runs improve stability.

### Spectral Entropy ΔSE
- What: entropy difference; flattening increases ΔSE.
- Heuristic: `delta_se_mean` ≥ 0.02 indicates information loss; closer to 0 is better.
- Inspect `delta_se_max` or time series for localized issues.

### Modulation Power Spectrum (MPS)
- What: texture similarity. `mps_correlation` (→1 good), `mps_distance` (→0 good).
- Heuristic: correlation ≥0.9 good; <0.8 suggests modulation texture degradation. Use `band_correlations` to locate bands.

### Temporal Fine Structure (TFS)
- What: high-band phase coherence and group-delay stability. `mean_correlation` (→1 good), `phase_coherence`, `group_delay_std_ms`.
- Heuristic: correlation ≥0.85 good; `group_delay_std_ms` > ~0.2 ms indicates notable inter-band delay spread. Check `band_group_delays_ms`.

## Reading examples
- “NPS +4 dB, ΔSE +0.03”: notch is filled and entropy degrades—likely dynamic IMD or added noise.
- “MPS corr 0.75, TFS corr 0.8”: both texture and high-band phase are degraded—could be heavy feedback or bandwidth limits.
- “THD/SINAD good but ΔSE/MPS/TFS degraded”: steady-state is fine but microstructure is harmed; re-check filters/gain structure.

## Residual risks / cautions
- Low-SNR references lower confidence for all metrics.
- Large drift or failed alignment: verify `drift`/`validation` and re-measure or adjust parameters.
- Extreme out-of-band noise or clipping can exaggerate entropy/notch penalties.

## Related docs
- Signal specs (EN): `docs/en/signal-specifications.md`
- Measurement setup (EN): `docs/en/measurement-setup.md`
- CLI/API options (JP): `docs/jp/api-cli-reference.md`
