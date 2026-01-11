# Metrics Interpretation (EN)

Purpose: explain how to read the metrics produced by `report` (JSON/CSV/Markdown) when comparing DUT vs reference.

## General guidance
- Values come after pilot-based alignment. Large drift or alignment failure lowers confidence (see `validation` / `drift` in the report).
- Some metrics use dB (bigger magnitude often means more degradation), others use correlation (closer to 1 is better). Check each metric.

## Selecting metrics
- Steady-state distortion / gain sanity → THD+N/SINAD.
- Texture/modulation differences → MPS.
- High-band fine structure/phase stability → TFS.
- Edge rounding / transient smearing → Transient metrics (needs impulse/edge stimulus).
- Bass waveform fidelity → LFCR.
- Spatial cue stability (stereo) → BCP.
- Distribution-distance comparisons → MDI.
- Residual “structure” vs noise → RMI.
- Always use the intended stimulus from `docs/*/signal-specifications.md`; wrong signals make metrics meaningless.

## Metric-specific notes

### THD+N
- What: harmonic + noise vs fundamental. See `thd_n_db`, `thd_n_percent`, `sinad_db`.
- Heuristic: higher `sinad_db` is better. >90 dB is decent; >110 dB is strong for high-end.
- If `fundamental_level_dbfs` deviates from expected, check level matching.

### Modulation Power Spectrum (MPS)
- What: texture similarity. `mps_correlation` (→1 good), `mps_distance` (→0 good).
- Heuristic: correlation ≥0.9 good; <0.8 suggests modulation texture degradation. Use `band_correlations` to locate bands.

### Temporal Fine Structure (TFS)
- What: high-band short-time phase correlation. Check `mean_correlation` (STCC mean), `percentile_05_correlation` (worst-case tail), `correlation_variance`, `phase_coherence`, and `group_delay_std_ms`.
- Heuristic: `mean_correlation` ≥0.85 is healthy; low `percentile_05_correlation` means intermittent breakdowns. `group_delay_std_ms` > ~0.2 ms indicates noticeable inter-band delay spread. `frame_length_ms`, `frame_hop_ms`, `max_lag_ms`, and `envelope_threshold_db` in the report document the STCC settings (low-envelope frames are dropped).

### Transient / Edge rounding
- What: rounded edges in impulses/clicks, now via multi-event scanning. The envelope is scanned with a -25 dB peak threshold, 2.5 ms refractory, 40 ms max event length, and 1.5 ms ref/dut matching tolerance.
- Keys: medians `attack_time_ms` (DUT), `attack_time_delta_ms` (DUT-ref), `edge_sharpness_ratio`, `transient_smearing_index` (width ratio) plus distribution views `edge_sharpness_ratio_p05/p95`, `transient_smearing_index_p95`, `attack_time_delta_p95_ms`, and `event_counts.*`. Width is measured at 30% peak crossings.
- Heuristic: `attack_time_delta_ms` > 0 or `edge_sharpness_ratio` < 1 → slower/rounded edge. `transient_smearing_index` > 1 → wider main lobe (more smearing). Large p95 values mean localized degradation.
- Best for: edge rounding, slew limiting, or windowing that blunts sharp attacks.
- Input requirements: impulse/edge stimulus (single Dirac or steep step per `signal-specifications.md`). Not meaningful on stationary noise or sinusoids.
- Robustness: uses smoothed envelope/energy to tolerate phase jitter and noise; if no events are detected, values fall back to 0.

### Low-Frequency Complex Reconstruction (LFCR)
- What: bass waveform fidelity via phase-conditioned cycle shapes, harmonic phase coherence, and envelope stability in low bands.
- Keys: `cycle_shape_corr_mean` / `cycle_shape_corr_p05` (→1 good), `harmonic_phase_coherence` (→1 good), `envelope_diff_outlier_rate` (→0 good), `bands_hz`, and per-band `band_metrics.*`.
- Defaults: bands 20–80 Hz / 80–200 Hz, Butterworth order 4 (zero-phase), 128 cycle samples, envelope threshold -50 dBFS, harmonic search 30–180 Hz up to 5th order; all recorded in the report.
- Heuristic: values drop when low-tap interpolation or phase-warped bass breaks waveform shape or harmonic alignment; outlier rate >0.1 indicates local envelope glitches.
- Input: complex bass / kick+bass composites or low-frequency multitone with intentional phase/PM/FM variation.

### Binaural Cue Preservation (BCP)
- What: estimates ITD/ILD/IACC over band×time and summarizes deltas vs reference (stereo required).
- Where in JSON: `metrics.binaural.summary.*` and `metrics.binaural.band_stats.*`.
- Keys: `median_abs_delta_itd_ms`, `p95_abs_delta_itd_ms`, `itd_outlier_rate`, `median_abs_delta_ild_db`, `p95_abs_delta_ild_db`, `iacc_p05`, `delta_iacc_median`.
- Heuristic: rising ITD/ILD deltas or outlier rate suggests unstable imaging; low `iacc_p05` suggests looser/blurred image.

### Residual Microstructure Information (RMI)
- What: removes best linear match (scale \(a\), micro delay \(\Delta\)) and analyzes the residual \(r(t)=dut(t)-a\cdot ref(t-\Delta)\) to see if the difference is noise-like or structured (ringing/smear/nonlinear artifacts).
- Where in JSON: `metrics.ch0.residual.*` / `metrics.ch1.residual.*` (CSV/MD flatten nested keys like `residual.whiteness.spectral_flatness`).
- Burstiness: `burstiness.kurtosis`, `burstiness.crest_factor`, `burstiness.p99_abs` (higher → more impulsive/structured).
- Whiteness: `whiteness.spectral_flatness` (closer to 1 → flatter), `whiteness.autocorr_peak_excess` (closer to 0 → more white), `whiteness.autocorr_peak_lag_ms`.
- Modulation structure: `modulation.high_mod_ratio_4_64`, `modulation.high_mod_ratio_10_64` (higher → more high-rate envelope modulation in residual).

### Microstructure Distribution Divergence (MDI)
- What: compares short-time feature distributions from TFS, Transient (and, when available, BCP) using 1D Wasserstein distances so that “mostly OK but sometimes broken” behavior is highlighted.
- Where in JSON: `metrics.divergence`. See `mdi_total`, `channels_total`/`binaural_total`, `component_totals`, and the list of `components` describing per-feature contributions (e.g., `ch0.tfs.correlation_to_ideal`, `ch0.transient.attack_time_ms`, `binaural.ild_db`).
- Heuristic: lower `mdi_total` signals more similar distributions. Component breakdowns show which feature (e.g., TFS correlation → 1.0, band delay → 0 ms, transient attack/width, binaural ITD/ILD/IACC) is drifting; even if averages look fine, a large component indicates sporadic glitched frames or localized degradation.

## Reading examples
- “MPS corr 0.75, TFS corr 0.8”: both texture and high-band phase are degraded—could be heavy feedback or bandwidth limits.
- “THD/SINAD good but MPS/TFS degraded”: steady-state is fine but microstructure is harmed; re-check filters/gain structure.

## Residual risks / cautions
- Low-SNR references lower confidence for all metrics.
- Large drift or failed alignment: verify `drift`/`validation` and re-measure or adjust parameters.
- Extreme out-of-band noise or clipping can exaggerate entropy/notch penalties.

## Related docs
- Signal specs (EN): `docs/en/signal-specifications.md`
- Measurement setup (EN): `docs/en/measurement-setup.md`
- CLI/API options (JP): `docs/jp/api-cli-reference.md`
