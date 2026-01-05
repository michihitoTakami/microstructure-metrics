# Metrics Interpretation (EN)

Purpose: explain how to read the metrics produced by `report` (JSON/CSV/Markdown) when comparing DUT vs reference.

## General guidance
- Values come after pilot-based alignment. Large drift or alignment failure lowers confidence (see `validation` / `drift` in the report).
- Some metrics use dB (bigger magnitude often means more degradation), others use correlation (closer to 1 is better). Check each metric.

## Selecting metrics (EPIC #38 context)
- Broad / low-Q notch fill or added broadband noise → use NPS (robust to moderate SNR).
- Very high-Q notch collapse → use PSD Notch Depth (narrowband Welch PSD; needs higher resolution/longer capture).
- Edge rounding / transient smearing → use Transient metrics (envelope attack/width; requires an impulse/edge stimulus).
- Always run metrics with the intended stimulus from `docs/*/signal-specifications.md` (e.g., `notched_noise` for NPS/PSD notch, impulse-like transient signal for edge tests). Running on the wrong signal produces meaningless values.

## Metric-specific notes

### THD+N
- What: harmonic + noise vs fundamental. See `thd_n_db`, `thd_n_percent`, `sinad_db`.
- Heuristic: higher `sinad_db` is better. >90 dB is decent; >110 dB is strong for high-end.
- If `fundamental_level_dbfs` deviates from expected, check level matching.

### Notch Preservation Score (NPS)
- What: how much the reference notch is filled in the DUT. `nps_db` (smaller/negative is better), `nps_ratio`.
- Heuristic: ≥0 dB means the notch is filling; >+3 dB suggests IMD/noise pollution.
- `is_noise_limited` true means noise floor is too high to trust depth.
- Best for: wide-ish notches (e.g., Q≈6–10) where you want a broadband sanity check that tolerates moderate SNR.
- Input requirements: use the notched-noise stimulus (`signal_type: notched_noise`, e.g., 8 kHz, Q≈8–10). If the capture lacks that notch, NPS is not meaningful.

### PSD Notch Depth (high-Q)
- What: Welch PSD around the notch center vs surrounding ring; narrowband depth for high-Q notches. Keys under `notch_psd`: `notch_fill_db`, `ref_notch_depth_db`, `dut_notch_depth_db`, `notch_bandwidth_hz`, `ring_bandwidth_hz`.
- Heuristic: `notch_fill_db` near 0 means preserved; +6 dB or more indicates notable fill. Negative depth implies the notch is effectively gone.
- Note: More sensitive to PSD resolution; low noise floor or multiple runs improve stability.
- Best for: narrow/high-Q notches (e.g., Q≈20+) where NPS is too coarse.
- Input requirements: same notched-noise stimulus as NPS but with sufficient capture length/FFT resolution (see `signal-specifications.md`). Mismatched notch center/Q makes results invalid.

### Spectral Entropy ΔSE
- What: entropy difference; flattening increases ΔSE.
- Heuristic: `delta_se_mean` ≥ 0.02 indicates information loss; closer to 0 is better.
- Inspect `delta_se_max` or time series for localized issues.

### Modulation Power Spectrum (MPS)
- What: texture similarity. `mps_correlation` (→1 good), `mps_distance` (→0 good).
- Heuristic: correlation ≥0.9 good; <0.8 suggests modulation texture degradation. Use `band_correlations` to locate bands.

### Temporal Fine Structure (TFS)
- What: high-band short-time phase correlation. Check `mean_correlation` (STCC mean), `percentile_05_correlation` (worst-case tail), `correlation_variance`, `phase_coherence`, and `group_delay_std_ms`.
- Heuristic: `mean_correlation` ≥0.85 is healthy; low `percentile_05_correlation` means intermittent breakdowns. `group_delay_std_ms` > ~0.2 ms indicates noticeable inter-band delay spread. `frame_length_ms`, `frame_hop_ms`, `max_lag_ms`, and `envelope_threshold_db` in the report document the STCC settings (low-envelope frames are dropped).

### Transient / Edge rounding
- What: rounded edges in impulses/clicks. Keys: `attack_time_ms` (DUT), `attack_time_delta_ms` (DUT-ref), `edge_sharpness_ratio`, `transient_smearing_index` (width ratio).
- Heuristic: `attack_time_delta_ms` > 0 or `edge_sharpness_ratio` < 1 → slower/rounded edge. `transient_smearing_index` > 1 → wider main lobe (more smearing).
- Best for: edge rounding, slew limiting, or windowing that blunts sharp attacks.
- Input requirements: impulse/edge stimulus (single Dirac or steep step per `signal-specifications.md`). Not meaningful on stationary noise or sinusoids.
- Robustness: uses smoothed envelope/energy to tolerate phase jitter and noise.

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
