# Binaural Cue Preservation (BCP)

## 1. Overview

### Purpose and Significance

**Binaural Cue Preservation (BCP)** evaluates how well a device preserves stereo
spatial cues compared to a reference recording. It focuses on three binaural
quantities that strongly influence perceived imaging and spatial stability:

- **ITD** (Interaural Time Difference): relative timing between L/R
- **ILD** (Interaural Level Difference): relative level between L/R
- **IACC** (Interaural Cross-Correlation): similarity/coherence between L/R

BCP is most useful when you want to detect imaging instability that may not be
visible in single-channel metrics (e.g., subtle channel-dependent delay, level
skew, or stereo decorrelation).

### What it Measures

BCP computes ITD/ILD/IACC for each **time frame** and **audio band** (via an
auditory filterbank), then summarizes how much the DUT deviates from the
reference using energy-weighted statistics.

### Typical Applications

- Detecting stereo image drift/blur introduced by DSP, resampling, or filters
- Comparing stereo stability across DAC/AMP chains
- Stress-testing L/R symmetry and channel coherence

---

## 2. Mathematical Definition

Let \(x_L[n], x_R[n]\) be a stereo signal sampled at \(f_s\) Hz. BCP is computed
for both reference and DUT, then compared.

### 2.1 Filterbank Analysis

Decompose each channel into \(B\) audio bands using a gammatone or mel filterbank:

$$
y_{b,L}[n] = \mathrm{FB}_b(x_L[n]),\quad y_{b,R}[n] = \mathrm{FB}_b(x_R[n])
$$

where \(b=1,\ldots,B\) indexes band center frequencies \(f_b\).

### 2.2 Short-Time Framing and Weights

For each band \(b\), split signals into frames of length \(L\) and hop \(H\).
For a frame \(m\), compute a joint RMS weight:

$$
w_{b,m} = \sqrt{\frac{1}{N}\sum_{n\in \mathrm{frame}} s[n]^2}
$$

where \(s[n]\) stacks ref/dut and L/R samples in that band and frame. Frames are
ignored when \(w_{b,m}\) is below a threshold derived from the global peak and
`envelope_threshold_db`.

### 2.3 ILD (Interaural Level Difference)

Define per-frame ILD as:

$$
\mathrm{ILD}_{b,m} = 20\log_{10}\left(\frac{\mathrm{RMS}(y_{b,L})}{\mathrm{RMS}(y_{b,R})}\right)
$$

### 2.4 ITD and IACC from Cross-Correlation

Compute the normalized cross-correlation between left and right within a lag
window \(\tau\in[-\tau_{\max},\tau_{\max}]\):

$$
\rho_{b,m}(\tau) =
\frac{\sum_n y_{b,L}[n]\,y_{b,R}[n-\tau]}
{\|y_{b,L}\|\cdot\|y_{b,R}\|}
$$

Choose the lag that maximizes the magnitude:

$$
\tau^\* = \arg\max_{\tau} |\rho_{b,m}(\tau)|
$$

Then:

$$
\mathrm{ITD}_{b,m} = \frac{\tau^\*}{f_s}\cdot 1000 \;\;[\mathrm{ms}],\quad
\mathrm{IACC}_{b,m} = |\rho_{b,m}(\tau^\*)|
$$

### 2.5 Reference vs DUT Deltas and Summary Statistics

Compute per-frame deltas:

$$
\Delta \mathrm{ITD}_{b,m} = \mathrm{ITD}^{(\mathrm{dut})}_{b,m} - \mathrm{ITD}^{(\mathrm{ref})}_{b,m}
$$

$$
\Delta \mathrm{ILD}_{b,m} = \mathrm{ILD}^{(\mathrm{dut})}_{b,m} - \mathrm{ILD}^{(\mathrm{ref})}_{b,m}
$$

$$
\Delta \mathrm{IACC}_{b,m} = \mathrm{IACC}^{(\mathrm{dut})}_{b,m} - \mathrm{IACC}^{(\mathrm{ref})}_{b,m}
$$

BCP reports energy-weighted percentiles over \(|\Delta \mathrm{ITD}|\) and
\(|\Delta \mathrm{ILD}|\), plus:

- `iacc_p05`: weighted 5th percentile of \(\mathrm{IACC}^{(\mathrm{dut})}\)
- `delta_iacc_median`: weighted median of \(\Delta \mathrm{IACC}\)
- `itd_outlier_rate`: weighted fraction where \(|\Delta \mathrm{ITD}| > T_{\mathrm{itd}}\)

Per-band stats use the same weighted percentile logic per band.

---

## 3. Implementation Details

Implementation is `src/microstructure_metrics/metrics/binaural.py`:
`calculate_binaural_cue_preservation()`.

In `report` output, BCP lives under `metrics.binaural.summary.*` and
`metrics.binaural.band_stats.*`.

### 3.1 Parameters and Recommended Values

| Parameter | Type | Default | Notes |
|-----------|------|---------|------|
| `sample_rate` | int | – | Sample rate (Hz) |
| `audio_freq_range` | (float,float) | (125, 8000) | Analysis bands (Hz) |
| `num_audio_bands` | int | 16 | Number of filterbank bands |
| `frame_length_ms` | float | 25.0 | Short-time frame length (ms) |
| `frame_hop_ms` | float | 10.0 | Frame hop (ms) |
| `max_itd_ms` | float | 1.0 | Max lag for L/R correlation (ms) |
| `envelope_threshold_db` | float | -50.0 | Drop low-energy frames (relative to peak) |
| `itd_outlier_threshold_ms` | float | 0.2 | Outlier threshold for \(|\Delta \mathrm{ITD}|\) |
| `filterbank` | str | "gammatone" | "gammatone" or "mel" |
| `filterbank_kwargs` | mapping | None | Filterbank-specific parameters |

### 3.2 Algorithm Overview

1. Filterbank-analyze L/R for both reference and DUT.
2. For each band and each time frame:
   - compute RMS weight and skip if below threshold
   - compute ILD from L/R RMS ratio
   - compute ITD and IACC via normalized cross-correlation within `max_itd_ms`
3. Compute ref-vs-dut deltas and summarize by energy-weighted percentiles.

### 3.3 Edge Cases and Special Handling

- Inputs must be stereo arrays `(samples, 2)` with matching shapes.
- If a frame has near-zero energy, ITD/ILD/IACC default to 0.0 (and are typically
  skipped by the envelope threshold).
- `audio_freq_range` is clipped to below Nyquist; invalid ranges raise
  `ValueError`.

### 3.4 Computational Complexity

Let \(N\) be samples, \(B\) bands, and \(M\) frames:

- Filterbank analysis: \(\mathcal{O}(B\cdot N)\)
- Per-frame correlation (FFT-based): \(\mathcal{O}(B\cdot M\cdot L\log L)\)

---

## 4. Interpretation Guidelines

### 4.1 Key Outputs

Lower is better:
- `median_abs_delta_itd_ms`, `p95_abs_delta_itd_ms`, `itd_outlier_rate`
- `median_abs_delta_ild_db`, `p95_abs_delta_ild_db`

Higher is better:
- `iacc_p05` (low tail indicates intermittent decorrelation)

Closer to 0 is better:
- `delta_iacc_median` (systematic decorrelation shows as negative bias)

### 4.2 Practical Heuristics

- Rising `itd_outlier_rate` suggests time-varying channel delay mismatch.
- Large `p95_abs_delta_ild_db` suggests sporadic level imbalance (e.g., limiter,
  channel-dependent EQ).
- Low `iacc_p05` suggests occasional L/R decorrelation and a “blurred” image.

Always inspect per-band `band_stats` to localize whether issues are low-band,
mid-band, or high-band dominant.

---

## 5. Recommended Test Signals

BCP requires stereo stimuli with known ITD/ILD structure.

### 5.1 Signal Types

- **binaural-cues** (`binaural-cues`): band-limited pink noise with injected ITD
  and ILD (designed for BCP).
- Real stereo music with stable imaging (for sanity checks).

### 5.2 Example (CLI)

```
uv run microstructure-metrics generate binaural-cues \
  --duration 10 --sample-rate 48000 --itd-ms 0.35 --ild-db 6 \
  --output ref.wav

uv run microstructure-metrics report ref.wav dut.wav \
  --output-json report.json --plot
```

---

## 6. References

### Theoretical Background

- Blauert, J. (1997). *Spatial Hearing* (Revised ed.). MIT Press.
- Moore, B. C. J. (2012). *An Introduction to the Psychology of Hearing* (6th ed.).
  Brill.

### Implementation References

- SciPy signal correlation: https://docs.scipy.org/doc/scipy/reference/signal.html

### Related Documentation

- Metrics interpretation: `docs/en/metrics-interpretation.md`
- Signal specifications: `docs/en/signal-specifications.md`
- Measurement setup: `docs/en/measurement-setup.md`

### Source Code

- BCP implementation: `src/microstructure_metrics/metrics/binaural.py`
- Signal generation: `src/microstructure_metrics/signals/generator.py`
- BCP visualization: `src/microstructure_metrics/visualization.py`

---

## Appendix: Common BCP Pitfalls

1. **Using mono or collapsed stereo**: BCP requires true 2ch content.
2. **Running `report --channels ch0/ch1`**: stereo-specific metrics become
   unavailable.
3. **Poor alignment or drift**: timing errors can masquerade as ITD changes.
4. **Low-SNR frames**: can produce unstable ITD/IACC; rely on the envelope
   threshold and sufficient signal energy.
