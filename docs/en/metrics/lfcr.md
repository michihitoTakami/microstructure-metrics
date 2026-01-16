# Low-Frequency Complex Reconstruction (LFCR)

## 1. Overview

### Purpose and Significance

**Low-Frequency Complex Reconstruction (LFCR)** evaluates how faithfully a device
reproduces **low-frequency waveform structure** (bass) when the signal contains
non-trivial phase / FM / PM variation. Unlike steady-state metrics (e.g. THD+N),
LFCR targets degradation that often shows up as “loose”, “blurred”, or
“phase-warped” bass even when harmonic distortion is low.

LFCR is designed to be sensitive to:
- Low-frequency group delay / phase distortion
- Poor interpolation / insufficient taps in resampling or reconstruction
- Bass envelope glitches (local instability) that do not appear as simple noise

### What it Measures

LFCR operates on one channel at a time (after alignment) and summarizes three
complementary checks, computed per low-frequency band:

1. **Cycle-shape correlation**: compare the waveform shape of each cycle after
   reparameterizing by instantaneous phase (phase-conditioned resampling).
2. **Harmonic phase coherence**: compare relative phase relationships between
   harmonics (2nd..N) conditioned on the fundamental.
3. **Envelope-difference outlier rate**: detect local envelope-gradient
   mismatches (glitches / instability) relative to the reference baseline.

### Typical Applications

- Evaluating resampler / reconstruction quality in bass-heavy content
- Detecting phase-warping or low-frequency timing smear in DAC/AMP chains
- Comparing bass “tightness” between devices under the same stimulus

---

## 2. Mathematical Definition

Let \(x^{(\mathrm{ref})}[n]\) and \(x^{(\mathrm{dut})}[n]\) be aligned 1-D
signals sampled at \(f_s\) Hz.

### 2.1 Bandpass Filtering

For each low-frequency band \(b = [f_{\ell}, f_h]\), apply a Butterworth
bandpass filter and use zero-phase filtering:

$$
y_b[n] = \mathrm{filtfilt}\left(\mathrm{Butter}(n, f_{\ell}, f_h), x[n]\right)
$$

The implementation uses second-order sections and `sosfiltfilt`.

### 2.2 Analytic Signal, Envelope, and Phase

For the reference band signal \(y_b^{(\mathrm{ref})}[n]\), form the analytic
signal via the Hilbert transform:

$$
z_b[n] = y_b^{(\mathrm{ref})}[n] + j\,\mathcal{H}\{y_b^{(\mathrm{ref})}[n]\}
$$

Envelope and unwrapped phase:

$$
A_b[n] = |z_b[n]|,\quad \phi_b[n] = \mathrm{unwrap}(\arg(z_b[n]))
$$

### 2.3 Cycle Segmentation (Phase-Conditioned)

Define a cycle index from the accumulated phase:

$$
c[n] = \left\lfloor \frac{\phi_b[n] - \phi_b[0]}{2\pi} \right\rfloor
$$

For each cycle \(c\), accept it if:
- it has at least 2 samples,
- its phase span is at least \(0.75\cdot 2\pi\),
- its mean envelope exceeds a threshold \(A_{\min}\).

The envelope threshold is derived from the global peak level of
\(x^{(\mathrm{ref})}\) and \(x^{(\mathrm{dut})}\) using
`envelope_threshold_db`.

### 2.4 Cycle Shape Resampling on a Phase Grid

For an accepted cycle, build a phase grid of \(P\) points:

$$
\theta_k = \frac{2\pi k}{P}, \quad k = 0,\ldots,P-1
$$

Let \(\phi_c[n]\) be the phase within the cycle shifted so it starts at 0.
Interpolate both reference and DUT band signals on \(\theta_k\):

$$
s^{(\mathrm{ref})}_c[k] = \mathrm{interp}(\theta_k,\phi_c, y_b^{(\mathrm{ref})}),
\quad
s^{(\mathrm{dut})}_c[k] = \mathrm{interp}(\theta_k,\phi_c, y_b^{(\mathrm{dut})})
$$

### 2.5 Cycle-Shape Correlation

Compute the Pearson correlation for each cycle:

$$
\rho_c =
\frac{\sum_k (s^{(\mathrm{ref})}_c[k]-\bar{s}^{(\mathrm{ref})}_c)
(s^{(\mathrm{dut})}_c[k]-\bar{s}^{(\mathrm{dut})}_c)}
{\sqrt{\sum_k (s^{(\mathrm{ref})}_c[k]-\bar{s}^{(\mathrm{ref})}_c)^2}
\sqrt{\sum_k (s^{(\mathrm{dut})}_c[k]-\bar{s}^{(\mathrm{dut})}_c)^2}}
$$

Use the mean envelope in the cycle as its weight:

$$
w_c = \frac{1}{|c|}\sum_{n\in c} A_b[n]
$$

Aggregate across all cycles and bands:
- `cycle_shape_corr_mean`: weighted mean of \(\rho_c\)
- `cycle_shape_corr_p05`: weighted 5th percentile of \(\rho_c\)

### 2.6 Fundamental Estimation (Per Band)

Estimate the fundamental frequency \(f_0\) from the reference band using a
Hann-windowed FFT, selecting the peak magnitude within a search range.

### 2.7 Harmonic Phase Coherence

Compute FFT phases at the nearest bin for \(f_0\) and each harmonic
\(h f_0\) (for \(h=2,\ldots,H\), below Nyquist).

Define relative harmonic phase (wrapped to \([-\pi,\pi]\)):

$$
\psi^{(\cdot)}_h =
\mathrm{wrap}\left(\angle X^{(\cdot)}(h f_0) - h\,\angle X^{(\cdot)}(f_0)\right)
$$

Then the per-harmonic phase delta is:

$$
\Delta\psi_h = \mathrm{wrap}\left(\psi^{(\mathrm{dut})}_h
- \psi^{(\mathrm{ref})}_h\right)
$$

The coherence score is the resultant length of these phase-difference vectors:

$$
\mathrm{coherence} =
\left|\frac{1}{K}\sum_{h=2}^{H} e^{j\Delta\psi_h}\right|
$$

### 2.8 Envelope-Difference Outlier Rate

Compute envelopes for both signals, normalize by a peak scale, and compare
envelope gradients:

$$
g[n] = \Delta\left(\frac{A[n]}{\mathrm{scale}}\right)
$$

Define a baseline threshold from the reference gradients:

$$
T = P_{95}(|g^{(\mathrm{ref})}|) + \mathrm{median}(|g^{(\mathrm{ref})}|)
$$

The outlier rate is:

$$
r = \frac{1}{N}\sum_n \mathbf{1}\left(|g^{(\mathrm{dut})}[n]
- g^{(\mathrm{ref})}[n]| > T\right)
$$

---

## 3. Implementation Details

Implementation lives in `src/microstructure_metrics/metrics/bass.py` as
`calculate_low_freq_complex_reconstruction()`. In the `report` JSON output this
metric appears under `metrics.bass.*` (LFCR is the conceptual name).

### 3.1 Parameters and Recommended Values

| Parameter | Type | Default | Notes |
|-----------|------|---------|------|
| `sample_rate` | int | – | Sample rate (Hz) |
| `bands_hz` | sequence of pairs | ((20,80),(80,200)) | Low-frequency bands (Hz) |
| `filter_order` | int | 4 | Butterworth bandpass order (zero-phase) |
| `cycle_points` | int | 128 | Samples per phase-conditioned cycle |
| `envelope_threshold_db` | float | -50.0 | Relative threshold vs global peak |
| `harmonic_max_order` | int | 5 | Use harmonics 2..N below Nyquist |
| `fundamental_search_hz` | (float,float) | (30,180) | Fundamental search range (Hz) |

### 3.2 Pseudo Code

The implementation follows the definitions in Section 2:

1. Bandpass `reference`/`dut` per band (`sosfiltfilt`).
2. Extract reference analytic phase/envelope, segment into cycles, and compute
   cycle-shape correlations on a fixed phase grid.
3. Estimate a per-band fundamental from the reference FFT, then compute harmonic
   phase coherence (resultant length of phase-delta vectors).
4. Compute envelope-gradient mismatch outlier rate.
5. Aggregate cycle metrics by cycle envelope weights; aggregate coherence/outlier
   rate by per-band RMS weight.

### 3.3 Edge Cases and Special Handling

- Inputs must be 1-D and equal length; empty inputs raise `ValueError`.
- Bands must satisfy `0 < low < high < Nyquist`; otherwise `ValueError`.
- If no valid cycles are found in a band (too short or too quiet), that band’s
  cycle metrics become 0.0 and contribute no cycle weights.
- If the fundamental estimate is invalid (\(f_0 \le 0\)), harmonic coherence is
  0.0 for that band.

### 3.4 Computational Complexity

For \(B\) bands and \(N\) samples:
- Filtering: \(\mathcal{O}(B\cdot N)\)
- Hilbert + FFT (per band): \(\mathcal{O}(B\cdot N\log N)\)
- Cycle interpolation: \(\mathcal{O}(C\cdot P)\) where \(C\) is used cycles

---

## 4. Interpretation Guidelines

### 4.1 Key Outputs

Higher is better:
- `cycle_shape_corr_mean`, `cycle_shape_corr_p05` (ideal: close to 1.0)
- `harmonic_phase_coherence` (ideal: close to 1.0)

Lower is better:
- `envelope_diff_outlier_rate` (ideal: close to 0.0)

Per-band details are available under `band_metrics`, including `fundamental_hz`
and the harmonic orders used.

### 4.2 Practical Heuristics

- **High mean, low p05**: mostly OK but intermittent cycle-shape failures
  (local glitches, time-varying warp, or occasional clipping).
- **Low harmonic coherence** with decent cycle correlations: waveform shape is
  similar, but harmonic timing/phase relationships drift (phase warping).
- **High outlier rate**: envelope micro-instability; often correlates with
  audible “flutter” or inconsistent bass punch.

LFCR is sensitive to alignment and low-frequency timing; ensure pilot-based
alignment is successful before trusting values.

---

## 5. Recommended Test Signals

LFCR is most informative on low-frequency content with intentional phase/FM/PM
structure.

### 5.1 Signal Types

- **complex-bass** (`complex-bass`): 8-component FM/PM multi-tone in ~30–220 Hz
  (recommended default for LFCR).
- **Kick + bass composites**: real-world bass waveforms with complex phase.
- **Low-frequency multitone** with controlled phase offsets.

### 5.2 Example (CLI)

```
uv run microstructure-metrics generate complex-bass \
  --duration 10 --sample-rate 48000 --lowcut 30 --highcut 220 \
  --output ref.wav

uv run microstructure-metrics report ref.wav dut.wav \
  --output-json report.json --plot
```

---

## 6. References

### Theoretical Background

- **Auditory Science**: Moore, B. C. J. (2012). *An Introduction to the
  Psychology of Hearing* (6th ed.). Brill.

### Implementation References

- **SciPy Signal Processing**: https://docs.scipy.org/doc/scipy/reference/signal.html
- **NumPy FFT**: https://numpy.org/doc/stable/reference/routines.fft.html

### Related Documentation

- **Metrics Interpretation Guide**: `docs/en/metrics-interpretation.md`
- **Signal Specifications**: `docs/en/signal-specifications.md`
- **Measurement Setup**: `docs/en/measurement-setup.md`

### Source Code

- **LFCR (bass) Implementation**: `src/microstructure_metrics/metrics/bass.py`
- **Signal Generation (complex-bass)**: `src/microstructure_metrics/signals/generator.py`
- **LFCR Visualization**: `src/microstructure_metrics/visualization.py`

---

## Appendix: Common LFCR Pitfalls

1. **Using the wrong stimulus**: pure sines or steady noise do not stress LFCR.
2. **Skipping alignment**: small time offsets in bass heavily affect cycle
   comparisons.
3. **Too short duration**: insufficient cycles yield unstable percentiles.
4. **Level mismatch / clipping**: distorts cycle shape and envelope gradients.
