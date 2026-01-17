# Residual Microstructure Information (RMI)

## 1. Overview

### Purpose and Significance

The **Residual Microstructure Information (RMI)** metric characterizes the nature of the difference between a reference signal and a Device Under Test (DUT) after removing the best linear match. While traditional metrics like THD+N measure absolute distortion levels, RMI focuses on the **structure** of the residual signal to distinguish between benign noise and perceptually significant artifacts such as ringing, nonlinear distortion, or modulation smearing.

### What it Measures

RMI first estimates the optimal linear transformation between reference and DUT:
- **Scale** \(a\): amplitude gain
- **Delay** \(\Delta\): micro-delay in samples (typically sub-millisecond)

After removing this best-fit linear match, the residual is defined as:

$$
r(t) = \text{dut}(t) - a \cdot \text{ref}(t - \Delta)
$$

RMI then analyzes this residual \(r(t)\) across three orthogonal dimensions:

1. **Burstiness**: measures impulsiveness via kurtosis, crest factor, and 99th percentile amplitude
2. **Modulation structure**: quantifies high-rate envelope modulation energy (4–64 Hz range)
3. **Whiteness**: evaluates spectral flatness and autocorrelation to distinguish noise from tonal artifacts

### Why RMI Matters

A device may exhibit low RMS error but still introduce perceptually salient artifacts. RMI reveals:
- **Filter ringing**: appears as high kurtosis and autocorrelation peaks
- **Nonlinear distortion**: shows up as structured spectral content (low spectral flatness)
- **Envelope artifacts**: detected via elevated high-modulation-rate energy
- **Phase distortion**: manifests as residual structure rather than white noise

Ideal devices produce noise-like residuals (high spectral flatness, low kurtosis, near-zero autocorrelation peaks). Artifacts produce structured residuals with distinct signatures in RMI metrics.

### Typical Applications

- Detecting subtle ringing or pre-echo from reconstruction filters
- Identifying nonlinear artifacts masked by low THD+N
- Assessing envelope preservation in lossy codecs or dynamic processors
- Comparing residual "cleanliness" across different analog amplifier topologies
- Diagnosing intermodulation or modulation distortion

---

## 2. Mathematical Definition

### 2.1 Best Linear Match Estimation

#### Delay Estimation

The micro-delay \(\Delta\) is estimated via normalized cross-correlation:

$$
\rho(\tau) = \frac{\sum_n \text{ref}(n) \cdot \text{dut}(n - \tau)}{\sqrt{\sum_n \text{ref}(n)^2} \cdot \sqrt{\sum_n \text{dut}(n)^2}}
$$

where \(\tau\) ranges over \([-T_{\max}, T_{\max}]\) (default: \(T_{\max} = 5\) ms).

**Coarse delay** (integer samples):
$$
\tau_{\text{coarse}} = \arg\max_\tau \rho(\tau)
$$

**Refined delay** (sub-sample via parabolic interpolation):

If `refine_delay=True`, fit a parabola to the correlation peak and its neighbors \(\rho(\tau_{\text{coarse}} - 1)\), \(\rho(\tau_{\text{coarse}})\), \(\rho(\tau_{\text{coarse}} + 1)\) to obtain fractional-sample precision:

$$
\Delta_{\text{refined}} = \tau_{\text{coarse}} + \frac{\rho(\tau-1) - \rho(\tau+1)}{2[\rho(\tau-1) - 2\rho(\tau) + \rho(\tau+1)]}
$$

**Residual energy refinement** (optional):

If `refine_fit=True`, perform a local search around \(\Delta_{\text{refined}}\) in steps of 0.05 samples over a window of ±0.75 samples to minimize the residual energy \(E_r\):

$$
\Delta = \arg\min_{\delta \in [\Delta_{\text{refined}} - 0.75, \Delta_{\text{refined}} + 0.75]} E_r(\delta)
$$

where the residual energy is computed after scale fitting (see below).

#### Signal Alignment and Trimming

Shift the reference by \(\Delta\) using linear interpolation:

$$
\text{ref}_{\text{shifted}}(i) = \text{ref}(i - \Delta)
$$

Trim both signals to the valid overlap region where interpolation is well-defined:
- Start index: \(\max(0, \lceil \Delta \rceil)\)
- End index: \(\min(N, \lfloor (N-1) + \Delta \rfloor + 1)\)

This produces aligned signals \(\text{ref}_{\text{aligned}}\) and \(\text{dut}_{\text{aligned}}\) of equal length.

#### Scale Estimation

Compute the least-squares optimal scale \(a\):

$$
a = \frac{\sum_n \text{dut}_{\text{aligned}}(n) \cdot \text{ref}_{\text{aligned}}(n)}{\sum_n \text{ref}_{\text{aligned}}(n)^2}
$$

If the denominator is below a threshold \(\epsilon = 10^{-12}\), set \(a = 0\).

### 2.2 Residual Computation

$$
r(n) = \text{dut}_{\text{aligned}}(n) - a \cdot \text{ref}_{\text{aligned}}(n)
$$

**Basic statistics**:
- RMS: \(\text{RMS}_r = \sqrt{\frac{1}{N} \sum_n r(n)^2}\)
- Peak: \(\text{Peak}_r = \max_n |r(n)|\)

### 2.3 Burstiness / Impulsiveness

#### Kurtosis

Fourth-order moment normalized by variance squared:

$$
\text{Kurtosis} = \frac{\frac{1}{N} \sum_n (r(n) - \mu_r)^4}{\left(\frac{1}{N} \sum_n (r(n) - \mu_r)^2\right)^2}
$$

where \(\mu_r = \frac{1}{N} \sum_n r(n)\) is the mean.

- **Gaussian noise**: \(\approx 3.0\)
- **Impulsive artifacts** (ringing, clicks): \(> 3.5\)
- **Uniform/flat residuals**: \(< 2.5\)

#### Crest Factor

Peak-to-RMS ratio:

$$
\text{Crest} = \frac{\text{Peak}_r}{\max(\text{RMS}_r, \epsilon)}
$$

- **White noise**: \(\approx 3\)–5 (in dB: 10–14 dB)
- **Impulsive**: \(> 6\) (> 15 dB)

#### 99th Percentile Absolute Value

$$
p_{99} = \text{quantile}_{0.99}(|r(n)|)
$$

Captures tail behavior; elevated values indicate occasional large excursions (dropout, clipping, ringing).

### 2.4 Modulation Structure

Extract the envelope of the residual via Hilbert transform:

$$
\text{env}(n) = |\mathcal{H}(r(n))|
$$

Remove DC component:

$$
\tilde{\text{env}}(n) = \text{env}(n) - \frac{1}{N} \sum_k \text{env}(k)
$$

Compute the modulation spectrum via FFT:

$$
M(f_{\text{mod}}) = |\text{FFT}(\tilde{\text{env}})| \quad \text{where } f_{\text{mod}} = \text{FFT frequency}
$$

Power in modulation frequency bands:

$$
E_{\text{band}} = \sum_{f \in [f_{\text{low}}, f_{\text{high}}]} |M(f)|^2
$$

**Total modulation energy** (default band: 0.5–64 Hz):

$$
E_{\text{total}} = E_{[0.5, 64]}
$$

**High-modulation ratios**:

$$
\text{HighModRatio}_{4-64} = \frac{E_{[4, 64]}}{E_{\text{total}}}
$$

$$
\text{HighModRatio}_{10-64} = \frac{E_{[10, 64]}}{E_{\text{total}}}
$$

- **Noise-like residual**: low ratios (\(< 0.5\)); energy spreads across all modulation frequencies including DC
- **Artifact-dominated**: high ratios (\(> 0.7\)); energy concentrates in higher modulation rates, indicating structured envelope variations

### 2.5 Whiteness

#### Spectral Flatness

Measures how uniform the power spectral density is:

$$
\text{SpectralFlatness} = \frac{\exp\left(\frac{1}{K} \sum_{k=0}^{K-1} \log(\text{PSD}(k))\right)}{\frac{1}{K} \sum_{k=0}^{K-1} \text{PSD}(k)}
$$

where \(\text{PSD}(k)\) is the power spectral density (computed via Welch's method with 4096-sample segments).

- **White noise**: \(\approx 1.0\)
- **Tonal artifacts** (ringing, harmonic distortion): \(< 0.5\)
- **Narrowband noise**: \(< 0.7\)

#### Autocorrelation Peak Excess

Compute the autocorrelation of the residual (DC-removed):

$$
\text{AC}(\ell) = \frac{1}{N} \sum_{n=0}^{N-|\ell|-1} r(n) \cdot r(n + |\ell|)
$$

Normalize by \(\text{AC}(0)\):

$$
\hat{\text{AC}}(\ell) = \frac{\text{AC}(\ell)}{\text{AC}(0)}
$$

Find the maximum absolute value excluding lag 0 within a search range \(|\ell| \leq \ell_{\max}\) (default: 20 ms):

$$
\text{ACPeakExcess} = \max_{\ell \neq 0} |\hat{\text{AC}}(\ell)|
$$

$$
\text{ACPeakLag} = \arg\max_{\ell \neq 0} |\hat{\text{AC}}(\ell)|
$$

- **White noise**: \(\text{ACPeakExcess} \approx 0\)
- **Ringing/resonance**: \(\text{ACPeakExcess} > 0.1\), lag corresponds to ringing period
- **Periodic artifacts**: \(\text{ACPeakExcess} > 0.2\)

---

## 3. Implementation Details

### 3.1 Parameters and Recommended Values

| Parameter | Type | Default | Range/Notes |
|-----------|------|---------|-------------|
| `sample_rate` | int | – | Audio sample rate (Hz) |
| `max_delay_lag_ms` | float | 5.0 | Maximum delay search range (ms); covers typical jitter and alignment uncertainty |
| `refine_delay` | bool | True | Enable sub-sample parabolic interpolation of correlation peak |
| `refine_fit` | bool | True | Enable local residual-energy minimization around refined delay |
| `autocorr_max_lag_ms` | float | 20.0 | Maximum lag for autocorrelation whiteness check (ms) |
| `modulation_total_band_hz` | tuple | (0.5, 64.0) | Denominator band for modulation energy ratios (Hz) |
| `modulation_high_band_hz` | tuple | (4.0, 64.0) | Numerator band for high-modulation ratio (4–64 Hz) |
| `modulation_very_high_band_hz` | tuple | (10.0, 64.0) | Numerator band for very-high-modulation ratio (10–64 Hz) |

### 3.2 Algorithm Overview

The RMI calculation follows these steps:

1. **Delay estimation**: Cross-correlation with optional sub-sample refinement
2. **Residual-energy refinement** (optional): Local search to minimize residual RMS
3. **Alignment and trimming**: Linearly interpolate reference by \(\Delta\), trim to valid overlap
4. **Scale fitting**: Least-squares optimal gain \(a\)
5. **Residual computation**: \(r(t) = \text{dut}(t) - a \cdot \text{ref}(t - \Delta)\)
6. **Burstiness metrics**: kurtosis, crest factor, 99th percentile
7. **Modulation structure**: Hilbert envelope → FFT → band energy ratios
8. **Whiteness metrics**: Welch PSD → spectral flatness; autocorrelation → peak excess and lag

Implementation details are available in `src/microstructure_metrics/metrics/residual.py`.

### 3.3 Edge Cases and Special Handling

1. **Empty or single-sample signals**: Raises `ValueError` if reference or DUT is empty or has mismatched length.

2. **Low-energy signals**: If the reference energy is below \(\epsilon = 10^{-12}\), scale defaults to 0 and residual equals DUT (unscaled).

3. **Large delay**: If the estimated delay is so large that trimming leaves insufficient samples, raises `ValueError` with message `"delay too large for trimming"` or `"insufficient samples after delay compensation"`. Increase signal length or reduce `max_delay_lag_ms`.

4. **Autocorrelation with short signals**: If the signal is shorter than the autocorrelation window, the function clamps `autocorr_max_lag_ms` to the available length.

5. **Modulation band out of Nyquist**: If the modulation band's upper frequency exceeds Nyquist, raises `ValueError`. Adjust `modulation_total_band_hz` based on sample rate (typically not an issue since modulation bands are <100 Hz).

6. **Infinite or NaN values**: Scale fitting and all metric computations check for non-finite values and default to 0 if overflow or underflow occurs.

### 3.4 Computational Complexity

- **Delay estimation**: \(\mathcal{O}(N \log N)\) (FFT-based cross-correlation)
- **Residual-energy refinement**: \(\mathcal{O}(k \cdot N)\) where \(k\) is the number of candidate delays (typically ~30)
- **Hilbert transform**: \(\mathcal{O}(N \log N)\) (FFT-based)
- **Welch PSD**: \(\mathcal{O}(N \log N)\) (overlapping FFTs)
- **Autocorrelation**: \(\mathcal{O}(N \log N)\) (FFT-based)
- **Overall**: \(\mathcal{O}(N \log N)\) dominated by FFT operations

Typical runtime: <50 ms for 10 s at 48 kHz on modern hardware.

**Memory**: Peak memory usage is \(\mathcal{O}(N)\) for storing aligned signals and intermediate transforms.

---

## 4. Interpretation Guidelines

### 4.1 Residual Basic Statistics

**RMS** (`residual_rms`):
- Absolute level of the residual in linear scale
- **Lower is better**; compare to reference RMS or expected noise floor
- Example: if reference RMS is 0.1 and residual RMS is 0.001, the device introduces 1% error

**Peak** (`residual_peak`):
- Maximum instantaneous residual amplitude
- Captures worst-case excursions (clipping, dropout, ringing peaks)
- **Lower is better**; ideally <0.01 for high-fidelity devices

### 4.2 Burstiness / Impulsiveness

**Kurtosis** (`kurtosis`):
- **≈ 3.0**: Gaussian-like residual (benign noise)
- **3.5–5.0**: Moderately impulsive (occasional clicks, mild ringing)
- **> 5.0**: Highly impulsive (strong ringing, dropout, clipping artifacts)
- **< 2.5**: Overly flat distribution (may indicate DC offset or extreme limiting)

**Crest Factor** (`crest_factor`):
- **3–5**: Noise-like (10–14 dB)
- **6–10**: Moderately impulsive (15–20 dB)
- **> 10**: Highly impulsive (> 20 dB); suspect ringing or clicks

**99th Percentile** (`p99_abs`):
- Captures tail amplitude without being as sensitive to single-sample outliers as peak
- Compare to RMS: if `p99_abs` is much larger than \(3 \times \text{RMS}\), suspect impulsive artifacts

### 4.3 Modulation Structure

**High-Modulation Ratios** (`high_mod_ratio_4_64`, `high_mod_ratio_10_64`):
- **< 0.5**: Energy is spread across all modulation frequencies including DC; noise-like
- **0.5–0.7**: Moderate structure; some envelope artifacts but not dominant
- **> 0.7**: High concentration in fast modulation rates (4–64 Hz or 10–64 Hz); suspect envelope smearing, modulation distortion, or lossy codec artifacts

These ratios reveal whether the residual has temporal envelope structure (high ratio) or is noise-like (low ratio).

### 4.4 Whiteness

**Spectral Flatness** (`spectral_flatness`):
- **≥ 0.9**: Very white; residual is noise-like across all frequencies
- **0.7–0.89**: Moderate coloration; some spectral structure but not severe
- **< 0.7**: Tonal or narrowband artifacts; suspect harmonic distortion, resonance, or filter ringing

**Autocorrelation Peak Excess** (`autocorr_peak_excess`):
- **< 0.05**: White noise-like; no significant periodic structure
- **0.05–0.1**: Weak periodic component; mild ringing or low-level resonance
- **> 0.1**: Strong periodic structure; suspect filter ringing, resonance, or modulation artifacts

**Autocorrelation Peak Lag** (`autocorr_peak_lag_ms`):
- Time lag of the peak in the autocorrelation (excluding lag 0)
- Corresponds to the period of ringing or resonance artifacts
- Example: `autocorr_peak_lag_ms = 0.5` ms suggests a ~2 kHz ringing artifact

### 4.5 Interpretation Tips

**Scenario: Low residual RMS, kurtosis ≈ 3, spectral flatness ≈ 0.95, autocorr peak ≈ 0**
- Interpretation: Ideal; residual is white noise-like. Device adds minimal structured artifacts.

**Scenario: Moderate residual RMS, kurtosis = 5.5, crest factor = 12 dB, autocorr peak = 0.15 at 0.8 ms**
- Interpretation: Impulsive ringing artifact with period ~0.8 ms (≈1.25 kHz resonance). Likely filter ringing or reconstruction artifact.

**Scenario: High spectral flatness (0.9), but high modulation ratios (0.8)**
- Interpretation: Residual is white in frequency but has structured envelope variations. Suspect envelope clipping or AM distortion.

**Scenario: Low spectral flatness (0.5), low kurtosis (2.5), low autocorr peak (0.03)**
- Interpretation: Tonal coloration without strong periodicity. May indicate low-level harmonic distortion or IMD spread across spectrum.

---

## 5. What RMI Reveals with Sample Signals

### 5.1 Using Test Signals from `generate.py`

The repository includes several test signal types (see `src/microstructure_metrics/cli/generate.py`) that are useful for interpreting RMI:

#### A. **White Noise** (`white-noise`)

**Generator Parameters**:
- Flat spectrum, 20–20k Hz
- Uniform amplitude distribution

**What RMI reveals**:
- **Ideal device**: residual should also be white noise-like (kurtosis ≈ 3, spectral flatness ≈ 0.9, autocorr peak ≈ 0)
- **Nonlinear device**: kurtosis increases (clipping, dropout); spectral flatness decreases (harmonic content)
- **Resonant device**: autocorr peak increases, lag reveals resonance frequency

---

#### B. **Tone Burst** (`tone-burst`)

**Generator Parameters**:
- 8 kHz sine, 10 cycles, ±2 ms Hann window fade

**What RMI reveals**:
- **Ringing**: elevated kurtosis, autocorr peak, and lag corresponding to ringing frequency
- **Pre-ringing** (linear-phase filter): high-mod-ratio increases, spectral flatness decreases
- **Phase distortion**: residual has tonal content (low spectral flatness) even if RMS is low

---

#### C. **Multitone** (`multitone`)

**Generator Parameters**:
- Multiple sine waves at different frequencies (e.g., 100, 500, 1000, 5000 Hz)

**What RMI reveals**:
- **Intermodulation distortion**: spectral flatness decreases, residual has tonal content at IMD frequencies
- **Phase nonlinearity**: high-mod-ratio may increase if IMD products modulate each other
- **Ideal device**: residual remains white with kurtosis ≈ 3

---

#### D. **Swept Sine** (`sweep`)

**Generator Parameters**:
- Logarithmic frequency sweep from 20 Hz to 20 kHz

**What RMI reveals**:
- **Frequency-dependent nonlinearity**: kurtosis and crest factor increase in bands where device is nonlinear
- **Resonance**: autocorr peak increases at specific frequencies
- **Group delay anomalies**: may manifest as modulation structure (high-mod-ratio increase)

---

### 5.2 Comparison Matrix: What Metrics Imply

| Signal Type | Expected RMI (Ideal Device) | Interpretation if Degraded |
|-------------|----------------------------|----------------------------|
| `white-noise` | Kurtosis ≈ 3, SF ≈ 0.9, AC peak ≈ 0 | Nonlinearity (kurtosis > 3.5), resonance (AC peak > 0.1), coloration (SF < 0.7) |
| `tone-burst` | Low kurtosis, SF ≈ 0.9, AC peak ≈ 0 | Ringing (AC peak > 0.1, lag = ringing period), pre-echo (high-mod-ratio > 0.7) |
| `multitone` | Kurtosis ≈ 3, SF ≈ 0.9 | IMD (SF < 0.7), phase distortion (high-mod-ratio > 0.7) |
| `sweep` | Kurtosis ≈ 3, SF ≈ 0.9 | Frequency-dependent distortion (kurtosis varies), resonance (AC peak spikes) |

### 5.3 Example Workflow

```bash
# 1. Generate reference signal
uv run microstructure-metrics generate white-noise \
  --duration 10 --sample-rate 48000 --output ref.wav

# 2. Pass through device or simulate degradation
# (e.g., via loopback recording or DSP processing)

# 3. Compute RMI using CLI report
uv run microstructure-metrics report ref.wav dut.wav \
  --metrics residual --output report.json

# 4. Or use Python API for programmatic access
python -c "
from microstructure_metrics.metrics.residual import calculate_residual_microstructure
import soundfile as sf
ref, sr = sf.read('ref.wav')
dut, _ = sf.read('dut.wav')
result = calculate_residual_microstructure(reference=ref, dut=dut, sample_rate=sr)
print(f'Kurtosis: {result.kurtosis:.3f}')
print(f'Spectral flatness: {result.spectral_flatness:.3f}')
print(f'AC peak excess: {result.autocorr_peak_excess:.3f} at {result.autocorr_peak_lag_ms:.3f} ms')
"
```

---

## 6. References

### Theoretical Background

- **Kurtosis and Impulsiveness**: Hyvarinen, A., & Oja, E. (2000). Independent component analysis: algorithms and applications. *Neural Networks*, 13(4-5), 411-430. – Higher-order statistics for signal characterization.
- **Spectral Flatness**: Johnston, J. D. (1988). Transform coding of audio signals using perceptual noise criteria. *IEEE Journal on Selected Areas in Communications*, 6(2), 314-323. – Tonality estimation via spectral flatness measure.
- **Modulation Transfer Function**: Drullman, R., Festen, J. M., & Plomp, R. (1994). Effect of temporal envelope smearing on speech reception. *Journal of the Acoustical Society of America*, 95(2), 1053-1064. – Perceptual relevance of envelope modulation.

### Implementation References

- **Scipy Signal Processing**: [https://docs.scipy.org/doc/scipy/reference/signal.html](https://docs.scipy.org/doc/scipy/reference/signal.html) – Hilbert transform, Welch PSD, correlation functions.
- **NumPy Statistics**: [https://numpy.org/doc/stable/reference/routines.statistics.html](https://numpy.org/doc/stable/reference/routines.statistics.html) – Kurtosis, percentile computations.

### Related Documentation

- **Metrics Interpretation Guide**: `docs/en/metrics-interpretation.md` – General guidance on RMI in the context of other metrics.
- **Signal Specifications**: `docs/en/signal-specifications.md` – Detailed parameters for each test signal type.
- **Measurement Setup**: `docs/en/measurement-setup.md` – Practical considerations for level matching and alignment.

### Source Code

- **RMI Implementation**: `src/microstructure_metrics/metrics/residual.py`
  - `calculate_residual_microstructure()`: Core RMI computation
  - `ResidualMicrostructureResult`: Data structure for RMI metrics

- **Test Signal Generation**: `src/microstructure_metrics/cli/generate.py`
  - Includes white-noise, tone-burst, multitone, and sweep signals for RMI evaluation

---

## Appendix: Common RMI Pitfalls

1. **Forgetting alignment**: RMI requires aligned signals. Use pilot tones or global cross-correlation before calling `calculate_residual_microstructure()`.

2. **Misinterpreting residual RMS**: A low residual RMS doesn't guarantee clean audio; check kurtosis, spectral flatness, and autocorr peak to assess artifact structure.

3. **Ignoring modulation ratios**: High spectral flatness but high modulation ratios indicate envelope artifacts; both dimensions must be considered.

4. **Using stationary signals only**: RMI is most informative with complex signals (noise, multitone, transients); pure sine waves may hide IMD or envelope issues.

5. **Comparing signals at different levels**: Level mismatch affects scale estimation and residual RMS. Always level-match reference and DUT before computing RMI.

6. **Over-interpreting kurtosis on short signals**: Kurtosis estimates require sufficient samples (>1000) for stability; use longer signals (≥10 s) for reliable kurtosis values.

7. **Autocorr lag out of expected range**: If `autocorr_peak_lag_ms` is very short (<0.1 ms) or very long (>10 ms), verify that the sample rate and `autocorr_max_lag_ms` are correct.
