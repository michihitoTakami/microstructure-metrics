# Temporal Fine Structure (TFS)

## 1. Overview

### Purpose and Significance

The **Temporal Fine Structure (TFS)** metric quantifies the preservation of high-frequency phase coherence in audio signals. Unlike traditional metrics that focus on steady-state distortion or envelope characteristics, TFS specifically targets the **fine-grained temporal structure**—the rapid oscillations within narrow frequency bands that carry critical perceptual information about tonal clarity, spatial localization, and "air" or "brilliance" in audio reproduction.

### What it Measures

TFS analyzes narrow frequency bands (typically 2–8 kHz) and separates each band into:
- **Envelope**: The slow-varying amplitude contour
- **Fine structure**: The normalized rapid oscillations (carrier-like waveform)
- **Instantaneous phase**: The unwrapped phase trajectory

The metric then computes **short-time cross-correlation (STCC)** between reference and DUT fine structures across multiple time frames, weighted by envelope energy. This reveals:
- **Temporal stability**: How consistently the fine structure is preserved across time
- **Phase coherence**: Whether the instantaneous phase relationship remains stable
- **Group delay variation**: Inter-band timing discrepancies that suggest filter artifacts

### Why TFS Matters

Human auditory perception is highly sensitive to temporal fine structure in the 2–8 kHz range because:
- **Pitch perception**: TFS cues contribute to pitch discrimination, especially for complex tones
- **Spatial hearing**: Interaural time differences (ITD) at high frequencies rely on envelope, but TFS supports low-frequency ITD cues via "binaural sluggishness"
- **Timbre and clarity**: The precise phase relationships between harmonics determine perceived brightness, harshness, or smoothness
- **Transient detail**: Attack characteristics of percussive sounds depend on preserved TFS

Degradation in TFS reveals:
- **Phase distortion**: Non-minimum-phase filters, all-pass artifacts, or group delay irregularities
- **Jitter and modulation**: Clock instability, intermodulation, or FM distortion
- **Slew-rate limiting**: Bandwidth restrictions that blur rapid oscillations
- **Nonlinear artifacts**: Harmonic generation or crossover distortion that disrupts carrier structure

### Typical Applications

- Evaluating DAC reconstruction filters (linear-phase vs minimum-phase trade-offs)
- Detecting jitter, phase noise, or clock instability
- Assessing the impact of resampling, bit-depth reduction, or lossy codecs
- Comparing analog amplifier designs (especially class-D vs class-A/B at high frequencies)
- Diagnosing group delay anomalies from equalization or crossover networks

---

## 2. Mathematical Definition

### 2.1 Bandpass Filtering

The input signal \(x(t)\) is decomposed into \(K\) narrow frequency bands using Butterworth bandpass filters:

$$
H_k(s) = \frac{(s/\omega_c)^n}{(s/\omega_c)^n + \cdots + 1}
$$

where:
- \(k\) indexes the band with center frequency \(f_k\) and bandwidth \(\Delta f_k\)
- Band edges: \(f_{\text{low},k} = f_k - \Delta f_k/2\), \(f_{\text{high},k} = f_k + \Delta f_k/2\)
- Filter order: typically \(n = 6\) (zero-phase via `sosfiltfilt`)

**Default bands** (can be customized):
- Band 1: 2000–3000 Hz
- Band 2: 3000–4000 Hz
- Band 3: 4000–6000 Hz
- Band 4: 6000–8000 Hz

Output: Band-limited signal \(x_k(t)\) for each band \(k\).

### 2.2 Hilbert Transform and TFS Extraction

For each band \(k\), compute the **analytic signal** via Hilbert transform:

$$
z_k(t) = x_k(t) + j \mathcal{H}[x_k(t)]
$$

where \(\mathcal{H}[\cdot]\) is the Hilbert transform.

**Envelope**:
$$
A_k(t) = |z_k(t)| = \sqrt{\text{Re}[z_k(t)]^2 + \text{Im}[z_k(t)]^2}
$$

**Instantaneous phase**:
$$
\phi_k(t) = \text{unwrap}(\arg[z_k(t)])
$$

**Fine structure** (normalized carrier):
$$
\text{TFS}_k(t) = \frac{\text{Re}[z_k(t)]}{\max(A_k(t), \epsilon)}
$$

where \(\epsilon = 10^{-12}\) prevents division by zero in silent regions.

The fine structure \(\text{TFS}_k(t)\) is a zero-mean, unit-amplitude oscillation that captures the "carrier" waveform within the envelope.

### 2.3 Short-Time Correlation (STCC)

Divide the signal into overlapping frames:
- **Frame length**: \(L\) samples (default: 25 ms)
- **Hop size**: \(H\) samples (default: 10 ms)
- **Window**: Hann window \(w(n)\) to reduce edge artifacts

For each frame \(m\) starting at sample \(n_m\):

**Windowed fine structure**:
$$
\text{TFS}_{k,m}^{(\text{ref})}(n) = \text{TFS}_k^{(\text{ref})}(n_m + n) \cdot w(n), \quad n = 0, \ldots, L-1
$$

$$
\text{TFS}_{k,m}^{(\text{dut})}(n) = \text{TFS}_k^{(\text{dut})}(n_m + n) \cdot w(n)
$$

**Normalized cross-correlation** with lag search:
$$
\rho_{k,m}(\tau) = \frac{\sum_n \text{TFS}_{k,m}^{(\text{ref})}(n) \cdot \text{TFS}_{k,m}^{(\text{dut})}(n - \tau)}{\|\text{TFS}_{k,m}^{(\text{ref})}\| \cdot \|\text{TFS}_{k,m}^{(\text{dut})}\|}
$$

where \(\tau\) ranges over \([-\tau_{\max}, \tau_{\max}]\) (default: \(\tau_{\max} = 1\) ms).

**Per-frame correlation**:
$$
\rho_{k,m} = \max_\tau \rho_{k,m}(\tau), \quad \tau_{k,m} = \arg\max_\tau \rho_{k,m}(\tau)
$$

**Frame weight** (envelope-based):
$$
w_{k,m} = \frac{1}{L} \sum_{n=0}^{L-1} \left[ A_k^{(\text{ref})}(n_m + n) + A_k^{(\text{dut})}(n_m + n) \right] / 2
$$

Frames with \(w_{k,m}\) below a threshold \(T_{\text{env}}\) (default: -40 dB relative to peak) are excluded to avoid correlation artifacts from noise.

### 2.4 Band and Global Aggregation

**Per-band correlation** (weighted mean across frames):
$$
\rho_k = \frac{\sum_m w_{k,m} \cdot \rho_{k,m}}{\sum_m w_{k,m}}
$$

**Per-band group delay** (weighted median of lags):
$$
\tau_k = \text{median}_w(\{\tau_{k,m}\}, \{w_{k,m}\})
$$

**Global mean correlation** (all bands, all frames):
$$
\rho_{\text{mean}} = \frac{\sum_{k,m} w_{k,m} \cdot \rho_{k,m}}{\sum_{k,m} w_{k,m}}
$$

**5th percentile correlation** (worst-case tail):
$$
\rho_{05} = \text{percentile}(\{\rho_{k,m}\}, 5)
$$

**Correlation variance** (temporal stability):
$$
\sigma^2_\rho = \frac{\sum_{k,m} w_{k,m} \cdot (\rho_{k,m} - \rho_{\text{mean}})^2}{\sum_{k,m} w_{k,m}}
$$

### 2.5 Phase Coherence

After compensating for the median lag \(\tau_k\) in each band, compute the **phase difference**:
$$
\Delta \phi_k(t) = \phi_k^{(\text{ref})}(t) - \phi_k^{(\text{dut})}(t - \tau_k)
$$

Wrap to \([-\pi, \pi]\):
$$
\Delta \phi_k(t) \leftarrow \arg(e^{j \Delta \phi_k(t)})
$$

**Circular mean (phase coherence)**:
$$
\text{coherence} = \left| \frac{1}{N_{\text{total}}} \sum_{k,t} e^{j \Delta \phi_k(t)} \right|
$$

where \(N_{\text{total}}\) is the total number of samples across all bands.

Values near 1.0 indicate stable phase alignment; values near 0 indicate random or drifting phase.

### 2.6 Group Delay Statistics

**Group delay standard deviation** (inter-band consistency):
$$
\sigma_{\tau} = \sqrt{\frac{1}{K} \sum_k (\tau_k - \bar{\tau})^2}
$$

where \(\bar{\tau}\) is the mean group delay across bands.

Large \(\sigma_{\tau}\) (e.g., > 0.2 ms) suggests frequency-dependent delay anomalies, often from non-minimum-phase filters or group delay ripple.

---

## 3. Implementation Details

### 3.1 Parameters and Recommended Values

| Parameter | Type | Default | Range/Notes |
|-----------|------|---------|-------------|
| `sample_rate` | int | – | Audio sample rate (Hz) |
| `freq_bands` | list of tuples | [(2000, 3000), (3000, 4000), (4000, 6000), (6000, 8000)] | (low, high) frequency pairs in Hz; must be below Nyquist |
| `filter_order` | int | 6 | Butterworth filter order; higher = steeper rolloff but more group delay ripple |
| `frame_length_ms` | float | 25.0 | Short-time frame duration (ms); trade-off: shorter = better time resolution, longer = better frequency resolution |
| `frame_hop_ms` | float | 10.0 | Frame hop size (ms); typical overlap is 50–75% |
| `max_lag_ms` | float | 1.0 | Maximum lag for correlation search (ms); should cover expected jitter and small alignment errors |
| `envelope_threshold_db` | float | -40.0 | Exclude frames below this dB level (relative to peak envelope); prevents noise-dominated correlation |
| `window` | str | "hann" | Window function for STCC; Hann provides good spectral leakage vs main-lobe width trade-off |

### 3.2 Pseudo Code

```
function calculate_tfs_correlation(reference, dut, sample_rate, params):
  // Initialization
  nyquist = sample_rate / 2
  frame_length_samples = round(params.frame_length_ms * sample_rate / 1000)
  hop_samples = round(params.frame_hop_ms * sample_rate / 1000)
  max_lag_samples = round(params.max_lag_ms * sample_rate / 1000)
  window = hann(frame_length_samples)
  frame_starts = range(0, len(reference) - frame_length_samples + 1, hop_samples)

  // Per-band processing
  for each (low, high) in freq_bands:
    // Extract band
    ref_band = bandpass_filter(reference, low, high, order=params.filter_order)
    dut_band = bandpass_filter(dut, low, high, order=params.filter_order)

    // Hilbert transform
    ref_analytic = hilbert(ref_band)
    dut_analytic = hilbert(dut_band)

    // Envelope and fine structure
    ref_envelope = abs(ref_analytic)
    dut_envelope = abs(dut_analytic)
    ref_fine = real(ref_analytic) / max(ref_envelope, EPS)
    dut_fine = real(dut_analytic) / max(dut_envelope, EPS)

    // Envelope threshold
    peak_envelope = max(max(ref_envelope), max(dut_envelope))
    threshold = peak_envelope * 10^(params.envelope_threshold_db / 20)

    // Short-time correlation
    for each frame_start in frame_starts:
      frame_end = frame_start + frame_length_samples
      envelope_mean = mean((ref_envelope[frame_start:frame_end] + dut_envelope[frame_start:frame_end]) / 2)
      if envelope_mean <= threshold:
        continue  // Skip low-energy frame

      ref_frame = ref_fine[frame_start:frame_end] * window
      dut_frame = dut_fine[frame_start:frame_end] * window

      // Normalized cross-correlation with lag search
      correlation, lag = max_normalized_xcorr(ref_frame, dut_frame, max_lag_samples)

      // Store correlation, lag, and weight
      correlations.append(correlation)
      lags.append(lag)
      weights.append(envelope_mean)

    // Per-band aggregation
    band_correlation[band] = weighted_mean(correlations, weights)
    band_group_delay[band] = weighted_median(lags, weights) * 1000 / sample_rate  // Convert to ms

    // Phase coherence (after lag compensation)
    ref_phase = unwrap(angle(ref_analytic))
    dut_phase = unwrap(angle(dut_analytic))
    lag_samples = weighted_median(lags, weights)
    ref_phase_aligned, dut_phase_aligned = overlap_with_lag(ref_phase, dut_phase, lag_samples)
    phase_diff = wrap(ref_phase_aligned - dut_phase_aligned)  // Wrap to [-pi, pi]
    phase_vector_sum += sum(exp(1j * phase_diff))
    phase_count += length(phase_diff)

  // Global aggregation
  mean_correlation = weighted_mean(all_correlations, all_weights)
  percentile_05_correlation = percentile(all_correlations, 5)
  correlation_variance = weighted_variance(all_correlations, all_weights)
  phase_coherence = abs(phase_vector_sum) / phase_count
  group_delay_std_ms = std(band_group_delay values)

  return result
```

### 3.3 Edge Cases and Special Handling

1. **Empty or low-energy signals**: If all frames in a band are below the envelope threshold, the band correlation defaults to 0.0 and group delay to 0.0 ms. The metric does not fail but reports degraded results.

2. **Misaligned signals**: If reference and DUT have different lengths, the function raises a `ValueError` with message: `"reference/dut length mismatch; align signals first"`. Always pre-align signals using pilot tones or cross-correlation before computing TFS.

3. **Band edges near Nyquist**: If any band's upper edge exceeds Nyquist frequency, the function raises `ValueError`. Users should adjust `freq_bands` based on the sample rate (e.g., for 44.1 kHz, the highest band should not exceed ~20 kHz).

4. **Short signals**: If the signal is shorter than one frame length, the effective frame length is reduced to the signal length, and hop size is set to the frame length (no overlap). A warning is logged if fewer than 3 frames are extracted.

5. **Lag search failure**: If `max_lag_ms` is too restrictive and no valid correlation peak is found, the frame correlation defaults to 0.0. Increase `max_lag_ms` if alignment uncertainty is large.

6. **Negative or zero envelope threshold**: The function enforces that `envelope_threshold_db` must be negative (dB relative). Values ≥ 0 raise `ValueError`.

### 3.4 Computational Complexity

- **Filtering**: \(\mathcal{O}(K \cdot N \cdot n)\) where \(K\) is the number of bands, \(N\) is the signal length, and \(n\) is the filter order.
- **Hilbert transform**: \(\mathcal{O}(K \cdot N \log N)\) (FFT-based)
- **STCC per band**: \(\mathcal{O}(M \cdot L \log L)\) where \(M\) is the number of frames and \(L\) is the frame length (FFT-based correlation)
- **Overall**: \(\mathcal{O}(K \cdot N \log N + K \cdot M \cdot L \log L)\)

Typical runtime: <200 ms for 10 s at 48 kHz, 4 bands, on modern hardware.

**Memory**: Peak memory usage is \(\mathcal{O}(K \cdot N)\) for storing band-filtered signals and analytic transforms.

---

## 4. Interpretation Guidelines

### 4.1 Similarity Metrics

When comparing reference and DUT:

**Mean Correlation** (`mean_correlation`):
$$
\rho_{\text{mean}} \in [0, 1] \quad \text{(higher is better)}
$$

- **≥ 0.90**: Excellent TFS preservation; fine structure is well maintained
- **0.85–0.89**: Very good; minor phase distortion or jitter
- **0.80–0.84**: Acceptable; noticeable but not severe TFS degradation
- **< 0.80**: Significant TFS loss; suspect filter artifacts, jitter, or nonlinear distortion

**5th Percentile Correlation** (`percentile_05_correlation`):
- Captures worst-case frames; useful for detecting intermittent glitches or dropouts
- **≥ 0.80**: Robust; no significant temporal outliers
- **< 0.70**: Some frames exhibit severe TFS breakdown

**Correlation Variance** (`correlation_variance`):
- Measures temporal stability of TFS correlation
- **< 0.01**: Stable; TFS quality is consistent across time
- **> 0.05**: Unstable; sporadic degradation or intermittent artifacts

### 4.2 Band-Specific Analysis

`band_correlations`: dict mapping (low, high) Hz → correlation value

- **All bands ≥ 0.9**: TFS is uniformly preserved across the frequency range
- **One band < 0.8**: Localized issue (e.g., resonance, group delay anomaly, or nonlinearity in that band)
- **High bands < 0.8, low bands ≥ 0.9**: Suggests bandwidth limitation, slew-rate limiting, or HF jitter

**Example**: If `band_correlations[(6000, 8000)] = 0.72` but other bands are ≥ 0.88, investigate:
- DAC reconstruction filter roll-off
- Amplifier slew rate at HF
- Jitter or phase noise concentrated above 6 kHz

### 4.3 Phase Coherence and Group Delay

**Phase Coherence** (`phase_coherence`):
$$
\text{coherence} \in [0, 1] \quad \text{(higher is better)}
$$

- **≥ 0.95**: Stable phase relationship; minimal phase jitter or drift
- **0.85–0.94**: Moderate phase instability; small jitter or group delay ripple
- **< 0.85**: Significant phase distortion; non-minimum-phase filter, all-pass artifacts, or clock instability

**Group Delay Standard Deviation** (`group_delay_std_ms`):
- Measures inter-band timing consistency
- **< 0.1 ms**: Excellent; minimal frequency-dependent delay
- **0.1–0.2 ms**: Acceptable; minor group delay variation (common with sharp filters)
- **> 0.2 ms**: Noticeable; may cause timbral shifts or "smearing" perception

**Per-band Group Delay** (`band_group_delays_ms`):
- Positive delay (> 0): DUT is delayed relative to reference in that band
- Negative delay (< 0): DUT is advanced (less common; may indicate pre-ringing)
- Large variation across bands: Non-flat group delay (e.g., elliptic filter, minimum-phase vs linear-phase mismatch)

### 4.4 Interpretation Tips

**Scenario: Mean correlation 0.88, phase coherence 0.92, group delay std 0.15 ms**
- **Interpretation**: Good overall TFS preservation with minor phase instability and moderate group delay variation. Likely a well-designed minimum-phase filter with slight ripple.
- **Action**: Compare with subjective listening; if no audible artifacts, acceptable. If harsh or unclear, investigate filter design.

**Scenario: Mean correlation 0.75, percentile_05 0.60, variance 0.06**
- **Interpretation**: Significant TFS degradation with intermittent breakdowns and high temporal variability. Suspect jitter, nonlinear distortion, or severe bandwidth limiting.
- **Action**: Check signal path for jitter sources, verify DAC clock quality, and inspect band_correlations to locate frequency regions of concern.

**Scenario: Band correlations [0.92, 0.90, 0.88, 0.70] (2–3, 3–4, 4–6, 6–8 kHz), phase coherence 0.89**
- **Interpretation**: HF band (6–8 kHz) exhibits severe TFS loss while lower bands are well preserved. Phase coherence is moderately affected.
- **Action**: Suspect HF-specific issue: slew-rate limiting, reconstruction filter roll-off, or jitter concentrated at high frequencies. Verify amplifier or DAC specifications.

**Scenario: All correlations ≥ 0.95, phase coherence 0.98, group delay std 0.05 ms**
- **Interpretation**: Excellent TFS preservation; minimal phase distortion and flat group delay. Device maintains fine structure integrity.
- **Action**: Reference-quality performance; no further investigation needed.

---

## 5. What TFS Reveals with Sample Signals

### 5.1 Using Test Signals from `generate.py`

The repository includes several test signal types (see `src/microstructure_metrics/cli/generate.py`) that are useful for understanding TFS:

#### A. **Multi-Tone** (`multitone`)

**Generator Parameters**:
- Multiple sine tones at different frequencies (e.g., 1 kHz, 2 kHz, 4 kHz, 8 kHz)
- Equal or weighted amplitudes

**What TFS reveals**:
- **Harmonic phase relationships**: TFS correlation drops if relative phase between tones is distorted
- **Intermodulation**: Nonlinear devices introduce IM products that disrupt fine structure
- **Group delay variation**: If different frequency components are delayed differently, phase coherence drops

**Expected TFS correlation for undistorted signal**: **> 0.95**

---

#### B. **Tone Burst** (`tone-burst`)

**Generator Parameters** (defaults):
- 8 kHz sine, 10 cycles, ±2 ms Hann fade
- Creates sharp transient start/stop

**What TFS reveals**:
- **Filter ringing**: Pre-ringing or post-ringing from sharp filters shows up as phase distortion
- **Phase nonlinearity**: Non-minimum-phase filters cause TFS correlation to drop during transient edges
- **Slew-rate limiting**: Rapid oscillations are blurred → lower TFS correlation

**Example**:
- Reference (`tone-burst` defaults): TFS correlation > 0.93
- DUT with minimum-phase filter (no pre-ring): TFS correlation ~0.90–0.92
- DUT with linear-phase FIR (pre-ringing): TFS correlation ~0.80–0.85, phase coherence drops

---

#### C. **Swept Sine** (`sweep`)

**Generator Parameters**:
- Frequency sweep from 20 Hz to 20 kHz over 10 seconds
- Logarithmic or linear sweep

**What TFS reveals**:
- **Frequency-dependent distortion**: TFS band correlations reveal which frequency regions are affected
- **Group delay anomalies**: Rapid phase changes during sweep cause TFS correlation to drop at specific frequencies
- **Nonlinear distortion**: Harmonic generation disrupts fine structure, especially at low frequencies (subharmonics) and high frequencies (aliasing)

**Example**:
- Reference (`sweep` defaults): TFS correlation > 0.90 across all bands
- DUT with group delay ripple: Band correlations vary (e.g., 0.92, 0.88, 0.85, 0.78 for 2–3, 3–4, 4–6, 6–8 kHz)
- DUT with HF roll-off: High band (6–8 kHz) correlation drops to < 0.75

---

#### D. **AM-Modulated Tone** (`modulated`)

**Generator Parameters**:
- Carrier: 4 kHz sine
- AM modulation: 10 Hz, depth 0.5 (50%)

**What TFS reveals**:
- **Envelope-carrier separation**: TFS extracts carrier phase independently of envelope; AM distortion shows up as phase jitter
- **Clock jitter**: Modulation sidebands are sensitive to timing errors → TFS correlation drops
- **Intermodulation**: Nonlinear devices mix carrier and modulation, disrupting fine structure

**Example**:
- Reference (`modulated` defaults): TFS correlation > 0.92
- DUT with clock jitter (±1 sample RMS): TFS correlation ~0.85–0.88, phase coherence ~0.90
- DUT with IM distortion: TFS correlation < 0.80, band correlations show localized issues

---

### 5.2 Comparison Matrix: What Correlations Imply

| Signal Type | Ideal Correlation | Interpretation if < 0.85 |
|-------------|-------------------|--------------------------|
| `multitone` | > 0.95 | Phase distortion between tones, IM distortion, or group delay ripple |
| `tone-burst` (transient) | > 0.92 | Filter ringing, phase nonlinearity, or slew-rate limiting |
| `sweep` (broadband) | > 0.90 | Frequency-dependent distortion, group delay anomalies, or HF roll-off |
| `modulated` (AM) | > 0.92 | Clock jitter, IM distortion, or envelope-carrier coupling |

### 5.3 How to Generate and Analyze Examples

1. **Generate reference signal** (ideal, high-quality output):
   ```bash
   python -m microstructure_metrics.cli generate multitone \
     --duration 10 --sample-rate 48000 \
     --frequencies 1000 2000 4000 8000 \
     --output multitone_ref.wav
   ```

2. **Simulate a degraded version** (e.g., add jitter):
   ```bash
   # Example: resample to introduce slight jitter/aliasing
   sox multitone_ref.wav -r 44100 dut_multitone.wav rate -v
   sox dut_multitone.wav -r 48000 dut_multitone_resampled.wav rate -v
   ```

3. **Compute TFS and similarity**:
   ```bash
   python -c "
   from microstructure_metrics.metrics.tfs import calculate_tfs_correlation
   import soundfile as sf
   ref, sr = sf.read('multitone_ref.wav')
   dut, sr = sf.read('dut_multitone_resampled.wav')
   result = calculate_tfs_correlation(
       reference=ref, dut=dut, sample_rate=sr,
       freq_bands=[(2000, 3000), (3000, 4000), (4000, 6000), (6000, 8000)]
   )
   print(f'TFS Mean Correlation: {result.mean_correlation:.3f}')
   print(f'Phase Coherence: {result.phase_coherence:.3f}')
   print(f'Group Delay Std: {result.group_delay_std_ms:.3f} ms')
   print(f'Band correlations: {result.band_correlations}')
   "
   ```

4. **Interpret results**:
   - If mean correlation drops to 0.80: significant TFS degradation from resampling artifacts
   - If phase coherence drops to 0.85: moderate phase jitter introduced
   - If group delay std > 0.2 ms: frequency-dependent timing errors from resampler

---

## 6. References

### Theoretical Background

- **Auditory Science**: Moore, B. C. J. (2012). *An Introduction to the Psychology of Hearing* (6th ed.). Brill. – Temporal fine structure and pitch perception.
- **Phase Perception**: Oxenham, A. J. (2018). How we hear: The perception and neural coding of sound. *Annual Review of Psychology*, 69, 27–50.
- **Binaural Hearing**: Bernstein, L. R., & Trahiotis, C. (2002). Enhancing sensitivity to interaural delays at high frequencies by using "transposed stimuli". *Journal of the Acoustical Society of America*, 112(3), 1026–1036.

### Implementation References

- **Hilbert Transform**: Marple, S. L. (1999). Computing the discrete-time analytic signal via FFT. *IEEE Transactions on Signal Processing*, 47(9), 2600–2603.
- **Scipy Signal Processing**: [https://docs.scipy.org/doc/scipy/reference/signal.html](https://docs.scipy.org/doc/scipy/reference/signal.html)
- **NumPy FFT**: [https://numpy.org/doc/stable/reference/routines.fft.html](https://numpy.org/doc/stable/reference/routines.fft.html)

### Related Documentation

- **Metrics Interpretation Guide**: `docs/en/metrics-interpretation.md` – General guidance on TFS in the context of other metrics.
- **Signal Specifications**: `docs/en/signal-specifications.md` – Detailed parameters for each test signal type.
- **Measurement Setup**: `docs/en/measurement-setup.md` – Practical considerations for level matching and alignment.

### Source Code

- **TFS Implementation**: `src/microstructure_metrics/metrics/tfs.py`
  - `extract_tfs()`: Extract TFS components from a single band
  - `calculate_tfs_correlation()`: Compute TFS correlation, phase coherence, and group delay

- **Test Signal Generation**: `src/microstructure_metrics/cli/generate.py`
  - Includes multitone, tone-burst, sweep, and modulated signals for TFS evaluation

---

## Appendix: Common TFS Pitfalls

1. **Forgetting alignment**: TFS requires pre-aligned signals. Use pilot tones or global cross-correlation before calling `calculate_tfs_correlation()`.

2. **Inappropriate bands**: Choosing bands below 1 kHz or above Nyquist/2 reduces TFS sensitivity. Stick to 2–8 kHz for typical audio.

3. **Frame length too short**: Frames shorter than ~10 ms reduce frequency resolution and increase noise sensitivity. Use at least 20–30 ms.

4. **Max lag too small**: If `max_lag_ms` is smaller than actual jitter or alignment uncertainty, correlation peaks are missed. Increase to ≥1 ms for typical scenarios.

5. **Ignoring band correlations**: A single overall correlation can hide localized problems. Always inspect `band_correlations` to identify frequency-specific issues.

6. **Comparing signals at different levels**: Level mismatch affects envelope weighting. Always level-match reference and DUT before computing TFS.

7. **Ignoring phase coherence**: High correlation but low phase coherence indicates amplitude-preserved but phase-distorted fine structure. Check both metrics.
