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

### 3.2 Algorithm Overview

The TFS correlation algorithm follows these steps:

1. **Band decomposition**: Apply Butterworth bandpass filters to extract each frequency band
2. **TFS extraction**: For each band, compute analytic signal via Hilbert transform, then separate envelope \(A(t)\) and fine structure \(\text{TFS}(t) = \text{Re}[z(t)] / A(t)\)
3. **Short-time correlation**: Divide signal into overlapping frames, apply Hann window, compute normalized cross-correlation with lag search \(\rho(\tau)\), and weight by envelope energy
4. **Band aggregation**: Compute weighted mean correlation and weighted median group delay per band
5. **Phase coherence**: After lag compensation, compute circular mean of phase differences \(\Delta\phi(t)\) across all bands
6. **Global statistics**: Aggregate correlations (mean, 5th percentile, variance), group delay standard deviation

Implementation details are available in `src/microstructure_metrics/metrics/tfs.py`.

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
- Good overall TFS preservation with minor phase instability. Likely a well-designed filter with slight ripple. Compare with subjective listening to assess audibility.

**Scenario: Mean correlation 0.75, percentile_05 0.60, variance 0.06**
- Significant TFS degradation with intermittent breakdowns. Suspect jitter, nonlinear distortion, or severe bandwidth limiting. Inspect band_correlations to locate frequency regions of concern.

**Scenario: Band correlations [0.92, 0.90, 0.88, 0.70] for 2–3, 3–4, 4–6, 6–8 kHz**
- HF band exhibits severe TFS loss while lower bands are preserved. Suspect slew-rate limiting, reconstruction filter roll-off, or jitter concentrated at high frequencies.

---

## 5. Recommended Test Signals

TFS metrics are most effective with signals containing high-frequency content (2–8 kHz) and clear phase structure. Suitable test signals include:

### 5.1 Signal Types and Expected Behavior

| Signal Type | Ideal Correlation | What TFS Reveals |
|-------------|-------------------|------------------|
| **Multi-tone** (`multitone`) | > 0.95 | Harmonic phase relationships, intermodulation, group delay variation between frequency components |
| **Tone Burst** (`tone-burst`) | > 0.92 | Filter ringing (pre/post), phase nonlinearity from non-minimum-phase filters, slew-rate limiting |
| **Swept Sine** (`sweep`) | > 0.90 | Frequency-dependent distortion, group delay anomalies, nonlinear artifacts across the spectrum |
| **AM-Modulated Tone** (`modulated`) | > 0.92 | Clock jitter, envelope-carrier coupling, intermodulation distortion |

### 5.2 Usage Notes

- Correlations below the ideal thresholds suggest phase distortion, jitter, or filter artifacts in the corresponding frequency range
- Always inspect `band_correlations` to identify which frequency bands are affected
- For detailed signal parameters and generation commands, see `docs/en/signal-specifications.md` and `docs/en/user-guide.md`

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
