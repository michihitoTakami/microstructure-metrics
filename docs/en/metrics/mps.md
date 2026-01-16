# Modulation Power Spectrum (MPS)

## 1. Overview

### Purpose and Significance

The **Modulation Power Spectrum (MPS)** quantifies how well the envelope structure of audio signals is preserved. While traditional metrics like THD+N measure steady-state distortion and harmonic content, MPS focuses on the **modulation texture**—the time-varying amplitude variations that carry perceptual information about transients, dynamic range, and musical character.

### What it Measures

MPS analyzes the temporal envelope of narrow audio bands (via a gammatone or mel filterbank) and decomposes the envelope fluctuations into frequency components (modulation frequencies). The result is a 2D spectrum where:

- **Audio frequency axis** (horizontal): center frequencies of the filterbank bands (typically 100–8000 Hz)
- **Modulation frequency axis** (vertical): frequencies of envelope variations (typically 0.5–64 Hz)
- **Power values**: magnitude of modulation content at each audio band and modulation frequency

### Why Modulation Matters

Human listeners are highly sensitive to modulation structure because:
- **Transient details**: Attack shapes, vibrato, tremolo
- **Spectral dynamics**: How energy distribution shifts over time
- **Auditory illusions**: Interplay between loudness fluctuations and frequency content

Degradation in MPS reveals filtering artifacts, bandwidth restrictions, slew-rate limiting, or envelope clipping that might not appear in harmonic distortion measurements.

### Typical Applications

- Evaluating DAC/amp microphone-less perceived fidelity
- Detecting artifacts from lossy codecs or aggressive DSP (EQ, compression)
- Assessing damage from nonlinear systems (soft-clipping, harmonic warping)
- Comparing preserved texture between audio devices under equivalent output levels

---

## 2. Mathematical Definition

### 2.1 Filterbank Analysis

Input signal is decomposed into \(M\) narrow bands using either:

**Gammatone Filterbank** (auditory-inspired):
$$G_i(f) = \frac{f^{n-1} e^{-2\pi B_{i} f / Q_i}}{(f^2 + 2 i \pi B_i f + Q_i^2 B_i^2)^n}$$

where \(Q_i\) is the Q factor (ratio of center frequency to bandwidth), \(B_i\) is bandwidth, and \(n\) is the filter order (typically 4).

**Mel Filterbank** (music-acoustics-inspired):
Triangular filters on mel scale with configurable order and bandwidth scaling factor.

Output: \(m \times N\) matrix where \(m\) is the number of bands and \(N\) is the number of samples.

### 2.2 Envelope Extraction

For each band \(i\), compute the envelope via:

**Hilbert Transform**:
$$\text{env}_i(n) = |\mathcal{H}(x_i(n))|$$

where \(\mathcal{H}(\cdot)\) is the analytic signal.

**Rectification**:
$$\text{env}_i(n) = |x_i(n)|$$

Optional DC removal:
$$\tilde{\text{env}}_i(n) = \text{env}_i(n) - \frac{1}{N}\sum_{k=0}^{N-1} \text{env}_i(k)$$

Optional low-pass filtering (Butterworth, order 4):
$$\hat{\text{env}}_i(n) = \text{LPF}(\tilde{\text{env}}_i(n), f_c)$$

where \(f_c\) is typically 64 Hz (removes rapid transients, preserves musicality).

### 2.3 Modulation Spectrum Computation

For each band \(i\), compute the modulation spectrum via FFT:
$$M_i(f_{\text{mod}}) = |\text{FFT}(\hat{\text{env}}_i(n))|$$

Restrict to modulation frequency range \([f_{\text{mod,low}}, f_{\text{mod,high}}]\), typically \([0.5, 64]\) Hz.

**Power**:
$$P_i(f_{\text{mod}}) = |M_i(f_{\text{mod}})|^2$$

**dB scale**:
$$P_i^{\text{dB}}(f_{\text{mod}}) = 10 \log_{10}(\max(P_i(f_{\text{mod}}), \epsilon))$$

where \(\epsilon = 10^{-12}\) prevents log underflow.

### 2.4 Result Matrix

Shape: \(M \times K\) where \(K\) is the number of modulation frequency bins.

Axes:
- `audio_freqs`: center frequencies of the \(M\) filterbank bands
- `mod_freqs`: modulation frequencies (linear or log-spaced)
- `mps_power`: power values (linear scale)
- `mps_db`: power values (dB scale)

---

## 3. Implementation Details

### 3.1 Parameters and Recommended Values

| Parameter | Type | Default | Range/Notes |
|-----------|------|---------|-------------|
| `sample_rate` | int | – | Audio sample rate (Hz) |
| `audio_freq_range` | tuple | (100, 8000) | Lower and upper audio band limits |
| `mod_freq_range` | tuple | (0.5, 64) | Lower and upper modulation frequency limits |
| `num_audio_bands` | int | 48 | Number of filterbank bands (32–64 typical) |
| `filterbank` | str | "gammatone" | "gammatone" or "mel" |
| `envelope_method` | str | "hilbert" | "hilbert" or "rectify" |
| `envelope_lowpass_hz` | float | 64 | LPF cutoff for envelope (Hz); None to disable |
| `envelope_lowpass_order` | int | 4 | LPF order (Butterworth) |
| `mod_scale` | str | "linear" | "linear" or "log" (log useful for perceptually even spacing) |
| `mps_scale` | str | "power" | "power" (linear) or "log" (dB) |

### 3.2 Pseudo Code

```
function compute_mps(signal, sample_rate, params):
  // Filterbank decomposition
  band_signals = filterbank(signal, num_bands, freq_range)

  // Envelope extraction per band
  for each band:
    env = hilbert(band) or rectify(band)
    if remove_dc:
      env -= mean(env)
    if lowpass_hz is not None:
      env = butterworth_lpf(env, lowpass_hz, order)

  // Modulation FFT
  n_fft = next_power_of_two(length(envelopes[0]))
  mod_spectrum = rfft(envelopes, n_fft)
  mod_freqs = rfftfreq(n_fft, 1/sample_rate)

  // Extract range
  mask = (mod_freqs >= freq_low) & (mod_freqs <= freq_high)
  mps_power = abs(mod_spectrum[:, mask]) ** 2
  mod_freqs = mod_freqs[mask]

  // Optional log rescaling
  if mod_scale == "log":
    log_freqs = logspace(freq_low, freq_high, num_bins)
    mps_power = interpolate_rows(mps_power, mod_freqs, log_freqs)
    mod_freqs = log_freqs

  // Power to dB
  if mps_scale == "log":
    mps_db = 10 * log10(max(mps_power, 1e-12))
  else:
    mps_db = mps_power

  return mps_power, mps_db, audio_freqs, mod_freqs
```

### 3.3 Edge Cases and Special Handling

1. **DC removal**: Always performed by default (`remove_dc=True`) to prevent energy concentration at modulation frequency = 0.
2. **Modulation range filtering**: All bins outside `mod_freq_range` are discarded; raises error if no bins remain.
3. **Log frequency interpolation**: When `mod_scale="log"`, linear interpolation is used on each band row independently to preserve per-band structure.
4. **Empty or single-sample signals**: Raises `ValueError` if signal is empty or shorter than a few samples.

### 3.4 Computational Complexity

- Filterbank analysis: \(\mathcal{O}(N \cdot M \cdot \text{filter\_order})\)
- FFT per band: \(\mathcal{O}(M \cdot N \log N)\)
- Overall: \(\mathcal{O}(N \log N)\) dominated by FFT

Typical runtime: <100 ms for 10 s at 48 kHz on modern hardware.

---

## 4. Interpretation Guidelines

### 4.1 Similarity Metrics

When comparing reference and DUT (Device Under Test):

**Correlation** (higher → better):
$$r_{\text{MPS}} = \frac{\text{cov}(\text{ref}_{\text{norm}}, \text{dut}_{\text{norm}})}{\sigma_{\text{ref}} \cdot \sigma_{\text{dut}}}$$

- **≥ 0.9**: Excellent preservation of modulation texture
- **0.85–0.89**: Very good; minor texture degradation
- **0.8–0.84**: Acceptable; noticeable but not severe texture change
- **< 0.8**: Significant texture loss; likely artifacts or filtering

**Distance** (lower → better):
$$d_{\text{MPS}} = \sqrt{\frac{1}{M \cdot K} \sum_{i,j} (\text{ref}_{i,j} - \text{dut}_{i,j})^2}$$

- Expressed in dB if `mps_scale="log"`
- Complements correlation; a high correlation can mask a globally shifted spectrum.

### 4.2 Band-Specific Analysis

`band_correlations`: dict mapping audio frequency → per-band correlation

- **High (>0.9)**: Band texture is well preserved
- **Low (<0.8)**: Suspect filtering, nonlinearity, or clipping in that band
- Trends across frequency reveal if degradation is bandlimited or systematic

### 4.3 Modulation Weighting

Optional weights can emphasize higher modulation frequencies (e.g., 4–64 Hz), which carry perceptually salient rapid envelope changes:

- `mod_weighting="high_mod"`: applies `[1, 1, ..., 2, 2, ..., 4, 4, ...]` (2x at 4 Hz, 4x at 10 Hz+)
- Useful for detecting subtle artifacts in the "musicality" range

### 4.4 Interpretation Tips

**Scenario: Correlation 0.75, distance moderate**
- Likely: Systematic bandwidth restriction (e.g., 20 kHz cutoff), envelope clipping, or slew-rate limiting
- Check: Band_correlations to see if specific frequency ranges are affected

**Scenario: Correlation 0.9, but band_correlations show one band at 0.7**
- Likely: Localized issue in that band (e.g., resonance, nonlinearity, digital clipping in a specific frequency region)
- Check: Spectrogram of residual signal

**Scenario: All correlations ≥0.95, THD+N and TFS also excellent**
- Conclusion: Device preserves both steady-state and dynamic content; likely high-fidelity

---

## 5. What MPS Reveals with Sample Signals

### 5.1 Using Test Signals from `generate.py`

The repository includes several test signal types (see `src/microstructure_metrics/cli/generate.py`) that are useful for understanding MPS:

#### A. **Modulated Signal** (`modulated`)

**Generator Parameters** (defaults):
- Carrier: 1000 Hz sine
- AM Modulation: 4 Hz depth 0.5 (50%)
- FM Modulation: ±50 Hz deviation

**What MPS reveals**:
- Clear **4 Hz peak** in modulation spectrum across audio bands
- Confirms envelope modulation is correctly extracted
- Reference vs. DUT comparison shows if modulation is preserved or smeared
- If AM/FM content is attenuated or frequency-shifted, MPS correlation drops significantly

**Example workflow**:
```bash
# Generate reference modulated signal
python -m microstructure_metrics.cli generate modulated \
  --duration 10 --sample-rate 48000 \
  --carrier 1000 --am-freq 4 --am-depth 0.5 \
  --output ref_modulated.wav

# Pass through a simulated device (e.g., 16-bit quantization, low-pass filter)
# Generate test file and compute MPS
python -m microstructure_metrics.cli report ref_modulated.wav dut_modulated.wav \
  --metrics mps --output report.json
```

Expected MPS correlation for undistorted signal: **> 0.95**

---

#### B. **AM Attack** (`am-attack`)

**Generator Parameters** (defaults):
- 1 kHz carrier with amplitude gating
- Attack: 2 ms, Release: 10 ms, Period: 100 ms
- Creates repeating envelope with clear rise/fall

**What MPS reveals**:
- Modulation spectrum shows **~10 Hz fundamental** (gating repetition rate) plus harmonics
- Sensitive to **edge rounding** or attack slowness in devices with low slew rate
- If device has slow response, attack edges blur → modulation energy spreads to lower frequencies
- Correlation degradation indicates loss of transient sharpness

**Example**:
- Reference (`am-attack` defaults): MPS shows sharp concentration at 10 Hz + harmonics
- DUT with 1 ms slew-rate limit: MPS broadens, energy shifts downward, correlation ~0.8–0.85
- Device with extreme low-pass: MPS flattens, correlation < 0.7

---

#### C. **Notched-Noise** (`notched-noise`)

**Generator Parameters**:
- White noise, 20–20k Hz
- Notch filter at center frequency (e.g., 8000 Hz) with Q = 8.6
- Removes narrowband content, preserves surrounding modulation

**What MPS reveals**:
- Modulation spectrum is broadband (0.5–64 Hz) but shows **local depth** at the notch audio frequency
- Ideal for stress-testing envelope extraction around filtering artifacts
- If device preserves the notch shape, MPS band_correlations at notch frequency ≈ correlation at surrounding bands
- Nonlinear devices may "fill in" the notch → mps_correlation increases artifactually (residual structure reappears)

---

#### D. **Tone Burst** (`tone-burst`)

**Generator Parameters** (defaults):
- 8 kHz sine, 10 cycles, ±2 ms Hann fade
- Creates sharp transient start/stop

**What MPS reveals**:
- Envelope has **rise and decay** → modulation energy in ~10–100 Hz range
- Highly sensitive to **filter ringing** or **phase distortion**
- Device that adds pre-ringing or post-ringing: MPS correlation drops due to envelope shape change
- Excellent for detecting phase nonlinearity disguised as "texture degradation"

---

### 5.2 Comparison Matrix: What Correlations Imply

| Signal Type | Ideal Correlation | Interpretation if < 0.9 |
|-------------|-------------------|-------------------------|
| `modulated` (4 Hz AM) | > 0.95 | Modulation smearing, envelope clipping, or AM reduction |
| `am-attack` (gating) | > 0.93 | Slew-rate limiting, edge rounding, or slow attack/release |
| `tone-burst` (transient) | > 0.92 | Filter ringing, pre-/post-echo, or phase nonlinearity |
| `notched-noise` (broadband with notch) | > 0.90 | Notch filling (nonlinearity), envelope artifacts, or bandwidth extension |
| `pink-noise` (natural) | > 0.88 | General envelope fidelity; less sensitive than modulated signals |

### 5.3 How to Generate and Analyze Examples

1. **Generate reference signal** (ideal, high-quality output):
   ```bash
   python -m microstructure_metrics.cli generate modulated \
     --duration 10 --sample-rate 48000 \
     --carrier 1000 --am-freq 4 --am-depth 0.5 \
     --output modulated_ref.wav
   ```

2. **Simulate a degraded version** (e.g., low-pass filter at 16 kHz):
   ```bash
   # Use sox or your own DSP to create dut_modulated.wav
   sox modulated_ref.wav dut_modulated.wav lowpass 16000
   ```

3. **Compute MPS and similarity**:
   ```bash
   python -c "
   from microstructure_metrics.metrics.mps import calculate_mps_similarity
   import soundfile as sf
   ref, sr = sf.read('modulated_ref.wav')
   dut, sr = sf.read('dut_modulated.wav')
   result = calculate_mps_similarity(
       reference=ref, dut=dut, sample_rate=sr,
       audio_freq_range=(100, 8000), mod_freq_range=(0.5, 64)
   )
   print(f'MPS Correlation: {result.mps_correlation:.3f}')
   print(f'Band correlations (sample): {list(result.band_correlations.items())[:3]}')
   "
   ```

4. **Interpret results**:
   - If correlation drops to 0.70: envelope significantly altered by DSP
   - If only high-frequency bands degrade: mid-band to treble filtering
   - If all bands degrade uniformly: global nonlinearity or level mismatch

---

## 6. References

### Theoretical Background

- **Auditory Science**: Moore, B. C. J. (2012). *An Introduction to the Psychology of Hearing* (6th ed.). Brill. – Modulation transfer function and envelope perception.
- **Gammatone Filterbank**: Holdsworth, J., Objé, C., Patterson, R., & Moore, B. C. (1988). Comparison of some auditory filter models. *Hearing Research*, 47(2–3), 103–120.
- **MPS in Codec Evaluation**: Thiede, T., Treurniet, W. C., Bitto, R., Schmidmer, C., Sporer, T., Beerends, J. G., & Colomes, C. (2000). PEAQ - The ITU standard for objective measurement of perceived audio quality. *Journal of the Audio Engineering Society*, 48(1/2), 3–29.

### Implementation References

- **Scipy Signal Processing**: [https://docs.scipy.org/doc/scipy/reference/signal.html](https://docs.scipy.org/doc/scipy/reference/signal.html)
- **NumPy FFT**: [https://numpy.org/doc/stable/reference/routines.fft.html](https://numpy.org/doc/stable/reference/routines.fft.html)
- **Filterbank Design**: Vilkamo, J., Mäkinen, T., & Huopaniemi, J. (2006). Reconstruction of time-frequency representation for audio via Fourier inverse transform. *Proc. DSP*, 1–7.

### Related Documentation

- **Metrics Interpretation Guide**: `docs/en/metrics-interpretation.md` – General guidance on MPS in the context of other metrics.
- **Signal Specifications**: `docs/en/signal-specifications.md` – Detailed parameters for each test signal type.
- **Measurement Setup**: `docs/en/measurement-setup.md` – Practical considerations for level matching and alignment.

### Source Code

- **MPS Implementation**: `src/microstructure_metrics/metrics/mps.py`
  - `calculate_mps()`: Core MPS computation
  - `calculate_mps_similarity()`: Comparative analysis

- **Test Signal Generation**: `src/microstructure_metrics/cli/generate.py`
  - Includes modulated, am-attack, tone-burst, and other signals for MPS evaluation

- **Filterbank**: `src/microstructure_metrics/filterbank/` – Gammatone and Mel filterbank implementations

---

## Appendix: Common MPS Pitfalls

1. **Forgetting DC removal**: Leads to huge energy at modulation frequency = 0, distorting the interpretation.
2. **Using the wrong filterbank**: Mel vs. Gammatone can give different band correlations; document which you used.
3. **Modulation range too narrow**: (e.g., 4–16 Hz) misses high-frequency texture; use at least 0.5–64 Hz.
4. **Comparing signals at vastly different levels**: Always level-match reference and DUT before computing similarity.
5. **Ignoring band_correlations**: A single overall correlation value can hide localized problems; always inspect per-band data.
