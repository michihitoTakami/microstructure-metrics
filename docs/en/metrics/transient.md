# Transient Metrics

## 1. Overview

### Purpose and Significance

**Transient Metrics** quantify how well devices preserve the sharpness and timing of rapid amplitude changes—such as impulses, clicks, or attack edges—that are critical to musical realism. While traditional metrics like THD+N measure steady-state distortion, transient analysis focuses on **edge fidelity**: the ability to reproduce sharp transients without rounding, smearing, or adding pre-ringing artifacts.

### What it Measures

Transient analysis detects multiple transient events in a signal by scanning the envelope for peaks above a threshold (typically -25 dB), then extracts features for each event:

- **Attack time**: how long the transient takes to rise from 10% to 90% of peak amplitude
- **Low-level attack time**: how long it takes to rise from 0.1% to 10% (captures subtle pre-ringing)
- **Edge sharpness**: maximum slope of the envelope near the peak
- **Width**: duration at 30% of peak amplitude (measures smearing/spreading)
- **Pre-energy fraction**: energy before the peak vs. total energy (detects pre-ringing)
- **Energy skewness**: asymmetry of energy distribution around the peak

By comparing reference and DUT (Device Under Test) events, the metric reveals:

- Edge rounding (slower attack, reduced sharpness)
- Transient smearing (wider peaks)
- Pre-ringing artifacts (increased pre-energy)
- Timing shifts (attack time deltas)

### Why Transients Matter

Human auditory perception is highly sensitive to transient structure because:

- **Spatial localization**: transient timing encodes directional cues
- **Source identification**: attack envelopes distinguish instruments and sounds
- **Dynamic perception**: edge sharpness correlates with perceived clarity and impact
- **Masking**: even slight smearing or pre-ringing can reduce intelligibility or create audible artifacts

Degradation in transient metrics indicates:

- Slew-rate limiting in amplifiers or DACs
- Filter ringing (especially linear-phase FIR or resonant analog filters)
- Windowing artifacts from time-domain processing
- Bandwidth restrictions that blur sharp edges

### Typical Applications

- Detecting slew-rate limitations in amplifiers or DACs
- Assessing filter ringing and pre-echo artifacts
- Comparing reconstruction quality of different sample-rate converters
- Diagnosing edge rounding from aggressive anti-aliasing or low-pass filters
- Evaluating transient preservation in lossy codecs or DSP chains

---

## 2. Mathematical Definition

### 2.1 Envelope Extraction

For a given signal \(x(t)\), compute the envelope via the **Hilbert transform**:

$$
\text{env}(t) = |\mathcal{H}(x(t))|
$$

where \(\mathcal{H}(\cdot)\) denotes the analytic signal. This preserves low-level pre-ringing better than energy-based windowing.

**Optional smoothing** (to reduce noise sensitivity):

$$
\hat{\text{env}}(t) = \text{env}(t) * w(t)
$$

where \(w(t)\) is a Hann window of duration \(T_{\text{smooth}}\) (default: 0.05 ms). For \(T_{\text{smooth}} = 0\), no smoothing is applied.

### 2.2 Peak Detection

Detect transient events by finding local maxima in \(\hat{\text{env}}(t)\) that exceed a threshold:

$$
\text{Threshold} = \max\left(\alpha \cdot \max(\hat{\text{env}}(t)), \quad 0.5 \cdot P_{90}(\hat{\text{env}}(t)), \quad \epsilon\right)
$$

where:
- \(\alpha = 10^{\text{peak\_threshold\_db}/20}\) (default: \(\alpha = 0.0562\) for -25 dB)
- \(P_{90}(\cdot)\) is the 90th percentile (noise floor estimate)
- \(\epsilon = 10^{-12}\) (prevents division by zero)

**Refractory period**: enforce a minimum separation \(T_{\text{refract}}\) (default: 2.5 ms) between consecutive peaks to avoid double-counting.

### 2.3 Feature Extraction

For each detected peak at index \(n_p\), extract features from a local segment of \(\hat{\text{env}}(t)\) around \(n_p\) (typically ±40 ms):

#### A. Peak Value and Time

$$
\text{peak\_value} = \hat{\text{env}}(n_p), \quad \text{peak\_time} = \frac{n_p}{f_s}
$$

#### B. Attack Time (10%–90%)

Find indices where envelope crosses 10% and 90% of peak value before \(n_p\):

$$
n_{10\%} = \max\{n < n_p : \hat{\text{env}}(n) \leq 0.1 \cdot \text{peak\_value}\}
$$

$$
n_{90\%} = \min\{n > n_{10\%} : \hat{\text{env}}(n) \geq 0.9 \cdot \text{peak\_value}\}
$$

$$
\text{attack\_time} = \frac{n_{90\%} - n_{10\%}}{f_s}
$$

#### C. Low-Level Attack Time (0.1%–10%)

To capture subtle pre-ringing:

$$
n_{0.1\%} = \max\{n < n_p : \hat{\text{env}}(n) \leq 0.001 \cdot \text{peak\_value}\}
$$

$$
n_{10\%}^{\text{low}} = \min\{n > n_{0.1\%} : \hat{\text{env}}(n) \geq 0.1 \cdot \text{peak\_value}\}
$$

$$
\text{low\_level\_attack\_time} = \frac{n_{10\%}^{\text{low}} - n_{0.1\%}}{f_s}
$$

#### D. Edge Sharpness

Maximum envelope slope near the peak (in the last 3 ms before \(n_p\)):

$$
\text{edge\_sharpness} = \max_{n \in [n_p - 3\,\text{ms}, n_p]} \left| \frac{d\hat{\text{env}}(n)}{dt} \right| \cdot f_s
$$

#### E. Width at 30%

Find left and right crossings at 30% of peak amplitude:

$$
n_{\text{left}} = \max\{n < n_p : \hat{\text{env}}(n) \leq 0.3 \cdot \text{peak\_value}\}
$$

$$
n_{\text{right}} = \min\{n > n_p : \hat{\text{env}}(n) < 0.3 \cdot \text{peak\_value}\}
$$

$$
\text{width} = \frac{n_{\text{right}} - n_{\text{left}}}{f_s}
$$

#### F. Pre-Energy Fraction

Using raw signal energy within a window ±\(T_{\text{asym}}\) (default: 3 ms) around \(n_p\):

$$
E_{\text{pre}} = \sum_{n=n_p-T_{\text{asym}}}^{n_p-1} x(n)^2, \quad E_{\text{post}} = \sum_{n=n_p+1}^{n_p+T_{\text{asym}}} x(n)^2
$$

$$
\text{pre\_energy\_fraction} = \frac{E_{\text{pre}}}{E_{\text{pre}} + E_{\text{post}}}
$$

#### G. Energy Skewness

Third moment of the energy-weighted time distribution:

$$
\mu = \frac{\sum_{n} (n - n_p) \cdot x(n)^2}{\sum_{n} x(n)^2}, \quad \sigma^2 = \frac{\sum_{n} (n - n_p - \mu)^2 \cdot x(n)^2}{\sum_{n} x(n)^2}
$$

$$
\text{energy\_skewness} = \frac{1}{\sigma^3} \cdot \frac{\sum_{n} (n - n_p - \mu)^3 \cdot x(n)^2}{\sum_{n} x(n)^2}
$$

Negative skewness indicates energy concentration before the peak (pre-ringing).

### 2.4 Event Matching

For each reference event at \(n_{\text{ref}}\), find the nearest DUT event at \(n_{\text{dut}}\) such that:

$$
|n_{\text{dut}} - n_{\text{ref}}| \leq T_{\text{match}} \cdot f_s
$$

where \(T_{\text{match}}\) is the matching tolerance (default: 1.5 ms). Use greedy nearest-neighbour matching to form pairs.

### 2.5 Statistical Summary

For all matched event pairs, compute pairwise differences:

$$
\Delta_{\text{attack}}^{(i)} = \text{attack\_time}_{\text{dut}}^{(i)} - \text{attack\_time}_{\text{ref}}^{(i)}
$$

$$
r_{\text{sharpness}}^{(i)} = \frac{\text{edge\_sharpness}_{\text{dut}}^{(i)}}{\max(\text{edge\_sharpness}_{\text{ref}}^{(i)}, \epsilon)}
$$

$$
r_{\text{width}}^{(i)} = \frac{\text{width}_{\text{dut}}^{(i)}}{\max(\text{width}_{\text{ref}}^{(i)}, \epsilon)}
$$

Summarize using **median**, **mean**, **5th/95th percentiles**, and **standard deviation** to capture central tendency and variability.

**Transient Smearing Index**: median of \(r_{\text{width}}\) (values > 1 indicate widening/smearing).

---

## 3. Implementation Details

### 3.1 Parameters and Recommended Values

| Parameter | Type | Default | Range/Notes |
|-----------|------|---------|-------------|
| `sample_rate` | int | – | Audio sample rate (Hz) |
| `smoothing_ms` | float | 0.05 | Envelope smoothing window (ms); 0 = Hilbert only |
| `peak_threshold_db` | float | -25.0 | Peak detection threshold relative to max envelope (dB) |
| `refractory_ms` | float | 2.5 | Minimum time between consecutive peaks (ms) |
| `match_tolerance_ms` | float | 1.5 | Maximum time difference for matching ref/DUT events (ms) |
| `max_event_duration_ms` | float | 40.0 | Half-window for feature extraction around each peak (ms) |
| `width_fraction` | float | 0.3 | Fraction of peak amplitude for width measurement (0–1) |
| `asymmetry_window_ms` | float | 3.0 | Half-window for pre-energy / skewness computation (ms) |

### 3.2 Pseudo Code

```
function calculate_transient_metrics(reference, dut, sample_rate, params):
  // Envelope extraction
  env_ref = hilbert_envelope(reference)
  env_dut = hilbert_envelope(dut)

  if smoothing_ms > 0:
    env_ref = smooth(env_ref, smoothing_ms)
    env_dut = smooth(env_dut, smoothing_ms)

  // Peak detection
  ref_events = detect_events(reference, env_ref, sample_rate, params)
  dut_events = detect_events(dut, env_dut, sample_rate, params)

  // Event matching
  pairs = match_events(ref_events, dut_events, tolerance=match_tolerance_ms)

  // Compute pairwise metrics
  attack_deltas = [dut.attack_time - ref.attack_time for ref, dut in pairs]
  sharpness_ratios = [dut.edge_sharpness / max(ref.edge_sharpness, eps) for ref, dut in pairs]
  width_ratios = [dut.width / max(ref.width, eps) for ref, dut in pairs]
  pre_energy_deltas = [dut.pre_energy_fraction - ref.pre_energy_fraction for ref, dut in pairs]
  skewness_deltas = [dut.energy_skewness - ref.energy_skewness for ref, dut in pairs]

  // Statistical summary
  return {
    attack_time_delta_ms: median(attack_deltas),
    edge_sharpness_ratio: median(sharpness_ratios),
    transient_smearing_index: median(width_ratios),
    pre_energy_fraction_delta: median(pre_energy_deltas),
    energy_skewness_delta: median(skewness_deltas),
    distribution_stats: {percentile_05, percentile_95, mean, std},
    matched_event_pairs: len(pairs),
    unmatched_ref_events: len(ref_events) - len(pairs),
    unmatched_dut_events: len(dut_events) - len(pairs),
  }
```

### 3.3 Edge Cases and Special Handling

1. **Empty signals or no peaks detected**: Returns zero values for all metrics; inspect signal amplitude and threshold.
2. **Length mismatch**: Raises error; signals must be aligned beforehand (use alignment module).
3. **Low SNR**: High noise floor may cause false peaks; increase `peak_threshold_db` or apply pre-filtering.
4. **Missing matched pairs**: If `unmatched_ref_events` or `unmatched_dut_events` is large, check `match_tolerance_ms` and signal alignment.
5. **Width crosses signal boundary**: Uses available segment up to signal edges; may underestimate width for edge events.

### 3.4 Computational Complexity

- Hilbert transform: \(\mathcal{O}(N \log N)\) (FFT-based)
- Smoothing convolution: \(\mathcal{O}(N \cdot W)\) where \(W\) is window size (typically small)
- Peak detection: \(\mathcal{O}(N)\)
- Feature extraction per event: \(\mathcal{O}(W_{\text{event}})\) (typically 40 ms × sample rate)
- Event matching: \(\mathcal{O}(M \cdot K)\) where \(M\) and \(K\) are number of ref and DUT events

**Overall**: \(\mathcal{O}(N \log N)\) dominated by Hilbert transform.

Typical runtime: <50 ms for 10 s at 48 kHz with ~10 events on modern hardware.

---

## 4. Interpretation Guidelines

### 4.1 Key Metrics (Medians)

| Metric | Symbol | Interpretation |
|--------|--------|----------------|
| **Attack Time Delta** | \(\Delta t_{\text{attack}}\) | Positive → DUT is slower; negative → DUT is faster |
| **Low-Level Attack Time Delta** | \(\Delta t_{\text{low}}\) | Captures subtle pre-ringing (0.1%–10% rise) |
| **Edge Sharpness Ratio** | \(r_{\text{sharpness}}\) | < 1 → rounder edge; > 1 → sharper edge |
| **Transient Smearing Index** | \(r_{\text{width}}\) | > 1 → wider peak (smearing); < 1 → narrower peak |
| **Pre-Energy Fraction Delta** | \(\Delta E_{\text{pre}}\) | Positive → more pre-ringing in DUT |
| **Energy Skewness Delta** | \(\Delta S\) | Negative → more energy shifted before peak |

**Reference values** (median):
- Attack time: typically 0.2–2 ms for clean impulses
- Edge sharpness: depends on signal bandwidth (higher for sharper transients)
- Width: depends on stimulus (impulse: <5 ms; click: 5–20 ms)
- Pre-energy fraction: ~0.5 for symmetric transients; <0.3 indicates pre-ringing

**DUT comparisons**:
- \(\Delta t_{\text{attack}} > 0.1\) ms: noticeable slowing
- \(r_{\text{sharpness}} < 0.9\): significant edge rounding
- \(r_{\text{width}} > 1.1\): noticeable smearing
- \(\Delta E_{\text{pre}} > 0.05\): detectable pre-ringing

### 4.2 Distribution Statistics

- **Percentile 05/95**: captures variability across events
  - Large spread (p95 - p05 > median) indicates inconsistent behavior
  - Check `edge_sharpness_ratio_p05` and `transient_smearing_index_p95` for worst-case events
- **Standard deviation**: high \(\sigma\) suggests non-uniform degradation (e.g., level-dependent slew-rate limiting)

### 4.3 Event Counts

- `matched_event_pairs`: number of successfully paired ref/DUT events
- `unmatched_ref_events`: events in reference not found in DUT (possible clipping or suppression)
- `unmatched_dut_events`: extra events in DUT (possible ringing or spurious peaks)

Large unmatched counts indicate:
- Poor signal alignment (check alignment module)
- Severe distortion or clipping
- Threshold mismatch (adjust `peak_threshold_db`)

### 4.4 Interpretation Tips

**Scenario: \(\Delta t_{\text{attack}} = +0.3\) ms, \(r_{\text{sharpness}} = 0.7\)**
- Likely: Low-pass filter or slew-rate limiting rounds edges
- Check: Frequency response, amplifier slew rate

**Scenario: \(r_{\text{width}} = 1.4\), \(\Delta E_{\text{pre}} = +0.1\)**
- Likely: Filter ringing adds pre-echo and widens transients
- Check: Impulse response for oscillations, consider minimum-phase alternative

**Scenario: Large p95 values but median OK**
- Likely: Intermittent artifacts (e.g., clipping on loud transients, level-dependent nonlinearity)
- Check: Per-event data, correlate with signal level

**Scenario: All metrics ≈ 1.0, deltas ≈ 0**
- Conclusion: Excellent transient preservation

---

## 5. What Transient Reveals with Sample Signals

### 5.1 Using Test Signals from `generate.py`

The repository includes several test signal types (see `src/microstructure_metrics/cli/generate.py`) that are useful for transient analysis:

#### A. **Impulse** (`impulse`)

**Generator Parameters** (defaults):
- Single Dirac delta (1 sample wide) or short pulse (few samples)
- Amplitude: -1 dBFS

**What Transient reveals**:
- **Cleanest reference**: sharp edge with minimal smoothing
- **Attack time**: typically <0.1 ms for ideal impulse
- **Edge sharpness**: maximum possible for given sample rate
- If DUT shows \(\Delta t_{\text{attack}} > 0.5\) ms or \(r_{\text{sharpness}} < 0.8\), suspect anti-aliasing filter or DAC reconstruction filter

**Example workflow**:
```bash
# Generate reference impulse
python -m microstructure_metrics.cli generate impulse \
  --duration 1 --sample-rate 48000 \
  --output ref_impulse.wav

# Pass through device and compare
python -m microstructure_metrics.cli report ref_impulse.wav dut_impulse.wav \
  --metrics transient --output report.json
```

Expected for transparent system: \(\Delta t_{\text{attack}} < 0.1\) ms, \(r_{\text{sharpness}} > 0.95\), \(r_{\text{width}} < 1.1\)

---

#### B. **Tone Burst** (`tone-burst`)

**Generator Parameters** (defaults):
- 8 kHz sine, 10 cycles, ±2 ms Hann window fade
- Creates sharp attack and decay edges

**What Transient reveals**:
- **Two transient events**: attack (rising edge) and release (falling edge)
- **Sensitive to filter ringing**: pre-ringing appears as increased low-level attack time and pre-energy fraction
- **Phase distortion**: can shift energy distribution (skewness delta)

**Example**:
- Reference: clean burst with symmetric energy (\(E_{\text{pre}} \approx 0.5\))
- DUT with linear-phase FIR: \(\Delta E_{\text{pre}} > 0.1\), low-level attack time increases
- DUT with resonant filter: \(r_{\text{width}} > 1.2\), energy skewness changes

---

#### C. **AM Attack** (`am-attack`)

**Generator Parameters** (defaults):
- 1 kHz carrier with amplitude gating
- Attack: 2 ms, Release: 10 ms, Period: 100 ms

**What Transient reveals**:
- **Multiple transient events** (one per gating period)
- **Statistical robustness**: median and percentiles across many events reveal consistent vs. intermittent degradation
- **Sensitive to slew-rate limiting**: \(\Delta t_{\text{attack}}\) increases if device cannot track fast amplitude changes

**Example**:
- Ideal device: all events show consistent metrics (low std, p05 ≈ p95 ≈ median)
- Slew-limited device: \(\Delta t_{\text{attack}}\) > 0.5 ms, \(r_{\text{sharpness}} < 0.8\)
- Level-dependent device: high std in attack time (some events faster than others)

---

### 5.2 Comparison Matrix: What Metrics Imply

| Signal Type | Expected Transient Count | Key Metric | Interpretation if Degraded |
|-------------|--------------------------|------------|----------------------------|
| `impulse` | 1 (single peak) | \(\Delta t_{\text{attack}}\), \(r_{\text{sharpness}}\) | Filter smoothing or DAC reconstruction artifacts |
| `tone-burst` | 2 (attack + release) | \(\Delta E_{\text{pre}}\), low-level attack time | Pre-ringing from linear-phase or resonant filter |
| `am-attack` | ~10–100 (periodic gating) | Distribution stats (p05, p95, std) | Intermittent artifacts, slew-rate limiting |

### 5.3 How to Generate and Analyze Examples

1. **Generate reference signal** (ideal, high-quality output):
   ```bash
   python -m microstructure_metrics.cli generate impulse \
     --duration 1 --sample-rate 48000 \
     --output impulse_ref.wav
   ```

2. **Simulate degradation** (e.g., low-pass filter at 10 kHz):
   ```bash
   sox impulse_ref.wav dut_impulse.wav lowpass 10000
   ```

3. **Compute transient metrics**:
   ```bash
   python -c "
   from microstructure_metrics.metrics.transient import calculate_transient_metrics
   import soundfile as sf
   ref, sr = sf.read('impulse_ref.wav')
   dut, sr = sf.read('dut_impulse.wav')
   result = calculate_transient_metrics(
       reference=ref, dut=dut, sample_rate=sr
   )
   print(f'Attack time delta: {result.attack_time_delta_ms:.3f} ms')
   print(f'Edge sharpness ratio: {result.edge_sharpness_ratio:.3f}')
   print(f'Transient smearing index: {result.transient_smearing_index:.3f}')
   "
   ```

4. **Interpret results**:
   - Attack time delta > 0.2 ms: significant slowing (filter or slew-rate issue)
   - Edge sharpness ratio < 0.85: noticeable edge rounding
   - Transient smearing index > 1.15: clear widening/smearing

---

## 6. References

### Theoretical Background

- **Auditory Perception**: Cariani, P. A., & Delgutte, B. (1996). Neural correlates of the pitch of complex tones. I. Pitch and pitch salience. *Journal of Neurophysiology*, 76(3), 1698–1716. – Temporal envelope and transient coding.
- **Transient Distortion**: Dunn, C., & Hawksford, M. O. (1993). Distortion immunity of MLS-derived impulse response measurements. *Journal of the Audio Engineering Society*, 41(5), 314–335.
- **Slew-Rate Limiting**: Cherry, E. M. (1982). A new distortion mechanism in class B amplifiers. *Journal of the Audio Engineering Society*, 30(11), 794–799.

### Implementation References

- **Scipy Signal Processing**: [https://docs.scipy.org/doc/scipy/reference/signal.html](https://docs.scipy.org/doc/scipy/reference/signal.html) – Hilbert transform, peak detection.
- **NumPy Gradient**: [https://numpy.org/doc/stable/reference/generated/numpy.gradient.html](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html) – Envelope slope computation.

### Related Documentation

- **Metrics Interpretation Guide**: `docs/en/metrics-interpretation.md` – General guidance on transient metrics in the context of other metrics.
- **Signal Specifications**: `docs/en/signal-specifications.md` – Detailed parameters for impulse, tone-burst, and AM-attack test signals.
- **Measurement Setup**: `docs/en/measurement-setup.md` – Practical considerations for level matching and alignment.

### Source Code

- **Transient Implementation**: `src/microstructure_metrics/metrics/transient.py`
  - `calculate_transient_metrics()`: Core transient analysis
  - `TransientEvent`, `TransientResult`: Data structures for event features and comparison results

- **Test Signal Generation**: `src/microstructure_metrics/cli/generate.py`
  - Includes impulse, tone-burst, and am-attack signals for transient evaluation

---

## Appendix: Common Transient Pitfalls

1. **Using stationary signals**: Transient metrics are meaningless on pure sine waves or white noise; always use signals with sharp edges.
2. **Poor alignment**: Misaligned signals lead to false positive timing shifts; always use the alignment module first.
3. **Too-strict threshold**: If no events are detected, lower `peak_threshold_db` (e.g., from -25 dB to -30 dB).
4. **Too-loose matching tolerance**: If `unmatched_dut_events` is high, reduce `match_tolerance_ms` to avoid false matches.
5. **Ignoring distribution stats**: A single median value can hide intermittent artifacts; always inspect p05/p95 and std.
