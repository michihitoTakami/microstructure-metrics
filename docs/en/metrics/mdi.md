# Microstructure Distribution Divergence (MDI)

## 1. Overview

### Purpose and Significance

The **Microstructure Distribution Divergence (MDI)** is a composite metric that quantifies the overall preservation of microstructure characteristics across temporal and spatial dimensions. Unlike traditional metrics that report only average values, MDI specifically targets **distributional differences**—the "mostly OK but sometimes broken" patterns that reveal intermittent artifacts, sporadic phase instability, or inconsistent transient handling that might be masked by simple averaging.

### What it Measures

MDI aggregates distributional divergences from multiple feature domains:

- **TFS (Temporal Fine Structure)**: Distribution of short-time correlation values across time frames, weighted by envelope energy, plus group delay deviations across frequency bands
- **Transient**: Distribution of attack time and width characteristics across detected transient events
- **Binaural** (stereo only): Distribution of ITD (Interaural Time Difference), ILD (Interaural Level Difference), and IACC (Interaural Cross-Correlation) across time frames

The metric uses **1D Wasserstein distance** (Earth Mover's Distance) to compare empirical distributions between reference and DUT (Device Under Test) signals. Each component divergence is normalized by a perceptually-motivated scale factor and weighted, then summed to produce the total MDI score.

### Why Distribution Matters

Human perception is sensitive to temporal consistency and statistical regularity of audio features:

- **Temporal stability**: Listeners detect sporadic breakdowns in correlation or timing even when average values are acceptable
- **Event consistency**: Transient characteristics (attack sharpness, decay shape) should remain stable across musical passages
- **Spatial coherence**: Binaural cues must be preserved consistently over time to maintain spatial image and localization accuracy

Degradation in MDI reveals:
- **Intermittent artifacts**: Jitter, dropouts, or sporadic nonlinearities that occur only occasionally
- **Statistical shift**: Systematic changes in feature distributions that indicate envelope clipping, slew-rate limiting, or dynamic range compression
- **Multi-domain coupling**: Correlated degradation across TFS, transient, and binaural domains that suggests a common root cause (e.g., NFB instability, clock issues)

### Typical Applications

- Comparing DACs, amplifiers, or signal chains for microstructure fidelity beyond steady-state distortion
- Detecting intermittent artifacts from thermal drift, power supply ripple, or adaptive processing
- Assessing codec quality, especially for perceptual codecs that may introduce time-varying artifacts
- Validating audio processing algorithms (DSP, resampling, dithering) for distributional consistency

---

## 2. Mathematical Definition

### 2.1 Wasserstein Distance (Earth Mover's Distance)

For two empirical distributions \(X = \{x_i\}\) and \(Y = \{y_j\}\) with optional weights \(w^X_i\) and \(w^Y_j\), the **1D Wasserstein distance** is:

$$
W_1(X, Y) = \int_{-\infty}^{\infty} |F_X(t) - F_Y(t)| \, dt
$$

where \(F_X(t)\) and \(F_Y(t)\) are the cumulative distribution functions (CDFs) of \(X\) and \(Y\).

**Discrete implementation**:
1. Sort samples: \(x_{(1)} \leq x_{(2)} \leq \cdots\), \(y_{(1)} \leq y_{(2)} \leq \cdots\)
2. Construct CDFs on the union of support points: \(S = \{x_i\} \cup \{y_j\}\)
3. Compute piecewise-linear CDFs: \(F_X(s)\), \(F_Y(s)\) for \(s \in S\)
4. Integrate absolute difference:

$$
W_1 = \sum_{k=1}^{|S|-1} |F_X(s_k) - F_Y(s_k)| \cdot (s_{k+1} - s_k)
$$

**Properties**:
- **Metric**: Satisfies triangle inequality; \(W_1(X, Y) = 0\) iff \(X\) and \(Y\) are identical
- **Robust**: Less sensitive to outliers than KL-divergence or \(\chi^2\)
- **Interpretable**: In units of the feature (e.g., ms, dB, correlation points)

### 2.2 MDI Components and Scaling

Each feature divergence \(D_i\) is scaled by a perceptually-motivated factor \(s_i\) and weighted by \(w_i\):

$$
\text{MDI}_{\text{total}} = \sum_i w_i \cdot \frac{D_i}{s_i}
$$

**Component definitions**:

#### A. TFS Correlation to Ideal (distance to 1.0)

For each channel \(c\), compute the Wasserstein distance between the TFS correlation series \(\{\rho_{k,m}\}\) and the constant distribution \(\{1.0\}\):

$$
D_{\text{tfs,corr}}^{(c)} = W_1(\{\rho_{k,m}\}, \{1.0\})
$$

- **Weights**: Frame weights \(w_{k,m}\) (envelope-based)
- **Scale**: \(s = 0.1\) (correlation deviation of 0.1 normalizes to 1.0)
- **Interpretation**: Measures how far the TFS correlation distribution deviates from perfect preservation (1.0)

#### B. TFS Group Delay to Ideal (distance to 0 ms)

For each channel \(c\), collect the per-band group delays \(\{\tau_k\}\) and compute:

$$
D_{\text{tfs,delay}}^{(c)} = W_1(\{\tau_k\}, \{0.0 \text{ ms}\})
$$

- **Scale**: \(s = 0.2\) ms (0.2 ms group delay deviation normalizes to 1.0)
- **Interpretation**: Measures frequency-dependent timing shifts

#### C. Transient Attack Time Distribution

For each channel \(c\), compare the attack time distributions of reference and DUT events:

$$
D_{\text{trans,attack}}^{(c)} = W_1(\{\text{attack}_{\text{ref}}\}, \{\text{attack}_{\text{dut}}\})
$$

- **Weights**: Peak amplitude of each event
- **Scale**: \(s = 1.0\) ms (1 ms attack shift normalizes to 1.0)
- **Interpretation**: Measures systematic or variable shifts in transient attack timing

#### D. Transient Width Distribution

$$
D_{\text{trans,width}}^{(c)} = W_1(\{\text{width}_{\text{ref}}\}, \{\text{width}_{\text{dut}}\})
$$

- **Weights**: Peak amplitude of each event
- **Scale**: \(s = 1.0\) ms
- **Interpretation**: Measures changes in transient duration (e.g., ringing, smearing)

#### E. Binaural ITD Distribution (stereo only)

$$
D_{\text{bin,itd}} = W_1(\{\text{ITD}_{\text{ref}}\}, \{\text{ITD}_{\text{dut}}\})
$$

- **Weights**: Frame envelope energy
- **Scale**: \(s = 0.2\) ms (0.2 ms ITD shift normalizes to 1.0)
- **Interpretation**: Measures changes in interaural timing cues

#### F. Binaural ILD Distribution

$$
D_{\text{bin,ild}} = W_1(\{\text{ILD}_{\text{ref}}\}, \{\text{ILD}_{\text{dut}}\})
$$

- **Scale**: \(s = 1.0\) dB
- **Interpretation**: Measures changes in interaural level cues

#### G. Binaural IACC Distribution

$$
D_{\text{bin,iacc}} = W_1(\{\text{IACC}_{\text{ref}}\}, \{\text{IACC}_{\text{dut}}\})
$$

- **Scale**: \(s = 0.1\)
- **Interpretation**: Measures changes in interchannel correlation

### 2.3 Aggregation and Weighting

**Default group weights** (\(w_i\)):
- TFS: 1.0
- Transient: 1.0
- Binaural: 1.0

**Total score**:
$$
\text{MDI}_{\text{total}} = \text{MDI}_{\text{channels}} + \text{MDI}_{\text{binaural}}
$$

where:
- \(\text{MDI}_{\text{channels}}\) = sum of all per-channel TFS and transient components
- \(\text{MDI}_{\text{binaural}}\) = sum of all binaural components (stereo only)

**Lower is better**: \(\text{MDI} = 0\) indicates perfect preservation; higher values indicate distributional divergence.

---

## 3. Implementation Details

### 3.1 Parameters and Recommended Values

MDI is computed from pre-calculated TFS, Transient, and Binaural results. The key parameters are the group weights:

| Parameter | Type | Default | Range/Notes |
|-----------|------|---------|-------------|
| `weights["tfs"]` | float | 1.0 | Relative importance of TFS divergence |
| `weights["transient"]` | float | 1.0 | Relative importance of transient divergence |
| `weights["binaural"]` | float | 1.0 | Relative importance of binaural divergence (stereo only) |

**Scale factors** (hardcoded for perceptual consistency):
- TFS correlation: 0.1 (correlation points)
- TFS group delay: 0.2 ms
- Transient attack/width: 1.0 ms
- Binaural ITD: 0.2 ms
- Binaural ILD: 1.0 dB
- Binaural IACC: 0.1

### 3.2 Algorithm Overview

1. **Prerequisite**: Compute TFS, Transient, and Binaural metrics for both reference and DUT signals
   - TFS: `calculate_tfs_correlation()` must return per-frame correlation series with weights
   - Transient: `calculate_transient()` must return detected events with attack time, width, and peak amplitude
   - Binaural: `calculate_binaural()` must return per-frame ITD, ILD, IACC series with weights

2. **Per-channel processing**:
   - For each channel, extract TFS correlation series and band group delays
   - Compute Wasserstein distance to ideal values (1.0 for correlation, 0 ms for delay)
   - Extract transient event features (attack, width) and compute Wasserstein distance between ref/DUT distributions

3. **Binaural processing** (stereo only):
   - Extract ITD, ILD, IACC series
   - Filter finite values with positive weights
   - Compute Wasserstein distance between ref/DUT distributions for each feature

4. **Aggregation**:
   - Scale each component divergence by its scale factor
   - Apply group weights
   - Sum all components to produce total MDI score

### 3.3 Edge Cases and Special Handling

1. **Empty distributions**: If a feature series is empty (e.g., no transients detected), Wasserstein distance defaults to 0.0 to avoid metric failure. This is conservative; interpret with caution.

2. **Mismatched sample counts**: Wasserstein distance handles unequal sample counts naturally (ref and DUT may have different numbers of transients or valid frames).

3. **Infinite or NaN values**: Samples with non-finite values or zero/negative weights are filtered out before computing Wasserstein distance.

4. **Mono signals**: Binaural component is not computed; MDI reflects only TFS and transient divergences.

5. **Zero weights**: If all frame weights are zero (e.g., silent signal), the corresponding component divergence is 0.0.

### 3.4 Computational Complexity

- **Per-component Wasserstein**: \(\mathcal{O}(N \log N)\) where \(N\) is the number of samples (dominated by sorting)
- **Total complexity**: \(\mathcal{O}(C \cdot K \cdot N \log N)\) where \(C\) is the number of channels and \(K\) is the number of feature types (~5-7)

Typical runtime: <50 ms for all components given pre-computed TFS/Transient/Binaural results.

---

## 4. Interpretation Guidelines

### 4.1 Overall Score Ranges

**MDI Total** (\(\text{MDI}_{\text{total}}\)):

- **< 1.0**: Excellent microstructure preservation; minimal distributional divergence
- **1.0–2.0**: Very good; minor distributional shifts, likely acceptable for most applications
- **2.0–5.0**: Acceptable; noticeable but not severe divergence, may be audible in critical listening
- **> 5.0**: Significant microstructure degradation; systematic or intermittent artifacts present

### 4.2 Component-Level Interpretation

**Component totals** (`component_totals` dict):
- `channels`: Sum of all per-channel TFS and transient divergences
- `binaural`: Sum of all binaural divergences (stereo only)

**Per-channel totals** (`channel_totals` dict):
- Maps channel name (e.g., "L", "R") to the sum of TFS + transient divergences for that channel
- Useful for identifying channel-specific issues

**Individual components** (`components` list):
- Each component has:
  - `name`: e.g., "L.tfs.correlation_to_ideal", "L.transient.attack_time_ms", "binaural.itd_ms"
  - `distance`: Raw Wasserstein distance (in feature units: ms, dB, correlation points)
  - `weight`: Group weight applied
  - `scale`: Scale factor used for normalization
  - `samples_ref`, `samples_dut`: Number of valid samples used in the distribution comparison

**Diagnostic workflow**:
1. Check `mdi_total`: If high (> 2.0), proceed to component analysis
2. Compare `channels_total` vs `binaural_total`: Identify whether degradation is spatial or temporal
3. Inspect `channel_totals`: Check if one channel is worse (asymmetric degradation)
4. Examine individual `components`: Identify specific feature types with high divergence (e.g., TFS correlation vs transient attack)

### 4.3 Correlation with Other Metrics

**MDI vs average TFS correlation**:
- High TFS correlation (e.g., 0.95) with high MDI suggests **temporal instability**: most frames are good, but a few outliers have very low correlation
- Low TFS correlation (e.g., 0.80) with low MDI suggests **systematic degradation**: consistently mediocre but not intermittent

**MDI vs THD+N**:
- Low THD+N (e.g., -100 dB) with high MDI indicates **dynamic nonlinearity** that doesn't appear in steady-state measurements
- High THD+N with low MDI suggests steady-state distortion without microstructure issues

### 4.4 Common Patterns

**Scenario: High TFS divergence, low transient divergence**
- Likely: Phase instability, jitter, or group delay anomalies that don't significantly affect transient timing
- Check: TFS band group delays for frequency-dependent delays

**Scenario: High transient divergence, low TFS divergence**
- Likely: Slew-rate limiting, envelope clipping, or attack rounding that affects event characteristics but not fine structure
- Check: Transient attack time and width distributions for systematic shifts

**Scenario: High binaural divergence, low other components**
- Likely: Stereo channel imbalance, phase mismatch, or interchannel crosstalk
- Check: ILD for level mismatch, ITD for timing errors, IACC for decorrelation

**Scenario: High divergence across all components**
- Likely: Severe nonlinearity, bandwidth limitation, or systematic signal degradation
- Check: Basic metrics (THD+N, frequency response) first; MDI confirms multi-domain impact

---

## 5. Recommended Test Signals

MDI is a **secondary metric** computed from TFS, Transient, and Binaural results, so suitable test signals are those that:
- Excite TFS bands with high-frequency content (2–8 kHz)
- Contain transient events with varying attack characteristics
- For stereo: Contain spatial information (ITD, ILD, IACC)

### 5.1 Signal Types and Expected Behavior

| Signal Type | MDI Sensitivity | What MDI Reveals |
|-------------|-----------------|------------------|
| **Multi-tone** (`multitone`) | Medium | TFS group delay variation, transient response if tones have gating |
| **Tone Burst** (`tone-burst`) | High | Transient attack/width consistency, TFS phase stability during attack/decay |
| **AM-Modulated Tone** (`modulated`) | Medium-High | TFS correlation stability during modulation, transient event detection if AM is deep |
| **Swept Sine** (`sweep`) | Medium | Frequency-dependent TFS and transient degradation |
| **Notched Noise** (`notched-noise`) | Low-Medium | Primarily tests spectral fidelity; MDI can reveal transient edge artifacts |
| **Music or Complex Program** | Very High | Real-world test; MDI captures distributional shifts across diverse transient and TFS conditions |

### 5.2 Ideal Ranges by Signal Type

- **Tone Burst**: MDI < 1.5 (transient-rich, should have consistent attack/width)
- **Modulated Tone**: MDI < 2.0 (TFS correlation may vary slightly with modulation)
- **Multi-tone**: MDI < 1.0 (steady-state-like, minimal distributional shifts expected)
- **Music/Complex**: MDI < 3.0 (higher tolerance due to signal complexity and diverse features)

### 5.3 Usage Notes

- Always inspect component breakdowns to understand which domain contributes most to total MDI
- Compare MDI across multiple test signals to identify signal-dependent artifacts
- Use MDI in conjunction with average metrics (TFS mean correlation, transient event counts) to distinguish systematic vs intermittent issues

---

## 6. References

### Theoretical Background

- **Wasserstein Distance**: Villani, C. (2009). *Optimal Transport: Old and New*. Springer. – Comprehensive treatment of optimal transport theory and Earth Mover's Distance.
- **Distributional Metrics in Audio**: Marins, M. A., et al. (2018). Improved similarity measures for histogram-based signatures. *Pattern Recognition Letters*, 105, 47–53.
- **Perceptual Significance of Temporal Variability**: Moore, B. C. J. (2012). *An Introduction to the Psychology of Hearing* (6th ed.). Brill. – Temporal integration and just-noticeable differences in timing and correlation.

### Implementation References

- **NumPy Statistical Functions**: [https://numpy.org/doc/stable/reference/routines.statistics.html](https://numpy.org/doc/stable/reference/routines.statistics.html)
- **SciPy Wasserstein Distance** (reference, though not used in this implementation): [https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)

### Related Documentation

- **TFS Metric**: `docs/en/metrics/tfs.md` – Temporal Fine Structure correlation and phase coherence
- **Transient Metric**: `docs/en/metrics/transient.md` – Transient event detection and characterization
- **Binaural Metric**: (if available) – ITD, ILD, IACC computation
- **Metrics Interpretation Guide**: `docs/en/metrics-interpretation.md` – General guidance on using MDI with other metrics
- **Measurement Setup**: `docs/en/measurement-setup.md` – Practical considerations for level matching and alignment

### Source Code

- **MDI Implementation**: `src/microstructure_metrics/metrics/divergence.py`
  - `calculate_microstructure_distribution_divergence()`: Main MDI computation
  - `wasserstein_1d()`: 1D Wasserstein distance with optional weights

---

## Appendix: Common MDI Pitfalls

1. **Interpreting MDI without component breakdown**: Always inspect `components` to understand which feature contributes most to total divergence.

2. **Comparing MDI across different signal types**: MDI is signal-dependent; a tone burst naturally has lower MDI than complex music. Compare like-with-like.

3. **Ignoring sample counts**: If `samples_dut` is very low (e.g., < 10), the distribution comparison is unreliable. Check sample counts in `components`.

4. **Over-reliance on total score**: Use `channel_totals` and `component_totals` to localize issues to specific channels or domains.

5. **Forgetting to align signals**: MDI assumes reference and DUT are pre-aligned. Misalignment will inflate transient divergence and TFS group delay.

6. **Not accounting for silent segments**: Silent or very low-level segments may have zero-weight frames, reducing effective sample count. Inspect weights.

7. **Assuming monotonic relationship with perceptual quality**: MDI measures distributional divergence, not perceptual distance. A high MDI may or may not be audible depending on the listener and context.
