# Signal Format Specification

## Purpose and scope
- Define standard test signal structure, file/metadata formats, and naming rules.
- Applicable to all metrics in this project: THD+N, MPS, TFS, Transient.
- Input/output are offline WAV files; realtime I/O is out of scope.

## Standard timeline
```
[Lead-in silence 500 ms] [Pilot tone 100 ms] [Test body 5–10 s] [Pilot tone 100 ms] [Tail silence 500 ms]
```
- Pilot tone: 1 kHz sine, -6 dBFS, duration 100 ms. Apply 5 ms cosine fade-in/out to avoid clicks.
- Silence: full-scale zero samples; keep 500 ms before first pilot and after last pilot. No dithering is required.
- Test body: metric-specific content; recommended length 5–10 s. Keep overall peak ≤ -1 dBFS for headroom.

## Metric-specific test signals
| Metric | Signal | Parameters (recommendation) |
|--------|--------|-----------------------------|
| THD+N  | Pure tone | 1 kHz, -3 dBFS, length 5 s |
| MPS    | AM/FM composite | Carrier 1 kHz, AM: 4 Hz depth 50%; optional FM: dev 50 Hz @ 4 Hz, peak ≈ -6 dBFS, length 8 s |
| TFS    | High-band multitone | Example tones: 4, 6, 8, 10, 12 kHz, equal amplitude, peak ≈ -6 dBFS, length 8 s |
| Transient | Impulse / click train | Spaced impulses or tone bursts, peak near -1 dBFS, length 0.3–1.0 s |

Notes:
- Use 24-bit or 32-bit float generation to avoid quantization during synthesis.
- Apply short fades (5 ms) at segment boundaries to suppress transients.

## File format
- Sample rate: 48 kHz required; 96 kHz optional (use only when explicitly needed).
- Bit depth: 24-bit PCM preferred; 32-bit float allowed.
- Channels: stereo (2ch) by default. Mono inputs are duplicated to stereo internally.
- Container: WAV (PCM or IEEE float). Avoid metadata that triggers DAW gain changes.

### Rationale (key parameters)
- 48 kHz mandatory: aligns with most DAC/ADC default clocking and downstream toolchains; sufficient for ≤20 kHz content (all test signals except optional high-band variants). 96 kHz is allowed for future TFS/high-band evaluation but increases drift risk and file size, so it is opt-in.
- Pilot 1 kHz / -6 dBFS / 100 ms: easy detectability by onset detectors and meters; far from Nyquist; level keeps 6 dB headroom to avoid clipping in mis-calibrated paths; 100 ms is long enough for stable RMS/FFT yet short versus total duration.
- Lead/tail silence 500 ms: provides margin for alignment, noise floor estimation, and hardware latency; long enough for windowed analysis without dominating file length.
- Fades 5 ms: short enough not to affect analysis bandwidth, long enough to suppress clicks.
- Test body peak ≤ -1 dBFS and typical RMS around -14 to -6 dBFS: leaves analog and digital headroom while keeping SNR high for distortion metrics.

## Naming convention
```
{signal_type}_{sample_rate}_{bit_depth}_{version}.wav

Examples:
thd_1khz_48000_24bit_v1.wav
notched_noise_8khz_q86_48000_24bit_v1.wav
pink_noise_48000_24bit_v1.wav
```
- `signal_type`: thd, notched_noise, pink_noise, mps, tfs, etc.
- `bit_depth`: `24bit` or `32f`.
- `version`: semantic version or vN tag; increment when parameters change.
- Optional parameters can be inserted after signal type (e.g., `mps_1khz_am4hz50_fm50hz_48000_24bit_v1.wav`).

## Metadata sidecar (JSON)
- Place alongside WAV with the same stem: `{filename}.json`.
- Required keys:
  - `signal_type` (string)
  - `sample_rate` (int, Hz)
  - `bit_depth` (string: `24bit` or `32f`)
  - `channels` (int; default 1)
  - `duration_sec` (float)
  - `pilot_tone_freq_hz` (int, default 1000)
  - `pilot_duration_ms` (int, default 100)
  - `pilot_level_dbfs` (float, default -6.0)
  - `lead_silence_ms` / `tail_silence_ms` (int, default 500)
  - `version` (string, e.g., `1.0.0`)
  - `created_at` (ISO 8601 string, UTC)
- Signal-specific keys (examples):
  - Notched noise: `notch_center_hz`, `notch_q`, `noise_color` (`pink`)
  - MPS: `carrier_hz`, `am_freq_hz`, `am_depth_ratio`, `fm_dev_hz` (optional), `mod_freq_hz`
  - TFS: `tones_hz` (array), `tone_level_dbfs`
- Example:
```json
{
  "signal_type": "notched_noise",
  "sample_rate": 48000,
  "bit_depth": "24bit",
  "channels": 2,
  "duration_sec": 10.0,
  "pilot_tone_freq_hz": 1000,
  "pilot_duration_ms": 100,
  "pilot_level_dbfs": -6.0,
  "lead_silence_ms": 500,
  "tail_silence_ms": 500,
  "notch_center_hz": 8000,
  "notch_q": 8.6,
  "noise_color": "pink",
  "created_at": "2026-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

## Acceptance checklist
- Structure matches the standard timeline with correct durations.
- Pilot tone is 1 kHz, -6 dBFS, 100 ms with fades; silence segments are present.
- Sample rate/bit depth match the naming and metadata.
- Peak level ≤ -1 dBFS; no clipping; fades applied at joins.
- Metadata JSON is present and consistent with filename and signal content.
