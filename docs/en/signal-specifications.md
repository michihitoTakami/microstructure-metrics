# Signal Format Specification (EN)

Scope: define test signal structure, file/metadata formats, and naming rules for all metrics (THD+N, MPS, TFS, Transient). Input/output are offline WAV files.

## Standard timeline
```
[Lead-in silence 500 ms] [Pilot tone 100 ms] [Test body 5–10 s] [Pilot tone 100 ms] [Tail silence 500 ms]
```
- Pilot: 1 kHz, -6 dBFS, 100 ms, 5 ms cosine fade in/out.
- Silence: 500 ms before/after. Do not trim.
- Test body: typically 5–10 s; keep peak ≤ -1 dBFS.

## Metric-specific signal examples
| Metric | Signal | Parameters (example) |
| --- | --- | --- |
| THD+N | Pure tone | 1 kHz, -3 dBFS, 5 s |
| MPS | AM/FM composite | 1 kHz carrier, AM 4 Hz depth 50%, FM dev 50 Hz @4 Hz, peak≈-6 dBFS, 8 s |
| TFS | High-band multitone | 4/6/8/10/12 kHz, equal amplitude, peak≈-6 dBFS, 8 s |
| Transient | Impulse / tone burst train | Spaced impulses or bursts, peak near -1 dBFS, 0.3–1.0 s |

## File format
- Sample rate: 48 kHz required (96 kHz only when explicitly needed).
- Bit depth: 24-bit PCM preferred; 32f allowed.
- Channels: mono; if stereo capture, analyze channels separately.
- Container: WAV (PCM or IEEE float). Avoid metadata that triggers gain changes.

## Naming convention
```
{signal_type}_{sample_rate}_{bit_depth}_{version}.wav

Examples:
thd_1khz_48000_24bit_v1.wav
notched_noise_8000hz_q8.6_48000_24bit_v1.wav
pink_noise_48000_24bit_v1.wav
```
- `signal_type`: thd, pink_noise, mps, tfs, transient, etc.
- `bit_depth`: `24bit` or `32f`.
- `version`: bump when parameters change (vN or semver).
- Optional parameters can follow signal_type (e.g., `mps_1khz_am4hz50_fm50hz_48000_24bit_v1.wav`).

## Metadata JSON (sidecar)
- Required keys:
  - `signal_type` (str), `sample_rate` (int), `bit_depth` (str), `channels` (int)
  - `duration_sec` (float), `pilot_tone_freq_hz` (int), `pilot_duration_ms` (int)
  - `pilot_level_dbfs` (float), `lead_silence_ms` / `tail_silence_ms` (int)
  - `version` (str), `created_at` (ISO 8601 UTC)
- Example:
```json
{
  "signal_type": "notched_noise",
  "sample_rate": 48000,
  "bit_depth": "24bit",
  "channels": 1,
  "duration_sec": 10.0,
  "pilot_tone_freq_hz": 1000,
  "pilot_duration_ms": 100,
  "pilot_level_dbfs": -6.0,
  "lead_silence_ms": 500,
  "tail_silence_ms": 500,
  "notch_center_hz": 8000,
  "notch_centers_hz": [8000],
  "notch_q": 8.6,
  "notch_cascade_stages": 1,
  "noise_color": "pink",
  "created_at": "2026-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

## Acceptance checklist
- Timeline matches spec (silence 500 ms, pilots 100 ms ×2).
- Sample rate / bit depth / channels match filename and metadata.
- Peak ≤ -1 dBFS; pilots ≈ -6 dBFS; no clipping; 5 ms fades applied.
- Metadata JSON exists and is consistent with the signal.

## Related docs
- Measurement setup (EN): `docs/en/measurement-setup.md`
- Metrics interpretation (EN): `docs/en/metrics-interpretation.md`
- CLI/API options (JP): `docs/jp/api-cli-reference.md`
