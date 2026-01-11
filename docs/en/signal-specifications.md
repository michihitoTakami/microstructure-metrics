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
| LFCR | complex-bass | 30–220 Hz FM/PM multi-tone, 8 components, peak≈-2 dBFS |
| BCP | binaural-cues | 150 Hz–Nyquist pink noise with ITD 0.35 ms and ILD 6 dB, replayed as stereo peak≈-3 dBFS |
| RMI/MDI | ms-side-texture | Mid: 80–3.2 kHz pink noise, Side: 4+ kHz modulated tones, peak≈-3 dBFS for side-rich microstructure |

## File format
- Sample rate: 48 kHz required (96 kHz only when explicitly needed).
- Bit depth: 24-bit PCM preferred; 32f allowed.
- Channels: stereo (2ch) by default. Mono inputs are duplicated to stereo internally.
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

-## Metadata JSON (sidecar)
- Required keys:
  - `signal_type` (str), `sample_rate` (int), `bit_depth` (str), `channels` (int)
  - `duration_sec` (float), `pilot_tone_freq_hz` (int), `pilot_duration_ms` (int)
  - `pilot_level_dbfs` (float), `lead_silence_ms` / `tail_silence_ms` (int)
  - `version` (str), `created_at` (ISO 8601 UTC)
- Optional stimulus-specific keys:
  - `complex-bass`: `bass_components_hz`, `bass_fm_dev_hz`, `bass_fm_rates_hz`, `bass_pm_depth_rad`, `bass_pm_rates_hz`, `band_lowcut_hz`, `band_highcut_hz`, `target_peak_dbfs`
  - `binaural-cues`: `itd_ms`, `ild_db`, `base_noise_lowcut_hz`, `base_noise_highcut_hz`, `target_peak_dbfs`
  - `ms-side-texture`: `mid_band_lowcut_hz`, `mid_band_highcut_hz`, `side_tones_hz`, `side_mod_freq_hz`, `side_mod_depth`, `side_target_peak_dbfs`, `target_peak_dbfs`
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

## New stimuli / stereo signals
- `complex-bass` layers 8 FM/PM-modulated tones between 30–220 Hz to exercise LFCR’s cycle-shape / phase-coherence checks. The resulting waveform is duplicated to stereo so the low-frequency phase structure can still be tracked per channel.
- `binaural-cues` generates pink noise above 150 Hz and introduces an ITD (default 0.35 ms) plus ILD (default 6 dB) between L/R. This signal targets BCP’s ITD/ILD/IACC sections and keeps the stereo timeline intact for `report --channels stereo|mid|side`.
- `ms-side-texture` mixes mid-range pink noise (80–3.2 kHz) with high-frequency side tones (≳4 kHz) modulated at 5 Hz. The resulting mid/side stereo is ideal for RMI/MDI to pick up residual microstructure differences in the Side channel while still preserving a natural Mid reference.
- When using these stereo stimuli, keep the `align`/`drift`/`report` pipeline on 2ch and prefer `report --channels stereo` (default) or `mid`/`side` so BCP/MDI can consume all available features (`ch0`/`ch1` disables stereo-specific metrics).

## Related docs
- Measurement setup (EN): `docs/en/measurement-setup.md`
- Metrics interpretation (EN): `docs/en/metrics-interpretation.md`
- CLI/API options (JP): `docs/jp/api-cli-reference.md`
