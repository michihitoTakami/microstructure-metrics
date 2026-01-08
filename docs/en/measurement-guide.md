# Measurement Guide (recommended environment)

## Purpose
Ensure repeatable offline evaluation by standardizing playback/recording conditions for the test signals defined in `signal-format-spec.md`.

## System requirements
- Shared clock: use the same master clock for DAC/ADC to minimize drift. Prefer digital loopback or a synchronized interface.
- Sample rate: set both playback and capture to 48 kHz. Use 96 kHz only when the provided signal explicitly targets high-band/TFS validation; higher rates increase drift sensitivity and file size.
- Bit depth: configure 24-bit (or 32-bit float) end-to-end. Disable SRC/dithering in players unless explicitly needed.
- Processing: disable all DSP/EQ/AGC/noise suppression in OS, driver, and player.
- Channel handling: use stereo (2ch) chain by default. Mono inputs are duplicated to stereo internally.

## Playback chain
- Player: use bit-perfect mode (e.g., exclusive/wasapi/alsa `hw:`). Avoid normalizers or volume leveling.
- Level: keep digital gain at unity; ensure peaks stay below -1 dBFS. Pilot tone is -6 dBFS; main body typically -3 to -6 dBFS.
- Start/stop: do not trim the lead-in/tail silence; they are used for alignment and noise floor measurement.

## Capture chain
- Interface: set the same sample rate/bit depth as playback; lock to the same clock when possible.
- Gain staging: keep at least 6 dB headroom; confirm no ADC clipping during pilot tone.
- File format: capture to WAV PCM 24-bit or 32-bit float; stereo (2ch) preferred.
- Noise floor: verify tail silence RMS to confirm environment and interface noise are acceptable.

## Operational steps
1) Set system/player/ADC to 48 kHz, 24-bit (or 32f). Confirm clock sync.
2) Load the provided WAV; ensure bit-perfect playback mode is active.
3) Play and capture the full file without trimming.
4) Verify the recorded file: duration matches, pilot tone amplitude ≈ -6 dBFS, peak < -1 dBFS, no clipping.
5) Name the captured file following the naming convention in `signal-format-spec.md` and keep the paired JSON metadata.

## Quality checks before analysis
- Peaks < -1 dBFS; no flat-topped sections.
- Pilot tone present at start and end; durations ~100 ms each; silence ~500 ms each side.
- Sample rate/bit depth/channels match metadata and filename.
- Optional: measure drift by comparing pilot onset offsets between channels (if stereo) or between playback/capture clocks.
