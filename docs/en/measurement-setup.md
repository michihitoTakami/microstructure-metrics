# Measurement Setup Guide (EN)

Purpose: ensure repeatable playback/recording of the test signals defined in `docs/en/signal-specifications.md` (or JP version), so alignment (`align`) and drift estimation (`drift`) remain stable.

## Common requirements
- Sample rate: 48 kHz (96 kHz only for high-band/TFS cases)
- Bit depth: 24-bit PCM (or 32-bit float)
- Channels: mono preferred. If stereo capture is unavoidable, keep identical content on L/R and pick a channel later.
- Player: bit-perfect/exclusive (`hw:` etc.). Disable EQ/AGC/DSP/resampling.
- Level: digital gain = 0 dB; peaks < -1 dBFS; pilot ≈ -6 dBFS.
- Do not trim: keep lead/tail silence intact.

## Recommended hardware chain
- Shared clock: drive DAC and ADC from the same clock; digital loopback is most stable. For external DAC, sync via word clock or digital output if possible.
- Audio interface: fixed 48 kHz / 24-bit without ASRC/DSP/noise suppression.
- Cabling: secure/short digital path; for analog, keep headroom and avoid unnecessary DI/effects.

## Scenarios

### 1) Digital loopback (smoke/regression)
1. Set playback/record to the same interface/clock at 48 kHz / 24-bit.
2. Prepare a test signal (e.g., `uv run microstructure-metrics generate notched-noise --with-metadata`).
3. Use bit-perfect playback; confirm headroom (peak < -1 dBFS).
4. Play and record the full file without trimming.
5. Run `align` → `drift` → `report` to validate the pipeline.

### 2) External DAC/AMP (device under test)
1. Playback: output the test WAV at 48 kHz / 24-bit; disable EQ/resampler.
2. Amplification: connect DUT to load; set level to avoid clipping.
3. Capture: ADC at 48 kHz / 24-bit. Adjust gain so pilot ≈ -6 dBFS with ≥6 dB headroom.
4. Save WAV (mono). Name files consistently (e.g., `{signal_name}_dut.wav`); keep metadata JSON alongside if available.
5. Verify using the checklist below.

## Post-recording checklist
- Timeline: lead/tail silence (~500 ms each) and two pilots (~100 ms each) are present.
- Levels: peak < -1 dBFS; pilot ≈ -6 dBFS; no clipping.
- Format matches metadata (sample rate/bit depth/channels).
- Drift: optional sanity check from pilot onset gaps (`drift` command recommended).

## Next steps
- Signal specs (EN): `docs/en/signal-specifications.md`
- Metrics interpretation (EN): `docs/en/metrics-interpretation.md`
- CLI/API options (JP): `docs/jp/api-cli-reference.md`
