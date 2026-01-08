from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf
from click.testing import CliRunner
from scipy import signal

from microstructure_metrics.cli import main


def _expected_samples(
    *,
    sample_rate: int,
    body_duration: float,
    pilot_ms: int,
    silence_ms: int,
) -> int:
    body_samples = int(body_duration * sample_rate)
    pilot_samples = int(sample_rate * pilot_ms / 1000)
    silence_samples = int(sample_rate * silence_ms / 1000)
    return body_samples + 2 * (pilot_samples + silence_samples)


def _pilot_rms_dbfs(
    data, *, sample_rate: int, pilot_ms: int, silence_ms: int, margin_ms: int = 20
) -> float:
    pilot_samples = int(sample_rate * pilot_ms / 1000)
    silence_samples = int(sample_rate * silence_ms / 1000)
    start = silence_samples + int(sample_rate * margin_ms / 1000)
    end = start + max(pilot_samples - 2 * int(sample_rate * margin_ms / 1000), 1)
    pilot_seg = np.asarray(data, dtype=np.float64)[start:end, 0]
    rms = (pilot_seg.astype("float64") ** 2).mean() ** 0.5
    rms = max(rms, 1e-12)
    return 20 * float(np.log10(rms))


def _body_segment(
    data: npt.NDArray[np.float64], *, sample_rate: int, pilot_ms: int, silence_ms: int
) -> npt.NDArray[np.float64]:
    silence_samples = int(sample_rate * silence_ms / 1000)
    pilot_samples = int(sample_rate * pilot_ms / 1000)
    start = silence_samples + pilot_samples
    end = data.shape[0] - (silence_samples + pilot_samples)
    return data[start:end, 0]


def test_generate_thd_creates_wav_and_json(tmp_path: Path) -> None:
    runner = CliRunner()
    wav_path = tmp_path / "thd.wav"
    result = runner.invoke(
        main,
        [
            "generate",
            "thd",
            "--duration",
            "0.5",
            "--freq",
            "1000",
            "--output",
            str(wav_path),
            "--with-metadata",
        ],
    )

    assert result.exit_code == 0, result.output
    assert wav_path.exists()
    assert wav_path.with_suffix(".json").exists()

    data, sample_rate = sf.read(wav_path, always_2d=True)
    assert sample_rate == 48000
    assert data.ndim == 2
    assert data.shape[1] == 2
    expected = _expected_samples(
        sample_rate=sample_rate,
        body_duration=0.5,
        pilot_ms=100,
        silence_ms=500,
    )
    assert data.shape[0] == expected
    pilot_dbfs = _pilot_rms_dbfs(
        data, sample_rate=sample_rate, pilot_ms=100, silence_ms=500
    )
    assert -10.5 <= pilot_dbfs <= -8.0


def test_generate_thd_stereo_channels(tmp_path: Path) -> None:
    runner = CliRunner()
    wav_path = tmp_path / "thd_stereo.wav"
    result = runner.invoke(
        main,
        [
            "generate",
            "thd",
            "--duration",
            "0.2",
            "--output",
            str(wav_path),
            "--with-metadata",
        ],
    )

    assert result.exit_code == 0, result.output
    data, sample_rate = sf.read(wav_path, always_2d=True)
    assert sample_rate == 48_000
    assert data.shape[1] == 2
    assert np.allclose(data[:, 0], data[:, 1], atol=1e-12)

    meta_path = wav_path.with_suffix(".json")
    payload = meta_path.read_text()
    assert '"channels": 2' in payload


def test_generate_tfs_tones_defaults(tmp_path: Path) -> None:
    runner = CliRunner()
    wav_path = tmp_path / "tfs.wav"
    result = runner.invoke(
        main,
        [
            "generate",
            "tfs-tones",
            "--duration",
            "0.25",
            "--tone-count",
            "3",
            "--output",
            str(wav_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert wav_path.exists()

    data, sample_rate = sf.read(wav_path, always_2d=True)
    expected = _expected_samples(
        sample_rate=sample_rate,
        body_duration=0.25,
        pilot_ms=100,
        silence_ms=500,
    )
    assert data.shape[0] == expected
    assert data.ndim == 2
    assert data.shape[1] == 2


@pytest.mark.parametrize(
    "signal_type,opts",
    [
        ("notched-noise", {"--center": "8000", "--q": "8.6"}),
        ("pink-noise", {"--highcut": "20000"}),
        (
            "modulated",
            {
                "--carrier": "1000",
                "--am-freq": "4",
                "--am-depth": "0.5",
                "--fm-dev": "50",
            },
        ),
        (
            "tone-burst",
            {
                "--burst-freq": "8000",
                "--burst-cycles": "10",
                "--burst-level-dbfs": "-6",
                "--burst-fade-cycles": "2",
            },
        ),
        (
            "am-attack",
            {
                "--carrier": "1000",
                "--attack-ms": "2",
                "--release-ms": "10",
                "--gate-period-ms": "100",
            },
        ),
        (
            "click",
            {"--click-level-dbfs": "-6", "--click-band-limit-hz": "20000"},
        ),
    ],
)
def test_generate_other_signals_structure_and_pilot(tmp_path: Path, signal_type, opts):
    runner = CliRunner()
    wav_path = tmp_path / f"{signal_type}.wav"
    args = [
        "generate",
        signal_type,
        "--duration",
        "0.3",
        "--output",
        str(wav_path),
        "--with-metadata",
    ]
    for k, v in opts.items():
        args.extend([k, v])
    result = runner.invoke(main, args)
    assert result.exit_code == 0, result.output

    data, sample_rate = sf.read(wav_path, always_2d=True)
    expected = _expected_samples(
        sample_rate=sample_rate, body_duration=0.3, pilot_ms=100, silence_ms=500
    )
    assert data.shape[0] == expected
    assert data.ndim == 2
    assert data.shape[1] == 2

    pilot_dbfs = _pilot_rms_dbfs(
        data, sample_rate=sample_rate, pilot_ms=100, silence_ms=500
    )
    assert -10.5 <= pilot_dbfs <= -8.0

    metadata = sf.info(wav_path)
    assert metadata.samplerate == 48000


@pytest.mark.parametrize(
    "signal_type,opts,expected_keys",
    [
        (
            "tone-burst",
            {
                "--burst-freq": "8000",
                "--burst-cycles": "10",
                "--burst-level-dbfs": "-6",
                "--burst-fade-cycles": "2",
            },
            ["burst_freq_hz", "burst_cycles", "burst_fade_cycles", "burst_level_dbfs"],
        ),
        (
            "am-attack",
            {
                "--carrier": "1000",
                "--attack-ms": "2",
                "--release-ms": "10",
                "--gate-period-ms": "100",
            },
            ["carrier_hz", "attack_ms", "release_ms", "gate_period_ms"],
        ),
        (
            "click",
            {"--click-level-dbfs": "-6", "--click-band-limit-hz": "20000"},
            ["click_level_dbfs", "click_band_limit_hz"],
        ),
    ],
)
def test_generate_transient_signals_metadata_keys(
    tmp_path: Path, signal_type: str, opts: dict[str, str], expected_keys: list[str]
) -> None:
    runner = CliRunner()
    wav_path = tmp_path / f"{signal_type}.wav"
    args = [
        "generate",
        signal_type,
        "--duration",
        "0.2",
        "--output",
        str(wav_path),
        "--with-metadata",
    ]
    for k, v in opts.items():
        args.extend([k, v])
    result = runner.invoke(main, args)
    assert result.exit_code == 0, result.output
    meta_path = wav_path.with_suffix(".json")
    assert meta_path.exists()
    payload = meta_path.read_text()
    for key in expected_keys:
        assert f'"{key}"' in payload


def test_pilot_frequency_and_level(tmp_path: Path) -> None:
    runner = CliRunner()
    wav_path = tmp_path / "pilot.wav"
    result = runner.invoke(
        main,
        [
            "generate",
            "thd",
            "--duration",
            "0.5",
            "--freq",
            "1000",
            "--output",
            str(wav_path),
        ],
    )
    assert result.exit_code == 0, result.output
    data, sr = sf.read(wav_path, always_2d=True)
    silence = int(sr * 0.5)
    pilot = int(sr * 0.1)
    start = silence
    end = silence + pilot
    pilot_seg = data[start:end, 0]
    freqs, psd = signal.welch(
        pilot_seg, sr, nperseg=min(2048, pilot_seg.shape[0]), scaling="spectrum"
    )
    peak_freq = freqs[np.argmax(psd)]
    assert abs(peak_freq - 1000) < 15.0
    rms_dbfs = _pilot_rms_dbfs(data, sample_rate=sr, pilot_ms=100, silence_ms=500)
    assert -10.5 <= rms_dbfs <= -8.0


def test_notched_noise_has_attenuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_default_rng = np.random.default_rng
    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda *_, **__: original_default_rng(0),  # deterministic
    )
    runner = CliRunner()
    wav_path = tmp_path / "notch.wav"
    result = runner.invoke(
        main,
        [
            "generate",
            "notched-noise",
            "--duration",
            "1.0",
            "--center",
            "8000",
            "--q",
            "8.6",
            "--output",
            str(wav_path),
        ],
    )
    assert result.exit_code == 0, result.output
    data, sr = sf.read(wav_path, always_2d=True)
    body = _body_segment(data, sample_rate=sr, pilot_ms=100, silence_ms=500)
    freqs, psd = signal.welch(body, sr, nperseg=8192, scaling="spectrum")
    center_idx = np.argmin(np.abs(freqs - 8000))
    band = (freqs > 7000) & (freqs < 9000)
    mean_band = np.mean(psd[band])
    notch_level = psd[center_idx]
    attenuation_db = 10 * np.log10(mean_band / max(notch_level, 1e-18))
    assert attenuation_db >= 10.0  # at least 10 dB attenuation at the notch


def test_generate_notched_noise_multi_centers_and_cascade_metadata(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    wav_path = tmp_path / "multi_notch.wav"
    result = runner.invoke(
        main,
        [
            "generate",
            "notched-noise",
            "--duration",
            "0.2",
            "--centers",
            "3000,5000,7000,9000",
            "--q",
            "8.6",
            "--notch-cascade-stages",
            "2",
            "--output",
            str(wav_path),
            "--with-metadata",
        ],
    )
    assert result.exit_code == 0, result.output
    meta_path = wav_path.with_suffix(".json")
    assert meta_path.exists()
    payload = meta_path.read_text()
    assert '"notch_centers_hz"' in payload
    assert '"notch_cascade_stages": 2' in payload


def test_generate_notched_noise_q_sweep_creates_multiple_files(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "q_sweep"
    result = runner.invoke(
        main,
        [
            "generate",
            "notched-noise",
            "--duration",
            "0.15",
            "--center",
            "8000",
            "--q",
            "2",
            "--q",
            "8.6",
            "--output",
            str(out_dir),
            "--with-metadata",
        ],
    )
    assert result.exit_code == 0, result.output
    wavs = sorted(out_dir.glob("*.wav"))
    jsons = sorted(out_dir.glob("*.json"))
    assert len(wavs) == 2
    assert len(jsons) == 2


def test_pink_noise_spectral_slope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_default_rng = np.random.default_rng
    monkeypatch.setattr(
        np.random, "default_rng", lambda *_, **__: original_default_rng(1)
    )
    runner = CliRunner()
    wav_path = tmp_path / "pink.wav"
    result = runner.invoke(
        main,
        [
            "generate",
            "pink-noise",
            "--duration",
            "1.0",
            "--output",
            str(wav_path),
        ],
    )
    assert result.exit_code == 0, result.output
    data, sr = sf.read(wav_path, always_2d=True)
    body = _body_segment(data, sample_rate=sr, pilot_ms=100, silence_ms=500)
    freqs, psd = signal.welch(body, sr, nperseg=8192, scaling="spectrum")
    mask = (freqs >= 200) & (freqs <= 10000)
    x = np.log10(freqs[mask])
    y = np.log10(psd[mask])
    slope, _ = np.polyfit(x, y, 1)
    assert -1.4 <= slope <= -0.6  # ideal pink ~ -1.0 in log-log


def test_modulated_has_am_depth(tmp_path: Path) -> None:
    runner = CliRunner()
    wav_path = tmp_path / "mod.wav"
    result = runner.invoke(
        main,
        [
            "generate",
            "modulated",
            "--duration",
            "0.8",
            "--carrier",
            "1000",
            "--am-freq",
            "4",
            "--am-depth",
            "0.5",
            "--output",
            str(wav_path),
        ],
    )
    assert result.exit_code == 0, result.output
    data, sr = sf.read(wav_path, always_2d=True)
    body = _body_segment(data, sample_rate=sr, pilot_ms=100, silence_ms=500)
    analytic = signal.hilbert(body)
    env = np.abs(analytic)
    env_max = env.max()
    env_min = env.min()
    depth = (env_max - env_min) / (env_max + env_min + 1e-12)
    assert 0.35 <= depth <= 0.65  # target depth 0.5 with tolerance


def test_tfs_tones_peak_frequencies(tmp_path: Path) -> None:
    runner = CliRunner()
    wav_path = tmp_path / "tfs_spec.wav"
    result = runner.invoke(
        main,
        [
            "generate",
            "tfs-tones",
            "--duration",
            "0.5",
            "--tone-count",
            "5",
            "--min-freq",
            "4000",
            "--tone-step",
            "2000",
            "--output",
            str(wav_path),
        ],
    )
    assert result.exit_code == 0, result.output
    data, sr = sf.read(wav_path, always_2d=True)
    body = _body_segment(data, sample_rate=sr, pilot_ms=100, silence_ms=500)
    freqs = np.fft.rfftfreq(body.size, 1 / sr)
    spectrum = np.abs(np.fft.rfft(body))
    peak_indices = np.argpartition(spectrum, -5)[-5:]
    peak_freqs = np.sort(freqs[peak_indices])
    expected = np.array([4000, 6000, 8000, 10000, 12000])
    assert np.allclose(peak_freqs, expected, atol=50.0)
