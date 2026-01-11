from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from scipy import signal as sp_signal  # noqa: E402

from microstructure_metrics.metrics import (
    BassResult,
    BinauralResult,
    MPSSimilarityResult,
    ResidualMicrostructureResult,
    TFSCorrelationResult,
)

EPS = 1e-12
TWO_PI = 2 * np.pi


def save_mps_delta_heatmap(
    *,
    result: MPSSimilarityResult,
    path: Path,
) -> Path:
    """Save a heatmap of MPS delta (ref - dut) in dB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    if result.delta_mps_db.size == 0:
        ax.text(0.5, 0.5, "No MPS data", ha="center", va="center")
    else:
        delta_db = np.asarray(result.delta_mps_db, dtype=np.float64)
        audio_freqs = np.asarray(result.audio_freqs, dtype=np.float64)
        mod_freqs = np.asarray(result.mod_freqs, dtype=np.float64)

        finite = delta_db[np.isfinite(delta_db)]
        max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
        if max_abs <= 0:
            max_abs = 1.0

        mesh = ax.pcolormesh(
            _axis_edges(mod_freqs),
            _axis_edges(audio_freqs),
            np.nan_to_num(
                delta_db,
                nan=0.0,
                posinf=max_abs,
                neginf=-max_abs,
            ),
            shading="auto",
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
        )
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("ΔMPS (dB, ref - dut)")
        ax.set_xlabel("Modulation Frequency (Hz)")
        ax.set_ylabel("Audio Center Frequency (Hz)")
        ax.set_title("MPS Delta Heatmap")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_tfs_correlation_timeseries(
    *,
    result: TFSCorrelationResult,
    path: Path,
) -> Path:
    """Save time-series plot of short-time TFS correlation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))

    times_ms = np.asarray(result.frame_times_ms, dtype=np.float64)
    series = np.asarray(result.correlation_series, dtype=np.float64)
    weights = np.asarray(result.correlation_weights, dtype=np.float64)

    if times_ms.size == 0 or series.size == 0:
        ax.text(0.5, 0.5, "No TFS data", ha="center", va="center")
    else:
        series = np.where(np.isfinite(series), series, np.nan)
        for idx, band in enumerate(result.freq_bands):
            band_series = series[idx]
            if np.all(np.isnan(band_series)):
                continue
            ax.plot(
                times_ms,
                band_series,
                alpha=0.25,
                linewidth=1.0,
                label=f"{band[0]:.0f}-{band[1]:.0f} Hz",
            )

        mean_series = _nan_weighted_mean(series, weights)
        if mean_series.size and not np.all(np.isnan(mean_series)):
            ax.plot(
                times_ms,
                mean_series,
                color="C0",
                linewidth=2.0,
                label="Weighted mean",
            )

        ax.axhline(0.0, color="0.6", linestyle="--", linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Correlation")
        ax.set_title("TFS Correlation Over Time")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_binaural_itd_heatmap(
    *,
    result: BinauralResult,
    path: Path,
) -> Path:
    """Save heatmap of ITD delta over time and band."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))

    delta = np.asarray(result.itd_dut_ms - result.itd_ref_ms, dtype=np.float64)
    times_ms = np.asarray(result.frame_times_ms, dtype=np.float64)
    freq_centers = np.asarray(result.freq_centers_hz, dtype=np.float64)

    if delta.size == 0 or times_ms.size == 0 or freq_centers.size == 0:
        ax.text(0.5, 0.5, "No BCP ITD data", ha="center", va="center")
    else:
        finite = delta[np.isfinite(delta)]
        max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
        if max_abs <= 0:
            max_abs = 1.0
        mesh = ax.pcolormesh(
            _axis_edges(times_ms),
            _axis_edges(freq_centers),
            np.nan_to_num(delta, nan=0.0, posinf=max_abs, neginf=-max_abs),
            shading="auto",
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
        )
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("ΔITD (ms, dut - ref)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Center Frequency (Hz)")
        ax.set_title("BCP ITD Delta Heatmap")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_binaural_ild_heatmap(
    *,
    result: BinauralResult,
    path: Path,
) -> Path:
    """Save heatmap of ILD delta over time and band."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))

    delta = np.asarray(result.ild_dut_db - result.ild_ref_db, dtype=np.float64)
    times_ms = np.asarray(result.frame_times_ms, dtype=np.float64)
    freq_centers = np.asarray(result.freq_centers_hz, dtype=np.float64)

    if delta.size == 0 or times_ms.size == 0 or freq_centers.size == 0:
        ax.text(0.5, 0.5, "No BCP ILD data", ha="center", va="center")
    else:
        finite = delta[np.isfinite(delta)]
        max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
        if max_abs <= 0:
            max_abs = 1.0
        mesh = ax.pcolormesh(
            _axis_edges(times_ms),
            _axis_edges(freq_centers),
            np.nan_to_num(delta, nan=0.0, posinf=max_abs, neginf=-max_abs),
            shading="auto",
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
        )
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("ΔILD (dB, dut - ref)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Center Frequency (Hz)")
        ax.set_title("BCP ILD Delta Heatmap")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_binaural_iacc_timeseries(
    *,
    result: BinauralResult,
    path: Path,
) -> Path:
    """Save time-series plot of IACC (ref/dut) and delta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))

    times_ms = np.asarray(result.frame_times_ms, dtype=np.float64)
    iacc_ref = np.asarray(result.iacc_ref, dtype=np.float64)
    iacc_dut = np.asarray(result.iacc_dut, dtype=np.float64)
    weights = np.asarray(result.weights, dtype=np.float64)

    if times_ms.size == 0 or iacc_ref.size == 0 or iacc_dut.size == 0:
        ax.text(0.5, 0.5, "No BCP IACC data", ha="center", va="center")
    else:
        mean_ref = _nan_weighted_mean(iacc_ref, weights)
        mean_dut = _nan_weighted_mean(iacc_dut, weights)
        delta = mean_dut - mean_ref

        if mean_ref.size:
            ax.plot(times_ms, mean_ref, label="Ref", color="C0", linewidth=1.5)
        if mean_dut.size:
            ax.plot(times_ms, mean_dut, label="DUT", color="C1", linewidth=1.5)
        if delta.size:
            ax.plot(times_ms, delta, label="Δ (dut - ref)", color="C3", linewidth=1.0)

        ax.axhline(0.0, color="0.6", linestyle="--", linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("IACC")
        ax.set_title("BCP IACC Over Time (Weighted Mean)")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_bass_cycle_shape_overlay(
    *,
    reference: npt.ArrayLike,
    dut: npt.ArrayLike,
    sample_rate: int,
    result: BassResult,
    path: Path,
) -> Path:
    """Save overlay of LFCR cycle shapes (ref vs dut) per band."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ref = np.asarray(reference, dtype=np.float64)
    du = np.asarray(dut, dtype=np.float64)
    fig, axes = plt.subplots(
        nrows=max(len(result.bands_hz), 1),
        ncols=1,
        figsize=(8, 3.0 * len(result.bands_hz)),
    )

    axes_list = axes if isinstance(axes, np.ndarray) else np.asarray([axes])
    if ref.size == 0 or du.size == 0 or sample_rate <= 0:
        for ax in axes_list:
            ax.text(0.5, 0.5, "No LFCR data", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    envelope_scale = float(
        np.max(np.abs(np.concatenate([ref, du]))) if ref.size else EPS
    )
    envelope_threshold = envelope_scale * 10 ** (result.envelope_threshold_db / 20.0)

    for idx, (low, high) in enumerate(result.bands_hz):
        ax = axes_list[idx]
        ref_band = _bandpass(
            data=ref,
            sample_rate=sample_rate,
            low=low,
            high=high,
            order=result.filter_order,
        )
        dut_band = _bandpass(
            data=du,
            sample_rate=sample_rate,
            low=low,
            high=high,
            order=result.filter_order,
        )
        phase_grid, mean_ref, mean_dut, cycles = _mean_cycle_shapes(
            ref_band=ref_band,
            dut_band=dut_band,
            cycle_points=result.cycle_points,
            envelope_threshold=envelope_threshold,
        )

        if phase_grid.size == 0:
            ax.text(0.5, 0.5, "No cycles extracted", ha="center", va="center")
        else:
            ax.plot(phase_grid, mean_ref, label="Ref", color="C0")
            ax.plot(phase_grid, mean_dut, label="DUT", color="C1")
            ax.set_xlabel("Phase (rad)")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.set_title(f"LFCR Cycle Shape {low:.0f}-{high:.0f} Hz (n={cycles})")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_residual_spectrogram(
    *,
    reference: npt.ArrayLike,
    dut: npt.ArrayLike,
    sample_rate: int,
    result: ResidualMicrostructureResult,
    path: Path,
) -> Path:
    """Save residual spectrogram."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))

    ref = np.asarray(reference, dtype=np.float64)
    du = np.asarray(dut, dtype=np.float64)
    if ref.size == 0 or du.size == 0 or sample_rate <= 0:
        ax.text(0.5, 0.5, "No residual data", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    residual = _residual_signal(
        reference=ref,
        dut=du,
        delay_samples=result.delay_samples,
        scale=result.scale,
    )
    if residual.size < 4:
        ax.text(0.5, 0.5, "Residual too short", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    nperseg = min(2048, residual.size)
    nperseg = max(nperseg, 64)
    noverlap = int(nperseg * 0.75)
    freqs, times, spec = sp_signal.stft(
        residual,
        fs=float(sample_rate),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    magnitude = 20.0 * np.log10(np.maximum(np.abs(spec), EPS))
    mesh = ax.pcolormesh(
        times * 1000.0,
        freqs,
        magnitude,
        shading="auto",
        cmap="magma",
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Residual Magnitude (dB)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Residual Spectrogram")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _axis_edges(axis: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert axis centers to edges for pcolormesh."""
    if axis.size == 0:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    if axis.size == 1:
        step = max(abs(axis[0]) * 0.1, 1.0)
        return np.asarray([axis[0] - step, axis[0] + step], dtype=np.float64)
    diffs = np.diff(axis)
    edges = np.empty(axis.size + 1, dtype=np.float64)
    edges[1:-1] = axis[:-1] + diffs / 2
    edges[0] = max(axis[0] - diffs[0] / 2, 0.0)
    edges[-1] = axis[-1] + diffs[-1] / 2
    return edges


def _nan_weighted_mean(
    values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Weighted mean ignoring NaNs."""
    if values.shape != weights.shape:
        raise ValueError("values and weights must share shape")
    if values.size == 0:
        return np.asarray([], dtype=np.float64)

    valid = np.isfinite(values)
    weighted_sum = np.sum(np.where(valid, values * weights, 0.0), axis=0)
    weight_sum = np.sum(np.where(valid, weights, 0.0), axis=0)
    mean = np.full(values.shape[1], np.nan, dtype=np.float64)
    mask = weight_sum > 0
    mean[mask] = weighted_sum[mask] / weight_sum[mask]
    return mean


def _bandpass(
    *,
    data: npt.NDArray[np.float64],
    sample_rate: int,
    low: float,
    high: float,
    order: int,
) -> npt.NDArray[np.float64]:
    nyquist = sample_rate / 2
    if nyquist <= 0:
        return np.asarray([], dtype=np.float64)
    sos = sp_signal.butter(
        order, [low / nyquist, high / nyquist], btype="band", output="sos"
    )
    return np.asarray(sp_signal.sosfiltfilt(sos, data), dtype=np.float64)


def _mean_cycle_shapes(
    *,
    ref_band: npt.NDArray[np.float64],
    dut_band: npt.NDArray[np.float64],
    cycle_points: int,
    envelope_threshold: float,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], int
]:
    analytic_ref = sp_signal.hilbert(ref_band)
    envelope_ref = np.abs(analytic_ref)
    phase_ref = np.unwrap(np.angle(analytic_ref))

    if phase_ref.size == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            0,
        )

    cycle_ids = np.floor((phase_ref - phase_ref[0]) / TWO_PI).astype(int)
    unique_cycles = np.unique(cycle_ids)
    phase_grid = np.linspace(0.0, TWO_PI, cycle_points, endpoint=False)

    ref_shapes: list[npt.NDArray[np.float64]] = []
    dut_shapes: list[npt.NDArray[np.float64]] = []
    weights: list[float] = []

    for cycle in unique_cycles:
        mask = cycle_ids == cycle
        if np.count_nonzero(mask) < 2:
            continue
        phase_segment = phase_ref[mask]
        span = float(phase_segment[-1] - phase_segment[0])
        if span < TWO_PI * 0.75:
            continue
        env_mean = float(np.mean(envelope_ref[mask]))
        if env_mean <= envelope_threshold:
            continue
        phase_rel = phase_segment - phase_segment[0]
        ref_seg = ref_band[mask]
        dut_seg = dut_band[mask]

        ref_shape = np.interp(
            phase_grid, phase_rel, ref_seg, left=ref_seg[0], right=ref_seg[-1]
        )
        dut_shape = np.interp(
            phase_grid, phase_rel, dut_seg, left=dut_seg[0], right=dut_seg[-1]
        )
        ref_shapes.append(ref_shape)
        dut_shapes.append(dut_shape)
        weights.append(env_mean)

    if not ref_shapes:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            0,
        )

    weight_arr = np.asarray(weights, dtype=np.float64)
    weight_arr = (
        weight_arr / np.sum(weight_arr) if np.sum(weight_arr) > 0 else weight_arr
    )
    ref_stack = np.vstack(ref_shapes)
    dut_stack = np.vstack(dut_shapes)
    mean_ref = np.average(ref_stack, axis=0, weights=weight_arr)
    mean_dut = np.average(dut_stack, axis=0, weights=weight_arr)
    return phase_grid, mean_ref, mean_dut, len(ref_shapes)


def _residual_signal(
    *,
    reference: npt.NDArray[np.float64],
    dut: npt.NDArray[np.float64],
    delay_samples: float,
    scale: float,
) -> npt.NDArray[np.float64]:
    n = int(reference.shape[0])
    shift = float(delay_samples)
    start = max(0, int(np.ceil(shift)))
    end = min(n, int(np.floor((n - 1) + shift)) + 1)
    if end <= start:
        return np.asarray([], dtype=np.float64)

    idx = np.arange(start, end, dtype=np.float64)
    base = np.arange(n, dtype=np.float64)
    ref_shifted = np.interp(idx - shift, base, reference, left=0.0, right=0.0)
    ref_out = np.asarray(ref_shifted, dtype=np.float64)
    dut_out = np.asarray(dut[start:end], dtype=np.float64)
    return np.asarray(dut_out - scale * ref_out, dtype=np.float64)
