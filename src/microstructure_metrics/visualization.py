from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402

from microstructure_metrics.metrics import MPSSimilarityResult, TFSCorrelationResult


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
