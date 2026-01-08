from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from microstructure_metrics.metrics.binaural import BinauralResult
from microstructure_metrics.metrics.tfs import TFSCorrelationResult
from microstructure_metrics.metrics.transient import TransientResult

EPS: Final = 1e-12


@dataclass(frozen=True)
class DivergenceComponent:
    """A single feature divergence component contributing to MDI."""

    name: str
    distance: float
    weight: float
    scale: float
    samples_ref: int
    samples_dut: int


@dataclass(frozen=True)
class MicrostructureDistributionDivergenceResult:
    """MDI: Microstructure Distribution Divergence (Wasserstein-based).

    The total score is a weighted sum of per-feature 1D Wasserstein distances,
    after optional per-feature scaling:

        mdi_total = Σ_i weight_i * (W1_i / scale_i)

    Lower is better (more similar distributions).
    """

    mdi_total: float
    channels_total: float
    binaural_total: float
    channel_totals: Mapping[str, float]
    component_totals: Mapping[str, float]
    components: tuple[DivergenceComponent, ...]


def calculate_microstructure_distribution_divergence(
    *,
    tfs_by_channel: Mapping[str, TFSCorrelationResult],
    transient_by_channel: Mapping[str, TransientResult],
    binaural: BinauralResult | None = None,
    weights: Mapping[str, float] | None = None,
) -> MicrostructureDistributionDivergenceResult:
    """Compute MDI from existing short-time feature series/distributions.

    This implementation focuses on detecting "mostly OK but sometimes broken"
    patterns by comparing distributions (not just averages).

    Args:
        tfs_by_channel: Per-channel TFS results.
        transient_by_channel: Per-channel transient results.
        binaural: Optional binaural (BCP) result (stereo only).
        weights: Optional group weights. Supported keys:
            - "tfs"
            - "transient"
            - "binaural"

    Returns:
        MicrostructureDistributionDivergenceResult
    """
    group_weights = {
        "tfs": 1.0,
        "transient": 1.0,
        "binaural": 1.0,
    }
    if weights is not None:
        for key, value in weights.items():
            if key in group_weights:
                group_weights[key] = float(value)

    components: list[DivergenceComponent] = []
    channel_totals: dict[str, float] = {}

    for channel, tfs in tfs_by_channel.items():
        transient = transient_by_channel.get(channel)
        if transient is None:
            continue
        channel_components: list[DivergenceComponent] = []
        channel_total = 0.0

        # --- TFS: correlation distribution "distance to ideal(=1)" ---
        tfs_corr_dist = _wasserstein_to_constant(
            values=tfs.correlation_series,
            weights=tfs.correlation_weights,
            constant=1.0,
        )
        # Scale: 0.1 correlation deviation -> 1.0 in normalized score.
        tfs_corr_scale = 0.1
        tfs_corr_score = (tfs_corr_dist / max(tfs_corr_scale, EPS)) * group_weights[
            "tfs"
        ]
        channel_total += tfs_corr_score
        channel_components.append(
            DivergenceComponent(
                name=f"{channel}.tfs.correlation_to_ideal",
                distance=float(tfs_corr_dist),
                weight=float(group_weights["tfs"]),
                scale=float(tfs_corr_scale),
                samples_ref=1,
                samples_dut=int(
                    _count_finite_with_positive_weights(
                        tfs.correlation_series, tfs.correlation_weights
                    )
                ),
            )
        )

        # --- TFS: band group delay "distance to ideal(=0 ms)" ---
        delays = np.asarray(list(tfs.band_group_delays_ms.values()), dtype=np.float64)
        delay_dist = _wasserstein_to_constant(values=delays, weights=None, constant=0.0)
        delay_scale_ms = 0.2
        delay_score = (delay_dist / max(delay_scale_ms, EPS)) * group_weights["tfs"]
        channel_total += delay_score
        channel_components.append(
            DivergenceComponent(
                name=f"{channel}.tfs.band_group_delay_to_ideal_ms",
                distance=float(delay_dist),
                weight=float(group_weights["tfs"]),
                scale=float(delay_scale_ms),
                samples_ref=1,
                samples_dut=int(np.count_nonzero(np.isfinite(delays))),
            )
        )

        # --- Transient: event-feature distribution distances ---
        ref_events = transient.ref_events
        dut_events = transient.dut_events
        ref_peak_weights = np.asarray(
            [e.peak_value for e in ref_events], dtype=np.float64
        )
        dut_peak_weights = np.asarray(
            [e.peak_value for e in dut_events], dtype=np.float64
        )

        ref_attack = np.asarray(
            [e.attack_time_ms for e in ref_events], dtype=np.float64
        )
        dut_attack = np.asarray(
            [e.attack_time_ms for e in dut_events], dtype=np.float64
        )
        attack_dist = wasserstein_1d(
            ref_attack,
            dut_attack,
            x_weights=ref_peak_weights,
            y_weights=dut_peak_weights,
        )
        attack_scale_ms = 1.0
        attack_score = (attack_dist / max(attack_scale_ms, EPS)) * group_weights[
            "transient"
        ]
        channel_total += attack_score
        channel_components.append(
            DivergenceComponent(
                name=f"{channel}.transient.attack_time_ms",
                distance=float(attack_dist),
                weight=float(group_weights["transient"]),
                scale=float(attack_scale_ms),
                samples_ref=int(ref_attack.size),
                samples_dut=int(dut_attack.size),
            )
        )

        ref_width = np.asarray([e.width_ms for e in ref_events], dtype=np.float64)
        dut_width = np.asarray([e.width_ms for e in dut_events], dtype=np.float64)
        width_dist = wasserstein_1d(
            ref_width,
            dut_width,
            x_weights=ref_peak_weights,
            y_weights=dut_peak_weights,
        )
        width_scale_ms = 1.0
        width_score = (width_dist / max(width_scale_ms, EPS)) * group_weights[
            "transient"
        ]
        channel_total += width_score
        channel_components.append(
            DivergenceComponent(
                name=f"{channel}.transient.width_ms",
                distance=float(width_dist),
                weight=float(group_weights["transient"]),
                scale=float(width_scale_ms),
                samples_ref=int(ref_width.size),
                samples_dut=int(dut_width.size),
            )
        )

        # Keep track
        components.extend(channel_components)
        channel_totals[channel] = float(channel_total)

    channels_total = float(sum(channel_totals.values()))

    binaural_total = 0.0
    if binaural is not None:
        binaural_components, binaural_total = _binaural_components(
            binaural=binaural, group_weight=group_weights["binaural"]
        )
        components.extend(binaural_components)

    component_totals: dict[str, float] = {
        "channels": channels_total,
        "binaural": float(binaural_total),
    }

    mdi_total = float(channels_total + binaural_total)
    return MicrostructureDistributionDivergenceResult(
        mdi_total=mdi_total,
        channels_total=channels_total,
        binaural_total=float(binaural_total),
        channel_totals=channel_totals,
        component_totals=component_totals,
        components=tuple(components),
    )


def wasserstein_1d(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    x_weights: npt.ArrayLike | None = None,
    y_weights: npt.ArrayLike | None = None,
) -> float:
    """Compute the 1D Wasserstein (Earth mover's) distance for empirical distributions.

    This implementation supports optional non-negative sample weights and avoids
    relying on SciPy so that typing stays strict.
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    if xs.size == 0 and ys.size == 0:
        return 0.0

    xw = (
        np.asarray(x_weights, dtype=np.float64).ravel()
        if x_weights is not None
        else np.ones(xs.shape, dtype=np.float64)
    )
    yw = (
        np.asarray(y_weights, dtype=np.float64).ravel()
        if y_weights is not None
        else np.ones(ys.shape, dtype=np.float64)
    )
    if xw.shape != xs.shape:
        raise ValueError("x_weights must match x shape")
    if yw.shape != ys.shape:
        raise ValueError("y_weights must match y shape")

    xs, xw = _filter_finite_positive(xs, xw)
    ys, yw = _filter_finite_positive(ys, yw)
    if xs.size == 0 and ys.size == 0:
        return 0.0
    if xs.size == 0 or ys.size == 0:
        # Distance to empty distribution is undefined in strict terms;
        # here we treat it as 0 to keep report stable when a feature is absent.
        return 0.0

    # Aggregate weights by unique support values (to handle duplicates efficiently).
    xs_sorted_idx = np.argsort(xs, kind="mergesort")
    ys_sorted_idx = np.argsort(ys, kind="mergesort")
    xs_sorted = xs[xs_sorted_idx]
    ys_sorted = ys[ys_sorted_idx]
    xw_sorted = xw[xs_sorted_idx]
    yw_sorted = yw[ys_sorted_idx]

    ux, invx = np.unique(xs_sorted, return_inverse=True)
    uy, invy = np.unique(ys_sorted, return_inverse=True)
    wx_agg = np.bincount(invx, weights=xw_sorted).astype(np.float64, copy=False)
    wy_agg = np.bincount(invy, weights=yw_sorted).astype(np.float64, copy=False)

    sumx = float(np.sum(wx_agg))
    sumy = float(np.sum(wy_agg))
    if sumx <= EPS or sumy <= EPS:
        return 0.0
    wx_agg = wx_agg / sumx
    wy_agg = wy_agg / sumy

    support = np.sort(np.unique(np.concatenate([ux, uy]))).astype(
        np.float64, copy=False
    )
    if support.size < 2:
        return 0.0

    cdf_x = np.cumsum(wx_agg)
    cdf_y = np.cumsum(wy_agg)
    idx_x = np.searchsorted(ux, support, side="right") - 1
    idx_y = np.searchsorted(uy, support, side="right") - 1
    Fx = np.where(idx_x >= 0, cdf_x[idx_x], 0.0)
    Fy = np.where(idx_y >= 0, cdf_y[idx_y], 0.0)
    dx = np.diff(support)
    return float(np.sum(np.abs(Fx[:-1] - Fy[:-1]) * dx))


def _wasserstein_to_constant(
    *,
    values: npt.ArrayLike,
    weights: npt.ArrayLike | None,
    constant: float,
) -> float:
    vals = np.asarray(values, dtype=np.float64).ravel()
    if weights is None:
        w = np.ones(vals.shape, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != vals.shape:
            raise ValueError("weights must match values shape")
    vals, w = _filter_finite_positive(vals, w)
    if vals.size == 0:
        return 0.0
    return wasserstein_1d(vals, np.asarray([constant], dtype=np.float64), x_weights=w)


def _filter_finite_positive(
    values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if values.shape != weights.shape:
        raise ValueError("values/weights shape mismatch")
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return values[mask], weights[mask]


def _count_finite_with_positive_weights(
    values: npt.ArrayLike, weights: npt.ArrayLike
) -> int:
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if v.shape != w.shape:
        return 0
    return int(np.count_nonzero(np.isfinite(v) & np.isfinite(w) & (w > 0)))


def _binaural_components(
    *, binaural: BinauralResult, group_weight: float
) -> tuple[list[DivergenceComponent], float]:
    comps: list[DivergenceComponent] = []
    total = 0.0
    w = np.asarray(binaural.weights, dtype=np.float64)

    itd_ref = np.asarray(binaural.itd_ref_ms, dtype=np.float64)
    itd_dut = np.asarray(binaural.itd_dut_ms, dtype=np.float64)
    itd_mask = np.isfinite(itd_ref) & np.isfinite(itd_dut) & np.isfinite(w) & (w > 0)
    itd_dist = wasserstein_1d(
        itd_ref[itd_mask],
        itd_dut[itd_mask],
        x_weights=w[itd_mask],
        y_weights=w[itd_mask],
    )
    itd_scale_ms = 0.2
    total += (itd_dist / max(itd_scale_ms, EPS)) * group_weight
    comps.append(
        DivergenceComponent(
            name="binaural.itd_ms",
            distance=float(itd_dist),
            weight=float(group_weight),
            scale=float(itd_scale_ms),
            samples_ref=int(np.count_nonzero(itd_mask)),
            samples_dut=int(np.count_nonzero(itd_mask)),
        )
    )

    ild_ref = np.asarray(binaural.ild_ref_db, dtype=np.float64)
    ild_dut = np.asarray(binaural.ild_dut_db, dtype=np.float64)
    ild_mask = np.isfinite(ild_ref) & np.isfinite(ild_dut) & np.isfinite(w) & (w > 0)
    ild_dist = wasserstein_1d(
        ild_ref[ild_mask],
        ild_dut[ild_mask],
        x_weights=w[ild_mask],
        y_weights=w[ild_mask],
    )
    ild_scale_db = 1.0
    total += (ild_dist / max(ild_scale_db, EPS)) * group_weight
    comps.append(
        DivergenceComponent(
            name="binaural.ild_db",
            distance=float(ild_dist),
            weight=float(group_weight),
            scale=float(ild_scale_db),
            samples_ref=int(np.count_nonzero(ild_mask)),
            samples_dut=int(np.count_nonzero(ild_mask)),
        )
    )

    iacc_ref = np.asarray(binaural.iacc_ref, dtype=np.float64)
    iacc_dut = np.asarray(binaural.iacc_dut, dtype=np.float64)
    iacc_mask = np.isfinite(iacc_ref) & np.isfinite(iacc_dut) & np.isfinite(w) & (w > 0)
    iacc_dist = wasserstein_1d(
        iacc_ref[iacc_mask],
        iacc_dut[iacc_mask],
        x_weights=w[iacc_mask],
        y_weights=w[iacc_mask],
    )
    iacc_scale = 0.1
    total += (iacc_dist / max(iacc_scale, EPS)) * group_weight
    comps.append(
        DivergenceComponent(
            name="binaural.iacc",
            distance=float(iacc_dist),
            weight=float(group_weight),
            scale=float(iacc_scale),
            samples_ref=int(np.count_nonzero(iacc_mask)),
            samples_dut=int(np.count_nonzero(iacc_mask)),
        )
    )

    return comps, float(total)
