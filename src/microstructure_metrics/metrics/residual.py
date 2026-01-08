from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy import signal as sp_signal

from microstructure_metrics.alignment.correlation import estimate_delay

EPS: Final = 1e-12


@dataclass(frozen=True)
class ResidualMicrostructureResult:
    """残差マイクロストラクチャ（RMI）指標の計算結果。

    残差は、最良線形一致（スケール a と微小遅延 Δ）を除去した
    r(t) = dut(t) - a * ref(t - Δ) により定義する。
    """

    # Best linear match parameters
    delay_samples: float
    scale: float
    used_samples: int

    # Residual basic stats
    residual_rms: float
    residual_peak: float

    # Burstiness / impulsiveness
    kurtosis: float
    crest_factor: float
    p99_abs: float

    # Modulation structure (envelope modulation energy ratios)
    high_mod_ratio_4_64: float
    high_mod_ratio_10_64: float

    # Whiteness
    spectral_flatness: float
    autocorr_peak_excess: float
    autocorr_peak_lag_ms: float


def calculate_residual_microstructure(
    *,
    reference: npt.ArrayLike,
    dut: npt.ArrayLike,
    sample_rate: int,
    max_delay_lag_ms: float = 5.0,
    refine_delay: bool = True,
    refine_fit: bool = True,
    autocorr_max_lag_ms: float = 20.0,
    modulation_total_band_hz: tuple[float, float] = (0.5, 64.0),
    modulation_high_band_hz: tuple[float, float] = (4.0, 64.0),
    modulation_very_high_band_hz: tuple[float, float] = (10.0, 64.0),
) -> ResidualMicrostructureResult:
    """RMI（Residual Microstructure Information）を計算する。

    Args:
        reference: 参照信号（モノラル、同一長を想定）。
        dut: DUT信号（モノラル、同一長を想定）。
        sample_rate: サンプルレート(Hz)。
        max_delay_lag_ms: Δ推定で探索する最大ラグ(ms)。
        refine_delay: 相互相関ピークのサブサンプル補間を行うか。
        refine_fit: 推定したΔの近傍で、残差エネルギー最小となるΔを局所探索するか。
        autocorr_max_lag_ms: whiteness評価用の自己相関を見る最大ラグ(ms)。
        modulation_total_band_hz: 包絡変調エネルギー比の分母となる周波数帯域(Hz)。
        modulation_high_band_hz: 高変調エネルギー比（4-64Hzなど）の分子帯域(Hz)。
        modulation_very_high_band_hz: より高変調エネルギー比（10-64Hzなど）の分子帯域(Hz)。

    Returns:
        ResidualMicrostructureResult
    """
    ref = np.asarray(reference, dtype=np.float64)
    du = np.asarray(dut, dtype=np.float64)
    if ref.ndim != 1 or du.ndim != 1:
        raise ValueError("reference/dut must be 1-D signals")
    if ref.size == 0 or du.size == 0:
        raise ValueError("reference/dut must not be empty")
    if ref.shape[0] != du.shape[0]:
        raise ValueError("reference/dut length mismatch; align signals first")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if max_delay_lag_ms <= 0:
        raise ValueError("max_delay_lag_ms must be positive")
    if autocorr_max_lag_ms <= 0:
        raise ValueError("autocorr_max_lag_ms must be positive")

    delay_samples = estimate_delay(
        reference=ref,
        dut=du,
        sample_rate=sample_rate,
        max_lag_ms=max_delay_lag_ms,
        refine=refine_delay,
    )
    if refine_delay and refine_fit:
        delay_samples = _refine_delay_by_residual_energy(
            ref=ref,
            dut=du,
            delay_samples=delay_samples,
            half_width_samples=0.75,
            step_samples=0.05,
        )

    ref_aligned, du_aligned = _shift_reference_and_trim(
        ref=ref, dut=du, delay_samples=delay_samples
    )
    used_samples = int(ref_aligned.shape[0])
    if used_samples <= 0:
        raise ValueError("insufficient samples after delay compensation")

    scale = _least_squares_scale(reference=ref_aligned, dut=du_aligned)
    residual = np.asarray(du_aligned - scale * ref_aligned, dtype=np.float64)

    residual_rms = float(np.sqrt(np.mean(residual**2)))
    residual_peak = float(np.max(np.abs(residual))) if residual.size else 0.0

    kurt = _kurtosis(residual)
    crest = residual_peak / max(residual_rms, EPS)
    p99_abs = float(np.quantile(np.abs(residual), 0.99)) if residual.size else 0.0

    high_mod_ratio_4_64, high_mod_ratio_10_64 = _modulation_structure(
        residual=residual,
        sample_rate=sample_rate,
        total_band_hz=modulation_total_band_hz,
        high_band_hz=modulation_high_band_hz,
        very_high_band_hz=modulation_very_high_band_hz,
    )

    spectral_flatness = _spectral_flatness(residual=residual, sample_rate=sample_rate)
    ac_peak, ac_lag_ms = _autocorr_peak_excess(
        residual=residual, sample_rate=sample_rate, max_lag_ms=autocorr_max_lag_ms
    )

    return ResidualMicrostructureResult(
        delay_samples=float(delay_samples),
        scale=float(scale),
        used_samples=used_samples,
        residual_rms=residual_rms,
        residual_peak=residual_peak,
        kurtosis=float(kurt),
        crest_factor=float(crest),
        p99_abs=float(p99_abs),
        high_mod_ratio_4_64=float(high_mod_ratio_4_64),
        high_mod_ratio_10_64=float(high_mod_ratio_10_64),
        spectral_flatness=float(spectral_flatness),
        autocorr_peak_excess=float(ac_peak),
        autocorr_peak_lag_ms=float(ac_lag_ms),
    )


def _least_squares_scale(
    *, reference: npt.NDArray[np.float64], dut: npt.NDArray[np.float64]
) -> float:
    denom = float(np.dot(reference, reference))
    if denom <= EPS or not np.isfinite(denom):
        return 0.0
    numer = float(np.dot(dut, reference))
    if not np.isfinite(numer):
        return 0.0
    return numer / denom


def _shift_reference_and_trim(
    *, ref: npt.NDArray[np.float64], dut: npt.NDArray[np.float64], delay_samples: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """ref(t-Δ) を線形補間で作り、境界影響を避けてトリムして返す。"""
    n = int(ref.shape[0])
    shift = float(delay_samples)

    # valid indices satisfy: 0 <= i - shift <= n-1
    start = max(0, int(np.ceil(shift)))
    end = min(n, int(np.floor((n - 1) + shift)) + 1)
    if end <= start:
        raise ValueError("delay too large for trimming")

    idx = np.arange(start, end, dtype=np.float64)
    base = np.arange(n, dtype=np.float64)
    # ref_shifted[i] = ref[i - shift]  <=>  ref(t-Δ)
    ref_shifted = np.interp(idx - shift, base, ref, left=0.0, right=0.0)
    ref_out = np.asarray(ref_shifted, dtype=np.float64)
    dut_out = np.asarray(dut[start:end], dtype=np.float64)
    return ref_out, dut_out


def _refine_delay_by_residual_energy(
    *,
    ref: npt.NDArray[np.float64],
    dut: npt.NDArray[np.float64],
    delay_samples: float,
    half_width_samples: float,
    step_samples: float,
) -> float:
    """delay_samples 近傍で、最小二乗スケール後の残差エネルギーが最小となるΔを選ぶ。"""
    if half_width_samples <= 0 or step_samples <= 0:
        return float(delay_samples)
    n = int(ref.shape[0])
    if n == 0:
        return float(delay_samples)

    center = float(delay_samples)
    candidates = np.arange(
        center - half_width_samples,
        center + half_width_samples + (step_samples / 2.0),
        step_samples,
        dtype=np.float64,
    )
    if candidates.size == 0:
        return float(delay_samples)

    start = max(0, int(np.ceil(float(np.max(candidates)))))
    end = min(n, int(np.floor((n - 1) + float(np.min(candidates)))) + 1)
    if end <= start:
        return float(delay_samples)

    idx = np.arange(start, end, dtype=np.float64)
    base = np.arange(n, dtype=np.float64)
    dut_seg = np.asarray(dut[start:end], dtype=np.float64)

    best_delay = float(delay_samples)
    best_mse = float("inf")
    for cand in candidates:
        ref_shifted = np.interp(idx - float(cand), base, ref, left=0.0, right=0.0)
        denom = float(np.dot(ref_shifted, ref_shifted))
        if denom <= EPS or not np.isfinite(denom):
            continue
        a = float(np.dot(dut_seg, ref_shifted)) / denom
        resid = dut_seg - a * ref_shifted
        mse = float(np.mean(resid**2))
        if mse < best_mse:
            best_mse = mse
            best_delay = float(cand)
    return best_delay


def _kurtosis(x: npt.NDArray[np.float64]) -> float:
    if x.size < 4:
        return 0.0
    mean = float(np.mean(x))
    dev = x - mean
    v = float(np.mean(dev**2))
    if v <= EPS or not np.isfinite(v):
        return 0.0
    m4 = float(np.mean(dev**4))
    if not np.isfinite(m4):
        return 0.0
    return m4 / (v * v)


def _modulation_structure(
    *,
    residual: npt.NDArray[np.float64],
    sample_rate: int,
    total_band_hz: tuple[float, float],
    high_band_hz: tuple[float, float],
    very_high_band_hz: tuple[float, float],
) -> tuple[float, float]:
    total_low, total_high = total_band_hz
    if total_low <= 0 or total_high <= total_low:
        raise ValueError("modulation_total_band_hz must be an increasing (low, high)")
    nyquist = sample_rate / 2
    if total_high >= nyquist:
        raise ValueError("modulation_total_band_hz high must be below Nyquist")

    env = np.abs(sp_signal.hilbert(residual))
    env = np.asarray(env - np.mean(env), dtype=np.float64)
    if env.size == 0:
        return 0.0, 0.0

    n_fft = 1 << (max(env.size - 1, 0).bit_length())
    spec = np.fft.rfft(env, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    power = np.abs(spec) ** 2

    def _band_energy(band: tuple[float, float]) -> float:
        low, high = band
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0
        return float(np.sum(power[mask]))

    total = _band_energy(total_band_hz)
    if total <= EPS or not np.isfinite(total):
        return 0.0, 0.0
    high = _band_energy(high_band_hz)
    very_high = _band_energy(very_high_band_hz)
    return high / total, very_high / total


def _spectral_flatness(*, residual: npt.NDArray[np.float64], sample_rate: int) -> float:
    if residual.size == 0:
        return 0.0
    nperseg = min(4096, residual.size)
    if nperseg < 16:
        nperseg = residual.size
    _, psd = sp_signal.welch(residual, fs=sample_rate, nperseg=nperseg)
    psd = np.asarray(psd, dtype=np.float64)
    if psd.size == 0:
        return 0.0
    psd = np.maximum(psd, EPS)
    geo = float(np.exp(np.mean(np.log(psd))))
    arith = float(np.mean(psd))
    if arith <= 0 or not np.isfinite(geo) or not np.isfinite(arith):
        return 0.0
    return geo / arith


def _autocorr_peak_excess(
    *, residual: npt.NDArray[np.float64], sample_rate: int, max_lag_ms: float
) -> tuple[float, float]:
    if residual.size == 0:
        return 0.0, 0.0
    max_lag_samples = max(int(sample_rate * max_lag_ms / 1000), 1)
    x = np.asarray(residual - float(np.mean(residual)), dtype=np.float64)
    ac = sp_signal.correlate(x, x, mode="full", method="fft")
    center = x.size - 1
    ac0 = float(ac[center])
    if ac0 <= EPS or not np.isfinite(ac0):
        return 0.0, 0.0

    start = max(0, center - max_lag_samples)
    end = min(ac.size, center + max_lag_samples + 1)
    window = np.asarray(ac[start:end] / ac0, dtype=np.float64)
    lags = np.arange(start - center, end - center, dtype=np.int64)
    nonzero = lags != 0
    if not np.any(nonzero):
        return 0.0, 0.0
    window = window[nonzero]
    lags = lags[nonzero]

    idx = int(np.argmax(np.abs(window)))
    peak = float(np.abs(window[idx]))
    lag_ms = float(lags[idx]) / float(sample_rate) * 1000.0
    return peak, abs(lag_ms)
