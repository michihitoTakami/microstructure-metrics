from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import signal


def estimate_delay(
    reference: npt.NDArray[np.float64],
    dut: npt.NDArray[np.float64],
    *,
    sample_rate: int,
    max_lag_ms: float = 100.0,
    refine: bool = True,
) -> float:
    """Estimate sample delay between reference and DUT using cross-correlation.

    Positive delay means the DUT starts later than the reference. The search
    window is limited by ``max_lag_ms`` to avoid spurious peaks.
    """
    ref = np.asarray(reference, dtype=np.float64)
    du = np.asarray(dut, dtype=np.float64)
    if ref.ndim != 1 or du.ndim != 1:
        raise ValueError("reference/dut must be 1-D signals.")
    if ref.size == 0 or du.size == 0:
        raise ValueError("reference/dut must not be empty.")

    max_len = min(ref.size, du.size)
    ref = ref[:max_len] - float(np.mean(ref[:max_len]))
    du = du[:max_len] - float(np.mean(du[:max_len]))

    max_lag_samples = max(int(sample_rate * max_lag_ms / 1000), 1)
    correlation = signal.correlate(du, ref, mode="full", method="fft")
    lags = np.arange(-ref.size + 1, du.size)
    window = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    if not np.any(window):
        raise ValueError("max_lag_ms is too small for the given signals.")
    correlation = correlation[window]
    lags = lags[window]

    peak_idx = int(np.argmax(correlation))
    peak_lag = float(lags[peak_idx])

    if refine and 0 < peak_idx < correlation.size - 1:
        y0 = correlation[peak_idx - 1]
        y1 = correlation[peak_idx]
        y2 = correlation[peak_idx + 1]
        denom = 2 * (y0 - 2 * y1 + y2)
        if denom != 0:
            peak_lag += (y0 - y2) / denom

    return peak_lag
