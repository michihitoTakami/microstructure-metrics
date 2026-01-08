from __future__ import annotations

import numpy as np

from microstructure_metrics.metrics.divergence import wasserstein_1d


def test_mdi_wasserstein_detects_local_breakdown_even_when_mean_matches() -> None:
    # "局所だけ崩す": 時間的に局所区間だけ値が大きく振れるが、平均は一致するケース
    n = 1000
    ref = np.zeros(n, dtype=np.float64)
    dut = ref.copy()

    # Local breakdown: symmetric spikes (+A then -A) so mean stays ~0.
    dut[100:150] = 1.0
    dut[150:200] = -1.0

    mean_diff = float(abs(np.mean(ref) - np.mean(dut)))
    mdi = wasserstein_1d(ref, dut)

    assert mean_diff == 0.0
    assert mdi > 0.05
