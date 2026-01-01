from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from gammatone import filters

EPS = 1e-12


@dataclass
class GammatoneFilterbank:
    """Wrapper around ``gammatone.filters`` to compute ERB-spaced band powers."""

    sample_rate: int
    num_filters: int = 64
    low_freq: float = 20.0
    high_freq: float | None = None
    width: float = 1.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.num_filters < 1:
            raise ValueError("num_filters must be at least 1")
        if self.low_freq <= 0:
            raise ValueError("low_freq must be positive")
        nyquist = self.sample_rate / 2
        high = self.high_freq or nyquist
        if high >= nyquist:
            high = nyquist * 0.99
        if high <= self.low_freq:
            raise ValueError("high_freq must be greater than low_freq")

        self._centre_freqs = np.asarray(
            filters.centre_freqs(
                self.sample_rate, self.num_filters, self.low_freq, f_max=high
            ),
            dtype=np.float64,
        )
        self._coefs = filters.make_erb_filters(
            self.sample_rate, self._centre_freqs, width=self.width
        )

    @property
    def center_frequencies(self) -> npt.NDArray[np.float64]:
        """ERB-spaced center frequencies (descending order)."""

        return self._centre_freqs

    def analyze(self, signal: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Apply the filterbank and return per-band signals (num_filters, n_samples)."""

        data = np.asarray(signal, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError("signal must be a 1-D array")
        if data.size == 0:
            raise ValueError("signal must not be empty")
        filtered = filters.erb_filterbank(data, self._coefs)
        return np.asarray(filtered, dtype=np.float64)

    def band_powers(self, signal: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return mean power per band."""

        bands = self.analyze(signal)
        power = np.mean(np.square(bands), axis=1)
        return np.asarray(power, dtype=np.float64)
