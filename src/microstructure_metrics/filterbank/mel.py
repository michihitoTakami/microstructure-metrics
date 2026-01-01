from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy import signal as sp_signal

EPS: Final = 1e-12


def _hz_to_mel(freq_hz: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """HTK-style Hz→mel変換。"""

    hz = np.asarray(freq_hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """HTK-style mel→Hz変換。"""

    m = np.asarray(mel, dtype=np.float64)
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


@dataclass
class MelFilterbank:
    """メル尺度の三角形帯域をIIR帯域通過で近似するシンプルなフィルタバンク。"""

    sample_rate: int
    num_filters: int = 64
    low_freq: float = 20.0
    high_freq: float | None = None
    order: int = 4
    bandwidth_scale: float = 1.0

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
        if self.order < 1:
            raise ValueError("order must be >= 1")
        if self.bandwidth_scale <= 0:
            raise ValueError("bandwidth_scale must be positive")

        mel_low = float(_hz_to_mel(self.low_freq))
        mel_high = float(_hz_to_mel(high))
        mel_points = np.linspace(mel_low, mel_high, self.num_filters)
        centers = _mel_to_hz(mel_points)

        # 中心間の中点を境界として帯域を決定（降順で揃える）。
        edges = np.zeros(self.num_filters + 1, dtype=np.float64)
        edges[0] = self.low_freq
        edges[-1] = high
        midpoints = (centers[:-1] + centers[1:]) / 2
        edges[1:-1] = midpoints

        # 高域→低域の降順に並べ、gammatoneと同じ順序に揃える。
        self._centre_freqs = centers[::-1]
        self._edges = edges[::-1]

        sos_list: list[npt.NDArray[np.float64]] = []
        for low, high_edge in zip(self._edges[1:], self._edges[:-1], strict=False):
            low_cut = max(low, EPS)
            high_cut = min(high_edge * self.bandwidth_scale, nyquist * 0.99)
            if high_cut <= low_cut:
                high_cut = min(low_cut * 1.01, nyquist * 0.99)
            norm_band = [low_cut, high_cut]
            sos = sp_signal.butter(
                self.order,
                norm_band,
                btype="band",
                output="sos",
                fs=self.sample_rate,
            )
            sos_list.append(np.asarray(sos, dtype=np.float64))
        self._sos = sos_list

    @property
    def center_frequencies(self) -> npt.NDArray[np.float64]:
        """メル間隔の中心周波数（降順）。"""

        return self._centre_freqs

    def analyze(self, signal: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """各帯域の出力波形を返す。"""

        data = np.asarray(signal, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError("signal must be a 1-D array")
        if data.size == 0:
            raise ValueError("signal must not be empty")

        outputs = []
        for sos in self._sos:
            # 位相歪みを抑えるためfiltfiltを使用。
            filtered = sp_signal.sosfiltfilt(sos, data)
            outputs.append(filtered)
        return np.asarray(outputs, dtype=np.float64)

    def band_powers(self, signal: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """各帯域の平均パワー。"""

        bands = self.analyze(signal)
        power = np.mean(np.square(bands), axis=1)
        return np.asarray(power, dtype=np.float64)
