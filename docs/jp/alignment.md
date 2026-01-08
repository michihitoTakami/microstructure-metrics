# アライメント検出モジュール (S-04)

## 目的
- パイロットトーンを用いて入力/出力信号の開始位置・遅延を検出し、テスト本体区間を整列する。
- 以降の指標計算（THD+N など）を同一時間軸で比較できるようにする。

## CLI の使い方

```bash
uv run microstructure-metrics align ref.wav dut.wav \
  --pilot-freq 1000 \
  --threshold 0.5 \
  --band-width-hz 200 \
  --margin-ms 5 \
  --max-lag-ms 100
```

出力:
- `ref.aligned_ref.wav`, `dut.aligned_dut.wav`
- `ref_alignment.json`（delay_samples, confidence などメタ情報）

主なオプション:
- `--threshold`: 包絡正規化値の閾値 (0-1)。ノイズが多い場合は下げる。
- `--band-width-hz`: パイロット周波数周辺の帯域幅（±Hz）。
- `--min-duration-ms`, `--pilot-duration-ms`: パイロット検出に用いる最小長/想定長。
- `--margin-ms`: パイロット端から本体切り出し時のマージン。
- `--max-lag-ms`: 相互相関で探索するラグ上限。
- `--no-refine-delay`: サブサンプル補間を無効化。

前提:
- 2ch WAV を想定（モノラル入力も内部で2chへ複製）。同一サンプルレート（測定信号は 48 kHz を想定）。

## Python API

```python
from microstructure_metrics.alignment import align_audio_pair

result = align_audio_pair(
    reference=ref_signal,
    dut=dut_signal,
    sample_rate=48000,
    pilot_freq=1000.0,
    threshold=0.5,
    band_width_hz=200.0,
    min_duration_ms=90.0,
    pilot_duration_ms=100.0,
    margin_ms=5.0,
    max_lag_ms=100.0,
)
# result.aligned_ref, result.aligned_dut, result.delay_samples, result.confidence
```

## パラメータ調整の指針
- 低SNRやドロップアウトがある場合: `threshold` を 0.3 〜 0.4 に下げる、`band_width_hz` をやや広げる。
- パイロットが短い/長い場合: `min_duration_ms` と `pilot_duration_ms` を実測に合わせる。
- 大きな機器遅延が予想される場合: `max_lag_ms` を広げる（計算コストが増える）。
