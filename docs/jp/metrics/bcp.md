# バイノーラルキュー保存 (BCP: Binaural Cue Preservation)

## 1. 概要

### 目的と意義

**バイノーラルキュー保存 (BCP)** は、ステレオ再生における空間手がかりが参照に対して
どれだけ保存されているかを評価します。特に、定位や像の安定性に強く関与する次の 3 量に
着目します：

- **ITD**（Interaural Time Difference）：左右の相対遅延
- **ILD**（Interaural Level Difference）：左右の相対レベル差
- **IACC**（Interaural Cross-Correlation）：左右の相関（コヒーレンス）

BCP は、片チャンネル指標では見えにくいステレオ固有の劣化（ch依存の遅延、レベル差、
デコリレーションなど）を検出したいときに有効です。

### 測定対象

BCP は、聴覚フィルタバンクで帯域分解した後、**帯域×時間フレーム**ごとに ITD/ILD/IACC
を推定し、DUT と参照の差分をエネルギー重み付き統計で要約します。

### 典型的な用途

- DSP／リサンプル／フィルタによるステレオ像の揺れ・にじみ検出
- DAC/AMP チェーンのステレオ安定性比較
- L/R の対称性・整合性のストレステスト

---

## 2. 数学的定義

ステレオ信号 \(x_L[n], x_R[n]\) をサンプルレート \(f_s\) Hz で観測するとします。
BCP は参照と DUT それぞれで算出し、その差分を評価します。

### 2.1 フィルタバンク分析

左右チャンネルを \(B\) 本の帯域へ分解します（gammatone / mel）：

$$
y_{b,L}[n] = \mathrm{FB}_b(x_L[n]),\quad y_{b,R}[n] = \mathrm{FB}_b(x_R[n])
$$

ここで \(b=1,\ldots,B\) は帯域中心周波数 \(f_b\) を持つ帯域インデックスです。

### 2.2 短時間フレームと重み

各帯域 \(b\) で、フレーム長 \(L\)、ホップ \(H\) の短時間フレームに分割します。フレーム
\(m\) の重みを joint RMS として定義します：

$$
w_{b,m} = \sqrt{\frac{1}{N}\sum_{n} v_{b,m}[n]^2}
$$

ここで \(v_{b,m}\) は当該帯域フレーム内の
\(\{y^{(\mathrm{ref})}_{b,L}, y^{(\mathrm{ref})}_{b,R},
y^{(\mathrm{dut})}_{b,L}, y^{(\mathrm{dut})}_{b,R}\}\) を結合したサンプル列で、
\(N\) は結合後のサンプル数です。`envelope_threshold_db` に基づく閾値より小さい
フレームは無視します。

### 2.3 ILD（左右レベル差）

フレームごとの ILD を次で定義します：

$$
\mathrm{ILD}_{b,m} = 20\log_{10}\left(\frac{\mathrm{RMS}(y_{b,L})}{\mathrm{RMS}(y_{b,R})}\right)
$$

### 2.4 相互相関による ITD と IACC

左右の正規化相互相関をラグ \(\tau\in[-\tau_{\max},\tau_{\max}]\) の範囲で計算します：

$$
\rho_{b,m}(\tau) =
\frac{\sum_n y_{b,L}[n]\,y_{b,R}[n-\tau]}
{\|y_{b,L}\|\cdot\|y_{b,R}\|}
$$

相関の絶対値が最大となるラグを選びます：

$$
\tau^\* = \arg\max_{\tau} |\rho_{b,m}(\tau)|
$$

そして：

$$
\mathrm{ITD}_{b,m} = \frac{\tau^\*}{f_s}\cdot 1000 \;\;[\mathrm{ms}],\quad
\mathrm{IACC}_{b,m} = |\rho_{b,m}(\tau^\*)|
$$

参照とDUTそれぞれについて、対応する帯域信号を代入して
\(\mathrm{ITD}_{b,m}\) と \(\mathrm{IACC}_{b,m}\) を計算します。

### 2.5 参照との差分と集約

フレームごとの差分：

$$
\Delta \mathrm{ITD}_{b,m} = \mathrm{ITD}^{(\mathrm{dut})}_{b,m} - \mathrm{ITD}^{(\mathrm{ref})}_{b,m}
$$

$$
\Delta \mathrm{ILD}_{b,m} = \mathrm{ILD}^{(\mathrm{dut})}_{b,m} - \mathrm{ILD}^{(\mathrm{ref})}_{b,m}
$$

$$
\Delta \mathrm{IACC}_{b,m} = \mathrm{IACC}^{(\mathrm{dut})}_{b,m} - \mathrm{IACC}^{(\mathrm{ref})}_{b,m}
$$

BCP は、\(|\Delta \mathrm{ITD}|\)、\(|\Delta \mathrm{ILD}|\) を重み \(w_{b,m}\) で
重み付けした分位点で要約し、加えて：

- `iacc_p05`：\(\mathrm{IACC}^{(\mathrm{dut})}\) の重み付き 5 パーセンタイル
- `delta_iacc_median`：\(\Delta \mathrm{IACC}\) の重み付き中央値
- `itd_outlier_rate`：\(|\Delta \mathrm{ITD}| > T_{\mathrm{itd}}\) の重み付き割合

帯域別統計も同様に帯域内で要約します。

---

## 3. 実装詳細

実装は `src/microstructure_metrics/metrics/binaural.py` の
`calculate_binaural_cue_preservation()` です。

`report` 出力では `metrics.binaural.summary.*` と `metrics.binaural.band_stats.*`
に格納されます。

### 3.1 パラメータと推奨値

| パラメータ | 型 | デフォルト | 備考 |
|-----------|---|---------|------|
| `sample_rate` | int | – | サンプルレート (Hz) |
| `audio_freq_range` | (float,float) | (125, 8000) | 解析帯域 (Hz) |
| `num_audio_bands` | int | 16 | フィルタバンク本数 |
| `frame_length_ms` | float | 25.0 | フレーム長 (ms) |
| `frame_hop_ms` | float | 10.0 | ホップ (ms) |
| `max_itd_ms` | float | 1.0 | ITD 探索最大ラグ (ms) |
| `envelope_threshold_db` | float | -50.0 | 低エネルギーフレーム除外 |
| `itd_outlier_threshold_ms` | float | 0.2 | \(|\Delta \mathrm{ITD}|\) 外れ値閾値 |
| `filterbank` | str | "gammatone" | "gammatone" / "mel" |
| `filterbank_kwargs` | mapping | None | フィルタバンク固有設定 |

### 3.2 アルゴリズム概要

1. 参照/DUT の L/R をフィルタバンクで帯域分解。
2. 帯域×フレームごとに：
   - RMS 重みを計算し閾値未満ならスキップ
   - ILD を左右 RMS 比から計算
   - 相互相関（`max_itd_ms` 範囲）から ITD と IACC を推定
3. 参照との差分を取り、重み付き分位点等で要約。

### 3.3 エッジケースと特別な処理

- 入力は `(samples, 2)` のステレオで、参照と DUT は同形状が必須。
- 低エネルギーのフレームは `envelope_threshold_db` により除外されます。
- `audio_freq_range` は Nyquist 未満へクリップされ、不正値は `ValueError`。

### 3.4 計算複雑度

\(N\) サンプル、\(B\) 帯域、\(M\) フレームとして：

- フィルタバンク：\(\mathcal{O}(B\cdot N)\)
- フレーム相互相関（FFT）：\(\mathcal{O}(B\cdot M\cdot L\log L)\)

---

## 4. 解釈ガイドライン

### 4.1 主な出力

低いほど良い：
- `median_abs_delta_itd_ms`, `p95_abs_delta_itd_ms`, `itd_outlier_rate`
- `median_abs_delta_ild_db`, `p95_abs_delta_ild_db`

高いほど良い：
- `iacc_p05`（下側の尾が低い＝断続的なデコリレーション）

0 に近いほど良い：
- `delta_iacc_median`（負に偏るほど系統的なデコリレーション）

### 4.2 実務的な読み方の目安

- `itd_outlier_rate` が上がる：時間変動する ch 遅延差を疑う。
- `p95_abs_delta_ild_db` が大きい：断続的なレベル偏り（リミッタ、ch依存 EQ 等）。
- `iacc_p05` が低い：局所的な L/R デコリレーションで像がぼやけやすい。

帯域別の `band_stats` を見て、低域/中域/高域のどこで崩れているかを特定します。

---

## 5. 推奨テスト信号

BCP は ITD/ILD 構造を持つステレオ刺激が必須です。

### 5.1 信号タイプ

- **binaural-cues** (`binaural-cues`)：既知の ITD/ILD を付加した帯域制限ピンクノイズ
  （BCP 用に設計）。
- 像が安定した実ステレオ音源（妥当性確認用）。

### 5.2 例（CLI）

```
uv run microstructure-metrics generate binaural-cues \
  --duration 10 --sample-rate 48000 --itd-ms 0.35 --ild-db 6 \
  --output ref.wav

uv run microstructure-metrics report ref.wav dut.wav \
  --output-json report.json --plot
```

---

## 6. 参考文献

### 理論的背景

- Blauert, J. (1997). *Spatial Hearing* (Revised ed.). MIT Press.
- Moore, B. C. J. (2012). *An Introduction to the Psychology of Hearing* (6th ed.).
  Brill.

### 実装参考資料

- SciPy signal: https://docs.scipy.org/doc/scipy/reference/signal.html

### 関連ドキュメント

- 指標の読み解き: `docs/jp/metrics-interpretation.md`
- 信号仕様: `docs/jp/signal-specifications.md`
- 測定手順: `docs/jp/measurement-setup.md`

### ソースコード

- BCP 実装: `src/microstructure_metrics/metrics/binaural.py`
- 信号生成: `src/microstructure_metrics/signals/generator.py`
- BCP 可視化: `src/microstructure_metrics/visualization.py`

---

## 付録：BCP の一般的な落とし穴

1. **真のステレオでない**：モノラル化や `report --channels ch0/ch1` では BCP が
   有効に働きません。
2. **整列失敗／ドリフト**：ITD の変化に見えてしまうことがあります。
3. **低SNRフレーム**：ITD/IACC が不安定になるため、十分な信号エネルギーと閾値設定が重要です。
