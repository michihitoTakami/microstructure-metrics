# 低周波複素再構成 (LFCR: Low-Frequency Complex Reconstruction)

## 1. 概要

### 目的と意義

**低周波複素再構成 (LFCR)** は、低域（ベース帯域）の**波形構造**がどれだけ忠実に
再現されているかを評価します。THD+N のような定常歪み指標では見えにくい
「ベースの締まり」「位相の回り込み」「低域のにじみ」などを、低域の位相構造に
敏感な手続きで定量化することを狙います。

LFCR が検出しやすい劣化例：
- 低域の群遅延や位相歪み（フィルタや再構成の影響）
- リサンプル／補間／タップ不足による低域の波形崩れ
- 包絡の局所的な不安定（グリッチ）によるパンチ感の乱れ

### 測定対象

LFCR は（整列後の）1ch 信号を対象に、低域帯域ごとに以下を併用して評価します：

1. **周期波形形状相関**：参照の瞬時位相で周期を区切り、位相に沿って再サンプルした
   1周期波形同士の相関をとる（位相条件付き比較）。
2. **倍音位相コヒーレンス**：基本周波数に条件付けた相対倍音位相の整合度を見る。
3. **包絡差アウトライア率**：包絡勾配（差分）のズレが参照の自然変動を超える割合を
   推定し、局所的な不安定を検出する。

### 典型的な用途

- 低域が強い素材での DAC/AMP チェーン比較
- 低域の位相ワープ、タイミングの smear（にじみ）、補間由来の波形崩れ検出
- デバイス間の「ベースの締まり」差の定量化

---

## 2. 数学的定義

整列済み 1-D 信号 \(x^{(\mathrm{ref})}[n]\), \(x^{(\mathrm{dut})}[n]\) を
サンプルレート \(f_s\) Hz で観測するとします。

### 2.1 帯域通過フィルタ

各低域帯域 \(b=[f_{\ell}, f_h]\) について Butterworth の帯域通過フィルタを作り、
ゼロ位相（前後フィルタ）で適用します：

$$
y_b[n] = \mathrm{filtfilt}\left(\mathrm{Butter}(n, f_{\ell}, f_h), x[n]\right)
$$

実装では SOS + `sosfiltfilt` を用います。

### 2.2 解析信号・包絡・位相

参照帯域信号 \(y_b^{(\mathrm{ref})}[n]\) からヒルベルト変換で解析信号を得ます：

$$
z_b[n] = y_b^{(\mathrm{ref})}[n] + j\,\mathcal{H}\{y_b^{(\mathrm{ref})}[n]\}
$$

包絡とアンラップ位相：

$$
A_b[n] = |z_b[n]|,\quad \phi_b[n] = \mathrm{unwrap}(\arg(z_b[n]))
$$

### 2.3 周期の切り出し（位相条件付き）

累積位相から周期IDを定義します：

$$
c[n] = \left\lfloor \frac{\phi_b[n] - \phi_b[0]}{2\pi} \right\rfloor
$$

周期 \(c\) は以下を満たすとき有効とします：
- サンプル数が 2 以上
- 位相スパンが \(0.75\cdot 2\pi\) 以上
- 周期内の平均包絡が閾値 \(A_{\min}\) を上回る

包絡閾値は `envelope_threshold_db` を用い、\(x^{(\mathrm{ref})}\) と
\(x^{(\mathrm{dut})}\) のグローバル最大振幅に対する相対値として決まります。

### 2.4 位相グリッドへの1周期再サンプル

1周期を \(P\) 点の位相グリッドで表します：

$$
\theta_k = \frac{2\pi k}{P}, \quad k = 0,\ldots,P-1
$$

周期内位相 \(\phi_c[n]\) を 0 始まりにシフトし、参照／DUT の帯域信号を
\(\theta_k\) 上へ線形補間します：

$$
s^{(\mathrm{ref})}_c[k] = \mathrm{interp}(\theta_k,\phi_c, y_b^{(\mathrm{ref})}),
\quad
s^{(\mathrm{dut})}_c[k] = \mathrm{interp}(\theta_k,\phi_c, y_b^{(\mathrm{dut})})
$$

### 2.5 周期波形形状の相関

周期ごとに Pearson 相関を計算します：

$$
\rho_c =
\frac{\sum_k (s^{(\mathrm{ref})}_c[k]-\bar{s}^{(\mathrm{ref})}_c)
(s^{(\mathrm{dut})}_c[k]-\bar{s}^{(\mathrm{dut})}_c)}
{\sqrt{\sum_k (s^{(\mathrm{ref})}_c[k]-\bar{s}^{(\mathrm{ref})}_c)^2}
\sqrt{\sum_k (s^{(\mathrm{dut})}_c[k]-\bar{s}^{(\mathrm{dut})}_c)^2}}
$$

重みは周期内平均包絡：

$$
w_c = \frac{1}{|c|}\sum_{n\in c} A_b[n]
$$

全周期・全帯域で集約し：
- `cycle_shape_corr_mean`：\(\rho_c\) の重み付き平均
- `cycle_shape_corr_p05`：\(\rho_c\) の重み付き 5 パーセンタイル

### 2.6 基本周波数推定（帯域ごと）

参照帯域に Hann 窓をかけた FFT を用い、探索範囲内の最大振幅ピークの周波数を
基本周波数 \(f_0\) とします。

### 2.7 倍音位相コヒーレンス

FFT の位相から、基本周波数と倍音 \(h f_0\)（\(h=2,\ldots,H\)）の位相を取得し、
基本位相に条件付けた相対倍音位相を定義します（\([-\pi,\pi]\) に wrap）：

$$
\psi^{(\cdot)}_h =
\mathrm{wrap}\left(\angle X^{(\cdot)}(h f_0) - h\,\angle X^{(\cdot)}(f_0)\right)
$$

参照との差：

$$
\Delta\psi_h = \mathrm{wrap}\left(\psi^{(\mathrm{dut})}_h
- \psi^{(\mathrm{ref})}_h\right)
$$

位相差ベクトルの平均長をコヒーレンスとします：

$$
\mathrm{coherence} =
\left|\frac{1}{K}\sum_{h=2}^{H} e^{j\Delta\psi_h}\right|
$$

### 2.8 包絡差アウトライア率

参照・DUT の包絡を最大値で正規化し、包絡勾配（差分）を比較します：

$$
g[n] = \Delta\left(\frac{A[n]}{\mathrm{scale}}\right)
$$

参照勾配から閾値を作ります：

$$
T = P_{95}(|g^{(\mathrm{ref})}|) + \mathrm{median}(|g^{(\mathrm{ref})}|)
$$

アウトライア率：

$$
r = \frac{1}{N}\sum_n \mathbf{1}\left(|g^{(\mathrm{dut})}[n]
- g^{(\mathrm{ref})}[n]| > T\right)
$$

---

## 3. 実装詳細

実装は `src/microstructure_metrics/metrics/bass.py` の
`calculate_low_freq_complex_reconstruction()` にあります。`report` の JSON 出力では
`metrics.bass.*` 配下に入ります（概念としては LFCR）。

### 3.1 パラメータと推奨値

| パラメータ | 型 | デフォルト | 備考 |
|-----------|---|---------|------|
| `sample_rate` | int | – | サンプルレート (Hz) |
| `bands_hz` | (low,high) の列 | ((20,80),(80,200)) | 低域帯域 (Hz) |
| `filter_order` | int | 4 | Butterworth 次数（ゼロ位相） |
| `cycle_points` | int | 128 | 1周期の位相グリッド点数 |
| `envelope_threshold_db` | float | -50.0 | グローバルピークに対する相対閾値 |
| `harmonic_max_order` | int | 5 | 2..N 倍音（Nyquist 未満） |
| `fundamental_search_hz` | (float,float) | (30,180) | 基本周波数探索範囲 (Hz) |

### 3.2 疑似コード

実装は 2 章の定義に対応して次の手順で計算します：

1. 帯域ごとに `reference`/`dut` を帯域通過（`sosfiltfilt`）。
2. 参照の解析信号から位相・包絡を取り、周期を切り出して位相グリッド上で
   周期波形相関を計算。
3. 参照 FFT から帯域ごとの基本周波数を推定し、倍音位相コヒーレンスを計算。
4. 包絡勾配の差分からアウトライア率を計算。
5. 周期指標は周期包絡重み、コヒーレンス／アウトライア率は帯域 RMS 重みで集約。

### 3.3 エッジケースと特別な処理

- 入力は 1-D かつ同一長である必要があり、空入力は `ValueError`。
- 帯域は `0 < low < high < Nyquist` を満たさないと `ValueError`。
- 有効周期が取れない（短すぎる／小さすぎる）場合、その帯域の周期指標は 0.0。
- 基本周波数推定が無効（\(f_0 \le 0\)）なら、その帯域のコヒーレンスは 0.0。

### 3.4 計算複雑度

\(B\) 帯域、\(N\) サンプルとして：
- フィルタ：\(\mathcal{O}(B\cdot N)\)
- ヒルベルト＋FFT：\(\mathcal{O}(B\cdot N\log N)\)
- 周期補間：\(\mathcal{O}(C\cdot P)\)（\(C\): 使用周期数）

---

## 4. 解釈ガイドライン

### 4.1 主な出力

高いほど良い：
- `cycle_shape_corr_mean`, `cycle_shape_corr_p05`（理想は 1.0 近傍）
- `harmonic_phase_coherence`（理想は 1.0 近傍）

低いほど良い：
- `envelope_diff_outlier_rate`（理想は 0.0 近傍）

帯域別の詳細は `band_metrics` にあり、`fundamental_hz` や使用倍音 `harmonic_orders`
も確認できます。

### 4.2 実務的な読み方の目安

- **平均が高いが p05 が低い**：概ね良いが、局所的／断続的に周期形状が崩れる
  （グリッチ、時間変動ワープ、局所クリップ等）可能性。
- **コヒーレンスが低い**（周期相関はそこそこ）：波形形状は似ているが、
  倍音位相の整合が崩れている（位相ワープ）可能性。
- **アウトライア率が高い**：包絡の微小不安定（パンチ感の乱れ）を疑う。

LFCR は低域タイミングに敏感なので、まずパイロット整列が正常であることを確認して
ください。

---

## 5. 推奨テスト信号

LFCR は、低域に意図的な位相/FM/PM 構造を含む信号で最も有効です。

### 5.1 信号タイプ

- **complex-bass** (`complex-bass`)：30–220 Hz 付近の FM/PM マルチトーン（推奨）
- **キック＋ベース合成**：実音源に近い複雑な位相／包絡
- **低域マルチトーン**：位相オフセットを制御した設計信号

### 5.2 例（CLI）

```
uv run microstructure-metrics generate complex-bass \
  --duration 10 --sample-rate 48000 --lowcut 30 --highcut 220 \
  --output ref.wav

uv run microstructure-metrics report ref.wav dut.wav \
  --output-json report.json --plot
```

---

## 6. 参考文献

### 理論的背景

- **聴覚科学**：Moore, B. C. J. (2012). *An Introduction to the Psychology of
  Hearing* (6th ed.). Brill.

### 実装参考資料

- **SciPy 信号処理**: https://docs.scipy.org/doc/scipy/reference/signal.html
- **NumPy FFT**: https://numpy.org/doc/stable/reference/routines.fft.html

### 関連ドキュメント

- **指標の読み解きガイド**：`docs/jp/metrics-interpretation.md`
- **信号仕様**：`docs/jp/signal-specifications.md`
- **測定手順**：`docs/jp/measurement-setup.md`

### ソースコード

- **LFCR（bass）実装**：`src/microstructure_metrics/metrics/bass.py`
- **信号生成（complex-bass）**：`src/microstructure_metrics/signals/generator.py`
- **LFCR 可視化**：`src/microstructure_metrics/visualization.py`

---

## 付録：LFCR の一般的な落とし穴

1. **信号が不適切**：純音や定常ノイズでは LFCR が効きにくい。
2. **整列を省略**：低域の微小ズレで周期比較が大きく崩れる。
3. **短すぎる信号**：周期数が不足し分位点が不安定になる。
4. **レベル不一致／クリップ**：周期形状と包絡勾配が不自然に崩れる。
