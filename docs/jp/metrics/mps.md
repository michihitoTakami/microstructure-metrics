# 変調パワースペクトラム (MPS: Modulation Power Spectrum)

## 1. 概要

### 目的と意義

**変調パワースペクトラム (MPS)** は、音声信号の包絡構造がどれだけ保存されているかを定量化します。THD+Nなどの従来の指標は定常的な歪みと高調波含有率を測定しますが、MPSは**変調テクスチャ**に焦点を当てます。つまり、過渡応答、ダイナミクスレンジ、音楽的なキャラクターに関する知覚的情報を担う時間変化する振幅変動を捉えます。

### 測定対象

MPSは、狭帯域フィルタバンク（ガンマトーンまたはメルフィルタバンク）を用いた音声の包絡を分析し、包絡の周波数成分（変調周波数）に分解します。結果は2次元スペクトラムとなり、以下の軸を持ちます：

- **聴覚周波数軸**（横軸）：フィルタバンドの中心周波数（通常100–8000 Hz）
- **変調周波数軸**（縦軸）：包絡変動の周波数（通常0.5–64 Hz）
- **パワー値**：各聴覚帯域および変調周波数での変調成分の大きさ

### なぜ変調構造が重要か

人間の聴覚は変調構造に非常に敏感です：
- **過渡特性の詳細**：立ち上がり形状、ビブラート、トレモロ
- **スペクトル動態**：エネルギー分布が時間とともにどのように変化するか
- **聴覚錯覚**：ラウドネス変動と周波数内容の相互作用

MPS劣化は、調和歪み測定では現れないフィルタリングアーティファクト、帯域制限、スルーレート制限、または包絡クリッピングを明らかにします。

### 典型的な応用場面

- 非可逆コーデックまたは積極的なDSP（EQ、圧縮）からのアーティファクト検出
- 非線形システムからの被害評価（ソフトクリッピング、調和波形変形）
- オーディオデバイス間の包絡保存度の比較
- スルーレート制限、帯域制限、またはクリッピングの診断

---

## 2. 数学的定義

### 2.1 フィルタバンク分析

入力信号は、以下いずれかを用いて \(M\) 本の狭帯域に分解されます：

**ガンマトーンフィルタバンク**（聴覚経路に基づく）：
$$G_i(f) = \frac{f^{n-1} e^{-2\pi B_{i} f / Q_i}}{(f^2 + 2 i \pi B_i f + Q_i^2 B_i^2)^n}$$

ここで \(Q_i\) は Q 値（中心周波数と帯域幅の比）、\(B_i\) は帯域幅、\(n\) はフィルタ次数（通常4）です。

**メルフィルタバンク**（音楽音響ベース）：
メル尺度上の三角形フィルタで、設定可能な次数と帯域幅スケーリング係数を使用します。

出力：\(m \times N\) 行列（\(m\) はバンド数、\(N\) はサンプル数）。

### 2.2 包絡抽出

各バンド \(i\) について、以下により包絡を計算します：

**ヒルベルト変換**：
$$\text{env}_i(n) = |\mathcal{H}(x_i(n))|$$

ここで \(\mathcal{H}(\cdot)\) は解析信号です。

**整流（絶対値）**：
$$\text{env}_i(n) = |x_i(n)|$$

オプション：DC成分除去
$$\tilde{\text{env}}_i(n) = \text{env}_i(n) - \frac{1}{N}\sum_{k=0}^{N-1} \text{env}_i(k)$$

オプション：低域カットフィルタ（バターワース、4次）
$$\hat{\text{env}}_i(n) = \text{LPF}(\tilde{\text{env}}_i(n), f_c)$$

ここで \(f_c\) は通常64 Hz（速い過渡を除去しながら音楽性を保持）。

### 2.3 変調スペクトラム計算

各バンド \(i\) について、FFTにより変調スペクトラムを計算します：
$$M_i(f_{\text{mod}}) = |\text{FFT}(\hat{\text{env}}_i(n))|$$

変調周波数範囲 \([f_{\text{mod,low}}, f_{\text{mod,high}}]\) に制限します。通常は \([0.5, 64]\) Hz。

**パワー**：
$$P_i(f_{\text{mod}}) = |M_i(f_{\text{mod}})|^2$$

**dB尺度**：
$$P_i^{\text{dB}}(f_{\text{mod}}) = 10 \log_{10}(\max(P_i(f_{\text{mod}}), \epsilon))$$

ここで \(\epsilon = 10^{-12}\) はログアンダーフローを防止します。

### 2.4 結果行列

形状：\(M \times K\)（\(K\) は変調周波数ビン数）

軸：
- `audio_freqs`：\(M\) 個のフィルタバンド中心周波数
- `mod_freqs`：変調周波数（線形またはログスケール）
- `mps_power`：パワー値（線形スケール）
- `mps_db`：パワー値（dBスケール）

---

## 3. 実装詳細

### 3.1 パラメータと推奨値

| パラメータ | 型 | デフォルト | 範囲/備考 |
|-----------|---|---------|---------|
| `sample_rate` | int | – | オーディオサンプルレート (Hz) |
| `audio_freq_range` | tuple | (100, 8000) | 聴覚帯域下限・上限 |
| `mod_freq_range` | tuple | (0.5, 64) | 変調周波数下限・上限 |
| `num_audio_bands` | int | 48 | フィルタバンド本数（32–64が典型的） |
| `filterbank` | str | "gammatone" | "gammatone" または "mel" |
| `envelope_method` | str | "hilbert" | "hilbert" または "rectify" |
| `envelope_lowpass_hz` | float | 64 | 包絡LPFカットオフ (Hz)；None で無効化 |
| `envelope_lowpass_order` | int | 4 | LPF次数（バターワース） |
| `mod_scale` | str | "linear" | "linear" または "log"（ログは知覚的等間隔に有用） |
| `mps_scale` | str | "power" | "power"（線形）または "log"（dB） |

### 3.2 疑似コード

```
function compute_mps(signal, sample_rate, params):
  // フィルタバンク分解
  band_signals = filterbank(signal, num_bands, freq_range)

  // バンドごとの包絡抽出
  for each band:
    env = hilbert(band) or rectify(band)
    if remove_dc:
      env -= mean(env)
    if lowpass_hz is not None:
      env = butterworth_lpf(env, lowpass_hz, order)

  // 変調FFT
  n_fft = next_power_of_two(length(envelopes[0]))
  mod_spectrum = rfft(envelopes, n_fft)
  mod_freqs = rfftfreq(n_fft, 1/sample_rate)

  // 範囲抽出
  mask = (mod_freqs >= freq_low) & (mod_freqs <= freq_high)
  mps_power = abs(mod_spectrum[:, mask]) ** 2
  mod_freqs = mod_freqs[mask]

  // オプション：ログ再スケーリング
  if mod_scale == "log":
    log_freqs = logspace(freq_low, freq_high, num_bins)
    mps_power = interpolate_rows(mps_power, mod_freqs, log_freqs)
    mod_freqs = log_freqs

  // パワーをdB変換
  if mps_scale == "log":
    mps_db = 10 * log10(max(mps_power, 1e-12))
  else:
    mps_db = mps_power

  return mps_power, mps_db, audio_freqs, mod_freqs
```

### 3.3 エッジケースと特別な処理

1. **DC除去**：デフォルトで常に実行（`remove_dc=True`）。変調周波数 = 0 でのエネルギー集中を防止します。
2. **変調周波数範囲フィルタリング**：`mod_freq_range` 外の全ビンは破棄；残りが無い場合はエラーを送出します。
3. **ログ周波数補間**：`mod_scale="log"` のとき、線形補間は各バンド行ごとに独立して実行され、バンドごとの構造を保持します。
4. **空または単一サンプル信号**：信号が空またはサンプル数不足の場合、`ValueError` を発生させます。

### 3.4 計算複雑度

- フィルタバンク分析：\(\mathcal{O}(N \cdot M \cdot \text{filter\_order})\)
- バンドごとのFFT：\(\mathcal{O}(M \cdot N \log N)\)
- 全体：FFTに支配されて \(\mathcal{O}(N \log N)\)

実行時間の目安：48 kHz、10秒で 100 ms未満（最新ハードウェア）。

---

## 4. 解釈ガイドライン

### 4.1 類似度指標

参照信号とDUT（被測定デバイス）を比較する際：

**相関**（高いほど良好）：
$$r_{\text{MPS}} = \frac{\text{cov}(\text{ref}_{\text{norm}}, \text{dut}_{\text{norm}})}{\sigma_{\text{ref}} \cdot \sigma_{\text{dut}}}$$

- **≥ 0.9**：変調テクスチャの優れた保存
- **0.85–0.89**：非常に良好；軽微なテクスチャ劣化
- **0.8–0.84**：許容可能；顕著だが深刻でないテクスチャ変化
- **< 0.8**：有意なテクスチャ損失；アーティファクトまたはフィルタリングの可能性

**距離**（低いほど良好）：
$$d_{\text{MPS}} = \sqrt{\frac{1}{M \cdot K} \sum_{i,j} (\text{ref}_{i,j} - \text{dut}_{i,j})^2}$$

- `mps_scale="log"` の場合、dB で表示
- 相関を補完；高相関でも全体的なシフトを隠せる場合がある

### 4.2 バンド別分析

`band_correlations`：聴覚周波数 → バンドごとの相関 の辞書

- **高（>0.9）**：バンドテクスチャは良好に保存
- **低（<0.8）**：そのバンドでのフィルタリング、非線形性、またはクリッピングを疑う
- 周波数全体のトレンドは、劣化が帯域限定か系統的かを明かします

### 4.3 変調周波数の重み付け

オプションの重みで、より高い変調周波数（例：4–64 Hz）を強調可能です。これは、知覚的に顕著な高速包絡変化を担います：

- `mod_weighting="high_mod"`：`[1, 1, ..., 2, 2, ..., 4, 4, ...]` を適用（4 Hz で2倍、10 Hz以上で4倍）
- 「音楽性」範囲の微妙なアーティファクト検出に有用

### 4.4 解釈のヒント

**シナリオ：相関 0.75、距離は中程度**
- 可能性：系統的な帯域制限（例：20 kHz カットオフ）、包絡クリッピング、またはスルーレート制限
- 確認：`band_correlations` で特定周波数範囲が影響を受けているかを確認

**シナリオ：相関 0.9 だが、あるバンド相関が 0.7**
- 可能性：そのバンド内の局所的問題（例：共鳴、非線形性、特定周波数のデジタルクリッピング）
- 確認：残差信号のスペクトログラム

**シナリオ：全バンド相関 ≥0.95**
- 結論：デバイスは変調テクスチャを良好に保存；優れた包絡忠実度

---

## 5. サンプル信号で何が分かるか

### 5.1 `generate.py` のテスト信号利用

リポジトリには複数のテスト信号タイプ（`src/microstructure_metrics/cli/generate.py` 参照）が含まれており、MPSの理解に有用です：

#### A. **変調信号** (`modulated`)

**生成パラメータ**（デフォルト値）：
- 搬送波：1000 Hz サイン波
- AM変調：4 Hz、深度 0.5（50%）
- FM変調：±50 Hz 偏移

**MPS が明かすこと**：
- 各聴覚バンドで明確な **4 Hz ピーク**
- 包絡変調が正しく抽出されていることを確認
- 参照 vs DUT 比較は、変調が保存されるか、またはスメア化されるかを示す
- AM/FM内容が減衰または周波数シフトする場合、MPS相関が大幅に低下

**ワークフロー例**：
```bash
# 参照変調信号を生成
python -m microstructure_metrics.cli generate modulated \
  --duration 10 --sample-rate 48000 \
  --carrier 1000 --am-freq 4 --am-depth 0.5 \
  --output ref_modulated.wav

# シミュレーション機器（例：16ビット量子化、ローパスフィルタ）を通す
# テストファイルを生成してMPS計算
python -m microstructure_metrics.cli report ref_modulated.wav dut_modulated.wav \
  --metrics mps --output report.json
```

無歪信号の期待MPS相関：**> 0.95**

---

#### B. **AM立ち上がり** (`am-attack`)

**生成パラメータ**（デフォルト値）：
- 1 kHz搬送波、振幅ゲーティング
- 立ち上がり：2 ms、立ち下がり：10 ms、周期：100 ms
- 明確な立ち上がり/立ち下がりを備えた繰り返し包絡を作成

**MPS が明かすこと**：
- 変調スペクトラムは **~10 Hz基本波**（ゲーティング繰り返し率）とその高調波を示す
- スルーレート不足のデバイスの **立ち上がり丸まり** に敏感
- デバイスのスルーレートが遅い場合、立ち上がりエッジがぼやける → 変調エネルギーが低周波へシフト
- 相関劣化は過渡シャープネスの損失を示す

**例**：
- 参照（`am-attack`デフォルト）：MPS は 10 Hz + 高調波に鋭く集中
- スルーレート制限 1 ms の DUT：MPS がブロード化、エネルギーが下方シフト、相関 ~0.8–0.85
- 極端なローパスのデバイス：MPS がフラット化、相関 < 0.7

---

#### C. **ノッチノイズ** (`notched-noise`)

**生成パラメータ**：
- ホワイトノイズ、20–20k Hz
- ノッチフィルタ中心周波数（例：8000 Hz）、Q = 8.6
- 狭帯域内容を除去、周辺の変調を保存

**MPS が明かすこと**：
- 変調スペクトラムは広帯域（0.5–64 Hz）だが、ノッチ聴覚周波数に **局所的な深さ** を示す
- フィルタリングアーティファクト周辺での包絡抽出のストレステスト
- デバイスがノッチ形状を保存する場合、MPS バンド相関はノッチ周波数 ≈ 周辺バンド相関
- 非線形デバイスはノッチを「埋める」可能 → mps_correlation がアーティファクト的に増加（残差構造が再出現）

---

#### D. **トーンバースト** (`tone-burst`)

**生成パラメータ**（デフォルト値）：
- 8 kHz サイン波、10周期、±2 ms ハン窓フェード
- 鋭い過渡開始/停止を作成

**MPS が明かすこと**：
- 包絡は **立ち上がりと減衰** を備える → 変調エネルギー ~10–100 Hz 範囲
- **フィルタリンギング** または **位相歪** に非常に敏感
- 前置きリンギングまたは後置きリンギングを追加するデバイス：包絡形状変化により MPS相関が低下
- 位相非線形性を「テクスチャ劣化」に偽装するのを検出するのに優秀

---

### 5.2 比較マトリックス：相関値の意味

| 信号タイプ | 理想的相関 | < 0.9 の場合の解釈 |
|-----------|----------|------------------|
| `modulated` (4 Hz AM) | > 0.95 | 変調スメア、包絡クリッピング、またはAM減衰 |
| `am-attack` (ゲーティング) | > 0.93 | スルーレート制限、立ち上がり丸まり、または遅い立ち上がり/立ち下がり |
| `tone-burst` (過渡) | > 0.92 | フィルタリンギング、前/後置きエコー、または位相非線形性 |
| `notched-noise` (ノッチ付広帯域) | > 0.90 | ノッチ埋め（非線形性）、包絡アーティファクト、または帯域拡張 |

### 5.3 生成と分析の例

1. **参照信号を生成**（理想、高品質出力）：
   ```bash
   python -m microstructure_metrics.cli generate modulated \
     --duration 10 --sample-rate 48000 \
     --carrier 1000 --am-freq 4 --am-depth 0.5 \
     --output modulated_ref.wav
   ```

2. **劣化版をシミュレート**（例：16 kHz ローパスフィルタ）：
   ```bash
   sox modulated_ref.wav dut_modulated.wav lowpass 16000
   ```

3. **MPS と類似度を計算**：
   ```bash
   python -c "
   from microstructure_metrics.metrics.mps import calculate_mps_similarity
   import soundfile as sf
   ref, sr = sf.read('modulated_ref.wav')
   dut, sr = sf.read('dut_modulated.wav')
   result = calculate_mps_similarity(
       reference=ref, dut=dut, sample_rate=sr,
       audio_freq_range=(100, 8000), mod_freq_range=(0.5, 64)
   )
   print(f'MPS相関: {result.mps_correlation:.3f}')
   print(f'バンド相関（サンプル）: {list(result.band_correlations.items())[:3]}')
   "
   ```

4. **結果を解釈**：
   - 相関が 0.70 に低下：DSP により包絡が大幅に変化
   - 高周波バンドのみ劣化：中域から高域のフィルタリング
   - 全バンドが均等に劣化：全体的な非線形性またはレベル不一致

---

## 6. 参考文献

### 理論的背景

- **聴覚科学**：Moore, B. C. J. (2012). *An Introduction to the Psychology of Hearing* (6th ed.). Brill. – 変調伝達関数と包絡知覚。
- **ガンマトーンフィルタバンク**：Holdsworth, J., Objé, C., Patterson, R., & Moore, B. C. (1988). Comparison of some auditory filter models. *Hearing Research*, 47(2–3), 103–120.
- **コーデック評価での MPS**：Thiede, T., Treurniet, W. C., Bitto, R., Schmidmer, C., Sporer, T., Beerends, J. G., & Colomes, C. (2000). PEAQ - The ITU standard for objective measurement of perceived audio quality. *Journal of the Audio Engineering Society*, 48(1/2), 3–29.

### 実装参考資料

- **Scipy 信号処理**：[https://docs.scipy.org/doc/scipy/reference/signal.html](https://docs.scipy.org/doc/scipy/reference/signal.html)
- **NumPy FFT**：[https://numpy.org/doc/stable/reference/routines.fft.html](https://numpy.org/doc/stable/reference/routines.fft.html)
- **フィルタバンク設計**：Vilkamo, J., Mäkinen, T., & Huopaniemi, J. (2006). Reconstruction of time-frequency representation for audio via Fourier inverse transform. *Proc. DSP*, 1–7.

### 関連ドキュメント

- **指標の読み解きガイド**：`docs/jp/metrics-interpretation.md` – MPS を他の指標との文脈で説明。
- **信号仕様**：`docs/jp/signal-specifications.md` – 各テスト信号タイプの詳細パラメータ。
- **測定手順**：`docs/jp/measurement-setup.md` – レベル合わせと整列の実務的考慮。

### ソースコード

- **MPS 実装**：`src/microstructure_metrics/metrics/mps.py`
  - `calculate_mps()`：MPS の中核計算
  - `calculate_mps_similarity()`：比較分析

- **テスト信号生成**：`src/microstructure_metrics/cli/generate.py`
  - MPS評価用の変調、am-attack、tone-burst その他信号を含む

- **フィルタバンク**：`src/microstructure_metrics/filterbank/` – ガンマトーンおよびメルフィルタバンク実装

---

## 付録：MPS の一般的な落とし穴

1. **DC除去を忘れる**：変調周波数 = 0 でのエネルギー巨大化、解釈を歪める。
2. **誤ったフィルタバンク選択**：メル vs ガンマトーンは異なるバンド相関を与える；どちらを使ったか記録する。
3. **変調周波数範囲が狭すぎ**：（例：4–16 Hz）高周波テクスチャを見落とす；最低でも 0.5–64 Hz を使用。
4. **大きく異なるレベルの信号比較**：相関計算前に必ず参照と DUT をレベル合わせ。
5. **バンド相関を無視**：単一の全体相関値は局所的問題を隠す可能性；常にバンド別データを確認。
