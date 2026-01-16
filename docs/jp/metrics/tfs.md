# 時間微細構造（TFS: Temporal Fine Structure）

## 1. 概要

### 目的と意義

**時間微細構造（TFS）** 指標は、音声信号の高周波位相コヒーレンスの保存度を定量化します。定常的な歪みや包絡特性に焦点を当てる従来の指標とは異なり、TFSは**微細な時間構造**、すなわち狭帯域内の高速振動に特に焦点を当てます。これらの振動は、音色の明瞭性、空間定位、そして音声再現における「空気感」や「輝き」に関する重要な知覚情報を担います。

### 測定対象

TFSは狭帯域（通常2–8 kHz）を分析し、各帯域を以下に分離します：
- **包絡（エンベロープ）**：ゆっくりと変化する振幅の輪郭
- **微細構造（ファインストラクチャ）**：正規化された高速振動（搬送波のような波形）
- **瞬時位相**：アンラップされた位相軌跡

この指標は、参照信号とDUT（被測定デバイス）の微細構造間の**短時間相互相関（STCC）**を複数の時間フレームにわたって計算し、包絡エネルギーで重み付けします。これにより、以下が明らかになります：
- **時間的安定性**：微細構造が時間にわたってどれだけ一貫して保存されているか
- **位相コヒーレンス**：瞬時位相関係が安定しているかどうか
- **群遅延変動**：フィルタアーティファクトを示唆する帯域間のタイミング不一致

### なぜTFSが重要か

人間の聴覚は、2–8 kHz範囲の時間微細構造に非常に敏感です。その理由は以下の通りです：
- **ピッチ知覚**：TFSキューは、特に複雑音のピッチ弁別に寄与します
- **空間聴覚**：高周波における両耳間時間差（ITD）は包絡に依存しますが、TFSは「両耳鈍感性」を介して低周波ITDキューをサポートします
- **音色と明瞭性**：高調波間の正確な位相関係は、知覚される明るさ、刺々しさ、または滑らかさを決定します
- **過渡特性の詳細**：打楽器音のアタック特性は、保存されたTFSに依存します

TFS劣化が明らかにするもの：
- **位相歪み**：非最小位相フィルタ、オールパスアーティファクト、または群遅延の不規則性
- **ジッタと変調**：クロック不安定性、相互変調、またはFM歪み
- **スルーレート制限**：高速振動をぼかす帯域幅制限
- **非線形アーティファクト**：高調波生成またはクロスオーバー歪みによる搬送波構造の破壊

### 典型的な応用場面

- DAC再構成フィルタの評価（線形位相 vs 最小位相のトレードオフ）
- ジッタ、位相ノイズ、またはクロック不安定性の検出
- リサンプリング、ビット深度削減、または非可逆コーデックの影響評価
- アナログアンプ設計の比較（特にクラスD vs クラスA/Bの高周波特性）
- イコライゼーションまたはクロスオーバーネットワークからの群遅延異常の診断

---

## 2. 数学的定義

### 2.1 バンドパスフィルタリング

入力信号 \(x(t)\) は、バターワースバンドパスフィルタを用いて \(K\) 本の狭帯域に分解されます：

$$
H_k(s) = \frac{(s/\omega_c)^n}{(s/\omega_c)^n + \cdots + 1}
$$

ここで：
- \(k\) は中心周波数 \(f_k\) と帯域幅 \(\Delta f_k\) を持つ帯域のインデックス
- 帯域エッジ：\(f_{\text{low},k} = f_k - \Delta f_k/2\)、\(f_{\text{high},k} = f_k + \Delta f_k/2\)
- フィルタ次数：通常 \(n = 6\)（`sosfiltfilt` によるゼロ位相）

**デフォルト帯域**（カスタマイズ可能）：
- 帯域1：2000–3000 Hz
- 帯域2：3000–4000 Hz
- 帯域3：4000–6000 Hz
- 帯域4：6000–8000 Hz

出力：各帯域 \(k\) の帯域制限信号 \(x_k(t)\)。

### 2.2 ヒルベルト変換とTFS抽出

各帯域 \(k\) について、ヒルベルト変換により**解析信号**を計算します：

$$
z_k(t) = x_k(t) + j \mathcal{H}[x_k(t)]
$$

ここで \(\mathcal{H}[\cdot]\) はヒルベルト変換です。

**包絡**：
$$
A_k(t) = |z_k(t)| = \sqrt{\text{Re}[z_k(t)]^2 + \text{Im}[z_k(t)]^2}
$$

**瞬時位相**：
$$
\phi_k(t) = \text{unwrap}(\arg[z_k(t)])
$$

**微細構造**（正規化搬送波）：
$$
\text{TFS}_k(t) = \frac{\text{Re}[z_k(t)]}{\max(A_k(t), \epsilon)}
$$

ここで \(\epsilon = 10^{-12}\) は無音領域でのゼロ除算を防ぎます。

微細構造 \(\text{TFS}_k(t)\) は、包絡内の「搬送波」波形を捉えるゼロ平均、単位振幅の振動です。

### 2.3 短時間相関（STCC）

信号を重複フレームに分割します：
- **フレーム長**：\(L\) サンプル（デフォルト：25 ms）
- **ホップサイズ**：\(H\) サンプル（デフォルト：10 ms）
- **ウィンドウ**：ハン窓 \(w(n)\) でエッジアーティファクトを削減

サンプル \(n_m\) から始まる各フレーム \(m\) について：

**窓掛け微細構造**：
$$
\text{TFS}_{k,m}^{(\text{ref})}(n) = \text{TFS}_k^{(\text{ref})}(n_m + n) \cdot w(n), \quad n = 0, \ldots, L-1
$$

$$
\text{TFS}_{k,m}^{(\text{dut})}(n) = \text{TFS}_k^{(\text{dut})}(n_m + n) \cdot w(n)
$$

**ラグ探索付き正規化相互相関**：
$$
\rho_{k,m}(\tau) = \frac{\sum_n \text{TFS}_{k,m}^{(\text{ref})}(n) \cdot \text{TFS}_{k,m}^{(\text{dut})}(n - \tau)}{\|\text{TFS}_{k,m}^{(\text{ref})}\| \cdot \|\text{TFS}_{k,m}^{(\text{dut})}\|}
$$

ここで \(\tau\) は \([-\tau_{\max}, \tau_{\max}]\) の範囲（デフォルト：\(\tau_{\max} = 1\) ms）。

**フレームごとの相関**：
$$
\rho_{k,m} = \max_\tau \rho_{k,m}(\tau), \quad \tau_{k,m} = \arg\max_\tau \rho_{k,m}(\tau)
$$

**フレーム重み**（包絡ベース）：
$$
w_{k,m} = \frac{1}{L} \sum_{n=0}^{L-1} \left[ A_k^{(\text{ref})}(n_m + n) + A_k^{(\text{dut})}(n_m + n) \right] / 2
$$

閾値 \(T_{\text{env}}\)（デフォルト：ピークに対して-40 dB）未満の \(w_{k,m}\) を持つフレームは、ノイズによる相関アーティファクトを避けるため除外されます。

### 2.4 帯域とグローバル集約

**帯域ごとの相関**（フレーム全体の重み付き平均）：
$$
\rho_k = \frac{\sum_m w_{k,m} \cdot \rho_{k,m}}{\sum_m w_{k,m}}
$$

**帯域ごとの群遅延**（ラグの重み付き中央値）：
$$
\tau_k = \text{median}_w(\{\tau_{k,m}\}, \{w_{k,m}\})
$$

**グローバル平均相関**（全帯域、全フレーム）：
$$
\rho_{\text{mean}} = \frac{\sum_{k,m} w_{k,m} \cdot \rho_{k,m}}{\sum_{k,m} w_{k,m}}
$$

**5パーセンタイル相関**（ワーストケーステール）：
$$
\rho_{05} = \text{percentile}(\{\rho_{k,m}\}, 5)
$$

**相関分散**（時間的安定性）：
$$
\sigma^2_\rho = \frac{\sum_{k,m} w_{k,m} \cdot (\rho_{k,m} - \rho_{\text{mean}})^2}{\sum_{k,m} w_{k,m}}
$$

### 2.5 位相コヒーレンス

各帯域の中央値ラグ \(\tau_k\) を補償した後、**位相差**を計算します：
$$
\Delta \phi_k(t) = \phi_k^{(\text{ref})}(t) - \phi_k^{(\text{dut})}(t - \tau_k)
$$

\([-\pi, \pi]\) にラップ：
$$
\Delta \phi_k(t) \leftarrow \arg(e^{j \Delta \phi_k(t)})
$$

**円周平均（位相コヒーレンス）**：
$$
\text{coherence} = \left| \frac{1}{N_{\text{total}}} \sum_{k,t} e^{j \Delta \phi_k(t)} \right|
$$

ここで \(N_{\text{total}}\) は全帯域のサンプル総数です。

1.0に近い値は安定した位相整列を示し、0に近い値はランダムまたはドリフトする位相を示します。

### 2.6 群遅延統計

**群遅延標準偏差**（帯域間一貫性）：
$$
\sigma_{\tau} = \sqrt{\frac{1}{K} \sum_k (\tau_k - \bar{\tau})^2}
$$

ここで \(\bar{\tau}\) は帯域全体の平均群遅延です。

大きな \(\sigma_{\tau}\)（例：> 0.2 ms）は、非最小位相フィルタまたは群遅延リップルからの周波数依存遅延異常を示唆します。

---

## 3. 実装詳細

### 3.1 パラメータと推奨値

| パラメータ | 型 | デフォルト | 範囲/備考 |
|-----------|---|---------|---------|
| `sample_rate` | int | – | オーディオサンプルレート (Hz) |
| `freq_bands` | list of tuples | [(2000, 3000), (3000, 4000), (4000, 6000), (6000, 8000)] | (低, 高) 周波数ペア（Hz）；ナイキスト以下である必要あり |
| `filter_order` | int | 6 | バターワースフィルタ次数；高いほど急峻なロールオフだが群遅延リップルが増加 |
| `frame_length_ms` | float | 25.0 | 短時間フレーム期間（ms）；トレードオフ：短い = 時間分解能向上、長い = 周波数分解能向上 |
| `frame_hop_ms` | float | 10.0 | フレームホップサイズ（ms）；典型的なオーバーラップは50–75% |
| `max_lag_ms` | float | 1.0 | 相関探索の最大ラグ（ms）；予想されるジッタと小さな整列誤差をカバーする必要あり |
| `envelope_threshold_db` | float | -40.0 | このdBレベル未満のフレームを除外（ピーク包絡に対する相対値）；ノイズ支配的な相関を防止 |
| `window` | str | "hann" | STCCのウィンドウ関数；ハンはスペクトル漏れとメインローブ幅の良好なトレードオフを提供 |

### 3.2 疑似コード

```
function calculate_tfs_correlation(reference, dut, sample_rate, params):
  // 初期化
  nyquist = sample_rate / 2
  frame_length_samples = round(params.frame_length_ms * sample_rate / 1000)
  hop_samples = round(params.frame_hop_ms * sample_rate / 1000)
  max_lag_samples = round(params.max_lag_ms * sample_rate / 1000)
  window = hann(frame_length_samples)
  frame_starts = range(0, len(reference) - frame_length_samples + 1, hop_samples)

  // 帯域ごとの処理
  for each (low, high) in freq_bands:
    // 帯域抽出
    ref_band = bandpass_filter(reference, low, high, order=params.filter_order)
    dut_band = bandpass_filter(dut, low, high, order=params.filter_order)

    // ヒルベルト変換
    ref_analytic = hilbert(ref_band)
    dut_analytic = hilbert(dut_band)

    // 包絡と微細構造
    ref_envelope = abs(ref_analytic)
    dut_envelope = abs(dut_analytic)
    ref_fine = real(ref_analytic) / max(ref_envelope, EPS)
    dut_fine = real(dut_analytic) / max(dut_envelope, EPS)

    // 包絡閾値
    peak_envelope = max(max(ref_envelope), max(dut_envelope))
    threshold = peak_envelope * 10^(params.envelope_threshold_db / 20)

    // 短時間相関
    for each frame_start in frame_starts:
      frame_end = frame_start + frame_length_samples
      envelope_mean = mean((ref_envelope[frame_start:frame_end] + dut_envelope[frame_start:frame_end]) / 2)
      if envelope_mean <= threshold:
        continue  // 低エネルギーフレームをスキップ

      ref_frame = ref_fine[frame_start:frame_end] * window
      dut_frame = dut_fine[frame_start:frame_end] * window

      // ラグ探索付き正規化相互相関
      correlation, lag = max_normalized_xcorr(ref_frame, dut_frame, max_lag_samples)

      // 相関、ラグ、重みを保存
      correlations.append(correlation)
      lags.append(lag)
      weights.append(envelope_mean)

    // 帯域ごとの集約
    band_correlation[band] = weighted_mean(correlations, weights)
    band_group_delay[band] = weighted_median(lags, weights) * 1000 / sample_rate  // ms に変換

    // 位相コヒーレンス（ラグ補償後）
    ref_phase = unwrap(angle(ref_analytic))
    dut_phase = unwrap(angle(dut_analytic))
    lag_samples = weighted_median(lags, weights)
    ref_phase_aligned, dut_phase_aligned = overlap_with_lag(ref_phase, dut_phase, lag_samples)
    phase_diff = wrap(ref_phase_aligned - dut_phase_aligned)  // [-pi, pi] にラップ
    phase_vector_sum += sum(exp(1j * phase_diff))
    phase_count += length(phase_diff)

  // グローバル集約
  mean_correlation = weighted_mean(all_correlations, all_weights)
  percentile_05_correlation = percentile(all_correlations, 5)
  correlation_variance = weighted_variance(all_correlations, all_weights)
  phase_coherence = abs(phase_vector_sum) / phase_count
  group_delay_std_ms = std(band_group_delay values)

  return result
```

### 3.3 エッジケースと特別な処理

1. **空または低エネルギー信号**：帯域内のすべてのフレームが包絡閾値未満の場合、帯域相関はデフォルトで0.0、群遅延は0.0 msになります。メトリックは失敗しませんが、劣化した結果を報告します。

2. **未整列信号**：参照とDUTの長さが異なる場合、関数は `ValueError` を送出し、メッセージ `"reference/dut length mismatch; align signals first"` を表示します。TFSを計算する前に、パイロットトーンまたは相互相関を使用して信号を事前整列してください。

3. **ナイキスト付近のバンドエッジ**：いずれかの帯域の上限エッジがナイキスト周波数を超える場合、関数は `ValueError` を送出します。ユーザーはサンプルレートに基づいて `freq_bands` を調整する必要があります（例：44.1 kHzの場合、最高帯域は約20 kHzを超えてはいけません）。

4. **短い信号**：信号が1フレーム長より短い場合、有効フレーム長は信号長に削減され、ホップサイズはフレーム長に設定されます（オーバーラップなし）。3フレーム未満が抽出された場合、警告がログに記録されます。

5. **ラグ探索失敗**：`max_lag_ms` が制限的すぎて有効な相関ピークが見つからない場合、フレーム相関はデフォルトで0.0になります。整列の不確実性が大きい場合は `max_lag_ms` を増やしてください。

6. **負またはゼロの包絡閾値**：関数は `envelope_threshold_db` が負（dB相対）である必要があることを強制します。≥ 0の値は `ValueError` を送出します。

### 3.4 計算複雑度

- **フィルタリング**：\(\mathcal{O}(K \cdot N \cdot n)\)（\(K\) は帯域数、\(N\) は信号長、\(n\) はフィルタ次数）
- **ヒルベルト変換**：\(\mathcal{O}(K \cdot N \log N)\)（FFTベース）
- **帯域ごとのSTCC**：\(\mathcal{O}(M \cdot L \log L)\)（\(M\) はフレーム数、\(L\) はフレーム長、FFTベース相関）
- **全体**：\(\mathcal{O}(K \cdot N \log N + K \cdot M \cdot L \log L)\)

実行時間の目安：48 kHz、10秒、4帯域で 200 ms未満（最新ハードウェア）。

**メモリ**：ピークメモリ使用量は \(\mathcal{O}(K \cdot N)\)（帯域フィルタ信号と解析変換の保存用）。

---

## 4. 解釈ガイドライン

### 4.1 類似度指標

参照とDUTを比較する際：

**平均相関** (`mean_correlation`)：
$$
\rho_{\text{mean}} \in [0, 1] \quad \text{（高いほど良好）}
$$

- **≥ 0.90**：優れたTFS保存；微細構造が良好に維持されている
- **0.85–0.89**：非常に良好；軽微な位相歪みまたはジッタ
- **0.80–0.84**：許容可能；顕著だが深刻でないTFS劣化
- **< 0.80**：有意なTFS損失；フィルタアーティファクト、ジッタ、または非線形歪みを疑う

**5パーセンタイル相関** (`percentile_05_correlation`)：
- ワーストケースフレームを捉える；断続的なグリッチまたはドロップアウトの検出に有用
- **≥ 0.80**：頑健；有意な時間的外れ値なし
- **< 0.70**：一部のフレームで深刻なTFS崩壊を示す

**相関分散** (`correlation_variance`)：
- TFS相関の時間的安定性を測定
- **< 0.01**：安定；TFS品質が時間にわたって一貫
- **> 0.05**：不安定；散発的な劣化または断続的アーティファクト

### 4.2 帯域別分析

`band_correlations`：(低, 高) Hz → 相関値の辞書

- **全帯域 ≥ 0.9**：TFSが周波数範囲全体で均一に保存
- **1帯域 < 0.8**：局所的問題（例：共鳴、群遅延異常、またはその帯域の非線形性）
- **高帯域 < 0.8、低帯域 ≥ 0.9**：帯域幅制限、スルーレート制限、またはHFジッタを示唆

**例**：`band_correlations[(6000, 8000)] = 0.72` だが他の帯域は ≥ 0.88 の場合、以下を調査：
- DAC再構成フィルタのロールオフ
- アンプのHFでのスルーレート
- 6 kHz以上に集中したジッタまたは位相ノイズ

### 4.3 位相コヒーレンスと群遅延

**位相コヒーレンス** (`phase_coherence`)：
$$
\text{coherence} \in [0, 1] \quad \text{（高いほど良好）}
$$

- **≥ 0.95**：安定した位相関係；最小限の位相ジッタまたはドリフト
- **0.85–0.94**：中程度の位相不安定性；小さなジッタまたは群遅延リップル
- **< 0.85**：有意な位相歪み；非最小位相フィルタ、オールパスアーティファクト、またはクロック不安定性

**群遅延標準偏差** (`group_delay_std_ms`)：
- 帯域間タイミング一貫性を測定
- **< 0.1 ms**：優秀；最小限の周波数依存遅延
- **0.1–0.2 ms**：許容可能；軽微な群遅延変動（急峻なフィルタで一般的）
- **> 0.2 ms**：顕著；音色シフトまたは「スメア」知覚を引き起こす可能性

**帯域ごとの群遅延** (`band_group_delays_ms`)：
- 正の遅延（> 0）：その帯域でDUTが参照に対して遅延
- 負の遅延（< 0）：DUTが進んでいる（あまり一般的でない；プレリンギングを示す可能性）
- 帯域間の大きな変動：非フラット群遅延（例：楕円フィルタ、最小位相 vs 線形位相の不一致）

### 4.4 解釈のヒント

**シナリオ：平均相関 0.88、位相コヒーレンス 0.92、群遅延標準偏差 0.15 ms**
- **解釈**：軽微な位相不安定性と中程度の群遅延変動を伴う良好な全体TFS保存。軽いリップルを持つ良好に設計された最小位相フィルタの可能性。
- **アクション**：主観的な聴取と比較；聴覚的アーティファクトがなければ許容可能。刺々しいまたは不明瞭であれば、フィルタ設計を調査。

**シナリオ：平均相関 0.75、5パーセンタイル 0.60、分散 0.06**
- **解釈**：断続的な崩壊と高い時間的変動を伴う有意なTFS劣化。ジッタ、非線形歪み、または深刻な帯域幅制限を疑う。
- **アクション**：信号経路でジッタ源を確認、DACクロック品質を検証、band_correlationsを検査して懸念周波数領域を特定。

**シナリオ：帯域相関 [0.92, 0.90, 0.88, 0.70]（2–3、3–4、4–6、6–8 kHz）、位相コヒーレンス 0.89**
- **解釈**：HF帯域（6–8 kHz）で深刻なTFS損失を示すが、低帯域は良好に保存。位相コヒーレンスは中程度の影響を受けている。
- **アクション**：HF特有の問題を疑う：スルーレート制限、再構成フィルタロールオフ、または高周波に集中したジッタ。アンプまたはDAC仕様を検証。

**シナリオ：全相関 ≥ 0.95、位相コヒーレンス 0.98、群遅延標準偏差 0.05 ms**
- **解釈**：優れたTFS保存；最小限の位相歪みとフラット群遅延。デバイスは微細構造の完全性を維持。
- **アクション**：参照品質パフォーマンス；さらなる調査不要。

---

## 5. サンプル信号で何が分かるか

### 5.1 `generate.py` のテスト信号利用

リポジトリには複数のテスト信号タイプ（`src/microstructure_metrics/cli/generate.py` 参照）が含まれており、TFSの理解に有用です：

#### A. **マルチトーン** (`multitone`)

**生成パラメータ**：
- 異なる周波数の複数のサイン波（例：1 kHz、2 kHz、4 kHz、8 kHz）
- 等振幅または重み付き振幅

**TFS が明かすこと**：
- **高調波位相関係**：トーン間の相対位相が歪むとTFS相関が低下
- **相互変調**：非線形デバイスはIM積を導入し、微細構造を破壊
- **群遅延変動**：異なる周波数成分が異なる遅延を受けると位相コヒーレンスが低下

**無歪信号の期待TFS相関**：**> 0.95**

---

#### B. **トーンバースト** (`tone-burst`)

**生成パラメータ**（デフォルト値）：
- 8 kHz サイン波、10周期、±2 ms ハン窓フェード
- 鋭い過渡開始/停止を作成

**TFS が明かすこと**：
- **フィルタリンギング**：急峻なフィルタからのプレリンギングまたはポストリンギングが位相歪みとして現れる
- **位相非線形性**：非最小位相フィルタは過渡エッジ中にTFS相関を低下させる
- **スルーレート制限**：高速振動がぼやける → TFS相関低下

**例**：
- 参照（`tone-burst` デフォルト）：TFS相関 > 0.93
- 最小位相フィルタ（プレリングなし）のDUT：TFS相関 ~0.90–0.92
- 線形位相FIR（プレリンギングあり）のDUT：TFS相関 ~0.80–0.85、位相コヒーレンス低下

---

#### C. **スイープサイン** (`sweep`)

**生成パラメータ**：
- 20 Hz から 20 kHz への周波数スイープ（10秒）
- 対数または線形スイープ

**TFS が明かすこと**：
- **周波数依存歪み**：TFS帯域相関がどの周波数領域が影響を受けているかを明らかにする
- **群遅延異常**：スイープ中の急速な位相変化により、特定周波数でTFS相関が低下
- **非線形歪み**：高調波生成が微細構造を破壊、特に低周波（サブハーモニクス）と高周波（エイリアシング）

**例**：
- 参照（`sweep` デフォルト）：全帯域でTFS相関 > 0.90
- 群遅延リップルのあるDUT：帯域相関が変動（例：2–3、3–4、4–6、6–8 kHz で 0.92、0.88、0.85、0.78）
- HFロールオフのあるDUT：高帯域（6–8 kHz）相関が < 0.75 に低下

---

#### D. **AM変調トーン** (`modulated`)

**生成パラメータ**：
- 搬送波：4 kHz サイン波
- AM変調：10 Hz、深度 0.5（50%）

**TFS が明かすこと**：
- **包絡-搬送波分離**：TFSは包絡とは独立して搬送波位相を抽出；AM歪みは位相ジッタとして現れる
- **クロックジッタ**：変調サイドバンドはタイミング誤差に敏感 → TFS相関低下
- **相互変調**：非線形デバイスは搬送波と変調を混合し、微細構造を破壊

**例**：
- 参照（`modulated` デフォルト）：TFS相関 > 0.92
- クロックジッタ（±1サンプル RMS）のあるDUT：TFS相関 ~0.85–0.88、位相コヒーレンス ~0.90
- IM歪みのあるDUT：TFS相関 < 0.80、帯域相関が局所的問題を示す

---

### 5.2 比較マトリックス：相関値の意味

| 信号タイプ | 理想的相関 | < 0.85 の場合の解釈 |
|-----------|----------|------------------|
| `multitone` | > 0.95 | トーン間の位相歪み、IM歪み、または群遅延リップル |
| `tone-burst` (過渡) | > 0.92 | フィルタリンギング、位相非線形性、またはスルーレート制限 |
| `sweep` (広帯域) | > 0.90 | 周波数依存歪み、群遅延異常、またはHFロールオフ |
| `modulated` (AM) | > 0.92 | クロックジッタ、IM歪み、または包絡-搬送波結合 |

### 5.3 生成と分析の例

1. **参照信号を生成**（理想、高品質出力）：
   ```bash
   python -m microstructure_metrics.cli generate multitone \
     --duration 10 --sample-rate 48000 \
     --frequencies 1000 2000 4000 8000 \
     --output multitone_ref.wav
   ```

2. **劣化版をシミュレート**（例：ジッタを追加）：
   ```bash
   # 例：リサンプリングで軽いジッタ/エイリアシングを導入
   sox multitone_ref.wav -r 44100 dut_multitone.wav rate -v
   sox dut_multitone.wav -r 48000 dut_multitone_resampled.wav rate -v
   ```

3. **TFSと類似度を計算**：
   ```bash
   python -c "
   from microstructure_metrics.metrics.tfs import calculate_tfs_correlation
   import soundfile as sf
   ref, sr = sf.read('multitone_ref.wav')
   dut, sr = sf.read('dut_multitone_resampled.wav')
   result = calculate_tfs_correlation(
       reference=ref, dut=dut, sample_rate=sr,
       freq_bands=[(2000, 3000), (3000, 4000), (4000, 6000), (6000, 8000)]
   )
   print(f'TFS平均相関: {result.mean_correlation:.3f}')
   print(f'位相コヒーレンス: {result.phase_coherence:.3f}')
   print(f'群遅延標準偏差: {result.group_delay_std_ms:.3f} ms')
   print(f'帯域相関: {result.band_correlations}')
   "
   ```

4. **結果を解釈**：
   - 平均相関が 0.80 に低下：リサンプリングアーティファクトからの有意なTFS劣化
   - 位相コヒーレンスが 0.85 に低下：中程度の位相ジッタ導入
   - 群遅延標準偏差 > 0.2 ms：リサンプラーからの周波数依存タイミング誤差

---

## 6. 参考文献

### 理論的背景

- **聴覚科学**：Moore, B. C. J. (2012). *An Introduction to the Psychology of Hearing* (6th ed.). Brill. – 時間微細構造とピッチ知覚。
- **位相知覚**：Oxenham, A. J. (2018). How we hear: The perception and neural coding of sound. *Annual Review of Psychology*, 69, 27–50.
- **両耳聴覚**：Bernstein, L. R., & Trahiotis, C. (2002). Enhancing sensitivity to interaural delays at high frequencies by using "transposed stimuli". *Journal of the Acoustical Society of America*, 112(3), 1026–1036.

### 実装参考資料

- **ヒルベルト変換**：Marple, S. L. (1999). Computing the discrete-time analytic signal via FFT. *IEEE Transactions on Signal Processing*, 47(9), 2600–2603.
- **Scipy 信号処理**：[https://docs.scipy.org/doc/scipy/reference/signal.html](https://docs.scipy.org/doc/scipy/reference/signal.html)
- **NumPy FFT**：[https://numpy.org/doc/stable/reference/routines.fft.html](https://numpy.org/doc/stable/reference/routines.fft.html)

### 関連ドキュメント

- **指標の読み解きガイド**：`docs/jp/metrics-interpretation.md` – TFS を他の指標との文脈で説明。
- **信号仕様**：`docs/jp/signal-specifications.md` – 各テスト信号タイプの詳細パラメータ。
- **測定手順**：`docs/jp/measurement-setup.md` – レベル合わせと整列の実務的考慮。

### ソースコード

- **TFS 実装**：`src/microstructure_metrics/metrics/tfs.py`
  - `extract_tfs()`：単一帯域からTFS成分を抽出
  - `calculate_tfs_correlation()`：TFS相関、位相コヒーレンス、群遅延を計算

- **テスト信号生成**：`src/microstructure_metrics/cli/generate.py`
  - TFS評価用のマルチトーン、トーンバースト、スイープ、変調信号を含む

---

## 付録：TFS の一般的な落とし穴

1. **整列を忘れる**：TFSは事前整列された信号を必要とします。`calculate_tfs_correlation()` を呼び出す前に、パイロットトーンまたはグローバル相互相関を使用してください。

2. **不適切な帯域**：1 kHz未満またはナイキスト/2を超える帯域を選択するとTFS感度が低下します。典型的なオーディオでは2–8 kHzに固執してください。

3. **フレーム長が短すぎる**：約10 ms未満のフレームは周波数分解能を低下させ、ノイズ感度を増加させます。少なくとも20–30 msを使用してください。

4. **最大ラグが小さすぎる**：`max_lag_ms` が実際のジッタまたは整列不確実性よりも小さい場合、相関ピークが見逃されます。典型的なシナリオでは ≥1 ms に増やしてください。

5. **帯域相関を無視する**：単一の全体相関は局所的問題を隠す可能性があります。常に `band_correlations` を検査して周波数特有の問題を特定してください。

6. **異なるレベルの信号を比較する**：レベル不一致は包絡重み付けに影響します。TFSを計算する前に必ず参照とDUTをレベル合わせしてください。

7. **位相コヒーレンスを無視する**：高相関だが低位相コヒーレンスは、振幅は保存されているが位相歪みのある微細構造を示します。両方の指標を確認してください。
