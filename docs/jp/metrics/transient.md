# Transient メトリクス

## 1. 概要

### 目的と意義

**Transient メトリクス**は、インパルス、クリック、または立ち上がりエッジなど、音楽のリアリズムに不可欠な急峻な振幅変化の鋭さとタイミングをデバイスがどれだけ保存できるかを定量化します。THD+Nなどの従来の指標は定常状態の歪みを測定しますが、過渡応答解析は**エッジ忠実度**に焦点を当てます：鋭い過渡応答を丸めたり、スメア化したり、プリリンギングアーティファクトを追加したりせずに再現する能力です。

### 測定対象

過渡応答解析は、包絡線を走査して閾値を超えるピーク（通常-25 dB）を検出することにより、信号内の複数の過渡イベントを検出し、各イベントの特徴を抽出します：

- **立ち上がり時間**：過渡応答がピーク振幅の10%から90%まで立ち上がるのにかかる時間
- **低レベル立ち上がり時間**：0.1%から10%まで立ち上がるのにかかる時間（微細なプリリンギングを捕捉）
- **エッジシャープネス**：ピーク付近の包絡線の最大傾き
- **幅**：ピーク振幅の30%での持続時間（スメア化/広がりを測定）
- **前エネルギー割合**：ピーク前のエネルギー対総エネルギー（プリリンギングを検出）
- **エネルギースキューネス**：ピーク周辺のエネルギー分布の非対称性

参照信号とDUT（被測定デバイス）のイベントを比較することで、以下を明らかにします：

- エッジの丸まり（遅い立ち上がり、低下したシャープネス）
- 過渡スメア化（より広いピーク）
- プリリンギングアーティファクト（増加した前エネルギー）
- タイミングシフト（立ち上がり時間のデルタ）

### なぜ過渡応答が重要か

人間の聴覚知覚は過渡構造に非常に敏感です：

- **空間定位**：過渡応答のタイミングは方向の手がかりをエンコード
- **音源識別**：立ち上がり包絡線は楽器と音を区別
- **ダイナミック知覚**：エッジのシャープネスは知覚される明瞭度とインパクトに関連
- **マスキング**：わずかなスメア化やプリリンギングでも、明瞭度が低下したり、可聴アーティファクトを作成したりする可能性

過渡メトリクスの劣化は以下を示します：

- アンプまたはDACのスルーレート制限
- フィルタリンギング（特に線形位相FIRまたは共振アナログフィルタ）
- 時間領域処理からのウィンドウイングアーティファクト
- 鋭いエッジをぼかす帯域制限

### 典型的な応用場面

- アンプやDACのスルーレート制限の検出
- フィルタリンギングとプリエコーアーティファクトの評価
- 異なるサンプルレートコンバータの再構成品質の比較
- 積極的なアンチエイリアシングまたはローパスフィルタによるエッジの丸まりの診断
- 非可逆コーデックやDSPチェーンでの過渡保存の評価

---

## 2. 数学的定義

### 2.1 包絡抽出

与えられた信号 \(x(t)\) について、**ヒルベルト変換**により包絡を計算します：

$$
\text{env}(t) = |\mathcal{H}(x(t))|
$$

ここで \(\mathcal{H}(\cdot)\) は解析信号を示します。これはエネルギーベースのウィンドウイングよりも低レベルのプリリンギングをよく保存します。

**オプション：スムージング**（ノイズ感度を低減するため）：

$$
\hat{\text{env}}(t) = \text{env}(t) * w(t)
$$

ここで \(w(t)\) は持続時間 \(T_{\text{smooth}}\) のハン窓です（デフォルト：0.05 ms）。\(T_{\text{smooth}} = 0\) の場合、スムージングは適用されません。

### 2.2 ピーク検出

閾値を超える \(\hat{\text{env}}(t)\) の局所最大値を見つけることで、過渡イベントを検出します：

$$
\text{Threshold} = \max\left(\alpha \cdot \max(\hat{\text{env}}(t)), \quad 0.5 \cdot P_{90}(\hat{\text{env}}(t)), \quad \epsilon\right)
$$

ここで：
- \(\alpha = 10^{\text{peak\_threshold\_db}/20}\)（デフォルト：-25 dBで \(\alpha = 0.0562\)）
- \(P_{90}(\cdot)\) は90パーセンタイル（ノイズフロア推定）
- \(\epsilon = 10^{-12}\)（ゼロ除算を防止）

**不感期間**：連続するピーク間の最小間隔 \(T_{\text{refract}}\)（デフォルト：2.5 ms）を強制して、二重カウントを回避します。

### 2.3 特徴抽出

インデックス \(n_p\) で検出された各ピークについて、\(n_p\) 周辺の \(\hat{\text{env}}(t)\) のローカルセグメント（通常±40 ms）から特徴を抽出します：

#### A. ピーク値と時刻

$$
\text{peak\_value} = \hat{\text{env}}(n_p), \quad \text{peak\_time} = \frac{n_p}{f_s}
$$

#### B. 立ち上がり時間（10%–90%）

\(n_p\) より前で包絡がピーク値の10%と90%を横切るインデックスを見つけます：

$$
n_{10\%} = \max\{n < n_p : \hat{\text{env}}(n) \leq 0.1 \cdot \text{peak\_value}\}
$$

$$
n_{90\%} = \min\{n > n_{10\%} : \hat{\text{env}}(n) \geq 0.9 \cdot \text{peak\_value}\}
$$

$$
\text{attack\_time} = \frac{n_{90\%} - n_{10\%}}{f_s}
$$

#### C. 低レベル立ち上がり時間（0.1%–10%）

微細なプリリンギングを捕捉するため：

$$
n_{0.1\%} = \max\{n < n_p : \hat{\text{env}}(n) \leq 0.001 \cdot \text{peak\_value}\}
$$

$$
n_{10\%}^{\text{low}} = \min\{n > n_{0.1\%} : \hat{\text{env}}(n) \geq 0.1 \cdot \text{peak\_value}\}
$$

$$
\text{low\_level\_attack\_time} = \frac{n_{10\%}^{\text{low}} - n_{0.1\%}}{f_s}
$$

#### D. エッジシャープネス

ピーク付近の最大包絡傾き（\(n_p\) 前の最後の3 ms）：

$$
\text{edge\_sharpness} = \max_{n \in [n_p - 3\,\text{ms}, n_p]} \left| \frac{d\hat{\text{env}}(n)}{dt} \right| \cdot f_s
$$

#### E. 30%での幅

ピーク振幅の30%での左右の交差を見つけます：

$$
n_{\text{left}} = \max\{n < n_p : \hat{\text{env}}(n) \leq 0.3 \cdot \text{peak\_value}\}
$$

$$
n_{\text{right}} = \min\{n > n_p : \hat{\text{env}}(n) < 0.3 \cdot \text{peak\_value}\}
$$

$$
\text{width} = \frac{n_{\text{right}} - n_{\text{left}}}{f_s}
$$

#### F. 前エネルギー割合

\(n_p\) 周辺のウィンドウ ±\(T_{\text{asym}}\)（デフォルト：3 ms）内の生信号エネルギーを使用：

$$
E_{\text{pre}} = \sum_{n=n_p-T_{\text{asym}}}^{n_p-1} x(n)^2, \quad E_{\text{post}} = \sum_{n=n_p+1}^{n_p+T_{\text{asym}}} x(n)^2
$$

$$
\text{pre\_energy\_fraction} = \frac{E_{\text{pre}}}{E_{\text{pre}} + E_{\text{post}}}
$$

#### G. エネルギースキューネス

エネルギー加重時間分布の3次モーメント：

$$
\mu = \frac{\sum_{n} (n - n_p) \cdot x(n)^2}{\sum_{n} x(n)^2}, \quad \sigma^2 = \frac{\sum_{n} (n - n_p - \mu)^2 \cdot x(n)^2}{\sum_{n} x(n)^2}
$$

$$
\text{energy\_skewness} = \frac{1}{\sigma^3} \cdot \frac{\sum_{n} (n - n_p - \mu)^3 \cdot x(n)^2}{\sum_{n} x(n)^2}
$$

負のスキューネスは、ピーク前のエネルギー集中（プリリンギング）を示します。

### 2.4 イベントマッチング

\(n_{\text{ref}}\) の各参照イベントについて、以下を満たす最も近い DUT イベント \(n_{\text{dut}}\) を見つけます：

$$
|n_{\text{dut}} - n_{\text{ref}}| \leq T_{\text{match}} \cdot f_s
$$

ここで \(T_{\text{match}}\) はマッチング許容範囲です（デフォルト：1.5 ms）。貪欲最近傍マッチングを使用してペアを形成します。

### 2.5 統計要約

すべてのマッチしたイベントペアについて、ペアごとの差分を計算します：

$$
\Delta_{\text{attack}}^{(i)} = \text{attack\_time}_{\text{dut}}^{(i)} - \text{attack\_time}_{\text{ref}}^{(i)}
$$

$$
r_{\text{sharpness}}^{(i)} = \frac{\text{edge\_sharpness}_{\text{dut}}^{(i)}}{\max(\text{edge\_sharpness}_{\text{ref}}^{(i)}, \epsilon)}
$$

$$
r_{\text{width}}^{(i)} = \frac{\text{width}_{\text{dut}}^{(i)}}{\max(\text{width}_{\text{ref}}^{(i)}, \epsilon)}
$$

**中央値**、**平均**、**5/95パーセンタイル**、**標準偏差**を使用して中心傾向と変動性を要約します。

**過渡スメア化指数**：\(r_{\text{width}}\) の中央値（> 1 の値は広がり/スメア化を示す）。

---

## 3. 実装詳細

### 3.1 パラメータと推奨値

| パラメータ | 型 | デフォルト | 範囲/備考 |
|-----------|---|---------|---------|
| `sample_rate` | int | – | オーディオサンプルレート (Hz) |
| `smoothing_ms` | float | 0.05 | 包絡スムージングウィンドウ (ms)；0 = ヒルベルトのみ |
| `peak_threshold_db` | float | -25.0 | 最大包絡に対するピーク検出閾値 (dB) |
| `refractory_ms` | float | 2.5 | 連続ピーク間の最小時間 (ms) |
| `match_tolerance_ms` | float | 1.5 | ref/DUTイベントマッチングの最大時間差 (ms) |
| `max_event_duration_ms` | float | 40.0 | 各ピーク周辺の特徴抽出用半ウィンドウ (ms) |
| `width_fraction` | float | 0.3 | 幅測定のピーク振幅の割合 (0–1) |
| `asymmetry_window_ms` | float | 3.0 | 前エネルギー/スキューネス計算用半ウィンドウ (ms) |

### 3.2 疑似コード

```
function calculate_transient_metrics(reference, dut, sample_rate, params):
  // 包絡抽出
  env_ref = hilbert_envelope(reference)
  env_dut = hilbert_envelope(dut)

  if smoothing_ms > 0:
    env_ref = smooth(env_ref, smoothing_ms)
    env_dut = smooth(env_dut, smoothing_ms)

  // ピーク検出
  ref_events = detect_events(reference, env_ref, sample_rate, params)
  dut_events = detect_events(dut, env_dut, sample_rate, params)

  // イベントマッチング
  pairs = match_events(ref_events, dut_events, tolerance=match_tolerance_ms)

  // ペアごとのメトリクス計算
  attack_deltas = [dut.attack_time - ref.attack_time for ref, dut in pairs]
  sharpness_ratios = [dut.edge_sharpness / max(ref.edge_sharpness, eps) for ref, dut in pairs]
  width_ratios = [dut.width / max(ref.width, eps) for ref, dut in pairs]
  pre_energy_deltas = [dut.pre_energy_fraction - ref.pre_energy_fraction for ref, dut in pairs]
  skewness_deltas = [dut.energy_skewness - ref.energy_skewness for ref, dut in pairs]

  // 統計要約
  return {
    attack_time_delta_ms: median(attack_deltas),
    edge_sharpness_ratio: median(sharpness_ratios),
    transient_smearing_index: median(width_ratios),
    pre_energy_fraction_delta: median(pre_energy_deltas),
    energy_skewness_delta: median(skewness_deltas),
    distribution_stats: {percentile_05, percentile_95, mean, std},
    matched_event_pairs: len(pairs),
    unmatched_ref_events: len(ref_events) - len(pairs),
    unmatched_dut_events: len(dut_events) - len(pairs),
  }
```

### 3.3 エッジケースと特別な処理

1. **空の信号またはピークが検出されない**：すべてのメトリクスにゼロ値を返します；信号振幅と閾値を確認してください。
2. **長さの不一致**：エラーを発生させます；事前に信号を整列する必要があります（整列モジュールを使用）。
3. **低SNR**：高いノイズフロアは誤ピークを引き起こす可能性があります；`peak_threshold_db` を増やすか、プリフィルタリングを適用してください。
4. **マッチしたペアがない**：`unmatched_ref_events` または `unmatched_dut_events` が大きい場合、`match_tolerance_ms` と信号整列を確認してください。
5. **幅が信号境界を越える**：信号エッジまでの利用可能なセグメントを使用します；エッジイベントの幅を過小評価する可能性があります。

### 3.4 計算複雑度

- ヒルベルト変換：\(\mathcal{O}(N \log N)\)（FFTベース）
- スムージング畳み込み：\(\mathcal{O}(N \cdot W)\)（\(W\) はウィンドウサイズ、通常小さい）
- ピーク検出：\(\mathcal{O}(N)\)
- イベントごとの特徴抽出：\(\mathcal{O}(W_{\text{event}})\)（通常40 ms × サンプルレート）
- イベントマッチング：\(\mathcal{O}(M \cdot K)\)（\(M\) と \(K\) はrefとDUTのイベント数）

**全体**：ヒルベルト変換に支配されて \(\mathcal{O}(N \log N)\)。

実行時間の目安：最新ハードウェアで48 kHz、10秒、~10イベントで 50 ms未満。

---

## 4. 解釈ガイドライン

### 4.1 主要メトリクス（中央値）

| メトリクス | 記号 | 解釈 |
|--------|--------|----------------|
| **立ち上がり時間デルタ** | \(\Delta t_{\text{attack}}\) | 正 → DUTが遅い；負 → DUTが速い |
| **低レベル立ち上がり時間デルタ** | \(\Delta t_{\text{low}}\) | 微細なプリリンギングを捕捉（0.1%–10%上昇） |
| **エッジシャープネス比** | \(r_{\text{sharpness}}\) | < 1 → 丸いエッジ；> 1 → 鋭いエッジ |
| **過渡スメア化指数** | \(r_{\text{width}}\) | > 1 → より広いピーク（スメア化）；< 1 → より狭いピーク |
| **前エネルギー割合デルタ** | \(\Delta E_{\text{pre}}\) | 正 → DUTでより多くのプリリンギング |
| **エネルギースキューネスデルタ** | \(\Delta S\) | 負 → ピーク前により多くのエネルギーシフト |

**参照値**（中央値）：
- 立ち上がり時間：クリーンインパルスで通常0.2–2 ms
- エッジシャープネス：信号帯域幅に依存（鋭い過渡応答ほど高い）
- 幅：刺激に依存（インパルス：<5 ms；クリック：5–20 ms）
- 前エネルギー割合：対称過渡応答で ~0.5；<0.3 はプリリンギングを示す

**DUT比較**：
- \(\Delta t_{\text{attack}} > 0.1\) ms：顕著な遅延
- \(r_{\text{sharpness}} < 0.9\)：有意なエッジの丸まり
- \(r_{\text{width}} > 1.1\)：顕著なスメア化
- \(\Delta E_{\text{pre}} > 0.05\)：検出可能なプリリンギング

### 4.2 分布統計

- **5/95パーセンタイル**：イベント間の変動性を捕捉
  - 大きな広がり（p95 - p05 > 中央値）は一貫性のない動作を示す
  - 最悪ケースイベントの `edge_sharpness_ratio_p05` と `transient_smearing_index_p95` を確認
- **標準偏差**：高い \(\sigma\) は非均一な劣化を示唆（例：レベル依存スルーレート制限）

### 4.3 イベント数

- `matched_event_pairs`：正常にペアリングされたref/DUTイベント数
- `unmatched_ref_events`：DUTで見つからない参照内のイベント（クリッピングまたは抑制の可能性）
- `unmatched_dut_events`：DUT内の余分なイベント（リンギングまたは偽ピークの可能性）

大きなアンマッチ数は以下を示します：
- 不十分な信号整列（整列モジュールを確認）
- 深刻な歪みまたはクリッピング
- 閾値の不一致（`peak_threshold_db` を調整）

### 4.4 解釈のヒント

**シナリオ：\(\Delta t_{\text{attack}} = +0.3\) ms、\(r_{\text{sharpness}} = 0.7\)**
- 可能性：ローパスフィルタまたはスルーレート制限がエッジを丸める
- 確認：周波数応答、アンプのスルーレート

**シナリオ：\(r_{\text{width}} = 1.4\)、\(\Delta E_{\text{pre}} = +0.1\)**
- 可能性：フィルタリンギングがプリエコーを追加し、過渡応答を広げる
- 確認：インパルス応答の振動、最小位相の代替を検討

**シナリオ：p95値が大きいが中央値はOK**
- 可能性：断続的なアーティファクト（例：大きな過渡応答でのクリッピング、レベル依存非線形性）
- 確認：イベントごとのデータ、信号レベルとの相関

**シナリオ：すべてのメトリクス ≈ 1.0、デルタ ≈ 0**
- 結論：優れた過渡保存

---

## 5. サンプル信号で何が分かるか

### 5.1 `generate.py` のテスト信号利用

リポジトリには複数のテスト信号タイプ（`src/microstructure_metrics/cli/generate.py` 参照）が含まれており、過渡解析に有用です：

#### A. **インパルス** (`impulse`)

**生成パラメータ**（デフォルト値）：
- 単一ディラックデルタ（1サンプル幅）または短いパルス（数サンプル）
- 振幅：-1 dBFS

**Transient が明かすこと**：
- **最もクリーンな参照**：最小限のスムージングで鋭いエッジ
- **立ち上がり時間**：理想的なインパルスで通常 <0.1 ms
- **エッジシャープネス**：与えられたサンプルレートで最大可能
- DUTが \(\Delta t_{\text{attack}} > 0.5\) ms または \(r_{\text{sharpness}} < 0.8\) を示す場合、アンチエイリアシングフィルタまたはDAC再構成フィルタを疑う

**ワークフロー例**：
```bash
# 参照インパルスを生成
python -m microstructure_metrics.cli generate impulse \
  --duration 1 --sample-rate 48000 \
  --output ref_impulse.wav

# デバイスを通して比較
python -m microstructure_metrics.cli report ref_impulse.wav dut_impulse.wav \
  --metrics transient --output report.json
```

透明なシステムの期待値：\(\Delta t_{\text{attack}} < 0.1\) ms、\(r_{\text{sharpness}} > 0.95\)、\(r_{\text{width}} < 1.1\)

---

#### B. **トーンバースト** (`tone-burst`)

**生成パラメータ**（デフォルト値）：
- 8 kHz サイン波、10周期、±2 ms ハン窓フェード
- 鋭い立ち上がりと減衰エッジを作成

**Transient が明かすこと**：
- **2つの過渡イベント**：立ち上がり（立ち上がりエッジ）と減衰（立ち下がりエッジ）
- **フィルタリンギングに敏感**：プリリンギングは低レベル立ち上がり時間と前エネルギー割合の増加として現れる
- **位相歪**：エネルギー分布をシフトできる（スキューネスデルタ）

**例**：
- 参照：対称エネルギーでクリーンバースト（\(E_{\text{pre}} \approx 0.5\)）
- 線形位相FIRのDUT：\(\Delta E_{\text{pre}} > 0.1\)、低レベル立ち上がり時間が増加
- 共振フィルタのDUT：\(r_{\text{width}} > 1.2\)、エネルギースキューネスが変化

---

#### C. **AM立ち上がり** (`am-attack`)

**生成パラメータ**（デフォルト値）：
- 1 kHz搬送波、振幅ゲーティング
- 立ち上がり：2 ms、立ち下がり：10 ms、周期：100 ms

**Transient が明かすこと**：
- **複数の過渡イベント**（ゲーティング周期ごとに1つ）
- **統計的頑健性**：多くのイベント間の中央値とパーセンタイルは、一貫した劣化と断続的劣化を明らかにする
- **スルーレート制限に敏感**：デバイスが速い振幅変化を追跡できない場合、\(\Delta t_{\text{attack}}\) が増加

**例**：
- 理想的なデバイス：すべてのイベントが一貫したメトリクスを示す（低std、p05 ≈ p95 ≈ 中央値）
- スルーレート制限デバイス：\(\Delta t_{\text{attack}}\) > 0.5 ms、\(r_{\text{sharpness}} < 0.8\)
- レベル依存デバイス：立ち上がり時間の高std（一部のイベントが他より速い）

---

### 5.2 比較マトリックス：メトリクスの意味

| 信号タイプ | 期待過渡数 | 主要メトリクス | 劣化時の解釈 |
|-------------|--------------------------|------------|----------------------------|
| `impulse` | 1（単一ピーク） | \(\Delta t_{\text{attack}}\)、\(r_{\text{sharpness}}\) | フィルタスムージングまたはDAC再構成アーティファクト |
| `tone-burst` | 2（立ち上がり + 減衰） | \(\Delta E_{\text{pre}}\)、低レベル立ち上がり時間 | 線形位相または共振フィルタからのプリリンギング |
| `am-attack` | ~10–100（周期ゲーティング） | 分布統計（p05、p95、std） | 断続的アーティファクト、スルーレート制限 |

### 5.3 生成と分析の例

1. **参照信号を生成**（理想、高品質出力）：
   ```bash
   python -m microstructure_metrics.cli generate impulse \
     --duration 1 --sample-rate 48000 \
     --output impulse_ref.wav
   ```

2. **劣化をシミュレート**（例：10 kHzローパスフィルタ）：
   ```bash
   sox impulse_ref.wav dut_impulse.wav lowpass 10000
   ```

3. **過渡メトリクスを計算**：
   ```bash
   python -c "
   from microstructure_metrics.metrics.transient import calculate_transient_metrics
   import soundfile as sf
   ref, sr = sf.read('impulse_ref.wav')
   dut, sr = sf.read('dut_impulse.wav')
   result = calculate_transient_metrics(
       reference=ref, dut=dut, sample_rate=sr
   )
   print(f'立ち上がり時間デルタ: {result.attack_time_delta_ms:.3f} ms')
   print(f'エッジシャープネス比: {result.edge_sharpness_ratio:.3f}')
   print(f'過渡スメア化指数: {result.transient_smearing_index:.3f}')
   "
   ```

4. **結果を解釈**：
   - 立ち上がり時間デルタ > 0.2 ms：有意な遅延（フィルタまたはスルーレートの問題）
   - エッジシャープネス比 < 0.85：顕著なエッジの丸まり
   - 過渡スメア化指数 > 1.15：明確な広がり/スメア化

---

## 6. 参考文献

### 理論的背景

- **聴覚知覚**：Cariani, P. A., & Delgutte, B. (1996). Neural correlates of the pitch of complex tones. I. Pitch and pitch salience. *Journal of Neurophysiology*, 76(3), 1698–1716. – 時間包絡と過渡コーディング。
- **過渡歪み**：Dunn, C., & Hawksford, M. O. (1993). Distortion immunity of MLS-derived impulse response measurements. *Journal of the Audio Engineering Society*, 41(5), 314–335.
- **スルーレート制限**：Cherry, E. M. (1982). A new distortion mechanism in class B amplifiers. *Journal of the Audio Engineering Society*, 30(11), 794–799.

### 実装参考資料

- **Scipy 信号処理**：[https://docs.scipy.org/doc/scipy/reference/signal.html](https://docs.scipy.org/doc/scipy/reference/signal.html) – ヒルベルト変換、ピーク検出。
- **NumPy Gradient**：[https://numpy.org/doc/stable/reference/generated/numpy.gradient.html](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html) – 包絡傾き計算。

### 関連ドキュメント

- **指標の読み解きガイド**：`docs/jp/metrics-interpretation.md` – 他の指標との文脈での過渡メトリクスの一般的ガイダンス。
- **信号仕様**：`docs/jp/signal-specifications.md` – インパルス、トーンバースト、AM立ち上がりテスト信号の詳細パラメータ。
- **測定手順**：`docs/jp/measurement-setup.md` – レベル合わせと整列の実務的考慮。

### ソースコード

- **Transient 実装**：`src/microstructure_metrics/metrics/transient.py`
  - `calculate_transient_metrics()`：過渡解析の中核
  - `TransientEvent`、`TransientResult`：イベント特徴と比較結果のデータ構造

- **テスト信号生成**：`src/microstructure_metrics/cli/generate.py`
  - 過渡評価用のインパルス、トーンバースト、AM立ち上がり信号を含む

---

## 付録：Transient の一般的な落とし穴

1. **定常信号の使用**：純粋なサイン波やホワイトノイズで過渡メトリクスは意味を持たない；常に鋭いエッジを持つ信号を使用する。
2. **不十分な整列**：不整列信号は誤検出タイミングシフトを引き起こす；常に整列モジュールを最初に使用する。
3. **厳しすぎる閾値**：イベントが検出されない場合、`peak_threshold_db` を下げる（例：-25 dBから-30 dBへ）。
4. **緩すぎるマッチング許容範囲**：`unmatched_dut_events` が高い場合、`match_tolerance_ms` を減らして誤マッチを回避する。
5. **分布統計を無視**：単一の中央値は断続的アーティファクトを隠す可能性；常にp05/p95とstdを確認する。
