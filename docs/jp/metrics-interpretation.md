# 指標の読み解きガイド

## 目的
- 本ツールが出力する指標（THD+N, NPS, ΔSE, MPS, TFS, Transient）の意味と読み方を整理し、機器比較や異常検出に活用する。
- 出力は `report` サブコマンドの JSON/CSV/Markdown に含まれる値を前提とする。

## 共通の見方
- 参照(ref)と被測定(DUT)をパイロットで整列した後の比較値。整列失敗やドリフト大のときは信頼度が下がる。
- dB値は大きいほど差が大きい（劣化）場合と、相関・コヒーレンス系で1.0に近いほど良好な場合がある。指標ごとに確認する。

## 指標の選び方（EPIC #38 の背景）
- 広帯域/低Qのノッチ埋まり・ノイズ底上げ → NPS（広めのノッチでSNRに比較的強い）。
- 非常に高Qなノッチの潰れ → PSDノッチ深さ（Welch PSDで狭帯域を測る。高分解能/長尺キャプチャが必要）。
- エッジ丸まり・過渡スメア → Transient指標（包絡の立ち上がり/幅を見る。インパルス/エッジ刺激が必須）。
- `docs/*/signal-specifications.md` にある想定刺激で実行すること（例: NPS/PSDノッチは `notched_noise`、エッジ試験はインパルス系）。異なる信号で実行すると値が無意味になる。

## 指標別のポイント

### THD+N
- 内容: 基本波に対する高調波+ノイズ比。`thd_n_db`, `thd_n_percent`, `sinad_db` を確認。
- 目安: `sinad_db` が高いほど良好。一般的なオーディオ計測では 90 dB 以上で良好、ハイエンドで 110 dB 以上を期待。
- 補足: `fundamental_level_dbfs` が想定とずれる場合、レベル合わせミスの可能性あり。

### Notch Preservation Score (NPS)
- 内容: リファレンスのノッチ深さが DUT でどれだけ埋まったか。`nps_db` (小さい/負が望ましい)、`nps_ratio`。
- 目安: 0 dB 以上でノッチが埋まり始めている。+3 dB 以上なら動的IMD/ノイズの影響が疑われる。
- `is_noise_limited` が true の場合、ノイズフロア不足で信頼度が下がる。
- 得意: 広めのノッチ（例: Q≈6–10）の埋まり検知や、SNRがほどほどの環境での健全性チェック。
- 入力条件: ノッチ付きピンクノイズ（`signal_type: notched_noise`、例: 8 kHz, Q≈8–10）を前提。ノッチが無い信号では意味を持たない。

### PSDノッチ深さ (Notch PSD)
- 内容: Welch PSD でノッチ中心±帯域のパワーを測定し、周辺リングとの比で深さを算出。高Qノッチの底上がり検出用。
- 出力: `notch_psd.notch_fill_db`（正で埋まり量）、`ref_notch_depth_db`、`dut_notch_depth_db`、`notch_bandwidth_hz`、`ring_bandwidth_hz`。
- 目安: `notch_fill_db` が 0 に近ければ維持。+6 dB 以上でノッチ埋まり傾向。深さが負ならノッチ消失の可能性。
- 補足: NPSより狭帯域で高Q設定に追従。ノイズフロアが浅い場合はばらつくため、複数試行の平均が安定。
- 得意: 高Q（例: Q≈20 以上）のノッチ潰れ評価。NPSが粗すぎるケースで使用。
- 入力条件: NPSと同じノッチ付きノイズだが、高分解能PSDを得るため十分な長さ/FFT分解能で収録（`signal-specifications.md`を参照）。ノッチ中心周波数やQがずれると結果が無効。

### Spectral Entropy ΔSE
- 内容: スペクトルエントロピー差分。DUTが平坦化すると ΔSE が正方向に増える。
- 目安: `delta_se_mean` が 0.02 以上で情報量劣化の兆候。0 に近いほどリファレンスに忠実。
- 局所変動を見るには `delta_se_max` や時系列 `delta_se_over_time` を確認。

### Modulation Power Spectrum (MPS)
- 内容: 変調テクスチャの類似度。`mps_correlation` (1に近いほど良好)、`mps_distance` (0に近いほど良好)。
- 目安: 相関 0.9 以上は良好、0.8 未満で変調成分の崩れを疑う。バンド別 `band_correlations` で帯域特定。
- 音楽的テクスチャが失われると相関が下がり距離が増える。

### Temporal Fine Structure (TFS)
- 内容: 高域微細位相の短時間相関。`mean_correlation`（STCC平均）、`percentile_05_correlation`（ワースト側分位点）、`correlation_variance`、`phase_coherence`、`group_delay_std_ms` を確認。
- 目安: `mean_correlation` 0.85 以上で良好。`percentile_05_correlation` が低い場合は一部フレームで崩れている可能性。`group_delay_std_ms` が大きい（>0.2 ms 程度）場合、帯域間の時間ずれが大きい可能性。
- 帯域別 `band_group_delays_ms` で特定の帯域の遅延ズレを確認。レポートには `frame_length_ms` `frame_hop_ms` `max_lag_ms` `envelope_threshold_db` が記録される（低包絡フレームは除外）。

### Transient / エッジ丸まり
- 内容: インパルス/クリックの立ち上がり丸まりをマルチイベントで検出。包絡を走査し、ピーク閾値(-25 dB相当)、不感時間(2.5 ms)、最大イベント長(40 ms)を用いて複数ピークを抽出し、マッチング許容1.5 msでref/dutを対応付ける。
- 確認指標: 中央値系の `attack_time_ms`(DUT), `attack_time_delta_ms`(DUT-ref), `edge_sharpness_ratio`, `transient_smearing_index`(幅比) に加え、分布評価として `edge_sharpness_ratio_p05/p95`, `transient_smearing_index_p95`, `attack_time_delta_p95_ms`、イベント数 `event_counts.*` を見る。幅はピーク30%の交差幅で評価。
- 目安: `attack_time_delta_ms` > 0 または `edge_sharpness_ratio` < 1 で鈍化傾向。`transient_smearing_index` > 1 で主峰が広がり(スメア)。p95 が大きい場合は局所的な悪化が混在。
- 得意: フィルタのロールオフやスルーレート不足による立ち上がり鈍化、ウィンドウ処理での丸まり検知。
- 入力条件: 単一インパルスや急峻ステップなどエッジを含む刺激（`signal-specifications.md`参照）。定常ノイズやサイン波では意味を持たない。
- ノイズ/位相ばらつき耐性: 包絡エネルギーを平滑してから特徴抽出。イベント数が0の場合は指標は0となる。

## 典型的な読み解き例
- 「NPS が +4 dB、ΔSE が +0.03」: ノッチが埋まり、情報量も損なわれている。動的IMDや高域ノイズの混入を疑う。
- 「MPS 相関 0.75, 距離大、TFS 相関 0.8」: テクスチャと位相微細構造がともに崩れており、フィードバックや帯域制限の影響が考えられる。
- 「THD+N/SINAD 良好だが ΔSE/MPS/TFS が劣化」: 定常指標では見えない微細構造の劣化。駆動系やフィルタ設定を再確認。

## 残留リスク・注意点
- 参照信号の質・SNRが低い場合、全指標の信頼度が低下する。
- 未整列や大きなドリフト時は `report` の `validation`/`drift` セクションを確認し、再測定またはパラメータ調整を行う。
- 極端な帯域外ノイズやクリッピングがあると、エントロピー・ノッチ系が過大に悪化して見えることがある。

## 関連ドキュメント
- 信号仕様・命名: `docs/jp/signal-specifications.md`
- 測定手順: `docs/jp/measurement-setup.md`
- CLI/API オプション: `docs/jp/api-cli-reference.md`
