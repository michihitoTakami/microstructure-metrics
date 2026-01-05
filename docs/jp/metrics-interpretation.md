# 指標の読み解きガイド

## 目的
- 本ツールが出力する指標（THD+N, MPS, TFS, Transient）の意味と読み方を整理し、機器比較や異常検出に活用する。
- 出力は `report` サブコマンドの JSON/CSV/Markdown に含まれる値を前提とする。

## 共通の見方
- 参照(ref)と被測定(DUT)をパイロットで整列した後の比較値。整列失敗やドリフト大のときは信頼度が下がる。
- dB値は大きいほど差が大きい（劣化）場合と、相関・コヒーレンス系で1.0に近いほど良好な場合がある。指標ごとに確認する。

## 指標の選び方
- 定常歪み/ゲイン確認 → THD+N/SINAD。
- テクスチャ・変調差分 → MPS。
- 高域微細位相の安定性 → TFS。
- エッジ丸まり・過渡スメア → Transient（インパルス/エッジ刺激が必須）。
- `docs/*/signal-specifications.md` にある想定刺激で実行すること。異なる信号で実行すると値が無意味になる。

## 指標別のポイント

### THD+N
- 内容: 基本波に対する高調波+ノイズ比。`thd_n_db`, `thd_n_percent`, `sinad_db` を確認。
- 目安: `sinad_db` が高いほど良好。一般的なオーディオ計測では 90 dB 以上で良好、ハイエンドで 110 dB 以上を期待。
- 補足: `fundamental_level_dbfs` が想定とずれる場合、レベル合わせミスの可能性あり。

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
- 「MPS 相関 0.75, 距離大、TFS 相関 0.8」: テクスチャと位相微細構造がともに崩れており、フィードバックや帯域制限の影響が考えられる。
- 「THD+N/SINAD 良好だが MPS/TFS が劣化」: 定常指標では見えない微細構造の劣化。駆動系やフィルタ設定を再確認。

## 残留リスク・注意点
- 参照信号の質・SNRが低い場合、全指標の信頼度が低下する。
- 未整列や大きなドリフト時は `report` の `validation`/`drift` セクションを確認し、再測定またはパラメータ調整を行う。
- 極端な帯域外ノイズやクリッピングがあると、エントロピー・ノッチ系が過大に悪化して見えることがある。

## 関連ドキュメント
- 信号仕様・命名: `docs/jp/signal-specifications.md`
- 測定手順: `docs/jp/measurement-setup.md`
- CLI/API オプション: `docs/jp/api-cli-reference.md`
