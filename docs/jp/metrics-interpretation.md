# 指標の読み解きガイド

## 目的
- 本ツールが出力する指標（THD+N, NPS, ΔSE, MPS, TFS）の意味と読み方を整理し、機器比較や異常検出に活用する。
- 出力は `report` サブコマンドの JSON/CSV/Markdown に含まれる値を前提とする。

## 共通の見方
- 参照(ref)と被測定(DUT)をパイロットで整列した後の比較値。整列失敗やドリフト大のときは信頼度が下がる。
- dB値は大きいほど差が大きい（劣化）場合と、相関・コヒーレンス系で1.0に近いほど良好な場合がある。指標ごとに確認する。

## 指標別のポイント

### THD+N
- 内容: 基本波に対する高調波+ノイズ比。`thd_n_db`, `thd_n_percent`, `sinad_db` を確認。
- 目安: `sinad_db` が高いほど良好。一般的なオーディオ計測では 90 dB 以上で良好、ハイエンドで 110 dB 以上を期待。
- 補足: `fundamental_level_dbfs` が想定とずれる場合、レベル合わせミスの可能性あり。

### Notch Preservation Score (NPS)
- 内容: リファレンスのノッチ深さが DUT でどれだけ埋まったか。`nps_db` (小さい/負が望ましい)、`nps_ratio`。
- 目安: 0 dB 以上でノッチが埋まり始めている。+3 dB 以上なら動的IMD/ノイズの影響が疑われる。
- `is_noise_limited` が true の場合、ノイズフロア不足で信頼度が下がる。

### PSDノッチ深さ (Notch PSD)
- 内容: Welch PSD でノッチ中心±帯域のパワーを測定し、周辺リングとの比で深さを算出。高Qノッチの底上がり検出用。
- 出力: `notch_psd.notch_fill_db`（正で埋まり量）、`ref_notch_depth_db`、`dut_notch_depth_db`、`notch_bandwidth_hz`、`ring_bandwidth_hz`。
- 目安: `notch_fill_db` が 0 に近ければ維持。+6 dB 以上でノッチ埋まり傾向。深さが負ならノッチ消失の可能性。
- 補足: NPSより狭帯域で高Q設定に追従。ノイズフロアが浅い場合はばらつくため、複数試行の平均が安定。

### Spectral Entropy ΔSE
- 内容: スペクトルエントロピー差分。DUTが平坦化すると ΔSE が正方向に増える。
- 目安: `delta_se_mean` が 0.02 以上で情報量劣化の兆候。0 に近いほどリファレンスに忠実。
- 局所変動を見るには `delta_se_max` や時系列 `delta_se_over_time` を確認。

### Modulation Power Spectrum (MPS)
- 内容: 変調テクスチャの類似度。`mps_correlation` (1に近いほど良好)、`mps_distance` (0に近いほど良好)。
- 目安: 相関 0.9 以上は良好、0.8 未満で変調成分の崩れを疑う。バンド別 `band_correlations` で帯域特定。
- 音楽的テクスチャが失われると相関が下がり距離が増える。

### Temporal Fine Structure (TFS)
- 内容: 高域微細位相の相関と群遅延のばらつき。`mean_correlation` (1に近いほど良好)、`phase_coherence`、`group_delay_std_ms`。
- 目安: 相関 0.85 以上で良好。`group_delay_std_ms` が大きい（>0.2 ms 程度）場合、帯域間の時間ずれが大きい可能性。
- 帯域別 `band_group_delays_ms` で特定の帯域の遅延ズレを確認。

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
