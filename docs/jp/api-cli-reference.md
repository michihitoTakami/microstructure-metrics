# API / CLI リファレンス

本書は CLI サブコマンドの主要オプションと典型的な使い方をまとめる。詳細は `uv run microstructure-metrics <cmd> --help` を参照。

## 前提
- WAV はステレオ(2ch)推奨・48 kHz / 24-bit（または 32f）。参照(ref)とDUTのサンプルレートを揃える。
- パイロット入りテスト信号の仕様は `docs/signal-specifications.md` を参照。

## generate — テスト信号生成
```
uv run microstructure-metrics generate <signal_type> [options]
```
- `signal_type`: thd / notched-noise / pink-noise / modulated / tfs-tones / tone-burst / am-attack / click
- 主なオプション:
  - `--sample-rate,-sr` (int, default 48000)
  - `--bit-depth,-bd` ("24bit" or "32f")
  - `--duration,-d` (sec, default 10.0) テスト本体長
  - 出力チャンネル: **常にステレオ(2ch)**（生成信号をL/Rに複製）
  - パイロット/無音: `--pilot-freq` `--pilot-duration` `--silence-duration`
  - THD: `--freq` `--level-dbfs`
  - MPS: `--carrier` `--am-freq` `--am-depth` `--fm-dev` `--fm-freq`
  - TFS: `--min-freq` `--tone-count` `--tone-step`
  - tone-burst: `--burst-freq` `--burst-cycles` `--burst-level-dbfs` `--burst-fade-cycles`
  - am-attack: `--carrier` `--attack-ms` `--release-ms` `--gate-period-ms`
  - click: `--click-level-dbfs` `--click-band-limit-hz`
  - 出力: `--output,-o` (WAV path), `--with-metadata` (JSONも出力)
- 例: ノッチノイズ + メタデータ
```
uv run microstructure-metrics generate notched-noise --with-metadata
```

- 例: **Qスイープ**（複数Qの一括生成。`--output` はディレクトリを指定）
```
uv run microstructure-metrics generate notched-noise --q 2 --q 8.6 --q 30 --q 80 \
  --output out/notch_q_sweep --with-metadata
```

- 例: **マルチノッチ + ノッチ強化**（複数中心周波数 + カスケード段数）
```
uv run microstructure-metrics generate notched-noise --centers "3000,5000,7000,9000" \
  --q 8.6 --notch-cascade-stages 2 --with-metadata
```

## align — パイロット整列
```
uv run microstructure-metrics align ref.wav dut.wav [options]
```
- 主なオプション: `--pilot-freq` `--threshold` `--band-width-hz` `--min-duration-ms`
  `--pilot-duration-ms` `--margin-ms` `--max-lag-ms` `--no-refine-delay`
  出力先: `--output-ref` `--output-dut` `--metadata`
- 出力: 整列済みWAV（`.aligned_ref.wav`, `.aligned_dut.wav`）とメタJSON。

## drift — クロックドリフト推定
```
uv run microstructure-metrics drift ref.wav dut.wav [options]
```
- 主なオプション: `--pilot-freq` `--pilot-duration-ms` `--band-width-hz`
  `--threshold` `--json-output` `--strict`
- 出力: 推定 ppm と警告。`--json-output` でレポートJSON保存。

## report — 全指標レポート
```
uv run microstructure-metrics report ref.wav dut.wav [options]
```
- 主なオプション:
  - 入力処理: `--allow-resample` `--target-sample-rate` `--channels`
    - `--channels`: `stereo|mid|side|ch0|ch1` を選択。I/Oは常に2chへ正規化され、mid/sideは2chへ写像、ch0/ch1は指定chを複製して解析する。
  - アライメント: `--align/--no-align` `--pilot-freq` `--pilot-threshold`
    `--pilot-band-width-hz` `--pilot-duration-ms` `--min-duration-ms`
    `--margin-ms` `--max-lag-ms`
  - THD: `--fundamental-freq` `--expected-level-dbfs`
  - LFCR (bass): `--bass-bands`(例:20-80,80-200) `--bass-filter-order` `--bass-cycle-points`
    `--bass-envelope-threshold-db` `--bass-harmonic-max-order` `--bass-fundamental-range`
  - 出力: `--output-json` (default `metrics_report.json`), `--output-csv`, `--output-md`
  - 可視化: `--plot` (MPS差分ヒートマップ / TFS相関時系列を保存), `--plot-dir` (プロット出力先ディレクトリ)
  - TFS出力項目: `mean_correlation` / `percentile_05_correlation` / `correlation_variance` に加え、`frame_length_ms` `frame_hop_ms` `max_lag_ms` `envelope_threshold_db`
  - Binaural出力項目: `median_abs_delta_itd_ms` / `p95_abs_delta_itd_ms` / `itd_outlier_rate` / `median_abs_delta_ild_db` / `p95_abs_delta_ild_db` / `iacc_p05` など
  - LFCR出力項目: `cycle_shape_corr_mean` / `cycle_shape_corr_p05` / `harmonic_phase_coherence` / `envelope_diff_outlier_rate` / 帯域別 `band_metrics.*`
- 例: JSON と Markdown を保存
```
uv run microstructure-metrics report ref.aligned_ref.wav dut.aligned_dut.wav \
  --output-json report.json --output-md report.md
```

## 関連ドキュメント
- 測定手順: `docs/jp/measurement-setup.md`
- 信号仕様: `docs/jp/signal-specifications.md`
- 指標の読み解き: `docs/jp/metrics-interpretation.md`

## Python API エントリポイント
- アライメント/ドリフト: `microstructure_metrics.alignment.align_audio_pair`, `estimate_clock_drift`
- 指標: `microstructure_metrics.metrics.calculate_thd_n`, `calculate_mps_similarity`, `calculate_tfs_correlation`, `calculate_transient_metrics`
- 信号生成: `microstructure_metrics.signals.build_signal`
各APIは numpy array とサンプルレートを受け取り、結果dataclassを返す。
