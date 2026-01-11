# 信号仕様書

## 目的と範囲
- テスト信号の構造・フォーマット・命名規則を定義し、測定と指標計算を再現性高く行う。
- 対象指標: THD+N, MPS, TFS, Transient。入出力はオフラインWAV。

## 標準タイムライン
```
[無音 500 ms] [パイロット 100 ms] [テスト本体 5–10 s] [パイロット 100 ms] [無音 500 ms]
```
- パイロット: 1 kHz, -6 dBFS, 100 ms、5 ms コサインフェード。
- 無音: 前後とも 500 ms。トリミング禁止。
- テスト本体: 指標ごとに 5–10 s を目安。ピークは -1 dBFS 以下。

## 指標別テスト信号の推奨例
| Metric | Signal | 主パラメータ例 |
| --- | --- | --- |
| THD+N | Pure tone | 1 kHz, -3 dBFS, 長さ 5 s |
| MPS | AM/FM composite | 1 kHz carrier, AM 4 Hz depth 50%, FM 50 Hz@4 Hz, peak≈-6 dBFS, 8 s |
| TFS | High-band multitone | 4/6/8/10/12 kHz 等振幅, peak≈-6 dBFS, 8 s |
| Transient | Impulse / tone burst train | インパルスまたはバースト列、ピーク -1 dBFS 付近、0.3–1.0 s |
| LFCR | complex-bass (低域複雑ベース) | 30–220 Hz RMSE, 8本のFM/PMトーン, peak≈-2 dBFS |
| BCP | binaural-cues (ステレオ ITD/ILD) | 150–10 kHz pink noise with controlled ITD 0.35 ms / ILD 6 dB, peak≈-3 dBFS |
| RMI/MDI | ms-side-texture (Mid/Side で構造化したテクスチャ) | Mid: 80–3.2 kHz pink, Side: 4+ kHz modulated tones, peak≈-3 dBFS |

## 追加信号（Issue #70）
- 共通: 標準タイムラインを維持し、48 kHz / 24-bit / stereo。WAVピークは -1 dBFS 以下。
- `complex-bass`
  - 25–260 Hz までに制限したランダム位相マルチトーン（8本）に軽い FM/PM（±3 Hz@0.3–1.1 Hz, 0.25 rad@0.4–1.6 Hz）を付与。
  - バンド制限: highcut ≤260 Hz（`--highcut`指定時も 260 Hz で頭打ち）。目標ピーク -2 dBFS。
  - 主なメタキー: `bass_components_hz`, `bass_fm_dev_hz`, `bass_fm_rates_hz`, `bass_pm_depth_rad`, `bass_pm_rates_hz`, `band_lowcut_hz`, `band_highcut_hz`, `target_peak_dbfs`。
- `binaural-cues`（ステレオ差分保持）
  - 150–12 kHz 帯域のピンクノイズに既知の ITD/ILD を埋め込み。デフォルト: `itd_ms=0.35`（右ch遅延）、`ild_db=+6`（右が6 dB低い）。
  - 出力2chをそのまま保持（L/R複製は行わない）。目標ピーク -3 dBFS。
  - 主なメタキー: `itd_ms`, `ild_db`, `base_noise_lowcut_hz`, `base_noise_highcut_hz`, `target_peak_dbfs`。
- `ms-side-texture`（Mid/Side合成）
  - Mid: 80–3200 Hz ピンクノイズ（約 -10 dBFS RMS）。Side: 4 kHz 以上の高域マルチトーン（`--min-freq`/`--tone-count`/`--tone-step`で調整）に5 Hz AM（深さ0.35）を掛けてテクスチャ化。
  - L/R = 0.5*(Mid ± Side)。ピークを -3 dBFS 付近に正規化し、高域TFSをSide成分に集約。
  - 主なメタキー: `mid_band_lowcut_hz`, `mid_band_highcut_hz`, `side_tones_hz`, `side_mod_freq_hz`, `side_mod_depth`, `side_target_peak_dbfs`, `target_peak_dbfs`。
- 参考stem例:
  - `complex_bass_25to260hz_48000_24bit_v1.wav`
  - `binaural_cues_itd0.35ms_ild6db_48000_24bit_v1.wav`
  - `ms_side_texture_side4000hz_48000_24bit_v1.wav`

## ファイルフォーマット
- サンプルレート: 48 kHz 必須（高帯域検証のみ 96 kHz 可）。
- ビット深度: 24-bit PCM 推奨（32f 可）。
- チャンネル: ステレオ(2ch)を基本とする（モノラル入力は内部で2chへ複製される）。
- コンテナ: WAV (PCM or IEEE float)。プレーヤーのゲイン変更を誘発するメタは避ける。

## 命名規則
```
{signal_type}_{sample_rate}_{bit_depth}_{version}.wav

例:
thd_1khz_48000_24bit_v1.wav
notched_noise_8000hz_q8.6_48000_24bit_v1.wav
pink_noise_48000_24bit_v1.wav
```
- signal_type: thd, pink_noise, mps, tfs, transient など。
- bit_depth: `24bit` または `32f`。
- version: パラメータ変更時は vN または semver で更新。
- 追加パラメータは signal_type 直後に挿入可（例: `mps_1khz_am4hz50_fm50hz_48000_24bit_v1.wav`）。

## メタデータJSON（WAVと同じstem）
- 必須キー:
  - `signal_type` (str), `sample_rate` (int Hz), `bit_depth` (str), `channels` (int)
  - `duration_sec` (float), `pilot_tone_freq_hz` (int), `pilot_duration_ms` (int)
  - `pilot_level_dbfs` (float), `lead_silence_ms` / `tail_silence_ms` (int)
  - `version` (str), `created_at` (ISO8601, UTC)
- 追加キー（刺激ごとに差分あり）:
  - `complex-bass`: `bass_components_hz`, `bass_fm_dev_hz`, `bass_fm_rates_hz`, `bass_pm_depth_rad`, `bass_pm_rates_hz`, `band_lowcut_hz`, `band_highcut_hz`, `target_peak_dbfs`
  - `binaural-cues`: `itd_ms`, `ild_db`, `base_noise_lowcut_hz`, `base_noise_highcut_hz`, `target_peak_dbfs`
  - `ms-side-texture`: `mid_band_lowcut_hz`, `mid_band_highcut_hz`, `side_tones_hz`, `side_mod_freq_hz`, `side_mod_depth`, `side_target_peak_dbfs`, `target_peak_dbfs`
- 例:
```json
{
  "signal_type": "notched_noise",
  "sample_rate": 48000,
  "bit_depth": "24bit",
  "channels": 2,
  "duration_sec": 10.0,
  "pilot_tone_freq_hz": 1000,
  "pilot_duration_ms": 100,
  "pilot_level_dbfs": -6.0,
  "lead_silence_ms": 500,
  "tail_silence_ms": 500,
  "notch_center_hz": 8000,
  "notch_centers_hz": [8000],
  "notch_q": 8.6,
  "notch_cascade_stages": 1,
  "noise_color": "pink",
  "created_at": "2026-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

## 受け入れチェックリスト
- タイムライン構造と長さが規定どおり（無音500 ms、パイロット100 ms×2）。
- サンプルレート/ビット深度/チャンネルがファイル名・メタデータと一致。
- ピーク ≤ -1 dBFS、クリップなし。パイロットはおよそ -6 dBFS。
- パイロットに5 msフェードが入り、無音区間が残されている。
- メタデータJSONが存在し、内容が信号と整合。

## 新刺激 / ステレオ構成
- `complex-bass` は 30–220 Hz の低域マルチトーン（FM/PM 調整）を重ねて LFCR の周期波形差に敏感な低域テクスチャを作る。LR はタイムライン内で複製されるが、波形のわずかな群遅延や位相差が再現性評価に反映される。
- `binaural-cues` は 150 Hz 以上のピンクノイズを生成し、右チャンネルに ITD（デフォルト 0.35 ms）、ILD（デフォルト 6 dB）を付加することで BCP の ITD/ILD/IACC セクションをストレステストする。`report --channels stereo`/`mid`/`side` で解析ができ、ステレオ依存の可視化（例: ITD/ILD ヒートマップ）も有効になる。
- `ms-side-texture` では Mid チャンネルに 80–3.2 kHz のピンクノイズ、Side に 4 kHz 以上の変調トーン群 + 5 Hz の振幅モジュレーションを与えて、Side 周波数帯の高変調/位相構造を RMI/MDI で観察する。Mid/Side の分解は `report --channels mid|side` で明示的に行うこともできる。
- これらのステレオ刺激を使う場合、`align`/`drift`/`report` は 2ch のまま処理し、`report --channels` に `stereo` を選ぶことで BCP/MDI が利用可能になる（`ch0`/`ch1` で片チャンネルを選ぶと BCP は利用不可になる）。

## 関連ドキュメント
- 測定手順: `docs/jp/measurement-setup.md`
- 指標の読み解き: `docs/jp/metrics-interpretation.md`
- CLI/API オプション: `docs/jp/api-cli-reference.md`
