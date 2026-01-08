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

## 関連ドキュメント
- 測定手順: `docs/jp/measurement-setup.md`
- 指標の読み解き: `docs/jp/metrics-interpretation.md`
- CLI/API オプション: `docs/jp/api-cli-reference.md`
