# Microstructure Metrics (日本語)

人間知覚指向のDAC/AMPマイクロダイナミクス評価スイート。
パイロットトーンを用いたアライメント・ドリフト推定・信号生成から、THD+N / MPS / TFS / Transient を一括算出するオフラインパイプラインを提供します。

## なぜ必要か

近年のDAC/AMPは SINAD 120 dB 以上が一般的で、定常波形指標だけでは「音の微細さ」「空間表現」など主観差を説明できません。以下の仮説に基づき、微小構造の劣化を測る指標群を実装しています。

- NFBが強い系では過渡応答が平滑化され微細構造が失われる
- ノッチ/ピークなどスペクトル特徴が動的IMDで埋まり情報欠損が生じる
- 高域でTFS（位相微細構造）の相関が崩れ定位が劣化する

## 実装済み指標

| Metric | 目的 |
| --- | --- |
| THD+N | 基本ゲイン/歪みの健全性確認 |
| Modulation Power Spectrum (MPS) | 変調テクスチャの保持度（相関/距離） |
| Temporal Fine Structure (TFS) | 高域微細位相の相関・群遅延ばらつき |
| Transient / エッジ丸まり | インパルス/クリックの立ち上がり鋭さ・スメア |

## 動作環境

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) パッケージマネージャ

## インストール

```bash
git clone https://github.com/michihitoTakami/microstructure-metrics.git
cd microstructure-metrics
uv sync
```

## クイックスタート

```bash
# バージョン確認・ヘルプ
uv run microstructure-metrics --version
uv run microstructure-metrics --help

# テスト信号生成（例: ノッチノイズ + メタデータ出力）
uv run microstructure-metrics generate notched-noise --with-metadata

# 録音した ref/dut WAV をパイロットで整列
uv run microstructure-metrics align ref.wav dut.wav

# ドリフト警告を表示（JSONも出力可能）
uv run microstructure-metrics drift ref.aligned_ref.wav dut.aligned_dut.wav

# 全指標を計算し JSON/CSV/Markdown でレポート
uv run microstructure-metrics report ref.aligned_ref.wav dut.aligned_dut.wav \
  --output-json report.json --output-md report.md
```

## ドキュメント

- 英語ユーザーガイド: `docs/en/user-guide.md`
- 測定セットアップ: `docs/en/measurement-setup.md` (EN) / `docs/jp/measurement-setup.md` (JP)
- 信号仕様: `docs/en/signal-specifications.md` (EN) / `docs/jp/signal-specifications.md` (JP)
- 指標の読み解き: `docs/en/metrics-interpretation.md` (EN) / `docs/jp/metrics-interpretation.md` (JP)
- CLI/APIリファレンス: `docs/jp/api-cli-reference.md` (JP)
- アライメント詳細: `docs/jp/alignment.md` (JP)

## 開発向けセットアップ

```bash
git clone https://github.com/michihitoTakami/microstructure-metrics.git
cd microstructure-metrics
uv sync --extra dev
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

### チェックコマンド

```bash
uv run ruff check src/
uv run ruff format src/
uv run mypy src/
uv run pytest
uv run pre-commit run --all-files
```

## プロジェクト構成

```
microstructure-metrics/
├── docs/                          # 仕様・ガイド
├── src/microstructure_metrics/    # パッケージ本体
├── tests/                         # ユニット/回帰テスト
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md
```

## ライセンス

MIT License
