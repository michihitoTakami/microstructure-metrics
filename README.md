# Microstructure Metrics

人間知覚指向DAC/AMPマイクロダイナミクス評価系

## 概要

SINADやTHD+Nといった従来の定常状態指標と並立し、人間の聴覚情報処理により適合した**マイクロダイナミクス評価指標群**をオフラインで計算するパイプラインを提供します。

### 背景

現代のDAC/AMPは120dB以上のSINADを容易に達成しますが、定常状態の測定だけでは「音楽性」や「マイクロダイナミクス」——微細な過渡応答、空間情報の再現性、音色のテクスチャ——における聴感上の差異を説明できません。

本プロジェクトでは、以下の仮説に基づく新指標群を実装します：
- 強力なNFB（負帰還）が過渡的な微細構造を平滑化している可能性
- スペクトルの凹凸（ノッチやピーク）の情報欠損が発生している可能性
- 時間微細構造（TFS）の位相コヒーレンスが劣化している可能性

### 実装する指標

| 指標 | 略称 | 評価対象 |
|------|------|----------|
| THD+N | THD+N | 従来指標（ベースライン・ゲイン適正確認） |
| ノッチ保存度 | NPS | 動的IMDによるスペクトル汚染 |
| スペクトルエントロピー差分 | ΔSE | 信号構造の平坦化・情報喪失 |
| 変調パワースペクトラム | MPS | テクスチャ・変調情報の保持 |
| 時間微細構造相関 | TFS | 高域位相コヒーレンス |

## 動作要件

- Python 3.13 以上
- [uv](https://github.com/astral-sh/uv) パッケージマネージャ

## インストール

### uv を使用（推奨）

```bash
# リポジトリをクローン
git clone https://github.com/michihitoTakami/microstructure-metrics.git
cd microstructure-metrics

# 依存関係をインストール
uv sync
```

### pip を使用

```bash
pip install .
```

## クイックスタート

```bash
# CLIのヘルプを表示
uv run microstructure-metrics --help

# バージョン確認
uv run microstructure-metrics --version
```

## 開発者向けセットアップ

### 開発環境のセットアップ

```bash
# リポジトリをクローン
git clone https://github.com/michihitoTakami/microstructure-metrics.git
cd microstructure-metrics

# 開発用依存を含めてインストール
uv sync --extra dev

# pre-commitフックをインストール
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

### コード品質チェック

```bash
# Linter/Formatter実行
uv run ruff check src/
uv run ruff format src/

# 型チェック
uv run mypy src/

# テスト実行
uv run pytest

# pre-commitを全ファイルに実行
uv run pre-commit run --all-files
```

## プロジェクト構造

```
microstructure-metrics/
├── docs/                          # ドキュメント
├── src/
│   └── microstructure_metrics/    # メインパッケージ
│       ├── __init__.py
│       ├── cli.py                 # CLIエントリポイント
│       └── py.typed               # PEP 561 型マーカー
├── tests/                         # テスト
├── .pre-commit-config.yaml        # pre-commit設定
├── pyproject.toml                 # プロジェクト設定
└── README.md
```

## 参考資料

- `docs/DAC_AMP評価指標 再実装指示.pdf`
- `docs/ミクロダイナミクス評価の新指標.pdf`

## ライセンス

MIT License
