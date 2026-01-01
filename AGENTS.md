# AGENTS.md - AI Agent Guidelines

このファイルはAIエージェント（Cursor、GitHub Copilot、Claude等）向けのガイドラインを記載しています。

## プロジェクト概要

- **プロジェクト名**: microstructure-metrics
- **目的**: 人間知覚指向DAC/AMPマイクロダイナミクス評価系の構築
- **言語**: Python 3.13+
- **パッケージマネージャ**: uv

## コーディング規約

### スタイル

- **Formatter/Linter**: ruff を使用
- **行の長さ**: 88文字
- **インポート順序**: isort互換（ruffで管理）
- **Docstring**: Google スタイル

### 型ヒント

- すべての関数に型ヒントを付与
- `mypy --strict` 相当の厳格な型チェック
- `py.typed` マーカーにより PEP 561 準拠

### ディレクトリ構造

```
src/microstructure_metrics/   # メインパッケージ
tests/                        # pytest テスト
docs/                         # ドキュメント
```

## 開発フロー

1. Issue駆動開発
2. feature/issue-N-* ブランチで作業
3. pre-commit（ruff）+ pre-push（mypy）でチェック
4. PR作成時に `Closes #N` で Issue をリンク

## 依存関係

### 主要依存

- `numpy`: 数値計算
- `scipy`: 信号処理
- `soundfile`: オーディオファイルI/O
- `gammatone`: ガンマトーンフィルタバンク
- `matplotlib`: 可視化
- `click`: CLI構築

### 開発依存

- `ruff`: Linter/Formatter
- `mypy`: 型チェック
- `pytest`: テスト
- `pytest-cov`: カバレッジ
- `pre-commit`: フック管理

## コマンドリファレンス

```bash
# 依存関係インストール
uv sync --extra dev

# Lintチェック
uv run ruff check src/

# フォーマット
uv run ruff format src/

# 型チェック
uv run mypy src/

# テスト
uv run pytest

# pre-commit全実行
uv run pre-commit run --all-files
```

## 注意事項

- オーディオファイル（.wav等）はリポジトリに含めない（.gitignoreで除外済み）
- テスト用フィクスチャは `tests/fixtures/` に配置
- 大きなファイル（1MB以上）のコミットは避ける
- 作業時はユーザー指示がない限り **worktree** を使用する
- worktree名は `mm-<ISSUE番号>` （例: `mm-13`）
- worktree作成時は必ず最新の `origin/main` から分岐する
- PR作成は GitHub CLI で実行すること（本タスクはAIエージェントが gh コマンドで対応）
