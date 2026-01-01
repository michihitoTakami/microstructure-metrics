# CLAUDE.md - Claude/Cursor Agent Instructions

このファイルはClaude（Anthropic）およびCursor AI向けの詳細な指示を記載しています。

## プロジェクトコンテキスト

### 目的

DAC/AMPの「音楽性」や「マイクロダイナミクス」を定量的に評価するための新しい指標群を実装するプロジェクトです。

従来のSINAD/THD+N指標では捉えられない以下の特性を評価します：
- 微細な過渡応答の保持
- スペクトル構造の保存
- 時間微細構造（TFS）の位相コヒーレンス

### 技術スタック

- **言語**: Python 3.13+
- **パッケージ管理**: uv
- **Lint/Format**: ruff
- **型チェック**: mypy（strict mode）
- **テスト**: pytest
- **CI/CD**: pre-commit hooks

## コーディングガイドライン

### 必須事項

1. **型ヒント必須**: すべての関数に型アノテーションを付与
2. **Docstring必須**: 公開関数/クラスにはGoogle形式のdocstring
3. **テスト必須**: 新機能には対応するテストを作成

### コードスタイル

```python
# 良い例
def calculate_thd_n(
    signal: np.ndarray,
    sample_rate: int,
    fundamental_freq: float,
) -> float:
    """Calculate THD+N for a given signal.

    Args:
        signal: Input audio signal (1D array).
        sample_rate: Sample rate in Hz.
        fundamental_freq: Fundamental frequency in Hz.

    Returns:
        THD+N value in dB.
    """
    ...
```

### インポート順序

```python
# 標準ライブラリ
from pathlib import Path
from typing import TYPE_CHECKING

# サードパーティ
import numpy as np
from scipy import signal

# ローカル
from microstructure_metrics.core import Analyzer
```

## 開発コマンド

```bash
# 環境セットアップ
uv sync --extra dev
uv run pre-commit install
uv run pre-commit install --hook-type pre-push

# 品質チェック
uv run ruff check src/ --fix
uv run ruff format src/
uv run mypy src/
uv run pytest -v

# 全チェック
uv run pre-commit run --all-files
```

## ファイル編集時の注意

1. 編集後は `uv run ruff format <file>` でフォーマット
2. 型エラーは `uv run mypy src/` で確認
3. 新規ファイル作成時は `__init__.py` にexportを追加

## 関連ドキュメント

- `docs/DAC_AMP評価指標 再実装指示.pdf` - 実装詳細
- `docs/ミクロダイナミクス評価の新指標.pdf` - 理論的背景

## Issue駆動開発

- 各IssueはEPIC #1の子Issueとして管理
- ブランチ名: `feature/issue-N-description`
- PRタイトル: `[S-XX] 機能名`
- コミットメッセージ: Conventional Commits形式
  - `feat:` 新機能
  - `fix:` バグ修正
  - `docs:` ドキュメント
  - `chore:` 雑務（依存更新等）
