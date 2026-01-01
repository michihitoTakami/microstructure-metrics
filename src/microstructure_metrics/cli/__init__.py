"""CLI entry point for microstructure-metrics."""

from __future__ import annotations

import click

from microstructure_metrics import __version__
from microstructure_metrics.cli.generate import generate


@click.group()
@click.version_option(__version__)
def main() -> None:
    """Microstructure Metrics - 人間知覚指向DAC/AMPマイクロダイナミクス評価系."""
    # サブコマンドは `main.add_command` で登録する
    return None


# サブコマンド登録
main.add_command(generate)


if __name__ == "__main__":
    main()
