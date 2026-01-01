"""CLI entry point for microstructure-metrics."""

import click


@click.group()
@click.version_option()
def main() -> None:
    """Microstructure Metrics - 人間知覚指向DAC/AMPマイクロダイナミクス評価系."""
    pass


if __name__ == "__main__":
    main()
