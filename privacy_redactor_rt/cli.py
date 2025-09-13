"""CLI interface for offline processing."""

import typer
from pathlib import Path
from .config import Config


app = typer.Typer()


@app.command()
def redact_video(
    input_file: Path = typer.Argument(..., help="Input video file"),
    output_file: Path = typer.Argument(..., help="Output video file"),
    config_file: Path = typer.Option("config.yaml", help="Configuration file"),
):
    """Redact sensitive information from video file."""
    # Placeholder for CLI processing
    typer.echo(f"Processing {input_file} -> {output_file}")


if __name__ == "__main__":
    app()