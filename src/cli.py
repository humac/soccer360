"""CLI interface for Soccer360 pipeline."""

from __future__ import annotations

from pathlib import Path

import click

from .utils import load_config, setup_logging


@click.group()
@click.option(
    "--config", "-c",
    default="configs/pipeline.yaml",
    envvar="SOCCER360_CONFIG",
    help="Path to pipeline config YAML.",
)
@click.pass_context
def cli(ctx: click.Context, config: str):
    """Soccer360 -- Automated 360 soccer video processing."""
    ctx.ensure_object(dict)
    cfg = load_config(config)

    paths = cfg.setdefault("paths", {})
    paths.setdefault("ingest", "/tank/ingest")
    paths.setdefault("scratch", "/scratch/work")
    paths.setdefault("processed", "/tank/processed")
    paths.setdefault("highlights", "/tank/highlights")
    paths.setdefault("models", "/tank/models")
    paths.setdefault("labeling", "/tank/labeling")
    paths.setdefault("archive_raw", "/tank/archive_raw")
    paths.setdefault("logs", "/tank/logs")

    Path("/scratch/work").mkdir(parents=True, exist_ok=True)
    log_dir = Path("/tank/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    paths["logs"] = str(log_dir)

    ctx.obj["config"] = cfg
    log_cfg = cfg.get("logging", {})
    log_name = Path(log_cfg.get("file") or "soccer360.log").name
    log_file = str(log_dir / log_name)
    setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_file)


@cli.command()
@click.pass_context
def watch(ctx: click.Context):
    """Start the watch-folder daemon on the ingest directory."""
    from .watcher import WatcherDaemon

    daemon = WatcherDaemon(ctx.obj["config"])
    daemon.run()


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--no-cleanup", is_flag=True, help="Keep scratch files after processing.")
@click.pass_context
def process(ctx: click.Context, file_path: str, no_cleanup: bool):
    """Process a single 360 video file."""
    from .pipeline import Pipeline

    pipe = Pipeline(ctx.obj["config"])
    pipe.run(file_path, cleanup=not no_cleanup)


@cli.command()
@click.option("--epochs", default=50, help="Training epochs.")
@click.option("--data", required=True, type=click.Path(exists=True), help="YOLO dataset YAML path.")
@click.pass_context
def train(ctx: click.Context, epochs: int, data: str):
    """Retrain / fine-tune the YOLO ball detection model."""
    from .trainer import Trainer

    trainer = Trainer(ctx.obj["config"])
    trainer.run(data=data, epochs=epochs)


@cli.command("export-hard-frames")
@click.argument("video_path", type=click.Path(exists=True))
@click.argument("detections_json", type=click.Path(exists=True))
@click.option("--threshold", default=0.3, help="Confidence threshold for hard frames.")
@click.option("--output-dir", default=None, help="Output directory (defaults to config labeling path).")
@click.pass_context
def export_hard_frames(
    ctx: click.Context, video_path: str, detections_json: str,
    threshold: float, output_dir: str | None,
):
    """Export low-confidence frames for labeling."""
    from .trainer import Trainer

    cfg = ctx.obj["config"]
    out = output_dir or cfg["paths"]["labeling"]
    trainer = Trainer(cfg)
    trainer.export_hard_frames(
        video_path=video_path,
        detections_path=Path(detections_json),
        threshold=threshold,
        output_dir=Path(out),
    )
