#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pretty training reports using the rich library.

Provides formatted output for training progress, metrics, and sample outputs.
Falls back to simple printing if rich is not available.
"""

from typing import Dict, List, Tuple, Optional, Any


def _get_console():
    """Get rich console or None if not available."""
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


def print_training_report(
    stage: str,
    metrics: Dict[str, float],
    config: Any = None,
    samples: Optional[List[Tuple[str, str]]] = None,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
):
    """
    Print a formatted training report with metrics and sample outputs.

    Args:
        stage: Stage name (e.g., "Stage 1 Alignment", "Stage 2 SFT")
        metrics: Dictionary of metric names to values
        config: Optional config object
        samples: Optional list of (prediction, reference) tuples
        epoch: Current epoch number
        total_epochs: Total number of epochs
    """
    console = _get_console()

    if console is not None:
        _print_rich_report(console, stage, metrics, config, samples, epoch, total_epochs)
    else:
        _print_simple_report(stage, metrics, config, samples, epoch, total_epochs)


def _print_rich_report(
    console,
    stage: str,
    metrics: Dict[str, float],
    config: Any = None,
    samples: Optional[List[Tuple[str, str]]] = None,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
):
    """Print report using rich library."""
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    # Header
    epoch_info = f" (Epoch {epoch}/{total_epochs})" if epoch is not None else ""
    header_text = f"{stage} Training Complete{epoch_info}"
    console.print(Panel(f"[bold green]{header_text}[/bold green]", border_style="green"))

    # Metrics table
    table = Table(title="Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="magenta", justify="right", width=15)

    for name, value in metrics.items():
        if isinstance(value, float):
            formatted = f"{value:.4f}"
        else:
            formatted = str(value)
        table.add_row(name, formatted)

    console.print(table)

    # Sample outputs
    if samples:
        console.print("\n[bold]Sample Outputs:[/bold]")
        for i, (pred, ref) in enumerate(samples[:3]):
            console.print(f"\n[dim]#{i + 1}[/dim]")

            # Truncate long outputs
            pred_display = pred[:150] + "..." if len(pred) > 150 else pred
            ref_display = ref[:150] + "..." if len(ref) > 150 else ref

            console.print(f"  [green]Pred:[/green] {pred_display}")
            console.print(f"  [blue]Ref:[/blue]  {ref_display}")

    # W&B link hint
    if config is not None and getattr(config, "use_wandb", False):
        console.print("\n[dim]View full metrics on Weights & Biases[/dim]")

    console.print()


def _print_simple_report(
    stage: str,
    metrics: Dict[str, float],
    config: Any = None,
    samples: Optional[List[Tuple[str, str]]] = None,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
):
    """Print simple text report without rich."""
    epoch_info = f" (Epoch {epoch}/{total_epochs})" if epoch is not None else ""

    print("\n" + "=" * 50)
    print(f"{stage} Training Complete{epoch_info}")
    print("=" * 50)

    print("\nMetrics:")
    print("-" * 30)
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name:20s}: {value:.4f}")
        else:
            print(f"  {name:20s}: {value}")

    if samples:
        print("\nSample Outputs:")
        print("-" * 30)
        for i, (pred, ref) in enumerate(samples[:3]):
            pred_display = pred[:150] + "..." if len(pred) > 150 else pred
            ref_display = ref[:150] + "..." if len(ref) > 150 else ref
            print(f"\n#{i + 1}")
            print(f"  Pred: {pred_display}")
            print(f"  Ref:  {ref_display}")

    print("=" * 50 + "\n")


def print_progress_header(stage: str, config: Any = None):
    """Print a header at the start of training."""
    console = _get_console()

    if console is not None:
        from rich.panel import Panel

        mode = getattr(config, "experiment_mode", "unknown") if config else "unknown"
        console.print(Panel(
            f"[bold blue]Starting {stage}[/bold blue]\n"
            f"[dim]Mode: {mode}[/dim]",
            border_style="blue"
        ))
    else:
        print(f"\n{'=' * 50}")
        print(f"Starting {stage}")
        if config:
            print(f"Mode: {getattr(config, 'experiment_mode', 'unknown')}")
        print("=" * 50 + "\n")


def print_config_summary(config: Any):
    """Print a summary of the configuration."""
    console = _get_console()

    config_dict = {
        "experiment_mode": config.experiment_mode,
        "llm_name": config.llm_name,
        "stage1_epochs": config.stage1_epochs,
        "stage2_epochs": config.stage2_epochs,
        "train_subset": config.train_subset or "all",
        "lora_r": config.lora_r,
    }

    if console is not None:
        from rich.table import Table

        table = Table(title="Configuration", show_header=True, header_style="bold")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="yellow")

        for name, value in config_dict.items():
            table.add_row(name, str(value))

        console.print(table)
        console.print()
    else:
        print("\nConfiguration:")
        print("-" * 30)
        for name, value in config_dict.items():
            print(f"  {name}: {value}")
        print()


def create_progress_bar(total: int, desc: str = "Training"):
    """
    Create a progress bar for training iterations.

    Args:
        total: Total number of iterations
        desc: Description text

    Returns:
        tqdm progress bar or simple iterator
    """
    try:
        from tqdm import tqdm
        return tqdm(range(total), desc=desc, ncols=100)
    except ImportError:
        print(f"{desc}: {total} iterations")
        return range(total)


def update_progress_bar(pbar, metrics: Dict[str, float]):
    """
    Update progress bar with current metrics.

    Args:
        pbar: tqdm progress bar
        metrics: Dictionary of metric values
    """
    try:
        # Format metrics for display
        postfix = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()}
        pbar.set_postfix(postfix)
    except AttributeError:
        # Not a tqdm progress bar, skip
        pass


def print_epoch_summary(epoch: int, total_epochs: int, train_loss: float, val_metrics: Dict[str, float]):
    """
    Print a summary line for an epoch.

    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        train_loss: Training loss
        val_metrics: Validation metrics dictionary
    """
    console = _get_console()

    metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())

    if console is not None:
        console.print(
            f"[bold]Epoch {epoch}/{total_epochs}[/bold] | "
            f"train_loss=[cyan]{train_loss:.4f}[/cyan] | {metrics_str}"
        )
    else:
        print(f"Epoch {epoch}/{total_epochs} | train_loss={train_loss:.4f} | {metrics_str}")


def print_best_model_saved(path: str, metric_name: str, metric_value: float):
    """Print notification when best model is saved."""
    console = _get_console()

    if console is not None:
        console.print(
            f"[green]Best model saved![/green] {metric_name}={metric_value:.4f} -> {path}"
        )
    else:
        print(f"Best model saved! {metric_name}={metric_value:.4f} -> {path}")
