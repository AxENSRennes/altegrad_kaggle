#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: Alignment Training

Trains the projector to align GNN graph embeddings with LLM text embeddings.
- GNN is frozen
- LLM is frozen
- Only the projector is trained

Loss: Cosine distance between projected graph embedding and mean-pooled LLM text embedding.
"""

from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from model_wrapper import MolCaptionModel
from dataset_caption import prepare_alignment_dataloaders
from utils import save_checkpoint, WandBLogger, get_grad_norm
from report import (
    print_progress_header,
    print_training_report,
    print_epoch_summary,
    print_best_model_saved,
)


# MSE Loss for scale-sensitive alignment
alignment_criterion = nn.MSELoss()


def train_stage1(
    model: MolCaptionModel,
    config: Config,
    logger: Optional[WandBLogger] = None,
) -> Tuple[Dict[str, float], int]:
    """
    Train Stage 1: Alignment of graph embeddings to LLM text space.

    Args:
        model: MolCaptionModel with frozen GNN and LLM
        config: Configuration object
        logger: Optional W&B logger

    Returns:
        Dictionary with final metrics
    """
    device = model.device
    print_progress_header("Stage 1: Alignment Training", config)

    # Prepare data
    train_loader, val_loader = prepare_alignment_dataloaders(config)

    # Freeze everything except projector
    for param in model.gnn.parameters():
        param.requires_grad = False
    for param in model.llm.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = True

    # Optimizer for projector only
    optimizer = torch.optim.AdamW(
        model.projector.parameters(),
        lr=config.stage1_lr,
        weight_decay=1e-4,
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * config.stage1_epochs
    warmup_steps = config.stage1_warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    use_amp = config.use_amp and device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    # Training state
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(config.stage1_epochs):
        model.projector.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Stage 1 Epoch {epoch + 1}/{config.stage1_epochs}")

        for batch in pbar:
            graphs, descriptions = batch
            graphs = graphs.to(device)

            # Skip batches with empty descriptions
            if not any(descriptions):
                continue

            optimizer.zero_grad()

            with torch.amp.autocast(device, enabled=use_amp):
                # Get graph embeddings
                graph_emb = model.encode_graphs(graphs)  # [B, 768]

                # Project to LLM space
                projected = model.project_to_llm_space(graph_emb)  # [B, 1, llm_hidden]

                # Get target text embeddings from LLM
                target_emb = model.get_batch_text_embeddings(descriptions)  # [B, llm_hidden]

                # Compute alignment loss (MSE)
                if projected.dim() == 3:
                    projected = projected.squeeze(1)
                loss = alignment_criterion(projected, target_emb)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.projector.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # Log to W&B
            if logger and global_step % config.log_every_n_steps == 0:
                logger.log({
                    "stage1/loss": loss.item(),
                    "stage1/lr": scheduler.get_last_lr()[0],
                    "stage1/grad_norm": get_grad_norm(model.projector),
                }, step=global_step)

        # Epoch metrics
        avg_train_loss = epoch_loss / max(num_batches, 1)

        # Validation
        val_loss, val_cos_sim = evaluate_alignment(model, val_loader, device, config)

        # Print epoch summary
        print_epoch_summary(
            epoch + 1,
            config.stage1_epochs,
            avg_train_loss,
            {"val_loss": val_loss, "val_cos_sim": val_cos_sim}
        )

        # Log to W&B (epoch metrics)
        if logger:
            logger.log({
                "stage1/epoch": epoch + 1,
                "stage1/train_loss_epoch": avg_train_loss,
                "stage1/val_loss": val_loss,
                "stage1/val_cos_sim": val_cos_sim,
            }, step=global_step)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                config.stage1_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch=epoch + 1,
                metrics={"val_loss": val_loss, "val_cos_sim": val_cos_sim},
                config=config,
            )
            print_best_model_saved(config.stage1_checkpoint_path, "val_loss", val_loss)

    # Final report
    final_metrics = {
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "val_cos_sim": val_cos_sim,
        "best_val_loss": best_val_loss,
    }

    print_training_report(
        "Stage 1: Alignment",
        final_metrics,
        config,
        epoch=config.stage1_epochs,
        total_epochs=config.stage1_epochs,
    )

    return final_metrics, global_step


@torch.no_grad()
def evaluate_alignment(
    model: MolCaptionModel,
    val_loader: DataLoader,
    device: str,
    config: Config,
) -> tuple:
    """
    Evaluate alignment on validation set.

    Args:
        model: MolCaptionModel
        val_loader: Validation dataloader
        device: Device string
        config: Configuration

    Returns:
        Tuple of (average_loss, average_cosine_similarity)
    """
    model.projector.eval()
    use_amp = config.use_amp and device == "cuda"

    total_loss = 0.0
    total_cos_sim = 0.0
    num_batches = 0

    for batch in val_loader:
        graphs, descriptions = batch
        graphs = graphs.to(device)

        if not any(descriptions):
            continue

        with torch.amp.autocast(device, enabled=use_amp):
            # Get embeddings
            graph_emb = model.encode_graphs(graphs)
            projected = model.project_to_llm_space(graph_emb)
            target_emb = model.get_batch_text_embeddings(descriptions)

            # Compute loss (MSE)
            if projected.dim() == 3:
                projected_for_loss = projected.squeeze(1)
            else:
                projected_for_loss = projected
            loss = alignment_criterion(projected_for_loss, target_emb)

            # Compute cosine similarity for metrics
            if projected.dim() == 3:
                projected = projected.squeeze(1)
            cos_sim = F.cosine_similarity(
                F.normalize(projected, dim=-1),
                F.normalize(target_emb, dim=-1),
                dim=-1,
            ).mean()

        total_loss += loss.item()
        total_cos_sim += cos_sim.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_cos_sim = total_cos_sim / max(num_batches, 1)

    return avg_loss, avg_cos_sim


def main():
    """Main function for standalone Stage 1 training."""
    from config import get_config
    from model_wrapper import create_model

    # Get config
    config = get_config(mode="quick")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    model = create_model(config, device=device)
    model.print_trainable_parameters()

    # Setup logger
    logger = WandBLogger(enabled=config.use_wandb)
    if config.use_wandb:
        logger.init(config.wandb_project, config, tags=[config.experiment_mode, "stage1"])

    # Train
    metrics = train_stage1(model, config, logger)

    # Cleanup
    if logger:
        logger.finish()

    return metrics


if __name__ == "__main__":
    main()
