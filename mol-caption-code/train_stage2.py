#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
# CRITICAL: This MUST be set before any other imports
os.environ["NPY_DISABLE_ARRAY_API"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from model_wrapper import MolCaptionModel
from dataset_caption import prepare_dataloaders
from utils import (
    load_checkpoint,
    save_checkpoint,
    WandBLogger,
    graph_to_smiles,
    get_grad_norm,
)
from report import (
    print_progress_header,
    print_training_report,
    print_epoch_summary,
    print_best_model_saved,
)
from metrics import compute_metrics


def train_stage2(
    model: MolCaptionModel,
    config: Config,
    logger: Optional[WandBLogger] = None,
    load_stage1: bool = True,
    start_step: int = 0,
) -> Dict[str, float]:
    """
    Train Stage 2: Supervised Fine-Tuning on caption generation.

    Args:
        model: MolCaptionModel
        config: Configuration object
        logger: Optional W&B logger
        load_stage1: Whether to load Stage 1 checkpoint

    Returns:
        Dictionary with final metrics
    """
    device = model.device
    print_progress_header("Stage 2: SFT Training", config)

    # Load Stage 1 checkpoint if available
    if load_stage1:
        try:
            load_checkpoint(config.stage1_checkpoint_path, model, device=device)
            print(f"Loaded Stage 1 checkpoint from {config.stage1_checkpoint_path}")
        except FileNotFoundError:
            print("No Stage 1 checkpoint found, starting from scratch")

    # Prepare data
    train_loader, val_loader = prepare_dataloaders(config, model.tokenizer)

    # Freeze GNN, unfreeze projector and LoRA
    for param in model.gnn.parameters():
        param.requires_grad = False

    for param in model.projector.parameters():
        param.requires_grad = True

    # Re-enable LoRA parameters (may have been frozen in Stage 1)
    for name, param in model.llm.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    model.llm.print_trainable_parameters()

    # Separate parameter groups with different learning rates
    projector_params = list(model.projector.parameters())
    lora_params = [p for n, p in model.llm.named_parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": projector_params, "lr": config.stage2_lr_proj},
        {"params": lora_params, "lr": config.stage2_lr_lora},
    ], weight_decay=1e-4)

    # Learning rate scheduler
    total_steps = len(train_loader) * config.stage2_epochs // config.stage2_grad_accum
    warmup_steps = config.stage2_warmup_steps

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
    best_bleu4 = 0.0
    global_step = start_step
    accum_loss = 0.0
    accum_steps = 0

    for epoch in range(config.stage2_epochs):
        model.projector.train()
        model.llm.train()

        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Stage 2 Epoch {epoch + 1}/{config.stage2_epochs}")

        for batch_idx, batch in enumerate(pbar):
            graphs = batch["graphs"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            with torch.amp.autocast(device, enabled=use_amp):
                outputs = model(
                    graphs=graphs,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"] / config.stage2_grad_accum

            # Backward pass
            scaler.scale(loss).backward()
            accum_loss += loss.item() * config.stage2_grad_accum
            accum_steps += 1

            # Gradient accumulation
            if (batch_idx + 1) % config.stage2_grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.projector.parameters()) + lora_params,
                    config.grad_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Log to W&B (every step)
                if logger:
                    logger.log({
                        "stage2/loss": accum_loss / accum_steps,
                        "stage2/lr_proj": optimizer.param_groups[0]["lr"],
                        "stage2/lr_lora": optimizer.param_groups[1]["lr"],
                        "stage2/grad_norm_proj": get_grad_norm(model.projector),
                        "stage2/grad_norm_lora": get_grad_norm(model.llm),
                    }, step=global_step)

                accum_loss = 0.0
                accum_steps = 0

            # Update metrics
            epoch_loss += outputs["loss"].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{outputs['loss'].item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # Periodic evaluation (can be skipped via config)
            if (config.compute_train_val and
                not config.skip_eval_during_training and
                (batch_idx + 1) % (config.eval_every_n_steps * config.stage2_grad_accum) == 0):
                val_metrics, samples = evaluate_generation(
                    model, val_loader, device, config, max_samples=50
                )
                bleu_str = f"bleu4={val_metrics['bleu4']:.2f}, " if config.compute_bleu_meteor else ""
                meteor_str = f"meteor={val_metrics['meteor']:.2f}" if config.compute_bleu_meteor else ""
                print(f"\n  [Step {global_step}] val_loss={val_metrics['loss']:.4f}, {bleu_str}{meteor_str}")

                if logger:
                    log_dict = {"eval/loss": val_metrics["loss"]}
                    if config.compute_bleu_meteor:
                        log_dict["eval/bleu4"] = val_metrics["bleu4"]
                        log_dict["eval/meteor"] = val_metrics["meteor"]
                    logger.log(log_dict, step=global_step)

                    # Log sample table
                    if samples:
                        table_data = []
                        for pred, ref, sm in samples:
                            table_data.append([sm, ref, pred])
                        logger.log_table(
                            "eval/samples",
                            columns=["SMILES", "Reference", "Prediction"],
                            data=table_data,
                            step=global_step
                        )

                model.projector.train()
                model.llm.train()

        # Epoch metrics
        avg_train_loss = epoch_loss / max(num_batches, 1)

        # Full validation (if enabled)
        if config.compute_train_val:
            val_metrics, samples = evaluate_generation(
                model, val_loader, device, config, max_samples=200
            )
        else:
            val_metrics = {"loss": float("nan"), "bleu4": 0.0, "meteor": 0.0, "token_f1": 0.0}
            samples = []

        # Print epoch summary
        print_epoch_summary(
            epoch + 1,
            config.stage2_epochs,
            avg_train_loss,
            val_metrics
        )

        # Log to W&B (epoch metrics)
        if logger:
            log_dict = {
                "stage2/epoch": epoch + 1,
                "stage2/train_loss_epoch": avg_train_loss,
            }
            if config.compute_train_val:
                log_dict["stage2/val_loss"] = val_metrics["loss"]
                if config.compute_bleu_meteor:
                    log_dict["stage2/val_bleu4"] = val_metrics["bleu4"]
                    log_dict["stage2/val_meteor"] = val_metrics["meteor"]
            logger.log(log_dict, step=global_step)

            # Log sample table for epoch
            if samples:
                table_data = [[sm, ref, pred] for pred, ref, sm in samples]
                logger.log_table(
                    "stage2/samples_epoch",
                    columns=["SMILES", "Reference", "Prediction"],
                    data=table_data,
                    step=global_step
                )

        # Save best model (by BLEU-4) - only if computing metrics
        if config.compute_train_val and config.compute_bleu_meteor:
            if val_metrics["bleu4"] > best_bleu4:
                best_bleu4 = val_metrics["bleu4"]
                bleu_path = config.stage2_checkpoint_path.replace(".pt", "_best_bleu.pt")
                save_checkpoint(
                    bleu_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch + 1,
                    metrics=val_metrics,
                    config=config,
                )
                print_best_model_saved(bleu_path, "bleu4", best_bleu4)

        # Save best model (by loss) - only if computing validation
        if config.compute_train_val and not math.isnan(val_metrics["loss"]):
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                loss_path = config.stage2_checkpoint_path.replace(".pt", "_best_loss.pt")
                save_checkpoint(
                    loss_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch + 1,
                    metrics=val_metrics,
                    config=config,
                )
                print(f"  [New Best Loss] {best_val_loss:.4f} - Saved to {loss_path}")

        # Save epoch checkpoint
        epoch_path = config.stage2_checkpoint_path.replace(".pt", f"_ep{epoch + 1}.pt")
        save_checkpoint(
            epoch_path,
            model,
            optimizer,
            scheduler,
            epoch=epoch + 1,
            metrics=val_metrics,
            config=config,
        )
        print(f"  Epoch {epoch + 1} checkpoint saved to {epoch_path}")

        # Always save a 'latest' checkpoint for safety
        latest_path = config.stage2_checkpoint_path.replace(".pt", "_latest.pt")
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            epoch=epoch + 1,
            metrics=val_metrics,
            config=config,
        )

    # Final report
    final_metrics = {
        "train_loss": avg_train_loss,
        "val_loss": val_metrics["loss"],
        "bleu4": val_metrics["bleu4"],
        "meteor": val_metrics["meteor"],
        "best_bleu4": best_bleu4,
    }

    print_training_report(
        "Stage 2: SFT",
        final_metrics,
        config,
        samples=samples[:3] if samples else None,
        epoch=config.stage2_epochs,
        total_epochs=config.stage2_epochs,
    )

    # Final redundant save
    final_path = config.stage2_checkpoint_path.replace(".pt", "_final.pt")
    save_checkpoint(
        final_path,
        model,
        optimizer,
        scheduler,
        epoch=config.stage2_epochs,
        metrics=final_metrics,
        config=config,
    )
    print(f"Final model saved to {final_path}")

    return final_metrics


@torch.no_grad()
def evaluate_generation(
    model: MolCaptionModel,
    val_loader: DataLoader,
    device: str,
    config: Config,
    max_samples: int = 100,
) -> Tuple[Dict[str, float], List[Tuple[str, str, str]]]:
    """
    Evaluate caption generation on validation set.

    Args:
        model: MolCaptionModel
        val_loader: Validation dataloader
        device: Device string
        config: Configuration
        max_samples: Maximum number of samples to evaluate

    Returns:
        Tuple of (metrics_dict, list of (prediction, reference, smiles) tuples)
    """
    model.projector.eval()
    model.llm.eval()
    use_amp = config.use_amp and device == "cuda"

    all_predictions = []
    all_references = []
    all_samples = []  # List of (pred, ref, smiles)
    total_loss = 0.0
    num_batches = 0
    num_samples = 0

    for batch in val_loader:
        if num_samples >= max_samples:
            break

        graphs = batch["graphs"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        descriptions = batch["descriptions"]
        smiles = batch["smiles"]

        # Compute loss
        with torch.amp.autocast(device, enabled=use_amp):
            outputs = model(
                graphs=graphs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs["loss"].item()
            num_batches += 1

        # Generate captions (only if computing BLEU/METEOR)
        if config.compute_bleu_meteor:
            try:
                predictions = model.generate(
                    graphs=graphs,
                    smiles_list=smiles,
                    max_new_tokens=128,
                    num_beams=1,
                    do_sample=False,
                )
            except Exception as e:
                print(f"Generation error: {e}")
                predictions = ["" for _ in descriptions]

            all_predictions.extend(predictions)
            all_references.extend(descriptions)

            # Zip with SMILES for the table
            for p, r, s in zip(predictions, descriptions, smiles):
                if len(all_predictions) <= max_samples:
                    all_samples.append((p, r, s))

        num_samples += len(descriptions)

    # Compute metrics
    metrics = {
        "loss": total_loss / max(num_batches, 1),
    }

    # Compute BLEU/METEOR only if enabled
    if config.compute_bleu_meteor and all_predictions and all_references:
        text_metrics = compute_metrics(all_predictions, all_references)
        metrics.update(text_metrics)
    else:
        # Provide placeholder metrics when not computing
        metrics["bleu4"] = 0.0
        metrics["meteor"] = 0.0
        metrics["token_f1"] = 0.0

    # Sample outputs
    samples = all_samples[:10]

    return metrics, samples


def main():
    """Main function for standalone Stage 2 training."""
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
        logger.init(config.wandb_project, config, tags=[config.experiment_mode, "stage2"])

    # Train
    metrics = train_stage2(model, config, logger)

    # Cleanup
    if logger:
        logger.finish()

    return metrics


if __name__ == "__main__":
    main()
