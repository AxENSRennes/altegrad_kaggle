#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 Training with TRL SFTTrainer for TPU/GPU/CPU compatibility.

Uses HuggingFace TRL and Accelerate for distributed training across
different hardware backends (TPU v5e-8, GPU, CPU).
"""

import os
# CRITICAL: This MUST be set before any other imports
os.environ["NPY_DISABLE_ARRAY_API"] = "1"

import math
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback

from config import Config
from model_wrapper import MolCaptionModel
from dataset_caption import MolCaptionDatasetTRL, prepare_trl_dataloaders
from data_collator import MolCaptionDataCollatorSimple
from utils import (
    load_checkpoint,
    save_checkpoint,
    WandBLogger,
    get_grad_norm,
)
from report import (
    print_progress_header,
    print_training_report,
    print_epoch_summary,
    print_best_model_saved,
)
from metrics import compute_metrics


class MolCaptionTrainer(Trainer):
    """
    Custom Trainer for molecular captioning with dual learning rates.

    Extends HuggingFace Trainer to support:
    - Separate learning rates for projector and LoRA parameters
    - Custom forward pass with graph injection
    - Compatibility with TPU/GPU/CPU via Accelerate
    """

    def __init__(
        self,
        *args,
        projector_lr: float = 1e-4,
        lora_lr: float = 1e-5,
        mol_model: Optional[MolCaptionModel] = None,
        **kwargs
    ):
        """
        Args:
            projector_lr: Learning rate for projector parameters
            lora_lr: Learning rate for LoRA parameters
            mol_model: The MolCaptionModel instance (for accessing components)
        """
        self.projector_lr = projector_lr
        self.lora_lr = lora_lr
        self.mol_model = mol_model
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """Create optimizer with separate parameter groups for projector and LoRA."""
        if self.optimizer is not None:
            return self.optimizer

        # Get parameter groups
        projector_params = list(self.mol_model.projector.parameters())
        lora_params = [
            p for n, p in self.mol_model.llm.named_parameters()
            if p.requires_grad and 'lora' in n.lower()
        ]

        optimizer_grouped_parameters = [
            {"params": projector_params, "lr": self.projector_lr, "weight_decay": self.args.weight_decay},
            {"params": lora_params, "lr": self.lora_lr, "weight_decay": self.args.weight_decay},
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation with graph embedding injection.

        The model expects:
        - graphs: Batched PyG graphs
        - input_ids: Token IDs
        - attention_mask: Attention mask
        - labels: Target labels
        """
        graphs = inputs.get("graphs")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")

        # Forward pass through MolCaptionModel
        outputs = self.mol_model(
            graphs=graphs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs["loss"]

        if return_outputs:
            return loss, outputs
        return loss


class GradientLoggingCallback(TrainerCallback):
    """Callback to log gradient norms during training."""

    def __init__(self, mol_model: MolCaptionModel, logger: Optional[WandBLogger] = None):
        self.mol_model = mol_model
        self.logger = logger

    def on_step_end(self, args, state, control, **kwargs):
        if self.logger and state.global_step % args.logging_steps == 0:
            grad_norm_proj = get_grad_norm(self.mol_model.projector)
            grad_norm_lora = get_grad_norm(self.mol_model.llm)
            self.logger.log({
                "stage2/grad_norm_proj": grad_norm_proj,
                "stage2/grad_norm_lora": grad_norm_lora,
            }, step=state.global_step)


def train_stage2(
    model: MolCaptionModel,
    config: Config,
    logger: Optional[WandBLogger] = None,
    load_stage1: bool = True,
    start_step: int = 0,
) -> Dict[str, float]:
    """
    Train Stage 2: Supervised Fine-Tuning on caption generation.

    Uses TRL-compatible training loop with Accelerate for hardware abstraction.

    Args:
        model: MolCaptionModel
        config: Configuration object
        logger: Optional W&B logger
        load_stage1: Whether to load Stage 1 checkpoint
        start_step: Starting global step (for logging continuity)

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

    # Prepare data with TRL-compatible format
    train_dataset, val_dataset = prepare_trl_dataloaders(config, model.tokenizer)

    # Create collator
    collator = MolCaptionDataCollatorSimple(
        tokenizer=model.tokenizer,
        max_length=config.max_seq_length,
    )

    # Freeze GNN, unfreeze projector and LoRA
    for param in model.gnn.parameters():
        param.requires_grad = False

    for param in model.projector.parameters():
        param.requires_grad = True

    # Re-enable LoRA parameters
    for name, param in model.llm.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    model.llm.print_trainable_parameters()

    # Calculate training steps
    total_steps = (len(train_dataset) // config.stage2_batch_size) * config.stage2_epochs
    warmup_steps = config.stage2_warmup_steps

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.stage2_epochs,
        per_device_train_batch_size=config.stage2_batch_size,
        per_device_eval_batch_size=config.stage2_batch_size,
        gradient_accumulation_steps=config.stage2_grad_accum,
        learning_rate=config.stage2_lr_lora,  # Base LR (LoRA), projector uses custom
        weight_decay=1e-4,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=config.log_every_n_steps,
        eval_strategy="epoch" if config.compute_train_val else "no",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=config.compute_train_val,
        metric_for_best_model="eval_loss" if config.compute_train_val else None,
        greater_is_better=False,
        bf16=config.use_amp and config.hardware_mode != "cpu",
        fp16=False,  # Prefer bf16 for TPU/GPU
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # Keep graph data
        max_grad_norm=config.grad_clip_norm,
        report_to="wandb" if config.use_wandb else "none",
        run_name=f"stage2_{config.experiment_mode}",
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,
    )

    # Create trainer
    trainer = MolCaptionTrainer(
        model=model.llm,  # Pass LLM for Trainer's model handling
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if config.compute_train_val else None,
        data_collator=collator,
        projector_lr=config.stage2_lr_proj,
        lora_lr=config.stage2_lr_lora,
        mol_model=model,
    )

    # Add gradient logging callback
    if logger:
        trainer.add_callback(GradientLoggingCallback(model, logger))

    # Train
    print(f"\nStarting training for {config.stage2_epochs} epochs...")
    train_result = trainer.train()

    # Save final model
    final_path = config.stage2_checkpoint_path.replace(".pt", "_final.pt")
    save_checkpoint(
        final_path,
        model,
        trainer.optimizer,
        trainer.lr_scheduler,
        epoch=config.stage2_epochs,
        metrics={"train_loss": train_result.training_loss},
        config=config,
    )
    print(f"Final model saved to {final_path}")

    # Evaluate on validation set
    if config.compute_train_val:
        print("\nRunning final evaluation...")
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.stage2_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=config.num_workers,
        )
        val_metrics, samples = evaluate_generation(model, val_loader, device, config, max_samples=200)
    else:
        val_metrics = {"loss": float("nan"), "bleu4": 0.0, "meteor": 0.0}
        samples = []

    # Final report
    final_metrics = {
        "train_loss": train_result.training_loss,
        "val_loss": val_metrics.get("loss", float("nan")),
        "bleu4": val_metrics.get("bleu4", 0.0),
        "meteor": val_metrics.get("meteor", 0.0),
    }

    print_training_report(
        "Stage 2: SFT",
        final_metrics,
        config,
        samples=samples[:3] if samples else None,
        epoch=config.stage2_epochs,
        total_epochs=config.stage2_epochs,
    )

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
    all_samples = []
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
            total_loss += outputs["loss"].detach().float()
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

            for p, r, s in zip(predictions, descriptions, smiles):
                if len(all_samples) <= max_samples:
                    all_samples.append((p, r, s))

        num_samples += len(descriptions)

    # Compute metrics
    avg_loss = (total_loss / max(num_batches, 1)).item()
    metrics = {"loss": avg_loss}

    if config.compute_bleu_meteor and all_predictions and all_references:
        text_metrics = compute_metrics(all_predictions, all_references)
        metrics.update(text_metrics)
    else:
        metrics["bleu4"] = 0.0
        metrics["meteor"] = 0.0
        metrics["token_f1"] = 0.0

    samples = all_samples[:10]
    return metrics, samples


# Legacy training loop (kept for backward compatibility with non-TRL mode)
def train_stage2_legacy(
    model: MolCaptionModel,
    config: Config,
    logger: Optional[WandBLogger] = None,
    load_stage1: bool = True,
    start_step: int = 0,
) -> Dict[str, float]:
    """
    Legacy training loop without TRL.

    This is the original training implementation, kept for backward compatibility
    and for cases where TRL/Accelerate is not available.
    """
    from dataset_caption import prepare_dataloaders

    device = model.device
    print_progress_header("Stage 2: SFT Training (Legacy)", config)

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
            # Use detach to avoid sync barrier
            accum_loss = accum_loss + loss.detach() * config.stage2_grad_accum
            accum_steps += 1

            # Gradient accumulation
            if (batch_idx + 1) % config.stage2_grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.projector.parameters()) + lora_params,
                    config.grad_clip_norm
                )

                grad_norm_proj = get_grad_norm(model.projector)
                grad_norm_lora = get_grad_norm(model.llm)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Log to W&B
                if logger:
                    avg_accum_loss = (accum_loss / accum_steps).item()
                    logger.log({
                        "stage2/loss": avg_accum_loss,
                        "stage2/lr_proj": optimizer.param_groups[0]["lr"],
                        "stage2/lr_lora": optimizer.param_groups[1]["lr"],
                        "stage2/grad_norm_proj": grad_norm_proj,
                        "stage2/grad_norm_lora": grad_norm_lora,
                    }, step=global_step)

                accum_loss = 0.0
                accum_steps = 0

            # Update metrics - use detach to avoid sync
            epoch_loss = epoch_loss + outputs["loss"].detach()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{outputs['loss'].detach().item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # Epoch metrics - only call .item() at epoch end
        avg_train_loss = (epoch_loss / max(num_batches, 1)).item()

        # Full validation
        if config.compute_train_val:
            val_metrics, samples = evaluate_generation(
                model, val_loader, device, config, max_samples=200
            )
        else:
            val_metrics = {"loss": float("nan"), "bleu4": 0.0, "meteor": 0.0, "token_f1": 0.0}
            samples = []

        print_epoch_summary(epoch + 1, config.stage2_epochs, avg_train_loss, val_metrics)

        # Save checkpoints
        if config.compute_train_val and not math.isnan(val_metrics["loss"]):
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                loss_path = config.stage2_checkpoint_path.replace(".pt", "_best_loss.pt")
                save_checkpoint(loss_path, model, optimizer, scheduler,
                              epoch=epoch + 1, metrics=val_metrics, config=config)
                print(f"  [New Best Loss] {best_val_loss:.4f} - Saved to {loss_path}")

        # Save epoch checkpoint
        epoch_path = config.stage2_checkpoint_path.replace(".pt", f"_ep{epoch + 1}.pt")
        save_checkpoint(epoch_path, model, optimizer, scheduler,
                      epoch=epoch + 1, metrics=val_metrics, config=config)

    # Final report
    final_metrics = {
        "train_loss": avg_train_loss,
        "val_loss": val_metrics["loss"],
        "bleu4": val_metrics["bleu4"],
        "meteor": val_metrics["meteor"],
        "best_bleu4": best_bleu4,
    }

    print_training_report("Stage 2: SFT", final_metrics, config,
                         samples=samples[:3] if samples else None,
                         epoch=config.stage2_epochs, total_epochs=config.stage2_epochs)

    # Final save
    final_path = config.stage2_checkpoint_path.replace(".pt", "_final.pt")
    save_checkpoint(final_path, model, optimizer, scheduler,
                  epoch=config.stage2_epochs, metrics=final_metrics, config=config)
    print(f"Final model saved to {final_path}")

    return final_metrics


def main():
    """Main function for standalone Stage 2 training."""
    from config import get_config
    from model_wrapper import create_model

    config = get_config(mode="quick")
    config.detect_hardware()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Hardware mode: {config.hardware_mode}")

    model = create_model(config, device=device)
    model.print_trainable_parameters()

    logger = WandBLogger(enabled=config.use_wandb)
    if config.use_wandb:
        logger.init(config.wandb_project, config, tags=[config.experiment_mode, "stage2"])

    # Use TRL-based training by default
    metrics = train_stage2(model, config, logger)

    if logger:
        logger.finish()

    return metrics


if __name__ == "__main__":
    main()
