#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MolCaptionModel: Combined model for molecular captioning.

Combines:
1. Frozen MolGNN encoder for graph embeddings
2. Trainable SolidBridgeProjector to map to LLM space
3. Qwen3-0.6B with 4-bit quantization and LoRA adapters

The model injects graph embeddings as "soft tokens" at the <|graph|> position
in the input sequence, enabling the LLM to generate molecule descriptions.
"""

from typing import Optional, List, Dict, Any

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from model_gnn import MolGNN
from model_projector import SolidBridgeProjector


class MolCaptionModel(nn.Module):
    """
    Molecular Captioning Model combining GNN, Projector, and LLM.

    Architecture:
    - GNN (frozen): Encodes molecular graphs to 768-dim embeddings
    - Projector (trainable): Maps GNN embeddings to LLM hidden space
    - LLM (LoRA): Qwen3-0.6B with 4-bit quantization and LoRA adapters

    The graph embedding is injected at the <|graph|> token position in the
    input sequence, allowing the LLM to "see" the molecular structure.
    """

    def __init__(
        self,
        config,
        gnn: MolGNN,
        device: str = "cuda",
    ):
        """
        Args:
            config: Configuration object
            gnn: Pre-trained MolGNN encoder
            device: Device to use
        """
        super().__init__()

        self.config = config
        self.device = device

        # 1. Freeze GNN encoder
        self.gnn = gnn
        for param in self.gnn.parameters():
            param.requires_grad = False
        self.gnn.eval()

        # 2. Create projector
        self.projector = SolidBridgeProjector(
            in_dim=config.gnn_out_dim,
            hidden_dim=config.proj_hidden,
            out_dim=config.llm_hidden,
            dropout=config.lora_dropout,
        ).to(device)

        # 3. Load tokenizer first to get vocab size
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_name,
            trust_remote_code=True,
        )

        # Add special tokens
        special_tokens = {"additional_special_tokens": ["<|graph|>"]}
        self.tokenizer.add_special_tokens(special_tokens)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # IMPORTANT: Use left padding for generation (decoder-only models)
        self.tokenizer.padding_side = "left"

        self.graph_token_id = self.tokenizer.convert_tokens_to_ids("<|graph|>")

        # 4. Load LLM (4-bit quantization on CUDA, full precision on CPU)
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        use_quantization = torch.cuda.is_available() and device != "cpu"

        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.llm_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # CPU mode: load in float32 without quantization
            print("Loading LLM without quantization (CPU mode)")
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.llm_name,
                dtype=torch.float32,
                device_map={"": device},
                trust_remote_code=True,
            )

        # Resize embeddings for new tokens
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # 5. Add LoRA adapters
        from peft import LoraConfig, get_peft_model

        # Only prepare for kbit training if using quantization
        if use_quantization:
            from peft import prepare_model_for_kbit_training
            self.llm = prepare_model_for_kbit_training(self.llm)

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=list(config.lora_target_modules),
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )


        self.llm = get_peft_model(self.llm, lora_config)

        # 6. Load txt_mean for centering
        self.txt_mean = None
        if config.center_embeddings and os.path.exists(config.txt_mean_path):
            print(f"Loading txt_mean from {config.txt_mean_path}")
            self.txt_mean = torch.load(config.txt_mean_path, map_location=device)
            # Ensure correct dtype (float16 if using quantization/amp)
            if use_quantization:
                self.txt_mean = self.txt_mean.half()
        elif config.center_embeddings:
            print(f"Warning: center_embeddings=True but {config.txt_mean_path} not found.")

    def process_gnn_embeddings(self, g: torch.Tensor) -> torch.Tensor:
        """
        Apply centering and normalization to match retrieval distribution.
        Logic: l2norm(g - txt_mean)
        """
        if self.txt_mean is not None:
            g = g - self.txt_mean
        
        return F.normalize(g, p=2, dim=-1)

    def encode_graphs(self, graphs: Batch) -> torch.Tensor:
        """
        Encode molecular graphs using the frozen GNN.

        Args:
            graphs: Batched PyG graphs

        Returns:
            Graph embeddings [batch_size, gnn_out_dim]
        """
        with torch.no_grad():
            self.gnn.eval()
            return self.gnn(graphs)

    def project_to_llm_space(self, graph_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project GNN embeddings to LLM hidden space.
        Applies centering and normalization beforehand.

        Args:
            graph_embeddings: [batch_size, gnn_out_dim]

        Returns:
            Projected tokens [batch_size, num_tokens, llm_hidden]
        """
        graph_embeddings = self.process_gnn_embeddings(graph_embeddings)
        return self.projector(graph_embeddings)

    def inject_graph_tokens(
        self,
        input_ids: torch.Tensor,
        graph_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace <|graph|> token positions with projected graph embeddings.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            graph_tokens: Projected graph embeddings [batch_size, num_tokens, llm_hidden]

        Returns:
            Modified input embeddings [batch_size, seq_len, llm_hidden]
        """
        # Get text embeddings from LLM (clone to avoid in-place modification of leaf variable)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids).clone()

        # Find <|graph|> token positions
        graph_positions = (input_ids == self.graph_token_id)

        # Replace with graph tokens
        batch_size = input_ids.size(0)
        for b in range(batch_size):
            pos_mask = graph_positions[b]
            if pos_mask.any():
                pos_indices = pos_mask.nonzero(as_tuple=True)[0]
                num_to_replace = min(len(pos_indices), graph_tokens.size(1))
                for t in range(num_to_replace):
                    inputs_embeds[b, pos_indices[t]] = graph_tokens[b, t]

        return inputs_embeds

    def forward(
        self,
        graphs: Batch,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            graphs: Batched molecular graphs
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target token IDs for loss computation [batch_size, seq_len]

        Returns:
            Dictionary with 'loss' and 'logits'
        """
        # Move inputs to device
        graphs = graphs.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # Encode graphs
        graph_embeddings = self.encode_graphs(graphs)

        # Project to LLM space
        graph_tokens = self.project_to_llm_space(graph_embeddings)

        # Inject into input embeddings
        inputs_embeds = self.inject_graph_tokens(input_ids, graph_tokens)

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def get_llm_text_embedding(self, text: str) -> torch.Tensor:
        """
        Get mean-pooled hidden state embedding for a text string.
        Used for alignment loss in Stage 1.

        Args:
            text: Input text string

        Returns:
            Text embedding [hidden_dim]
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.llm.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

        # Mean pool last hidden state
        last_hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        
        # Apply centering and normalization
        emb = pooled.squeeze(0)
        return self.process_gnn_embeddings(emb)

    def get_batch_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get mean-pooled embeddings for a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            Text embeddings [batch_size, hidden_dim]
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.llm.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

        # Mean pool last hidden state
        last_hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

        # Apply centering and normalization
        return self.process_gnn_embeddings(pooled)

    @torch.no_grad()
    def generate(
        self,
        graphs: Batch,
        smiles_list: List[str],
        max_new_tokens: int = 128,
        num_beams: int = 1,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Generate captions for a batch of molecular graphs.

        Args:
            graphs: Batched molecular graphs
            smiles_list: SMILES strings for each molecule
            max_new_tokens: Maximum tokens to generate
            num_beams: Beam search width (1 = greedy/sampling)
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Nucleus sampling threshold

        Returns:
            List of generated caption strings
        """
        from config import PROMPT_TEMPLATE

        self.eval()
        batch_size = len(smiles_list)

        # Build prompts
        prompts = [PROMPT_TEMPLATE.format(smiles=s) for s in smiles_list]

        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.device)

        # Encode and project graphs
        graphs = graphs.to(self.device)
        graph_embeddings = self.encode_graphs(graphs)
        graph_tokens = self.project_to_llm_space(graph_embeddings)

        # Inject graph tokens
        inputs_embeds = self.inject_graph_tokens(inputs["input_ids"], graph_tokens)

        # Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode outputs
        # Note: When using inputs_embeds, generate() returns only new tokens (not the prompt)
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text.strip())

        return generated_texts

    def print_trainable_parameters(self):
        """Print trainable parameter statistics."""
        total_params = 0
        trainable_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")


def create_model(config, device: str = "cuda") -> MolCaptionModel:
    """
    Create a MolCaptionModel with loaded GNN checkpoint.

    Args:
        config: Configuration object
        device: Device to use

    Returns:
        Initialized MolCaptionModel
    """
    from model_gnn import load_gnn_checkpoint

    # Load GNN
    gnn = load_gnn_checkpoint(config, device=device)

    # Create full model
    model = MolCaptionModel(config, gnn, device=device)

    return model
