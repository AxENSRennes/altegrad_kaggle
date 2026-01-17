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

from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from model_gnn import MolGNN
from model_projector import SolidBridgeProjector
from metadata import TXT_MEAN_V1


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

        # 4. Load LLM with hardware-aware configuration
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        # Detect hardware mode from config or infer from device
        hardware_mode = getattr(config, 'hardware_mode', 'auto')
        use_quantization = getattr(config, 'use_quantization', True)

        if hardware_mode == "auto":
            # Auto-detect: TPU check, then GPU, then CPU
            try:
                import importlib.util
                if importlib.util.find_spec("torch_xla") is not None:
                    hardware_mode = "tpu"
                else:
                    hardware_mode = "gpu" if torch.cuda.is_available() else "cpu"
            except Exception:
                hardware_mode = "gpu" if torch.cuda.is_available() else "cpu"

        # Disable quantization for TPU/CPU
        if hardware_mode in ("tpu", "cpu"):
            use_quantization = False

        if hardware_mode == "tpu":
            # TPU: bfloat16, no quantization, no device_map
            print("Loading LLM for TPU (bfloat16, no quantization)")
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.llm_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        elif hardware_mode == "cpu":
            # CPU: float32, no quantization, for local testing
            print("Loading LLM for CPU (float32, no quantization)")
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.llm_name,
                torch_dtype=torch.float32,
                device_map={"": device},
                trust_remote_code=True,
            )
        elif use_quantization and torch.cuda.is_available():
            # GPU with BitsAndBytes 4-bit quantization
            print("Loading LLM for GPU (4-bit quantization)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.llm_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # GPU without quantization (bfloat16)
            print("Loading LLM for GPU (bfloat16, no quantization)")
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.llm_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        # Store hardware mode for later use
        self.hardware_mode = hardware_mode

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
        if config.center_embeddings:
            # Load from embedded metadata
            self.txt_mean = torch.tensor(TXT_MEAN_V1, device=device)
            # Ensure correct dtype based on hardware mode
            if hardware_mode in ("tpu", "gpu"):
                self.txt_mean = self.txt_mean.to(torch.bfloat16)

    def process_gnn_embeddings(self, g: torch.Tensor) -> torch.Tensor:
        """
        Apply centering and normalization to match retrieval distribution.
        Logic: l2norm(g - txt_mean) if dimensions match.
        """
        if self.txt_mean is not None:
            # Only apply centering if dimensions match (e.g., GNN = 768, LLM = 1024)
            if g.size(-1) == self.txt_mean.size(-1):
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

        Uses XLA-compatible static masking with torch.where() instead of
        dynamic .nonzero() indexing to avoid graph recompilation on TPU.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            graph_tokens: Projected graph embeddings [batch_size, 1, llm_hidden]

        Returns:
            Modified input embeddings [batch_size, seq_len, llm_hidden]
        """
        # Get text embeddings from LLM (clone to avoid in-place modification of leaf variable)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids).clone()

        # Find <|graph|> token positions
        graph_mask = (input_ids == self.graph_token_id)

        # Expand graph token to match sequence length for broadcasting
        graph_token_expanded = graph_tokens.expand(-1, input_ids.size(1), -1)
        inputs_embeds = torch.where(
            graph_mask.unsqueeze(-1),
            graph_token_expanded,
            inputs_embeds
        )

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
        enable_thinking: bool = False,
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
            enable_thinking: Enable thinking mode (model reasons before answering)

        Returns:
            List of generated caption strings
        """
        from config import SYSTEM_PROMPT, SYSTEM_PROMPT_THINK, USER_PROMPT_FORMAT

        self.eval()

        # Select system prompt based on thinking mode
        system_prompt = SYSTEM_PROMPT_THINK if enable_thinking else SYSTEM_PROMPT

        # Build prompts using tokenizer.apply_chat_template
        prompts = []
        for s in smiles_list:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT_FORMAT.format(smiles=s)}
            ]
            # Pass enable_thinking to chat template
            p = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            prompts.append(p)

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
            # Post-process: strip thinking content if thinking was enabled
            if enable_thinking and "</think>" in text:
                text = text.split("</think>", 1)[1]
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
