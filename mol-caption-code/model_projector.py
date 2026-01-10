#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SolidBridgeProjector: MLP projector to bridge GNN embeddings to LLM space.

Projects 768-dim GNN embeddings to the LLM's hidden dimension (896 for Qwen3-0.6B).
Uses a 3-layer MLP with GELU activation, LayerNorm, and Dropout.
"""

import torch
import torch.nn as nn


class SolidBridgeProjector(nn.Module):
    """
    Multi-layer projector that bridges molecular graph embeddings to LLM embedding space.

    Architecture:
    - Layer 1: in_dim → hidden (Linear + GELU + LayerNorm + Dropout)
    - Layer 2: hidden → hidden (Linear + GELU + LayerNorm + Dropout)
    - Layer 3: hidden → out_dim (Linear + LayerNorm)

    The output is reshaped to [batch, 1, out_dim] to serve as a single "graph token"
    that can be injected into the LLM's input embeddings.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 1024,
        out_dim: int = 896,
        dropout: float = 0.1,
        num_tokens: int = 1,
    ):
        """
        Args:
            in_dim: Input dimension (GNN embedding size, typically 768)
            hidden_dim: Hidden layer dimension
            out_dim: Output dimension (LLM hidden size, 896 for Qwen3-0.6B)
            dropout: Dropout rate
            num_tokens: Number of graph tokens to produce (default 1)
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.out_dim = out_dim

        # Layer 1: in_dim -> hidden_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Layer 2: hidden_dim -> hidden_dim
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Layer 3: hidden_dim -> out_dim * num_tokens
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim * num_tokens),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights specifically to target an output norm of ~1.0.
        Intermediate LayerNorms force a norm of ~32, so the final layer
        must scale this down (32 * 0.03nd-layer-gain * 0.03final-layer-gain).
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # If it's the last layer (layer3), we use a very small scale
                # to cancel out the ~32 norm from the previous LayerNorm.
                if 'layer3' in name:
                    std = 0.001 # 32 * 0.001 * sqrt(1024) approx 1.0
                else:
                    std = 0.03
                
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project GNN embeddings to LLM space.

        Args:
            x: GNN embeddings of shape [batch_size, in_dim]

        Returns:
            Projected embeddings of shape [batch_size, num_tokens, out_dim]
        """
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)

        # Reshape to [batch, num_tokens, out_dim]
        batch_size = x.size(0)
        return h.view(batch_size, self.num_tokens, self.out_dim)


class ResidualProjector(nn.Module):
    """
    Alternative projector with residual connections.

    Useful when in_dim == out_dim or with an initial projection layer.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 1024,
        out_dim: int = 896,
        dropout: float = 0.1,
        num_layers: int = 3,
    ):
        super().__init__()

        self.out_dim = out_dim

        # Initial projection if dimensions don't match
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                )
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_dim]

        Returns:
            [batch_size, 1, out_dim]
        """
        h = self.input_proj(x)

        for block in self.blocks:
            h = h + block(h)  # Residual connection

        h = self.output_proj(h)
        return h.unsqueeze(1)  # [batch, 1, out_dim]
