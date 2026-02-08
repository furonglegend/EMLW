"""
hsf.py

Heterogeneous Stream Fusion (HSF) implemented as a cross-attention fusion module.

- Accepts a list of modality tensors (B, C_i, T_i) or pre-computed embeddings (B, D)
- Applies cross-attention (multi-head) to fuse modalities into a single embedding per example.
- Lightweight and designed for edge use.

This file uses torch.nn.MultiheadAttention on sequence embeddings; modalities are first projected.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based fusion for heterogeneous streams.
    Example usage:
       fusion = CrossAttentionFusion(modality_dims=[32, 16], model_dim=128, num_heads=4)
       fused = fusion([mod1_emb, mod2_emb])  # returns (B, model_dim)
    """
    def __init__(self, modality_dims: List[int], model_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.modalities = len(modality_dims)
        self.model_dim = model_dim
        # per-modality linear projection to model_dim
        self.projections = nn.ModuleList([nn.Linear(d, model_dim) for d in modality_dims])
        # single MultiheadAttention used for cross-attention (query from target modality)
        self.mha = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim, model_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, modality_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        modality_embeddings: list of tensors, each either:
            - (B, D_i) : a pooled embedding per modality
            - (B, L, D_i) : sequence embeddings per modality (we will pool if needed)
        Returns:
            fused embedding (B, model_dim)
        """
        # normalize and project
        projected = []
        for i, emb in enumerate(modality_embeddings):
            if emb.dim() == 2:
                # (B, D_i) -> (B, 1, D_i)
                seq = emb.unsqueeze(1)
            elif emb.dim() == 3:
                seq = emb  # (B, L, D_i)
                # optionally pool or keep sequence - we use mean pooling for simplicity
                seq = seq.mean(dim=1, keepdim=True)  # (B,1,D_i)
            else:
                raise ValueError("Each modality embedding must be 2D or 3D tensor.")
            proj = self.projections[i](seq)  # (B,1,model_dim)
            projected.append(proj)

        # Concatenate along sequence dimension to build key/value bank
        # bank: (B, M, model_dim) where M = number of modalities
        bank = torch.cat(projected, dim=1)

        # Use the first modality as query (could use learned query or pooled bank)
        query = projected[0]  # (B,1,model_dim)

        # MultiheadAttention expects (B, L, E) when batch_first=True
        attn_out, attn_weights = self.mha(query, bank, bank, need_weights=True)  # out: (B,1,E)
        attn_out = attn_out.squeeze(1)  # (B, E)
        # FFN + residual + norm
        res = self.layernorm(attn_out + self.dropout(self.ffn(attn_out)))
        return res
