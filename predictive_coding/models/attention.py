from typing import List, Tuple, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import math
from xformers.ops import memory_efficient_attention as sdpa_attention
from xformers.ops import LowerTriangularMask


class MultiHeadAttention(nn.Module):
    """
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)
    Args:
        d_model (int): The dimension of keys / values / queries (default: 512)
        num_heads (int): The number of attention heads. (default: 8)
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): In transformer, three different ways:
            Case 1: come from previoys decoder layer
            Case 2: come from the input embedding
            Case 3: come from the output embedding (masked)
        - **key** (batch, k_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)
        - **value** (batch, v_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)
        - **mask** (-): tensor containing indices to be masked
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 2):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.query_proj = nn.Conv2d(
            d_model, self.d_head * num_heads, kernel_size=1, bias=False
        )
        self.key_proj = nn.Conv2d(
            d_model, self.d_head * num_heads, kernel_size=1, bias=False
        )
        self.value_proj = nn.Conv2d(
            d_model, self.d_head * num_heads, kernel_size=1, bias=False
        )

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size = value.size(0)
        channels = value.size(2)
        height = value.size(3)
        width = value.size(4)

        query = query.reshape(-1, channels, height, width)
        query = self.query_proj(query)
        query = query.reshape(
            batch_size, -1, self.num_heads, self.d_head, height, width
        )

        key = key.reshape(-1, channels, height, width)
        key = self.key_proj(key)
        key = key.reshape(batch_size, -1, self.num_heads, self.d_head, height, width)

        value = value.reshape(-1, channels, height, width)
        value = self.value_proj(value)
        value = value.reshape(
            batch_size, -1, self.num_heads, self.d_head, height, width
        )

        query = query.contiguous().view(
            batch_size, -1, self.num_heads, self.d_head * height * width
        )
        key = key.contiguous().view(
            batch_size, -1, self.num_heads, self.d_head * height * width
        )
        value = value.contiguous().view(
            batch_size, -1, self.num_heads, self.d_head * height * width
        )

        context = sdpa_attention(query, key, value, LowerTriangularMask())
        attn = None
        context = context.view(
            batch_size, -1, self.num_heads * self.d_head, height, width
        )

        return context, attn


class RotaryEncoding(nn.Module):
    def __init__(self, in_channels, max_len=2048, base=10000, device="cuda:0"):
        super().__init__()
        self.in_channels = in_channels
        self.max_len = max_len
        self.base = base
        self.inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.in_channels, 2, dtype=torch.int64)
                .float()
                .to(device)
                / self.in_channels
            )
        )
        self.cache_len = 0

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.cache_len = seq_len
        t = torch.arange(self.cache_len, device=device, dtype=torch.int64).type_as(
            self.inv_freq
        )

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos().to(dtype)
        self.sin = emb.sin().to(dtype)

    def forward(self, x, seq_len=None):
        # x: [batch_dim, time_dim, num_heads, head_dim]
        if seq_len > self.cache_len:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos[:seq_len].to(dtype=x.dtype),
            self.sin[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_position_embedding(q, k, cos, sin, position_ids, unsqueeze_dim=2):
    # q: [batch_dim, time_dim, num_heads, head_dim]
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
