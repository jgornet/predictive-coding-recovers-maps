from typing import List, Tuple, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import math


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
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.query_proj = nn.Conv2d(d_model, self.d_head * num_heads, kernel_size=1, bias=False)
        self.key_proj = nn.Conv2d(d_model, self.d_head * num_heads, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(d_model, self.d_head * num_heads, kernel_size=1, bias=False)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)
        channels = value.size(2)
        height = value.size(3)
        width = value.size(4)

        query = query.reshape(-1, channels, height, width)
        query = self.query_proj(query)
        query = query.reshape(batch_size, -1, self.num_heads, self.d_head, height, width)

        key = key.reshape(-1, channels, height, width)
        key = self.key_proj(key)
        key = key.reshape(batch_size, -1, self.num_heads, self.d_head, height, width)

        value = value.reshape(-1, channels, height, width)
        value = self.value_proj(value)
        value = value.reshape(batch_size, -1, self.num_heads, self.d_head, height, width)

        query = query.permute(2, 0, 1, 3, 4, 5).contiguous().view(batch_size * self.num_heads, -1, self.d_head, height, width)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3, 4, 5).contiguous().view(batch_size * self.num_heads, -1, self.d_head, height, width)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3, 4, 5).contiguous().view(batch_size * self.num_heads, -1, self.d_head, height, width)  # BNxV_LENxD

        # if mask is not None:
            # mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head, height, width)  # Nx(B)xQ_LENxD
        context = context.permute(1, 2, 0, 3, 4, 5).contiguous().view(batch_size, -1, self.num_heads * self.d_head, height, width)  # BxTxND

        return context, attn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.einsum("b i d h w, b j d h w -> b i j", query, key) / self.sqrt_dim  # BxQ_LENxK_LEN

        if mask is not None:
            score[:, mask] = -float('Inf')
            # score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.einsum("b i j, b j d h w -> b i d h w", attn, value)  # BxQ_LENxD

        return context, attn


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
