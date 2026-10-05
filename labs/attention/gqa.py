"""Readable GQA reference; explicit KV repetition is not a production backend."""

import math

import torch
from torch import nn


class GQA(nn.Module):
    def __init__(self, hidden_dim, num_q_heads, num_kv_heads, head_dim):
        super().__init__()
        if min(hidden_dim, num_q_heads, num_kv_heads, head_dim) <= 0:
            raise ValueError("Dimensions and head counts must be positive")
        if num_q_heads % num_kv_heads:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        self.hidden_dim = hidden_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_q_heads // num_kv_heads
        self.w_q = nn.Linear(hidden_dim, num_q_heads * head_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.w_o = nn.Linear(num_q_heads * head_dim, hidden_dim, bias=False)

    def forward(self, x, *, is_causal=True, past_key_value=None, use_cache=False):
        """All batch entries have the same length; no padding or position encoding.

        past_key_value stores compact [B, Hkv, Tpast, dh] K/V tensors.
        If use_cache is True, return (output, present_key_value).
        """
        if x.ndim != 3 or x.shape[-1] != self.hidden_dim or x.shape[1] == 0:
            raise ValueError("x must have shape [B, S>0, hidden_dim]")
        batch, seq, _ = x.shape

        def split(tensor, heads):
            return tensor.view(batch, seq, heads, self.head_dim).transpose(1, 2)

        q = split(self.w_q(x), self.num_q_heads)
        k = split(self.w_k(x), self.num_kv_heads)
        v = split(self.w_v(x), self.num_kv_heads)
        past_len = 0
        if past_key_value is not None:
            if not is_causal:
                raise ValueError("This reference supports cached execution only in causal mode")
            past_k, past_v = past_key_value
            expected = (batch, self.num_kv_heads)
            if (past_k.ndim != 4 or past_v.shape != past_k.shape
                    or past_k.shape[:2] != expected
                    or past_k.shape[-1] != self.head_dim):
                raise ValueError("Cache must have shape [B, Hkv, Tpast, dh]")
            if (past_k.device != k.device or past_v.device != v.device
                    or past_k.dtype != k.dtype or past_v.dtype != v.dtype):
                raise ValueError("Cache device and dtype must match projected K/V")
            past_len = past_k.shape[-2]
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        # Save compact K/V; repetition below is only for a readable reference.
        present = (k, v)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if is_causal:
            query_pos = past_len + torch.arange(seq, device=x.device)
            key_pos = torch.arange(k.shape[-2], device=x.device)
            allowed = key_pos[None, :] <= query_pos[:, None]
            scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(
            batch, seq, self.num_q_heads * self.head_dim
        )
        out = self.w_o(out)
        return (out, present) if use_cache else out
