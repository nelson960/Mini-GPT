from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,d->td", t, self.inv_freq)  # (T, D/2)
        cos = freqs.cos().to(dtype=dtype)[None, :, None, :]  # (1,T,1,D/2)
        sin = freqs.sin().to(dtype=dtype)[None, :, None, :]  # (1,T,1,D/2)
        return cos, sin

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: (B,T,H,D)
        _, seq_len, _, _ = q.shape
        cos, sin = self._cos_sin(seq_len, q.device, q.dtype)
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w_gate = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_up = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w_gate(x)) * self.w_up(x)
        x = self.w_down(x)
        return self.dropout(x)


class MoESwiGLU(nn.Module):
    """Dense-compute MoE (small-model friendly): route top-k, mix expert outputs."""

    def __init__(
        self,
        n_embd: int,
        hidden_dim: int,
        dropout: float,
        n_experts: int,
        top_k: int,
    ):
        super().__init__()
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if top_k < 1 or top_k > n_experts:
            raise ValueError("top_k must be in [1, n_experts]")

        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(n_embd, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(n_embd, hidden_dim, dropout) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,T,C)
        gate_logits = self.router(x)  # (B,T,E)
        top_vals, top_idx = torch.topk(gate_logits, k=self.top_k, dim=-1)  # (B,T,K)
        top_weights = F.softmax(top_vals, dim=-1)  # (B,T,K)

        expert_out = torch.stack([expert(x) for expert in self.experts], dim=2)  # (B,T,E,C)
        dispatch = torch.zeros_like(gate_logits)
        dispatch.scatter_(-1, top_idx, top_weights)
        out = torch.einsum("bte,btec->btc", dispatch, expert_out)

        # Switch-like load balancing term.
        router_probs = F.softmax(gate_logits, dim=-1)  # (B,T,E)
        mean_prob = router_probs.mean(dim=(0, 1))
        top1 = top_idx[..., 0].reshape(-1)
        token_fraction = F.one_hot(top1, num_classes=self.n_experts).float().mean(dim=0)
        aux_loss = self.n_experts * torch.sum(mean_prob * token_fraction)

        return out, aux_loss


class ModernCausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        block_size: int,
        rope_theta: float,
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, theta=rope_theta)

        mask = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C)
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.n_head, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.n_head, self.head_dim)

        q, k = self.rope.apply_rotary(q, k)

        # (B,H,T,D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask, float("-inf"))

        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)

        out = torch.matmul(probs, v)  # (B,H,T,D)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        out = self.o_proj(out)
        out = self.resid_dropout(out)
        return out


class ModernBlock(nn.Module):
    def __init__(self, cfg: ModernGPTConfig):
        super().__init__()
        hidden_dim = int(cfg.ffn_mult * cfg.n_embd)

        self.norm1 = RMSNorm(cfg.n_embd, eps=cfg.rms_norm_eps)
        self.attn = ModernCausalSelfAttention(
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            dropout=cfg.dropout,
            block_size=cfg.block_size,
            rope_theta=cfg.rope_theta,
        )
        self.norm2 = RMSNorm(cfg.n_embd, eps=cfg.rms_norm_eps)

        self.use_moe = cfg.use_moe
        if self.use_moe:
            self.ff = MoESwiGLU(
                n_embd=cfg.n_embd,
                hidden_dim=hidden_dim,
                dropout=cfg.dropout,
                n_experts=cfg.n_experts,
                top_k=cfg.moe_top_k,
            )
        else:
            self.ff = SwiGLU(cfg.n_embd, hidden_dim, cfg.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x))

        aux = x.new_zeros(())
        if self.use_moe:
            ff_out, aux = self.ff(self.norm2(x))
        else:
            ff_out = self.ff(self.norm2(x))

        x = x + ff_out
        return x, aux


@dataclass
class ModernGPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192
    dropout: float = 0.1
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-8
    ffn_mult: float = 4.0

    use_moe: bool = False
    n_experts: int = 4
    moe_top_k: int = 1
    moe_aux_loss_coef: float = 1e-2

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if self.moe_top_k < 1 or self.moe_top_k > self.n_experts:
            raise ValueError("moe_top_k must be in [1, n_experts]")


class ModernGPT(nn.Module):
    def __init__(self, cfg: ModernGPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([ModernBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        # Weight tying improves sample efficiency on small-data setups.
        self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # idx: (B,T)
        bsz, seq_len = idx.shape
        if seq_len > self.cfg.block_size:
            raise ValueError(f"T={seq_len} exceeds block_size={self.cfg.block_size}")

        x = self.tok_emb(idx)
        x = self.drop(x)

        total_aux = x.new_zeros(())
        for block in self.blocks:
            x, aux = block(x)
            total_aux = total_aux + aux

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(bsz * seq_len, -1), targets.reshape(bsz * seq_len))
            if self.cfg.use_moe:
                loss = loss + self.cfg.moe_aux_loss_coef * total_aux

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B,T,H,D), cos/sin: (1,T,1,D/2)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)
