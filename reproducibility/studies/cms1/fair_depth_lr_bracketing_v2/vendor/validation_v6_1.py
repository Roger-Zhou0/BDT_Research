#!/usr/bin/env python3
"""
V6.1 — round-aware recurrence and latent topology controls.

Goal:
  Diagnose the two bottlenecks exposed by V6.0 language modeling:
  (1) recurrent weight tying without a depth/time coordinate, and
  (2) learned sparse masks collapsing into highly overlapping subspaces.

New conditions:
  looped_dense_roundaware
  recurrent_sparse_roundaware_learned
  recurrent_sparse_roundaware_identical
  recurrent_sparse_roundaware_random
  recurrent_sparse_roundaware_low_overlap

All conditions use the same cached FineWeb-Edu token stream and V6.0 training
budget. The sparse topology controls keep the router learned and unsupervised;
only the parameter-subspace masks are fixed. Scientific evaluation uses the
dense hard-mask implementation; physical sparse dispatch is used only for
throughput benchmarking.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VERSION = "v6_1_roundaware_topology_v1"
CONDITIONS = (
    "looped_dense_roundaware",
    "recurrent_sparse_roundaware_learned",
    "recurrent_sparse_roundaware_identical",
    "recurrent_sparse_roundaware_random",
    "recurrent_sparse_roundaware_low_overlap",
)
DEFAULT_SEEDS = (41, 42, 43)


# ============================================================================
# Utilities
# ============================================================================


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def stats(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else math.nan,
        "n": int(arr.size),
    }


def paired_stats(a: Iterable[float], b: Iterable[float]) -> dict[str, float]:
    aa = np.asarray(list(a), dtype=np.float64)
    bb = np.asarray(list(b), dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError("paired arrays must have identical shape")
    d = aa - bb
    mean = float(d.mean())
    if d.size <= 1:
        return {
            "mean": mean,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "n": int(d.size),
        }
    t95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}.get(
        int(d.size), 1.96
    )
    se = float(d.std(ddof=1) / math.sqrt(d.size))
    return {
        "mean": mean,
        "ci95_low": mean - t95 * se,
        "ci95_high": mean + t95 * se,
        "n": int(d.size),
    }


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ExperimentConfig:
    vocab_size: int = 50_257  # GPT-2 tokenizer
    context_length: int = 512

    d_model: int = 384
    n_heads: int = 6
    n_layers_or_rounds: int = 8
    max_recurrent_rounds: int = 12
    ffn_hidden: int = 1024

    n_modes: int = 8
    sparse_bank_hidden: int = 4096
    router_hidden: int = 128
    mask_tau: float = 1.0
    router_tau_start: float = 1.5
    router_tau_end: float = 0.5
    router_balance_weight: float = 0.01
    router_z_weight: float = 1.0e-4

    train_steps: int = 3000
    warmup_steps: int = 200
    validate_every: int = 250
    checkpoint_every: int = 100
    micro_batch: int = 8
    grad_accum: int = 8
    eval_batches: int = 32

    learning_rate: float = 3.0e-4
    min_lr_ratio: float = 0.10
    weight_decay: float = 0.10
    beta1: float = 0.90
    beta2: float = 0.95
    grad_clip: float = 1.0

    precision: str = "bf16"
    data_dir: str = "data/v6_fineweb_edu"
    output_root: str = "runs/v6_0_latent_mode_lm"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.sparse_bank_hidden < self.ffn_hidden:
            raise ValueError("sparse_bank_hidden must be >= ffn_hidden")
        if self.vocab_size > np.iinfo(np.uint16).max:
            raise ValueError("current token cache expects uint16 token IDs")

    @property
    def tokens_per_update(self) -> int:
        return self.context_length * self.micro_batch * self.grad_accum

    @property
    def train_tokens(self) -> int:
        return self.tokens_per_update * self.train_steps

    def signature(self, condition: str, seed: int) -> str:
        payload = {
            "version": VERSION,
            "condition": condition,
            "seed": seed,
            "config": asdict(self),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ============================================================================
# Token store
# ============================================================================


class TokenStore:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        self.path = path
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")

    def batch(
        self,
        batch_size: int,
        context_length: int,
        seed: int,
        step: int,
        microstep: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.tokens) - context_length - 1
        if max_start <= 0:
            raise RuntimeError(f"token store {self.path} is too short")
        g = np.random.default_rng(
            np.uint64(seed) * np.uint64(1_000_003)
            + np.uint64(step) * np.uint64(1009)
            + np.uint64(microstep) * np.uint64(9176)
        )
        starts = g.integers(0, max_start, size=batch_size, dtype=np.int64)
        offsets = np.arange(context_length + 1, dtype=np.int64)
        arr = np.asarray(self.tokens[starts[:, None] + offsets[None, :]], dtype=np.int64)
        x = torch.from_numpy(arr[:, :-1]).to(device=device, non_blocking=True)
        y = torch.from_numpy(arr[:, 1:]).to(device=device, non_blocking=True)
        return x, y


# ============================================================================
# Model components
# ============================================================================


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y.to(dtype) * self.weight)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q,k: [B,H,T,D]
        t = q.shape[-2]
        cos = self.cos[:t].to(device=q.device, dtype=q.dtype)[None, None, :, :]
        sin = self.sin[:t].to(device=q.device, dtype=q.dtype)[None, None, :, :]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.context_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.rope.apply(q, k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DenseBlock(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SharedSparseSwiGLU(nn.Module):
    def __init__(
        self,
        config: ExperimentConfig,
        fixed_masks: torch.Tensor | None = None,
    ):
        super().__init__()
        d = config.d_model
        k = config.sparse_bank_hidden
        self.bank_hidden = k
        self.active_hidden = config.ffn_hidden
        self.n_modes = config.n_modes
        self.mask_tau = config.mask_tau

        self.gate_weight = nn.Parameter(torch.empty(k, d))
        self.up_weight = nn.Parameter(torch.empty(k, d))
        self.down_weight = nn.Parameter(torch.empty(d, k))
        self.mask_logits = nn.Parameter(0.01 * torch.randn(config.n_modes, k))
        nn.init.normal_(self.gate_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_weight, mean=0.0, std=0.02)

        self.fixed_topology = fixed_masks is not None
        if fixed_masks is not None:
            fixed_masks = fixed_masks.to(dtype=torch.bool, device="cpu")
            expected = (config.n_modes, config.sparse_bank_hidden)
            if tuple(fixed_masks.shape) != expected:
                raise ValueError(f"fixed mask shape {tuple(fixed_masks.shape)} != {expected}")
            if not torch.equal(
                fixed_masks.sum(dim=1),
                torch.full((config.n_modes,), config.ffn_hidden, dtype=torch.long),
            ):
                raise ValueError("every fixed mode must activate exactly ffn_hidden units")
            with torch.no_grad():
                self.mask_logits.copy_(
                    torch.where(
                        fixed_masks,
                        torch.ones_like(self.mask_logits),
                        -torch.ones_like(self.mask_logits),
                    )
                )
            # Preserve the tensor in state_dict / parameter counts, but do not
            # optimize topology in fixed-mask controls.
            self.mask_logits.requires_grad_(False)

    def hard_masks(self) -> torch.Tensor:
        top = torch.topk(self.mask_logits, k=self.active_hidden, dim=-1).indices
        hard = torch.zeros_like(self.mask_logits)
        hard.scatter_(1, top, 1.0)
        return hard

    def straight_through_masks(self) -> torch.Tensor:
        hard = self.hard_masks()
        if not self.mask_logits.requires_grad:
            return hard
        soft = torch.sigmoid(self.mask_logits / self.mask_tau)
        soft = soft * (self.active_hidden / soft.sum(dim=-1, keepdim=True).clamp_min(1e-6))
        return hard + soft - soft.detach()

    def forward_dense_train(self, x: torch.Tensor, route_weights: torch.Tensor) -> torch.Tensor:
        # x [N,D], route_weights [N,M], hard in forward / soft in backward.
        masks = self.straight_through_masks().to(dtype=x.dtype)
        effective = route_weights.to(dtype=x.dtype) @ masks
        gate = F.linear(x, self.gate_weight)
        up = F.linear(x, self.up_weight)
        hidden = F.silu(gate) * up
        hidden = hidden * effective
        return F.linear(hidden, self.down_weight)

    def forward_sparse_eval(self, x: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        active = torch.topk(self.mask_logits, k=self.active_hidden, dim=-1).indices
        for mode in range(self.n_modes):
            rows = torch.nonzero(route_ids == mode, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            idx = active[mode]
            xr = x.index_select(0, rows)
            gate = F.linear(xr, self.gate_weight.index_select(0, idx))
            up = F.linear(xr, self.up_weight.index_select(0, idx))
            hidden = F.silu(gate) * up
            down_w = self.down_weight.index_select(1, idx)
            yr = F.linear(hidden, down_w)
            result.index_copy_(0, rows, yr.to(result.dtype))
        return result

    @torch.no_grad()
    def topology_stats(self) -> dict[str, float]:
        masks = self.hard_masks().bool().cpu()
        pair_i = []
        pair_j = []
        for i in range(self.n_modes):
            for j in range(i + 1, self.n_modes):
                inter = int((masks[i] & masks[j]).sum())
                union = int((masks[i] | masks[j]).sum())
                pair_i.append(inter / self.active_hidden)
                pair_j.append(inter / max(union, 1))
        usage_count = masks.sum(dim=0)
        return {
            "active_fraction": self.active_hidden / self.bank_hidden,
            "mean_pairwise_intersection_over_active": float(np.mean(pair_i)),
            "mean_pairwise_jaccard": float(np.mean(pair_j)),
            "bank_union_fraction": float(masks.any(dim=0).float().mean()),
            "shared_by_all_fraction": float(masks.all(dim=0).float().mean()),
            "mean_memberships_per_used_unit": float(
                usage_count.sum().item() / max(int((usage_count > 0).sum()), 1)
            ),
        }


class LatentSparseBlock(nn.Module):
    def __init__(
        self,
        config: ExperimentConfig,
        fixed_masks: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = config
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.router_in = nn.Linear(config.d_model, config.router_hidden, bias=False)
        self.router_out = nn.Linear(config.router_hidden, config.n_modes, bias=False)
        self.ffn = SharedSparseSwiGLU(config, fixed_masks=fixed_masks)

    def forward(
        self,
        x: torch.Tensor,
        tau: float,
        stochastic: bool,
        sparse_eval: bool,
        round_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # The round embedding is a time/depth coordinate, not an operation cue.
        # It conditions the shared transformation without being permanently
        # accumulated into the recurrent state.
        attn_input = x if round_embedding is None else x + round_embedding
        x = x + self.attn(self.norm1(attn_input))
        ffn_input = x if round_embedding is None else x + round_embedding
        y = self.norm2(ffn_input)
        logits = self.router_out(F.silu(self.router_in(y))).float()
        probs = torch.softmax(logits, dim=-1)

        if stochastic:
            route_weights = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        else:
            route_ids = logits.argmax(dim=-1)
            route_weights = F.one_hot(route_ids, self.config.n_modes).float()
        route_ids = route_weights.argmax(dim=-1)

        flat_y = y.reshape(-1, y.shape[-1])
        flat_routes = route_weights.reshape(-1, self.config.n_modes)
        flat_ids = route_ids.reshape(-1)
        if sparse_eval:
            ffn_out = self.ffn.forward_sparse_eval(flat_y, flat_ids)
        else:
            ffn_out = self.ffn.forward_dense_train(flat_y, flat_routes)
        x = x + ffn_out.view_as(y)

        importance = probs.mean(dim=(0, 1))
        load = F.one_hot(route_ids, self.config.n_modes).float().mean(dim=(0, 1))
        balance = self.config.n_modes * torch.sum(importance * load.detach())
        z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2)
        entropy = -torch.sum(probs * probs.clamp_min(1e-9).log(), dim=-1).mean()
        aux = (
            self.config.router_balance_weight * (balance - 1.0)
            + self.config.router_z_weight * z_loss
        )
        return x, {
            "auxiliary_loss": aux,
            "route_ids": route_ids,
            "route_probs": probs,
            "balance": balance.detach(),
            "z_loss": z_loss.detach(),
            "entropy": entropy.detach(),
        }


# ============================================================================
# Causal LMs
# ============================================================================


class BaseLM(nn.Module):
    condition: str

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.final_norm = RMSNorm(config.d_model)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        # Tied input/output embeddings.
        return F.linear(self.final_norm(h), self.token_embedding.weight)


class DenseStackLM(BaseLM):
    condition = "dense_stack"

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.blocks = nn.ModuleList([DenseBlock(config) for _ in range(config.n_layers_or_rounds)])

    def forward(self, input_ids: torch.Tensor, **_: Any):
        h = self.token_embedding(input_ids)
        for block in self.blocks:
            h = block(h)
        return self.logits(h), {"auxiliary_loss": h.new_zeros(())}


class LoopedDenseLM(BaseLM):
    condition = "looped_dense"

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.block = DenseBlock(config)

    def forward(self, input_ids: torch.Tensor, rounds: int | None = None, **_: Any):
        h = self.token_embedding(input_ids)
        n = self.config.n_layers_or_rounds if rounds is None else rounds
        for _ in range(n):
            h = self.block(h)
        return self.logits(h), {"auxiliary_loss": h.new_zeros(())}


class RoundAwareDenseBlock(DenseBlock):
    def forward(self, x: torch.Tensor, round_embedding: torch.Tensor) -> torch.Tensor:
        # Condition the shared computation on recurrent depth while keeping the
        # evolving state itself free of accumulated round embeddings.
        x = x + self.attn(self.norm1(x + round_embedding))
        x = x + self.ffn(self.norm2(x + round_embedding))
        return x


class LoopedDenseRoundAwareLM(BaseLM):
    condition = "looped_dense_roundaware"

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.block = RoundAwareDenseBlock(config)
        self.round_embedding = nn.Embedding(config.max_recurrent_rounds, config.d_model)

    def forward(self, input_ids: torch.Tensor, rounds: int | None = None, **_: Any):
        h = self.token_embedding(input_ids)
        n = self.config.n_layers_or_rounds if rounds is None else rounds
        if n > self.config.max_recurrent_rounds:
            raise ValueError(f"rounds={n} exceeds max_recurrent_rounds")
        for rr in range(n):
            e = self.round_embedding.weight[rr][None, None, :]
            h = self.block(h, e)
        return self.logits(h), {"auxiliary_loss": h.new_zeros(())}


class RecurrentSparseLM(BaseLM):
    condition = "recurrent_sparse"

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        round_aware: bool = False,
        fixed_masks: torch.Tensor | None = None,
    ):
        super().__init__(config)
        self.round_aware = round_aware
        self.block = LatentSparseBlock(config, fixed_masks=fixed_masks)
        self.round_embedding = (
            nn.Embedding(config.max_recurrent_rounds, config.d_model)
            if round_aware else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        tau: float | None = None,
        stochastic: bool = False,
        sparse_eval: bool = True,
        rounds: int | None = None,
        collect_routes: bool = False,
    ):
        h = self.token_embedding(input_ids)
        n = self.config.n_layers_or_rounds if rounds is None else rounds
        if self.round_aware and n > self.config.max_recurrent_rounds:
            raise ValueError(f"rounds={n} exceeds max_recurrent_rounds")
        tau = self.config.router_tau_end if tau is None else tau
        aux = h.new_zeros(())
        route_ids = []
        entropies = []
        balances = []
        for rr in range(n):
            e = None
            if self.round_embedding is not None:
                e = self.round_embedding.weight[rr][None, None, :]
            h, diag = self.block(
                h,
                tau=tau,
                stochastic=stochastic,
                sparse_eval=sparse_eval,
                round_embedding=e,
            )
            aux = aux + diag["auxiliary_loss"]
            if collect_routes:
                route_ids.append(diag["route_ids"].detach())
                entropies.append(float(diag["entropy"]))
                balances.append(float(diag["balance"]))
        out = {"auxiliary_loss": aux / n}
        if collect_routes:
            out.update(
                {
                    "route_ids": route_ids,
                    "router_entropy": entropies,
                    "router_balance": balances,
                }
            )
        return self.logits(h), out


# ============================================================================
# Construction / compute accounting
# ============================================================================


def initialize_model(model: nn.Module) -> None:
    # Qwen/GPT-style small normal initialization. In particular, this avoids
    # PyTorch Embedding's much larger default scale from dominating tied logits.
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
    if isinstance(model, RecurrentSparseLM):
        nn.init.normal_(model.block.ffn.gate_weight, mean=0.0, std=0.02)
        nn.init.normal_(model.block.ffn.up_weight, mean=0.0, std=0.02)
        nn.init.normal_(model.block.ffn.down_weight, mean=0.0, std=0.02)


def identical_masks(config: ExperimentConfig, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(61_000_000 + seed)
    idx = torch.randperm(config.sparse_bank_hidden, generator=g)[: config.ffn_hidden]
    masks = torch.zeros(config.n_modes, config.sparse_bank_hidden, dtype=torch.bool)
    masks[:, idx] = True
    return masks


def random_masks(config: ExperimentConfig, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(62_000_000 + seed)
    masks = torch.zeros(config.n_modes, config.sparse_bank_hidden, dtype=torch.bool)
    for mode in range(config.n_modes):
        idx = torch.randperm(config.sparse_bank_hidden, generator=g)[: config.ffn_hidden]
        masks[mode, idx] = True
    return masks


def low_overlap_masks(config: ExperimentConfig, seed: int) -> torch.Tensor:
    # V6.1 preregistered construction for M=8, K=4096, active=1024.
    # There are 8192 total memberships. The minimum-overlap balanced solution
    # uses every bank unit in exactly two modes. Each of 28 mode pairs gets 146
    # units, plus one extra unit on each edge of an 8-cycle. Hence every mode
    # has degree 7*146 + 2 = 1024 and every unit is used exactly twice.
    if (
        config.n_modes != 8
        or config.sparse_bank_hidden != 4096
        or config.ffn_hidden != 1024
    ):
        raise ValueError(
            "low_overlap_masks is preregistered for n_modes=8, bank=4096, active=1024"
        )
    pair_counts = {(i, j): 146 for i in range(8) for j in range(i + 1, 8)}
    cycle = [(i, (i + 1) % 8) for i in range(8)]
    for a, b in cycle:
        if a > b:
            a, b = b, a
        pair_counts[(a, b)] += 1

    unit_pairs = []
    for pair in sorted(pair_counts):
        unit_pairs.extend([pair] * pair_counts[pair])
    if len(unit_pairs) != 4096:
        raise RuntimeError("low-overlap construction must allocate exactly 4096 units")

    g = torch.Generator(device="cpu").manual_seed(63_000_000 + seed)
    perm = torch.randperm(4096, generator=g)
    masks = torch.zeros(8, 4096, dtype=torch.bool)
    for pos, pair in enumerate(unit_pairs):
        unit = int(perm[pos])
        masks[pair[0], unit] = True
        masks[pair[1], unit] = True
    expected = torch.full((8,), 1024, dtype=torch.long)
    if not torch.equal(masks.sum(dim=1), expected):
        raise RuntimeError(f"invalid low-overlap row sums: {masks.sum(dim=1).tolist()}")
    if not torch.all(masks.sum(dim=0) == 2):
        raise RuntimeError("every low-overlap bank unit must belong to exactly two modes")
    return masks


def construct_model(
    config: ExperimentConfig,
    condition: str,
    seed: int = 0,
) -> BaseLM:
    if condition == "looped_dense_roundaware":
        model = LoopedDenseRoundAwareLM(config)
    elif condition.startswith("recurrent_sparse_roundaware_"):
        suffix = condition.removeprefix("recurrent_sparse_roundaware_")
        if suffix == "learned":
            fixed = None
        elif suffix == "identical":
            fixed = identical_masks(config, seed)
        elif suffix == "random":
            fixed = random_masks(config, seed)
        elif suffix == "low_overlap":
            fixed = low_overlap_masks(config, seed)
        else:
            raise ValueError(condition)
        model = RecurrentSparseLM(config, round_aware=True, fixed_masks=fixed)
    else:
        raise ValueError(condition)
    initialize_model(model)
    return model


def logical_macs_per_sequence(config: ExperimentConfig, condition: str) -> int:
    t = config.context_length
    d = config.d_model
    f = config.ffn_hidden
    n = config.n_layers_or_rounds
    # Approximate multiply-accumulate count. QKV+O plus QK/AV plus SwiGLU.
    attention = 4 * t * d * d + 2 * t * t * d
    swiglu = 3 * t * d * f
    per = attention + swiglu
    if condition.startswith("recurrent_sparse"):
        router = t * (
            d * config.router_hidden + config.router_hidden * config.n_modes
        )
        per += router
    # Tied embeddings save parameters but the vocabulary projection is still
    # executed for next-token logits in every condition.
    lm_head = t * d * config.vocab_size
    return n * per + lm_head


def router_tau(config: ExperimentConfig, step: int) -> float:
    p = min(max(step / config.train_steps, 0.0), 1.0)
    return config.router_tau_start * (1 - p) + config.router_tau_end * p


def learning_rate(config: ExperimentConfig, step: int) -> float:
    if step <= config.warmup_steps:
        return config.learning_rate * step / max(config.warmup_steps, 1)
    p = (step - config.warmup_steps) / max(config.train_steps - config.warmup_steps, 1)
    p = min(max(p, 0.0), 1.0)
    min_lr = config.learning_rate * config.min_lr_ratio
    return min_lr + 0.5 * (config.learning_rate - min_lr) * (1 + math.cos(math.pi * p))


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


# ============================================================================
# Checkpointing
# ============================================================================


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    condition: str,
    seed: int,
    step: int,
    best_val: float,
    best_step: int,
) -> None:
    payload = {
        "version": VERSION,
        "signature": config.signature(condition, seed),
        "condition": condition,
        "seed": seed,
        "step": step,
        "best_val": best_val,
        "best_step": best_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def restore_rng(checkpoint: dict[str, Any]) -> None:
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_states"):
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng_states"]])


# ============================================================================
# Evaluation / route diagnostics
# ============================================================================


@torch.no_grad()
def evaluate(
    model: BaseLM,
    store: TokenStore,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
    rounds: int | None = None,
    collect_routes: bool = False,
) -> dict[str, Any]:
    model.eval()
    losses = []
    route_usage = None
    route_by_round = None
    transitions = None
    entropy_rows: list[float] = []
    balance_rows: list[float] = []

    if collect_routes and isinstance(model, RecurrentSparseLM):
        r = config.n_layers_or_rounds if rounds is None else rounds
        route_usage = np.zeros(config.n_modes, dtype=np.int64)
        route_by_round = np.zeros((r, config.n_modes), dtype=np.int64)
        transitions = np.zeros((config.n_modes, config.n_modes), dtype=np.int64)

    for i in range(config.eval_batches):
        x, y = store.batch(
            config.micro_batch,
            config.context_length,
            seed=9_000_000 + seed,
            step=i,
            microstep=0,
            device=device,
        )
        with autocast_context(device, config.precision):
            logits, diag = model(
                x,
                stochastic=False,
                # Scientific evaluation uses dense hard masks so BF16 kernel
                # rounding cannot choose a different later computation graph.
                sparse_eval=False,
                rounds=rounds,
                collect_routes=collect_routes,
            )
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(float(loss))

        if route_usage is not None:
            routes = diag["route_ids"]
            for rr, ids in enumerate(routes):
                flat = ids.cpu().numpy().reshape(-1)
                route_usage += np.bincount(flat, minlength=config.n_modes)
                route_by_round[rr] += np.bincount(flat, minlength=config.n_modes)
                if rr + 1 < len(routes):
                    nxt = routes[rr + 1].cpu().numpy().reshape(-1)
                    np.add.at(transitions, (flat, nxt), 1)
            entropy_rows.extend(diag["router_entropy"])
            balance_rows.extend(diag["router_balance"])

    mean_loss = float(np.mean(losses))
    result: dict[str, Any] = {
        "loss": mean_loss,
        "perplexity": float(math.exp(min(mean_loss, 20.0))),
    }
    if route_usage is not None:
        usage = route_usage / max(route_usage.sum(), 1)
        round_usage = route_by_round / np.maximum(route_by_round.sum(axis=1, keepdims=True), 1)
        joint = route_by_round / max(route_by_round.sum(), 1)
        p_r = joint.sum(axis=1, keepdims=True)
        p_m = joint.sum(axis=0, keepdims=True)
        expected = p_r @ p_m
        nz = joint > 0
        mi = float(np.sum(joint[nz] * np.log(joint[nz] / expected[nz])))
        usage_entropy = float(-np.sum(usage[usage > 0] * np.log(usage[usage > 0])))
        within_round_effective = []
        for row in round_usage:
            nz_row = row > 0
            ent = float(-np.sum(row[nz_row] * np.log(row[nz_row])))
            within_round_effective.append(math.exp(ent))
        result["routing"] = {
            "usage": usage.tolist(),
            "effective_modes": float(math.exp(usage_entropy)),
            "usage_by_round": round_usage.tolist(),
            "mean_within_round_effective_modes": float(np.mean(within_round_effective)),
            "min_within_round_effective_modes": float(np.min(within_round_effective)),
            "round_mode_mutual_information_nats": mi,
            "round_mode_mi_fraction_of_mode_entropy": float(mi / max(usage_entropy, 1.0e-12)),
            "transition_matrix": (
                transitions / np.maximum(transitions.sum(axis=1, keepdims=True), 1)
            ).tolist(),
            "mean_router_entropy": float(np.mean(entropy_rows)),
            "mean_router_balance": float(np.mean(balance_rows)),
            "topology": model.block.ffn.topology_stats(),
        }
    return result


@torch.no_grad()
def benchmark_tokens_per_second(
    model: BaseLM,
    store: TokenStore,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> float:
    model.eval()
    if device.type != "cuda":
        return math.nan
    x, _ = store.batch(
        config.micro_batch,
        config.context_length,
        seed=7_777_777 + seed,
        step=0,
        microstep=0,
        device=device,
    )
    for _ in range(3):
        with autocast_context(device, config.precision):
            model(x, stochastic=False, sparse_eval=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    iters = 10
    for _ in range(iters):
        with autocast_context(device, config.precision):
            model(x, stochastic=False, sparse_eval=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return float(iters * x.numel() / elapsed)


# ============================================================================
# Training
# ============================================================================


def train_one(
    config: ExperimentConfig,
    condition: str,
    seed: int,
    device: torch.device,
    smoke: bool = False,
) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(condition)

    seed_all(seed)
    root = Path(config.output_root) / f"seed_{seed}" / condition
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "result.json"
    if result_path.exists() and not smoke:
        print(f"Already complete: {result_path}")
        return result_path

    train_store = TokenStore(Path(config.data_dir) / "train.bin")
    val_store = TokenStore(Path(config.data_dir) / "val.bin")

    model = construct_model(config, condition, seed=seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        fused=(device.type == "cuda"),
    )

    latest = root / "latest.pt"
    best = root / "best.pt"
    start_step = 0
    best_val = math.inf
    best_step = 0
    if latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        if ckpt.get("signature") != config.signature(condition, seed):
            raise RuntimeError(f"Checkpoint signature mismatch: {latest}")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
        best_val = float(ckpt["best_val"])
        best_step = int(ckpt["best_step"])
        restore_rng(ckpt)
        print(f"Resuming {condition} seed={seed} from step {start_step}")

    steps = 20 if smoke else config.train_steps
    validate_every = 10 if smoke else config.validate_every
    checkpoint_every = 10 if smoke else config.checkpoint_every
    curve = []
    start_time = time.time()

    model.train()
    for step in range(start_step + 1, steps + 1):
        lr = learning_rate(config, step if not smoke else min(step, config.train_steps))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_task = 0.0
        train_aux = 0.0
        tau = router_tau(config, step)

        for micro in range(config.grad_accum):
            x, y = train_store.batch(
                config.micro_batch,
                config.context_length,
                seed=seed,
                step=step,
                microstep=micro,
                device=device,
            )
            with autocast_context(device, config.precision):
                logits, diag = model(
                    x,
                    tau=tau,
                    stochastic=isinstance(model, RecurrentSparseLM),
                    sparse_eval=False,
                )
                task_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
                )
                aux = diag["auxiliary_loss"]
                loss = task_loss + aux
                scaled = loss / config.grad_accum
            scaled.backward()
            train_loss += float(loss.detach()) / config.grad_accum
            train_task += float(task_loss.detach()) / config.grad_accum
            train_aux += float(aux.detach()) / config.grad_accum

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip))
        optimizer.step()

        should_validate = step == 1 or step % validate_every == 0 or step == steps
        if should_validate:
            val = evaluate(model, val_store, config, device, seed, collect_routes=False)
            row = {
                "step": step,
                "train_loss": train_loss,
                "train_task": train_task,
                "train_aux": train_aux,
                "val_loss": val["loss"],
                "val_ppl": val["perplexity"],
                "lr": lr,
                "tau": tau,
                "grad_norm": grad_norm,
            }
            curve.append(row)
            if val["loss"] < best_val:
                best_val = val["loss"]
                best_step = step
                save_checkpoint(
                    best, model, optimizer, config, condition, seed, step, best_val, best_step
                )
            print(
                f"seed={seed} condition={condition:16s} step={step:4d}/{steps} "
                f"train={train_task:.4f} aux={train_aux:.4f} "
                f"val={val['loss']:.4f} ppl={val['perplexity']:.2f} "
                f"best={best_val:.4f}@{best_step}",
                flush=True,
            )

        if step % checkpoint_every == 0 or step == steps:
            save_checkpoint(
                latest, model, optimizer, config, condition, seed, step, best_val, best_step
            )

    # Evaluate best checkpoint, not last checkpoint.
    ckpt = torch.load(best, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    final = evaluate(
        model,
        val_store,
        config,
        device,
        seed,
        collect_routes=isinstance(model, RecurrentSparseLM),
    )
    depth_sweep = {}
    if isinstance(model, (LoopedDenseRoundAwareLM, RecurrentSparseLM)):
        for rounds in (4, 6, 8, 10, 12):
            depth_sweep[str(rounds)] = evaluate(
                model,
                val_store,
                config,
                device,
                seed,
                rounds=rounds,
                collect_routes=False,
            )
    throughput = benchmark_tokens_per_second(model, val_store, config, device, seed)

    result = {
        "version": VERSION,
        "condition": condition,
        "seed": seed,
        "config": asdict(config),
        "derived": {
            "parameters": parameter_count(model),
            "logical_macs_per_context": logical_macs_per_sequence(config, condition),
            "logical_macs_per_token": logical_macs_per_sequence(config, condition)
            / config.context_length,
            "external_training_tokens": config.train_tokens if not smoke else steps * config.tokens_per_update,
            "note": (
                "Sparse learned-mask conditions train through the full bank using a straight-through "
                "mask surrogate; fixed-mask controls retain dense training execution for a matched code path. "
                "Physical training FLOPs exceed logical sparse inference FLOPs."
            ),
        },
        "training": {
            "best_val_loss": best_val,
            "best_step": best_step,
            "elapsed_seconds": time.time() - start_time,
            "curve": curve,
        },
        "validation": final,
        "round_sweep": depth_sweep,
        "tokens_per_second_sparse_or_dense_eval": throughput,
    }
    atomic_json(result, result_path)
    print(f"Saved: {result_path}")
    return result_path


# ============================================================================
# Preflight / aggregation
# ============================================================================


def preflight(config: ExperimentConfig, device: torch.device) -> None:
    print("=" * 108)
    print("V6.1 ROUND-AWARE + TOPOLOGY-CONTROL PREFLIGHT")
    print("=" * 108)
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        print("bf16:", torch.cuda.is_bf16_supported())
    print("context:", config.context_length)
    print("train tokens:", f"{config.train_tokens:,}")
    print("tokens/update:", f"{config.tokens_per_update:,}")

    for condition in CONDITIONS:
        seed_all(123)
        model = construct_model(config, condition, seed=123).to(device)
        x = torch.randint(0, config.vocab_size, (2, 32), device=device)
        with autocast_context(device, config.precision):
            logits, diag = model(
                x,
                tau=1.0,
                stochastic=isinstance(model, RecurrentSparseLM),
                sparse_eval=False,
            )
            loss = logits.float().square().mean() * 1e-5 + diag["auxiliary_loss"]
        loss.backward()
        print(
            f"{condition:42s} params={parameter_count(model):,} "
            f"logical_MAC/context={logical_macs_per_sequence(config, condition):,}"
        )
        if isinstance(model, RecurrentSparseLM):
            topo = model.block.ffn.topology_stats()
            print(
                "  topology "
                f"jaccard={topo['mean_pairwise_jaccard']:.4f} "
                f"overlap/active={topo['mean_pairwise_intersection_over_active']:.4f} "
                f"union={topo['bank_union_fraction']:.4f}"
            )

    # True fixed-route sparse-kernel equivalence test.
    seed_all(456)
    sparse = construct_model(
        config, "recurrent_sparse_roundaware_learned", seed=456
    ).to(device).eval()
    assert isinstance(sparse, RecurrentSparseLM)
    n = 8 * config.n_modes
    ffn_x = torch.randn(n, config.d_model, device=device)
    route_ids = torch.arange(n, device=device) % config.n_modes
    route_weights = F.one_hot(route_ids, config.n_modes).float()
    with torch.no_grad(), autocast_context(device, config.precision):
        dense_ffn = sparse.block.ffn.forward_dense_train(ffn_x, route_weights)
        sparse_ffn = sparse.block.ffn.forward_sparse_eval(ffn_x, route_ids)
    diff = float((dense_ffn.float() - sparse_ffn.float()).abs().max())
    print(f"sparse fixed-route kernel max diff: {diff:.3e}")
    if diff > 5.0e-2:
        raise RuntimeError(f"Sparse fixed-route kernel mismatch is too large: {diff:.3e}")

    # Fail closed on the preregistered topology controls.
    expected = {
        "identical": (1.0, 0.25),
        "low_overlap": (146.28571428571428 / (2048 - 146.28571428571428), 1.0),
    }
    for suffix, (target_j, target_union) in expected.items():
        m = construct_model(config, f"recurrent_sparse_roundaware_{suffix}", seed=7)
        topo = m.block.ffn.topology_stats()
        if abs(topo["mean_pairwise_jaccard"] - target_j) > 2e-3:
            raise RuntimeError(f"{suffix} topology Jaccard mismatch: {topo}")
        if abs(topo["bank_union_fraction"] - target_union) > 2e-3:
            raise RuntimeError(f"{suffix} topology union mismatch: {topo}")

    print("V6.1 PREFLIGHT PASSED")


def _routing_mi_fraction(r: dict[str, Any]) -> float:
    if "round_mode_mi_fraction_of_mode_entropy" in r:
        return float(r["round_mode_mi_fraction_of_mode_entropy"])
    entropy = math.log(max(float(r["effective_modes"]), 1.0))
    return float(
        r["round_mode_mutual_information_nats"]
        / max(entropy, 1.0e-12)
    )


def _routing_within_round_modes(r: dict[str, Any]) -> float:
    if "mean_within_round_effective_modes" in r:
        return float(r["mean_within_round_effective_modes"])

    rows = np.asarray(r["usage_by_round"], dtype=np.float64)
    values = []

    for row in rows:
        total = row.sum()
        if total <= 0:
            continue
        row = row / total
        nz = row > 0
        entropy = -float(np.sum(row[nz] * np.log(row[nz])))
        values.append(math.exp(entropy))

    return float(np.mean(values)) if values else float("nan")


def aggregate(
    config: ExperimentConfig,
    seeds: list[int],
    conditions: list[str],
    source_v60_root: str,
) -> Path:
    root = Path(config.output_root)
    source_root = Path(source_v60_root)
    raw: dict[str, dict[str, Any]] = {}
    missing = []
    reference_conditions = ["dense_stack", "looped_dense", "recurrent_sparse"]

    for seed in seeds:
        raw[str(seed)] = {}
        for condition in reference_conditions:
            path = source_root / f"seed_{seed}" / condition / "result.json"
            if path.exists():
                raw[str(seed)][condition] = json.loads(path.read_text())
            else:
                # References are optional for seeds not yet run in V6.0.
                pass
        for condition in conditions:
            path = root / f"seed_{seed}" / condition / "result.json"
            if not path.exists():
                missing.append(str(path))
            else:
                raw[str(seed)][condition] = json.loads(path.read_text())
    if missing:
        raise RuntimeError("Missing V6.1 result files:\n" + "\n".join(missing))

    available = []
    for c in reference_conditions + conditions:
        if all(c in raw[str(s)] for s in seeds):
            available.append(c)

    summary: dict[str, Any] = {
        "seeds": seeds,
        "conditions": available,
        "metrics": {},
        "paired": {},
        "routing": {},
    }
    for condition in available:
        summary["metrics"][condition] = {
            "val_loss": stats(raw[str(s)][condition]["validation"]["loss"] for s in seeds),
            "perplexity": stats(raw[str(s)][condition]["validation"]["perplexity"] for s in seeds),
            "parameters": stats(raw[str(s)][condition]["derived"]["parameters"] for s in seeds),
            "logical_macs_per_token": stats(
                raw[str(s)][condition]["derived"]["logical_macs_per_token"] for s in seeds
            ),
            "tokens_per_second": stats(
                raw[str(s)][condition]["tokens_per_second_sparse_or_dense_eval"] for s in seeds
            ),
        }
        if condition.startswith("recurrent_sparse"):
            rr = [raw[str(s)][condition]["validation"].get("routing") for s in seeds]
            rr = [r for r in rr if r]
            if rr:
                summary["routing"][condition] = {
                    "effective_modes": stats(r["effective_modes"] for r in rr),
                    "round_mode_mi": stats(r["round_mode_mutual_information_nats"] for r in rr),
                    "round_mi_fraction": stats(_routing_mi_fraction(r) for r in rr),
                    "within_round_modes": stats(_routing_within_round_modes(r) for r in rr),
                    "jaccard": stats(r["topology"]["mean_pairwise_jaccard"] for r in rr),
                    "overlap_over_active": stats(
                        r["topology"]["mean_pairwise_intersection_over_active"] for r in rr
                    ),
                    "bank_union": stats(r["topology"]["bank_union_fraction"] for r in rr),
                }

    comparisons = [
        ("looped_dense_roundaware", "looped_dense", "roundaware_looped_minus_looped"),
        (
            "recurrent_sparse_roundaware_learned",
            "recurrent_sparse",
            "roundaware_sparse_minus_v60_sparse",
        ),
        (
            "recurrent_sparse_roundaware_learned",
            "looped_dense_roundaware",
            "roundaware_sparse_minus_roundaware_looped",
        ),
    ]
    for alt, base, label in comparisons:
        if alt in available and base in available:
            summary["paired"][label] = {
                "val_loss": paired_stats(
                    [raw[str(s)][alt]["validation"]["loss"] for s in seeds],
                    [raw[str(s)][base]["validation"]["loss"] for s in seeds],
                )
            }
    learned = "recurrent_sparse_roundaware_learned"
    if learned in available:
        for suffix in ("identical", "random", "low_overlap"):
            fixed = f"recurrent_sparse_roundaware_{suffix}"
            if fixed in available:
                summary["paired"][f"{fixed}_minus_learned"] = {
                    "val_loss": paired_stats(
                        [raw[str(s)][fixed]["validation"]["loss"] for s in seeds],
                        [raw[str(s)][learned]["validation"]["loss"] for s in seeds],
                    )
                }

    out = root / "aggregate_results.json"
    atomic_json({"version": VERSION, "raw": raw, "summary": summary}, out)

    print("=" * 132)
    print("V6.1 ROUND-AWARE RECURRENCE + TOPOLOGY-CONTROL SUMMARY")
    print("=" * 132)
    print(f"{'condition':44s} {'params':>11s} {'val loss':>10s} {'ppl':>10s} {'MAC/tok':>13s} {'tok/s':>11s}")
    for c in available:
        m = summary["metrics"][c]
        print(
            f"{c:44s} {m['parameters']['mean']:11.0f} {m['val_loss']['mean']:10.4f} "
            f"{m['perplexity']['mean']:10.2f} {m['logical_macs_per_token']['mean']:13.0f} "
            f"{m['tokens_per_second']['mean']:11.0f}"
        )
    print("\nPAIRED VAL-LOSS DIFFERENCES (negative favors first-named condition)")
    for name, payload in summary["paired"].items():
        p = payload["val_loss"]
        print(
            f"{name:62s} {p['mean']:+.4f} "
            f"95%CI [{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}]"
        )
    if summary["routing"]:
        print("\nROUTING / TOPOLOGY")
        for c, r in summary["routing"].items():
            print(
                f"{c:44s} modes={r['effective_modes']['mean']:.3f} "
                f"withinRound={r['within_round_modes']['mean']:.3f} "
                f"roundMI={r['round_mode_mi']['mean']:.4f} "
                f"MIfrac={r['round_mi_fraction']['mean']:.3f} "
                f"J={r['jaccard']['mean']:.4f} "
                f"overlap={r['overlap_over_active']['mean']:.4f} "
                f"union={r['bank_union']['mean']:.4f}"
            )
    print("Saved:", out)
    return out


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        sp.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
        sp.add_argument("--data-dir", default="data/v6_fineweb_edu")
        sp.add_argument("--output-root", default="runs/v6_1_roundaware_topology")
        sp.add_argument("--train-steps", type=int)
        sp.add_argument("--micro-batch", type=int)
        sp.add_argument("--grad-accum", type=int)

    sp = sub.add_parser("preflight")
    common(sp)

    sp = sub.add_parser("train")
    common(sp)
    sp.add_argument("--condition", choices=CONDITIONS, required=True)
    sp.add_argument("--seed", type=int, required=True)
    sp.add_argument("--smoke", action="store_true")

    sp = sub.add_parser("aggregate")
    common(sp)
    sp.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    sp.add_argument("--conditions", default=",".join(CONDITIONS))
    sp.add_argument("--source-v60-root", default="runs/v6_0_latent_mode_lm")

    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    values = asdict(ExperimentConfig())
    for key in ("precision", "data_dir", "output_root", "train_steps", "micro_batch", "grad_accum"):
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    return ExperimentConfig(**values)


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    if args.command == "preflight":
        preflight(config, device)
    elif args.command == "train":
        train_one(config, args.condition, args.seed, device, smoke=args.smoke)
    elif args.command == "aggregate":
        seeds = [int(x) for x in args.seeds.split(",") if x]
        conditions = [x for x in args.conditions.split(",") if x]
        aggregate(config, seeds, conditions, args.source_v60_root)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
