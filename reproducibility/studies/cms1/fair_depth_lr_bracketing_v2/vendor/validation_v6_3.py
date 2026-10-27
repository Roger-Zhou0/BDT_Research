#!/usr/bin/env python3
"""
V6.3 — attention-role decomposition for latent recurrent language modeling.

This experiment keeps the successful V6.1 fixed low-overlap sparse FFN
unchanged and varies only how self-attention is specialized across recurrent
rounds. It addresses the main remaining structural difference between the best
recurrent model and the fixed-stack Transformer: all recurrent rounds still
share one attention operator.

New attention conditions:
  recurrent_sparse_roundaware_attn_round_specific
      Eight full attention modules, one per recurrent round. Diagnostic upper
      bound: active attention MACs are unchanged, but stored attention
      parameters are no longer tied across depth.

  recurrent_sparse_roundaware_attn_round_lora
      One shared attention module plus small rank-r QKV/O adapters indexed by
      recurrent round. This tests whether a small amount of depth-specific
      modulation recovers the tied-attention penalty.

  recurrent_sparse_roundaware_attn_latent_lora
      One shared attention module plus the same low-rank adapter bank, but a
      learned state-dependent router selects the adapter independently at each
      token/round. There are no operation labels or routing targets.

All three conditions retain:
  - the same FineWeb-Edu token stream,
  - the same 8 recurrent rounds and round embeddings,
  - the same fixed low-overlap FFN bank topology,
  - the same learned state-dependent FFN router,
  - the same 98.3M-token external training budget.

Scientific evaluation uses dense hard routing; sparse/indexed adapter execution
is used only for throughput benchmarking. This prevents BF16 kernel rounding
from changing the recurrent computation graph during quality evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import validation_v6_1 as v61

VERSION = "v6_3_attention_role_decomposition_v1"
NEW_CONDITIONS = (
    "recurrent_sparse_roundaware_attn_round_specific",
    "recurrent_sparse_roundaware_attn_round_lora",
    "recurrent_sparse_roundaware_attn_latent_lora",
)
TRAIN_CONDITIONS = ("dense_stack",) + NEW_CONDITIONS
DEFAULT_SEEDS = (41, 42, 43, 44, 45, 46, 47)

# Preserve tested V6.1 functions before patching dynamic globals.
_BASE_LOGICAL_MACS = v61.logical_macs_per_sequence
_BASE_EVALUATE = v61.evaluate


@dataclass
class ExperimentConfig(v61.ExperimentConfig):
    attention_adapter_rank: int = 16
    attention_adapter_modes: int = 8
    attention_router_hidden: int = 128
    attention_router_balance_weight: float = 0.01
    attention_router_z_weight: float = 1.0e-4
    attention_adapter_a_std: float = 0.02
    attention_adapter_b_std: float = 0.002
    output_root: str = "runs/v6_3_attention_roles"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attention_adapter_modes != self.n_layers_or_rounds:
            raise ValueError(
                "V6.3 preregisters one attention adapter mode per nominal recurrent round"
            )
        if self.attention_adapter_rank <= 0:
            raise ValueError("attention_adapter_rank must be positive")

    def signature(self, condition: str, seed: int) -> str:
        payload = {
            "version": VERSION,
            "condition": condition,
            "seed": seed,
            "config": asdict(self),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ============================================================================
# Attention helpers
# ============================================================================


def _router_aux(
    logits: torch.Tensor,
    route_ids: torch.Tensor,
    n_modes: int,
    balance_weight: float,
    z_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    importance = probs.mean(dim=(0, 1))
    load = F.one_hot(route_ids, n_modes).float().mean(dim=(0, 1))
    balance = n_modes * torch.sum(importance * load.detach())
    z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2)
    entropy = -torch.sum(probs * probs.clamp_min(1e-9).log(), dim=-1).mean()
    aux = balance_weight * (balance - 1.0) + z_weight * z_loss
    return aux, probs, balance.detach(), entropy.detach()


class LowRankAdapterBank(nn.Module):
    """Mode-indexed low-rank linear updates.

    A: [M,R,Din], B: [M,Dout,R]. Dense hard routing is used for scientific
    training/evaluation so the straight-through router receives gradients.
    Indexed execution is available for throughput benchmarking.
    """

    def __init__(
        self,
        n_modes: int,
        in_dim: int,
        out_dim: int,
        rank: int,
        a_std: float,
        b_std: float,
    ):
        super().__init__()
        self.n_modes = n_modes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.a = nn.Parameter(torch.empty(n_modes, rank, in_dim))
        self.b = nn.Parameter(torch.empty(n_modes, out_dim, rank))
        nn.init.normal_(self.a, mean=0.0, std=a_std)
        nn.init.normal_(self.b, mean=0.0, std=b_std)

    def forward_dense(
        self, x: torch.Tensor, route_weights: torch.Tensor
    ) -> torch.Tensor:
        # x [B,T,D], route_weights [B,T,M]
        low = torch.einsum("btd,mrd->btmr", x, self.a)
        all_out = torch.einsum("btmr,mor->btmo", low, self.b)
        return torch.einsum(
            "btm,btmo->bto", route_weights.to(dtype=all_out.dtype), all_out
        )

    def forward_sparse(self, x: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        flat_x = x.reshape(-1, x.shape[-1])
        flat_ids = route_ids.reshape(-1)
        result = torch.zeros(
            flat_x.shape[0], self.out_dim, device=x.device, dtype=x.dtype
        )
        for mode in range(self.n_modes):
            rows = torch.nonzero(flat_ids == mode, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            xr = flat_x.index_select(0, rows)
            low = F.linear(xr, self.a[mode])
            yr = F.linear(low, self.b[mode])
            result.index_copy_(0, rows, yr.to(result.dtype))
        return result.view(b, t, self.out_dim)


class LoRAAttention(nn.Module):
    """Shared attention plus either round-indexed or latent low-rank adapters."""

    def __init__(self, config: ExperimentConfig, latent: bool):
        super().__init__()
        self.config = config
        self.latent = latent
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_modes = config.attention_adapter_modes

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = v61.RotaryEmbedding(self.head_dim, config.context_length)
        self.qkv_adapters = LowRankAdapterBank(
            self.n_modes,
            config.d_model,
            3 * config.d_model,
            config.attention_adapter_rank,
            config.attention_adapter_a_std,
            config.attention_adapter_b_std,
        )
        self.out_adapters = LowRankAdapterBank(
            self.n_modes,
            config.d_model,
            config.d_model,
            config.attention_adapter_rank,
            config.attention_adapter_a_std,
            config.attention_adapter_b_std,
        )

        if latent:
            self.router_in = nn.Linear(
                config.d_model, config.attention_router_hidden, bias=False
            )
            self.router_out = nn.Linear(
                config.attention_router_hidden, self.n_modes, bias=False
            )
        else:
            self.router_in = None
            self.router_out = None

    def _routes(
        self,
        x: torch.Tensor,
        round_index: int,
        tau: float,
        stochastic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        b, t, _ = x.shape
        if not self.latent:
            route_ids = torch.full(
                (b, t), round_index % self.n_modes, device=x.device, dtype=torch.long
            )
            route_weights = F.one_hot(route_ids, self.n_modes).float()
            return route_weights, route_ids, {
                "auxiliary_loss": x.new_zeros(()),
                "entropy": x.new_zeros(()),
                "balance": x.new_ones(()),
            }

        logits = self.router_out(F.silu(self.router_in(x))).float()
        if stochastic:
            route_weights = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        else:
            route_ids = logits.argmax(dim=-1)
            route_weights = F.one_hot(route_ids, self.n_modes).float()
        route_ids = route_weights.argmax(dim=-1)
        aux, _probs, balance, entropy = _router_aux(
            logits,
            route_ids,
            self.n_modes,
            self.config.attention_router_balance_weight,
            self.config.attention_router_z_weight,
        )
        return route_weights, route_ids, {
            "auxiliary_loss": aux,
            "entropy": entropy,
            "balance": balance,
        }

    def forward(
        self,
        x: torch.Tensor,
        *,
        round_index: int,
        tau: float,
        stochastic: bool,
        sparse_eval: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, t, d = x.shape
        route_weights, route_ids, diag = self._routes(
            x, round_index=round_index, tau=tau, stochastic=stochastic
        )

        if sparse_eval:
            qkv_delta = self.qkv_adapters.forward_sparse(x, route_ids)
        else:
            qkv_delta = self.qkv_adapters.forward_dense(x, route_weights)
        qkv = self.qkv(x) + qkv_delta
        qkv = qkv.view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.rope.apply(q, k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, d)

        base_out = self.out(y)
        if sparse_eval:
            out_delta = self.out_adapters.forward_sparse(y, route_ids)
        else:
            out_delta = self.out_adapters.forward_dense(y, route_weights)
        return base_out + out_delta, {
            **diag,
            "route_ids": route_ids,
        }


class AttentionRoleSparseBlock(nn.Module):
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        seed: int,
        attention_kind: str,
    ):
        super().__init__()
        self.config = config
        self.attention_kind = attention_kind
        self.norm1 = v61.RMSNorm(config.d_model)
        self.norm2 = v61.RMSNorm(config.d_model)

        if attention_kind == "round_specific":
            self.attn_rounds = nn.ModuleList(
                [v61.CausalSelfAttention(config) for _ in range(config.n_layers_or_rounds)]
            )
            self.attn = None
        elif attention_kind == "round_lora":
            self.attn_rounds = None
            self.attn = LoRAAttention(config, latent=False)
        elif attention_kind == "latent_lora":
            self.attn_rounds = None
            self.attn = LoRAAttention(config, latent=True)
        else:
            raise ValueError(attention_kind)

        self.router_in = nn.Linear(config.d_model, config.router_hidden, bias=False)
        self.router_out = nn.Linear(config.router_hidden, config.n_modes, bias=False)
        self.ffn = v61.SharedSparseSwiGLU(
            config, fixed_masks=v61.low_overlap_masks(config, seed)
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        tau: float,
        stochastic: bool,
        sparse_eval: bool,
        round_embedding: torch.Tensor,
        round_index: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attn_input = self.norm1(x + round_embedding)
        if self.attention_kind == "round_specific":
            attn_out = self.attn_rounds[round_index](attn_input)
            attn_diag = {
                "auxiliary_loss": x.new_zeros(()),
                "entropy": x.new_zeros(()),
                "balance": x.new_ones(()),
                "route_ids": None,
            }
        else:
            attn_out, attn_diag = self.attn(
                attn_input,
                round_index=round_index,
                tau=tau,
                stochastic=stochastic,
                sparse_eval=sparse_eval,
            )
        x = x + attn_out

        y = self.norm2(x + round_embedding)
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

        ffn_aux, _ffn_probs, ffn_balance, ffn_entropy = _router_aux(
            logits,
            route_ids,
            self.config.n_modes,
            self.config.router_balance_weight,
            self.config.router_z_weight,
        )
        return x, {
            "auxiliary_loss": ffn_aux + attn_diag["auxiliary_loss"],
            "route_ids": route_ids,
            "entropy": ffn_entropy,
            "balance": ffn_balance,
            "attn_route_ids": attn_diag.get("route_ids"),
            "attn_entropy": attn_diag["entropy"],
            "attn_balance": attn_diag["balance"],
        }


class AttentionRoleSparseLM(v61.RecurrentSparseLM):
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        seed: int,
        attention_kind: str,
    ):
        # Build tested embedding/round-embedding structure, then replace only
        # the recurrent block. The replacement removes the old block from the
        # registered module tree, so there are no unused parameters.
        super().__init__(config, round_aware=True, fixed_masks=None)
        self.attention_kind = attention_kind
        self.block = AttentionRoleSparseBlock(
            config, seed=seed, attention_kind=attention_kind
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
        if n > self.config.max_recurrent_rounds:
            raise ValueError(f"rounds={n} exceeds max_recurrent_rounds")
        tau = self.config.router_tau_end if tau is None else tau
        aux = h.new_zeros(())
        route_ids = []
        entropies = []
        balances = []
        attn_route_ids = []
        attn_entropies = []
        attn_balances = []

        for rr in range(n):
            e = self.round_embedding.weight[rr][None, None, :]
            # The nominal model has eight distinct attention roles. For the
            # diagnostic >8-round sweep, cycle that learned role schedule rather
            # than introducing untrained attention parameters. Nominal 8-round
            # training/evaluation is unaffected.
            attn_rr = rr
            if self.attention_kind == "round_specific":
                attn_rr = rr % len(self.block.attn_rounds)
            h, diag = self.block(
                h,
                tau=tau,
                stochastic=stochastic,
                sparse_eval=sparse_eval,
                round_embedding=e,
                round_index=attn_rr,
            )
            aux = aux + diag["auxiliary_loss"]
            if collect_routes:
                route_ids.append(diag["route_ids"].detach())
                entropies.append(float(diag["entropy"]))
                balances.append(float(diag["balance"]))
                if diag["attn_route_ids"] is not None:
                    attn_route_ids.append(diag["attn_route_ids"].detach())
                    attn_entropies.append(float(diag["attn_entropy"]))
                    attn_balances.append(float(diag["attn_balance"]))

        out = {"auxiliary_loss": aux / n}
        if collect_routes:
            out.update(
                {
                    "route_ids": route_ids,
                    "router_entropy": entropies,
                    "router_balance": balances,
                }
            )
            if attn_route_ids:
                out.update(
                    {
                        "attn_route_ids": attn_route_ids,
                        "attn_router_entropy": attn_entropies,
                        "attn_router_balance": attn_balances,
                    }
                )
        return self.logits(h), out


# ============================================================================
# Construction / accounting
# ============================================================================


def construct_model(config: ExperimentConfig, condition: str, seed: int = 0):
    if condition == "dense_stack":
        model = v61.DenseStackLM(config)
        v61.initialize_model(model)
        return model

    mapping = {
        "recurrent_sparse_roundaware_attn_round_specific": "round_specific",
        "recurrent_sparse_roundaware_attn_round_lora": "round_lora",
        "recurrent_sparse_roundaware_attn_latent_lora": "latent_lora",
    }
    if condition not in mapping:
        raise ValueError(condition)
    model = AttentionRoleSparseLM(
        config, seed=seed, attention_kind=mapping[condition]
    )
    v61.initialize_model(model)

    # v61.initialize_model covers all Linear/Embedding modules and sparse FFN
    # weights. Adapter A/B matrices are direct Parameters and intentionally get
    # a much smaller B initialization so the model starts near shared attention
    # while still exposing task gradients to a latent attention router at step 1.
    if model.block.attn is not None:
        for bank in (model.block.attn.qkv_adapters, model.block.attn.out_adapters):
            nn.init.normal_(bank.a, mean=0.0, std=config.attention_adapter_a_std)
            nn.init.normal_(bank.b, mean=0.0, std=config.attention_adapter_b_std)
    return model


def logical_macs_per_sequence(config: ExperimentConfig, condition: str) -> int:
    if condition == "dense_stack":
        return _BASE_LOGICAL_MACS(config, "dense_stack")

    # Base fixed-low-overlap recurrent sparse model.
    base = _BASE_LOGICAL_MACS(config, "recurrent_sparse_roundaware_low_overlap")
    if condition == "recurrent_sparse_roundaware_attn_round_specific":
        # Unique stored attention parameters; active attention computation is
        # identical to the shared-attention reference.
        return base

    t = config.context_length
    d = config.d_model
    r = config.attention_adapter_rank
    n = config.n_layers_or_rounds
    # QKV LoRA: D->R->3D = 4DR. O LoRA: D->R->D = 2DR.
    adapter_per_round = t * 6 * d * r
    extra = n * adapter_per_round
    if condition == "recurrent_sparse_roundaware_attn_latent_lora":
        extra += n * t * (
            d * config.attention_router_hidden
            + config.attention_router_hidden * config.attention_adapter_modes
        )
    return base + extra


def _patch_v61_training_globals() -> None:
    v61.VERSION = VERSION
    v61.CONDITIONS = TRAIN_CONDITIONS
    v61.construct_model = construct_model
    v61.logical_macs_per_sequence = logical_macs_per_sequence
    v61.evaluate = evaluate


def train_one(
    config: ExperimentConfig,
    condition: str,
    seed: int,
    device: torch.device,
    smoke: bool = False,
):
    _patch_v61_training_globals()
    return v61.train_one(config, condition, seed, device, smoke=smoke)


# ============================================================================
# Routing diagnostics
# ============================================================================


def _summarize_routes(route_lists: list[torch.Tensor], n_modes: int) -> dict[str, Any]:
    r = len(route_lists)
    usage = np.zeros(n_modes, dtype=np.int64)
    by_round = np.zeros((r, n_modes), dtype=np.int64)
    transitions = np.zeros((n_modes, n_modes), dtype=np.int64)
    for rr, ids in enumerate(route_lists):
        flat = ids.cpu().numpy().reshape(-1)
        usage += np.bincount(flat, minlength=n_modes)
        by_round[rr] += np.bincount(flat, minlength=n_modes)
        if rr + 1 < r:
            nxt = route_lists[rr + 1].cpu().numpy().reshape(-1)
            np.add.at(transitions, (flat, nxt), 1)

    p = usage / max(usage.sum(), 1)
    usage_entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    round_usage = by_round / np.maximum(by_round.sum(axis=1, keepdims=True), 1)
    joint = by_round / max(by_round.sum(), 1)
    p_r = joint.sum(axis=1, keepdims=True)
    p_m = joint.sum(axis=0, keepdims=True)
    expected = p_r @ p_m
    nz = joint > 0
    mi = float(np.sum(joint[nz] * np.log(joint[nz] / expected[nz])))
    within = []
    for row in round_usage:
        nzr = row > 0
        ent = float(-np.sum(row[nzr] * np.log(row[nzr])))
        within.append(math.exp(ent))
    return {
        "usage": p.tolist(),
        "effective_modes": float(math.exp(usage_entropy)),
        "usage_by_round": round_usage.tolist(),
        "mean_within_round_effective_modes": float(np.mean(within)),
        "min_within_round_effective_modes": float(np.min(within)),
        "round_mode_mutual_information_nats": mi,
        "round_mode_mi_fraction_of_mode_entropy": float(
            mi / max(usage_entropy, 1.0e-12)
        ),
        "transition_matrix": (
            transitions / np.maximum(transitions.sum(axis=1, keepdims=True), 1)
        ).tolist(),
    }


def _joint_mode_mi(
    a_lists: list[torch.Tensor], b_lists: list[torch.Tensor], n_modes: int
) -> float:
    counts = np.zeros((n_modes, n_modes), dtype=np.int64)
    for aa, bb in zip(a_lists, b_lists):
        a = aa.cpu().numpy().reshape(-1)
        b = bb.cpu().numpy().reshape(-1)
        np.add.at(counts, (a, b), 1)
    joint = counts / max(counts.sum(), 1)
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    expected = pa @ pb
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log(joint[nz] / expected[nz])))


def evaluate(
    model: v61.BaseLM,
    store: v61.TokenStore,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
    rounds: int | None = None,
    collect_routes: bool = False,
) -> dict[str, Any]:
    # Non-V6.3 reference models can use the tested V6.1 evaluator directly.
    if not isinstance(model, AttentionRoleSparseLM):
        return _BASE_EVALUATE(
            model,
            store,
            config,
            device,
            seed,
            rounds=rounds,
            collect_routes=collect_routes,
        )

    model.eval()
    losses = []
    all_ffn_routes: list[list[torch.Tensor]] = []
    all_attn_routes: list[list[torch.Tensor]] = []
    entropy_rows: list[float] = []
    balance_rows: list[float] = []
    attn_entropy_rows: list[float] = []
    attn_balance_rows: list[float] = []

    for i in range(config.eval_batches):
        x, y = store.batch(
            config.micro_batch,
            config.context_length,
            seed=9_000_000 + seed,
            step=i,
            microstep=0,
            device=device,
        )
        with v61.autocast_context(device, config.precision):
            logits, diag = model(
                x,
                stochastic=False,
                sparse_eval=False,
                rounds=rounds,
                collect_routes=collect_routes,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
            )
        losses.append(float(loss.detach()))
        if collect_routes:
            all_ffn_routes.append(diag["route_ids"])
            entropy_rows.extend(diag["router_entropy"])
            balance_rows.extend(diag["router_balance"])
            if "attn_route_ids" in diag:
                all_attn_routes.append(diag["attn_route_ids"])
                attn_entropy_rows.extend(diag["attn_router_entropy"])
                attn_balance_rows.extend(diag["attn_router_balance"])

    mean_loss = float(np.mean(losses))
    result: dict[str, Any] = {
        "loss": mean_loss,
        "perplexity": float(math.exp(min(mean_loss, 20.0))),
    }

    if collect_routes:
        # Concatenate evaluation batches round-wise.
        n_rounds = len(all_ffn_routes[0])
        ffn_by_round = [
            torch.cat([batch[rr].reshape(-1) for batch in all_ffn_routes], dim=0)
            for rr in range(n_rounds)
        ]
        ffn_summary = _summarize_routes(ffn_by_round, config.n_modes)
        ffn_summary.update(
            {
                "mean_router_entropy": float(np.mean(entropy_rows)),
                "mean_router_balance": float(np.mean(balance_rows)),
                "topology": model.block.ffn.topology_stats(),
            }
        )
        result["routing"] = ffn_summary

        if all_attn_routes:
            attn_by_round = [
                torch.cat([batch[rr].reshape(-1) for batch in all_attn_routes], dim=0)
                for rr in range(n_rounds)
            ]
            attn_summary = _summarize_routes(
                attn_by_round, config.attention_adapter_modes
            )
            attn_summary.update(
                {
                    "mean_router_entropy": float(np.mean(attn_entropy_rows)),
                    "mean_router_balance": float(np.mean(attn_balance_rows)),
                    "attention_ffn_mutual_information_nats": _joint_mode_mi(
                        attn_by_round, ffn_by_round, config.n_modes
                    ),
                }
            )
            result["attention_routing"] = attn_summary
    return result


# ============================================================================
# Preflight
# ============================================================================


def preflight(config: ExperimentConfig, device: torch.device) -> None:
    print("=" * 124)
    print("V6.3 ATTENTION-ROLE DECOMPOSITION PREFLIGHT")
    print("=" * 124)
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        print("bf16:", torch.cuda.is_bf16_supported())
    print("attention adapter rank:", config.attention_adapter_rank)
    print("external train tokens:", f"{config.train_tokens:,}")

    for condition in TRAIN_CONDITIONS:
        v61.seed_all(123)
        model = construct_model(config, condition, seed=123).to(device)
        x = torch.randint(0, config.vocab_size, (2, 32), device=device)
        with v61.autocast_context(device, config.precision):
            logits, diag = model(
                x,
                tau=1.0,
                stochastic=isinstance(model, v61.RecurrentSparseLM),
                sparse_eval=False,
            )
            loss = logits.float().square().mean() * 1e-5 + diag["auxiliary_loss"]
        loss.backward()
        macs = logical_macs_per_sequence(config, condition)
        print(
            f"{condition:56s} params={v61.parameter_count(model):,} "
            f"MAC/context={macs:,} ratio_to_dense={macs / logical_macs_per_sequence(config, 'dense_stack'):.4f}x"
        )
        if isinstance(model, AttentionRoleSparseLM):
            topo = model.block.ffn.topology_stats()
            print(
                "  FFN topology "
                f"J={topo['mean_pairwise_jaccard']:.4f} "
                f"overlap={topo['mean_pairwise_intersection_over_active']:.4f} "
                f"union={topo['bank_union_fraction']:.4f}"
            )
            if abs(topo["bank_union_fraction"] - 1.0) > 1e-6:
                raise RuntimeError(f"FFN low-overlap union mismatch: {topo}")
            if abs(topo["mean_pairwise_jaccard"] - (1.0 / 13.0)) > 2e-3:
                raise RuntimeError(f"FFN low-overlap Jaccard mismatch: {topo}")

    # Round-specific full attention modules must really be distinct tensors.
    full = construct_model(
        config, "recurrent_sparse_roundaware_attn_round_specific", seed=7
    )
    ptrs = [m.qkv.weight.data_ptr() for m in full.block.attn_rounds]
    if len(set(ptrs)) != config.n_layers_or_rounds:
        raise RuntimeError("round-specific attention parameters are unexpectedly shared")

    # Fixed-route LoRA adapter kernel equivalence; this tests the adapter
    # implementation itself rather than allowing recurrent routers to bifurcate.
    lora = construct_model(
        config, "recurrent_sparse_roundaware_attn_latent_lora", seed=456
    ).to(device).eval()
    xx = torch.randn(2, 32, config.d_model, device=device)
    route_ids = torch.arange(64, device=device).view(2, 32) % config.attention_adapter_modes
    route_weights = F.one_hot(route_ids, config.attention_adapter_modes).float()
    with torch.no_grad(), v61.autocast_context(device, config.precision):
        dense_qkv = lora.block.attn.qkv_adapters.forward_dense(xx, route_weights)
        sparse_qkv = lora.block.attn.qkv_adapters.forward_sparse(xx, route_ids)
        dense_out = lora.block.attn.out_adapters.forward_dense(xx, route_weights)
        sparse_out = lora.block.attn.out_adapters.forward_sparse(xx, route_ids)
    qdiff = float((dense_qkv.float() - sparse_qkv.float()).abs().max())
    odiff = float((dense_out.float() - sparse_out.float()).abs().max())
    print(f"attention adapter fixed-route max diff: qkv={qdiff:.3e} out={odiff:.3e}")
    if max(qdiff, odiff) > 5e-2:
        raise RuntimeError("attention adapter indexed execution mismatch")

    print("V6.3 PREFLIGHT PASSED")


# ============================================================================
# Aggregation
# ============================================================================


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _stats(values: Iterable[float]) -> dict[str, float]:
    return v61.stats(values)


def _cohort_summary(
    raw: dict[str, dict[str, Any]], seeds: list[int], conditions: list[str]
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "seeds": seeds,
        "metrics": {},
        "ffn_routing": {},
        "attention_routing": {},
        "paired": {},
    }
    for c in conditions:
        if not all(c in raw[str(s)] for s in seeds):
            continue
        out["metrics"][c] = {
            "val_loss": _stats(raw[str(s)][c]["validation"]["loss"] for s in seeds),
            "perplexity": _stats(
                raw[str(s)][c]["validation"]["perplexity"] for s in seeds
            ),
            "parameters": _stats(raw[str(s)][c]["derived"]["parameters"] for s in seeds),
            "logical_macs_per_token": _stats(
                raw[str(s)][c]["derived"]["logical_macs_per_token"] for s in seeds
            ),
            "tokens_per_second": _stats(
                raw[str(s)][c]["tokens_per_second_sparse_or_dense_eval"] for s in seeds
            ),
        }
        if c.startswith("recurrent_sparse"):
            rr = [raw[str(s)][c]["validation"].get("routing") for s in seeds]
            rr = [r for r in rr if r]
            if rr:
                out["ffn_routing"][c] = {
                    "effective_modes": _stats(r["effective_modes"] for r in rr),
                    "within_round_modes": _stats(
                        r["mean_within_round_effective_modes"] for r in rr
                    ),
                    "mi_fraction": _stats(
                        r["round_mode_mi_fraction_of_mode_entropy"] for r in rr
                    ),
                    "jaccard": _stats(
                        r["topology"]["mean_pairwise_jaccard"] for r in rr
                    ),
                    "union": _stats(r["topology"]["bank_union_fraction"] for r in rr),
                }
            ar = [
                raw[str(s)][c]["validation"].get("attention_routing") for s in seeds
            ]
            ar = [r for r in ar if r]
            if ar:
                out["attention_routing"][c] = {
                    "effective_modes": _stats(r["effective_modes"] for r in ar),
                    "within_round_modes": _stats(
                        r["mean_within_round_effective_modes"] for r in ar
                    ),
                    "mi_fraction": _stats(
                        r["round_mode_mi_fraction_of_mode_entropy"] for r in ar
                    ),
                    "attention_ffn_mi": _stats(
                        r["attention_ffn_mutual_information_nats"] for r in ar
                    ),
                }

    comparisons = []
    current = "recurrent_sparse_roundaware_low_overlap"
    looped = "looped_dense_roundaware"
    dense = "dense_stack"
    for c in NEW_CONDITIONS:
        comparisons += [
            (c, current, f"{c}_minus_shared_attention_sparse"),
            (c, looped, f"{c}_minus_roundaware_looped"),
            (c, dense, f"{c}_minus_dense_stack"),
        ]
    comparisons += [
        (
            "recurrent_sparse_roundaware_attn_latent_lora",
            "recurrent_sparse_roundaware_attn_round_lora",
            "latent_lora_minus_round_lora",
        ),
        (
            "recurrent_sparse_roundaware_attn_round_lora",
            "recurrent_sparse_roundaware_attn_round_specific",
            "round_lora_minus_full_round_specific",
        ),
    ]
    for a, b, label in comparisons:
        if a in out["metrics"] and b in out["metrics"]:
            out["paired"][label] = v61.paired_stats(
                [raw[str(s)][a]["validation"]["loss"] for s in seeds],
                [raw[str(s)][b]["validation"]["loss"] for s in seeds],
            )
    return out


def aggregate(
    config: ExperimentConfig,
    seeds: list[int],
    source_v60_root: str,
    source_v61_root: str,
) -> Path:
    root = Path(config.output_root)
    src60 = Path(source_v60_root)
    src61 = Path(source_v61_root)
    references = (
        "dense_stack",
        "looped_dense_roundaware",
        "recurrent_sparse_roundaware_low_overlap",
    )
    conditions = list(references) + list(NEW_CONDITIONS)
    raw: dict[str, dict[str, Any]] = {}
    missing = []
    for seed in seeds:
        raw[str(seed)] = {}
        # Dense stack seed 41 can be reused from V6.0; newly confirmed dense
        # seeds live under the V6.3 output root.
        dense_new = root / f"seed_{seed}" / "dense_stack" / "result.json"
        dense_old = src60 / f"seed_{seed}" / "dense_stack" / "result.json"
        dense_path = dense_new if dense_new.exists() else dense_old
        if dense_path.exists():
            raw[str(seed)]["dense_stack"] = _load(dense_path)
        else:
            missing.append(str(dense_path))

        for c in references[1:]:
            p = src61 / f"seed_{seed}" / c / "result.json"
            if p.exists():
                raw[str(seed)][c] = _load(p)
            else:
                missing.append(str(p))
        for c in NEW_CONDITIONS:
            p = root / f"seed_{seed}" / c / "result.json"
            if p.exists():
                raw[str(seed)][c] = _load(p)
            else:
                missing.append(str(p))
    if missing:
        raise RuntimeError("Missing V6.3/reference results:\n" + "\n".join(missing))

    cohorts: dict[str, list[int]] = {}
    dev = [s for s in seeds if s in (41, 42, 43)]
    confirm = [s for s in seeds if s in (44, 45, 46, 47)]
    if dev:
        cohorts["development_41_43"] = dev
    if confirm:
        cohorts["confirmation_44_47"] = confirm
    cohorts["all_requested"] = seeds
    summary = {
        name: _cohort_summary(raw, cohort, conditions)
        for name, cohort in cohorts.items()
    }

    out = root / "aggregate_results.json"
    v61.atomic_json({"version": VERSION, "raw": raw, "summary": summary}, out)

    print("=" * 152)
    print("V6.3 ATTENTION-ROLE DECOMPOSITION SUMMARY")
    print("=" * 152)
    for name, ss in summary.items():
        print(f"\n[{name}] seeds={ss['seeds']}")
        print(
            f"{'condition':58s} {'loss':>9s} {'ppl':>9s} {'params':>11s} "
            f"{'MAC/tok':>11s} {'tok/s':>11s}"
        )
        for c, m in ss["metrics"].items():
            print(
                f"{c:58s} {m['val_loss']['mean']:9.4f} {m['perplexity']['mean']:9.2f} "
                f"{m['parameters']['mean']:11.0f} {m['logical_macs_per_token']['mean']:11.0f} "
                f"{m['tokens_per_second']['mean']:11.0f}"
            )
        print("  paired val-loss differences (negative favors first named):")
        for label, p in ss["paired"].items():
            print(
                f"    {label:96s} {p['mean']:+.4f} "
                f"95%CI [{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}]"
            )
        if ss["ffn_routing"]:
            print("  FFN latent routing:")
            for c, r in ss["ffn_routing"].items():
                if c not in NEW_CONDITIONS:
                    continue
                print(
                    f"    {c:56s} modes={r['effective_modes']['mean']:.3f} "
                    f"within={r['within_round_modes']['mean']:.3f} "
                    f"MIfrac={r['mi_fraction']['mean']:.3f} "
                    f"J={r['jaccard']['mean']:.4f} union={r['union']['mean']:.4f}"
                )
        if ss["attention_routing"]:
            print("  attention-adapter routing (round_lora is deliberately round-indexed):")
            for c, r in ss["attention_routing"].items():
                print(
                    f"    {c:56s} modes={r['effective_modes']['mean']:.3f} "
                    f"within={r['within_round_modes']['mean']:.3f} "
                    f"MIfrac={r['mi_fraction']['mean']:.3f} "
                    f"attn↔ffnMI={r['attention_ffn_mi']['mean']:.4f}"
                )
    print("Saved:", out)
    return out


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train", "aggregate"):
        sp = sub.add_parser(name)
        sp.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
        sp.add_argument(
            "--precision", choices=("bf16", "fp16", "fp32"), default="bf16"
        )
        sp.add_argument("--data-dir", default="data/v6_fineweb_edu")
        sp.add_argument("--output-root", default="runs/v6_3_attention_roles")
        sp.add_argument("--attention-adapter-rank", type=int, default=16)
    tr = sub.choices["train"]
    tr.add_argument("--condition", required=True, choices=TRAIN_CONDITIONS)
    tr.add_argument("--seed", type=int, required=True)
    tr.add_argument("--smoke", action="store_true")

    ag = sub.choices["aggregate"]
    ag.add_argument("--seeds", default="41,42,43,44,45,46,47")
    ag.add_argument("--source-v60-root", default="runs/v6_0_latent_mode_lm")
    ag.add_argument("--source-v61-root", default="runs/v6_1_roundaware_topology")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        precision=args.precision,
        data_dir=args.data_dir,
        output_root=args.output_root,
        attention_adapter_rank=args.attention_adapter_rank,
    )
    device = torch.device(args.device)
    if args.command == "preflight":
        preflight(cfg, device)
    elif args.command == "train":
        train_one(cfg, args.condition, args.seed, device, smoke=args.smoke)
    elif args.command == "aggregate":
        seeds = [int(x) for x in args.seeds.split(",") if x]
        aggregate(cfg, seeds, args.source_v60_root, args.source_v61_root)


if __name__ == "__main__":
    main()
