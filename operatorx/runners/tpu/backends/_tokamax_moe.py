"""MoE layer on Tokamax routing_moe: router -> routed experts (+ gated shared expert).

Depends only on jax + tokamax (nothing from OperatorX), so it also runs on its own:
    python -m operatorx.runners.tpu.backends._tokamax_moe   # Qwen3.5-397B-A17B vs Tokamax's reference
(Not `python _tokamax_moe.py`: that puts backends/ on sys.path and tokamax.py shadows tokamax.)
The leading "_" keeps the OperatorX runner from loading this module as a backend.
"""
from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.routing_moe.gmm import plan as gmm_plan_lib
from tokamax._src.ops.experimental.routing_moe.layer import modular
from tokamax._src.ops.experimental.routing_moe.test.helper.helper_inputs import sub_channel_quantize

_F32 = jnp.float32
_DTYPES = {"e4m3": jnp.float8_e4m3fn}


@dataclasses.dataclass(frozen=True)
class MoeConfig:
    hidden: int
    num_experts: int
    top_k: int
    inter: int
    renormalize: bool = True
    # One shared expert of width `inter`, folded into routing_moe as expert `num_experts`
    # (as Tokamax's fold_shared_expert); its weight is sigmoid(x @ w_shared_gate) or 1.
    shared: bool = False
    shared_gate: bool = False
    weight_dtype: str = "bf16"  # "bf16" | "e4m3"
    weight_block: int | None = None  # e4m3: K rows per fp32 weight scale (one scale per column)
    x_quant_block: int | None = None  # e4m3: block of the dynamic fp8 quantization of x / intermediate


def init_params(cfg: MoeConfig, key: jax.Array) -> dict:
    """Random weights (std 0.1, as Tokamax's test inputs) in routing_moe's layout:
    w1 [E', H, 2I] with [gate | up] columns, w2 [E', I, H]; E' = E (+1 for the folded shared expert)."""
    H, E, I = cfg.hidden, cfg.num_experts + cfg.shared, cfg.inter
    ks = jax.random.split(key, 4)
    rand = lambda k, shape: jax.random.normal(k, shape, jnp.bfloat16) / 10
    p = {"w_router": rand(ks[0], (H, cfg.num_experts)), "w1": rand(ks[1], (E, H, 2 * I)),
         "w2": rand(ks[2], (E, I, H))}
    if cfg.shared_gate:
        p["w_shared_gate"] = rand(ks[3], (H, 1))
    if cfg.weight_dtype != "bf16":
        for w in ("w1", "w2"):
            p[w], p[f"{w}_scale"] = sub_channel_quantize(p[w], _DTYPES[cfg.weight_dtype], cfg.weight_block)
    return p


def gmm_plan(cfg: MoeConfig) -> gmm_plan_lib.RoutingGmmPlan | None:
    """fp8 weights: also feed x and the intermediate to the MXU as fp8, quantized dynamically."""
    if cfg.weight_dtype == "bf16":
        return None
    return gmm_plan_lib.RoutingGmmPlan(matmul_dtype=_DTYPES[cfg.weight_dtype],
                                       x_quant_block_size=cfg.x_quant_block)


def route(logits: jax.Array, cfg: MoeConfig, forced_ids: jax.Array | None = None):
    """softmax (fp32) -> top_k (or forced ids, keeping the router's weights) -> renormalize."""
    s = jax.nn.softmax(logits.astype(_F32), axis=-1)
    if forced_ids is None:
        w, ids = jax.lax.top_k(s, cfg.top_k)
    else:
        ids = forced_ids
        w = jnp.take_along_axis(s, ids, axis=-1)
    if cfg.renormalize:
        w = w / w.sum(axis=-1, keepdims=True)
    return w, ids.astype(jnp.int32)


def forced_ids(dist, tokens: int, experts: int, top_k: int, seed: int) -> np.ndarray:
    """[tokens, top_k] expert ids with the requested load distribution (numpy port of
    OperatorX's runners/common/vllm/moe.py:_forced_ids; same distributions, different RNG)."""
    rng = np.random.default_rng(seed)
    if dist == "balanced":  # round-robin: every expert gets tokens*top_k/experts slots
        ids = (np.arange(tokens * top_k) % experts).reshape(tokens, top_k)
        return ids[:, rng.permutation(top_k)]
    if dist == "single_hot":  # expert 0 + top_k-1 distinct random others
        rest = np.stack([rng.permutation(experts - 1)[: top_k - 1] + 1 for _ in range(tokens)])
        return np.concatenate([np.zeros((tokens, 1), np.int64), rest], axis=1)
    p = 1.0 / np.arange(1, experts + 1, dtype=np.float64) ** dist["s"]  # zipf over expert rank
    ranked = rng.permutation(experts)
    return np.stack([ranked[rng.choice(experts, top_k, replace=False, p=p / p.sum())]
                     for _ in range(tokens)])


def single_device_mesh() -> jax.sharding.Mesh:
    return jax.sharding.Mesh(np.array(jax.devices()[:1]), ("x",))


def moe_forward(p: dict, x: jax.Array, *, cfg: MoeConfig, mesh: jax.sharding.Mesh) -> jax.Array:
    """x [T, H] bf16 -> [T, H] bf16."""
    logits = jnp.matmul(x, p["w_router"])  # bf16 logits, as vLLM's GateLinear by default
    w, ids = route(logits, cfg, p.get("forced_ids"))
    if cfg.shared:  # the shared expert as one more pick of every token
        if cfg.shared_gate:
            sw = jax.nn.sigmoid(jnp.matmul(x, p["w_shared_gate"], preferred_element_type=_F32))
        else:
            sw = jnp.ones((x.shape[0], 1), _F32)
        w = jnp.concatenate([w, sw.astype(w.dtype)], axis=-1)
        ids = jnp.concatenate([ids, jnp.full_like(ids[:, :1], cfg.num_experts)], axis=-1)
    return modular.routing_moe(
        x, w, ids, p["w1"], p["w2"], p.get("w1_scale"), p.get("w2_scale"), None, None,
        mesh=mesh, activation="silu", tp_axis=None, ep_axis=None, gmm_plan=gmm_plan(cfg))


def make_inputs(cfg: MoeConfig, tokens: int, seed: int, distribution="natural") -> tuple[dict, jax.Array]:
    kp, kx = jax.random.split(jax.random.key(seed))
    # randn, as OperatorX's GPU MoE (runners/common/vllm/moe.py): zero-mean x spreads natural
    # routing across experts, where all-positive x sends every token to the same few
    x = jax.random.normal(kx, (tokens, cfg.hidden), jnp.bfloat16)
    p = init_params(cfg, kp)
    if distribution != "natural":
        p["forced_ids"] = jnp.asarray(forced_ids(distribution, tokens, cfg.num_experts, cfg.top_k, seed),
                                      jnp.int32)
    return p, x


def check(cfg: MoeConfig, tokens: int, seed: int = 0) -> float:
    """Max abs error of moe_forward vs Tokamax's reference_moe (shared expert unfolded,
    unquantized x); raises if out of Tokamax's own test tolerance."""
    from tokamax._src.ops.experimental.routing_moe.test.helper import (
        helper_inputs, helper_reference, helper_shapes)
    p, x = make_inputs(cfg, tokens, seed)
    out = jax.jit(partial(moe_forward, cfg=cfg, mesh=single_device_mesh()))(p, x)
    E = cfg.num_experts
    routed = {k: p[k][:E] for k in ("w1", "w2", "w1_scale", "w2_scale") if k in p}
    shared = {f"{k}_shared": p[k][E:] for k in ("w1", "w2") if cfg.shared}
    shared |= {f"{k}_shared_scale": p[f"{k}_scale"][E:] for k in ("w1", "w2") if cfg.shared and f"{k}_scale" in p}
    moe = helper_inputs.MoEInputs(hidden_states=x, gating_output=jnp.matmul(x, p["w_router"]),
                                  w_shared_gate=p.get("w_shared_gate"), **routed, **shared)
    shape = helper_shapes.MoEShape(name="check", num_experts=E, top_k=cfg.top_k, hidden_size=cfg.hidden,
                                   intermediate_size=cfg.inter, renormalize=cfg.renormalize,
                                   has_shared_expert=cfg.shared)
    ref = helper_reference.reference_moe(moe, shape)
    out, ref = np.asarray(out, np.float32), np.asarray(ref, np.float32)
    # Tokamax tests create x via `uniform[0, 1) / 10` but operatorX uses
    # randn, larger. So we need to slightly loosen the rtol from 5e-2.
    rtol = 2e-2 if cfg.weight_dtype == "bf16" else 8e-2
    np.testing.assert_allclose(out, ref, rtol=rtol, atol=rtol * np.abs(ref).max())
    return float(np.abs(out - ref).max())


if __name__ == "__main__":
    qwen35 = MoeConfig(hidden=4096, num_experts=512, top_k=10, inter=1024, shared=True, shared_gate=True)
    qwen35_fp8 = dataclasses.replace(qwen35, weight_dtype="e4m3", weight_block=128, x_quant_block=128)
    for name, cfg in (("bf16", qwen35), ("fp8", qwen35_fp8)):
        for t in (1, 256, 2048):
            print(f"Qwen3.5-397B-A17B {name} T={t}: max abs err {check(cfg, t):.3g}", flush=True)
    print("CHECK OK")
