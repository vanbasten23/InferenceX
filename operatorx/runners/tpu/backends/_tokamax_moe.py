"""MoE layer on Tokamax routing_moe: router -> routed experts -> + (gated) shared expert.

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
from tokamax._src.ops.experimental.routing_moe.layer import modular

_F32 = jnp.float32


@dataclasses.dataclass(frozen=True)
class MoeConfig:
    hidden: int
    num_experts: int
    top_k: int
    inter: int
    renormalize: bool = True
    shared_inter: int | None = None  # None: no shared expert
    shared_gate: bool = False  # Qwen: sigmoid(x @ w_shared_gate) scales the shared output


def init_params(cfg: MoeConfig, key: jax.Array) -> dict:
    """Random bf16 weights (std 0.1, as Tokamax's test inputs) in routing_moe's layout:
    w1 [E, H, 2I] with [gate | up] columns, w2 [E, I, H]; shared expert as one more [1, ...] expert."""
    H, E, I = cfg.hidden, cfg.num_experts, cfg.inter
    ks = jax.random.split(key, 6)
    rand = lambda k, shape: jax.random.normal(k, shape, jnp.bfloat16) / 10
    p = {"w_router": rand(ks[0], (H, E)), "w1": rand(ks[1], (E, H, 2 * I)), "w2": rand(ks[2], (E, I, H))}
    if cfg.shared_inter:
        S = cfg.shared_inter
        p["w1_shared"] = rand(ks[3], (1, H, 2 * S))
        p["w2_shared"] = rand(ks[4], (1, S, H))
        if cfg.shared_gate:
            p["w_shared_gate"] = rand(ks[5], (H, 1))
    return p


def route(logits: jax.Array, cfg: MoeConfig) -> tuple[jax.Array, jax.Array]:
    """softmax (fp32) -> top_k -> renormalize; same as Tokamax's helper_inputs.top_k."""
    w = jax.nn.softmax(logits.astype(_F32), axis=-1)
    w, ids = jax.lax.top_k(w, cfg.top_k)
    if cfg.renormalize:
        w = w / w.sum(axis=-1, keepdims=True)
    return w, ids.astype(jnp.int32)


def _shared_expert(x: jax.Array, p: dict) -> jax.Array:
    gate, up = jnp.split(jnp.matmul(x, p["w1_shared"][0], preferred_element_type=_F32), 2, axis=-1)
    h = (jax.nn.silu(gate) * up).astype(x.dtype)
    out = jnp.matmul(h, p["w2_shared"][0], preferred_element_type=_F32).astype(x.dtype)
    if "w_shared_gate" in p:
        g = jnp.matmul(x, p["w_shared_gate"], preferred_element_type=_F32)
        out = out * jax.nn.sigmoid(g).astype(x.dtype)
    return out


def single_device_mesh() -> jax.sharding.Mesh:
    return jax.sharding.Mesh(np.array(jax.devices()[:1]), ("x",))


def moe_forward(p: dict, x: jax.Array, *, cfg: MoeConfig, mesh: jax.sharding.Mesh) -> jax.Array:
    """x [T, H] bf16 -> [T, H] bf16."""
    logits = jnp.matmul(x, p["w_router"])  # bf16 logits, as vLLM's GateLinear by default
    w, ids = route(logits, cfg)
    out = modular.routing_moe(
        x, w, ids, p["w1"], p["w2"], None, None, None, None,
        mesh=mesh, activation="silu", tp_axis=None, ep_axis=None)
    if "w1_shared" in p:
        out = out + _shared_expert(x, p)
    return out


def make_inputs(cfg: MoeConfig, tokens: int, seed: int) -> tuple[dict, jax.Array]:
    kp, kx = jax.random.split(jax.random.key(seed))
    x = jax.random.uniform(kx, (tokens, cfg.hidden), jnp.bfloat16) / 10  # as Tokamax's test inputs
    return init_params(cfg, kp), x


def check(cfg: MoeConfig, tokens: int, seed: int = 0) -> float:
    """Max abs error of moe_forward vs Tokamax's reference_moe; raises if out of tolerance."""
    from tokamax._src.ops.experimental.routing_moe.test.helper import (
        helper_inputs, helper_reference, helper_shapes)
    p, x = make_inputs(cfg, tokens, seed)
    out = jax.jit(partial(moe_forward, cfg=cfg, mesh=single_device_mesh()))(p, x)
    shape = helper_shapes.MoEShape(
        name="check", num_experts=cfg.num_experts, top_k=cfg.top_k, hidden_size=cfg.hidden,
        intermediate_size=cfg.inter, renormalize=cfg.renormalize, has_shared_expert=bool(cfg.shared_inter))
    moe = helper_inputs.MoEInputs(
        hidden_states=x, gating_output=jnp.matmul(x, p["w_router"]), w1=p["w1"], w2=p["w2"],
        w1_shared=p.get("w1_shared"), w2_shared=p.get("w2_shared"), w_shared_gate=p.get("w_shared_gate"))
    ref = helper_reference.reference_moe(moe, shape)
    out, ref = np.asarray(out, np.float32), np.asarray(ref, np.float32)
    rtol = 2e-2  # Tokamax moe_test_lib's bf16 tolerance
    np.testing.assert_allclose(out, ref, rtol=rtol, atol=rtol * np.abs(ref).max())
    return float(np.abs(out - ref).max())


if __name__ == "__main__":
    qwen35 = MoeConfig(hidden=4096, num_experts=512, top_k=10, inter=1024,
                       shared_inter=1024, shared_gate=True)
    for t in (1, 256, 2048):
        print(f"Qwen3.5-397B-A17B bf16 T={t}: max abs err {check(qwen35, t):.3g}", flush=True)
    print("CHECK OK")
