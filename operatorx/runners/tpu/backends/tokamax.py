"""Tokamax backend (TPU): the moe op on Tokamax routing_moe (layer in _tokamax_moe.py).

Supported so far: bf16, or fp8 e4m3 weights with static block scales + dynamic e4m3
activations; softmax top-k router; silu experts; optional (sigmoid-gated) shared expert
of the experts' width and format; any routing distribution; one device.
Everything else is UnsupportedOpError.
"""
from __future__ import annotations

from functools import partial

import jax

from operatorx.core import BackendImpl, UnsupportedOpError, lookup_versions
from operatorx.runners.tpu.backends._tokamax_moe import (MoeConfig, gmm_plan, make_inputs, moe_forward,
                                                          single_device_mesh)

_BF16 = {"dtype": "bf16"}


def _require(ok: bool, msg: str) -> None:
    """Like assert, but the row is reported as unsupported (main.py), not as an error."""
    if not ok:
        raise UnsupportedOpError(f"tokamax moe: {msg}")


def _quant(ex: dict) -> dict:
    """MoeConfig quant fields for the experts' operands; raises if routing_moe can't take them."""
    a, w = ex["a1"], ex["w1"]
    _require(ex["a2"] == a and ex["w2"] == w, "a2/w2 must use the same format as a1/w1")
    if a == _BF16 and w == _BF16:
        return {"weight_dtype": "bf16"}
    a_s, w_s = a.get("scale") or {}, w.get("scale") or {}
    # Eg. "group" is the size of the block that shares one scale, not the
    # shape of the scale tensor. So [1, 128] means one fp32 scale for every
    # 1 token x 128 hidden values.
    # group[0] is not passed to the routing_moe. We check `group[0] == 1`,
    # the only layout the kernel's dynamic quantization does.
    # group[1] is passed to the kernel as
    # RoutingGmmPlan(matmul_dtype=e4m3fn, x_quant_block_size=cfg.x_quant_block)
    if (a.get("dtype") == "e4m3" and a_s.get("dtype") == "fp32" and not a_s.get("static")
            and a_s.get("group", [0])[0] == 1 and "scale2" not in a
            and w.get("dtype") == "e4m3" and w_s.get("dtype") in ("bf16", "fp32") and w_s.get("static")
            and "scale2" not in w):
        # [kb, nb] weight blocks -> one fp32 scale per kb rows per column (exact, finer)
        return {"weight_dtype": "e4m3", "weight_block": w_s["group"][0], "x_quant_block": a_s["group"][1]}
    raise UnsupportedOpError(f"tokamax moe: expert operands a1={a} w1={w} are neither bf16 nor "
                             "e4m3 weights (static scales) + dynamic e4m3 activations")


def _to_config(a: dict) -> MoeConfig:
    ex, rt, act, sh = a["experts"], a["router"], a["activation"], a.get("shared")
    quant = _quant(ex)
    if sh is not None:  # folded into routing_moe as one more expert, so it must look like one
        _require(sh["count"] == 1, f"only 1 shared expert, got {sh['count']}")
        _require(sh["inter"] == ex["inter"], f"shared inter {sh['inter']} != experts inter {ex['inter']}")
        _require(all(sh[k] == ex[k] for k in ("a1", "w1", "w2")), "shared operands differ from the experts'")
    _require(not any(ex.get(k) for k in ("bias", "latent", "zero")), "expert bias/latent/zero experts")
    _require(rt["gate"].get("dtype") == "bf16" and rt["gate"].get("logits", "bf16") == "bf16",
             f"router gate/logits must be bf16, got {rt['gate']}")
    _require(rt["scoring"] == "softmax", f"scoring must be softmax, got {rt['scoring']!r}")
    _require(rt["select"].get("kind") == "topk", f"select must be topk, got {rt['select'].get('kind')!r}")
    _require(not any(rt.get(k) for k in ("bias", "scale", "weight_on_input")),
             "router bias/scale/weight_on_input")
    _require(act == {"kind": "silu"}, f"activation must be plain silu, got {act}")
    _require(a.get("out", "bf16") == "bf16", f"out must be bf16, got {a.get('out')!r}")
    _require(ex["inter"] % 256 == 0, f"routing_moe needs experts.inter % 256 == 0, got {ex['inter']}")
    return MoeConfig(hidden=a["hidden"], num_experts=ex["num"], top_k=ex["top_k"], inter=ex["inter"],
                     renormalize=bool(rt.get("renormalize")), shared=sh is not None,
                     shared_gate=bool(sh and sh.get("gate") == "sigmoid"), **quant)


def _prepare_moe(op) -> dict:
    a = op.args
    cfg = _to_config(a)
    routing = a.get("routing") or {"distribution": "natural", "seed": 0}
    p, x = make_inputs(cfg, a["tokens"], routing["seed"], routing["distribution"])
    fn = jax.jit(partial(moe_forward, cfg=cfg, mesh=single_device_mesh()))
    fn(p, x).block_until_ready()  # compile outside timing
    return {"fn": fn, "p": p, "x": x,
            "meta": {"impl": "routing_moe.modular", "devices": 1, "shared_expert": "folded" if cfg.shared else None,
                     "gmm_plan": repr(gmm_plan(cfg))}}


def _kernel_moe(ctx: dict) -> None:
    ctx["fn"](ctx["p"], ctx["x"]).block_until_ready()


IMPLS = [BackendImpl(op_type="moe", prepare=_prepare_moe, kernel=_kernel_moe)]


def versions() -> dict[str, str]:
    return lookup_versions("tokamax")
