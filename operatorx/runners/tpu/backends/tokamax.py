"""Tokamax backend (TPU): the moe op on Tokamax routing_moe (layer in _tokamax_moe.py).

Supported so far: bf16, softmax top-k router, silu experts, optional sigmoid-gated shared
expert, natural routing, one device. Everything else is UnsupportedOpError.
"""
from __future__ import annotations

from functools import partial

import jax

from operatorx.core import BackendImpl, UnsupportedOpError, lookup_versions
from operatorx.runners.tpu.backends._tokamax_moe import (MoeConfig, make_inputs, moe_forward,
                                                          single_device_mesh)

_BF16 = {"dtype": "bf16"}


def _to_config(a: dict) -> MoeConfig:
    ex, rt, act, sh = a["experts"], a["router"], a["activation"], a.get("shared")
    routing = a.get("routing") or {"distribution": "natural"}
    checks = {
        "non-bf16 expert operands": any(ex[k] != _BF16 for k in ("a1", "w1", "w2", "a2")),
        "non-bf16 shared operands": sh is not None and any(sh[k] != _BF16 for k in ("a1", "w1", "w2")),
        "expert bias/latent/zero": any(ex.get(k) for k in ("bias", "latent", "zero")),
        "router gate/logits not bf16": rt["gate"].get("dtype") != "bf16" or rt["gate"].get("logits", "bf16") != "bf16",
        f"scoring {rt['scoring']!r}": rt["scoring"] != "softmax",
        f"select {rt['select'].get('kind')!r}": rt["select"].get("kind") != "topk",
        "router bias/scale/weight_on_input": any(rt.get(k) for k in ("bias", "scale", "weight_on_input")),
        f"activation {act}": act != {"kind": "silu"},
        "shared count != 1": sh is not None and sh["count"] != 1,
        f"routing {routing['distribution']!r}": routing["distribution"] != "natural",
        f"out {a.get('out', 'bf16')!r}": a.get("out", "bf16") != "bf16",
        "experts.inter % 256 != 0": ex["inter"] % 256 != 0,
    }
    bad = [why for why, hit in checks.items() if hit]
    if bad:
        raise UnsupportedOpError("tokamax moe: " + "; ".join(bad))
    return MoeConfig(hidden=a["hidden"], num_experts=ex["num"], top_k=ex["top_k"], inter=ex["inter"],
                     renormalize=bool(rt.get("renormalize")),
                     shared_inter=sh["inter"] if sh else None,
                     shared_gate=bool(sh and sh.get("gate") == "sigmoid"))


def _prepare_moe(op) -> dict:
    a = op.args
    cfg = _to_config(a)
    p, x = make_inputs(cfg, a["tokens"], (a.get("routing") or {}).get("seed", 0))
    fn = jax.jit(partial(moe_forward, cfg=cfg, mesh=single_device_mesh()))
    fn(p, x).block_until_ready()  # compile outside timing
    return {"fn": fn, "p": p, "x": x, "meta": {"impl": "routing_moe.modular", "devices": 1}}


def _kernel_moe(ctx: dict) -> None:
    ctx["fn"](ctx["p"], ctx["x"]).block_until_ready()


IMPLS = [BackendImpl(op_type="moe", prepare=_prepare_moe, kernel=_kernel_moe)]


def versions() -> dict[str, str]:
    return lookup_versions("tokamax")
