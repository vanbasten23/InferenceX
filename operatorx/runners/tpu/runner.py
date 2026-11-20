"""TPU runner (JAX), single device: host wall-clock timing of a blocking call.

Unlike the CUDA runners there is no L2 flush, CUDA graph or telemetry. A backend's
kernel(ctx) (or launcher callable) must block until the device work is done, e.g. by
calling .block_until_ready() on its output, or the timing measures only the dispatch.
"""
from __future__ import annotations

import pkgutil
import time
from importlib import import_module

from operatorx.core import BackendImpl, Op, Result, UnsupportedOpError

# Same counts as runners/common/timing.py, which imports torch and so can't be reused here.
WARMUP = 5
ITERS = 10

_DISPATCH: dict[tuple[str, str], BackendImpl] = {}


def _load() -> None:
    if _DISPATCH:
        return
    from operatorx.runners.tpu import backends
    for info in pkgutil.iter_modules(backends.__path__):
        if info.name.startswith("_"):
            continue
        try:
            mod = import_module(f"operatorx.runners.tpu.backends.{info.name}")
        except ImportError:
            continue
        for impl in getattr(mod, "IMPLS", []):
            _DISPATCH[(impl.op_type, info.name)] = impl


def time_op(fn) -> float:
    """Median over ITERS calls of the blocking fn, in us."""
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    return sorted(times)[ITERS // 2]


def run(op: Op) -> Result:
    _load()
    impl = _DISPATCH.get((op.type, op.backend))
    if impl is None:
        raise UnsupportedOpError(f"tpu/{op.backend} has no impl for op_type={op.type!r}")
    ctx = impl.prepare(op)

    fn = impl.launcher(ctx)[0] if impl.launcher else (lambda: impl.kernel(ctx))
    metrics = {"latency_us": time_op(fn), "cuda_graph": False}
    if isinstance(ctx, dict) and ctx.get("meta"):
        metrics["backend_meta"] = ctx["meta"]
    return Result(op=op, metrics=metrics)
