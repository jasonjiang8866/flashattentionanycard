"""Disposable CUDA worker: a timeout/illegal access cannot poison the tuner.

Run as a module. The parent owns trial accounting, stderr, and failure state.
"""

import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from bo_kernel_tuner.configs.search_spaces import Workload


@dataclass(frozen=True)
class Measurement:
    warmup: int = 20
    repeats: int = 30
    batch: int = 32
    validation_seeds: int = 2

    def __post_init__(self):
        if self.warmup < 1 or self.repeats < 3 or self.batch < 1 or self.validation_seeds < 1:
            raise ValueError("Require warmup >= 1, repeats >= 3, batch/seeds >= 1")


def prepare(workload, device, seed, sm_count):
    import torch

    from bo_kernel_tuner.kernels import gemm, moe_grouped_gemm, rmsnorm

    w = workload
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[w.dtype]
    generator = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape):
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    if w.kernel == "gemm":
        a, b = rand(w.m, w.k), rand(w.k, w.n)
        out = torch.empty((w.m, w.n), device=device, dtype=dtype)
        ref = a.float() @ b.float()
        call = lambda p: gemm.launch(a, b, out, p)
    elif w.kernel == "rmsnorm":
        x, weight = rand(w.m, w.n), rand(w.n)
        out = torch.empty_like(x)
        ref = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + w.epsilon)
        ref = ref * weight.float()
        call = lambda p: rmsnorm.launch(x, weight, out, p, w.epsilon)
    else:
        tokens = w.expert_tokens
        x, weight = rand(sum(tokens), w.k), rand(len(tokens), w.k, w.n)
        starts = [0]
        for count in tokens:
            starts.append(starts[-1] + count)
        offsets = torch.tensor(starts, device=device, dtype=torch.int64)
        out = torch.empty((sum(tokens), w.n), device=device, dtype=dtype)
        ref = torch.cat(
            [x[starts[e] : starts[e + 1]].float() @ weight[e].float() for e in range(len(tokens))]
        )
        call = lambda p: moe_grouped_gemm.launch(x, weight, offsets, out, p, sm_count)
    return call, out, ref


def check_output(out, reference, dtype):
    import torch

    rtol, atol = (0.01, 0.02) if dtype == "fp16" else (0.02, 0.15)
    if not torch.isfinite(out).all() or not torch.isfinite(reference).all():
        raise AssertionError("Nonfinite output or reference")
    torch.testing.assert_close(out.float(), reference, rtol=rtol, atol=atol)
    delta = (out.float() - reference).abs()
    return {"max_abs_error": delta.max().item(), "rtol": rtol, "atol": atol}


def measure(call, config):
    import torch

    for _ in range(config.warmup):
        call()
    torch.cuda.synchronize()
    # Captured repeated launches exclude Python dispatch gaps for small kernels.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(config.batch):
            call()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(config.repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) / config.batch)
    median = statistics.median(samples)
    if median <= 0:
        raise RuntimeError("CUDA timer returned nonpositive latency")
    return {
        "latency_ms": median,
        "samples_ms": samples,
        "mad_ms": statistics.median(abs(x - median) for x in samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def evaluate(spec):
    import torch

    from bo_kernel_tuner.tuner.hardware import Hardware

    torch.cuda.set_device(spec["device"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    hardware = Hardware.detect(spec["device"])
    w = Workload(**spec["workload"])
    config = Measurement(**spec["measurement"])
    p = spec["params"]
    device = torch.device("cuda", spec["device"])
    started = time.perf_counter()
    validation = []
    for offset in range(config.validation_seeds):
        call, out, ref = prepare(w, device, w.data_seed + offset, hardware.sm_count)
        out.fill_(float("nan"))  # Detect incomplete stores, including ragged tails.
        compiled = call(p)
        torch.cuda.synchronize()
        validation.append(check_output(out, ref, w.dtype))
    # Benchmark on the first validation dataset, independent of search seed.
    call, out, ref = prepare(w, device, w.data_seed, hardware.sm_count)
    shared = int(getattr(compiled.metadata, "shared", 0))
    if shared > hardware.shared_memory_per_block:
        raise RuntimeError("Compiled shared memory exceeds device limit")
    setup_s = time.perf_counter() - started
    result = measure(lambda: call(p), config)
    result.update(
        {
            "validation": validation,
            "setup_compile_validate_s": setup_s,
            "compiled_shared_bytes": shared,
            "compiled_registers": getattr(compiled, "n_regs", None),
            "compiled_spills": getattr(compiled, "n_spills", None),
        }
    )
    return result


def main():
    source, destination = map(Path, sys.argv[1:3])
    try:
        result = evaluate(json.loads(source.read_text()))
        result["status"] = "ok"
    except Exception as exc:
        import traceback

        traceback.print_exc()
        result = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
    destination.write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
