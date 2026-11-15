"""Disposable CUDA worker: a timeout/illegal access cannot poison the tuner.

Run as a module. The parent owns trial accounting, stderr, and failure state.
"""

import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from bo_kernel_tuner.configs.search_spaces import Workload

MEASUREMENT_PROTOCOL = "cuda_graph_v2;side_stream_warmup;hot_reused_buffers"
BASELINES = {
    "gemm": "torch.mm_out",
    "rmsnorm": "torch.nn.functional.rms_norm",
    "moe_grouped_gemm": "torch.mm_out_per_nonempty_expert",
}
PRECISION_POLICY = {
    "tf32": False,
    "fp16_reduced_precision_reduction": False,
    "bf16_reduced_precision_reduction": False,
    "float32_matmul_precision": "highest",
}


@dataclass(frozen=True)
class Measurement:
    warmup: int = 20
    repeats: int = 30
    batch: int = 32
    validation_seeds: int = 2

    def __post_init__(self):
        counts = (self.warmup, self.repeats, self.batch, self.validation_seeds)
        if any(type(v) is not int for v in counts):
            raise ValueError("Measurement counts must be integers")
        if self.warmup < 1 or self.repeats < 3 or self.batch < 1 or self.validation_seeds < 1:
            raise ValueError("Require warmup >= 1, repeats >= 3, batch/seeds >= 1")


def prepare(workload, device, seed, sm_count, implementation="triton"):
    import torch

    if implementation not in ("triton", "torch"):
        raise ValueError("Unknown implementation")
    baseline = implementation == "torch"
    if not baseline:
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
        call = (
            (lambda p: torch.mm(a, b, out=out))
            if baseline
            else (lambda p: gemm.launch(a, b, out, p))
        )
    elif w.kernel == "rmsnorm":
        x, weight = rand(w.m, w.n), rand(w.n)
        out = None if baseline else torch.empty_like(x)
        ref = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + w.epsilon)
        ref = ref * weight.float()
        call = (
            (lambda p: torch.nn.functional.rms_norm(x, (w.n,), weight, w.epsilon))
            if baseline
            else (lambda p: rmsnorm.launch(x, weight, out, p, w.epsilon))
        )
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
        if baseline:
            # Fixed views built outside capture; no routing/scatter or padded FLOPs.
            views = [
                (x[starts[e] : starts[e + 1]], weight[e], out[starts[e] : starts[e + 1]])
                for e, count in enumerate(tokens)
                if count
            ]

            def call(p):
                for a, b, y in views:
                    torch.mm(a, b, out=y)
                return out
        else:
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


def sample_statistics(samples):
    """Summaries are derived from retained samples; NaN/inf never pass admission."""
    if len(samples) < 3 or any(
        type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples
    ):
        raise ValueError("Require at least three finite positive CUDA timing samples")
    median = statistics.median(samples)
    return {
        "latency_ms": median,
        "samples_ms": samples,
        "mad_ms": statistics.median(abs(x - median) for x in samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def measure(call, config):
    import torch

    # PyTorch/cuBLAS initialization must occur on a side stream before capture.
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(config.warmup):
            call()
    current.wait_stream(stream)
    stream.synchronize()
    # Captured repeated launches exclude Python dispatch gaps for small kernels.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
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
    return sample_statistics(samples)


def evaluate(spec):
    import torch

    from bo_kernel_tuner.tuner.hardware import Hardware

    torch.cuda.set_device(spec["device"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    hardware = Hardware.detect(spec["device"])
    w = Workload(**spec["workload"])
    config = Measurement(**spec["measurement"])
    p = spec["params"]
    implementation = spec.get("implementation", "triton")
    baseline = implementation == "torch"
    device = torch.device("cuda", spec["device"])
    started = time.perf_counter()
    validation = []
    for offset in range(config.validation_seeds):
        call, out, ref = prepare(w, device, w.data_seed + offset, hardware.sm_count, implementation)
        if out is not None:
            out.fill_(float("nan"))  # Detect incomplete stores, including ragged tails.
        compiled = call(p)
        torch.cuda.synchronize()
        validation.append(check_output(compiled if baseline else out, ref, w.dtype))
    # Benchmark on the first validation dataset, independent of search seed.
    call, out, ref = prepare(w, device, w.data_seed, hardware.sm_count, implementation)
    shared = None if baseline else int(getattr(compiled.metadata, "shared", 0))
    if shared is not None and shared > hardware.shared_memory_per_block:
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
            "implementation": BASELINES[w.kernel] if baseline else "triton",
            "protocol": MEASUREMENT_PROTOCOL,
            "precision_policy": PRECISION_POLICY,
        }
    )
    return result


def evaluate_confirmation(spec):
    """A frozen candidate and its baseline share input seeds and timing settings."""
    from bo_kernel_tuner.tuner.hardware import Hardware

    order = spec["order"]
    if sorted(order) != ["candidate", "torch"]:
        raise ValueError("Confirmation requires exactly one candidate and one baseline")
    result = {"order": order, "data_seed": spec["workload"]["data_seed"]}
    for name in order:
        result[name] = evaluate(
            dict(spec, implementation="triton" if name == "candidate" else "torch")
        )
        result[name]["status"] = "ok"
    result["hardware"] = Hardware.detect(spec["device"]).to_dict()
    return result


def main():
    source, destination = map(Path, sys.argv[1:3])
    try:
        spec = json.loads(source.read_text())
        result = evaluate_confirmation(spec) if spec.get("order") is not None else evaluate(spec)
        result["status"] = "ok"
    except Exception as exc:
        import traceback

        traceback.print_exc()
        result = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
    destination.write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
