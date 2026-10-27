"""CLI: tune one sampler or compare paired seeds in randomized method order."""

import json
import random
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from bo_kernel_tuner.configs.search_spaces import Workload, grid_size, search_space
from bo_kernel_tuner.tuner.engine import run_search
from bo_kernel_tuner.tuner.hardware import Hardware, prune_reason
from bo_kernel_tuner.tuner.transfer import load_warm_start
from bo_kernel_tuner.tuner.worker import Measurement

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
console = Console()


@app.command()
def main(
    kernel: str = "gemm",
    mode: str = "compare",
    method: str = "tpe",
    device: int = 0,
    dtype: str = "fp16",
    m: int = 1024,
    n: int = 1024,
    k: int = 1024,
    expert_tokens: str = "64,128,256,512",
    epsilon: float = 1e-6,
    data_seed: int = 17,
    seeds: str = "0,1,2,3,4",
    trials: int = 100,
    profile: str = "full",
    output: Path = Path("results/experiment"),
    time_budget: Optional[float] = None,
    trial_timeout: float = 180,
    startup_trials: int = 10,
    warm_start: Optional[Path] = None,
    top_k: int = 5,
    warmup: int = 20,
    repeats: int = 30,
    batch: int = 32,
    validation_seeds: int = 2,
    heuristic_smem: bool = False,
    dry_run: bool = False,
):
    """Run independent studies; --dry-run prints the design without loading CUDA."""
    try:
        if mode not in ("tune", "compare") or method not in ("grid", "random", "tpe"):
            raise ValueError("mode must be tune/compare; method must be grid/random/tpe")
        seed_list = [int(x) for x in seeds.split(",")]
        if not seed_list or min(seed_list) < 0 or len(set(seed_list)) != len(seed_list):
            raise ValueError("Seeds must be distinct nonnegative integers")
        w = Workload(
            kernel,
            m,
            n,
            k,
            dtype,
            tuple(int(x) for x in expert_tokens.split(",")),
            epsilon,
            data_seed,
        )
        space = search_space(kernel, profile)
        measure = Measurement(warmup, repeats, batch, validation_seeds)
        if trials < 1 or trial_timeout <= 0 or startup_trials < 1 or top_k < 1:
            raise ValueError("Budgets, timeout, startup trials and top-k must be positive")
        if time_budget is not None and time_budget <= 0:
            raise ValueError("Time budget must be positive")
        if warm_start and (top_k > trials or (mode == "tune" and method != "tpe")):
            raise ValueError("Warm start requires TPE and top-k <= trials")
        plan = []
        for seed in seed_list:
            methods = ["grid", "random", "tpe"] if mode == "compare" else [method]
            if warm_start:
                methods = methods + ["tpe_warm"] if mode == "compare" else ["tpe_warm"]
            random.Random(seed).shuffle(methods)
            plan.extend((seed, item) for item in methods)
        console.print(
            {
                "workload": w.signature(),
                "grid_size": grid_size(space),
                "attempt_budget": trials,
                "time_budget": time_budget,
                "run_order": plan,
            }
        )
        if dry_run:
            return
        hardware = Hardware.detect(device)
        warm_params, provenance = [], None
        if warm_start:
            warm_params, provenance = load_warm_start(
                warm_start,
                w,
                space,
                top_k,
                hardware,
                lambda p: prune_reason(hardware, kernel, p, n, heuristic_smem),
            )
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new output directory to avoid overwriting or mixing runs")
        output.mkdir(parents=True, exist_ok=True)
        (output / "plan.json").write_text(
            json.dumps({"order": plan, "hardware": hardware.to_dict()}, indent=2)
        )
        failed = []
        for seed, label in plan:
            console.print(f"[bold]{label} seed={seed}[/bold] on {hardware.name}")
            result = run_search(
                w,
                hardware,
                space,
                "tpe" if label == "tpe_warm" else label,
                seed,
                trials,
                output / f"{label}-seed-{seed}",
                measure,
                time_budget,
                trial_timeout,
                startup_trials,
                warm_params if label == "tpe_warm" else (),
                provenance if label == "tpe_warm" else None,
                heuristic_smem,
            )
            if result["status"] != "complete":
                failed.append(f"{label}-seed-{seed}")
            console.print(
                {
                    "status": result["status"],
                    "attempts": result["attempts"],
                    "successful": result["successful_evaluations"],
                    "best_ms": result["best"]["value"] if result["best"] else None,
                }
            )
        if failed:
            console.print(f"[red]No valid measurements in: {failed}[/red]")
            raise typer.Exit(1)
    except (ValueError, RuntimeError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


if __name__ == "__main__":
    app()
