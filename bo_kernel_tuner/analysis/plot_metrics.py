"""Per-device comparisons with matched workloads and explicit censored outcomes."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def label(run):
    meta = run["metadata"]
    return meta["method"] + ("_warm" if meta["transfer"] else "")


def cohort_key(run, include_hardware=True):
    meta = run["metadata"]
    keys = [
        "workload",
        "search_space",
        "measurement",
        "protocol",
        "code_fingerprint",
        "heuristic_smem",
        "budget",
        "time_budget_s",
        "startup_trials",
        "trial_timeout_s",
    ]
    data = {key: meta[key] for key in keys}
    data["software_versions"] = meta["environment"]["versions"]
    if include_hardware:
        data["hardware"] = {k: v for k, v in meta["hardware"].items() if k != "device"}
    return json.dumps(data, sort_keys=True)


def incumbent(run, axis="attempt"):
    x, y, best = [], [], float("inf")
    for t in run["trials"]:
        if t["state"] == "COMPLETE":
            best = min(best, t["value"])
        x.append(t["number"] + 1 if axis == "attempt" else t["attrs"]["elapsed_s"])
        y.append(best)
    return np.asarray(x), np.asarray(y)


def step_values(x, y, grid):
    indices = np.searchsorted(x, grid, side="right") - 1
    out = np.full(len(grid), np.inf)
    if len(x):
        valid = indices >= 0
        out[valid] = y[indices[valid]]
    return out


def summarize(runs, target_factor=1.05):
    if target_factor < 1:
        raise ValueError("target_factor must be >= 1")
    observed = [t["value"] for r in runs for t in r["trials"] if t["state"] == "COMPLETE"]
    target = min(observed) * target_factor if observed else None
    rows = []
    for run in runs:
        hits = [t for t in run["trials"] if t["state"] == "COMPLETE" and t["value"] <= target]
        first = hits[0] if hits else None
        rows.append(
            {
                "method": label(run),
                "seed": run["metadata"]["seed"],
                "status": run["status"],
                "attempts": run["attempts"],
                "successful": run["successful_evaluations"],
                "unique_successful": run["unique_successful_configs"],
                "pruned": sum(t["state"] == "PRUNED" for t in run["trials"]),
                "failed": sum(t["state"] == "FAIL" for t in run["trials"]),
                "best_ms": run["best"]["value"] if run["best"] else None,
                "total_s": run["elapsed_s"],
                "target_ms": target,
                "target_reached": first is not None,
                "attempts_to_target": first["number"] + 1 if first else None,
                "seconds_to_target": first["attrs"]["elapsed_s"] if first else None,
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig, stem):
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def paired_comparisons(frame):
    """Exploratory paired ratios; never discard unsuccessful/censored seeds."""
    rows = []
    methods = set(frame.method)
    comparisons = [("grid", name) for name in sorted(methods - {"grid"}) if "grid" in methods]
    if {"tpe", "tpe_warm"} <= methods:
        comparisons.append(("tpe", "tpe_warm"))
    for baseline, candidate in comparisons:
        a = frame[frame.method == baseline].set_index("seed")
        b = frame[frame.method == candidate].set_index("seed").reindex(a.index)
        for metric in ("best_ms", "attempts_to_target", "seconds_to_target"):
            valid = a[metric].notna() & b[metric].notna()
            row = {
                "baseline": baseline,
                "candidate": candidate,
                "metric": metric,
                "paired_seeds": len(a),
                "observed_pairs": int(valid.sum()),
                "baseline_hit_rate": float(a.target_reached.mean()),
                "candidate_hit_rate": float(b.target_reached.mean()),
                "median_ratio": None,
                "ci95_low": None,
                "ci95_high": None,
            }
            if valid.all():
                ratios = a[metric].to_numpy() / b[metric].to_numpy()
                row["median_ratio"] = float(np.median(ratios))
                if len(ratios) >= 3:
                    rng = np.random.default_rng(0)
                    medians = np.median(
                        rng.choice(ratios, (5000, len(ratios)), replace=True), axis=1
                    )
                    row["ci95_low"], row["ci95_high"] = np.quantile(medians, [0.025, 0.975])
            rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "baseline",
            "candidate",
            "metric",
            "paired_seeds",
            "observed_pairs",
            "baseline_hit_rate",
            "candidate_hit_rate",
            "median_ratio",
            "ci95_low",
            "ci95_high",
        ],
    )


def render_cohort(runs, output, target_factor):
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    frame = summarize(runs, target_factor)
    frame.to_csv(output / "summary.csv", index=False)
    groups = {}
    seen = set()
    for run in runs:
        key = (label(run), run["metadata"]["seed"])
        if key in seen:
            raise ValueError(f"Duplicate method/seed in cohort: {key}")
        seen.add(key)
        groups.setdefault(label(run), []).append(run)
    if len({tuple(sorted(r["metadata"]["seed"] for r in group)) for group in groups.values()}) > 1:
        raise ValueError("Methods must have matched seed sets within each cohort")
    paired_comparisons(frame).to_csv(output / "paired_comparisons.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for axis, ax in zip(("attempt", "time"), axes):
        horizon = min((r["attempts"] if axis == "attempt" else r["elapsed_s"]) for r in runs)
        grid = np.arange(1, horizon + 1) if axis == "attempt" else np.linspace(0, horizon, 250)
        for name, group in sorted(groups.items()):
            values = np.asarray([step_values(*incumbent(r, axis), grid) for r in group])
            # No survivor bias: draw only where every seed has a valid incumbent.
            valid = np.all(np.isfinite(values), axis=0)
            med = np.full(len(grid), np.nan)
            low, high = med.copy(), med.copy()
            if valid.any():
                low[valid], med[valid], high[valid] = np.quantile(
                    values[:, valid], [0.25, 0.5, 0.75], axis=0
                )
            ax.step(grid, med, where="post", label=f"{name} (n={len(group)})")
            ax.fill_between(grid, low, high, alpha=0.18, step="post")
        ax.set(
            xlabel="Attempt (includes failures/pruning)"
            if axis == "attempt"
            else "End-to-end search time (s)",
            ylabel="Best measured latency (ms)",
            title="Median and interquartile range",
        )
        ax.legend(fontsize=8)
    save_figure(fig, output / "convergence")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.boxplot(data=frame, x="method", y="total_s", ax=ax, color="#a9c7df")
    sns.stripplot(data=frame, x="method", y="total_s", ax=ax, color="#24344d", jitter=False)
    ax.set(xlabel="", ylabel="End-to-end tuning time (s)")
    save_figure(fig, output / "tuning_time")
    notes = {
        "target": f"{target_factor} x best observed within this device cohort; not an oracle",
        "unreached_targets": "right-censored; retained with blank time/attempt, never dropped",
        "curves": "median/IQR across seeds; common horizon; all seeds need finite incumbents",
        "cache_policy": "CUDA graph replay reuses buffers (hot-cache workload)",
        "paired_ratios": "baseline/candidate (>1 favors candidate); omitted if any pair is censored; percentile bootstrap 5000 resamples seed=0, >=3 pairs; exploratory",
        "cohort": json.loads(cohort_key(runs[0])),
    }
    (output / "methodology.json").write_text(json.dumps(notes, indent=2))


def parameter_heatmap(runs, output):
    """Ordinal best-parameter locations, not a false continuous hardware correlation."""
    valid = [r for r in runs if r["best"]]
    if not valid:
        return
    space = valid[0]["metadata"]["search_space"]
    rows, names = [], []
    for run in valid:
        p = run["best"]["params"]
        rows.append(
            {k: choices.index(p[k]) / max(1, len(choices) - 1) for k, choices in space.items()}
        )
        names.append(
            f"{run['metadata']['hardware']['name']} / {label(run)} / s{run['metadata']['seed']}"
        )
    frame = pd.DataFrame(rows, index=names)
    actual = pd.DataFrame([r["best"]["params"] for r in valid], index=names)[frame.columns]
    actual.to_csv(output / "best_parameters.csv")
    fig, ax = plt.subplots(figsize=(max(6, len(space) * 1.3), max(3, len(valid) * 0.35)))
    sns.heatmap(
        frame,
        annot=actual.to_numpy(),
        fmt="d",
        cmap="viridis",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Ordinal position within search domain [0,1]"},
    )
    ax.set_title("Best-parameter shifts across devices and seeds (annotations = actual values)")
    save_figure(fig, output / "parameter_shift")


def generate(paths, output, target_factor=1.05):
    files = sorted(
        {
            p.resolve()
            for root in paths
            for p in ([root] if root.is_file() else root.rglob("result.json"))
        }
    )
    runs = [json.loads(path.read_text()) for path in files]
    if not runs or any(
        r.get("schema_version") != 1 or r.get("status") not in ("complete", "failed") for r in runs
    ):
        raise ValueError(
            "Supply completed/failed schema-v1 studies; interrupted studies need investigation"
        )
    output.mkdir(parents=True, exist_ok=True)
    cohorts, heatmaps = {}, {}
    for run in runs:
        cohorts.setdefault(cohort_key(run), []).append(run)
        heatmaps.setdefault(cohort_key(run, False), []).append(run)
    for key, group in cohorts.items():
        name = hashlib.sha256(key.encode()).hexdigest()[:10]
        render_cohort(group, output / name, target_factor)
    for key, group in heatmaps.items():
        folder = output / ("parameters-" + hashlib.sha256(key.encode()).hexdigest()[:10])
        folder.mkdir(exist_ok=True)
        parameter_heatmap(group, folder)
    return len(cohorts)


@app.command()
def main(
    inputs: list[Path] = typer.Argument(...),
    output: Path = Path("results/figures"),
    target_factor: float = 1.05,
):
    """Read one or more experiment directories (or result.json files)."""
    typer.echo(
        f"Rendered {generate(inputs, output, target_factor)} independent device cohorts to {output}"
    )


if __name__ == "__main__":
    app()
