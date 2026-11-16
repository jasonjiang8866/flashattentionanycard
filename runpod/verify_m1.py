"""Bind M1 paired confirmations to every accepted search and its published CSV."""

import csv
import json

CONFIRM_REPETITIONS = 6


def verify_confirmations(output, plan, manifest, state):
    # Lazy import keeps inventory/cleanup and the stdlib controller tests usable
    # without Optuna. The paid verification job installs the pinned dependency.
    from bo_kernel_tuner.tuner.confirmation import verify_confirmation

    if manifest.get("confirmation_repetitions") != CONFIRM_REPETITIONS or state.get(
        "completed_confirmations"
    ) != len(plan):
        raise RuntimeError("M1 confirmations were not fully executed")
    expected_paths = {
        output / "confirmations" / item["path"] / "confirmation.json" for item in plan
    }
    if set((output / "confirmations").rglob("confirmation.json")) != expected_paths:
        raise RuntimeError("Unexpected M1 confirmation set")
    rows = []
    for item in plan:
        source = output / item["path"] / "result.json"
        folder = output / "confirmations" / item["path"]
        if (folder / "source-result.json").read_bytes() != source.read_bytes():
            raise RuntimeError("Confirmation is not bound to its accepted search")
        summary = verify_confirmation(folder)
        report = json.loads((folder / "confirmation.json").read_text())
        if report["repetitions"] != CONFIRM_REPETITIONS:
            raise RuntimeError("Unexpected confirmation repetition count")
        rows.append(
            {
                "case": item["case"]["id"],
                "method": item["label"],
                "seed": item["seed"],
                "baseline": report["baseline"],
                "search_best_ms": report["search_best_ms"],
                **{k: v for k, v in summary.items() if k != "paired_speedups"},
            }
        )
    with (output / "baseline-summary.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        actual = list(reader)
        if reader.fieldnames != list(rows[0]) or actual != [
            {k: str(v) for k, v in row.items()} for row in rows
        ]:
            raise RuntimeError("Baseline summary differs from verified paired evidence")
    return {"confirmations": len(rows), "paired_workers": len(rows) * CONFIRM_REPETITIONS}
