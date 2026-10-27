"""Rank-based transfer: enqueue source parameters, never import source latencies."""

import hashlib
import json
import math
from pathlib import Path

from bo_kernel_tuner.configs.search_spaces import validate_params


def load_warm_start(path, workload, space, top_k, hardware, prune):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    raw = Path(path).read_bytes()
    source = json.loads(raw)
    if source.get("schema_version") != 1 or source.get("status") != "complete":
        raise ValueError("Warm start requires a completed schema-v1 result")
    meta = source["metadata"]
    if meta["workload"] != workload.signature() or meta["search_space"] != space:
        raise ValueError("Warm start workload/dtype/data seed/search space mismatch")
    # Code changes can alter the meaning of the same parameter names.
    from bo_kernel_tuner.tuner.engine import code_fingerprint

    if meta["code_fingerprint"] != code_fingerprint():
        raise ValueError("Source and target kernel/measurement code differ")
    candidates = []
    seen = set()
    valid = [
        t
        for t in source["trials"]
        if t["state"] == "COMPLETE"
        and isinstance(t["value"], (float, int))
        and math.isfinite(t["value"])
        and t["value"] > 0
    ]
    for trial in sorted(valid, key=lambda t: t["value"]):
        params = trial["params"]
        key = json.dumps(params, sort_keys=True)
        if not validate_params(params, space) or key in seen or prune(params):
            continue
        seen.add(key)
        candidates.append(params)
        if len(candidates) == top_k:
            break
    if not candidates:
        raise ValueError("No legal source configurations available")
    return candidates, {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hardware": meta["hardware"],
        "target_hardware": hardware.to_dict(),
        "requested_top_k": top_k,
        "enqueued": len(candidates),
        "strategy": "ranked_parameters_remeasured_on_target",
    }
