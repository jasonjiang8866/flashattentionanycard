"""Bounded RunPod API access. Account credentials never enter the Pod."""

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

API = "https://api.runpod.io/v2"
REPOSITORY = "jasonjiang8866/flashattentionanycard"
MIG = "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb"
IMAGE = "ghcr.io/jasonjiang8866/flashattentionanycard-runpod@sha256:079df18c23caee6769ee0b70570c5c12863d7e5d0f847e7fb7cec1fb4d1136c3"
SOURCE = "14118cbec5aba88654232b1a2f232be4ca35d1db"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class RemoteHTTPError(RuntimeError):
    """Only a status and fixed categories may leave the HTTP boundary."""

    def __init__(self, status, categories):
        self.status = status
        self.categories = sorted(set(categories)) or ["unclassified"]
        super().__init__(f"Remote HTTP {status}; categories={','.join(self.categories)}")


def request(url, *, token, method="GET", body=None, limit=8_000_000):
    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "User-Agent": "flashattentionanycard-actions/0.1",
        },
        data=json.dumps(body).encode() if body is not None else None,
    )
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=30) as response:
            data = response.read(limit + 1)
        if len(data) > limit:
            raise RuntimeError("Response exceeds limit")
        return json.loads(data) if data else None
    except urllib.error.HTTPError as exc:
        # Only fixed diagnostic categories leave this handler, never provider text.
        raw = exc.read(16000).decode(errors="replace").lower()
        categories = [
            label
            for word, label in (
                ("capacity", "capacity"),
                ("stock", "capacity"),
                ("disk", "disk"),
                ("cpu", "cpu"),
                ("vcpu", "vcpu"),
                ("image", "image"),
                ("global", "global-volume"),
                ("mount", "mount"),
                ("port", "port"),
                ("entrypoint", "entrypoint"),
                ("command", "command"),
                ("argument", "argument"),
                ("not supported", "unsupported"),
                ("minimum", "minimum"),
                ("maximum", "maximum"),
                ("unavailable", "unavailable"),
                ("not found", "not-found"),
                ("sold out", "capacity"),
                ("no longer any", "capacity"),
                ("no available", "capacity"),
                ("not enough", "insufficient-resources"),
                ("insufficient", "insufficient-resources"),
                ("resource", "resource"),
                ("gpu", "gpu"),
                ("mig", "mig"),
                ("cuda", "cuda"),
                ("memory", "memory"),
                ("ram", "memory"),
                ("datacenter", "data-center"),
                ("data center", "data-center"),
                ("invalid", "invalid-input"),
                ("validation", "validation"),
                ("additional properties", "unknown-field"),
                ("credit", "credit"),
                ("balance", "credit"),
            )
            if word in raw
        ]
        raise RemoteHTTPError(exc.code, categories) from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("Remote request timeout/unavailable; outcome may be unknown") from None


def finite(value, minimum=0):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= minimum
    )


class Runpod:
    def __init__(self):
        self.key = os.environ.get("RUNPOD_API_KEY", "")
        if not self.key:
            raise RuntimeError("RUNPOD_API_KEY secret is missing")

    def call(self, path, method="GET", body=None):
        return request(API + path, token=self.key, method=method, body=body)

    def listing(self, path, key):
        items, seen = [], set()
        base = path
        while True:
            response = self.call(path)
            if not isinstance(response.get(key), list):
                raise RuntimeError("Invalid inventory")
            items.extend(response[key])
            page = response.get("pagination", {})
            if page.get("hasNextPage") is False:
                return items
            cursor = page.get("nextCursor")
            if not isinstance(cursor, str) or not cursor or cursor in seen:
                raise RuntimeError("Incomplete inventory")
            seen.add(cursor)
            path = base + "?" + urllib.parse.urlencode({"cursor": cursor})


def budget(api, reserve=1):
    if not finite(reserve, 0.01):
        raise ValueError("Invalid budget reserve")
    now = datetime.now(timezone.utc)
    result = request(
        "https://api.runpod.io/graphql",
        token=api.key,
        method="POST",
        body={
            "query": "query SmokeBudget($start: DateTime!, $end: DateTime!) { myself { clientBalance currentSpendPerHr dailyCharges(startDate: $start, endDate: $end) { amount } } }",
            "variables": {
                "start": now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat(),
                "end": now.isoformat(),
            },
        },
    )
    if result.get("errors"):
        raise RuntimeError("Budget query failed")
    account = result.get("data", {}).get("myself") or {}
    balance, spend, charges = (
        account.get("clientBalance"),
        account.get("currentSpendPerHr"),
        account.get("dailyCharges"),
    )
    if not finite(balance) or not finite(spend) or not isinstance(charges, list):
        raise RuntimeError("Missing budget information")
    amounts = [c.get("amount") for c in charges if isinstance(c, dict)]
    if len(amounts) != len(charges) or not all(finite(x, -math.inf) for x in amounts):
        raise RuntimeError("Invalid billing history")
    if balance < reserve or sum(abs(x) for x in amounts) + spend + reserve > 500:
        raise RuntimeError("Insufficient experiment reserve or $500 monthly budget headroom")
    return {
        "reserve_usd": reserve,
        "monthly_budget_usd": 500,
        "balance_sufficient": True,
        "month_headroom_sufficient": True,
        "month_basis": "UTC",
        "checked_utc": now.isoformat(),
        "scope": "reported charges; billing may lag; storage/request fees additional",
    }
