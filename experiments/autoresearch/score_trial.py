#!/usr/bin/env python3
"""
Score a trial run from metrics files in a run directory.

This script is intentionally lightweight and robust:
- works with an explicit metrics file, or
- auto-discovers known ADN/MATDO-E experiment outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import operator
from pathlib import Path
from typing import Any, Callable


OPS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_known_metrics(run_dir: Path) -> dict[str, float]:
    """
    Best-effort extraction from known report files in this repo.
    """
    metrics: dict[str, float] = {}

    # throughput_result.json
    for path in run_dir.rglob("throughput_result.json"):
        data = _load_json(path) or {}
        tps = data.get("throughput_tokens_per_sec")
        p99 = data.get("p99_latency_ms")
        mem = data.get("peak_memory_gb") or data.get("peak_allocated_gb")
        if isinstance(tps, (int, float)):
            metrics["throughput_tokens_per_sec"] = float(tps)
        if isinstance(p99, (int, float)):
            metrics["p99_latency_ms"] = float(p99)
        if isinstance(mem, (int, float)):
            metrics["peak_memory_gb"] = float(mem)

    # needle_haystack_real.json
    for path in run_dir.rglob("needle_haystack_real.json"):
        data = _load_json(path) or {}
        results = data.get("results", {})
        for key in ("131072", 131072):
            row = results.get(key) if isinstance(results, dict) else None
            if not row and isinstance(results, dict):
                row = results.get(str(key))
            if isinstance(row, dict):
                acc = row.get("accuracy")
                if isinstance(acc, (int, float)):
                    metrics["needle_128k_accuracy"] = float(acc)
                    break

    # p0_summary.json style pass counts
    for path in run_dir.rglob("p0_summary.json"):
        data = _load_json(path) or {}
        checks = data.get("checks")
        if isinstance(checks, dict):
            total = 0
            passed = 0
            for v in checks.values():
                if isinstance(v, dict) and "passed" in v:
                    total += 1
                    passed += int(bool(v["passed"]))
            if total > 0:
                metrics["p0_pass_rate"] = passed / total

    # table6_math-like accuracy key if present
    for path in run_dir.rglob("*.json"):
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        for k in ("math_accuracy", "accuracy", "val_accuracy"):
            v = data.get(k)
            if isinstance(v, (int, float)):
                # keep only if a more specific key wasn't already set
                metrics.setdefault("math_accuracy", float(v))

    return metrics


def _parse_constraint(text: str) -> tuple[str, str, float]:
    for op in ("<=", ">=", "==", "<", ">"):
        if op in text:
            left, right = text.split(op, 1)
            return left.strip(), op, float(right.strip())
    raise ValueError(f"Invalid constraint: {text}")


def _evaluate_constraints(
    metrics: dict[str, float],
    constraints: list[str],
) -> tuple[bool, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    ok_all = True
    for raw in constraints:
        key, op, bound = _parse_constraint(raw)
        value = metrics.get(key)
        if value is None or not math.isfinite(value):
            ok = False
        else:
            ok = OPS[op](value, bound)
        details.append(
            {
                "constraint": raw,
                "metric": key,
                "op": op,
                "bound": bound,
                "value": value,
                "passed": ok,
            }
        )
        ok_all = ok_all and ok
    return ok_all, details


def main() -> int:
    parser = argparse.ArgumentParser(description="Score an autoresearch trial.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics-file", type=Path, default=None)
    parser.add_argument("--primary-metric", type=str, default="needle_128k_accuracy")
    parser.add_argument(
        "--primary-direction",
        choices=("max", "min"),
        default="max",
    )
    parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="Constraint like p99_latency_ms<=400 (can repeat).",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metrics: dict[str, float] = {}

    if args.metrics_file:
        data = _load_json(args.metrics_file.resolve()) or {}
        for k, v in data.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
    else:
        metrics = _extract_known_metrics(run_dir)

    primary_value = metrics.get(args.primary_metric)
    if primary_value is None or not math.isfinite(primary_value):
        score = float("-inf")
        primary_found = False
    else:
        primary_found = True
        score = primary_value if args.primary_direction == "max" else -primary_value

    constraints_ok, constraint_details = _evaluate_constraints(metrics, args.constraint)
    valid = primary_found and constraints_ok

    result = {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "primary_metric": args.primary_metric,
        "primary_direction": args.primary_direction,
        "primary_value": primary_value,
        "score": score,
        "primary_found": primary_found,
        "constraints_ok": constraints_ok,
        "constraint_details": constraint_details,
        "valid": valid,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
