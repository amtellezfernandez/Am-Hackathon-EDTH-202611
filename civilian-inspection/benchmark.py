#!/usr/bin/env python3
"""Run the fixed civilian inspection benchmark and retain reviewable evidence.

This is an offline scheduling experiment, not the original fleet leaderboard.
No job execution, physical motion, authorization, or network activity is tested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from inspectionkit.generate import Case, cases, generate, protocol
from inspectionkit.model import plan_to_dict, scenario_to_dict
from inspectionkit.solver import greedy_baseline, solve
from inspectionkit.validate import evaluate, evaluation_to_dict


class BenchmarkError(RuntimeError):
    """A comparison failed validation; partial evidence is not a success."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> str:
    raw = _json_bytes(value)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _source_inventory(root: Path) -> Dict[str, str]:
    paths = [root / name for name in
             ("benchmark.py", "planner.py", "check_plan.py", "verify_benchmark.py")]
    paths.extend(sorted((root / "inspectionkit").glob("*.py")))
    for name in ("benchmark-protocol.json", "benchmark-inputs.json"):
        frozen = root / "examples" / name
        if frozen.is_file():
            paths.append(frozen)
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths if path.is_file()
    }


def _summarize(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(results)
    base = sum(row["baseline"]["evaluation"]["score"] for row in rows)
    improved = sum(row["planner"]["evaluation"]["score"] for row in rows)
    deltas = [row["delta_score"] for row in rows]
    runtime = [row["planner"]["elapsed_s"] for row in rows]
    return {
        "cases": len(rows),
        "valid_baseline_plans": sum(row["baseline"]["evaluation"]["valid"] for row in rows),
        "valid_planner_plans": sum(row["planner"]["evaluation"]["valid"] for row in rows),
        "baseline_score": base,
        "planner_score": improved,
        "delta_score": improved - base,
        "score_ratio": improved / base if base > 0 else None,
        "wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "baseline_elapsed_s": sum(row["baseline"]["elapsed_s"] for row in rows),
        "planner_elapsed_s": sum(runtime),
        "planner_median_elapsed_s": statistics.median(runtime) if runtime else None,
        "planner_max_elapsed_s": max(runtime) if runtime else None,
    }


def _run_case(case: Case, scenario: Any, case_dir: Path,
              time_limit_s: float, max_states: int) -> Dict[str, Any]:
    started = time.perf_counter()
    baseline = greedy_baseline(scenario)
    baseline_elapsed = time.perf_counter() - started
    baseline_eval = evaluate(scenario, baseline)
    baseline_hash = _write_json(case_dir / "baseline-plan.json", plan_to_dict(baseline))

    started = time.perf_counter()
    solution = solve(scenario, time_limit_s=time_limit_s, max_states=max_states)
    solution_elapsed = time.perf_counter() - started
    solution_eval = evaluate(scenario, solution.plan)
    solution_hash = _write_json(case_dir / "planner-plan.json", plan_to_dict(solution.plan))
    baseline_matches = plan_to_dict(solution.baseline) == plan_to_dict(baseline)
    result = {
        "scenario_id": case.scenario_id,
        "split": case.split,
        "tier": case.tier,
        "seed": case.seed,
        "baseline": {
            "elapsed_s": baseline_elapsed,
            "plan_sha256": baseline_hash,
            "evaluation": evaluation_to_dict(baseline_eval),
        },
        "planner": {
            "elapsed_s": solution_elapsed,
            "plan_sha256": solution_hash,
            "evaluation": evaluation_to_dict(solution_eval),
            "stats": solution.stats,
            "baseline_identity_matches": baseline_matches,
        },
        "delta_score": (
            solution_eval.score - baseline_eval.score
            if baseline_eval.valid and solution_eval.valid else None
        ),
    }
    _write_json(case_dir / "result.json", result)
    if not baseline_eval.valid or not solution_eval.valid:
        raise BenchmarkError("{}: independent validation failed; inspect result.json".format(case.scenario_id))
    if not baseline_matches:
        raise BenchmarkError("{}: solver baseline differs from independently run baseline".format(case.scenario_id))
    if solution_eval.score < baseline_eval.score:
        raise BenchmarkError("{}: planner score is below baseline".format(case.scenario_id))
    return result


def run_benchmark(output: Path, *, split: str = "development",
                  time_limit_s: float = 0.2, max_states: int = 5000) -> Dict[str, Any]:
    """Retain every input before evaluating the fixed manifest exactly once.

    An existing output directory is rejected rather than overwritten. An error
    leaves partial case files for diagnosis, without a successful report.
    """
    selected = cases(split)
    if (isinstance(time_limit_s, bool) or not isinstance(time_limit_s, (int, float))
            or not math.isfinite(time_limit_s) or not 0.01 <= time_limit_s <= 50):
        raise ValueError("time_limit_s must be finite and between 0.01 and 50")
    if type(max_states) is not int or not 1 <= max_states <= 200000:
        raise ValueError("max_states must be an integer between 1 and 200000")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parent
    source_hashes = _source_inventory(source_root)
    frozen = source_root / "examples" / "benchmark-protocol.json"
    if not frozen.is_file() or json.loads(frozen.read_text(encoding="utf-8")) != protocol():
        raise BenchmarkError("frozen examples/benchmark-protocol.json is missing or differs from generator")
    frozen_inputs_path = source_root / "examples" / "benchmark-inputs.json"
    if not frozen_inputs_path.is_file():
        raise BenchmarkError("frozen examples/benchmark-inputs.json is missing")
    frozen_inputs = json.loads(frozen_inputs_path.read_text(encoding="utf-8"))
    if frozen_inputs.get("generator_sha256") != source_hashes["inspectionkit/generate.py"]:
        raise BenchmarkError("generator source differs from the frozen input manifest")
    if frozen_inputs.get("protocol_sha256") != hashlib.sha256(frozen.read_bytes()).hexdigest():
        raise BenchmarkError("protocol identity differs from the frozen input manifest")
    expected_inputs = {entry["scenario_id"]: entry["sha256"] for entry in frozen_inputs["cases"]}
    _write_json(output / "protocol.json", protocol())
    _write_json(output / "sources.json", source_hashes)
    inputs: List[Dict[str, Any]] = []
    prepared = []
    for case in selected:
        scenario = generate(case.tier, case.seed, split=case.split)
        case_dir = output / "cases" / case.scenario_id
        case_dir.mkdir(parents=True)
        digest = _write_json(case_dir / "scenario.json", scenario_to_dict(scenario))
        if expected_inputs.get(case.scenario_id) != digest:
            raise BenchmarkError("{}: generated input differs from frozen manifest".format(case.scenario_id))
        inputs.append({"scenario_id": case.scenario_id, "split": case.split,
                       "tier": case.tier, "seed": case.seed, "sha256": digest})
        prepared.append((case, scenario, case_dir))
    # All inputs and identities are fixed before the first call to either solver.
    input_manifest_hash = _write_json(output / "inputs.json", inputs)
    results = [_run_case(case, scenario, path, float(time_limit_s), max_states)
               for case, scenario, path in prepared]
    if _source_inventory(source_root) != source_hashes:
        raise BenchmarkError("benchmark source changed during the run")
    report = {
        "schema": "civilian-inspection-benchmark/v1",
        "status": "passed",
        "scope": "Offline schedules of local industrial image-processing jobs",
        "leaderboard_compatible": False,
        "job_execution_tested": False,
        "selected_split": split,
        "inputs_manifest_sha256": input_manifest_hash,
        "limits": {"time_limit_s": float(time_limit_s), "max_states": max_states},
        "runtime": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "perf_counter_resolution_s": time.get_clock_info("perf_counter").resolution,
        },
        "summary": _summarize(results),
        "by_split": {
            name: _summarize(row for row in results if row["split"] == name)
            for name in ("development", "heldout") if any(row["split"] == name for row in results)
        },
        "by_tier": {
            name: _summarize(row for row in results if row["tier"] == name)
            for name in ("small", "medium", "large")
        },
        "cases": results,
        "limitations": [
            "This does not submit to or measure the original fleet leaderboard.",
            "Schedules do not prove that image jobs ran or were authorized.",
            "One deterministic generated instance per manifest case; no population claim.",
            "Wall-clock cutoffs and host load can change explored search prefixes.",
            "Runtime is measured once and includes baseline construction inside solve.",
        ],
    }
    _write_json(output / "report.json", report)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    checksums = "".join(
        "{}  {}\n".format(hashlib.sha256(path.read_bytes()).hexdigest(), path.relative_to(output).as_posix())
        for path in files
    )
    (output / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new evidence directory; never overwritten")
    parser.add_argument("--split", choices=("development", "heldout", "all"), default="development")
    parser.add_argument("--time-limit", type=float, default=0.2, help="per-case search budget in seconds")
    parser.add_argument("--max-states", type=int, default=5000)
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args.out, split=args.split, time_limit_s=args.time_limit,
                               max_states=args.max_states)
    except (ValueError, OSError, BenchmarkError) as exc:
        parser.exit(2, "benchmark: {}\n".format(exc))
    print(json.dumps({"evidence": str(args.out), "summary": report["summary"],
                      "by_split": report["by_split"]}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
