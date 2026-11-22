#!/usr/bin/env python3
"""Plan non-actuating civilian image inspection on local compute workers.

Usage: python3 planner.py scenario.json plan.json

This entry point accepts only civilian-inspection/v1 scenarios. It does not
read fleet-kit scenarios or produce aircraft routes, reconnaissance or engagement
actions. The output is a proposed job schedule, not an execution authorization.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Optional, Sequence

from inspectionkit.io import ensure_distinct_paths, read_json, write_json_atomic
from inspectionkit.model import parse_scenario, plan_to_dict
from inspectionkit.solver import greedy_baseline, solve
from inspectionkit.validate import evaluate, evaluation_to_dict


def _time_limit(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("time limit must be a number") from exc
    if not math.isfinite(parsed) or not 0.01 <= parsed <= 50.0:
        raise argparse.ArgumentTypeError("time limit must be between 0.01 and 50 seconds")
    return parsed


def _state_limit(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max states must be an integer") from exc
    if not 1 <= parsed <= 200000:
        raise argparse.ArgumentTypeError("max states must be between 1 and 200000")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", help="civilian-inspection/v1 input JSON")
    parser.add_argument("plan", help="destination JSON; parent directory must exist")
    parser.add_argument("--time-limit", type=_time_limit, default=5.0,
                        help="search budget in seconds, 0.01..50 (default: 5)")
    parser.add_argument("--max-states", type=_state_limit, default=50000,
                        help="search work bound, 1..200000 (default: 50000)")
    parser.add_argument("--baseline", action="store_true", help="run the reference greedy scheduler")
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        ensure_distinct_paths(args.scenario, args.plan)
        scenario = parse_scenario(read_json(args.scenario))
        if args.baseline:
            plan = greedy_baseline(scenario)
            stats = {"algorithm": "greedy_baseline", "optimal_proven": False}
        else:
            result = solve(scenario, time_limit_s=args.time_limit, max_states=args.max_states)
            plan, stats = result.plan, result.stats
        evaluation = evaluate(scenario, plan)
        if not evaluation.valid:
            # An invalid candidate must never overwrite a previous good result.
            raise ValueError("planner produced an invalid schedule: {}".format(
                "; ".join(evaluation.violations)))
        report = {
            "schema": "civilian-inspection-run/v1",
            "scenario_id": scenario.scenario_id,
            "evaluation": evaluation_to_dict(evaluation),
            "search": stats,
            "planning_and_validation_elapsed_s": time.monotonic() - started,
            "execution_performed": False,
            "official_leaderboard_submission": False,
        }
        report_json = json.dumps(report, allow_nan=False, sort_keys=True)
        write_json_atomic(args.plan, plan_to_dict(plan))
        print(report_json, file=sys.stderr)
        return 0
    except (OSError, ValueError) as exc:
        print("planner: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
