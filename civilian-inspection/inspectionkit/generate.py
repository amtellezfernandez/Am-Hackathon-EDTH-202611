"""Seeded industrial image-job workloads; no physical action is represented.

The public development and held-out seed lists are fixed independently of solver
results. The latter is a reproducibility holdout, not a secret evaluation set.
Each mode means that a worker already has the required data and capability.
There is deliberately no network-transfer or actuator model.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .model import Scenario, parse_scenario, scenario_to_dict


DEVELOPMENT_SEEDS = (11, 23, 37, 53)
HELDOUT_SEEDS = (1009, 1019, 1031, 1049)
TIERS = ("small", "medium", "large")
SIZES = {
    "small": (8, 2, 24000),
    "medium": (24, 3, 48000),
    "large": (64, 4, 90000),
}


@dataclass(frozen=True)
class Case:
    split: str
    tier: str
    seed: int

    @property
    def scenario_id(self) -> str:
        return "inspection-{}-{}-{}".format(self.split, self.tier, self.seed)


def cases(split: str = "development") -> Tuple[Case, ...]:
    """Return the entire named manifest in a stable order."""
    if split not in ("development", "heldout", "all"):
        raise ValueError("split must be development, heldout, or all")
    selections = []
    if split in ("development", "all"):
        selections.append(("development", DEVELOPMENT_SEEDS))
    if split in ("heldout", "all"):
        selections.append(("heldout", HELDOUT_SEEDS))
    return tuple(Case(name, tier, seed) for name, seeds in selections
                 for tier in TIERS for seed in seeds)


def protocol() -> Dict[str, Any]:
    """Machine-readable, fixed benchmark design, without machine-specific data."""
    return {
        "schema": "civilian-inspection-benchmark-protocol/v1",
        "purpose": "Non-actuating local industrial image-processing schedules",
        "leaderboard_compatible": False,
        "holdout_note": (
            "Public seed holdout fixed before solver measurements; this is not "
            "the original competition validation set or its hidden evaluation."
        ),
        "score": (
            "assigned task rewards - energy_cost_per_mwh * assigned energy_mwh "
            "- activation_cost of each used worker"
        ),
        "tiers": {
            name: {"tasks": size[0], "workers": size[1], "horizon_ms": size[2]}
            for name, size in SIZES.items()
        },
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "heldout_seeds": list(HELDOUT_SEEDS),
        "cases": [
            {"split": case.split, "tier": case.tier, "seed": case.seed,
             "scenario_id": case.scenario_id}
            for case in cases("all")
        ],
        "comparison": "greedy_baseline and solve on identical parsed scenarios",
        "acceptance": "every plan valid, improved score at least baseline score",
        "default_time_limit_s": 0.2,
        "default_max_states": 5000,
        "repetitions": 1,
        "timing_note": "Descriptive single-process timings, not a latency benchmark",
    }


def generate(tier: str, seed: int, *, split: str = "development") -> Scenario:
    """Create a bounded, deterministic case with heterogeneous local workers.

    Jobs are optional. Preprocessing jobs can have no direct reward but unlock
    useful later image analyses. Some allowed modes intentionally lie on an
    unavailable worker: eligibility does not imply current availability.
    """
    if tier not in SIZES:
        raise ValueError("tier must be small, medium, or large")
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be an integer between 0 and 2**31 - 1")
    if split not in ("development", "heldout"):
        raise ValueError("split must be development or heldout")
    rng = random.Random(seed)
    n_tasks, n_workers, horizon = SIZES[tier]
    worker_ids = ["edge-{}".format(i + 1) for i in range(n_workers)]
    # Independent variation in local compute throughput and energy demand.
    speed_percent = [rng.randint(75, 135) for _ in worker_ids]
    energy_percent = [rng.randint(70, 140) for _ in worker_ids]
    workers = []
    for i, wid in enumerate(worker_ids):
        workers.append({
            "id": wid,
            "available": not (i == n_workers - 1 and seed % 4 == 1),
            "available_from_ms": rng.choice((0, 0, horizon // 12)),
            "energy_budget_mwh": max(40, n_tasks * rng.randint(10, 15) // n_workers),
            "activation_cost": rng.randint(5, 25),
        })
    tasks = []
    for i in range(n_tasks):
        # Each batch starts with a non-rewarding preprocessing task, followed
        # by analyses of images already resident on eligible workers.
        is_preprocess = i % 8 == 0
        task_id = "{}-{:03d}".format("preprocess" if is_preprocess else "inspect", i)
        duration = rng.randint(900, 4500) if is_preprocess else rng.randint(2500, 10500)
        energy = rng.randint(3, 10) if is_preprocess else rng.randint(8, 28)
        reward = 0 if is_preprocess else rng.randint(35, 180)
        release = rng.randrange(0, max(1, horizon // 3), 500)
        deadline = min(horizon, release + rng.randint(horizon // 4, horizon))
        dependencies = []
        if not is_preprocess and i % 8 in (1, 2, 3):
            dependencies.append(tasks[i - i % 8]["id"])
        elif not is_preprocess and i > 1 and rng.random() < 0.25:
            dependencies.append(tasks[rng.randrange(i)]["id"])
        if dependencies:
            # A deadline wide enough to admit some dependency bundles, while
            # retaining real capacity and alternative-mode tradeoffs.
            deadline = max(deadline, 3 * horizon // 4)
        eligible = rng.sample(range(n_workers), rng.randint(1, n_workers))
        modes = []
        for w in sorted(eligible):
            modes.append({
                "worker_id": worker_ids[w],
                "duration_ms": max(1, duration * 100 // speed_percent[w]),
                "energy_mwh": max(1, energy * energy_percent[w] // 100),
            })
        tasks.append({
            "id": task_id,
            "reward": reward,
            "release_ms": release,
            "deadline_ms": deadline,
            "dependencies": dependencies,
            "modes": modes,
        })
    return parse_scenario({
        "schema": "civilian-inspection/v1",
        "scenario_id": Case(split, tier, seed).scenario_id,
        "horizon_ms": horizon,
        "energy_cost_per_mwh": 1,
        "workers": workers,
        "tasks": tasks,
    })


def main(argv: Optional[List[str]] = None) -> int:
    """Write one reproducible scenario to standard output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", choices=TIERS)
    parser.add_argument("seed", type=int)
    parser.add_argument("--split", choices=("development", "heldout"), default="development")
    args = parser.parse_args(argv)
    try:
        scenario = generate(args.tier, args.seed, split=args.split)
    except ValueError as exc:
        parser.exit(2, "generate: {}\n".format(exc))
    print(json.dumps(scenario_to_dict(scenario), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
