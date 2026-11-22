# Civilian inspection scheduler

`planner.py` schedules non-actuating image-inspection jobs on local compute
workers. It chooses which optional jobs to run, their worker, and their start
time while respecting dependencies, deadlines, worker eligibility, and energy
budgets. The objective is earned task reward minus energy charges and the
activation cost of each worker used.

This is a working civilian planning entry with its own explicit input contract.
The published fleet leaderboard expects fleet-kit scenarios and aircraft routes;
these job schedules are a different format. The measurements here do **not**
establish compatibility, a submission, or a score on that leaderboard. The
original `fleet-orchestration/fleet-kit/handin/planner.py` is unchanged.

## Run the entry point

Python 3.9 or newer, standard library only. No installation, service, network
connection, GPU, or fleet-kit package is needed.

```sh
cd civilian-inspection
python3 planner.py examples/factory-small.json plan.json --time-limit 1
python3 check_plan.py examples/factory-small.json plan.json
```

The required invocation shape is `python3 planner.py scenario.json plan.json`.
The output file contains only the schedule JSON. A separate JSON record on
stderr contains the independently recomputed score and search diagnostics.
The output's parent directory must exist. Input, validation, and serialization
failures preserve an existing output; a successful write uses atomic replacement.
The input and output cannot name the same file.

For the reference scheduler:

```sh
python3 planner.py examples/factory-small.json baseline.json --baseline
python3 check_plan.py examples/factory-small.json baseline.json
```

`check_plan.py` returns exit code 0 for a valid schedule, 1 for an invalid
schedule, and 2 for malformed input or a file error. Invalid schedules have
`score: null`; a valid empty schedule has score zero. This distinction prevents
invalid work being mistaken for a valid zero-reward result.

## What is scheduled

An inspection job is a local image-processing operation, such as decoding an
already available image, checking a panel, or assembling an inspection result.
Workers are local execution slots. This model contains no motion, trajectories,
actuator commands, target classification, or engagement actions.

A task's modes declare the workers on which its data and capability are already
available, together with the predicted duration and energy for each worker.
Dependencies express completion order; they do not schedule data transfer.
Dependent outputs must already be accessible under the declared local execution
arrangement, for example through shared storage on one host. A caller must not
use a mode that would require an unmodelled transfer. Worker availability is a
local resource fact; the planner does not probe cloud or Internet availability.

Every assignment is nonpreemptive. A worker processes at most one job at a time,
and each task appears at most once. A selected task requires all its predecessors
to finish before it starts. Jobs may remain unscheduled when they are infeasible
or when their cost exceeds their value.

The result is a proposed offline schedule. It does not execute jobs, reserve
resources, prove estimated durations, or grant runtime authorization. Integration
with BRIDGE would still have to authorize and validate each actual execution.
Live rescheduling, already-completed work, transport failures, and hardware
deadline guarantees are outside this version's contract.

## Exact input and scoring contract

See [the small example](examples/factory-small.json) for a complete scenario.
All keys are required, all undeclared keys are rejected, and quantities use
integer milliseconds, milli-watt-hours, or score units.

| Object | Required fields |
|---|---|
| Scenario | `schema`, `scenario_id`, `horizon_ms`, `energy_cost_per_mwh`, `workers`, `tasks` |
| Worker | `id`, `available`, `available_from_ms`, `energy_budget_mwh`, `activation_cost` |
| Task | `id`, `reward`, `release_ms`, `deadline_ms`, `dependencies`, `modes` |
| Worker mode | `worker_id`, `duration_ms`, `energy_mwh` |
| Plan | `schema`, `scenario_id`, `assignments` |
| Assignment | `task_id`, `worker_id`, `start_ms`, `finish_ms` |

Scenario schema: `civilian-inspection/v1`. Plan schema:
`civilian-inspection-plan/v1`. Schema mismatch is an error; there is no conversion
from the original fleet scenario or inference from renamed fields.

```text
score = sum(reward for assigned tasks)
        - energy_cost_per_mwh * sum(energy_mwh for selected modes)
        - sum(activation_cost for workers with at least one assignment)
```

Activation is charged once per used worker. Prerequisites consume time and
energy even when their own reward is zero. All score arithmetic is exact integer
arithmetic. The independent validator computes the objective from the scenario
and assignments; it never trusts the solver's score or completion claims.

Limits are 128 tasks, 16 workers, one mode per worker per task, and 2 MiB per
input file. At least one worker and zero or more tasks are required. IDs are
1–64 characters from `[A-Za-z0-9_.-]` and begin with an alphanumeric character.
Dependency cycles, unknown references, duplicate JSON keys, noninteger numbers,
booleans in integer fields, and nonfinite numbers are rejected. Feasibility is
separate from parsing: an unavailable worker or an impossible task deadline
does not make the scenario malformed.

The horizon is at most 1,000,000,000 ms. Rewards, activation costs, and the
energy coefficient are at most 1,000,000,000; individual energy values are at
most 1,000,000,000,000 mWh. These are parser bounds, not engineering estimates
or recommended operating values. Actual deployments must supply measured costs.

## Search and limits

The solver builds a deterministic greedy reference, explores several constructive
priorities, and then improves the incumbent with bounded search. Small instances
use exhaustive branch-and-bound when the search budget permits; larger instances
use a bounded beam. Zero-reward preprocessing can be selected when it enables
valuable dependent jobs.

The returned schedule is independently checked and cannot score below the
reference schedule returned by that same call. Global optimality is reported
only when exhaustive search finishes without truncation. A bounded beam or a
time/state cutoff is a heuristic result. See `search` in the stderr JSON for
the actual termination status.

`--time-limit` defaults to 5 seconds and accepts 0.01–50 seconds.
`--max-states` defaults to 50,000 and accepts 1–200,000. Time is measured with a
monotonic clock. These limits bound search effort; input handling, reference
construction, final validation, filesystem operations, host scheduling, and
diagnostic output also take time. This is not a hard real-time deadline promise.
The state limit makes an untruncated explored prefix reproducible; a wall-clock
cutoff can change that prefix across machines or loads.

## Reproduce the comparison

The generator and all 24 scenario identities were frozen before solver
measurements in [benchmark-protocol.json](examples/benchmark-protocol.json) and
[benchmark-inputs.json](examples/benchmark-inputs.json). There are 12 development
cases and 12 public seed holdouts across 8-, 24-, and 64-task workloads. The
holdout is our reproducibility split, not a hidden organizer evaluation set.

```sh
python3 -m inspectionkit.generate small 11 > scenario.json
python3 benchmark.py --out reports/civilian-v1 --split all --time-limit 0.2 --max-states 5000
python3 verify_benchmark.py reports/civilian-v1
```

The output directory must be new. Each case retains the exact input, both plans,
independent evaluations, score difference, timings, and search diagnostics.
The run also records source hashes, input hashes, the complete protocol, and
interpreter/platform metadata. A failed comparison leaves diagnostic files and
does not produce a successful report. All input identities are recorded before
the first optimizer call. Source changes during measurement invalidate the run.

The replay verifier checks file inventory and hashes, source and input identities,
every recorded plan, and the aggregate results without rerunning optimization.
The comparison is with a declared greedy reference on this finite generated set;
it is not a claim of superiority to other planners, unseen industrial workloads,
or entries on the original leaderboard. Timing is one observation per case.

## Recorded comparison: 22 November 2026

The [complete evidence package](evidence/2026-11-22/README.md) retains the frozen
run, both plans for every case, and independent replay results.

| Split | Cases | Greedy score | Planner score | Change | Wins / ties / losses |
|---|---:|---:|---:|---:|---:|
| Development | 12 | 14,918 | 18,240 | +22.27% | 12 / 0 / 0 |
| Public seed holdout | 12 | 12,493 | 17,083 | +36.74% | 11 / 1 / 0 |
| Total | 24 | 27,411 | 35,323 | +28.86% | 23 / 1 / 0 |

All 48 schedules were independently valid. The run used a 0.2-second search
budget and 5,000 nonbaseline transitions per case. Measured planner time was
2.256 seconds in total, with a 74.5 ms median and 201.3 ms maximum. These are
single-run observations on the recorded host, including baseline construction
inside `solve`; they are not a latency guarantee. All eight small cases completed
exact search; the sixteen larger cases remain heuristic results.

## Validation

```sh
python3 -m unittest discover -s tests -v
```

Tests cover strict parsing, independent feasibility and scoring, prerequisite
unlocks, small scheduling oracles, cutoff behavior, atomic output retention,
malformed and tampered evidence, and running a copied hand-in folder from an
unrelated working directory with third-party package loading disabled.
No Actions or CI workflow is added.

The published run passed 88 tests, including 88 tiny cases compared with an
independent exhaustive integer-time oracle. Tests ran on Python 3.12.14.
Python 3.9 syntax was checked, but a Python 3.9 interpreter was not executed.
The evidence package records the exact validation and source identities.
