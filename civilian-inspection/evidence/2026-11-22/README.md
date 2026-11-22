# Civilian inspection planning evidence — 22 November 2026

This is one frozen comparison of local image-inspection schedules against a
declared greedy reference. The input, output, and scores belong to the civilian
schema. No result was submitted to the original fleet leaderboard, and these
results do not establish compatibility or rank on that board.

## Results

| Split | Cases | Greedy score | Planner score | Gain | Wins / ties / losses |
|---|---:|---:|---:|---:|---:|
| Development | 12 | 14,918 | 18,240 | 3,322 (+22.27%) | 12 / 0 / 0 |
| Public seed holdout | 12 | 12,493 | 17,083 | 4,590 (+36.74%) | 11 / 1 / 0 |
| All cases | 24 | 27,411 | 35,323 | 7,912 (+28.86%) | 23 / 1 / 0 |

All 24 greedy and 24 planner schedules passed independent feasibility checks.
The fixed tiers contain 8, 24, and 64 tasks, each with four development and four
held-out seeds. Generator source, protocol, and all scenario identities were
fixed before observing solver measurements. These seeds are public; the split
is not the organizer's hidden evaluation.

The optimizer used 0.2 seconds and at most 5,000 nonbaseline transitions per
case. Its total measured runtime was 2.256 seconds (median 74.5 ms; maximum
201.3 ms). Eight small cases completed exact search. Eight large cases reached
the time cutoff. All sixteen medium/large results remain heuristics, including
those whose bounded beam exhausted early. No global optimum is claimed for them.

Times were observed once using Python 3.12.14 on the host identified in
`report.json`; host load and wall-clock cutoffs can affect results. This is a
comparison with one reference heuristic, not evidence of superiority over other
planners or a population estimate for industrial workloads.

## Files and replay

- `report.json`: all per-case results and summaries for convenient review.
- `campaign.tar.xz`: the complete run, including all exact scenarios, both plans,
  per-case evaluations, frozen protocol, source/input identities, and checksums.
- `verification.json`: independent replay of the original run directory.
- `portable-verification.json`: replay after extracting the published archive
  into a different directory.
- `validation.json` and `tests.log`: test commands, counts, runtime, and limits.
- `factory-small-*.json`: execution of the documented sample CLI and checker.
- `maximum-dimensions.json`: a 128-task, 16-worker, 2,048-mode input check. All
  128 tasks were scheduled and validated. Its 0.01-second search budget expired
  during reference construction; total observed time was 177.6 ms, illustrating
  the documented startup/final-validation overhead rather than a hard deadline.
- `SHA256SUMS`: the exact contents of this published package.

The campaign archive is 41,448 bytes, SHA-256:

```text
6b9617f80f97af719c938efad2abeee7c9148a05b4fb78ee8e2f1f9c1f33a794
```

From `civilian-inspection/` on this PR's source revision:

```sh
mkdir -p reports
mkdir reports/replay-20261122
tar -xf evidence/2026-11-22/campaign.tar.xz -C reports/replay-20261122
python3 verify_benchmark.py reports/replay-20261122/campaign
```

Replay verified 100 inventoried files and independently recomputed all 48 plan
evaluations and aggregate scores without running either optimizer. The archive
also contains its `SHA256SUMS`, which is not recursively hashed into itself.
The verifier rejects missing/extra files, symlinks, unsafe paths, source/input
changes, and false result claims even when their file checksums are recomputed.
The current checkout is its source reference; this proves consistency with that
source and those inputs, not an external attestation of source execution.

## Validation and limits

`python3 -m unittest discover -s tests -v`: **88 tests passed, zero skipped**.
The solver tests include **88 independently enumerated tiny scheduling cases**,
with every integer start time, eligible worker, and omission considered. They
also exercise costly prerequisite unlocks, activation sharing, cutoff
fallback, exact arithmetic, and input-order stability.

Other checks cover strict schemas, malformed/ambiguous JSON, invalid plan
rejection, preserved previous output, standalone execution from a copied folder
using `python -S`, and rehashed evidence tampering. All 17 Python source/test
files parsed using the Python 3.9 syntax grammar; runtime validation used Python
3.12.14, not a Python 3.9 interpreter.

The replay verifier does not authenticate observed timings, optimizer execution,
or optimality claims. No image jobs, robots, physical systems, BRIDGE execution
authorization, or leaderboard submission were exercised. Task durations and
energy are declared scenario inputs, not hardware measurements.
