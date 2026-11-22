# Civilian inputs and frozen benchmark

`factory-small.json` is a readable five-job example: decode an image, inspect its
label and surface, inspect packaging, and review an anomaly. Two local compute
workers have different execution times and energy requirements. A mode declares
that a worker already holds the necessary data and can run the job; it does not
authorize a transfer or require Internet access.

`benchmark-protocol.json` declares twelve development and twelve held-out seed
cases across three sizes. `benchmark-inputs.json` freezes the generator source,
protocol, and exact canonical JSON digest of every scenario. These manifests were
fixed before measuring the solver. The held-out seeds are public; this is a
reproducibility holdout, not a hidden challenge evaluation.

Run commands from `civilian-inspection/`:

```sh
python3 planner.py examples/factory-small.json plan.json
python3 -m inspectionkit.generate small 11 > scenario.json
python3 benchmark.py --out reports/civilian-v1 --split all --time-limit 0.2 --max-states 5000
```

The benchmark requires a new output directory. It freezes all selected inputs
before running either planner, validates both schedules independently, and fails
if the improved planner loses score or any plan is invalid. It retains exact
scenario and plan JSON, per-case score breakdowns, source hashes, limits,
interpreter/platform details, descriptive timings, and a `SHA256SUMS` inventory.
An incomplete run has no successful final report.

These are non-actuating image-processing schedules. They neither submit to nor
measure the original fleet leaderboard; they do not establish job execution,
BRIDGE authorization, hardware energy measurements, or global solver optimality.
