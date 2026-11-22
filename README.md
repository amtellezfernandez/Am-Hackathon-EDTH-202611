# AA-Hackathon-EDTH-2026

Starter material from Alta Ares for the EDTH Paris hackathon (20 to 22 November 2026). One folder per subject; each has its own README with everything you need.

| Subject | What you build | Folder |
|---|---|---|
| Fleet Orchestration | A planner that tasks a mixed fleet of scouts and effector drones against several contacts under ten written rules. | [`fleet-orchestration/`](fleet-orchestration/) |
| Search & Destroy | A 30 Hz FPV flight policy that finds an orange box hidden in a simulated terrain and touches it without hitting anything else, using a detector, noisy depth and drifting odometry. | [`search-destroy/`](search-destroy/) |

Everything in this repository is synthetic.

## Civilian inspection planner

This fork also includes a [civilian inspection planner](civilian-inspection/README.md)
for scheduling non-actuating image-analysis jobs across local compute workers.
It provides a runnable `planner.py`, an independent validator, bounded search,
and a reproducible comparison with a greedy scheduler.

Its input schema describes local compute jobs. The existing fleet leaderboard
accepts fleet-kit scenarios and routes; the civilian benchmark is a separate
measurement and does not establish a result on that board.

