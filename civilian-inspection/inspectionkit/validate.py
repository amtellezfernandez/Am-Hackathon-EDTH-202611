"""Independent feasibility checks and exact scoring for inspection schedules.

This module does not import the solver or trust solver-reported totals. It checks
the complete schedule and recomputes the score from the supplied scenario.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .model import (
    Plan, Scenario, ValidationError, parse_plan, parse_scenario,
    plan_to_dict, scenario_to_dict,
)


@dataclass(frozen=True)
class Evaluation:
    valid: bool
    score: Optional[int]
    reward: int
    energy_mwh: int
    energy_cost: int
    activation_cost: int
    completed_tasks: Tuple[str, ...]
    unscheduled_tasks: Tuple[str, ...]
    violations: Tuple[str, ...]


def _invalid(violations: list, task_ids: tuple = ()) -> Evaluation:
    # Invalid schedules earn no partial credit; None cannot be confused with a
    # valid empty schedule's zero score. Diagnostics are the only trusted result.
    return Evaluation(False, None, 0, 0, 0, 0, (), task_ids, tuple(violations))


def evaluate(scenario: Scenario, plan: Plan) -> Evaluation:
    """Validate every assignment and recompute its exact objective.

    Direct dataclass construction is also checked. This prevents an internal
    caller from bypassing the same type and range checks used at the JSON boundary.
    """
    if not isinstance(scenario, Scenario):
        return _invalid(["scenario must be a Scenario"])
    try:
        scenario = parse_scenario(scenario_to_dict(scenario))
    except (ValidationError, AttributeError, TypeError) as error:
        return _invalid([f"invalid scenario: {error}"])
    task_ids = tuple(sorted(task.id for task in scenario.tasks))
    if not isinstance(plan, Plan):
        return _invalid(["plan must be a Plan"], task_ids)
    try:
        plan = parse_plan(plan_to_dict(plan))
    except (ValidationError, AttributeError, TypeError) as error:
        return _invalid([f"invalid plan: {error}"], task_ids)

    violations = []
    if plan.scenario_id != scenario.scenario_id:
        violations.append("plan.scenario_id does not match scenario.scenario_id")
    tasks = {task.id: task for task in scenario.tasks}
    workers = {worker.id: worker for worker in scenario.workers}
    by_task = {}
    by_worker: Dict[str, list] = {worker_id: [] for worker_id in workers}
    worker_energy = {worker_id: 0 for worker_id in workers}
    reward = 0

    for assignment in plan.assignments:
        name = assignment.task_id
        if name in by_task:
            violations.append(f"task {name!r} is assigned more than once")
        else:
            by_task[name] = assignment
        task = tasks.get(name)
        worker = workers.get(assignment.worker_id)
        if task is None:
            violations.append(f"unknown task {name!r}")
        if worker is None:
            violations.append(f"unknown worker {assignment.worker_id!r}")
        if task is None or worker is None:
            continue
        by_worker[worker.id].append(assignment)
        mode = next((mode for mode in task.modes if mode.worker_id == worker.id), None)
        if mode is None:
            violations.append(f"task {name!r} has no mode for worker {worker.id!r}")
        else:
            if assignment.finish_ms - assignment.start_ms != mode.duration_ms:
                violations.append(f"task {name!r} duration does not match its worker mode")
            worker_energy[worker.id] += mode.energy_mwh
        if not worker.available:
            violations.append(f"task {name!r} uses unavailable worker {worker.id!r}")
        if assignment.start_ms < worker.available_from_ms:
            violations.append(f"task {name!r} starts before worker {worker.id!r} is available")
        if assignment.start_ms < task.release_ms:
            violations.append(f"task {name!r} starts before its release")
        if assignment.finish_ms <= assignment.start_ms:
            violations.append(f"task {name!r} must have positive elapsed time")
        if assignment.finish_ms > task.deadline_ms:
            violations.append(f"task {name!r} finishes after its deadline")
        if assignment.finish_ms > scenario.horizon_ms or assignment.start_ms > scenario.horizon_ms:
            violations.append(f"task {name!r} lies outside the scenario horizon")
        reward += task.reward

    for assignment in plan.assignments:
        task = tasks.get(assignment.task_id)
        if task is None:
            continue
        for dependency in task.dependencies:
            predecessor = by_task.get(dependency)
            if predecessor is None:
                violations.append(f"task {task.id!r} is missing dependency {dependency!r}")
            elif predecessor.finish_ms > assignment.start_ms:
                violations.append(f"task {task.id!r} starts before dependency {dependency!r} finishes")

    for worker_id, assignments in by_worker.items():
        assignments.sort(key=lambda item: (item.start_ms, item.finish_ms, item.task_id))
        previous = None
        for assignment in assignments:
            if previous is not None and assignment.start_ms < previous.finish_ms:
                violations.append(
                    f"worker {worker_id!r} overlaps tasks {previous.task_id!r} and {assignment.task_id!r}"
                )
            if previous is None or assignment.finish_ms > previous.finish_ms:
                previous = assignment
        if worker_energy[worker_id] > workers[worker_id].energy_budget_mwh:
            violations.append(f"worker {worker_id!r} exceeds its energy budget")

    if violations:
        return _invalid(violations, task_ids)
    energy = sum(worker_energy.values())
    energy_cost = scenario.energy_cost_per_mwh * energy
    activation_cost = sum(workers[worker_id].activation_cost
                          for worker_id, assignments in by_worker.items() if assignments)
    completed = tuple(sorted(by_task))
    unscheduled = tuple(task_id for task_id in task_ids if task_id not in by_task)
    return Evaluation(True, reward - energy_cost - activation_cost, reward, energy,
                      energy_cost, activation_cost, completed, unscheduled, ())


def evaluation_to_dict(evaluation: Evaluation) -> Dict[str, Any]:
    return {
        "valid": evaluation.valid,
        "score": evaluation.score,
        "reward": evaluation.reward,
        "energy_mwh": evaluation.energy_mwh,
        "energy_cost": evaluation.energy_cost,
        "activation_cost": evaluation.activation_cost,
        "completed_tasks": list(evaluation.completed_tasks),
        "unscheduled_tasks": list(evaluation.unscheduled_tasks),
        "violations": list(evaluation.violations),
    }
