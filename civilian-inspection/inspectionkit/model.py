"""Strict contracts for non-actuating industrial inspection jobs.

Modes describe processing data already available to a local worker. A scenario
does not describe vehicles, trajectories, targets, transfers, or actuator commands.
All quantities are integers so validation and scoring are exact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple


SCENARIO_SCHEMA = "civilian-inspection/v1"
PLAN_SCHEMA = "civilian-inspection-plan/v1"
MAX_TASKS = 128
MAX_WORKERS = 16
MAX_TIME_MS = 1_000_000_000
MAX_SCORE_VALUE = 1_000_000_000
MAX_ENERGY_MWH = 1_000_000_000_000
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class ValidationError(ValueError):
    """An input violates the declared data contract."""


@dataclass(frozen=True)
class Mode:
    worker_id: str
    duration_ms: int
    energy_mwh: int


@dataclass(frozen=True)
class Task:
    id: str
    reward: int
    release_ms: int
    deadline_ms: int
    dependencies: Tuple[str, ...]
    modes: Tuple[Mode, ...]


@dataclass(frozen=True)
class Worker:
    id: str
    available: bool
    available_from_ms: int
    energy_budget_mwh: int
    activation_cost: int


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    horizon_ms: int
    energy_cost_per_mwh: int
    workers: Tuple[Worker, ...]
    tasks: Tuple[Task, ...]


@dataclass(frozen=True)
class Assignment:
    task_id: str
    worker_id: str
    start_ms: int
    finish_ms: int


@dataclass(frozen=True)
class Plan:
    scenario_id: str
    assignments: Tuple[Assignment, ...]


def _object(value: Any, keys: Tuple[str, ...], path: str) -> Dict[str, Any]:
    if type(value) is not dict:
        raise ValidationError(f"{path} must be an object")
    if any(type(key) is not str for key in value):
        raise ValidationError(f"{path} keys must be strings")
    missing = sorted(set(keys) - value.keys())
    unknown = sorted(value.keys() - set(keys))
    if missing:
        raise ValidationError(f"{path} missing keys: {', '.join(missing)}")
    if unknown:
        raise ValidationError(f"{path} unknown keys: {', '.join(unknown)}")
    return value


def _array(value: Any, path: str, maximum: int, minimum: int = 0) -> list:
    if type(value) is not list:
        raise ValidationError(f"{path} must be an array")
    if not minimum <= len(value) <= maximum:
        raise ValidationError(f"{path} must contain {minimum}..{maximum} items")
    return value


def _integer(value: Any, path: str, maximum: int, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValidationError(f"{path} must be an integer in {minimum}..{maximum}")
    return value


def _identifier(value: Any, path: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValidationError(
            f"{path} must match [A-Za-z0-9][A-Za-z0-9_.-]{{0,63}}"
        )
    return value


def _unique(value: str, seen: set, path: str) -> None:
    if value in seen:
        raise ValidationError(f"{path} duplicates {value!r}")
    seen.add(value)


def _check_dependencies(tasks: Tuple[Task, ...]) -> None:
    """Validate references and detect cycles with an iterative topological walk."""
    by_id = {task.id: task for task in tasks}
    incoming = {task.id: len(task.dependencies) for task in tasks}
    children = {task.id: [] for task in tasks}
    for task in tasks:
        for dependency in task.dependencies:
            if dependency == task.id:
                raise ValidationError(f"task {task.id!r} cannot depend on itself")
            if dependency not in by_id:
                raise ValidationError(
                    f"task {task.id!r} has unknown dependency {dependency!r}"
                )
            children[dependency].append(task.id)
    ready = [task.id for task in tasks if incoming[task.id] == 0]
    visited = 0
    while ready:
        current = ready.pop()
        visited += 1
        for child in children[current]:
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
    if visited != len(tasks):
        raise ValidationError("task dependencies contain a cycle")


def parse_scenario(data: Any) -> Scenario:
    """Parse a bounded scenario; reject unsupported schemas and undeclared fields.

    Feasibility is not a schema condition: a task may have no available worker or
    a deadline it cannot meet. Such a task must remain unscheduled.
    """
    obj = _object(data, (
        "schema", "scenario_id", "horizon_ms", "energy_cost_per_mwh", "workers", "tasks"
    ), "scenario")
    if obj["schema"] != SCENARIO_SCHEMA or type(obj["schema"]) is not str:
        raise ValidationError(f"scenario.schema must be {SCENARIO_SCHEMA!r}")
    scenario_id = _identifier(obj["scenario_id"], "scenario.scenario_id")
    horizon = _integer(obj["horizon_ms"], "scenario.horizon_ms", MAX_TIME_MS, 1)
    energy_cost = _integer(
        obj["energy_cost_per_mwh"], "scenario.energy_cost_per_mwh", MAX_SCORE_VALUE
    )
    workers = []
    worker_ids: set = set()
    for index, item in enumerate(_array(obj["workers"], "scenario.workers", MAX_WORKERS, 1)):
        path = f"scenario.workers[{index}]"
        item = _object(item, (
            "id", "available", "available_from_ms", "energy_budget_mwh", "activation_cost"
        ), path)
        worker_id = _identifier(item["id"], f"{path}.id")
        _unique(worker_id, worker_ids, f"{path}.id")
        if type(item["available"]) is not bool:
            raise ValidationError(f"{path}.available must be a boolean")
        workers.append(Worker(
            worker_id,
            item["available"],
            _integer(item["available_from_ms"], f"{path}.available_from_ms", horizon),
            _integer(item["energy_budget_mwh"], f"{path}.energy_budget_mwh", MAX_ENERGY_MWH),
            _integer(item["activation_cost"], f"{path}.activation_cost", MAX_SCORE_VALUE),
        ))
    tasks = []
    task_ids: set = set()
    for index, item in enumerate(_array(obj["tasks"], "scenario.tasks", MAX_TASKS)):
        path = f"scenario.tasks[{index}]"
        item = _object(item, (
            "id", "reward", "release_ms", "deadline_ms", "dependencies", "modes"
        ), path)
        task_id = _identifier(item["id"], f"{path}.id")
        _unique(task_id, task_ids, f"{path}.id")
        release = _integer(item["release_ms"], f"{path}.release_ms", horizon)
        deadline = _integer(item["deadline_ms"], f"{path}.deadline_ms", horizon)
        if release > deadline:
            raise ValidationError(f"{path}.release_ms exceeds deadline_ms")
        dependency_ids: set = set()
        dependencies = []
        for number, dependency in enumerate(_array(
            item["dependencies"], f"{path}.dependencies", MAX_TASKS - 1
        )):
            dependency = _identifier(dependency, f"{path}.dependencies[{number}]")
            _unique(dependency, dependency_ids, f"{path}.dependencies[{number}]")
            dependencies.append(dependency)
        modes = []
        mode_workers: set = set()
        for number, mode in enumerate(_array(item["modes"], f"{path}.modes", MAX_WORKERS, 1)):
            mode_path = f"{path}.modes[{number}]"
            mode = _object(mode, ("worker_id", "duration_ms", "energy_mwh"), mode_path)
            worker_id = _identifier(mode["worker_id"], f"{mode_path}.worker_id")
            if worker_id not in worker_ids:
                raise ValidationError(f"{mode_path}.worker_id refers to unknown worker {worker_id!r}")
            _unique(worker_id, mode_workers, f"{mode_path}.worker_id")
            modes.append(Mode(
                worker_id,
                _integer(mode["duration_ms"], f"{mode_path}.duration_ms", horizon, 1),
                _integer(mode["energy_mwh"], f"{mode_path}.energy_mwh", MAX_ENERGY_MWH),
            ))
        tasks.append(Task(
            task_id,
            _integer(item["reward"], f"{path}.reward", MAX_SCORE_VALUE),
            release, deadline, tuple(dependencies), tuple(modes),
        ))
    task_tuple = tuple(tasks)
    _check_dependencies(task_tuple)
    return Scenario(scenario_id, horizon, energy_cost, tuple(workers), task_tuple)


def parse_plan(data: Any) -> Plan:
    """Parse plan structure; schedule feasibility is independently evaluated."""
    obj = _object(data, ("schema", "scenario_id", "assignments"), "plan")
    if obj["schema"] != PLAN_SCHEMA or type(obj["schema"]) is not str:
        raise ValidationError(f"plan.schema must be {PLAN_SCHEMA!r}")
    scenario_id = _identifier(obj["scenario_id"], "plan.scenario_id")
    assignments = []
    for index, item in enumerate(_array(obj["assignments"], "plan.assignments", MAX_TASKS)):
        path = f"plan.assignments[{index}]"
        item = _object(item, ("task_id", "worker_id", "start_ms", "finish_ms"), path)
        assignments.append(Assignment(
            _identifier(item["task_id"], f"{path}.task_id"),
            _identifier(item["worker_id"], f"{path}.worker_id"),
            _integer(item["start_ms"], f"{path}.start_ms", MAX_TIME_MS),
            _integer(item["finish_ms"], f"{path}.finish_ms", MAX_TIME_MS),
        ))
    return Plan(scenario_id, tuple(assignments))


def scenario_to_dict(scenario: Scenario) -> Dict[str, Any]:
    """Return the declared scenario representation, without inferred defaults."""
    return {
        "schema": SCENARIO_SCHEMA,
        "scenario_id": scenario.scenario_id,
        "horizon_ms": scenario.horizon_ms,
        "energy_cost_per_mwh": scenario.energy_cost_per_mwh,
        "workers": [
            {"id": worker.id, "available": worker.available,
             "available_from_ms": worker.available_from_ms,
             "energy_budget_mwh": worker.energy_budget_mwh,
             "activation_cost": worker.activation_cost}
            for worker in scenario.workers
        ],
        "tasks": [
            {"id": task.id, "reward": task.reward, "release_ms": task.release_ms,
             "deadline_ms": task.deadline_ms, "dependencies": list(task.dependencies),
             "modes": [
                 {"worker_id": mode.worker_id, "duration_ms": mode.duration_ms,
                  "energy_mwh": mode.energy_mwh} for mode in task.modes
             ]}
            for task in scenario.tasks
        ],
    }


def plan_to_dict(plan: Plan) -> Dict[str, Any]:
    return {
        "schema": PLAN_SCHEMA,
        "scenario_id": plan.scenario_id,
        "assignments": [
            {"task_id": assignment.task_id, "worker_id": assignment.worker_id,
             "start_ms": assignment.start_ms, "finish_ms": assignment.finish_ms}
            for assignment in plan.assignments
        ],
    }
