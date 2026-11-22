"""Bounded scheduling of non-actuating inspection jobs on local compute workers.

The dispatcher first establishes a reproducible greedy incumbent, tries several
constructive priorities, then searches job/worker choices.  Small instances use
complete branch-and-bound when the supplied limits permit it.  Larger instances
use a bounded beam and never claim optimality.  All objective arithmetic is
integer arithmetic; rational numbers are used only to order search candidates.

Jobs are appended at their earliest feasible time.  For a fixed order of jobs
on each worker, moving a job earlier cannot hurt a release/deadline, energy or
precedence constraint.  Enumerating all ready jobs and eligible workers therefore
covers an optimum for this schema.  It does not cover other scheduling models
such as preemption, data transfers, periodic reservations or shared accelerators.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from .model import (
    Assignment,
    Plan,
    Scenario,
    parse_scenario,
    scenario_to_dict,
)
from .validate import evaluate


EXACT_TASK_LIMIT = 9
BEAM_WIDTH = 32
BRANCH_WIDTH = 12
MAX_STATES = 200_000
_SEEN_LIMIT = 20_000


@dataclass(frozen=True)
class PlanningResult:
    plan: Plan
    baseline: Plan
    stats: dict


class _Option(NamedTuple):
    worker: int
    duration: int
    energy: int
    gain: int


class _Job(NamedTuple):
    task_id: str
    reward: int
    release: int
    deadline: int
    dependencies: int
    dependency_indices: Tuple[int, ...]
    options: Tuple[_Option, ...]


class _Problem(NamedTuple):
    scenario: Scenario
    jobs: Tuple[_Job, ...]
    worker_ids: Tuple[str, ...]
    worker_available_from: Tuple[int, ...]
    energy_budgets: Tuple[int, ...]
    activation_costs: Tuple[int, ...]
    unlock_values: Tuple[int, ...]


class _Choice(NamedTuple):
    job: int
    option: _Option
    start: int
    finish: int
    gain: int


class _State(NamedTuple):
    mask: int
    ends: Tuple[int, ...]
    energy: Tuple[int, ...]
    finishes: Tuple[int, ...]
    active: int
    score: int
    assignments: Tuple[Assignment, ...]


@dataclass
class _Budget:
    deadline: float
    max_states: int
    states: int = 0
    timed_out: bool = False
    state_limit_reached: bool = False

    def check_time(self) -> bool:
        if time.monotonic() >= self.deadline:
            self.timed_out = True
            return False
        return True

    def available(self) -> bool:
        if not self.check_time():
            return False
        if self.states >= self.max_states:
            self.state_limit_reached = True
            return False
        return True

    def take(self) -> bool:
        if not self.available():
            return False
        self.states += 1
        return True


def _compile(scenario: Scenario) -> _Problem:
    # Dataclasses are public API too: callers cannot bypass validation by
    # constructing one directly rather than going through the JSON parser.
    scenario = parse_scenario(scenario_to_dict(scenario))
    workers = tuple(sorted(scenario.workers, key=lambda item: item.id))
    tasks = tuple(sorted(scenario.tasks, key=lambda item: item.id))
    wi = {worker.id: index for index, worker in enumerate(workers)}
    ti = {task.id: index for index, task in enumerate(tasks)}
    jobs = []
    for task in tasks:
        options = []
        for mode in sorted(task.modes, key=lambda item: item.worker_id):
            worker = workers[wi[mode.worker_id]]
            earliest = max(task.release_ms, worker.available_from_ms)
            if (
                worker.available
                and earliest + mode.duration_ms <= task.deadline_ms
                and mode.energy_mwh <= worker.energy_budget_mwh
            ):
                options.append(_Option(
                    wi[mode.worker_id], mode.duration_ms, mode.energy_mwh,
                    task.reward - scenario.energy_cost_per_mwh * mode.energy_mwh,
                ))
        dependencies = tuple(sorted(ti[item] for item in task.dependencies))
        jobs.append(_Job(
            task.id, task.reward, task.release_ms, task.deadline_ms,
            sum(1 << index for index in dependencies), dependencies,
            tuple(options),
        ))

    # A heuristic value for unlocking downstream inspection stages.  It is
    # deliberately separate from the admissible bound used to prune search.
    successors: List[List[int]] = [[] for _ in jobs]
    for index, job in enumerate(jobs):
        for dependency in job.dependency_indices:
            successors[dependency].append(index)
    descendant_masks: Dict[int, int] = {}

    def descendants(index: int) -> int:
        if index not in descendant_masks:
            mask = 1 << index
            for successor in successors[index]:
                mask |= descendants(successor)
            descendant_masks[index] = mask
        return descendant_masks[index]

    positive_gains = tuple(max([0] + [mode.gain for mode in job.options]) for job in jobs)
    unlock_values = tuple(
        sum(positive_gains[j] for j in range(len(jobs)) if descendants(i) & (1 << j))
        for i in range(len(jobs))
    )
    return _Problem(
        scenario, tuple(jobs), tuple(worker.id for worker in workers),
        tuple(worker.available_from_ms for worker in workers),
        tuple(worker.energy_budget_mwh for worker in workers),
        tuple(worker.activation_cost for worker in workers), unlock_values,
    )


def _initial(problem: _Problem) -> _State:
    return _State(
        0, problem.worker_available_from, (0,) * len(problem.worker_ids),
        (-1,) * len(problem.jobs), 0, 0, (),
    )


def _choices(
    problem: _Problem, state: _State, budget: Optional[_Budget] = None,
) -> Iterable[_Choice]:
    for index, job in enumerate(problem.jobs):
        if budget is not None and not budget.check_time():
            return
        if state.mask & (1 << index) or state.mask & job.dependencies != job.dependencies:
            continue
        ready = max([job.release] + [state.finishes[d] for d in job.dependency_indices])
        for option in job.options:
            worker = option.worker
            start = max(ready, state.ends[worker])
            finish = start + option.duration
            if finish > job.deadline or state.energy[worker] + option.energy > problem.energy_budgets[worker]:
                continue
            activation = 0 if state.active & (1 << worker) else problem.activation_costs[worker]
            yield _Choice(index, option, start, finish, option.gain - activation)


def _advance(problem: _Problem, state: _State, choice: _Choice) -> _State:
    worker = choice.option.worker
    ends, energy, finishes = list(state.ends), list(state.energy), list(state.finishes)
    ends[worker] = choice.finish
    energy[worker] += choice.option.energy
    finishes[choice.job] = choice.finish
    assignment = Assignment(
        problem.jobs[choice.job].task_id, problem.worker_ids[worker],
        choice.start, choice.finish,
    )
    return _State(
        state.mask | (1 << choice.job), tuple(ends), tuple(energy), tuple(finishes),
        state.active | (1 << worker), state.score + choice.gain,
        state.assignments + (assignment,),
    )


def _upper_bound(problem: _Problem, state: _State) -> int:
    """Optimistic reward remaining in this append-only search branch.

    Every remaining task gets its best individually feasible net energy reward.
    Contention, the cost of unscheduled dependencies and future activation costs
    are ignored.  Those omissions can only raise the bound.  In particular, a
    costly or zero-reward prerequisite is never treated as an impossible branch.
    """
    upper = state.score
    for index, job in enumerate(problem.jobs):
        if state.mask & (1 << index):
            continue
        ready = max([job.release] + [state.finishes[d] for d in job.dependency_indices])
        gain = 0
        for option in job.options:
            worker = option.worker
            if (
                max(ready, state.ends[worker]) + option.duration <= job.deadline
                and state.energy[worker] + option.energy <= problem.energy_budgets[worker]
            ):
                gain = max(gain, option.gain)
        upper += gain
    return upper


def _priority(problem: _Problem, choice: _Choice, style: str) -> tuple:
    job, option = problem.jobs[choice.job], choice.option
    rate = Fraction(choice.gain, option.duration)
    if style == "unlock":
        primary = (Fraction(problem.unlock_values[choice.job], option.duration), rate)
    elif style == "deadline":
        primary = (-job.deadline, rate)
    elif style == "reward":
        primary = (job.reward, rate)
    elif style == "energy":
        primary = (-option.energy, choice.gain)
    elif style == "shortest":
        primary = (-option.duration, choice.gain)
    else:
        primary = (rate, choice.gain)
    # Options are generated in canonical ID order.  Keeping the first exact tie
    # makes the output independent of input list order without using float keys.
    return primary + (-choice.finish, -option.energy)


def _construct(
    problem: _Problem, style: str, budget: Optional[_Budget] = None,
) -> Tuple[_State, int]:
    state = best = _initial(problem)
    count = 0
    while budget is None or budget.available():
        choices = list(_choices(problem, state, budget))
        if not choices or (budget is not None and not budget.take()):
            break
        state = _advance(problem, state, max(choices, key=lambda choice: _priority(problem, choice, style)))
        count += 1
        if state.score > best.score:
            best = state
    return best, count


def _as_plan(problem: _Problem, state: _State) -> Plan:
    assignments = tuple(sorted(
        state.assignments, key=lambda item: (item.start_ms, item.worker_id, item.task_id),
    ))
    return Plan(problem.scenario.scenario_id, assignments)


def greedy_baseline(scenario: Scenario) -> Plan:
    """Deterministic ready-job reward-rate dispatch, retaining its best prefix.

    Non-positive immediate work may unlock later stages or share a worker's
    activation charge.  Consequently the baseline explores such work rather
    than categorically discarding prerequisites.  An empty plan is always a
    candidate.  This is a reference heuristic, not an optimal scheduler.
    """
    problem = _compile(scenario)
    best, _ = _construct(problem, "rate")
    plan = _as_plan(problem, best)
    result = evaluate(problem.scenario, plan)
    if not result.valid or result.score != best.score:
        raise RuntimeError("independent validation rejected the greedy schedule")
    return plan


def _state_key(state: _State) -> tuple:
    # Future readiness depends on individual predecessor finish times, not just
    # worker end times.  Omitting them would make duplicate pruning unsound.
    return state.mask, state.ends, state.energy, state.finishes, state.active


def _exact_search(
    problem: _Problem, incumbent: _State, budget: _Budget,
) -> Tuple[_State, bool, int]:
    initial = _initial(problem)
    frontier = [initial]
    seen = {_state_key(initial)}
    pruned = 0
    while frontier:
        if not budget.available():
            return incumbent, False, pruned
        state = frontier.pop()
        if _upper_bound(problem, state) <= incumbent.score:
            pruned += 1
            continue
        choices = sorted(
            _choices(problem, state, budget),
            key=lambda choice: _priority(problem, choice, "unlock"),
        )
        if budget.timed_out:
            return incumbent, False, pruned
        # A LIFO frontier explores the highest-priority choice first.
        for choice in choices:
            if not budget.take():
                return incumbent, False, pruned
            child = _advance(problem, state, choice)
            if child.score > incumbent.score:
                incumbent = child
            if _upper_bound(problem, child) <= incumbent.score:
                pruned += 1
                continue
            key = _state_key(child)
            if key in seen:
                continue
            # Capping this cache loses only deduplication, not any search branch.
            if len(seen) < _SEEN_LIMIT:
                seen.add(key)
            frontier.append(child)
    return incumbent, True, pruned


def _shortlist(problem: _Problem, choices: Sequence[_Choice]) -> List[_Choice]:
    """Retain alternatives across rate, prerequisite value and deadline order."""
    orders = [
        sorted(choices, key=lambda item: _priority(problem, item, style), reverse=True)
        for style in ("rate", "unlock", "deadline")
    ]
    selected: List[_Choice] = []
    keys = set()
    for rank in range(len(choices)):
        for order in orders:
            choice = order[rank]
            key = choice.job, choice.option.worker
            if key not in keys:
                keys.add(key)
                selected.append(choice)
                if len(selected) == BRANCH_WIDTH:
                    return selected
    return selected


def _beam_search(
    problem: _Problem, incumbent: _State, budget: _Budget,
) -> Tuple[_State, int, int, int]:
    beam = [_initial(problem)]
    bound_pruned = branch_pruned = beam_pruned = 0
    while beam:
        next_states = {}
        for state in beam:
            if not budget.available():
                return incumbent, bound_pruned, branch_pruned, beam_pruned
            if _upper_bound(problem, state) <= incumbent.score:
                bound_pruned += 1
                continue
            choices = list(_choices(problem, state, budget))
            if budget.timed_out:
                return incumbent, bound_pruned, branch_pruned, beam_pruned
            selected = _shortlist(problem, choices)
            branch_pruned += len(choices) - len(selected)
            for choice in selected:
                if not budget.take():
                    return incumbent, bound_pruned, branch_pruned, beam_pruned
                child = _advance(problem, state, choice)
                if child.score > incumbent.score:
                    incumbent = child
                upper = _upper_bound(problem, child)
                if upper <= incumbent.score:
                    bound_pruned += 1
                    continue
                next_states.setdefault(_state_key(child), (upper, child))
        ranked = sorted(next_states.values(), key=lambda item: (item[0], item[1].score), reverse=True)
        beam_pruned += max(0, len(ranked) - BEAM_WIDTH)
        beam = [state for _, state in ranked[:BEAM_WIDTH]]
    return incumbent, bound_pruned, branch_pruned, beam_pruned


def solve(
    scenario: Scenario, *, time_limit_s: float = 5.0, max_states: int = 50_000,
) -> PlanningResult:
    """Return a validated schedule scoring at least as well as the baseline.

    ``time_limit_s`` sets a monotonic search deadline measured from function
    entry.  The bounded baseline and final independent validation always run;
    they can add overhead beyond very small deadlines.  This is not a hard
    real-time guarantee.  ``max_states`` caps non-baseline constructive/search
    transitions; the baseline performs at most one transition per input job.
    """
    if (
        isinstance(time_limit_s, bool)
        or not isinstance(time_limit_s, (int, float))
        or not math.isfinite(time_limit_s)
        or not 0.01 <= time_limit_s <= 50.0
    ):
        raise ValueError("time_limit_s must be finite and between 0.01 and 50 seconds")
    if isinstance(max_states, bool) or not isinstance(max_states, int) or not 1 <= max_states <= MAX_STATES:
        raise ValueError("max_states must be an integer between 1 and 200000")
    started = time.monotonic()
    budget = _Budget(started + float(time_limit_s), max_states)
    problem = _compile(scenario)
    best, baseline_states = _construct(problem, "rate")
    baseline = _as_plan(problem, best)
    baseline_evaluation = evaluate(problem.scenario, baseline)
    if not baseline_evaluation.valid or baseline_evaluation.score != best.score:
        raise RuntimeError("independent validation rejected the greedy schedule")
    baseline_elapsed = time.monotonic() - started

    priorities_completed = 0
    for style in ("unlock", "deadline", "reward", "energy", "shortest"):
        if not budget.available():
            break
        candidate, _ = _construct(problem, style, budget)
        if candidate.score > best.score:
            best = candidate
        if not budget.timed_out and not budget.state_limit_reached:
            priorities_completed += 1
    constructive_states = budget.states

    exact = len(problem.jobs) <= EXACT_TASK_LIMIT
    optimal_proven = False
    bound_pruned = branch_pruned = beam_pruned = 0
    if budget.available():
        if exact:
            best, optimal_proven, bound_pruned = _exact_search(problem, best, budget)
        else:
            best, bound_pruned, branch_pruned, beam_pruned = _beam_search(problem, best, budget)
    plan = _as_plan(problem, best)
    result = evaluate(problem.scenario, plan)
    if not result.valid or result.score != best.score:
        raise RuntimeError("independent validation rejected the selected schedule")
    if result.score < baseline_evaluation.score:
        raise RuntimeError("selected schedule scores below its greedy incumbent")
    elapsed = time.monotonic() - started
    stats = {
        "algorithm": "portfolio+branch-and-bound" if exact else "portfolio+bounded-beam",
        "optimal_proven": optimal_proven,
        "time_limit_s": float(time_limit_s),
        "elapsed_s": elapsed,
        "baseline_elapsed_s": baseline_elapsed,
        "max_states": max_states,
        "states": budget.states,
        "baseline_states": baseline_states,
        "constructive_states": constructive_states,
        "search_states": budget.states - constructive_states,
        "timed_out": budget.timed_out,
        "state_limit_reached": budget.state_limit_reached,
        "constructive_priorities_completed": priorities_completed,
        "bound_pruned": bound_pruned,
        "branch_pruned": branch_pruned,
        "beam_pruned": beam_pruned,
        "baseline_score": baseline_evaluation.score,
        "score": result.score,
        "completed_tasks": len(plan.assignments),
        "total_tasks": len(problem.jobs),
    }
    return PlanningResult(plan, baseline, stats)
