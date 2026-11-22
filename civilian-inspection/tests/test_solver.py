"""Scheduling checks with a discrete-time oracle independent of the solver."""
import itertools
import random
import unittest
from unittest.mock import patch

from inspectionkit.model import Assignment, Mode, Plan, Scenario, Task, Worker
from inspectionkit.solver import greedy_baseline, solve
from inspectionkit.validate import evaluate


def worker(name="a", *, available=True, start=0, energy=100, activation=0):
    return Worker(name, available, start, energy, activation)


def task(name, reward, duration, *, deadline, release=0, energy=0,
         on="a", dependencies=(), modes=None):
    if modes is None:
        modes = (Mode(on, duration, energy),)
    return Task(name, reward, release, deadline, tuple(dependencies), tuple(modes))


def scenario(tasks, *, workers=None, horizon=10, energy_cost=1):
    return Scenario("inspection-test", horizon, energy_cost,
                    tuple(workers or (worker(),)), tuple(tasks))


def discrete_oracle(case):
    """Enumerate job omission, worker and EVERY integer start time.

    This does not use the solver's earliest-append rule, priorities or bound.
    Occupancy and energy are only enumeration reductions; the independent
    validator decides each complete candidate's feasibility and objective.
    Tiny test horizons keep the grid exhaustive without a time cutoff.
    """
    workers = {item.id: item for item in case.workers}
    possibilities = []
    for job in case.tasks:
        alternatives = [None]
        for mode in job.modes:
            node = workers[mode.worker_id]
            if not node.available:
                continue
            for start in range(max(job.release_ms, node.available_from_ms),
                               job.deadline_ms - mode.duration_ms + 1):
                alternatives.append((Assignment(job.id, node.id, start,
                                                start + mode.duration_ms),
                                     mode.energy_mwh))
        possibilities.append(alternatives)
    best = 0
    occupied = {item: set() for item in workers}
    energy = {item: 0 for item in workers}

    def visit(index, assignments):
        nonlocal best
        if index == len(case.tasks):
            result = evaluate(case, Plan(case.scenario_id, tuple(assignments)))
            if result.valid:
                best = max(best, result.score)
            return
        for alternative in possibilities[index]:
            if alternative is None:
                visit(index + 1, assignments)
                continue
            assignment, cost = alternative
            node = assignment.worker_id
            slots = set(range(assignment.start_ms, assignment.finish_ms))
            if slots & occupied[node] or energy[node] + cost > workers[node].energy_budget_mwh:
                continue
            occupied[node].update(slots)
            energy[node] += cost
            visit(index + 1, assignments + [assignment])
            occupied[node].difference_update(slots)
            energy[node] -= cost

    visit(0, [])
    return best


class SolverTests(unittest.TestCase):
    def assert_plan(self, case, result):
        score = evaluate(case, result.plan)
        baseline = evaluate(case, result.baseline)
        self.assertTrue(score.valid, score.violations)
        self.assertTrue(baseline.valid, baseline.violations)
        self.assertGreaterEqual(score.score, baseline.score)
        self.assertEqual(result.stats["score"], score.score)
        self.assertEqual(result.stats["baseline_score"], baseline.score)
        self.assertEqual(result.baseline, greedy_baseline(case))
        self.assertLessEqual(result.stats["states"], result.stats["max_states"])

    def test_discrete_grid_optima(self):
        cases = [
            scenario([task("a", 10, 3, deadline=5),
                      task("b", 7, 2, deadline=5),
                      task("c", 7, 3, deadline=5)], horizon=5),
            scenario([task("short", 1, 1, deadline=3),
                      task("released", 100, 4, release=3, deadline=7)], horizon=7),
            scenario([task("decode", 0, 1, deadline=4, energy=2),
                      task("inspect", 20, 1, deadline=4, dependencies=("decode",))], horizon=4),
            scenario([task("one", 6, 1, deadline=2),
                      task("two", 6, 1, deadline=2)],
                     workers=[worker(activation=10)], horizon=2),
            scenario([task("prep", 2, 2, deadline=5, on="a"),
                      task("inspect", 10, 1, deadline=5, on="b", dependencies=("prep",)),
                      task("independent", 6, 2, deadline=4, on="b")],
                     workers=[worker("a"), worker("b")], horizon=5),
            scenario([task("a", 12, 1, deadline=5,
                           modes=(Mode("a", 3, 1), Mode("b", 1, 6))),
                      task("b", 10, 2, deadline=5, energy=2)],
                     workers=[worker("a", energy=3), worker("b", energy=8, activation=2)], horizon=5),
            scenario([task("a", 100, 1, deadline=3, on="a"),
                      task("b", 4, 2, deadline=4, on="b")],
                     workers=[worker("a", available=False), worker("b", start=2)], horizon=4),
            scenario([task("a", 5, 2, deadline=4, energy=3),
                      task("b", 10, 3, deadline=4, energy=4)],
                     workers=[worker(energy=4)], horizon=4),
        ]
        for index, case in enumerate(cases):
            with self.subTest(case=index):
                expected = discrete_oracle(case)
                result = solve(case, time_limit_s=2.0, max_states=200_000)
                self.assert_plan(case, result)
                self.assertEqual(result.stats["score"], expected)
                self.assertTrue(result.stats["optimal_proven"])
                self.assertFalse(result.stats["timed_out"])
                self.assertFalse(result.stats["state_limit_reached"])

    def test_improves_a_density_trap(self):
        case = scenario([task("a", 10, 6, deadline=10),
                         task("b", 7, 5, deadline=10),
                         task("c", 7, 5, deadline=10)])
        result = solve(case)
        self.assert_plan(case, result)
        self.assertEqual(result.stats["baseline_score"], 10)
        self.assertEqual(result.stats["score"], 14)
        self.assertTrue(result.stats["optimal_proven"])

    def test_seeded_tiny_cases_match_discrete_oracle(self):
        rng = random.Random(14791)
        for index in range(80):
            horizon = rng.randrange(3, 8)
            nodes = [worker("a", energy=rng.randrange(1, 10), activation=rng.randrange(0, 5)),
                     worker("b", available=rng.random() > 0.1, start=rng.randrange(0, 2),
                            energy=rng.randrange(1, 10), activation=rng.randrange(0, 5))]
            jobs = []
            for job_index in range(4):
                release = rng.randrange(0, horizon)
                deadline = rng.randrange(release, horizon + 1)
                modes = tuple(Mode(node.id, rng.randrange(1, min(horizon, 3) + 1),
                                   rng.randrange(0, 5))
                              for node in nodes if rng.random() < 0.7)
                if not modes:
                    modes = (Mode("a", 1, 0),)
                dependencies = tuple("job-%d" % previous for previous in range(job_index)
                                     if rng.random() < 0.25)
                jobs.append(task("job-%d" % job_index, rng.randrange(0, 21), 1,
                                 release=release, deadline=deadline,
                                 dependencies=dependencies, modes=modes))
            case = scenario(jobs, workers=nodes, horizon=horizon, energy_cost=rng.randrange(0, 4))
            with self.subTest(case=index):
                expected = discrete_oracle(case)
                result = solve(case, time_limit_s=2.0, max_states=200_000)
                self.assertEqual(result.stats["score"], expected)
                self.assertTrue(result.stats["optimal_proven"])

    def test_zero_reward_chain_unlocks_inspection(self):
        case = scenario([task("decode", 0, 1, deadline=10, energy=3),
                         task("normalize", 0, 1, deadline=10, energy=2,
                              dependencies=("decode",)),
                         task("inspect", 50, 1, deadline=10,
                              dependencies=("normalize",))])
        result = solve(case)
        self.assert_plan(case, result)
        self.assertEqual(result.stats["score"], 45)
        self.assertEqual({item.task_id for item in result.plan.assignments},
                         {"decode", "normalize", "inspect"})

    def test_empty_and_unprofitable_inputs(self):
        for case in (
            scenario([]),
            scenario([task("expensive", 1_000_000_000, 1, deadline=10,
                           energy=1_000_000_000_000)],
                     workers=[worker(energy=1_000_000_000_000)], energy_cost=1_000_000_000),
            scenario([task("unavailable", 100, 1, deadline=10)],
                     workers=[worker(available=False)]),
        ):
            result = solve(case)
            self.assert_plan(case, result)
            self.assertEqual(result.plan.assignments, ())
            self.assertEqual(result.stats["score"], 0)
            self.assertTrue(result.stats["optimal_proven"])

    def test_state_cutoff_retains_valid_incumbent(self):
        case = scenario([task("a", 10, 6, deadline=10),
                         task("b", 7, 5, deadline=10),
                         task("c", 7, 5, deadline=10)])
        result = solve(case, max_states=1)
        self.assert_plan(case, result)
        self.assertEqual(result.stats["states"], 1)
        self.assertTrue(result.stats["state_limit_reached"])
        self.assertFalse(result.stats["optimal_proven"])

    def test_deadline_cutoff_keeps_baseline(self):
        case = scenario([task("a", 10, 6, deadline=10),
                         task("b", 7, 5, deadline=10),
                         task("c", 7, 5, deadline=10)])
        clock = itertools.count(0.0, 0.02)
        with patch("inspectionkit.solver.time.monotonic", side_effect=lambda: next(clock)):
            result = solve(case, time_limit_s=0.01)
        self.assert_plan(case, result)
        self.assertEqual(result.plan, result.baseline)
        self.assertTrue(result.stats["timed_out"])
        self.assertFalse(result.stats["optimal_proven"])
        self.assertEqual(result.stats["states"], 0)

    def test_large_case_never_claims_optimality(self):
        case = scenario([task("inspect-%02d" % index, 10 + index, 1,
                              deadline=8, energy=1) for index in range(16)], horizon=8)
        result = solve(case, time_limit_s=1.0, max_states=250)
        self.assert_plan(case, result)
        self.assertEqual(result.stats["algorithm"], "portfolio+bounded-beam")
        self.assertFalse(result.stats["optimal_proven"])

    def test_input_order_does_not_change_result(self):
        case = scenario([
            task("decode", 0, 1, deadline=8, modes=(Mode("a", 1, 1), Mode("b", 2, 0))),
            task("inspect", 20, 2, deadline=8, dependencies=("decode",),
                 modes=(Mode("a", 2, 4), Mode("b", 3, 1))),
            task("independent", 10, 3, deadline=6, modes=(Mode("a", 3, 2), Mode("b", 2, 3))),
        ], workers=[worker("a", activation=2), worker("b", activation=1)], horizon=8)
        reversed_case = Scenario(
            case.scenario_id, case.horizon_ms, case.energy_cost_per_mwh,
            tuple(reversed(case.workers)),
            tuple(Task(job.id, job.reward, job.release_ms, job.deadline_ms,
                       tuple(reversed(job.dependencies)), tuple(reversed(job.modes)))
                  for job in reversed(case.tasks)),
        )
        first, second = solve(case, max_states=5000), solve(reversed_case, max_states=5000)
        self.assertEqual(first.plan, second.plan)
        self.assertEqual(first.baseline, second.baseline)
        self.assertEqual(first.stats["states"], second.stats["states"])

    def test_invalid_budget_options(self):
        case = scenario([])
        for value in (True, False, 0, -1, 0.001, 50.01, float("inf"), float("nan"), "1", None):
            with self.subTest(time_limit=value), self.assertRaises(ValueError):
                solve(case, time_limit_s=value)
        for value in (True, False, 0, -1, 1.5, 200_001, "1", None):
            with self.subTest(max_states=value), self.assertRaises(ValueError):
                solve(case, max_states=value)


if __name__ == "__main__":
    unittest.main()
