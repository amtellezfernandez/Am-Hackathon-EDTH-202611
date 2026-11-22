"""Evidence checks are independent of the planner's claimed score and stats."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark
from inspectionkit.generate import DEVELOPMENT_SEEDS, HELDOUT_SEEDS, cases, generate, protocol
from inspectionkit.model import Assignment, Plan, parse_scenario, scenario_to_dict


ROOT = Path(__file__).resolve().parents[1]


def simple_scenario():
    return parse_scenario({
        "schema": "civilian-inspection/v1", "scenario_id": "inspection-development-small-11",
        "horizon_ms": 100, "energy_cost_per_mwh": 1,
        "workers": [{"id": "local", "available": True, "available_from_ms": 0,
                     "energy_budget_mwh": 10, "activation_cost": 1}],
        "tasks": [{"id": "image", "reward": 20, "release_ms": 0, "deadline_ms": 100,
                   "dependencies": [],
                   "modes": [{"worker_id": "local", "duration_ms": 10, "energy_mwh": 2}]}],
    })


class GeneratorTests(unittest.TestCase):
    def test_fixed_sets_are_disjoint_and_each_has_all_tiers(self):
        self.assertFalse(set(DEVELOPMENT_SEEDS) & set(HELDOUT_SEEDS))
        self.assertEqual(len(cases("development")), 12)
        self.assertEqual(len(cases("heldout")), 12)
        self.assertEqual(len({case.scenario_id for case in cases("all")}), 24)
        for split in ("development", "heldout"):
            self.assertEqual({case.tier for case in cases(split)}, {"small", "medium", "large"})

    def test_frozen_source_protocol_and_every_input_identity(self):
        fixed = json.loads((ROOT / "examples/benchmark-inputs.json").read_text())
        protocol_file = ROOT / "examples/benchmark-protocol.json"
        self.assertEqual(json.loads(protocol_file.read_text()), protocol())
        self.assertEqual(hashlib.sha256(protocol_file.read_bytes()).hexdigest(), fixed["protocol_sha256"])
        self.assertEqual(hashlib.sha256((ROOT / "inspectionkit/generate.py").read_bytes()).hexdigest(),
                         fixed["generator_sha256"])
        by_id = {row["scenario_id"]: row["sha256"] for row in fixed["cases"]}
        self.assertEqual(len(by_id), 24)
        for case in cases("all"):
            with self.subTest(case=case.scenario_id):
                one = generate(case.tier, case.seed, split=case.split)
                two = generate(case.tier, case.seed, split=case.split)
                self.assertEqual(one, two)
                raw = benchmark._json_bytes(scenario_to_dict(one))
                self.assertEqual(hashlib.sha256(raw).hexdigest(), by_id[case.scenario_id])

    def test_local_job_structure_and_nontrivial_constraints(self):
        for case in cases("all"):
            with self.subTest(case=case.scenario_id):
                scenario = generate(case.tier, case.seed, split=case.split)
                data = scenario_to_dict(scenario)
                self.assertEqual(data["schema"], "civilian-inspection/v1")
                self.assertGreaterEqual(sum(worker.available for worker in scenario.workers), 1)
                self.assertTrue(any(task.dependencies for task in scenario.tasks))
                self.assertTrue(any(task.reward == 0 for task in scenario.tasks))
                self.assertTrue(any(task.release_ms > 0 for task in scenario.tasks))
                self.assertTrue(any(task.deadline_ms < scenario.horizon_ms for task in scenario.tasks))
                self.assertNotIn("routes", data)
                self.assertNotIn("contacts", data)
                self.assertNotIn("aircraft", data)

    def test_generator_rejects_invalid_arguments(self):
        for tier, seed, split in (("unknown", 1, "development"), ("small", True, "development"),
                                  ("small", -1, "heldout"), ("small", 2**31, "heldout"),
                                  ("small", 1, "all")):
            with self.subTest(tier=tier, seed=seed, split=split), self.assertRaises(ValueError):
                generate(tier, seed, split=split)

    def test_human_readable_example_is_valid_input(self):
        scenario = parse_scenario(json.loads((ROOT / "examples/factory-small.json").read_text()))
        self.assertEqual(len(scenario.workers), 2)
        self.assertEqual(len(scenario.tasks), 5)


class EvidenceTests(unittest.TestCase):
    def test_report_recomputes_score_instead_of_trusting_solver_stats(self):
        scenario = simple_scenario()
        baseline = Plan(scenario.scenario_id, ())
        improved = Plan(scenario.scenario_id, (Assignment("image", "local", 0, 10),))
        solution = SimpleNamespace(plan=improved, baseline=baseline, stats={"score": 999999})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(benchmark, "greedy_baseline", return_value=baseline), \
                 patch.object(benchmark, "solve", return_value=solution):
                result = benchmark._run_case(cases()[0], scenario, Path(directory), 0.01, 1)
            self.assertEqual(result["planner"]["evaluation"]["score"], 17)
            self.assertEqual(result["delta_score"], 17)
            self.assertEqual(result["baseline"]["evaluation"]["score"], 0)

    def test_invalid_plan_fails_and_retains_failure_evidence(self):
        scenario = simple_scenario()
        baseline = Plan(scenario.scenario_id, ())
        invalid = Plan(scenario.scenario_id, (Assignment("image", "local", 0, 11),))
        solution = SimpleNamespace(plan=invalid, baseline=baseline, stats={"score": 99})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch.object(benchmark, "greedy_baseline", return_value=baseline), \
                 patch.object(benchmark, "solve", return_value=solution):
                with self.assertRaises(benchmark.BenchmarkError):
                    benchmark._run_case(cases()[0], scenario, path, 0.01, 1)
            result = json.loads((path / "result.json").read_text())
            self.assertFalse(result["planner"]["evaluation"]["valid"])
            self.assertIsNone(result["planner"]["evaluation"]["score"])

    def test_lower_scoring_valid_plan_is_not_reported_as_pass(self):
        scenario = simple_scenario()
        baseline = Plan(scenario.scenario_id, (Assignment("image", "local", 0, 10),))
        solution = SimpleNamespace(plan=Plan(scenario.scenario_id, ()), baseline=baseline, stats={})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(benchmark, "greedy_baseline", return_value=baseline), \
                 patch.object(benchmark, "solve", return_value=solution):
                with self.assertRaisesRegex(benchmark.BenchmarkError, "below baseline"):
                    benchmark._run_case(cases()[0], scenario, Path(directory), 0.01, 1)

    def test_evidence_inventory_covers_every_output_and_cannot_overwrite(self):
        case = cases()[0]
        empty = Plan(case.scenario_id, ())
        solution = SimpleNamespace(plan=empty, baseline=empty, stats={})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence"
            with patch.object(benchmark, "cases", return_value=(case,)), \
                 patch.object(benchmark, "greedy_baseline", return_value=empty), \
                 patch.object(benchmark, "solve", return_value=solution):
                report = benchmark.run_benchmark(output, time_limit_s=0.01, max_states=1)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["summary"]["cases"], 1)
            self.assertFalse(report["leaderboard_compatible"])
            self.assertFalse(report["job_execution_tested"])
            observed = set()
            for line in (output / "SHA256SUMS").read_text().splitlines():
                digest, name = line.split("  ", 1)
                observed.add(name)
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            expected = {path.relative_to(output).as_posix() for path in output.rglob("*")
                        if path.is_file() and path.name != "SHA256SUMS"}
            self.assertEqual(observed, expected)
            with self.assertRaises(FileExistsError):
                benchmark.run_benchmark(output)

    def test_generator_change_requires_explicit_new_freeze(self):
        changed = benchmark._source_inventory(ROOT)
        changed["inspectionkit/generate.py"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(benchmark, "_source_inventory", return_value=changed):
                with self.assertRaisesRegex(benchmark.BenchmarkError, "generator source differs"):
                    benchmark.run_benchmark(Path(directory) / "evidence", time_limit_s=0.01)

    def test_source_change_during_measurement_prevents_success_report(self):
        case = cases()[0]
        empty = Plan(case.scenario_id, ())
        solution = SimpleNamespace(plan=empty, baseline=empty, stats={})
        before = benchmark._source_inventory(ROOT)
        after = dict(before)
        after["benchmark.py"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence"
            with patch.object(benchmark, "cases", return_value=(case,)), \
                 patch.object(benchmark, "greedy_baseline", return_value=empty), \
                 patch.object(benchmark, "solve", return_value=solution), \
                 patch.object(benchmark, "_source_inventory", side_effect=(before, after)):
                with self.assertRaisesRegex(benchmark.BenchmarkError, "source changed"):
                    benchmark.run_benchmark(output, time_limit_s=0.01, max_states=1)
            self.assertFalse((output / "report.json").exists())

    def test_invalid_limits_fail_before_creating_output(self):
        for seconds, states in ((float("nan"), 1), (float("inf"), 1), (True, 1),
                                (0, 1), (51, 1), (0.01, 0), (0.01, True),
                                (0.01, 200001)):
            with self.subTest(seconds=seconds, states=states), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "evidence"
                with self.assertRaises(ValueError):
                    benchmark.run_benchmark(output, time_limit_s=seconds, max_states=states)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
