"""Fresh-folder hand-in, error handling and independent checking regressions."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import planner
from inspectionkit.model import Assignment, Plan


ROOT = Path(__file__).resolve().parents[1]


def sample_scenario():
    return {
        "schema": "civilian-inspection/v1", "scenario_id": "factory-smoke",
        "horizon_ms": 100, "energy_cost_per_mwh": 1,
        "workers": [
            {"id": "edge-a", "available": True, "available_from_ms": 0,
             "energy_budget_mwh": 12, "activation_cost": 2},
            {"id": "edge-b", "available": False, "available_from_ms": 0,
             "energy_budget_mwh": 100, "activation_cost": 0},
        ],
        "tasks": [
            {"id": "decode", "reward": 0, "release_ms": 0, "deadline_ms": 80,
             "dependencies": [], "modes": [
                 {"worker_id": "edge-a", "duration_ms": 10, "energy_mwh": 1}]},
            {"id": "inspect-panel", "reward": 30, "release_ms": 0, "deadline_ms": 100,
             "dependencies": ["decode"], "modes": [
                 {"worker_id": "edge-a", "duration_ms": 20, "energy_mwh": 4}]},
            {"id": "inspect-seam", "reward": 12, "release_ms": 0, "deadline_ms": 50,
             "dependencies": [], "modes": [
                 {"worker_id": "edge-a", "duration_ms": 20, "energy_mwh": 3},
                 {"worker_id": "edge-b", "duration_ms": 5, "energy_mwh": 1}]},
        ],
    }


class CLITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.scenario = self.root / "scenario.json"
        self.output = self.root / "plan.json"
        self.scenario.write_text(json.dumps(sample_scenario()), encoding="utf-8")

    def run_cli(self, script, *arguments):
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [sys.executable, "-S", str(script), *map(str, arguments)],
            cwd=str(self.root), env=environment, capture_output=True,
            text=True, timeout=10,
        )

    def test_standalone_folder_plans_and_checks_without_installed_packages(self):
        handin = self.root / "handin"
        handin.mkdir()
        for name in ("planner.py", "check_plan.py"):
            shutil.copyfile(ROOT / name, handin / name)
        shutil.copytree(ROOT / "inspectionkit", handin / "inspectionkit",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        result = self.run_cli(handin / "planner.py", self.scenario, self.output,
                              "--time-limit", "1", "--max-states", "2000")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        report = json.loads(result.stderr)
        self.assertTrue(report["evaluation"]["valid"])
        self.assertFalse(report["execution_performed"])
        self.assertFalse(report["official_leaderboard_submission"])
        self.assertEqual(report["evaluation"]["score"], 32)
        self.assertEqual({a["worker_id"] for a in json.loads(self.output.read_text())["assignments"]},
                         {"edge-a"})
        checked = self.run_cli(handin / "check_plan.py", self.scenario, self.output)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(json.loads(checked.stdout)["score"], 32)

    def test_rejects_original_scenario_and_retains_existing_output(self):
        self.scenario.write_text(json.dumps({"seed": 7, "aircraft": [], "contacts": [], "rules": {}}))
        self.output.write_text("previous-valid-output")
        result = self.run_cli(ROOT / "planner.py", self.scenario, self.output)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.output.read_text(), "previous-valid-output")

    def test_checker_rejects_tampered_duration(self):
        self.output.write_text(json.dumps({
            "schema": "civilian-inspection-plan/v1", "scenario_id": "factory-smoke",
            "assignments": [{"task_id": "decode", "worker_id": "edge-a",
                             "start_ms": 0, "finish_ms": 1}],
        }))
        result = self.run_cli(ROOT / "check_plan.py", self.scenario, self.output)
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["valid"])
        self.assertIsNone(report["score"])

    def test_final_validation_prevents_invalid_candidate_publication(self):
        bad_plan = Plan("factory-smoke", (Assignment("inspect-panel", "edge-a", 0, 20),))
        self.output.write_text("previous-valid-output")
        with mock.patch("planner.solve", return_value=SimpleNamespace(plan=bad_plan, stats={})), \
                contextlib.redirect_stderr(io.StringIO()):
            code = planner.main([str(self.scenario), str(self.output)])
        self.assertEqual(code, 2)
        self.assertEqual(self.output.read_text(), "previous-valid-output")

    def test_invalid_cli_budget_leaves_output_untouched(self):
        for flag, value in (("--time-limit", "nan"), ("--time-limit", "51"),
                            ("--max-states", "0"), ("--max-states", "200001")):
            with self.subTest(flag=flag, value=value):
                self.output.write_text("previous-valid-output")
                result = self.run_cli(ROOT / "planner.py", self.scenario, self.output, flag, value)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(self.output.read_text(), "previous-valid-output")

    def test_invalid_report_statistics_cannot_replace_output(self):
        self.output.write_text("previous-valid-output")
        candidate = SimpleNamespace(plan=Plan("factory-smoke", ()), stats={"elapsed": float("nan")})
        with mock.patch("planner.solve", return_value=candidate), \
                contextlib.redirect_stderr(io.StringIO()):
            code = planner.main([str(self.scenario), str(self.output)])
        self.assertEqual(code, 2)
        self.assertEqual(self.output.read_text(), "previous-valid-output")

    def test_no_network_required_by_planning_path(self):
        with mock.patch("socket.socket", side_effect=AssertionError("network disabled")), \
                contextlib.redirect_stderr(io.StringIO()):
            code = planner.main([str(self.scenario), str(self.output), "--time-limit", "1"])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
