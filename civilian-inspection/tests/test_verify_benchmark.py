"""Independent replay rejects rehashed claims as well as broken inventories."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark
import verify_benchmark
from inspectionkit.model import Plan


def write_json(path, value):
    path.write_bytes(benchmark._json_bytes(value))


def load_json(path):
    return json.loads(path.read_text())


class ReplayVerificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "evidence"

        def empty(scenario):
            return Plan(scenario.scenario_id, ())

        def solve_fixture(scenario, **kwargs):
            return SimpleNamespace(plan=empty(scenario), baseline=empty(scenario),
                                   stats={"fixture": "no optimizer executed"})

        # Keep the real complete frozen development manifest and scenarios, but
        # replace both optimizers with valid empty plans. No holdout/search run.
        with patch.object(benchmark, "greedy_baseline", side_effect=empty), \
                patch.object(benchmark, "solve", side_effect=solve_fixture):
            benchmark.run_benchmark(self.root, split="development", time_limit_s=0.01, max_states=1)
        self.case_id = benchmark.cases("development")[0].scenario_id
        self.case_dir = self.root / "cases" / self.case_id

    def rehash(self):
        lines = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(self.root).as_posix()}\n")
        (self.root / "SHA256SUMS").write_text("".join(lines))

    def reject(self, message):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = verify_benchmark.main([str(self.root)])
        self.assertEqual(code, 2, output.getvalue())
        result = json.loads(output.getvalue())
        self.assertFalse(result["valid"])
        self.assertIn(message, result["error"])

    def update_row_in_report(self, row):
        report = load_json(self.root / "report.json")
        report["cases"][0] = row
        report["summary"] = benchmark._summarize(report["cases"])
        report["by_split"] = {"development": benchmark._summarize(report["cases"])}
        report["by_tier"] = {tier: benchmark._summarize(
            item for item in report["cases"] if item["tier"] == tier)
            for tier in ("small", "medium", "large")}
        write_json(self.root / "report.json", report)

    def test_normal_complete_fixture_replays_without_optimizer(self):
        with patch.object(benchmark, "solve", side_effect=AssertionError("optimizer must not run")), \
                patch.object(benchmark, "greedy_baseline", side_effect=AssertionError("baseline must not run")):
            result = verify_benchmark.verify(self.root)
        self.assertTrue(result["valid"])
        self.assertEqual(result["cases_verified"], 12)
        self.assertEqual(result["plans_verified"], 24)
        self.assertEqual(result["files_verified"], 52)
        self.assertEqual(result["summary"]["ties"], 12)
        self.assertEqual(result["summary"]["planner_score"], 0)
        self.assertFalse(result["optimization_rerun"])
        self.assertFalse(result["timings_or_solver_optimality_verified"])
        self.assertFalse(result["leaderboard_compatible"])

    def test_changed_file_without_new_checksum_fails(self):
        path = self.root / "report.json"
        path.write_bytes(path.read_bytes() + b" ")
        self.reject("checksum mismatch")

    def test_rehashed_false_evaluation_and_consistent_summaries_fail(self):
        path = self.case_dir / "result.json"
        row = load_json(path)
        row["planner"]["evaluation"]["score"] = 123
        row["planner"]["evaluation"]["reward"] = 123
        row["delta_score"] = 123
        write_json(path, row)
        self.update_row_in_report(row)
        self.rehash()
        self.reject("planner evaluation mismatch")

    def test_rehashed_false_overall_summary_fails(self):
        report = load_json(self.root / "report.json")
        report["summary"]["wins"] = 12
        report["summary"]["ties"] = 0
        write_json(self.root / "report.json", report)
        self.rehash()
        self.reject("overall summary mismatch")

    def test_rehashed_undeclared_report_claim_fails(self):
        report = load_json(self.root / "report.json")
        report["official_rank"] = 1
        write_json(self.root / "report.json", report)
        self.rehash()
        self.reject("report fields mismatch")

    def test_rehashed_invalid_plan_fails_even_with_updated_plan_identity(self):
        data = load_json(self.case_dir / "scenario.json")
        task, mode = data["tasks"][0], data["tasks"][0]["modes"][0]
        plan = {"schema": "civilian-inspection-plan/v1", "scenario_id": self.case_id,
                "assignments": [{"task_id": task["id"], "worker_id": mode["worker_id"],
                                 "start_ms": task["release_ms"],
                                 "finish_ms": task["release_ms"] + mode["duration_ms"] + 1}]}
        plan_path = self.case_dir / "planner-plan.json"
        write_json(plan_path, plan)
        row = load_json(self.case_dir / "result.json")
        row["planner"]["plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        write_json(self.case_dir / "result.json", row)
        self.update_row_in_report(row)
        self.rehash()
        self.reject("invalid planner plan")

    def test_rehashed_input_change_fails_frozen_identity(self):
        path = self.case_dir / "scenario.json"
        data = load_json(path)
        data["tasks"][0]["reward"] += 1
        write_json(path, data)
        self.rehash()
        self.reject("scenario differs from frozen input")

    def test_rehashed_source_identity_change_fails(self):
        path = self.root / "sources.json"
        data = load_json(path)
        data["planner.py"] = "0" * 64
        write_json(path, data)
        self.rehash()
        self.reject("current source identities mismatch")

    def test_rehashed_protocol_change_fails(self):
        path = self.root / "protocol.json"
        data = load_json(path)
        data["development_seeds"] = [99]
        write_json(path, data)
        self.rehash()
        self.reject("recorded protocol mismatch")

    def test_missing_inventory_entry_fails(self):
        manifest = self.root / "SHA256SUMS"
        lines = manifest.read_text().splitlines(True)
        manifest.write_text("".join(lines[1:]))
        self.reject("inventory file set mismatch")

    def test_missing_file_cannot_be_hidden_by_rehashing(self):
        (self.case_dir / "baseline-plan.json").unlink()
        self.rehash()
        self.reject("required evidence file set mismatch")

    def test_extra_file_cannot_be_added_even_when_inventoried(self):
        (self.root / "unrelated.json").write_text("{}")
        self.rehash()
        self.reject("required evidence file set mismatch")

    def test_extra_empty_directory_fails(self):
        (self.root / "extra").mkdir()
        self.reject("inventory directory set mismatch")

    def test_symlink_fails_before_reading_target(self):
        (self.root / "alias.json").symlink_to(self.root / "report.json")
        self.reject("symlink")

    def test_traversal_and_duplicate_inventory_paths_fail(self):
        path = self.root / "SHA256SUMS"
        original = path.read_text()
        path.write_text(original + "0" * 64 + "  ../outside.json\n")
        self.reject("unsafe inventory path")
        path.write_text(original + original.splitlines(True)[0])
        self.reject("duplicate inventory path")

    def test_rehashed_duplicate_keys_and_nonfinite_numbers_fail(self):
        path = self.root / "report.json"
        original = path.read_bytes()
        path.write_bytes(original.replace(b'"status": "passed"',
                                         b'"status": "failed", "status": "passed"', 1))
        self.rehash()
        self.reject("duplicate JSON key")
        data = json.loads(original)
        data["limits"]["time_limit_s"] = float("nan")
        path.write_text(json.dumps(data))
        self.rehash()
        self.reject("nonfinite JSON number")

    def test_malformed_report_returns_json_failure_without_traceback(self):
        write_json(self.root / "report.json", [])
        self.rehash()
        self.reject("report fields mismatch")


if __name__ == "__main__":
    unittest.main()
