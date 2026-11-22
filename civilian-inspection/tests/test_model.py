"""Data-contract boundary tests, independent of the planner."""
import copy
import dataclasses
import unittest

from inspectionkit.model import (
    Assignment, MAX_TASKS, MAX_WORKERS, Plan, ValidationError,
    parse_plan, parse_scenario, plan_to_dict, scenario_to_dict,
)


def example():
    return {
        "schema": "civilian-inspection/v1", "scenario_id": "factory-001",
        "horizon_ms": 100, "energy_cost_per_mwh": 2,
        "workers": [{"id": "edge-a", "available": True, "available_from_ms": 0,
                     "energy_budget_mwh": 20, "activation_cost": 3}],
        "tasks": [{"id": "image-001", "reward": 40, "release_ms": 0,
                   "deadline_ms": 100, "dependencies": [],
                   "modes": [{"worker_id": "edge-a", "duration_ms": 10, "energy_mwh": 4}]}],
    }


class ScenarioContractTests(unittest.TestCase):
    def test_round_trip_and_frozen_values(self):
        original = example()
        scenario = parse_scenario(original)
        self.assertEqual(scenario_to_dict(scenario), original)
        self.assertIsInstance(scenario.workers, tuple)
        self.assertIsInstance(scenario.tasks[0].modes, tuple)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            scenario.horizon_ms = 1
        original["workers"][0]["id"] = "changed"
        self.assertEqual(scenario.workers[0].id, "edge-a")

    def test_missing_and_unknown_fields_fail_closed_at_each_level(self):
        paths = [(), ("workers", 0), ("tasks", 0), ("tasks", 0, "modes", 0)]
        for path in paths:
            with self.subTest(path=path):
                data = example()
                item = data
                for key in path:
                    item = item[key]
                item["undeclared"] = 1
                with self.assertRaisesRegex(ValidationError, "unknown keys"):
                    parse_scenario(data)
                del item["undeclared"]
                del item[next(iter(item))]
                with self.assertRaisesRegex(ValidationError, "missing keys"):
                    parse_scenario(data)

    def test_unsupported_or_original_fleet_schema_is_rejected(self):
        for schema in ("fleetkit/v1", "civilian-inspection/v2", "", None, 1):
            data = example()
            data["schema"] = schema
            with self.subTest(schema=schema), self.assertRaises(ValidationError):
                parse_scenario(data)
        with self.assertRaises(ValidationError):
            parse_scenario({"seed": 1, "aircraft": [], "contacts": [], "rules": {}})

    def test_no_action_or_transfer_fields_are_inferred(self):
        for key, value in (("action", "engage"), ("network_url", "https://example.test"),
                           ("coordinates", [0, 0, 0])):
            data = example()
            data["tasks"][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValidationError, "unknown keys"):
                parse_scenario(data)

    def test_numeric_fields_reject_bool_float_string_and_nonfinite(self):
        paths = [
            ("horizon_ms",), ("energy_cost_per_mwh",),
            ("workers", 0, "available_from_ms"), ("workers", 0, "energy_budget_mwh"),
            ("workers", 0, "activation_cost"), ("tasks", 0, "reward"),
            ("tasks", 0, "release_ms"), ("tasks", 0, "deadline_ms"),
            ("tasks", 0, "modes", 0, "duration_ms"), ("tasks", 0, "modes", 0, "energy_mwh"),
        ]
        for path in paths:
            for value in (True, False, 1.0, "1", float("nan"), float("inf"), -1):
                data = example()
                item = data
                for key in path[:-1]:
                    item = item[key]
                item[path[-1]] = value
                with self.subTest(path=path, value=value), self.assertRaises(ValidationError):
                    parse_scenario(data)

    def test_available_requires_actual_boolean(self):
        for value in (0, 1, "true", None):
            data = example()
            data["workers"][0]["available"] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                parse_scenario(data)

    def test_identifiers_are_bounded_ascii_without_whitespace(self):
        for value in ("", "_leading", "bad id", "a\n", "é", "a" * 65, 1):
            data = example()
            data["scenario_id"] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                parse_scenario(data)
        data = example()
        data["scenario_id"] = "a" * 64
        self.assertEqual(parse_scenario(data).scenario_id, "a" * 64)

    def test_invalid_shapes_and_array_bounds(self):
        for value in (None, [], "text", 1, {1: "value"}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                parse_scenario(value)
        for key, value in (("workers", []), ("workers", {}), ("tasks", {}),
                           ("workers", [example()["workers"][0]] * (MAX_WORKERS + 1)),
                           ("tasks", [example()["tasks"][0]] * (MAX_TASKS + 1))):
            data = example()
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError):
                parse_scenario(data)
        data = example()
        data["tasks"] = []
        self.assertEqual(parse_scenario(data).tasks, ())

    def test_duplicate_workers_tasks_and_modes(self):
        for key in ("workers", "tasks"):
            data = example()
            data[key].append(copy.deepcopy(data[key][0]))
            with self.subTest(key=key), self.assertRaisesRegex(ValidationError, "duplicates"):
                parse_scenario(data)
        data = example()
        data["tasks"][0]["modes"] *= 2
        with self.assertRaisesRegex(ValidationError, "duplicates"):
            parse_scenario(data)

    def test_unknown_mode_worker_and_empty_modes_rejected(self):
        data = example()
        data["tasks"][0]["modes"][0]["worker_id"] = "elsewhere"
        with self.assertRaisesRegex(ValidationError, "unknown worker"):
            parse_scenario(data)
        data["tasks"][0]["modes"] = []
        with self.assertRaises(ValidationError):
            parse_scenario(data)

    def test_time_and_numeric_upper_bounds(self):
        mutations = [
            (("horizon_ms",), 0), (("horizon_ms",), 1_000_000_001),
            (("energy_cost_per_mwh",), 1_000_000_001),
            (("workers", 0, "energy_budget_mwh"), 1_000_000_000_001),
            (("workers", 0, "available_from_ms"), 101),
            (("tasks", 0, "release_ms"), 101), (("tasks", 0, "deadline_ms"), 101),
            (("tasks", 0, "reward"), 1_000_000_001),
            (("tasks", 0, "modes", 0, "duration_ms"), 0),
            (("tasks", 0, "modes", 0, "duration_ms"), 101),
            (("tasks", 0, "modes", 0, "energy_mwh"), 1_000_000_000_001),
        ]
        for path, value in mutations:
            data = example()
            item = data
            for key in path[:-1]:
                item = item[key]
            item[path[-1]] = value
            with self.subTest(path=path, value=value), self.assertRaises(ValidationError):
                parse_scenario(data)

    def test_release_must_not_exceed_deadline(self):
        data = example()
        data["tasks"][0].update(release_ms=40, deadline_ms=30)
        with self.assertRaisesRegex(ValidationError, "release_ms exceeds"):
            parse_scenario(data)

    def test_infeasible_but_well_formed_task_is_not_a_schema_error(self):
        data = example()
        data["workers"][0].update(available=False, energy_budget_mwh=0)
        data["tasks"][0].update(release_ms=100, deadline_ms=100)
        self.assertFalse(parse_scenario(data).workers[0].available)

    def test_dependency_references_self_duplicates_and_cycles(self):
        for dependencies, message in ((["missing"], "unknown dependency"),
                                       (["image-001"], "itself"),
                                       (["missing", "missing"], "duplicates")):
            data = example()
            data["tasks"][0]["dependencies"] = dependencies
            with self.subTest(dependencies=dependencies), self.assertRaisesRegex(ValidationError, message):
                parse_scenario(data)
        data = example()
        other = copy.deepcopy(data["tasks"][0])
        other.update(id="image-002", dependencies=["image-001"])
        data["tasks"][0]["dependencies"] = ["image-002"]
        data["tasks"].append(other)
        with self.assertRaisesRegex(ValidationError, "cycle"):
            parse_scenario(data)

    def test_maximum_dependency_chain_and_out_of_order_declarations(self):
        data = example()
        template = data["tasks"][0]
        data["tasks"] = []
        for index in range(MAX_TASKS):
            task = copy.deepcopy(template)
            task.update(id=f"task-{index}", dependencies=[] if index == 0 else [f"task-{index - 1}"])
            data["tasks"].append(task)
        data["tasks"].reverse()
        self.assertEqual(len(parse_scenario(data).tasks), MAX_TASKS)


class PlanContractTests(unittest.TestCase):
    def test_round_trip(self):
        plan = Plan("factory-001", (Assignment("image-001", "edge-a", 0, 10),))
        self.assertEqual(parse_plan(plan_to_dict(plan)), plan)

    def test_schema_fields_and_numeric_types(self):
        valid = plan_to_dict(Plan("factory-001", (Assignment("image-001", "edge-a", 0, 10),)))
        for mutation in ("schema", "unknown", "missing", "nested", "time", "negative", "large"):
            data = copy.deepcopy(valid)
            if mutation == "schema":
                data["schema"] = "fleet-plan/v1"
            elif mutation == "unknown":
                data["score"] = 999
            elif mutation == "missing":
                del data["scenario_id"]
            elif mutation == "nested":
                data["assignments"][0]["reward"] = 999
            elif mutation == "time":
                data["assignments"][0]["start_ms"] = False
            elif mutation == "negative":
                data["assignments"][0]["start_ms"] = -1
            else:
                data["assignments"][0]["finish_ms"] = 1_000_000_001
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                parse_plan(data)

    def test_plan_size_is_bounded(self):
        data = plan_to_dict(Plan("factory-001", (Assignment("image-001", "edge-a", 0, 10),)))
        data["assignments"] *= MAX_TASKS + 1
        with self.assertRaises(ValidationError):
            parse_plan(data)


if __name__ == "__main__":
    unittest.main()
