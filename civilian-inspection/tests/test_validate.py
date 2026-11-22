"""Independent schedule invariants with hand-computed expected outcomes."""
import dataclasses
import unittest

from inspectionkit.model import Assignment, Mode, Plan, Scenario, Task, Worker
from inspectionkit.validate import evaluate, evaluation_to_dict


def scenario():
    return Scenario(
        "factory-001", 100, 2,
        (Worker("edge-a", True, 0, 20, 3), Worker("edge-b", True, 0, 10, 7)),
        (Task("image-a", 40, 0, 60, (), (Mode("edge-a", 10, 4), Mode("edge-b", 20, 6))),
         Task("image-b", 30, 0, 80, ("image-a",), (Mode("edge-a", 15, 5), Mode("edge-b", 10, 3))),
         Task("image-c", 20, 5, 100, (), (Mode("edge-a", 5, 2),))),
    )


class IndependentValidationTests(unittest.TestCase):
    def assertInvalid(self, assignments, expected, source=None):
        result = evaluate(source or scenario(), Plan("factory-001", tuple(assignments)))
        self.assertFalse(result.valid)
        self.assertIsNone(result.score)
        self.assertEqual(result.reward, 0)
        self.assertEqual(result.completed_tasks, ())
        self.assertTrue(any(expected in message for message in result.violations), result.violations)
        return result

    def test_empty_plan_has_zero_score_without_activation_cost(self):
        result = evaluate(scenario(), Plan("factory-001", ()))
        self.assertTrue(result.valid)
        self.assertEqual((result.score, result.reward, result.energy_mwh, result.activation_cost), (0, 0, 0, 0))
        self.assertEqual(result.unscheduled_tasks, ("image-a", "image-b", "image-c"))

    def test_exact_score_charges_activation_once(self):
        assignments = (Assignment("image-a", "edge-a", 0, 10),
                       Assignment("image-b", "edge-a", 10, 25),
                       Assignment("image-c", "edge-a", 25, 30))
        result = evaluate(scenario(), Plan("factory-001", assignments))
        self.assertTrue(result.valid, result.violations)
        # Rewards: 40+30+20=90. Energy: (4+5+2)*2=22. Activation: 3.
        self.assertEqual((result.score, result.reward, result.energy_mwh, result.energy_cost,
                          result.activation_cost), (65, 90, 11, 22, 3))
        self.assertEqual(result.completed_tasks, ("image-a", "image-b", "image-c"))
        self.assertEqual(result.unscheduled_tasks, ())

    def test_parallel_workers_and_exact_dependency_boundary(self):
        assignments = (Assignment("image-a", "edge-a", 0, 10),
                       Assignment("image-b", "edge-b", 10, 20),
                       Assignment("image-c", "edge-a", 10, 15))
        result = evaluate(scenario(), Plan("factory-001", tuple(reversed(assignments))))
        self.assertTrue(result.valid, result.violations)
        self.assertEqual(result.score, 90 - 2 * 9 - 10)

    def test_duplicate_task_never_collects_duplicate_reward(self):
        self.assertInvalid((Assignment("image-a", "edge-a", 0, 10),
                            Assignment("image-a", "edge-a", 10, 20)), "more than once")

    def test_unknown_task_worker_and_wrong_worker_mode(self):
        for assignment, expected in ((Assignment("missing", "edge-a", 0, 10), "unknown task"),
                                     (Assignment("image-a", "missing", 0, 10), "unknown worker"),
                                     (Assignment("image-c", "edge-b", 5, 10), "no mode")):
            with self.subTest(expected=expected):
                self.assertInvalid((assignment,), expected)

    def test_exact_duration_and_positive_time(self):
        for finish in (0, 9, 11):
            with self.subTest(finish=finish):
                self.assertInvalid((Assignment("image-a", "edge-a", 0, finish),), "duration")
        self.assertInvalid((Assignment("image-a", "edge-a", 10, 0),), "positive elapsed")

    def test_same_worker_overlap_including_nested_intervals(self):
        self.assertInvalid((Assignment("image-a", "edge-a", 0, 10),
                            Assignment("image-c", "edge-a", 5, 10)), "overlaps")
        source = scenario()
        tasks = tuple(dataclasses.replace(task, dependencies=(), modes=(Mode("edge-a", duration, 1),))
                      for task, duration in zip(source.tasks, (40, 5, 5)))
        source = dataclasses.replace(source, tasks=tasks)
        result = self.assertInvalid((Assignment("image-a", "edge-a", 0, 40),
                                     Assignment("image-b", "edge-a", 5, 10),
                                     Assignment("image-c", "edge-a", 20, 25)), "overlaps", source)
        self.assertEqual(sum("overlaps" in message for message in result.violations), 2)

    def test_energy_is_summed_per_worker_not_pooled(self):
        source = scenario()
        source = dataclasses.replace(source, workers=(
            dataclasses.replace(source.workers[0], energy_budget_mwh=8),
            dataclasses.replace(source.workers[1], energy_budget_mwh=1000)))
        self.assertInvalid((Assignment("image-a", "edge-a", 0, 10),
                            Assignment("image-b", "edge-a", 10, 25)), "energy budget", source)

    def test_energy_equal_to_budget_is_valid(self):
        source = scenario()
        source = dataclasses.replace(source, workers=(
            dataclasses.replace(source.workers[0], energy_budget_mwh=9), source.workers[1]))
        result = evaluate(source, Plan("factory-001", (
            Assignment("image-a", "edge-a", 0, 10), Assignment("image-b", "edge-a", 10, 25))))
        self.assertTrue(result.valid, result.violations)

    def test_missing_and_unfinished_dependencies(self):
        self.assertInvalid((Assignment("image-b", "edge-b", 0, 10),), "missing dependency")
        self.assertInvalid((Assignment("image-a", "edge-a", 0, 10),
                            Assignment("image-b", "edge-b", 9, 19)), "before dependency")

    def test_unavailable_and_not_yet_available_worker(self):
        source = scenario()
        disabled = dataclasses.replace(source, workers=(
            dataclasses.replace(source.workers[0], available=False), source.workers[1]))
        self.assertInvalid((Assignment("image-a", "edge-a", 0, 10),), "unavailable worker", disabled)
        delayed = dataclasses.replace(source, workers=(
            dataclasses.replace(source.workers[0], available_from_ms=1), source.workers[1]))
        self.assertInvalid((Assignment("image-a", "edge-a", 0, 10),), "before worker", delayed)

    def test_release_deadline_and_horizon_are_independent_checks(self):
        self.assertInvalid((Assignment("image-c", "edge-a", 4, 9),), "before its release")
        self.assertInvalid((Assignment("image-a", "edge-a", 51, 61),), "after its deadline")
        self.assertInvalid((Assignment("image-c", "edge-a", 96, 101),), "outside the scenario horizon")

    def test_release_deadline_equality_is_allowed(self):
        result = evaluate(scenario(), Plan("factory-001", (Assignment("image-a", "edge-a", 50, 60),)))
        self.assertTrue(result.valid, result.violations)

    def test_wrong_scenario_identifier_is_rejected(self):
        result = evaluate(scenario(), Plan("different", ()))
        self.assertFalse(result.valid)
        self.assertIsNone(result.score)
        self.assertIn("does not match", result.violations[0])

    def test_valid_negative_score_is_not_clamped_to_zero(self):
        source = scenario()
        source = dataclasses.replace(source, tasks=(dataclasses.replace(source.tasks[0], reward=0),))
        result = evaluate(source, Plan("factory-001", (Assignment("image-a", "edge-a", 0, 10),)))
        self.assertTrue(result.valid)
        self.assertEqual(result.score, -11)

    def test_direct_dataclass_types_cannot_bypass_validation(self):
        self.assertInvalid((Assignment("image-a", "edge-a", False, 10),), "invalid plan")
        self.assertInvalid((Assignment("image-a", "edge-a", -1, 9),), "invalid plan")
        source = dataclasses.replace(scenario(), energy_cost_per_mwh=-1)
        result = evaluate(source, Plan("factory-001", ()))
        self.assertFalse(result.valid)
        self.assertIn("invalid scenario", result.violations[0])
        result = evaluate(scenario(), {"assignments": []})
        self.assertFalse(result.valid)

    def test_large_integer_score_is_exact(self):
        source = Scenario("large", 100, 1_000_000_000,
                          (Worker("edge", True, 0, 1_000_000_000_000, 1_000_000_000),),
                          (Task("job", 1_000_000_000, 0, 100, (),
                                (Mode("edge", 1, 999_999_999_999),)),))
        result = evaluate(source, Plan("large", (Assignment("job", "edge", 0, 1),)))
        self.assertTrue(result.valid)
        self.assertEqual(result.score, -999_999_999_999_000_000_000)
        self.assertIs(type(result.score), int)

    def test_result_dictionary_is_json_compatible_and_invalid_score_is_null(self):
        import json
        valid = evaluate(scenario(), Plan("factory-001", ()))
        self.assertEqual(json.loads(json.dumps(evaluation_to_dict(valid)))["score"], 0)
        invalid = evaluate(scenario(), Plan("wrong", ()))
        result = evaluation_to_dict(invalid)
        self.assertIsNone(json.loads(json.dumps(result))["score"])
        self.assertIsInstance(result["violations"], list)


if __name__ == "__main__":
    unittest.main()
