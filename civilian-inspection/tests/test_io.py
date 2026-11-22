"""Boundary tests for untrusted JSON and transactional plan replacement."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from inspectionkit.io import MAX_INPUT_BYTES, ensure_distinct_paths, read_json, write_json_atomic


class IOTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_rejects_ambiguous_and_noninteger_json(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}',
                     '{"x":1.0}', '{"x":1e2}', '{"x":1234567890123456789}'):
            with self.subTest(text=text):
                path = self.root / "input.json"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_json(path)

    def test_rejects_oversized_and_invalid_utf8(self):
        path = self.root / "input.json"
        for payload in (b" " * (MAX_INPUT_BYTES + 1), b"\xff"):
            path.write_bytes(payload)
            with self.assertRaises(ValueError):
                read_json(path)

    def test_same_file_and_hard_link_cannot_be_output(self):
        source, alias = self.root / "source.json", self.root / "alias.json"
        source.write_text("{}", encoding="utf-8")
        os.link(source, alias)
        for path in (source, alias):
            with self.assertRaises(ValueError):
                ensure_distinct_paths(source, path)

    def test_failed_replace_retains_previous_output_and_removes_temp(self):
        destination = self.root / "plan.json"
        destination.write_bytes(b"previous-output")
        with mock.patch("inspectionkit.io.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                write_json_atomic(destination, {"valid": True})
        self.assertEqual(destination.read_bytes(), b"previous-output")
        self.assertEqual(list(self.root.iterdir()), [destination])

    def test_serialization_failure_cannot_replace_output(self):
        destination = self.root / "plan.json"
        destination.write_bytes(b"previous-output")
        with self.assertRaises(ValueError):
            write_json_atomic(destination, {"score": float("nan")})
        self.assertEqual(destination.read_bytes(), b"previous-output")

    def test_successful_atomic_output_round_trips(self):
        destination = self.root / "plan.json"
        write_json_atomic(destination, {"schema": "civilian-inspection-plan/v1", "n": 4})
        self.assertEqual(read_json(destination), {"schema": "civilian-inspection-plan/v1", "n": 4})
        self.assertTrue(destination.read_bytes().endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
