#!/usr/bin/env python3
"""Replay civilian benchmark evidence without running either optimizer.

The current checkout is the trusted source reference. This checks artifact
consistency, input identity, schedule validity and arithmetic. It does not prove
source execution, timing, solver optimality, job execution or leaderboard results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
from typing import Any, Dict, Optional

from inspectionkit.generate import cases, protocol
from inspectionkit.model import parse_plan, parse_scenario
from inspectionkit.validate import evaluate, evaluation_to_dict


ROOT = Path(__file__).resolve().parent
MAX_FILE_BYTES = 4 * 1024 * 1024


class VerificationError(ValueError):
    """Evidence does not support its recorded claims."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path, limit: int = MAX_FILE_BYTES) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"not a regular file: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    _require(len(raw) <= limit, f"file exceeds verification limit: {path.name}")
    return raw


def _pairs(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _number(raw: str, integer: bool = False) -> Any:
    _require(len(raw) <= 128, "JSON number exceeds digit limit")
    value = int(raw) if integer else float(raw)
    _require(integer or math.isfinite(value), "nonfinite JSON number")
    return value


def _json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_int=lambda text: _number(text, True),
                          parse_float=_number, parse_constant=_number)
    except RecursionError as exc:
        raise VerificationError("JSON nesting exceeds supported depth") from exc


def _same(actual: Any, expected: Any, label: str) -> None:
    # Canonical JSON also distinguishes bool/int and float/int, unlike Python ==.
    encode = lambda value: json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    _require(encode(actual) == encode(expected), f"{label} mismatch")


def _inventory(root: Path) -> Dict[str, bytes]:
    _require(root.is_dir() and not root.is_symlink(), "evidence must be a real directory")
    manifest = _read(root / "SHA256SUMS", 65536).decode("utf-8")
    declared = {}
    for line in manifest.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        _require(match is not None, "malformed SHA256SUMS line")
        digest, name = match.groups()
        path = PurePosixPath(name)
        _require(not path.is_absolute() and path.as_posix() == name
                 and all(part not in (".", "..") for part in path.parts)
                 and "\\" not in name and name != "SHA256SUMS", "unsafe inventory path")
        _require(name not in declared, "duplicate inventory path")
        declared[name] = digest
    observed = {}
    directories = set()
    total_bytes = 0
    for count, path in enumerate(root.rglob("*"), 1):
        _require(count <= 256 and not path.is_symlink(), "excess entries or symlink in evidence")
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.add(relative)
        elif relative != "SHA256SUMS":
            observed[relative] = _read(path)
            total_bytes += len(observed[relative])
            _require(total_bytes <= 32 * 1024 * 1024, "evidence exceeds total size limit")
    _same(sorted(observed), sorted(declared), "inventory file set")
    expected_dirs = {parent.as_posix() for name in declared
                     for parent in PurePosixPath(name).parents if parent.as_posix() != "."}
    _same(sorted(directories), sorted(expected_dirs), "inventory directory set")
    for name, raw in observed.items():
        _require(_digest(raw) == declared[name], f"checksum mismatch: {name}")
    return observed


def _summary(rows: list) -> dict:
    baseline = sum(row["baseline"]["evaluation"]["score"] for row in rows)
    planner = sum(row["planner"]["evaluation"]["score"] for row in rows)
    times = [row["planner"]["elapsed_s"] for row in rows]
    deltas = [row["delta_score"] for row in rows]
    return {"cases": len(rows), "valid_baseline_plans": len(rows), "valid_planner_plans": len(rows),
            "baseline_score": baseline, "planner_score": planner, "delta_score": planner - baseline,
            "score_ratio": planner / baseline if baseline > 0 else None,
            "wins": sum(value > 0 for value in deltas), "ties": sum(value == 0 for value in deltas),
            "losses": sum(value < 0 for value in deltas),
            "baseline_elapsed_s": sum(row["baseline"]["elapsed_s"] for row in rows),
            "planner_elapsed_s": sum(times),
            "planner_median_elapsed_s": statistics.median(times) if times else None,
            "planner_max_elapsed_s": max(times) if times else None}


def verify(evidence: Path, *, source_root: Optional[Path] = None) -> dict:
    """Verify the complete selected split against frozen inputs and current code."""
    root = ROOT if source_root is None else Path(source_root)
    files = _inventory(Path(evidence))
    for name in ("sources.json", "protocol.json", "inputs.json", "report.json"):
        _require(name in files, f"missing required file: {name}")
    source_names = {"benchmark.py", "planner.py", "check_plan.py", "verify_benchmark.py",
                    "examples/benchmark-protocol.json", "examples/benchmark-inputs.json"}
    source_names.update("inspectionkit/" + name + ".py"
                        for name in ("__init__", "generate", "io", "model", "solver", "validate"))
    source_names.update(path.relative_to(root).as_posix() for path in (root / "inspectionkit").glob("*.py"))
    source_hashes = {name: _digest(_read(root / name)) for name in sorted(source_names)}
    _same(_json(files["sources.json"]), source_hashes, "current source identities")
    frozen_protocol = _read(root / "examples/benchmark-protocol.json")
    _same(_json(frozen_protocol), protocol(), "frozen protocol")
    _same(_json(files["protocol.json"]), protocol(), "recorded protocol")
    frozen = _json(_read(root / "examples/benchmark-inputs.json"))
    _require(frozen["generator_sha256"] == source_hashes["inspectionkit/generate.py"],
             "frozen generator identity mismatch")
    _require(frozen["protocol_sha256"] == _digest(frozen_protocol), "frozen protocol identity mismatch")
    definitions = [{"scenario_id": case.scenario_id, "split": case.split,
                    "tier": case.tier, "seed": case.seed} for case in cases("all")]
    _same([{key: row[key] for key in ("scenario_id", "split", "tier", "seed")}
           for row in frozen["cases"]], definitions, "frozen case manifest")
    report = _json(files["report.json"])
    _same(sorted(report), sorted(("schema", "status", "scope", "leaderboard_compatible", "job_execution_tested",
          "selected_split", "inputs_manifest_sha256", "limits", "runtime", "summary", "by_split", "by_tier",
          "cases", "limitations")), "report fields")
    _require(report["schema"] == "civilian-inspection-benchmark/v1" and report["status"] == "passed",
             "report is not a completed civilian benchmark")
    _require(report["scope"] == "Offline schedules of local industrial image-processing jobs", "unsupported scope claim")
    _require(report["leaderboard_compatible"] is False and report["job_execution_tested"] is False,
             "unsupported leaderboard or execution claim")
    split = report["selected_split"]
    selected = cases(split)
    wanted = {case.scenario_id for case in selected}
    inputs = [row for row in frozen["cases"] if row["scenario_id"] in wanted]
    _same(_json(files["inputs.json"]), inputs, "selected frozen inputs")
    _require(report["inputs_manifest_sha256"] == _digest(files["inputs.json"]), "input manifest hash mismatch")
    expected_files = {"sources.json", "protocol.json", "inputs.json", "report.json"}
    for case in selected:
        expected_files.update(f"cases/{case.scenario_id}/{name}.json"
                              for name in ("scenario", "baseline-plan", "planner-plan", "result"))
    _same(sorted(files), sorted(expected_files), "required evidence file set")
    limits = report["limits"]
    _same(sorted(limits), ["max_states", "time_limit_s"], "search limit fields")
    seconds, states = limits["time_limit_s"], limits["max_states"]
    _require(type(seconds) in (int, float) and 0.01 <= seconds <= 50
             and type(states) is int and 1 <= states <= 200000, "invalid recorded search limits")
    rows = []
    for entry in inputs:
        prefix = f"cases/{entry['scenario_id']}/"
        raw = files[prefix + "scenario.json"]
        _require(_digest(raw) == entry["sha256"], "scenario differs from frozen input")
        scenario = parse_scenario(_json(raw))
        _require(scenario.scenario_id == entry["scenario_id"], "scenario identifier mismatch")
        row = _json(files[prefix + "result.json"])
        _same(sorted(row), sorted(("scenario_id", "split", "tier", "seed", "baseline", "planner", "delta_score")),
              "case result fields")
        _same({key: row[key] for key in ("scenario_id", "split", "tier", "seed")},
              {key: entry[key] for key in ("scenario_id", "split", "tier", "seed")}, "result identity")
        scores = []
        for label in ("baseline", "planner"):
            plan_raw = files[prefix + label + "-plan.json"]
            _require(row[label]["plan_sha256"] == _digest(plan_raw), "recorded plan identity mismatch")
            result = evaluate(scenario, parse_plan(_json(plan_raw)))
            _require(result.valid, f"invalid {label} plan: {entry['scenario_id']}")
            _same(row[label]["evaluation"], evaluation_to_dict(result), f"{label} evaluation")
            elapsed = row[label]["elapsed_s"]
            _require(type(elapsed) in (int, float) and elapsed >= 0, "invalid recorded elapsed time")
            scores.append(result.score)
        _require(row["planner"]["baseline_identity_matches"] is True, "reported baseline mismatch")
        _same(row["delta_score"], scores[1] - scores[0], "case score difference")
        _require(scores[1] >= scores[0], "recorded planner is worse than baseline")
        rows.append(row)
    _same(report["cases"], rows, "report case results")
    _same(report["summary"], _summary(rows), "overall summary")
    _same(report["by_split"], {name: _summary([row for row in rows if row["split"] == name])
                              for name in ("development", "heldout") if any(row["split"] == name for row in rows)},
          "split summaries")
    _same(report["by_tier"], {name: _summary([row for row in rows if row["tier"] == name])
                             for name in ("small", "medium", "large")}, "tier summaries")
    return {"valid": True, "files_verified": len(files), "cases_verified": len(rows),
            "plans_verified": 2 * len(rows), "summary": _summary(rows),
            "source_identities_verified": True, "optimization_rerun": False,
            "timings_or_solver_optimality_verified": False, "leaderboard_compatible": False}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.evidence)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
