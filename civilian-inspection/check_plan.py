#!/usr/bin/env python3
"""Independently check a civilian inspection schedule and recompute its score."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from inspectionkit.io import read_json
from inspectionkit.model import parse_plan, parse_scenario
from inspectionkit.validate import evaluate, evaluation_to_dict


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario")
    parser.add_argument("plan")
    args = parser.parse_args(argv)
    try:
        result = evaluate(parse_scenario(read_json(args.scenario)), parse_plan(read_json(args.plan)))
    except (OSError, ValueError) as exc:
        print("check_plan: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps(evaluation_to_dict(result), allow_nan=False, sort_keys=True, indent=2))
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
