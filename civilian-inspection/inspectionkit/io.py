"""Bounded strict JSON input and atomic output for the civilian CLI tools."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple, Union


MAX_INPUT_BYTES = 2 * 1024 * 1024
PathLike = Union[str, os.PathLike]


def _unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {!r}".format(key))
        result[key] = value
    return result


def _integer(text: str) -> int:
    # All scenario/plan numeric values have much smaller schema bounds. Limit
    # conversion itself so malformed files cannot trigger expensive big integers.
    if len(text.lstrip("-")) > 18:
        raise ValueError("JSON integer exceeds the input digit limit")
    return int(text)


def _noninteger(text: str) -> None:
    raise ValueError("scenario and plan numbers must be integers: {!r}".format(text))


def read_json(path: PathLike) -> Any:
    """Read at most 2 MiB, rejecting duplicate keys, floats and nonfinite numbers."""
    with open(path, "rb") as stream:
        data = stream.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("input exceeds {} bytes".format(MAX_INPUT_BYTES))
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_int=_integer,
            parse_float=_noninteger,
            parse_constant=_noninteger,
        )
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds the supported depth") from exc


def ensure_distinct_paths(input_path: PathLike, output_path: PathLike) -> None:
    """Protect the scenario even when two paths name the same file."""
    source, destination = Path(input_path), Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("input and output must be different files")
    if source.exists() and destination.exists() and os.path.samefile(source, destination):
        raise ValueError("input and output refer to the same file")


def write_json_atomic(path: PathLike, value: Any) -> None:
    """Replace the output only after serialization and a complete file write.

    The parent directory must already exist. Temporary files are in that same
    directory, so replacement does not cross filesystems. This guarantees atomic
    replacement on supported local filesystems, not transactional publication
    across several output files or distributed storage.
    """
    payload = (json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".{}-".format(destination.name), suffix=".tmp",
            dir=str(destination.parent), delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
