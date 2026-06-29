"""Parsing of VASP OUTCAR files for benchmarking.

Two pieces of information are extracted:

  * the parallel layout, from the header line
    ``running N mpi-ranks, with M threads/rank, on K nodes``;
  * per-electronic-step wall times, from lines
    ``LOOP:  cpu time X: real time Y`` (the ``LOOP+`` ionic-step lines are
    deliberately ignored).
"""

from __future__ import annotations

import re
from pathlib import Path

# "running    8 mpi-ranks, with    4 threads/rank, on    1 nodes"
_HEADER_RE = re.compile(
    r"running\s+(\d+)\s+mpi-ranks,\s+with\s+(\d+)\s+threads/rank,\s+on\s+(\d+)\s+nodes"
)

# "      LOOP:  cpu time     10.7003: real time     10.7910"
# Anchored on "LOOP:" so the "LOOP+:" ionic lines do not match.
_LOOP_RE = re.compile(
    r"^\s*LOOP:\s+cpu time\s+([\d.]+)\s*:\s*real time\s+([\d.]+)", re.MULTILINE
)


def parse_outcar_header(path: str | Path) -> tuple[int, int, int] | None:
    """Return ``(ntasks, cpus_per_task, nodes)`` from the OUTCAR header.

    Returns ``None`` if the header line is absent (e.g. a truncated/empty file).
    """
    text = Path(path).read_text(errors="replace")
    match = _HEADER_RE.search(text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def parse_loop_times(path: str | Path) -> list[float]:
    """Return the per-electronic-step ``real time`` values (seconds)."""
    text = Path(path).read_text(errors="replace")
    return [float(m.group(2)) for m in _LOOP_RE.finditer(text)]
