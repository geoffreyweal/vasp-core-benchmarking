"""Parsing of VASP OUTCAR files for benchmarking.

Three kinds of information are extracted:

  * the parallel layout, from the header line
    ``running N mpi-ranks, with M threads/rank, on K nodes``;
  * per-electronic-step wall times, from lines
    ``LOOP:  cpu time X: real time Y`` (the ``LOOP+`` ionic-step lines are
    deliberately ignored) - both the timing metric and, via their count, the
    "did the benchmark run?" signal used by ``status``/``reset``;
  * an identifiable VASP abort message near the end of the file, used to label
    a run with no usable result as an ``error`` rather than a bare ``failed``.
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


# Error signatures VASP prints into the OUTCAR / stdout when it aborts.
_VASP_ERROR_SIGNATURES = (
    "VERY BAD NEWS",
    "I REFUSE TO CONTINUE",
    "ZBRENT: fatal",
    "Error EDDDAV",
    "EDWAV: internal error",
    "LAPACK: Routine ZPOTRF failed",
    "forrtl: severe",
)

# How much of the end of the file to inspect: the abort messages sit at the end,
# and OUTCARs can be hundreds of MB.
_TAIL_BYTES = 200_000


def _tail(path: str | Path) -> str | None:
    """The last ``_TAIL_BYTES`` of a file as text, or None if it is missing."""
    p = Path(path)
    if not p.is_file():
        return None
    with open(p, "rb") as fh:
        fh.seek(max(0, p.stat().st_size - _TAIL_BYTES))
        return fh.read().decode(errors="replace")


def error_signature(path: str | Path) -> str | None:
    """A VASP abort message found near the end of the OUTCAR, or None."""
    tail = _tail(path)
    if tail is None:
        return None
    for sig in _VASP_ERROR_SIGNATURES:
        if sig in tail:
            return sig
    return None
