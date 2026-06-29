"""Parsing of core-count range specifications and factorization into
``(ntasks, cpus_per_task)`` layouts.

A range spec is a comma-separated list of terms, where each term is either a
single integer or an inclusive ``start-end`` range with an optional ``:step``::

    "1,2,4,8,16-128:8"

The ``:step`` stride follows SLURM array syntax (e.g. ``--array=0-15:4``), where
``:`` means "increment by". (Note ``%`` is *not* used here: in SLURM ``%`` is the
array concurrency throttle, a different meaning.)
"""

from __future__ import annotations


def parse_core_ranges(spec: str) -> list[int]:
    """Expand a range spec like ``"1,2,4,8,16-128:8"`` into a sorted unique list.

    Single values (``64``) and ranges (``18-32``) are both accepted; a trailing
    ``:step`` on a range sets the stride (default 1), matching SLURM's array
    increment syntax. The end of a range is inclusive.
    """
    if spec is None:
        raise ValueError("core range spec is empty")

    values: set[int] = set()
    for raw_term in spec.split(","):
        term = raw_term.strip()
        if not term:
            continue

        step = 1
        if ":" in term:
            term, step_str = term.split(":", 1)
            term = term.strip()
            step = int(step_str.strip())
            if step <= 0:
                raise ValueError(f"step must be positive in term: {raw_term!r}")

        if "-" in term:
            start_str, end_str = term.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if end < start:
                raise ValueError(f"range end before start in term: {raw_term!r}")
            values.update(range(start, end + 1, step))
        else:
            values.add(int(term))

    if not values:
        raise ValueError(f"no core counts parsed from spec: {spec!r}")

    return sorted(values)


def factorize(
    total: int,
    max_cpus_per_task: int | None = None,
    allowed_cpus_per_task: list[int] | None = None,
) -> list[tuple[int, int]]:
    """Return all ``(ntasks, cpus_per_task)`` pairs whose product equals ``total``.

    Pairs are returned ordered by descending ``ntasks`` (i.e. pure-MPI first,
    then increasing OpenMP threads).

    Parameters
    ----------
    total:
        Total core count to factorize.
    max_cpus_per_task:
        If given, drop any layout whose cpus-per-task exceeds this value (e.g.
        the number of cores in one socket).
    allowed_cpus_per_task:
        If given, only keep layouts whose cpus-per-task is in this list (e.g.
        ``[1, 2, 4, 8]`` to restrict to powers of two).
    """
    if total <= 0:
        raise ValueError(f"total cores must be positive, got {total}")

    allowed = set(allowed_cpus_per_task) if allowed_cpus_per_task else None

    pairs: list[tuple[int, int]] = []
    for cpus_per_task in range(1, total + 1):
        if total % cpus_per_task != 0:
            continue
        if max_cpus_per_task is not None and cpus_per_task > max_cpus_per_task:
            continue
        if allowed is not None and cpus_per_task not in allowed:
            continue
        ntasks = total // cpus_per_task
        pairs.append((ntasks, cpus_per_task))

    pairs.sort(key=lambda p: (-p[0], p[1]))
    return pairs


def build_layouts(
    cores_spec: str,
    max_cpus_per_task: int | None = None,
    allowed_cpus_per_task: list[int] | None = None,
) -> list[tuple[int, int, int]]:
    """Expand a cores spec into ``(total, ntasks, cpus_per_task)`` triples.

    Combines :func:`parse_core_ranges` with :func:`factorize`. Triples are
    ordered by total core count, then by descending ntasks.
    """
    layouts: list[tuple[int, int, int]] = []
    for total in parse_core_ranges(cores_spec):
        for ntasks, cpus_per_task in factorize(
            total, max_cpus_per_task, allowed_cpus_per_task
        ):
            layouts.append((total, ntasks, cpus_per_task))
    return layouts
