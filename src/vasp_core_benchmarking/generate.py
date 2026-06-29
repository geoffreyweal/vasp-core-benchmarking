"""Part 1: generate the VASP benchmarking directory tree.

Creates one directory per ``(ntasks, cpus_per_task)`` layout, copies the VASP
input files into each, and writes a ``submit.sl`` SLURM script. The INCAR is
copied unchanged; only ``OMP_NUM_THREADS`` (= cpus-per-task) is varied via the
submit script.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .core_ranges import build_layouts

# Input files VASP expects. Any other files found in the VASP_Files directory are
# also copied (e.g. ML_FF, WAVECAR, CHGCAR), but these four are required.
REQUIRED_INPUTS = ["INCAR", "POSCAR", "POTCAR", "KPOINTS"]

DEFAULT_INCLUDE = "vasp_core_benchmarking_submit_include.txt"


def time_to_seconds(time_str: str) -> int:
    """Convert a SLURM time string to seconds.

    Accepts ``HH:MM:SS``, ``MM:SS``, ``D-HH:MM:SS`` and ``D-HH:MM``.
    """
    days = 0
    rest = time_str.strip()
    if "-" in rest:
        day_str, rest = rest.split("-", 1)
        days = int(day_str)

    parts = [int(p) for p in rest.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 1:
        h, m, s = parts[0], 0, 0
    else:
        raise ValueError(f"unrecognised time format: {time_str!r}")

    return days * 86400 + h * 3600 + m * 60 + s


def seconds_to_dhms(seconds: int) -> str:
    """Human-readable ``Dd HH:MM:SS`` form for a total-walltime summary."""
    days, rem = divmod(int(seconds), 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if days:
        return f"{days}d {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def config_dirname(total: int, ntasks: int, cpus_per_task: int) -> str:
    return f"{total}cores_{ntasks}tasks_{cpus_per_task}cpt"


# Directives the tool sets itself, per benchmark configuration. If the user puts
# any of these in the include file they are dropped and replaced, so the layout
# being benchmarked is always authoritative.
MANAGED_DIRECTIVES = ("--job-name", "--ntasks", "--cpus-per-task")


def parse_mem_to_mb(mem_str: str) -> float:
    """Convert a memory string (``8G``, ``8GB``, ``4000M``, ``4000``) to MB.

    A bare number is treated as MB, matching SLURM's default unit.
    """
    s = mem_str.strip().upper()
    try:
        if s.endswith("GB"):
            return float(s[:-2]) * 1024
        if s.endswith("G"):
            return float(s[:-1]) * 1024
        if s.endswith("MB"):
            return float(s[:-2])
        if s.endswith("M"):
            return float(s[:-1])
        return float(s)  # bare number = MB
    except ValueError:
        raise ValueError(
            f"unrecognised memory value {mem_str!r}; use e.g. 8G, 8GB, 4000M or 4000"
        ) from None


def memory_directive(
    total_cores: int,
    mem: str | None = None,
    mem_per_cpu: str | None = None,
) -> str | None:
    """Return the ``#SBATCH`` memory line for a job, or None to leave it alone.

    Mirrors the cp2k-benchmarking policy: ``--mem`` is a flat total-memory floor
    and ``--mem-per-cpu`` scales with core count. When both are given, each job
    gets whichever yields more memory - the per-cpu value once
    ``total_cores * mem_per_cpu`` exceeds the floor, otherwise the flat total.
    If only one is given it is used as-is; if neither, returns None.
    """
    if mem is not None and mem_per_cpu is not None:
        if total_cores * parse_mem_to_mb(mem_per_cpu) > parse_mem_to_mb(mem):
            return f"#SBATCH --mem-per-cpu={mem_per_cpu}"
        return f"#SBATCH --mem={mem}"
    if mem_per_cpu is not None:
        return f"#SBATCH --mem-per-cpu={mem_per_cpu}"
    if mem is not None:
        return f"#SBATCH --mem={mem}"
    return None


def parse_time_policy(policy: str) -> tuple[list[str], list[int]]:
    """Parse a time-policy string into (times, thresholds).

    Format ``"T1,T2,...@C1,C2,..."`` - N+1 walltimes for N ascending core
    thresholds, e.g. ``"30:00,15:00,10:00@16,64"`` means: <=16 cores -> 30:00,
    <=64 cores -> 15:00, otherwise -> 10:00. Mirrors cp2k-benchmarking.
    """
    if "@" not in policy:
        raise ValueError(
            "invalid --time-policy: expected 'T1,T2,...@C1,C2,...' "
            "(e.g. '30:00,15:00,10:00@16,64')"
        )
    times_part, thresholds_part = policy.split("@", 1)
    times = [t.strip() for t in times_part.split(",") if t.strip()]
    thresholds = [int(x.strip()) for x in thresholds_part.split(",") if x.strip()]
    if len(times) != len(thresholds) + 1:
        raise ValueError(
            "--time-policy must have one more time than thresholds "
            f"(got {len(times)} times, {len(thresholds)} thresholds)"
        )
    # Validate each time is a recognised SLURM walltime.
    for t in times:
        time_to_seconds(t)
    return times, thresholds


def select_time(total_cores: int, times: list[str], thresholds: list[int]) -> str:
    """Pick the walltime for ``total_cores`` from a parsed time policy."""
    for time_str, max_cores in zip(times, thresholds):
        if total_cores <= max_cores:
            return time_str
    return times[-1]


def time_directive(total_cores: int, time_policy: str | None = None) -> str | None:
    """Return the ``#SBATCH --time`` line for a job, or None to leave it alone."""
    if time_policy is None:
        return None
    times, thresholds = parse_time_policy(time_policy)
    return f"#SBATCH --time={select_time(total_cores, times, thresholds)}"


def directive_name(sbatch_line: str) -> str:
    """Return the option name of a ``#SBATCH`` line, e.g. ``--ntasks``."""
    rest = sbatch_line.split("#SBATCH", 1)[1].strip()
    if not rest:
        return ""
    return rest.split()[0].split("=", 1)[0]


def directive_value(sbatch_line: str) -> str | None:
    """Return the value of a ``#SBATCH`` line, handling ``=`` or space form.

    Trailing comments are ignored, so ``#SBATCH --time=00:15:00  # foo`` yields
    ``00:15:00``.
    """
    rest = sbatch_line.split("#SBATCH", 1)[1].strip()
    tokens = rest.split()
    if not tokens:
        return None
    first = tokens[0]
    if "=" in first:
        return first.split("=", 1)[1]
    return tokens[1] if len(tokens) > 1 else None


def split_include(include_content: str) -> tuple[list[str], str]:
    """Separate an include file into hoisted ``#SBATCH`` lines and the body.

    SLURM only honours ``#SBATCH`` directives that appear before the first
    executable line, so any ``#SBATCH`` lines the user writes in the include are
    lifted out and placed in the header. Everything else (module loads, srun
    commands, ordinary comments) is kept, in order, as the body.
    """
    sbatch_lines: list[str] = []
    body_lines: list[str] = []
    for line in include_content.splitlines():
        if line.lstrip().startswith("#SBATCH"):
            sbatch_lines.append(line.strip())
        else:
            body_lines.append(line)
    return sbatch_lines, "\n".join(body_lines).strip("\n")


def build_submit_script(
    *,
    job_name: str,
    ntasks: int,
    cpus_per_task: int,
    include_content: str,
    mem_line: str | None = None,
    time_line: str | None = None,
) -> str:
    """Assemble a submit.sl: SBATCH header + OMP export + include body.

    Most cluster/job directives (account, partition, time, node binding, ...)
    come from the include file's ``#SBATCH`` lines. The tool sets ``--job-name``,
    ``--ntasks`` and ``--cpus-per-task`` (and the matching ``OMP_NUM_THREADS``).
    If ``mem_line``/``time_line`` are given (per-config memory/walltime), they are
    injected too and the matching ``--mem``/``--mem-per-cpu``/``--time`` from the
    include is dropped.
    """
    include_sbatch, include_body = split_include(include_content)
    # Drop any user copies of the directives we manage so they cannot conflict.
    managed = set(MANAGED_DIRECTIVES)
    if mem_line is not None:
        managed |= {"--mem", "--mem-per-cpu"}
    if time_line is not None:
        managed |= {"--time"}
    include_sbatch = [
        ln for ln in include_sbatch if directive_name(ln) not in managed
    ]

    lines = [
        "#!/bin/bash -e",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
    ]
    if mem_line is not None:
        lines.append(mem_line)
    if time_line is not None:
        lines.append(time_line)
    # Everything else (account, partition, time, node binding, ...).
    lines += include_sbatch

    lines += [
        "",
        # OpenMP threads = cpus-per-task; read it from SLURM so the export always
        # matches the --cpus-per-task above. OMP_STACKSIZE is set large because
        # VASP's OpenMP regions otherwise segfault on the default per-thread stack.
        "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}",
        "export OMP_STACKSIZE=512m",
        "",
        include_body,
        "",
    ]
    return "\n".join(lines)


def setup(
    *,
    cores: str,
    jobname_prefix: str = "vasp_bench",
    vasp_files: str = "VASP_Files",
    include: str | None = None,
    root: str = "VASP_Benchmarking",
    max_cpus_per_task: int | None = None,
    allowed_cpus_per_task: list[int] | None = None,
    mem: str | None = None,
    mem_per_cpu: str | None = None,
    time_policy: str | None = None,
) -> list[Path]:
    """Generate the benchmarking tree. Returns the list of created directories.

    Most SLURM job settings (account, partition, node binding, ...) are taken
    from the include file; this function varies the parallel layout (job-name,
    ntasks, cpus-per-task). If ``mem``/``mem_per_cpu`` are given the memory
    directive is set per config (see :func:`memory_directive`); if
    ``time_policy`` is given the ``--time`` directive is set per config (see
    :func:`time_directive`). Either overrides the matching line in the include.
    """
    # Validate memory/time strings up front so a bad value fails before any files
    # are written.
    if mem is not None:
        parse_mem_to_mb(mem)
    if mem_per_cpu is not None:
        parse_mem_to_mb(mem_per_cpu)
    if time_policy is not None:
        parse_time_policy(time_policy)

    vasp_files_dir = Path(vasp_files)
    if not vasp_files_dir.is_dir():
        raise FileNotFoundError(f"VASP input directory not found: {vasp_files_dir}")

    missing = [f for f in REQUIRED_INPUTS if not (vasp_files_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing required VASP input(s) in {vasp_files_dir}: {', '.join(missing)}"
        )

    # The submit-include file is required - there is no built-in default.
    include_path = Path(include) if include else Path(DEFAULT_INCLUDE)
    if not include_path.is_file():
        raise FileNotFoundError(
            f"submit-include file not found: {include_path}\n"
            f"A '{DEFAULT_INCLUDE}' is required to run setup. Create one (see the "
            f"README for an example) or point at it with --include."
        )
    include_body = include_path.read_text()

    layouts = build_layouts(cores, max_cpus_per_task, allowed_cpus_per_task)
    if not layouts:
        raise ValueError("no valid (ntasks, cpus-per-task) layouts generated")

    # Everything in VASP_Files to copy into each run: inputs plus any extras,
    # including subdirectories (copied recursively) in case they are needed.
    input_items = sorted(vasp_files_dir.iterdir())

    root_dir = Path(root)
    root_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for total, ntasks, cpus_per_task in layouts:
        run_dir = root_dir / config_dirname(total, ntasks, cpus_per_task)
        run_dir.mkdir(parents=True, exist_ok=True)

        for src in input_items:
            dest = run_dir / src.name
            if src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dest)

        script = build_submit_script(
            job_name=f"{jobname_prefix}_{total}cores_{ntasks}MPI_{cpus_per_task}OMP",
            ntasks=ntasks,
            cpus_per_task=cpus_per_task,
            include_content=include_body,
            mem_line=memory_directive(total, mem, mem_per_cpu),
            time_line=time_directive(total, time_policy),
        )
        (run_dir / "submit.sl").write_text(script)
        created.append(run_dir)

    print(f"Created {len(created)} benchmark configurations under {root_dir}/")
    print(f"Core counts: {sorted({t for t, _, _ in layouts})}")

    # Summarise total requested walltime.
    if time_policy is not None:
        total_seconds = sum(
            time_to_seconds(select_time(total, *parse_time_policy(time_policy)))
            for total, _, _ in layouts
        )
        print(
            f"Walltime from --time-policy; total across all jobs: "
            f"{seconds_to_dhms(total_seconds)}"
        )
    else:
        include_sbatch, _ = split_include(include_body)
        time_value = next(
            (directive_value(ln) for ln in include_sbatch if directive_name(ln) == "--time"),
            None,
        )
        if time_value:
            total_seconds = time_to_seconds(time_value) * len(created)
            print(
                f"Walltime per job (from include --time): {time_value}  |  "
                f"total across all jobs: {seconds_to_dhms(total_seconds)}"
            )
    return created
