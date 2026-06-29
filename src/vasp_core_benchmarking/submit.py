"""Part 2: submit every generated ``submit.sl`` to SLURM."""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from .outcar import parse_loop_times

# Pause briefly after this many submissions to avoid hammering the scheduler /
# tripping QOS submission-rate limits.
PAUSE_EVERY = 10
PAUSE_SECONDS = 2

# Directory names look like "8cores_4tasks_2cpt" (see generate.config_dirname).
_CONFIG_DIR_RE = re.compile(r"(\d+)cores_(\d+)tasks_(\d+)cpt")

# Files kept when resetting a failed run before resubmitting it.
RESET_KEEP = {"INCAR", "KPOINTS", "POTCAR", "POSCAR", "submit.sl"}


def has_result(run_dir: Path) -> bool:
    """True if this run produced a usable average-electronic-step result.

    Matches the report's validity test: an OUTCAR with at least two ``LOOP``
    lines (so the first, setup-heavy step can be dropped).
    """
    outcar = run_dir / "OUTCAR"
    if not outcar.is_file():
        return False
    return len(parse_loop_times(outcar)) >= 2


def reset_run_dir(run_dir: Path) -> int:
    """Delete everything in ``run_dir`` except the inputs and submit.sl.

    Keeps only INCAR, KPOINTS, POTCAR, POSCAR and submit.sl, so the directory is
    a clean starting point for a fresh run. Returns the number of files removed.
    """
    removed = 0
    for p in run_dir.iterdir():
        if p.is_file() and p.name not in RESET_KEEP:
            p.unlink()
            removed += 1
    return removed


def _sort_key(script: Path) -> tuple:
    """Sort submit.sl scripts numerically by (total cores, ntasks, cpus-per-task).

    A plain ``sorted()`` would order names lexicographically ("128cores" before
    "16cores" before "8cores"); parsing the numbers gives ascending core counts.
    Directories that do not match the expected pattern sort last, by path.
    """
    match = _CONFIG_DIR_RE.search(script.parent.name)
    if match:
        total, ntasks, cpt = (int(g) for g in match.groups())
        return (0, total, ntasks, cpt, "")
    return (1, 0, 0, 0, str(script))


def find_submit_scripts(root: str) -> list[Path]:
    """Return every ``submit.sl`` beneath ``root``, sorted by ascending cores."""
    root_dir = Path(root)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"benchmark root not found: {root_dir}")
    return sorted(root_dir.rglob("submit.sl"), key=_sort_key)


def submit(
    root: str = "VASP_Benchmarking",
    dry_run: bool = False,
    yes: bool = False,
    retry_failed: bool = False,
) -> int:
    """Submit benchmark jobs. Returns the number successfully submitted.

    By default every config is submitted. With ``retry_failed``, only configs
    that have **not** produced a usable average-electronic-step result are
    (re)submitted, and each such directory is first reset to just its inputs and
    submit.sl.
    """
    scripts = find_submit_scripts(root)
    if not scripts:
        print(f"No submit.sl files found under {root}/")
        return 0

    if retry_failed:
        all_n = len(scripts)
        scripts = [s for s in scripts if not has_result(s.parent)]
        print(
            f"Found {all_n} configs under {root}/; "
            f"{all_n - len(scripts)} already have a result, "
            f"{len(scripts)} to retry."
        )
        if not scripts:
            print("Nothing to retry - every config has a usable result.")
            return 0
    else:
        print(f"Found {len(scripts)} submit.sl scripts under {root}/")

    if dry_run:
        for script in scripts:
            prefix = "reset + " if retry_failed else ""
            print(f"[dry-run] {prefix}sbatch (cwd={script.parent}) submit.sl")
        return 0

    if not yes:
        action = (
            f"Reset and resubmit {len(scripts)} failed/incomplete jobs"
            if retry_failed
            else f"Submit all {len(scripts)} jobs to SLURM"
        )
        reply = input(f"{action}? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 0

    submitted = 0
    for i, script in enumerate(scripts, start=1):
        if retry_failed:
            removed = reset_run_dir(script.parent)
            print(f"[{i}/{len(scripts)}] reset {script.parent} ({removed} files removed)")
        try:
            result = subprocess.run(
                ["sbatch", script.name],
                cwd=script.parent,
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"[{i}/{len(scripts)}] {script.parent}: {result.stdout.strip()}")
            submitted += 1
        except FileNotFoundError:
            print("ERROR: 'sbatch' not found - are you on a SLURM login node?")
            break
        except subprocess.CalledProcessError as exc:
            print(f"[{i}/{len(scripts)}] FAILED {script.parent}: {exc.stderr.strip()}")

        if i % PAUSE_EVERY == 0 and i < len(scripts):
            time.sleep(PAUSE_SECONDS)

    print(f"Submitted {submitted}/{len(scripts)} jobs.")
    return submitted
