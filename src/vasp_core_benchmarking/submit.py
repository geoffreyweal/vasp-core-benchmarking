"""Part 2: submit the generated jobs to SLURM (and reset unusable ones).

``submit`` classifies every config first (same rules as the folder status page)
and only submits the ones that need running - **pending** configs, and
**failed** ones (which are reset to their inputs first). Runs that already
produced a usable result (more than ``skip_steps`` electronic steps) count as
**done** even if SLURM later killed them, and are left alone; **running** and
**errored** configs are also skipped. Errors usually need attention (more
memory, a longer time limit, a fixed input) before rerunning, so clear them
explicitly with ``reset``.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from . import status as status_mod
from .status import DEFAULT_SKIP_STEPS

# Pause briefly after this many submissions to avoid hammering the scheduler /
# tripping QOS submission-rate limits.
PAUSE_EVERY = 10
PAUSE_SECONDS = 2

# Files kept when resetting a run before resubmitting it. These are the
# per-config inputs copied in by setup, plus the generated submit.sl.
RESET_KEEP = {"INCAR", "KPOINTS", "POTCAR", "POSCAR", "submit.sl"}


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


def find_submit_scripts(root: str) -> list[Path]:
    """Return every ``submit.sl`` beneath ``root``, sorted by ascending cores."""
    root_dir = Path(root)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"benchmark root not found: {root_dir}")
    return sorted(
        root_dir.rglob("submit.sl"),
        key=lambda p: status_mod.config_sort_key(p.parent),
    )


def _print_plan(to_submit: list[tuple[Path, bool]]) -> None:
    """Show exactly which folders would be submitted, and why."""
    print(f"Will submit {len(to_submit)} job(s):")
    for script, needs_reset in to_submit:
        why = "failed - will reset first" if needs_reset else "pending"
        print(f"  {script.parent.name}  ({why})")


def submit(
    root: str = "VASP_Benchmarking",
    dry_run: bool = False,
    yes: bool = False,
    skip_steps: int = DEFAULT_SKIP_STEPS,
) -> int:
    """Submit the configs that need running. Returns the number submitted.

    Every config is classified first (same rules as the folder status page) and
    only **pending** and **failed** ones are submitted - failed directories are
    reset to their inputs first. Runs that already logged more than
    ``skip_steps`` electronic steps count as done and are skipped, as are
    still-running and errored configs; clear errors explicitly with
    :func:`reset` once you have addressed their cause. The exact list of folders
    to be submitted is shown before the confirmation prompt.

    Each script is submitted exactly as it sits in its folder.
    """
    scripts = find_submit_scripts(root)
    if not scripts:
        print(f"No submit.sl files found under {root}/")
        return 0

    to_submit: list[tuple[Path, bool]] = []  # (script, needs_reset)
    counts = {"done": 0, "running": 0, "error": 0}
    for s in scripts:
        st, _detail = status_mod.run_status(s.parent, skip_steps=skip_steps)
        if st in ("pending", "failed"):
            to_submit.append((s, st == "failed"))
        else:
            counts[st] += 1
    print(
        f"Found {len(scripts)} configs under {root}/: "
        f"{counts['done']} run, {counts['running']} running, "
        f"{counts['error']} error (all skipped); {len(to_submit)} eligible."
    )
    if counts["error"]:
        print(
            "Errored configs are never resubmitted as-is - fix the cause, then "
            "run 'vasp-core-benchmarking reset' to make them pending."
        )

    if not to_submit:
        print("Nothing to submit.")
        return 0

    # Always show exactly what would be launched before doing anything.
    _print_plan(to_submit)

    if dry_run:
        print("[dry-run] nothing was submitted.")
        return 0

    if not yes:
        reply = (
            input(f"Submit these {len(to_submit)} job(s) to SLURM? [y/N] ")
            .strip()
            .lower()
        )
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 0

    submitted = 0
    for i, (script, needs_reset) in enumerate(to_submit, start=1):
        if needs_reset:
            removed = reset_run_dir(script.parent)
            print(f"[{i}/{len(to_submit)}] reset {script.parent} ({removed} files removed)")
        try:
            result = subprocess.run(
                ["sbatch", script.name],
                cwd=script.parent,
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"[{i}/{len(to_submit)}] {script.parent}: {result.stdout.strip()}")
            submitted += 1
        except FileNotFoundError:
            print("ERROR: 'sbatch' not found - are you on a SLURM login node?")
            break
        except subprocess.CalledProcessError as exc:
            print(f"[{i}/{len(to_submit)}] FAILED {script.parent}: {exc.stderr.strip()}")

        if i % PAUSE_EVERY == 0 and i < len(to_submit):
            time.sleep(PAUSE_SECONDS)

    print(f"Submitted {submitted}/{len(to_submit)} jobs.")
    return submitted


def reset(
    root: str = "VASP_Benchmarking",
    dry_run: bool = False,
    yes: bool = False,
    skip_steps: int = DEFAULT_SKIP_STEPS,
) -> int:
    """Reset every finished config **without a usable result**. Returns the count.

    A config is reset when it has been launched, is no longer running, and did
    **not** log more than ``skip_steps`` electronic steps - i.e. it produced no
    usable timing data (the ``error`` and ``failed`` states). Resetting deletes
    everything except the inputs (:data:`RESET_KEEP`), returning the config to
    **pending** so the next ``submit`` picks it up.

    Runs that already logged more than ``skip_steps`` steps are **never** reset,
    even if SLURM killed them afterwards (e.g. at the walltime) - their timing
    data is usable, so it is kept. Running and pending configs are untouched.

    Fix the cause of any error first (e.g. more memory or a longer time limit);
    ``reset`` only clears the unusable run's artefacts so the folder starts clean.
    """
    root_dir = Path(root)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"benchmark root not found: {root_dir}")

    targets: list[tuple[Path, str]] = []
    for d in status_mod.config_dirs(root_dir):
        st, detail = status_mod.run_status(d, skip_steps=skip_steps)
        if st in ("error", "failed"):
            n = status_mod.n_electronic_steps(d)
            reason = f"{n} electronic step(s), need > {skip_steps}"
            if detail:
                reason += f"; {detail}"
            targets.append((d, reason))

    if not targets:
        print(
            f"No configs under {root_dir}/ lack a usable result "
            f"(> {skip_steps} electronic steps) - nothing to reset."
        )
        return 0

    print(f"{len(targets)} config(s) under {root_dir}/ with no usable result:")
    for d, reason in targets:
        print(f"  {d.name}: {reason}")

    if dry_run:
        print("[dry-run] no files were removed.")
        return 0

    if not yes:
        reply = (
            input(f"Reset {len(targets)} config(s) to their inputs? [y/N] ")
            .strip()
            .lower()
        )
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 0

    for d, _reason in targets:
        removed = reset_run_dir(d)
        print(f"  reset {d} ({removed} files removed)")

    # Refresh the status page so these configs show as pending again.
    try:
        index_path, _entries = status_mod.refresh_index(root_dir, skip_steps=skip_steps)
        print(f"Refreshed folder status page -> {index_path}")
    except FileNotFoundError:
        pass  # unusual; the reset itself is done

    print(f"Reset {len(targets)} config(s); run 'submit' to relaunch them.")
    return len(targets)
