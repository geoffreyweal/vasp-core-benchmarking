"""Command-line interface for the VASP benchmarking toolkit.

Subcommands:

  vasp-core-benchmarking setup   - Part 1: create the benchmarking directory tree.
  vasp-core-benchmarking submit  - Part 2: submit the configs that need running.
  vasp-core-benchmarking report  - Part 3: collect utilisation + efficiency results.
  vasp-core-benchmarking status  - re-scan folders + refresh folder_index.html.
  vasp-core-benchmarking reset   - reset errored configs back to their inputs.
  vasp-core-benchmarking clean   - delete bulky outputs, keep inputs + results.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .options_file import DEFAULT_OPTIONS_FILE, load_setup_options


def _parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


# setup options that can also be supplied via the options file. These are the
# argparse dest names; keep in sync with the setup subparser and
# options_file.KNOWN_OPTIONS. A dest listed in _SETUP_OPTION_CONVERTERS has its
# raw file value passed through that converter (command-line values are already
# typed by argparse); the rest are used as plain strings.
_SETUP_OPTION_DESTS = (
    "cores",
    "jobname_prefix",
    "vasp_files",
    "include",
    "mem",
    "mem_per_cpu",
    "time_policy",
    "root",
    "max_cpus_per_task",
    "allowed_cpus_per_task",
)
_SETUP_OPTION_CONVERTERS = {
    "max_cpus_per_task": int,
    "allowed_cpus_per_task": _parse_int_list,
}


def _merge_setup_options(args, file_opts, source):
    """Merge command-line args over options-file values into ``setup()`` kwargs.

    Precedence is command-line flag > options file > ``setup()``'s own default,
    the last achieved by omitting any option that was given in neither place.
    """
    kwargs = {}
    for dest in _SETUP_OPTION_DESTS:
        cli_val = getattr(args, dest)
        if cli_val is not None:
            kwargs[dest] = cli_val
        elif dest in file_opts:
            raw = file_opts[dest]
            converter = _SETUP_OPTION_CONVERTERS.get(dest)
            if converter is None:
                kwargs[dest] = raw
            else:
                try:
                    kwargs[dest] = converter(raw)
                except ValueError as exc:
                    flag = dest.replace("_", "-")
                    raise ValueError(
                        f"{source}: invalid value for '{flag}': {raw!r} ({exc})"
                    ) from None
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-core-benchmarking",
        description=(
            "Benchmark VASP across SLURM ntasks x cpus-per-task layouts. "
            "Here MPI ranks = --ntasks and OpenMP threads = --cpus-per-task "
            "(OMP_NUM_THREADS); total cores = ntasks x cpus-per-task."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- setup -----------------------------------------------------------
    # Every setup option below defaults to None (rather than its real default) so
    # that _merge_setup_options can tell "given on the command line" from "left to
    # the options file / built-in default". The real defaults live in setup()'s
    # signature and are noted in the help text here.
    p_setup = sub.add_parser("setup", help="Part 1: create benchmarking files.")
    p_setup.add_argument(
        "--options",
        help="Read setup options from this key=value file. If omitted, "
        f"'{DEFAULT_OPTIONS_FILE}' in the working directory is used automatically "
        "when present. Command-line flags override values from the file.",
    )
    p_setup.add_argument(
        "--cores",
        help='Total core counts to benchmark, e.g. "1,2,4,8,16-128:8" '
        "(ranges take a :step stride, SLURM array increment syntax). Required, "
        "unless given as 'cores' in the options file.",
    )
    p_setup.add_argument(
        "--jobname-prefix",
        help="Prefix for the SLURM job name; the layout "
        "(e.g. _16cores_8MPI_2OMP) is appended (default: vasp_bench).",
    )
    p_setup.add_argument(
        "--vasp-files", help="Directory of VASP inputs (default: VASP_Files)."
    )
    p_setup.add_argument(
        "--include",
        help="Submit-include file (default: vasp_core_benchmarking_submit_include.txt). "
        "Holds all #SBATCH job settings plus the run commands.",
    )
    p_setup.add_argument(
        "--mem",
        help="Flat total-memory floor per job (e.g. 8G, 4000M). Used at low core "
        "counts. If set, overrides any memory line in the include.",
    )
    p_setup.add_argument(
        "--mem-per-cpu",
        help="Per-CPU memory (e.g. 2G, 2000). Scales with core count. With --mem, "
        "each job uses whichever gives more memory.",
    )
    p_setup.add_argument(
        "--time-policy",
        help="Walltime that varies with core count: 'T1,T2,...@C1,C2,...' "
        "(N+1 times, N thresholds), e.g. '30:00,15:00,10:00@16,64' = <=16 cores "
        "30:00, <=64 cores 15:00, else 10:00. Overrides --time in the include.",
    )
    p_setup.add_argument(
        "--root", help="Output root directory (default: VASP_Benchmarking)."
    )
    p_setup.add_argument(
        "--max-cpus-per-task",
        type=int,
        help="Drop layouts whose cpus-per-task exceeds this (e.g. socket size).",
    )
    p_setup.add_argument(
        "--allowed-cpus-per-task",
        type=_parse_int_list,
        help='Restrict cpus-per-task to these values, e.g. "1,2,4,8".',
    )

    # ---- submit ----------------------------------------------------------
    p_submit = sub.add_parser(
        "submit",
        help="Part 2: submit the configs that need running (pending + failed; "
        "completed/running/errored are skipped).",
    )
    p_submit.add_argument("--root", default="VASP_Benchmarking", help="Benchmark root directory.")
    p_submit.add_argument("--dry-run", action="store_true", help="List jobs without submitting.")
    p_submit.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    p_submit.add_argument(
        "--skip-steps",
        type=int,
        default=5,
        help="Warm-up electronic steps a run must exceed to count as done "
        "(default 5); a config with no more than this is treated as needing "
        "a (re)run. Match report's --skip-steps.",
    )

    # ---- report ----------------------------------------------------------
    p_report = sub.add_parser("report", help="Part 3: collect results into CSV + HTML.")
    p_report.add_argument("--root", default="VASP_Benchmarking", help="Benchmark root directory.")
    p_report.add_argument("--out", default="report", help="Report output directory.")
    p_report.add_argument("--no-sacct", action="store_true", help="Skip sacct utilisation queries.")
    p_report.add_argument(
        "--skip-steps",
        type=int,
        default=5,
        help="Number of leading (warm-up) electronic steps to drop from each "
        "run's timing average (default 5). Runs with no more steps than this "
        "are skipped.",
    )
    p_report.add_argument(
        "--baseline",
        help="Run to use as the t_1 speedup baseline: a config name "
        "(e.g. 1cores_1tasks_1cpt) or a path to a run directory (e.g. a "
        "non-hyperthreaded single-core run). Default: the 1 MPI x 1 OMP run.",
    )

    # ---- status ----------------------------------------------------------
    p_status = sub.add_parser(
        "status",
        help="Re-scan folders and refresh folder_index.html (run/running/error/"
        "failed/pending), printing a summary.",
    )
    p_status.add_argument("--root", default="VASP_Benchmarking", help="Benchmark root directory.")
    p_status.add_argument(
        "--no-sacct",
        action="store_true",
        help="Skip sacct queries; 'running' is then inferred from recent "
        "output-file activity instead of the scheduler.",
    )
    p_status.add_argument(
        "--skip-steps",
        type=int,
        default=5,
        help="A config counts as 'run' only once it has more than this many "
        "electronic steps (default 5). Match report's --skip-steps.",
    )

    # ---- reset -----------------------------------------------------------
    p_reset = sub.add_parser(
        "reset",
        help="Reset errored configs back to their inputs (they become pending, "
        "so the next 'submit' relaunches them). Fix the error's cause first.",
    )
    p_reset.add_argument("--root", default="VASP_Benchmarking", help="Benchmark root directory.")
    p_reset.add_argument(
        "--dry-run", action="store_true", help="List configs to reset without resetting."
    )
    p_reset.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    p_reset.add_argument(
        "--skip-steps",
        type=int,
        default=5,
        help="Reset configs with no more than this many electronic steps "
        "(default 5); runs that exceeded it are kept. Match report's --skip-steps.",
    )

    # ---- clean -----------------------------------------------------------
    p_clean = sub.add_parser(
        "clean",
        help="Delete unnecessary files, keeping inputs, OUTCAR/OSZICAR and slurm logs.",
    )
    p_clean.add_argument("--root", default="VASP_Benchmarking", help="Benchmark root directory.")
    p_clean.add_argument("--dry-run", action="store_true", help="List files without deleting.")
    p_clean.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "setup":
            from .generate import setup

            # Pull any options.txt (auto-loaded, or the file named with --options)
            # and merge, with command-line flags taking precedence.
            file_opts, source = load_setup_options(args.options)
            kwargs = _merge_setup_options(args, file_opts, source)
            if source is not None and file_opts:
                loaded = ", ".join(k.replace("_", "-") for k in sorted(file_opts))
                print(f"Loaded setup options from {source}: {loaded}")

            if "cores" not in kwargs:
                raise ValueError(
                    "no core counts given: pass --cores, or add a 'cores' line to "
                    f"'{DEFAULT_OPTIONS_FILE}' (auto-loaded) or the file named with "
                    "--options."
                )
            setup(**kwargs)
        elif args.command == "submit":
            from .submit import submit

            submit(
                root=args.root,
                dry_run=args.dry_run,
                yes=args.yes,
                skip_steps=args.skip_steps,
            )
        elif args.command == "reset":
            from .submit import reset

            reset(
                root=args.root,
                dry_run=args.dry_run,
                yes=args.yes,
                skip_steps=args.skip_steps,
            )
        elif args.command == "report":
            from .report import report

            report(
                root=args.root,
                out=args.out,
                no_sacct=args.no_sacct,
                baseline=args.baseline,
                skip_steps=args.skip_steps,
            )
        elif args.command == "status":
            from .status import STATUS_TEXT, refresh_index

            out_path, entries = refresh_index(
                args.root, use_sacct=not args.no_sacct, skip_steps=args.skip_steps
            )
            counts: dict[str, int] = {}
            for e in entries:
                counts[e["status"]] = counts.get(e["status"], 0) + 1
            order = ["done", "running", "error", "failed", "pending"]
            summary = ", ".join(
                f"{counts.get(k, 0)} {STATUS_TEXT[k].split(' ', 1)[-1]}" for k in order
            )
            print(f"Rewrote {out_path} ({len(entries)} folder(s): {summary}).")
            print("Refresh the page in your browser to see the updated statuses.")
        elif args.command == "clean":
            from .clean import clean

            clean(root=args.root, dry_run=args.dry_run, yes=args.yes)
        else:  # pragma: no cover - argparse enforces a valid command
            return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
