"""Command-line interface for the VASP benchmarking toolkit.

Subcommands:

  vasp-core-benchmarking setup   - Part 1: create the benchmarking directory tree.
  vasp-core-benchmarking submit  - Part 2: submit all jobs to SLURM.
  vasp-core-benchmarking report  - Part 3: collect utilisation + efficiency results.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


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
    p_setup = sub.add_parser("setup", help="Part 1: create benchmarking files.")
    p_setup.add_argument(
        "--cores",
        required=True,
        help='Total core counts to benchmark, e.g. "1,2,4,8,16-128:8" '
        '(ranges take a :step stride, SLURM array increment syntax).',
    )
    p_setup.add_argument(
        "--jobname-prefix",
        default="vasp_bench",
        help="Prefix for the SLURM job name; the layout "
        "(e.g. _16cores_8MPI_2OMP) is appended.",
    )
    p_setup.add_argument("--vasp-files", default="VASP_Files", help="Directory of VASP inputs.")
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
    p_setup.add_argument("--root", default="VASP_Benchmarking", help="Output root directory.")
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
    p_submit = sub.add_parser("submit", help="Part 2: submit all jobs to SLURM.")
    p_submit.add_argument("--root", default="VASP_Benchmarking", help="Benchmark root directory.")
    p_submit.add_argument("--dry-run", action="store_true", help="List jobs without submitting.")
    p_submit.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    p_submit.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only (re)submit configs with no usable result; reset each to its "
        "inputs + submit.sl first.",
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

            setup(
                cores=args.cores,
                jobname_prefix=args.jobname_prefix,
                vasp_files=args.vasp_files,
                include=args.include,
                root=args.root,
                max_cpus_per_task=args.max_cpus_per_task,
                allowed_cpus_per_task=args.allowed_cpus_per_task,
                mem=args.mem,
                mem_per_cpu=args.mem_per_cpu,
                time_policy=args.time_policy,
            )
        elif args.command == "submit":
            from .submit import submit

            submit(
                root=args.root,
                dry_run=args.dry_run,
                yes=args.yes,
                retry_failed=args.retry_failed,
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
