# vasp-core-benchmarking

Benchmark [VASP](https://www.vasp.at/) across SLURM parallel layouts to find the
most efficient combination of MPI ranks (`--ntasks`) and OpenMP threads
(`--cpus-per-task`) for a given system.

**Terminology** — throughout the tool and its plots:

- **MPI ranks = `--ntasks`**.
- **OpenMP threads = `--cpus-per-task`** (exported as `OMP_NUM_THREADS`).
- **Total cores = ntasks × cpus-per-task**.

## Install

Install the latest version straight from GitHub with `pip`:

```bash
pip install git+https://github.com/geoffreyweal/vasp-core-benchmarking.git
```

Check it installed with:

```bash
vasp-core-benchmarking --version
```

## Workflow

The tool runs in three parts, plus status/reset helpers and an optional cleanup step.

| Subcommand | Purpose |
| --- | --- |
| `setup`  | Generate a benchmark directory per `ntasks × cpus-per-task` layout. |
| `submit` | `sbatch` the configs that need running (pending + failed). |
| `report` | Collect utilisation and electronic-step efficiency into CSV + HTML. |
| `status` | Re-scan folders and refresh `folder_index.html` with each config's run state. |
| `reset`  | Reset errored configs back to their inputs so `submit` will relaunch them. |
| `clean`  | Delete bulky VASP outputs once you're done. |

### Part 1 — `setup`: create the benchmarking files

Provide two things in the directory you run from:

1. A `VASP_Files/` directory of inputs (or point at it with `--vasp-files`).
2. A `vasp_core_benchmarking_submit_include.txt` (or pass `--include`).

#### `VASP_Files`

The following files are needed in the `VASP_Files` folder:

```text
VASP_Files/
├── INCAR      # required
├── POSCAR     # required
├── POTCAR     # required
├── KPOINTS    # required
└── ...        # any extras (ML_FF, WAVECAR, CHGCAR, …) are copied too
```

The four required files must be present. Every file in `VASP_Files/` (the inputs
plus any extras) is copied into each benchmark directory **unchanged** — the
parallel layout is varied only through the generated `submit.sl`
(`--ntasks`, `--cpus-per-task` and `OMP_NUM_THREADS`).

> `POTCAR` files are distributed under the VASP licence, so provide your own.

#### `vasp_core_benchmarking_submit_include.txt`

The `vasp_core_benchmarking_submit_include.txt` file holds what you would like to
include in the `submit.sl` files created when setting up your VASP benchmarking
environment.

An example of what this looks like is shown below:

```bash
#SBATCH --account=nesi12345
#SBATCH --partition=genoa
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=2000
#SBATCH --extra-node-info=1:*:1     # Use 1 socket, 1 thread/core (disables hyperthreading).
#SBATCH --distribution=*:block:*    # Bind tasks to one socket, fill it before the next.
#SBATCH --mem-bind=local
#SBATCH --profile=task

module purge 2> /dev/null
module load VASP/6.4.3-foss-2023a

# Print the per-rank CPU binding before launching VASP - handy for confirming
# that ntasks x cpus-per-task mapped onto cores the way you intended.
srun --job-name=print_binding_stats bash -c "echo -e \"Task #\${SLURM_PROCID} is running on node \$(hostname). \n\$(hostname) has the following NUMA configuration:\n\$(lscpu | grep -i --color=none numa)\nTask #\${SLURM_PROCID} has \$(nproc) CPUs, their core IDs are \$(taskset -c -p \$\$ | awk '{print \$NF}')\""
echo -e "\n====== Launching VASP ======\n"

# Run VASP. -K1 makes srun kill the whole step if any task fails.
srun -K1 vasp_std
```

#### Running `vasp-core-benchmarking setup`

Once you have set up the `VASP_Files` folder and `vasp_core_benchmarking_submit_include.txt`
file, describe the set of parallel layouts you want to benchmark in a plain-text
**`options.txt`** and run `setup`
to create your benchmarking environments.

Write one `key = value` per line, using the option names described below. Blank
lines and lines starting with `#` are ignored, `-` and `_` are interchangeable in
keys, and quotes around a value are optional. A typical `options.txt` looks like:

```text
# options.txt — VASP core-benchmarking setup
cores        = 1,2,4,8,16-128:8
mem          = 32G
mem-per-cpu  = 2G
time-policy  = 30:00,15:00@32
```

If an `options.txt` is present in the directory you run from, `setup` picks it up
automatically; point at a differently named file with `--options path/to/file`:

```bash
vasp-core-benchmarking setup                    # auto-loads ./options.txt
vasp-core-benchmarking setup --options my.txt   # use a differently named file
```

This writes `VASP_Benchmarking/<total>cores_<ntasks>tasks_<cpt>cpt/`, each holding
copies of the inputs and a `submit.sl`. An unknown key, a missing value or a
duplicated key is reported with its line number, so typos are caught before any
files are written.

##### `cores` (required)

A comma-separated list of single values and inclusive `start-end` ranges. A range
may take a `:step` stride (SLURM array increment syntax, where `:` means
"increment by"), so `cores = 1,2,4,8,16-128:8` expands to `1, 2, 4, 8, 16, 24, 32, …, 128`.

For each total core count, `setup` generates **every** `ntasks × cpus-per-task`
factorisation — e.g. `16` → `16×1, 8×2, 4×4, 2×8, 1×16`.

##### `max-cpus-per-task` / `allowed-cpus-per-task` (optional)

Prune the layout grid. `max-cpus-per-task` drops any layout whose OpenMP thread
count exceeds the given value (e.g. a socket size); `allowed-cpus-per-task`
restricts OpenMP threads to a fixed set, e.g. `allowed-cpus-per-task = 1,2,4,8`.

##### `jobname-prefix` (optional)

Sets the SLURM job-name prefix (default `vasp_bench`). Each job is named
`<prefix>_<total>cores_<ntasks>MPI_<cpt>OMP` (e.g. `vasp_bench_16cores_8MPI_2OMP`),
so you can read the layout straight off the queue.

##### `mem` / `mem-per-cpu` (optional)

Make memory scale with core count instead of using a fixed value from the include:

- `mem` is a flat total-memory floor (e.g. `mem = 32G`).
- `mem-per-cpu` scales with cores (e.g. `mem-per-cpu = 2G`).

With both set, each job gets whichever yields more — the floor at low core counts,
switching to per-cpu once `total_cores × mem_per_cpu` exceeds it. Giving only one
applies it flat to every job.

##### `time-policy` (optional)

Make walltime depend on core count, in the form `T1,T2,...@C1,C2,...` (N+1
walltimes for N ascending thresholds): `total_cores ≤ C1` gets `T1`, `≤ C2` gets
`T2`, …, anything larger gets the last. For example, `time-policy = 30:00,15:00,10:00@16,64`
gives 30:00 up to 16 cores, 15:00 up to 64, and 10:00 beyond.

##### Other options

`vasp-files` (default `VASP_Files`) and `include` (default
`vasp_core_benchmarking_submit_include.txt`) point at the inputs and the include file;
`root` (default `VASP_Benchmarking`) sets the output directory.

> When `mem`/`mem-per-cpu` or `time-policy` is set, the tool writes that directive
> itself and **overrides** the matching memory/walltime directive in the include.
> With a time set, `setup` also reports the total requested walltime.

### Part 2 — `submit`: send the jobs to SLURM

```bash
vasp-core-benchmarking submit            # prompts for confirmation
vasp-core-benchmarking submit --dry-run  # list what would be submitted
vasp-core-benchmarking submit --yes      # no prompt
```

Every config is classified first (the same run-state rules `status` uses) and only
the ones that need running are submitted, in ascending core order, pausing briefly
every 10 submissions to avoid scheduler rate limits:

- **pending** (never launched — only inputs present) → submitted;
- **failed** (launched, but produced no usable timing data — see below) → **reset to
  its inputs first**, then submitted;
- **run**, **running** and **errored** configs are skipped.

A config counts as having *run* once its `OUTCAR` logged **more than `--skip-steps`
electronic (`LOOP:`) steps** (default 5) — enough that at least one usable step
remains after the warm-up steps are dropped. This is the same usable-result test the
report applies, so a job that hit the walltime but still logged enough steps is a
successful benchmark and is left alone. Pass `--skip-steps` to match the value you
report with.

`submit` prints the exact plan (which folders, and why) before the confirmation
prompt, so you always see what will launch.

Resetting a failed config restores the directory to just `INCAR`, `KPOINTS`,
`POTCAR`, `POSCAR` and `submit.sl` (deleting the old OUTCAR, slurm logs and other
leftovers) before resubmitting.

> The reset keeps only those five files. If your system needs extra inputs (e.g.
> `ML_FF`), re-run `setup` to repopulate them before retrying, or copy them back in.

Errored configs — those that produced no usable result **and** ended with an
identifiable failure (a VASP abort message in the `OUTCAR`, an abnormal SLURM
terminal state such as `TIMEOUT` or `OUT_OF_MEMORY`, or an error line in
`slurm-<id>.out`) — are **never** resubmitted automatically, because they usually
need attention first (more memory, a longer walltime, a fixed input). Fix the cause,
then clear them with `reset` (below).

### Part 3 — `report`: measure utilisation and efficiency

```bash
vasp-core-benchmarking report                 # reads VASP_Benchmarking/
vasp-core-benchmarking report --no-sacct      # skip SLURM accounting queries
vasp-core-benchmarking report --skip-steps 10 # drop the first 10 warm-up steps (default 5)
```

For each completed run this collects:

- **Electronic-step time** — mean & std-dev of the per-step `LOOP: … real time`
  from `OUTCAR`. The first few steps carry setup/warm-up overhead and are dropped;
  the count is set by `--skip-steps` (default 5). A run is skipped entirely if it
  has no more steps than that.
- **Speedup** — t<sub>1</sub> / t<sub>N</sub>: the **1 MPI × 1 OMP** step time over
  this run's step time (>1 means faster than single-core). Only computed when a
  1 MPI × 1 OMP run produced a result; otherwise the cell is left blank. Use
  `--baseline` to choose a different t<sub>1</sub> run (below).
- **SLURM utilisation** — elapsed time, CPU utilisation
  `TotalCPU / (Elapsed × cores)`, and **Max Memory utilisation** via `sacct --json`
  (the `tres.requested.total` mem entry). These are left blank with `--no-sacct`.

Outputs go to `report/` (change with `--out`): `results.csv` (all metrics),
`skipped.txt` (unusable runs), and a self-contained `vasp_benchmark_results.html`.

The HTML shows four metrics — **CPU utilisation**, **Max Memory utilisation**,
**Time / electronic step** and **Speedup (t<sub>1</sub> / t<sub>N</sub>)** — on a
2×2 grid, with one radio toggle and one shared slider driving every cell:

- **3D** — MPI ranks vs OpenMP threads vs the metric.
- **2D** — MPI ranks vs the metric at the slider's OpenMP-thread count, with fixed
  axes for like-for-like comparison. The time/step cell carries std-dev error bars,
  and the speedup cell adds a dotted 1:1 perfect-scaling line.

#### Choosing the speedup baseline (`--baseline`)

By default t<sub>1</sub> is the 1 MPI × 1 OMP run inside `--root`. Use `--baseline`
to anchor speedup elsewhere — e.g. a **non-hyperthreaded** single-core run when the
rest of the benchmark runs are hyperthreaded:

```bash
vasp-core-benchmarking report --baseline 1cores_1tasks_1cpt              # config name in --root
vasp-core-benchmarking report --baseline /path/to/no_hyperthreading/run # external directory
```

Speedup becomes `t_baseline / t_N`, and the ideal line scales by the baseline's
core count (`total_cores / baseline_cores`). An unknown config name, or a path
whose OUTCAR has no usable result, is reported as a clear error.

### `status`: see the state of every config

```bash
vasp-core-benchmarking status               # re-scan + refresh folder_index.html
vasp-core-benchmarking status --no-sacct    # classify from local files only
vasp-core-benchmarking status --skip-steps 10  # match the report's warm-up count
```

Re-scans every layout folder under `--root`, classifies each run, prints a summary,
and (re)writes a self-contained `folder_index.html` in the root — a snapshot table
of every config's layout (total cores, MPI ranks, OpenMP threads), its electronic-step
count, and current state, with a status filter. Open it in a browser and re-run
`status` (or `report`) to refresh it.

Each config is classified as one of:

| State | Meaning |
| --- | --- |
| **run** | Finished (no longer active) with a usable result — **more than `--skip-steps` electronic (`LOOP:`) steps** (default 5). Counts even if SLURM later killed the job (e.g. at the walltime): the timing data is usable. |
| **running** | Launched, not yet a usable result, and its SLURM job is still active (via `sacct`; without it, its output files were written to within the last 30 minutes). |
| **error** | Finished with no usable result **and** an identifiable failure — a VASP abort in the `OUTCAR`, an abnormal SLURM state (`TIMEOUT`, `OUT_OF_MEMORY`, …), or an error line in `slurm-<id>.out`. |
| **failed** | Finished with no usable result and not still running, but no specific error could be identified (e.g. killed without a message, or it ran no more than `--skip-steps` steps). |
| **pending** | No sign the run has been launched yet (only input files). |

Because "run" is defined by the step count, `--skip-steps` sets the same threshold
`report` uses; pass the value you intend to report with so the two agree.

With `--no-sacct` the scheduler is not queried: "running" is inferred from recent
output-file activity instead, so the classification still works from the folder
contents alone.

### `reset`: clear configs with no usable result

```bash
vasp-core-benchmarking reset --dry-run       # list what would be reset, and why
vasp-core-benchmarking reset                 # prompts for confirmation
vasp-core-benchmarking reset --yes           # no prompt
vasp-core-benchmarking reset --skip-steps 10 # match the report's warm-up count
```

Resets every finished config that produced **no usable result** — the **error** and
**failed** states, i.e. those that logged no more than `--skip-steps` electronic
steps — back to its inputs (`INCAR`, `KPOINTS`, `POTCAR`, `POSCAR`, `submit.sl`),
deleting the failed run's artefacts and returning it to **pending** so the next
`submit` relaunches it.

Runs that already logged more than `--skip-steps` steps are **never** reset, even if
SLURM killed them afterwards — their timing data is usable, so it is kept. Running and
pending configs are left untouched too.

Fix the cause of any error first (e.g. raise the memory or walltime in your include
and re-run `setup` for any brand-new layouts), then `reset` and `submit`.

### Optional — `clean`: reclaim disk space

Benchmark runs leave large outputs behind (WAVECAR, CHGCAR, vaspout.h5, vasprun.xml,
ML_FF, …). After `report`, delete everything non-essential:

```bash
vasp-core-benchmarking clean --dry-run   # list what would go + total size
vasp-core-benchmarking clean             # prompts for confirmation
vasp-core-benchmarking clean --yes       # no prompt
```

In every directory under `--root` this keeps `INCAR`, `KPOINTS`, `POTCAR`,
`POSCAR`, `OUTCAR`, `OSZICAR`, scripts (`*.sh`, `*.sl`) and slurm logs
(`slurm-*.out` / `slurm-*.err`), deletes the rest, and reports space freed.
