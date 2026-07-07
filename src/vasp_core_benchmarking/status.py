"""Classify each benchmark config folder and refresh the folder status page.

Config directories are named by their parallel layout (``8cores_4tasks_2cpt``,
see :func:`generate.config_dirname`). This module reads each folder's output
files to decide whether the run completed, is still running, errored, failed or
is still pending, and writes a self-contained ``folder_index.html`` snapshot
table so the state of the whole sweep can be seen at a glance.

The same classifier drives ``submit`` (which only launches pending/failed
configs) and ``reset`` (which clears errored configs back to pending).
"""

from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime
from pathlib import Path

from . import sacct
from .outcar import error_signature, parse_loop_times

INDEX_FILENAME = "folder_index.html"

# Warm-up electronic steps dropped from each run's timing average (report's
# --skip-steps default). A run is only a usable benchmark result once it has run
# MORE than this many electronic steps, so at least one remains after the drop.
DEFAULT_SKIP_STEPS = 5

# Directory names look like "8cores_4tasks_2cpt" (see generate.config_dirname).
_CONFIG_DIR_RE = re.compile(r"(\d+)cores_(\d+)tasks_(\d+)cpt")

# Per-status display text and CSS class, shared by the table and the summary.
STATUS_TEXT = {
    "done": "✓ run",
    "running": "⏳ running",
    "error": "✗ error",
    "failed": "✗ failed",
    "pending": "— pending",
}

# Error signatures SLURM / the MPI runtime print into slurm-<id>.out.
_SLURM_ERROR_PATTERNS = (
    "DUE TO TIME LIMIT",
    "CANCELLED AT",
    "Out Of Memory",
    "oom-kill",
    "oom_kill",
    "srun: error",
    "slurmstepd: error",
    "Segmentation fault",
    "BAD TERMINATION",
    "Traceback (most recent call last)",
    "forrtl: severe",
)


def parse_layout(name: str) -> tuple[int, int, int] | None:
    """Return ``(total_cores, ntasks, cpus_per_task)`` parsed from a dir name."""
    match = _CONFIG_DIR_RE.search(name)
    if not match:
        return None
    return tuple(int(g) for g in match.groups())  # type: ignore[return-value]


def config_sort_key(path: Path) -> tuple:
    """Sort config directories numerically by (total cores, ntasks, cpus-per-task).

    A plain ``sorted()`` would order names lexicographically ("128cores" before
    "16cores" before "8cores"); parsing the numbers gives ascending core counts.
    Directories that do not match the expected pattern sort last, by path.
    """
    layout = parse_layout(path.name)
    if layout is not None:
        total, ntasks, cpt = layout
        return (0, total, ntasks, cpt, "")
    return (1, 0, 0, 0, str(path))


def config_dirs(root_dir: Path) -> list[Path]:
    """Layout config directories under ``root`` (immediate subdirs with an INCAR)."""
    dirs = [
        p
        for p in root_dir.iterdir()
        if p.is_dir() and (p / "INCAR").is_file() and parse_layout(p.name) is not None
    ]
    return sorted(dirs, key=config_sort_key)


def _error_evidence(run_dir: Path, use_sacct: bool) -> str | None:
    """Why this run ended abnormally, or None if no error can be identified.

    Checked in order: a VASP abort message near the end of the OUTCAR; the
    SLURM job's terminal state via ``sacct`` (TIMEOUT, OUT_OF_MEMORY, ...);
    error signatures in the newest ``slurm-<id>.out`` (a plain file in the run
    directory - no scheduler query needed).
    """
    sig = error_signature(run_dir / "OUTCAR")
    if sig:
        return sig
    if use_sacct:
        state = sacct.error_state(run_dir)
        if state:
            return state
    outs = sorted(
        (p for p in run_dir.glob("slurm-*.out")),
        key=lambda p: int(sacct._SLURM_OUT_RE.search(p.name).group(1)),
    )
    if outs:
        text = outs[-1].read_text(errors="replace")
        for pat in _SLURM_ERROR_PATTERNS:
            if pat in text:
                return pat
    return None


# A launched, incomplete run whose OUTCAR/OSZICAR was written to this recently
# is treated as still running when sacct can't say. VASP writes at least once
# per electronic step, so half an hour of silence means the job is dead (or a
# single step takes >30 min, in which case it briefly shows as failed).
ACTIVITY_WINDOW_S = 30 * 60


def _recently_active(run_dir: Path) -> bool:
    """Whether the run's output files were modified within ACTIVITY_WINDOW_S."""
    mtimes = [
        p.stat().st_mtime
        for p in (run_dir / "OUTCAR", run_dir / "OSZICAR")
        if p.is_file()
    ]
    return bool(mtimes) and (time.time() - max(mtimes)) < ACTIVITY_WINDOW_S


def n_electronic_steps(run_dir: Path) -> int:
    """Number of electronic (``LOOP:``) steps timed in this run's OUTCAR."""
    outcar = run_dir / "OUTCAR"
    return len(parse_loop_times(outcar)) if outcar.is_file() else 0


def has_usable_result(run_dir: Path, skip_steps: int = DEFAULT_SKIP_STEPS) -> bool:
    """Whether this run produced a usable average-electronic-step result.

    Matches the report's validity test: MORE than ``skip_steps`` electronic
    (``LOOP:``) steps, so at least one remains after the warm-up steps are
    dropped. This is what "the benchmark ran" means for this tool - a job that
    logged enough steps has usable timing data even if SLURM later killed it
    (e.g. at the walltime), so its result must not be thrown away.
    """
    return n_electronic_steps(run_dir) > skip_steps


def run_status(
    run_dir: Path, use_sacct: bool = True, skip_steps: int = DEFAULT_SKIP_STEPS
) -> tuple[str, str | None]:
    """Classify a config folder; returns ``(status, detail)``.

    "Ran" is defined by the timing data, not by SLURM's exit: a run counts as
    **done** once it has logged more than ``skip_steps`` electronic steps (so at
    least one usable step remains after the warm-up steps are dropped).

    * **done** - finished (no longer active) with a usable result, i.e. more
      than ``skip_steps`` electronic steps. A job that hit the walltime but
      still logged enough steps counts as done - its timing data is usable.
    * **running** - launched, not yet done, and its SLURM job is still active
      according to ``sacct``; without sacct, "output files written to within
      the last :data:`ACTIVITY_WINDOW_S`" is used instead, so this works from
      the folder contents alone;
    * **error** - finished without a usable result and with an identifiable
      error; ``detail`` says what was found (a VASP abort message in the OUTCAR,
      an abnormal SLURM terminal state such as ``TIMEOUT``, or an error line in
      ``slurm-<id>.out``);
    * **failed** - finished without a usable result and not still running, but
      no specific error could be identified (e.g. killed without a message, or
      it ran no more than ``skip_steps`` steps);
    * **pending** - no sign the run has been launched yet (only input files).

    ``detail`` is None for every status except ``error``. Everything except the
    sacct refinement comes from files in the run directory, so the
    classification works with ``use_sacct=False`` too.
    """
    started = (
        (run_dir / "OUTCAR").is_file()
        or (run_dir / "OSZICAR").is_file()
        or sacct.find_job_id(run_dir) is not None
    )
    if not started:
        return "pending", None
    # Is it still going? Prefer the scheduler's answer; fall back to recent
    # write activity so the classification never depends on sacct being there.
    active = sacct.is_running(run_dir) if use_sacct else None
    if active is None:
        active = _recently_active(run_dir)
    # A finished run that logged enough electronic steps is a successful
    # benchmark run, regardless of how SLURM recorded the job's exit.
    if not active and has_usable_result(run_dir, skip_steps):
        return "done", None
    if active:
        return "running", None
    detail = _error_evidence(run_dir, use_sacct)
    if detail is not None:
        return "error", detail
    return "failed", None


def scan_configs(
    root_dir: Path, use_sacct: bool = True, skip_steps: int = DEFAULT_SKIP_STEPS
) -> list[dict]:
    """Read every config folder into ``{folder, total_cores, ntasks, ...}``."""
    entries: list[dict] = []
    for d in config_dirs(root_dir):
        total, ntasks, cpt = parse_layout(d.name)  # type: ignore[misc]
        status, detail = run_status(d, use_sacct=use_sacct, skip_steps=skip_steps)
        entries.append(
            {
                "folder": d.name,
                "total_cores": total,
                "ntasks": ntasks,
                "cpus_per_task": cpt,
                "n_steps": n_electronic_steps(d),
                "status": status,
                "detail": detail,
                "has_result": status == "done",
            }
        )
    return entries


def build_index_html(
    entries: list[dict],
    generated_at: str | None = None,
    skip_steps: int = DEFAULT_SKIP_STEPS,
) -> str:
    """Build the self-contained folder status page.

    A summary line, a status filter, and a reference table of every config
    (layout, electronic-step count + run state). ``generated_at`` stamps the
    page with when it was scanned; since the page is a static snapshot (a
    browser opening a local file cannot re-scan the folders), this tells the
    reader how fresh it is. ``skip_steps`` is noted so the "> N steps = run"
    rule behind the statuses is visible.
    """

    def esc(x: str) -> str:
        return html.escape(str(x), quote=True)

    counts: dict[str, int] = {}
    for e in entries:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    order = ["done", "running", "error", "failed", "pending"]
    summary_bits = [
        f'<span class="status-{k}">{esc(STATUS_TEXT[k])}: {counts.get(k, 0)}</span>'
        for k in order
    ]
    summary_html = " &nbsp;·&nbsp; ".join(summary_bits)

    rows = []
    for e in entries:
        status_cls = e["status"]
        status = STATUS_TEXT[status_cls]
        if e.get("detail"):
            status += f" ({esc(e['detail'])})"
        rows.append(
            f'<tr data-status="{status_cls}">'
            f'<td class="num">{esc(e["folder"])}</td>'
            f'<td class="num">{esc(e["total_cores"])}</td>'
            f'<td class="num">{esc(e["ntasks"])}</td>'
            f'<td class="num">{esc(e["cpus_per_task"])}</td>'
            f'<td class="num">{esc(e.get("n_steps", 0))}</td>'
            f'<td class="{status_cls}">{status}</td></tr>'
        )
    table_rows = "\n".join(rows)

    asof_html = (
        f'<p class="asof">Status as of <b>{esc(generated_at)}</b> — this page is a '
        "snapshot; re-run <code>vasp-core-benchmarking status</code> "
        "(or <code>report</code>) to refresh.</p>"
        if generated_at
        else ""
    )

    return f"""<style>
  :root {{ --fg:#222; --muted:#667; --accent:#2c7fb8; --line:#e2e4e8; }}
  body {{ font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; color: var(--fg);
          margin: 0; padding: 28px 32px; background: #fafafb; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  p.lead {{ color: var(--muted); margin: 0 0 8px; }}
  p.asof {{ color: var(--muted); font-size: 12px; margin: 0 0 22px; }}
  p.asof code {{ font-size: 11px; background: #eef1f4; padding: 1px 5px; border-radius: 4px; }}
  .panel {{ background: #fff; border: 1px solid var(--line); border-radius: 10px;
            padding: 20px; margin-bottom: 22px; }}
  .summary {{ font-size: 14px; margin-bottom: 14px; }}
  label.sel {{ display: inline-flex; flex-direction: column; font-size: 12px; color: var(--muted); }}
  label.sel span {{ margin-bottom: 4px; font-weight: 600; }}
  select {{ font-size: 14px; padding: 6px 8px; border: 1px solid #ccd; border-radius: 6px;
            background: #fff; min-width: 140px; }}
  .status-done {{ color: #2ca25f; font-weight: 600; }}
  .status-running {{ color: #2c7fb8; font-weight: 600; }}
  .status-error {{ color: #c0392b; font-weight: 600; }}
  .status-failed {{ color: #c0392b; font-weight: 600; }}
  .status-pending {{ color: #d08000; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--muted); font-weight: 600; }}
  td.num {{ font-variant-numeric: tabular-nums; }}
  td:first-child {{ font-weight: 700; color: var(--accent); }}
  td.done {{ color: #2ca25f; }} td.running {{ color: #2c7fb8; }}
  td.error {{ color: #c0392b; }} td.failed {{ color: #c0392b; }} td.pending {{ color: #999; }}
  .wrap {{ overflow-x: auto; }}
</style>

<h1>VASP core benchmarking — folder status</h1>
<p class="lead">Every benchmark layout under the root, with its current run state.
The folder's own <code>submit.sl</code> defines its layout; folder names encode
<b>total cores</b>, <b>MPI ranks</b> (ntasks) and <b>OpenMP threads</b> (cpus-per-task).
A config counts as <b>run</b> once it has logged more than <b>{esc(skip_steps)}</b>
electronic steps (one usable step remains after the warm-up steps are dropped).</p>
{asof_html}

<div class="panel">
  <div class="summary">{summary_html}</div>
  <label class="sel"><span>Show</span>
    <select id="filter" onchange="applyFilter()">
      <option value="__all__" selected>all statuses</option>
      <option value="done">done</option>
      <option value="running">running</option>
      <option value="error">error</option>
      <option value="failed">failed</option>
      <option value="pending">pending</option>
    </select>
  </label>
</div>

<div class="panel wrap">
  <table>
    <thead><tr><th>Folder</th><th>Total cores</th><th>MPI ranks</th>
      <th>OpenMP threads</th><th>Electronic steps</th><th>Status</th></tr></thead>
    <tbody id="rows">
{table_rows}
    </tbody>
  </table>
</div>

<script>
function applyFilter() {{
  const want = document.getElementById("filter").value;
  for (const tr of document.querySelectorAll("#rows tr")) {{
    tr.style.display = (want === "__all__" || tr.dataset.status === want) ? "" : "none";
  }}
}}
applyFilter();
</script>
"""


def _scan_and_write(
    root_dir: Path, use_sacct: bool, skip_steps: int
) -> tuple[Path, list[dict]]:
    """Scan ``root`` and write the status page, stamped with the scan time."""
    entries = scan_configs(root_dir, use_sacct=use_sacct, skip_steps=skip_steps)
    out_path = root_dir / INDEX_FILENAME
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_path.write_text(
        build_index_html(entries, generated_at=generated_at, skip_steps=skip_steps)
    )
    return out_path, entries


def refresh_index(
    root_dir: str | Path, use_sacct: bool = True, skip_steps: int = DEFAULT_SKIP_STEPS
) -> tuple[Path, list[dict]]:
    """Re-scan ``root`` and (re)write ``folder_index.html``. Returns ``(path, entries)``.

    ``use_sacct`` lets the classifier query SLURM to tell a still-running job
    apart from a failed one; set it False to rely only on local files.
    ``skip_steps`` sets the "> N electronic steps = a usable run" threshold.
    """
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"benchmark root not found: {root_dir}")
    return _scan_and_write(root_dir, use_sacct=use_sacct, skip_steps=skip_steps)
