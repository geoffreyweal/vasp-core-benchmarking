"""Part 3: collect benchmarking results and write a CSV + HTML report.

For every run directory under ``--root`` that contains a usable OUTCAR, this
collects:

  * the parallel layout (ntasks, cpus-per-task, total cores) from the OUTCAR
    header;
  * the mean & std-dev of the per-electronic-step wall time (LOOP real time),
    excluding the first ``--skip-steps`` warm-up steps (default 5);
  * SLURM utilisation from sacct (elapsed, CPU utilisation, peak memory), unless
    ``--no-sacct`` is given;
  * speedup and parallel efficiency relative to the fastest single-core run.

The HTML report shows every metric on a 2x2 grid. A single radio toggle switches
all four cells between a 3D view (MPI x OpenMP x metric) and a 2D view
(MPI vs metric), and a single shared slider selects which OpenMP-thread value the
2D view shows - so all metrics move together.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import sacct
from .outcar import parse_loop_times, parse_outcar_header

# Metrics shown in the report grid: (column, axis label, std-dev column or None).
# Order maps to the 2x2 cells below (top row, then bottom row):
#   top:    CPU utilisation, Max Memory utilisation
#   bottom: Time / electronic step, Speedup
METRICS = [
    ("cpu_utilisation_pct", "CPU utilisation (%)", None),
    ("max_memory_utilisation_gb", "Max Memory utilisation (GB)", None),
    ("loop_real_mean_s", "Time / electronic step (s)", "loop_real_std_s"),
    ("speedup", "Speedup (t<sub>1</sub> / t<sub>N</sub>)", None),
]

# A distinct accent colour per metric cell (blue, green, orange, purple).
METRIC_COLORS = ["#2c7fb8", "#2ca25f", "#e6550d", "#756bb1"]
FONT_FAMILY = "Helvetica Neue, Helvetica, Arial, sans-serif"

# 2x2 paper-domain layout for the four metric cells (top row, then bottom row).
_CELLS = [
    {"xd": [0.00, 0.45], "yd": [0.57, 0.93]},
    {"xd": [0.55, 1.00], "yd": [0.57, 0.93]},
    {"xd": [0.00, 0.45], "yd": [0.08, 0.44]},
    {"xd": [0.55, 1.00], "yd": [0.08, 0.44]},
]

# Shared styling for the 3D scene axes and the 2D cartesian axes.
_SCENE_AXIS_STYLE = dict(
    backgroundcolor="rgba(245,245,247,0.6)",
    gridcolor="rgba(0,0,0,0.12)",
    showbackground=True,
    zerolinecolor="rgba(0,0,0,0.2)",
)
_AXIS_STYLE = dict(
    gridcolor="rgba(0,0,0,0.08)",
    zeroline=False,
    linecolor="rgba(0,0,0,0.25)",
    ticks="outside",
    ticklen=4,
    tickcolor="rgba(0,0,0,0.25)",
)


def collect_run(run_dir: Path, use_sacct: bool, skip_steps: int = 5) -> dict | None:
    """Build a result row for one run directory, or None if it is unusable.

    The first ``skip_steps`` electronic steps are dropped from the timing average:
    early steps carry setup/warm-up overhead and are not representative. A run is
    only usable if it has more than ``skip_steps`` electronic steps (so at least
    one remains to average).
    """
    outcar = run_dir / "OUTCAR"
    if not outcar.is_file():
        return None

    header = parse_outcar_header(outcar)
    if header is None:
        return None
    ntasks, cpus_per_task, nodes = header
    total_cores = ntasks * cpus_per_task

    loops = parse_loop_times(outcar)
    if len(loops) <= skip_steps:
        # Need at least one step left after dropping the warm-up steps.
        return None
    steady = loops[skip_steps:]
    loop_mean = statistics.fmean(steady)
    loop_std = statistics.pstdev(steady) if len(steady) > 1 else 0.0

    row = {
        "config": run_dir.name,
        "ntasks": ntasks,
        "cpus_per_task": cpus_per_task,
        "nodes": nodes,
        "total_cores": total_cores,
        "n_electronic_steps": len(loops),
        "loop_real_mean_s": loop_mean,
        "loop_real_std_s": loop_std,
        "elapsed_s": None,
        "cpu_utilisation_pct": None,
        "max_memory_utilisation_gb": None,
        "job_id": None,
    }

    if use_sacct:
        job_id = sacct.find_job_id(run_dir)
        row["job_id"] = job_id
        util = sacct.get_utilisation(run_dir)
        if util is not None:
            elapsed, total_cpu, max_rss_gb = util
            row["elapsed_s"] = elapsed
            row["max_memory_utilisation_gb"] = max_rss_gb
            if elapsed > 0 and total_cores > 0:
                util_pct = total_cpu / (elapsed * total_cores) * 100.0
                row["cpu_utilisation_pct"] = max(0.0, min(100.0, util_pct))

    return row


def add_scaling_metrics(
    df: pd.DataFrame,
    base_time: float | None = None,
    base_cores: int | None = None,
    base_config: str | None = None,
) -> pd.DataFrame:
    """Add speedup and parallel-efficiency columns.

    Speedup is ``t_1 / t_N`` against a baseline t_1. The baseline is chosen as:

    * ``base_time`` (+ ``base_cores``) if given directly (e.g. from an external
      run supplied via ``report(baseline=...)``);
    * else the run whose ``config`` equals ``base_config`` if given;
    * else the 1 MPI x 1 OMP run (total cores = 1).

    If no baseline can be resolved (default mode and no 1x1 run), the
    speedup/efficiency columns are left as NaN so the report's speedup cell is
    blank rather than scaled to an arbitrary baseline.
    """
    df = df.sort_values("total_cores").reset_index(drop=True)

    if base_time is None:
        if base_config is not None:
            base = df[df["config"] == base_config]
            if base.empty:
                raise ValueError(
                    f"baseline config {base_config!r} not found among the runs"
                )
            base_time = float(base["loop_real_mean_s"].iloc[0])
            base_cores = int(base["total_cores"].iloc[0])
        else:
            base = df[(df["ntasks"] == 1) & (df["cpus_per_task"] == 1)]
            if base.empty:
                df["speedup"] = float("nan")
                df["ideal_speedup"] = float("nan")
                df["parallel_efficiency_pct"] = float("nan")
                return df
            base_time = float(base["loop_real_mean_s"].iloc[0])
            base_cores = 1

    if not base_cores or base_cores <= 0:
        base_cores = 1

    df["speedup"] = base_time / df["loop_real_mean_s"]
    df["ideal_speedup"] = df["total_cores"] / base_cores
    df["parallel_efficiency_pct"] = (df["speedup"] / df["ideal_speedup"]) * 100.0
    return df


def _axis_suffix(i: int) -> str:
    """Plotly axis/scene suffix: '' for the first subplot, '2', '3', ... after."""
    return "" if i == 0 else str(i + 1)


def _build_figure(df: pd.DataFrame):
    """Build the 2x2 metric grid with a shared 3D/2D toggle and OMP slider.

    Each metric occupies one cell, present as both a 3D trace (MPI x OMP x metric)
    and one 2D trace per OpenMP value (MPI vs metric). A radio toggle flips every
    cell between 3D and 2D at once; a single slider chooses the OpenMP value shown
    by every 2D cell at once.
    """
    import plotly.graph_objects as go

    df = df.sort_values(["cpus_per_task", "ntasks"]).copy()
    # Coerce metric columns to numeric so missing values become NaN (plotly
    # rejects None in marker colour/position arrays, e.g. when --no-sacct leaves
    # CPU utilisation empty).
    for col, _label, err_col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if err_col is not None:
            df[err_col] = pd.to_numeric(df[err_col], errors="coerce")

    omp_values = sorted(df["cpus_per_task"].unique())
    n_omp = len(omp_values)
    n_metrics = len(METRICS)

    fig = go.Figure()
    # Bookkeeping: which trace indices belong to the 3D view vs each OMP's 2D view.
    threed_idx: list[int] = []
    twod_idx: dict[float, list[int]] = {omp: [] for omp in omp_values}

    for i, (col, label, err_col) in enumerate(METRICS):
        suffix = _axis_suffix(i)

        # 3D trace for this metric (visible at start). Colour by the metric
        # value, unless the metric has no data (then use the cell accent colour).
        marker = dict(size=5, opacity=0.9, line=dict(width=0))
        if df[col].notna().any():
            marker.update(
                color=df[col],
                colorscale="Viridis",
                # low time = good, so reverse the scale for the time metric
                reversescale=(col == "loop_real_mean_s"),
                showscale=False,        # no colourbars in the grid
            )
        else:
            marker.update(color=METRIC_COLORS[i])
        threed_idx.append(len(fig.data))
        fig.add_trace(
            go.Scatter3d(
                x=df["ntasks"],
                y=df["cpus_per_task"],
                z=df[col],
                mode="markers",
                marker=marker,
                text=df["config"],
                customdata=df["total_cores"],
                hovertemplate=(
                    "Total cores = %{customdata}<br>"
                    "MPI ranks = %{x}<br>OpenMP threads = %{y}<br>"
                    + label
                    + " = %{z:.4g}<br>%{text}<extra></extra>"
                ),
                scene="scene" + suffix,
                name=label,
                showlegend=False,
                visible=True,
            )
        )

        # 2D traces for this metric, one per OMP value (hidden at start).
        for omp in omp_values:
            sub = df[df["cpus_per_task"] == omp].sort_values("ntasks")
            err = (
                dict(type="data", array=sub[err_col], visible=True)
                if err_col is not None
                else None
            )
            twod_idx[omp].append(len(fig.data))
            fig.add_trace(
                go.Scatter(
                    x=sub["ntasks"],
                    y=sub[col],
                    error_y=err,
                    mode="markers+lines",
                    line=dict(color=METRIC_COLORS[i], width=2.5),
                    marker=dict(
                        size=9, color=METRIC_COLORS[i],
                        line=dict(width=1.5, color="white"),
                    ),
                    text=sub["config"],
                    customdata=sub["total_cores"],
                    hovertemplate=(
                        "Total cores = %{customdata}<br>"
                        "MPI ranks = %{x}<br>"
                        f"OpenMP threads = {omp}<br>"
                        f"{label} = %{{y:.4g}}<br>"
                        "%{text}<extra></extra>"
                    ),
                    xaxis="x" + suffix,
                    yaxis="y" + suffix,
                    name=label,
                    showlegend=False,
                    visible=False,
                )
            )
            # For speedup, add a dotted ideal (perfect-scaling) reference line.
            if col == "speedup":
                twod_idx[omp].append(len(fig.data))
                fig.add_trace(
                    go.Scatter(
                        x=sub["ntasks"],
                        y=sub["ideal_speedup"],
                        mode="lines",
                        line=dict(dash="dot", color="rgba(120,120,120,0.7)", width=1.5),
                        customdata=sub["total_cores"],
                        hovertemplate=(
                            "Total cores = %{customdata}<br>"
                            "MPI ranks = %{x}<br>"
                            f"OpenMP threads = {omp}<br>"
                            "ideal speedup = %{y:.4g}<extra></extra>"
                        ),
                        xaxis="x" + suffix,
                        yaxis="y" + suffix,
                        name="ideal (1:1)",
                        showlegend=False,
                        visible=False,
                    )
                )

    # Visibility patterns across all traces.
    def vis_3d() -> list[bool]:
        v = [False] * len(fig.data)
        for idx in threed_idx:
            v[idx] = True
        return v

    def vis_2d(k: int) -> list[bool]:
        v = [False] * len(fig.data)
        for idx in twod_idx[omp_values[k]]:
            v[idx] = True
        return v

    # Fixed x range (MPI ranks) shared by every 2D cell, so the axes stay put as
    # the OMP slider moves.
    x_max = float(df["ntasks"].max())
    x_range = [0, x_max * 1.08]

    # Per-metric scenes, cartesian axes and titles (all sharing the cell domains).
    layout: dict = {}
    annotations = []
    for i, (col, label, err_col) in enumerate(METRICS):
        suffix = _axis_suffix(i)
        xd = _CELLS[i]["xd"]
        yd = _CELLS[i]["yd"]
        layout["scene" + suffix] = dict(
            domain=dict(x=xd, y=yd),
            xaxis=dict(title="MPI ranks (ntasks)", visible=True, **_SCENE_AXIS_STYLE),
            yaxis=dict(title="OpenMP (cpus-per-task)", visible=True, **_SCENE_AXIS_STYLE),
            zaxis=dict(title=label, visible=True, **_SCENE_AXIS_STYLE),
        )
        # Fixed y range from all OMP values of this metric, so switching the
        # slider keeps the same limits (None -> autorange if the metric has no
        # data, e.g. CPU utilisation with --no-sacct). For speedup, include the
        # ideal line so it always fits.
        y_max = df[col].max()
        if col == "speedup":
            y_max = max(y_max, df["ideal_speedup"].max())
        y_range = [0, float(y_max) * 1.08] if pd.notna(y_max) and y_max > 0 else None
        layout["xaxis" + suffix] = dict(
            domain=xd, anchor="y" + suffix, title="MPI ranks (ntasks)",
            range=list(x_range), autorange=False, visible=False, **_AXIS_STYLE,
        )
        layout["yaxis" + suffix] = dict(
            domain=yd, anchor="x" + suffix, title=label, visible=False, **_AXIS_STYLE,
            **({"range": y_range, "autorange": False} if y_range else {"rangemode": "tozero"}),
        )
        annotations.append(
            dict(
                text=f"<b>{label}</b>",
                x=(xd[0] + xd[1]) / 2,
                y=min(1.0, yd[1] + 0.04),
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, color=METRIC_COLORS[i], family=FONT_FAMILY),
            )
        )
        # Speedup needs a 1 MPI x 1 OMP baseline; note it in the cell if absent.
        if col == "speedup" and not df[col].notna().any():
            annotations.append(
                dict(
                    text="needs a 1 MPI &times; 1 OMP run",
                    x=(xd[0] + xd[1]) / 2,
                    y=(yd[0] + yd[1]) / 2,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="rgba(0,0,0,0.45)", family=FONT_FAMILY),
                )
            )

    # A tiny off-the-plots domain to park scenes in when showing 2D, so their
    # (empty) WebGL canvas no longer overlaps - and steals hover events from -
    # the 2D plots underneath.
    _COLLAPSED = {"x": [0.0, 0.0001], "y": [0.0, 0.0001]}

    # Layout dicts that show one view and hide the other across every cell.
    def show_3d() -> dict:
        d = {}
        for i in range(n_metrics):
            s = _axis_suffix(i)
            d[f"scene{s}.domain"] = dict(x=_CELLS[i]["xd"], y=_CELLS[i]["yd"])
            d[f"scene{s}.xaxis.visible"] = True
            d[f"scene{s}.yaxis.visible"] = True
            d[f"scene{s}.zaxis.visible"] = True
            d[f"xaxis{s}.visible"] = False
            d[f"yaxis{s}.visible"] = False
        return d

    def show_2d() -> dict:
        d = {}
        for i in range(n_metrics):
            s = _axis_suffix(i)
            d[f"scene{s}.domain"] = dict(_COLLAPSED)
            d[f"scene{s}.xaxis.visible"] = False
            d[f"scene{s}.yaxis.visible"] = False
            d[f"scene{s}.zaxis.visible"] = False
            d[f"xaxis{s}.visible"] = True
            d[f"yaxis{s}.visible"] = True
        return d

    buttons = [
        dict(
            label="3D (MPI x OMP x metric)",
            method="update",
            args=[{"visible": vis_3d()}, show_3d()],
        ),
        dict(
            label="2D (MPI vs metric, OMP slider)",
            method="update",
            args=[{"visible": vis_2d(0)}, show_2d()],
        ),
    ]
    steps = [
        dict(
            label=str(omp_values[k]),
            method="update",
            args=[{"visible": vis_2d(k)}, show_2d()],
        )
        for k in range(n_omp)
    ]

    layout["updatemenus"] = [
        dict(
            type="buttons",
            direction="right",
            showactive=True,
            active=0,
            x=0.0,
            xanchor="left",
            y=1.10,
            yanchor="bottom",
            buttons=buttons,
            bgcolor="white",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=12, family=FONT_FAMILY),
            pad=dict(t=4, b=4, l=6, r=6),
        )
    ]
    layout["sliders"] = [
        dict(
            active=0,
            currentvalue={
                "prefix": "OpenMP / cpus-per-task (2D view) = ",
                "font": dict(size=13, color="#444", family=FONT_FAMILY),
            },
            pad={"t": 20},
            steps=steps,
            bgcolor="rgba(0,0,0,0.12)",
            bordercolor="rgba(0,0,0,0.0)",
            tickcolor="rgba(0,0,0,0.4)",
            font=dict(size=11, family=FONT_FAMILY),
        )
    ]
    layout["annotations"] = annotations
    layout["template"] = "plotly_white"
    layout["height"] = 900
    layout["margin"] = dict(t=100, b=50, l=60, r=40)
    layout["title"] = dict(
        text="VASP benchmarking &#8226; MPI &times; OpenMP scaling",
        x=0.5,
        xanchor="center",
        font=dict(size=20, family=FONT_FAMILY, color="#222"),
    )
    layout["font"] = dict(family=FONT_FAMILY, size=12, color="#333")
    layout["paper_bgcolor"] = "white"
    layout["plot_bgcolor"] = "white"
    layout["hovermode"] = "closest"  # show a hover box for the nearest point
    layout["hoverlabel"] = dict(
        bgcolor="white",
        bordercolor="black",
        font=dict(size=12, family=FONT_FAMILY, color="black"),
    )

    fig.update_layout(**layout)
    return fig


def write_html(df: pd.DataFrame, out_path: Path) -> None:
    """Write the report HTML (self-contained, plotly.js embedded)."""
    _build_figure(df).write_html(str(out_path), include_plotlyjs=True)


def report(
    root: str = "VASP_Benchmarking",
    out: str = "report",
    no_sacct: bool = False,
    baseline: str | None = None,
    skip_steps: int = 5,
) -> pd.DataFrame:
    """Run the full report pipeline. Returns the results DataFrame.

    ``skip_steps`` is the number of leading (warm-up) electronic steps dropped
    from each run's timing average; runs with no more than ``skip_steps`` steps
    are skipped.

    ``baseline`` selects the t_1 run for the speedup metric: either a config name
    (e.g. ``1cores_1tasks_1cpt``) present in ``root``, or a path to a separate run
    directory (e.g. a non-hyperthreaded single-core run). Default: the
    1 MPI x 1 OMP run.
    """
    if skip_steps < 0:
        raise ValueError(f"--skip-steps must be >= 0, got {skip_steps}")

    root_dir = Path(root)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"benchmark root not found: {root_dir}")

    use_sacct = not no_sacct
    rows: list[dict] = []
    skipped: list[str] = []

    # A run directory is any directory directly containing an OUTCAR.
    print(f"Scanning {root_dir}/ for runs (OUTCAR files)...")
    run_dirs = sorted({p.parent for p in root_dir.rglob("OUTCAR")})
    print(
        f"Found {len(run_dirs)} run director{'y' if len(run_dirs) == 1 else 'ies'}. "
        f"Reading OUTCARs"
        + (" and querying sacct" if use_sacct else " (sacct disabled)")
        + f" (dropping the first {skip_steps} electronic step(s) per run)..."
    )

    progress = tqdm(run_dirs, desc="Collecting", unit="run")
    for run_dir in progress:
        progress.set_postfix_str(run_dir.name)
        row = collect_run(run_dir, use_sacct, skip_steps=skip_steps)
        if row is None:
            skipped.append(str(run_dir))
        else:
            rows.append(row)
    progress.close()
    print(f"  parsed {len(rows)} usable run(s); skipped {len(skipped)}.")

    if not rows:
        print(f"No usable runs found under {root_dir}/")
        if skipped:
            print(
                f"Skipped {len(skipped)} directories "
                f"(no header / <= {skip_steps} LOOP steps)."
            )
        return pd.DataFrame()

    print("Computing scaling metrics (speedup, parallel efficiency)...")
    df = pd.DataFrame(rows)

    # Resolve the t_1 baseline for speedup.
    base_time = base_cores = base_config = None
    if baseline is not None:
        bpath = Path(baseline)
        if bpath.is_dir():
            brow = collect_run(bpath, use_sacct=False, skip_steps=skip_steps)
            if brow is None:
                raise ValueError(
                    f"baseline run {bpath} has no usable OUTCAR "
                    f"(need more than {skip_steps} LOOP steps)"
                )
            base_time = brow["loop_real_mean_s"]
            base_cores = brow["total_cores"]
            print(
                f"  t_1 baseline from {bpath} "
                f"({base_cores} cores, {base_time:.3f} s/step)"
            )
        else:
            base_config = baseline
            print(f"  t_1 baseline from config '{base_config}'")

    df = add_scaling_metrics(
        df, base_time=base_time, base_cores=base_cores, base_config=base_config
    )
    if not df["speedup"].notna().any():
        print("  note: no 1 MPI x 1 OMP run, so speedup is left blank.")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    print(f"Writing results table -> {csv_path}")
    df.to_csv(csv_path, index=False)

    html_path = out_dir / "vasp_benchmark_results.html"
    print(f"Building interactive plot -> {html_path} (embedding plotly.js)...")
    write_html(df, html_path)

    if skipped:
        (out_dir / "skipped.txt").write_text("\n".join(skipped) + "\n")
        print(f"Wrote list of skipped directories -> {out_dir / 'skipped.txt'}")

    print(
        f"Done: {len(df)} run(s) reported, {len(skipped)} skipped. "
        f"Open {html_path} to view."
    )
    return df
