"""VASP benchmarking toolkit.

Three-part workflow for benchmarking VASP across SLURM ntasks x cpus-per-task
configurations:

  1. ``setup``  - create one directory + submit.sl per (ntasks, cpus-per-task).
  2. ``submit`` - submit the configs that need running (pending + failed).
  3. ``report`` - collect sacct utilisation and OUTCAR electronic-step
                  efficiency, then write a CSV and an interactive HTML report.

Plus ``status`` (re-scan the folders and refresh a folder_index.html snapshot of
each config's run state, where "run" means it logged more than --skip-steps
electronic steps), ``reset`` (clear configs that produced no usable result back to
pending), and an optional ``clean`` step that deletes the large output files,
keeping only the inputs, OUTCAR/OSZICAR and slurm logs.
"""

__version__ = "0.1.0"
