"""VASP benchmarking toolkit.

Three-part workflow for benchmarking VASP across SLURM ntasks x cpus-per-task
configurations:

  1. ``setup``  - create one directory + submit.sl per (ntasks, cpus-per-task).
  2. ``submit`` - submit every generated submit.sl to SLURM.
  3. ``report`` - collect sacct utilisation and OUTCAR electronic-step
                  efficiency, then write a CSV and an interactive HTML report.

Plus an optional ``clean`` step that deletes the large output files, keeping only
the inputs, OUTCAR/OSZICAR and slurm logs.
"""

__version__ = "0.1.0"
