"""Read ``setup`` options from a ``key = value`` file (default: ``options.txt``).

An entire ``vasp-core-benchmarking setup`` invocation can be saved to a file
instead of being typed on the command line. When an ``options.txt`` exists in the
working directory it is picked up automatically; point at a differently named
file with ``--options``. Command-line flags override values from the file, which
in turn override the built-in defaults.

File format - one ``key = value`` per line::

    # blank lines and whole-line '#' comments are ignored
    cores        = 1,2,4,8,16-128:8
    mem          = 32G
    mem-per-cpu  = 2G
    time-policy  = 30:00,15:00@32

Keys are the long option names without the leading ``--`` (e.g. ``mem-per-cpu``);
``-`` and ``_`` are interchangeable, and a leading ``--`` is tolerated. Any
surrounding quotes on a value are stripped, so ``cores = "1,2,4"`` and
``cores = 1,2,4`` are equivalent.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_OPTIONS_FILE = "options.txt"

# Option keys accepted in the file, as their canonical hyphenated CLI names.
# Keep in sync with the `setup` subparser and _SETUP_OPTION_DESTS in cli.py.
KNOWN_OPTIONS = (
    "cores",
    "jobname-prefix",
    "vasp-files",
    "include",
    "mem",
    "mem-per-cpu",
    "time-policy",
    "root",
    "max-cpus-per-task",
    "allowed-cpus-per-task",
)


def _canonical_key(key: str) -> str:
    """Normalise a file key to its canonical hyphenated form (a CLI flag name)."""
    return key.strip().lower().lstrip("-").replace("_", "-")


def parse_options_file(path: Path) -> dict[str, str]:
    """Parse a ``key = value`` options file into a ``{dest: value}`` dict.

    Returned keys are argparse *dest* names (hyphens converted to underscores),
    ready to merge with parsed CLI arguments. Values are the raw strings; type
    conversion (e.g. int, comma lists) is left to the caller so the same
    converters as the command line are used.

    Unknown keys, missing ``=`` separators, empty values and duplicate keys each
    raise ``ValueError`` (with the file name and line number) so mistakes surface
    instead of being silently ignored.
    """
    path = Path(path)
    options: dict[str, str] = {}
    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(
                f"{path}:{lineno}: expected 'key = value', got {line!r}"
            )

        key_part, value = line.split("=", 1)
        key = _canonical_key(key_part)
        value = value.strip()
        # Strip a single pair of surrounding quotes, if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]

        if key not in KNOWN_OPTIONS:
            valid = ", ".join(KNOWN_OPTIONS)
            raise ValueError(
                f"{path}:{lineno}: unknown option {key!r}. Valid keys are: {valid}"
            )
        dest = key.replace("-", "_")
        if dest in options:
            raise ValueError(f"{path}:{lineno}: duplicate option {key!r}")
        if not value:
            raise ValueError(f"{path}:{lineno}: no value given for {key!r}")
        options[dest] = value

    return options


def load_setup_options(explicit_path: str | None) -> tuple[dict[str, str], Path | None]:
    """Load setup options from the options file.

    Returns ``(options, source)``, where ``options`` is a ``{dest: value}`` dict
    and ``source`` is the :class:`~pathlib.Path` read, or ``None`` when no file
    was found. With ``explicit_path`` set (from ``--options``) that file must
    exist; otherwise ``options.txt`` in the working directory is used when present
    and skipped when absent.
    """
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.is_file():
            raise FileNotFoundError(f"options file not found: {path}")
        return parse_options_file(path), path

    default_path = Path(DEFAULT_OPTIONS_FILE)
    if default_path.is_file():
        return parse_options_file(default_path), default_path
    return {}, None
