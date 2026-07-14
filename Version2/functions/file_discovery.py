"""Discover CESM history files on the GLADE filesystem.

This replaces the original ``collect_files`` function. Two behavioral
changes from the original:

1. ``comps`` is no longer a bare module-level global the function silently
   closes over -- it's looked up from ``config.VAR_COMPONENT`` (or any dict
   you pass in), so this module can be imported and tested without the rest
   of the pipeline being initialized first.
2. Year parsing no longer relies on ``f.split('.')[-2][:4]``. CESM history
   filenames vary in how many dot-delimited fields they have depending on
   variable name (variable names with dots in them, or case/member-id
   strings containing extra dots, shift the field you're slicing). We parse
   the trailing ``YYYY-MM`` (or ``YYYY``) date stamp with a regex anchored
   to the ``.nc`` suffix instead, which is robust to that variation.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Mapping, Sequence

# Matches the trailing CESM timeseries date stamp just before `.nc`, e.g.
#   ...cice.h.hi.192001-200512.nc      -> 192001
#   ...cam.h0.U.1920-01.nc             -> 1920
#   ...pop.h.SST.0001-01-0100-12.nc    -> 0001  (first stamp found wins)
_DATE_STAMP_RE = re.compile(r"\.(\d{4})(?:-\d{2})?[^./]*\.nc$")


def parse_file_start_year(path: str) -> int:
    """Extract the first 4-digit year from a CESM history filename.

    Raises ValueError if no date stamp can be found, rather than silently
    mis-slicing (the original code's `f.split('.')[-2][:4]` would happily
    return a wrong-but-plausible-looking year on an unexpected filename
    pattern instead of failing).
    """
    m = _DATE_STAMP_RE.search(path)
    if m is None:
        raise ValueError(
            f"Could not parse a year from filename: {path!r}. "
            f"Expected a CESM-style trailing date stamp before '.nc'."
        )
    return int(m.group(1))


def collect_member_files(
    member_dir: str,
    variables: Sequence[str],
    var_component: Mapping[str, str],
    start_year: int,
) -> dict[str, list[str]]:
    """Collect, per variable, the sorted list of monthly history files for
    one ensemble member directory, filtered to ``year >= start_year``.
    """
    out: dict[str, list[str]] = {}
    for v in variables:
        try:
            component = var_component[v]
        except KeyError as e:
            raise KeyError(
                f"Variable '{v}' has no entry in var_component; "
                f"add it to config.VAR_COMPONENT."
            ) from e
        pattern = f"{member_dir}/{component}/proc/tseries/month_1/*.{v}.*.nc"
        files = sorted(glob.glob(pattern))
        filtered = [f for f in files if parse_file_start_year(f) >= start_year]
        if not filtered:
            # Don't fail hard here -- an empty list for one variable in one
            # member is a real possibility (e.g. a member that hasn't run
            # this far yet) but it WILL break time-alignment downstream, so
            # surface it loudly rather than padding silently.
            print(
                f"[collect_member_files] WARNING: no files found for "
                f"var={v!r} in {member_dir!r} (pattern={pattern!r})"
            )
        out[v] = filtered
    return out


def collect_files(
    member_dirs: Sequence[str],
    variables: Sequence[str],
    var_component: Mapping[str, str],
    start_year: int,
) -> list[dict[str, list[str]]]:
    """Collect files for every ensemble member directory.

    Returns a list (one entry per ensemble member) of
    ``{variable: [filepaths...]}`` dicts, matching the shape the original
    notebook's ``collect_files`` produced.
    """
    return [
        collect_member_files(d, variables, var_component, start_year)
        for d in member_dirs
    ]


def discover_member_dirs(glob_pattern: str) -> list[str]:
    """Sorted, deduplicated list of ensemble member directories.

    Thin wrapper so call sites don't need a bare ``glob.glob`` + ``sorted``
    and so this is easy to mock in tests.
    """
    dirs = sorted(set(glob.glob(glob_pattern)))
    if not dirs:
        raise FileNotFoundError(f"No member directories matched pattern: {glob_pattern!r}")
    return dirs


def summarize_collection(label: str, files: list[dict[str, list[str]]]) -> None:
    """Print the same summary line the original script printed, plus a
    per-variable file count so a silently-empty variable is obvious
    immediately rather than surfacing as a confusing shape error later.
    """
    n_ens = len(files)
    n_vars = len(files[0]) if files else 0
    print(f"{label} | # ens: {n_ens} | # vars: {n_vars}")
    for i, member in enumerate(files):
        counts = {v: len(fs) for v, fs in member.items()}
        if any(c == 0 for c in counts.values()):
            print(f"  ensemble[{i}] file counts: {counts}  <-- contains an empty variable")
