# utils/nomad_io.py
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple, Union, Optional, List
import os


PointKey = Tuple[float, ...]

__all__ = [
    "NomadCacheFormat",
    "write_nomad_cache",
    "scale_first_column_in_stats_file",
    "NomadStatsColumns",
    "AppendGlobalStatsResult",
    "append_global_stats",
    "FilterConsecutiveResult",
    "filter_consecutive_duplicates_by_column",
]


@dataclass(frozen=True)
class NomadCacheFormat:
    """
    Describe the minimal information needed to write a NOMAD cache file.
    """
    bb_output_type: str = "OBJ"
    coord_fmt: str = "%.17g"
    obj_fmt: str = "%.17g"
    cache_hits: int = 0


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Write text to `path` atomically (best effort).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_nomad_cache(
    doe_s: Mapping[PointKey, float],
    cache_path: Union[str, Path],
    fmt: NomadCacheFormat = NomadCacheFormat(),
    sort_items: bool = True,
) -> None:
    """
    Write a NOMAD CACHE_FILE from a mapping {s_tuple: fval}.

    Parameters
    ----------
    doe_s
        Mapping from point coordinates (tuple) to objective value.
    cache_path
        Output path.
    fmt
        Output formatting options.
    sort_items
        If True, write entries in a deterministic order (by f then coordinates).

    Notes
    -----
    This writer generates a cache compatible with the typical NOMAD format:
      CACHE_HITS <int>
      BB_OUTPUT_TYPE <string>
      ( s1 ... sp ) BB_EVAL_OK ( f ) SURROGATE_EVAL_NOT_STARTED ( )

    If you later use multi-output (e.g., "OBJ CNT_EVAL"), this writer must be adapted.
    """
    cache_path = Path(cache_path)

    # Empty cache is allowed; still write header for NOMAD.
    if not doe_s:
        text = f"CACHE_HITS {int(fmt.cache_hits)}\nBB_OUTPUT_TYPE {fmt.bb_output_type}\n"
        _atomic_write_text(cache_path, text)
        return

    # Check consistent dimensions
    dims = {len(k) for k in doe_s.keys()}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent point dimensions in doe_s: {sorted(dims)}")
    p = next(iter(dims))
    if p <= 0:
        raise ValueError("Points must have positive dimension.")

    def fmt_vec(s: PointKey) -> str:
        vals = "  ".join(fmt.coord_fmt % float(v) for v in s)
        return f"( {vals} )"

    def fmt_obj(v: float) -> str:
        return f"( {fmt.obj_fmt % float(v)} )"

    items = list(doe_s.items())
    if sort_items:
        items.sort(key=lambda kv: (float(kv[1]), tuple(float(x) for x in kv[0])))

    lines: List[str] = []
    lines.append(f"CACHE_HITS {int(fmt.cache_hits)}")
    lines.append(f"BB_OUTPUT_TYPE {fmt.bb_output_type}")

    for s_tuple, fval in items:
        if len(s_tuple) != p:
            raise ValueError("Unexpected point dimension while writing cache.")
        line = (
            f"{fmt_vec(tuple(float(v) for v in s_tuple))} "
            f"BB_EVAL_OK {fmt_obj(float(fval))} "
            f"SURROGATE_EVAL_NOT_STARTED ( )"
        )
        lines.append(line)

    _atomic_write_text(cache_path, "\n".join(lines) + "\n")


def scale_first_column_in_stats_file(
    stats_path: Union[str, Path],
    scale: int,
) -> None:
    """
    Multiply the first numeric column of a NOMAD stats file by `scale`, in-place.

    Parameters
    ----------
    stats_path
        Path to the NOMAD stats file.
    scale
        Integer multiplier (must be > 0).

    Notes
    -----
    - Non-parseable lines are preserved as-is.
    - The file is rewritten atomically.
    """
    stats_path = Path(stats_path)
    if scale <= 0:
        raise ValueError("scale must be a positive integer.")
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    out_lines: List[str] = []
    for raw in stats_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            out_lines.append(raw)
            continue

        parts = raw.split()
        if len(parts) < 2:
            out_lines.append(raw)
            continue

        # First column is often an evaluation counter; allow "141" or "141.0".
        try:
            col0 = int(float(parts[0]))
        except ValueError:
            out_lines.append(raw)
            continue

        parts[0] = str(col0 * int(scale))
        out_lines.append(" ".join(parts))

    _atomic_write_text(stats_path, "\n".join(out_lines) + "\n")


@dataclass(frozen=True)
class AppendGlobalStatsResult:
    lines_written: int
    last_bbe: Optional[int]
    last_obj: Optional[float]
    last_mesh: Optional[List[float]]
    last_poll: Optional[List[float]]


@dataclass(frozen=True)
class NomadStatsColumns:
    """
    Column indices in a NOMAD/PyNomad stats line.

    If you set DISPLAY_STATS / STATS_FILE with keywords, NOMAD writes those values
    in the same order. (BBE, OBJ, etc.)
    """
    bbe: int = 0
    obj: int = 1
    # If mesh/poll are written, a common layout is:
    # BBE OBJ (mesh_1 ... mesh_p) (poll_1 ... poll_p)
    mesh_start: Optional[int] = 2  # start index for mesh vector (if present)


def _needs_leading_newline_for_append(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as f:
        f.seek(-1, 2)
        last = f.read(1)
    return last != b"\n"


def append_global_stats(
    nomad_stats_path: Union[str, Path],
    global_stats_path: Union[str, Path],
    offset_bbe: int,
    *,
    columns: NomadStatsColumns = NomadStatsColumns(),
    # mesh/poll extraction (optional)
    extract_mesh_poll: bool = True,
    p: Optional[int] = None,
    obj_format: str = ".15g",
) -> AppendGlobalStatsResult:
    """
    Append NOMAD/PyNomad stats into a global file as two columns: (BBE+offset) OBJ.

    Parameters
    ----------
    nomad_stats_path
        Path to the NOMAD/PyNomad stats file created via STATS_FILE.
    global_stats_path
        Destination file to append: each line "<bbe+offset> <obj>".
    offset_bbe
        Offset added to BBE to make a global evaluation counter.
    columns
        Indices of BBE/OBJ (and optionally mesh_start) in the stats lines.
        This stays valid even if you add/remove other DISPLAY_STATS keywords, as long
        as you update these indices accordingly. 
    extract_mesh_poll
        If True, tries to extract mesh/poll vectors from the *last parseable line*,
        assuming the standard contiguous layout after OBJ:
          mesh (p values) then poll (p values)
        If you changed the stats layout, set this to False or set columns.mesh_start
        and provide p.
    p
        Dimension for mesh/poll vectors. If None, we will only infer p if the remaining
        number of columns after mesh_start is exactly even (2*p).
    obj_format
        Python float formatting spec (without the leading ':', e.g. '.15g').

    Returns
    -------
    AppendGlobalStatsResult
        Includes last parsed mesh/poll (if extractable), and how many lines were appended.

    Notes
    -----
    NOMAD writes STATS_FILE lines according to DISPLAY_STATS keywords (e.g., BBE, OBJ,
    MESH_SIZE, POLL_SIZE).
    """
    src = Path(nomad_stats_path)
    dst = Path(global_stats_path)

    if not src.exists():
        raise FileNotFoundError(f"Stats source not found: {src}")
    if not isinstance(offset_bbe, int):
        raise TypeError("offset_bbe must be an int.")

    out_lines: List[str] = []
    last_bbe: Optional[int] = None
    last_obj: Optional[float] = None
    last_mesh: Optional[List[float]] = None
    last_poll: Optional[List[float]] = None

    with src.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            parts = line.split()
            # Need at least BBE/OBJ columns
            if len(parts) <= max(columns.bbe, columns.obj):
                continue

            try:
                bbe = int(float(parts[columns.bbe]))  # tolerant to "141" or "141.0"
                obj = float(parts[columns.obj])
            except ValueError:
                continue

            out_lines.append(f"{bbe + offset_bbe} {obj:{obj_format}}")
            last_bbe, last_obj = bbe, obj

            # Optional mesh/poll extraction from the last parseable line
            if extract_mesh_poll and columns.mesh_start is not None:
                start = columns.mesh_start
                if len(parts) > start:
                    rest = parts[start:]
                    p_eff: Optional[int] = p
                    if p_eff is None and (len(rest) % 2 == 0) and (len(rest) >= 2):
                        p_eff = len(rest) // 2

                    if p_eff is not None and len(rest) >= 2 * p_eff:
                        try:
                            mesh = [float(x) for x in rest[:p_eff]]
                            poll = [float(x) for x in rest[p_eff:2 * p_eff]]
                            last_mesh, last_poll = mesh, poll
                        except ValueError:
                            # keep previous extracted vectors (best effort)
                            pass

    # Append to destination file
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a", encoding="utf-8") as g:
        if out_lines:
            if _needs_leading_newline_for_append(dst):
                g.write("\n")
            g.write("\n".join(out_lines))
            g.write("\n")

    return AppendGlobalStatsResult(
        lines_written=len(out_lines),
        last_bbe=last_bbe,
        last_obj=last_obj,
        last_mesh=last_mesh,
        last_poll=last_poll,
    )


@dataclass(frozen=True)
class FilterConsecutiveResult:
    lines_in: int
    lines_out: int
    removed: int


def filter_consecutive_duplicates_by_column(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    *,
    value_col: int = 1,
    tol: float = 0.0,
    keep_empty_lines: bool = False,
) -> FilterConsecutiveResult:
    """
    Remove consecutive duplicate values (within tolerance) from a text file.

    Parameters
    ----------
    src_path
        Input file.
    dst_path
        Output file (written atomically).
    value_col
        Zero-based index of the numeric column used for duplicate detection.
        Example: if the file contains "BBE OBJ", then OBJ is value_col=1.
    tol
        Tolerance. If tol=0, duplicates are exact equality. If tol>0, duplicates satisfy
        |v - v_prev| <= tol.
    keep_empty_lines
        If False, empty/blank lines are dropped. If True, they are preserved.

    Behavior
    --------
    - Lines that cannot be parsed for the chosen column are kept as-is and reset the
      duplicate tracking (so a following numeric line won't be compared to a previous one).
    - Only consecutive duplicates are removed.

    Returns
    -------
    FilterConsecutiveResult
        Basic statistics about filtering.
    """
    src = Path(src_path)
    dst = Path(dst_path)

    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")
    if value_col < 0:
        raise ValueError("value_col must be >= 0.")
    if tol < 0:
        raise ValueError("tol must be >= 0.")

    lines_in = 0
    out_lines: List[str] = []

    last_val: Optional[float] = None
    last_val_valid = False

    for raw in src.read_text(encoding="utf-8").splitlines():
        lines_in += 1
        line = raw.rstrip("\n")

        if not line.strip():
            if keep_empty_lines:
                out_lines.append(line)
            # empty lines do not affect duplicate tracking
            continue

        parts = line.split()
        if len(parts) <= value_col:
            out_lines.append(line)
            last_val = None
            last_val_valid = False
            continue

        try:
            v = float(parts[value_col])
        except ValueError:
            out_lines.append(line)
            last_val = None
            last_val_valid = False
            continue

        if last_val_valid:
            if tol == 0.0:
                if v == last_val:
                    continue
            else:
                if abs(v - last_val) <= tol:
                    continue

        out_lines.append(line)
        last_val = v
        last_val_valid = True

    _atomic_write_text(dst, "\n".join(out_lines) + ("\n" if out_lines else ""))

    lines_out = len(out_lines)
    return FilterConsecutiveResult(
        lines_in=lines_in,
        lines_out=lines_out,
        removed=max(0, lines_in - lines_out),
    )

