"""SU2 ASCII mesh I/O helpers for IDWarp.

This module provides pure-Python (no MPI) helpers used by
``USMesh`` when ``fileType`` is set to ``"SU2"``. It mirrors the way the
OpenFOAM support is split between IDWarp's core (``UnstructuredMesh.py``)
and an external reader (``pyofm``): here the parsing/streaming logic lives
in this module while the parallel partitioning and warping glue lives in
``UnstructuredMesh.py``.

Functions
---------
read_su2_points_and_markers
    Parse NDIME, NPOIN coordinates and the boundary markers from a SU2
    ASCII mesh.
stream_write_su2_with_new_points
    Copy a SU2 mesh, replacing only the NPOIN coordinate block with new
    (deformed) coordinates. The remainder of the file (elements, markers,
    comments, ordering) is preserved byte-for-byte.
update_su2_restart
    Stream-update an ASCII SU2 restart/solution file (CSV), replacing the
    x/y(/z) columns with the deformed coordinates (no interpolation; the
    field values stay attached to their PointID).

Notes
-----
SU2 boundary elements use VTK element ids in practice:
``3 = line (2 nodes)``, ``5 = triangle (3 nodes)``, ``9 = quad (4 nodes)``.
Line elements (``3``) are the surface boundary of a 2D mesh; triangles and
quads are surface boundaries of a 3D mesh.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

VTK_NNODES = {3: 2, 5: 3, 9: 4}

IO_CHUNK_LINES = 250_000


def _strip_comment(line: str) -> str:
    """Strip a trailing SU2 ``%`` comment and surrounding whitespace."""
    if "%" in line:
        line = line.split("%", 1)[0]
    return line.strip()


def _next_data_line(it: Iterator[str]) -> str:
    """Return the next non-empty, comment-stripped line from ``it``."""
    for raw in it:
        s = _strip_comment(raw)
        if s:
            return s
    raise EOFError("Unexpected EOF while parsing SU2 file.")


def _read_npoin_block_fast(f, npoin: int, ndime: int, chunk_lines: int = IO_CHUNK_LINES) -> np.ndarray:
    """Read ``npoin`` coordinate lines quickly.

    Each line is ``x y (z) id``; the trailing id is ignored here and the
    coordinates are stored in ``pts[:, :ndime]``. A chunked
    ``numpy.fromstring`` is used to avoid per-line Python ``float`` overhead
    on large meshes, with a robust line-by-line fallback for chunks that
    contain blank/comment-only lines.
    """
    pts = np.zeros((npoin, 3), dtype=float)
    need_cols = ndime + 1

    i0 = 0
    while i0 < npoin:
        n = min(chunk_lines, npoin - i0)

        lines = [f.readline() for _ in range(n)]
        if any(l == "" for l in lines):
            raise EOFError("Unexpected EOF while reading NPOIN block.")

        s = "".join(l.split("%", 1)[0] for l in lines)
        arr = np.fromstring(s, sep=" ")
        expected = n * need_cols

        if arr.size < expected:
            for k, raw in enumerate(lines):
                parts = _strip_comment(raw).split()
                while not parts:
                    parts = _strip_comment(f.readline()).split()
                pts[i0 + k, :ndime] = [float(x) for x in parts[:ndime]]
            i0 += n
            continue

        arr = arr[:expected].reshape((n, need_cols))
        pts[i0:i0 + n, :ndime] = arr[:, :ndime]
        i0 += n

    return pts


# ---------------------------------------------------------------------------
# Public readers / writers
# ---------------------------------------------------------------------------
class _SU2FastPathError(Exception):
    """Raised internally when the fast SU2 reader cannot be used safely."""


def _parse_markers(next_data) -> Dict[str, List[List[int]]]:
    while True:
        line = next_data()
        if line.startswith("NMARK"):
            nmark = int(line.split("=", 1)[1].strip())
            break

    markers: Dict[str, List[List[int]]] = {}
    for _ in range(nmark):
        line = next_data()
        while not line.startswith("MARKER_TAG"):
            line = next_data()
        tag = line.split("=", 1)[1].strip()

        line = next_data()
        while not line.startswith("MARKER_ELEMS"):
            line = next_data()
        ne = int(line.split("=", 1)[1].strip())

        markers[tag] = [[int(x) for x in next_data().split()] for _ in range(ne)]
    return markers


def read_su2_points_and_markers(
    path: str, mode: str = "auto"
) -> Tuple[int, np.ndarray, Dict[str, List[List[int]]]]:
    """Parse NDIME, the NPOIN coordinate block and the boundary markers.

    Parameters
    ----------
    path : str
        Path to a SU2 ASCII mesh file.
    mode : {"auto", "fast", "robust"}, optional
        Reader selection:

        - ``"fast"``  : use only the vectorized reader (raw skip of the
          element block by its declared count, then a single buffered read of
          the coordinate block with ``numpy.fromstring(count=...)``). Raises
          ``_SU2FastPathError`` if the file layout is not canonical.
        - ``"robust"`` : use only the line-by-line reader (tolerant of
          interspersed comments / blank lines / non-canonical ordering). Use
          this to avoid a wasted fast attempt on files known to be
          non-canonical.
        - ``"auto"`` (default) : try ``"fast"`` first and fall back to
          ``"robust"`` transparently. Both paths produce identical results.

    Returns
    -------
    ndime : int
        Spatial dimension (2 or 3).
    pts : numpy.ndarray, shape (NPOIN, 3)
        Node coordinates. For 2D meshes the third column is zero.
    markers : dict[str, list[list[int]]]
        Mapping ``MARKER_TAG -> list of boundary elements``, where each
        element is ``[etype, n0, n1, ...]`` using global node ids.
    """
    mode = (mode or "auto").lower()
    if mode == "fast":
        return _read_su2_fast(path)
    if mode == "robust":
        return _read_su2_robust(path)
    if mode != "auto":
        raise ValueError(f"su2 read mode must be 'auto', 'fast' or 'robust', got {mode!r}")

    try:
        return _read_su2_fast(path)
    except _SU2FastPathError:
        return _read_su2_robust(path)


def _read_su2_fast(path: str) -> Tuple[int, np.ndarray, Dict[str, List[List[int]]]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        ndime = None
        npoin = None
        while npoin is None:
            line = f.readline()
            if line == "":
                raise _SU2FastPathError("EOF reached before NPOIN")
            s = _strip_comment(line)
            if not s:
                continue
            if s.startswith("NDIME"):
                ndime = int(s.split("=", 1)[1].strip())
            elif s.startswith("NELEM"):
                nelem = int(s.split("=", 1)[1].strip())
                for _ in range(nelem):
                    f.readline()
            elif s.startswith("NPOIN"):
                npoin = int(s.split("=", 1)[1].strip())

        if ndime is None:
            raise _SU2FastPathError("NDIME not found before NPOIN")

        ncol = ndime + 1
        rest = f.read()

    nnum = npoin * ncol
    arr = np.fromstring(rest, sep=" ", count=nnum)
    if arr.size != nnum:
        raise _SU2FastPathError("coordinate block was not cleanly numeric")

    pts = np.zeros((npoin, 3), dtype=float)
    pts[:, :ndime] = arr.reshape(npoin, ncol)[:, :ndime]

    midx = rest.find("NMARK")
    if midx < 0:
        raise _SU2FastPathError("NMARK not found after coordinate block")

    lineiter = iter(rest[midx:].splitlines())

    def next_data() -> str:
        for raw in lineiter:
            s = _strip_comment(raw)
            if s:
                return s
        raise EOFError("Unexpected EOF while parsing SU2 markers.")

    markers = _parse_markers(next_data)
    return ndime, pts, markers


def _read_su2_robust(path: str) -> Tuple[int, np.ndarray, Dict[str, List[List[int]]]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        ndime = None
        while True:
            line = f.readline()
            if line == "":
                raise EOFError("Unexpected EOF while searching NDIME.")
            s = _strip_comment(line)
            if s.startswith("NDIME"):
                ndime = int(s.split("=", 1)[1].strip())
                break

        npoin = None
        while True:
            line = f.readline()
            if line == "":
                raise EOFError("Unexpected EOF while searching NPOIN.")
            s = _strip_comment(line)
            if s.startswith("NPOIN"):
                npoin = int(s.split("=", 1)[1].strip())
                break

        pts = _read_npoin_block_fast(f, npoin=npoin, ndime=ndime)

        it = iter(f)

        def next_data() -> str:
            return _next_data_line(it)

        markers = _parse_markers(next_data)

    return ndime, pts, markers



def stream_write_su2_with_new_points(in_path: str, out_path: str, ndime: int, new_pts: np.ndarray) -> None:
    """Write a deformed SU2 mesh by replacing only the NPOIN coordinate block.

    The element/marker sections and all formatting outside the coordinate
    block are streamed through unchanged, so the output differs from the
    input only in the node coordinates. Point ids are re-emitted in their
    canonical sequential ordering (``0 .. NPOIN-1``), matching the SU2
    convention.

    Parameters
    ----------
    in_path : str
        Original SU2 mesh (template for everything except coordinates).
    out_path : str
        Output path.
    ndime : int
        Spatial dimension (2 or 3).
    new_pts : numpy.ndarray, shape (NPOIN, >=ndime)
        Deformed node coordinates in the original node ordering.
    """
    chunk = IO_CHUNK_LINES

    with open(in_path, "r", encoding="utf-8", errors="ignore") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        it = iter(fin)
        for raw in it:
            stripped = _strip_comment(raw)

            if not stripped:
                fout.write(raw)
                continue

            if stripped.startswith("NPOIN"):
                fout.write(raw)
                npoin = int(stripped.split("=", 1)[1].strip())

                for start in range(0, npoin, chunk):
                    end = min(start + chunk, npoin)

                    for _ in range(end - start):
                        next(it)

                    ids = np.arange(start, end, dtype=np.int64)
                    if ndime == 2:
                        out = np.column_stack((new_pts[start:end, 0], new_pts[start:end, 1], ids))
                        np.savetxt(fout, out, fmt="%.16e\t%.16e\t%d")
                    else:
                        out = np.column_stack(
                            (new_pts[start:end, 0], new_pts[start:end, 1], new_pts[start:end, 2], ids)
                        )
                        np.savetxt(fout, out, fmt="%.16e\t%.16e\t%.16e\t%d")
                continue

            fout.write(raw)


# ---------------------------------------------------------------------------
# Restart / solution file update
# ---------------------------------------------------------------------------
def _detect_csv_delimiter(sample_line: str) -> str:
    """Best-effort delimiter detection for SU2 ASCII restart/solution files."""
    if "," in sample_line:
        return ","
    return " "


def _parse_header_fields(line: str, delim: str) -> Tuple[List[str], bool]:
    """Return ``(field_names, quoted)`` with names stripped and de-quoted."""
    if delim == ",":
        raw = [x.strip() for x in line.strip().split(",")]
    else:
        raw = line.strip().split()

    quoted = any((len(x) >= 2 and x[0] == '"' and x[-1] == '"') for x in raw)
    names = [x.strip().strip('"') for x in raw]
    return names, quoted


def update_su2_restart(
    restart_in: str,
    restart_out: str,
    ndime: int,
    new_pts: np.ndarray,
) -> None:
    """Stream-update an ASCII SU2 restart/solution file with deformed coords.

    This is a "no-interpolation" update: it assumes the PointID ordering is
    consistent, i.e. the field values stay attached to the same PointID while
    the x/y(/z) columns are overwritten with the deformed coordinates. If the
    restart file has a PointID column but no coordinate columns, x/y(/z) are
    inserted right after PointID.

    Parameters
    ----------
    restart_in : str
        Input ASCII restart/solution file (typically a ``.csv``).
    restart_out : str
        Output path.
    ndime : int
        Spatial dimension (2 or 3).
    new_pts : numpy.ndarray, shape (N, >=2)
        Deformed node coordinates indexed by PointID.
    """
    if ndime not in (2, 3):
        raise ValueError(f"ndime must be 2 or 3, got {ndime}")
    if new_pts.ndim != 2 or new_pts.shape[1] < 2:
        raise ValueError("new_pts must be (N,3) or (N,>=2)")

    npts = int(new_pts.shape[0])

    def fmt(v: float) -> str:
        return f"{v:.16e}"

    preamble: List[str] = []
    with open(restart_in, "r", encoding="utf-8", errors="ignore", newline="") as fin:
        header = ""
        while True:
            line = fin.readline()
            if line == "":
                raise EOFError("Empty restart file (no header found).")
            if not line.strip():
                preamble.append(line)
                continue
            header = line.rstrip("\n")
            break

    delim = _detect_csv_delimiter(header)
    names, quoted = _parse_header_fields(header, delim)
    names_lc = [n.strip().lower() for n in names]

    def find_col(key: str) -> Optional[int]:
        key = key.lower()
        for i, n in enumerate(names_lc):
            if n == key:
                return i
        return None

    pid_i = find_col("pointid")
    if pid_i is None:
        raise RuntimeError("Restart header must contain a PointID column.")

    x_i = find_col("x")
    y_i = find_col("y")
    z_i = find_col("z")
    has_xyz = (x_i is not None) and (y_i is not None) and (ndime == 2 or z_i is not None)
    insert_xyz = not has_xyz

    old_nfields = len(names)

    if insert_xyz:
        new_names = list(names)
        ins_pos = pid_i + 1
        new_names[ins_pos:ins_pos] = ["x", "y"] + (["z"] if ndime == 3 else [])
        names = new_names
        names_lc = [n.strip().lower() for n in names]
        pid_i = names_lc.index("pointid")
        x_i = names_lc.index("x")
        y_i = names_lc.index("y")
        z_i = names_lc.index("z") if ndime == 3 else None

    def render_header(fields: List[str]) -> str:
        f = [f'"{x}"' for x in fields] if quoted else fields
        if delim == ",":
            return ", ".join(f)
        return " ".join(f)

    new_nfields = len(names)

    with open(restart_in, "r", encoding="utf-8", errors="ignore", newline="") as fin, open(
        restart_out, "w", encoding="utf-8", newline=""
    ) as fout:
        for ln in preamble:
            fout.write(ln)
        fout.write(render_header(names) + "\n")

        for _ in range(len(preamble) + 1):
            fin.readline()

        line_no = 0
        for raw in fin:
            if not raw.strip():
                fout.write(raw)
                continue

            if delim == ",":
                parts = [x.strip() for x in raw.rstrip("\n").split(",")]
            else:
                parts = raw.rstrip("\n").split()

            if len(parts) < 2:
                fout.write(raw)
                continue

            try:
                pid = int(float(parts[pid_i]))
            except Exception as e:
                raise RuntimeError(f"Failed to parse PointID at data line {line_no + 1}: {e}")

            if pid < 0 or pid >= npts:
                raise RuntimeError(f"PointID {pid} out of range [0,{npts - 1}] at data line {line_no + 1}.")

            if insert_xyz and len(parts) == old_nfields:
                ins_pos = pid_i + 1
                parts[ins_pos:ins_pos] = ["0", "0"] + (["0"] if ndime == 3 else [])

            if len(parts) < new_nfields:
                parts = parts + [""] * (new_nfields - len(parts))

            parts[x_i] = fmt(float(new_pts[pid, 0]))
            parts[y_i] = fmt(float(new_pts[pid, 1]))
            if ndime == 3 and z_i is not None:
                parts[z_i] = fmt(float(new_pts[pid, 2]))

            if delim == ",":
                fout.write(", ".join(parts) + "\n")
            else:
                fout.write(" ".join(parts) + "\n")

            line_no += 1
