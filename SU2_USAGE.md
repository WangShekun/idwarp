# IDWarp SU2 Usage Guide

This branch adds native `fileType="SU2"` support to IDWarp. SU2 ASCII meshes can be read, warped, and written back through `USMesh` without a separate `SU2USMesh` subclass.

## Scope

- SU2 mesh reader and writer helpers live in `idwarp/su2io.py`
- `USMesh` integration lives in `idwarp/UnstructuredMesh.py`
- The Fortran core and compiled extension are unchanged

## Installation

The SU2 changes are Python-only, but a normal install is still the cleanest way to use this branch.

```bash
pip install --no-build-isolation .
```

If IDWarp is already installed on the target machine, you can also update only the Python package files after rebuilding or reinstalling is not convenient.

## Quick Start

```python
from mpi4py import MPI
from idwarp import USMesh

opts = {
    "gridFile": "mesh.su2",
    "fileType": "SU2",
    "su2ReadMode": "auto",
    "specifiedSurfaces": ["wall"],
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
    "symmetrySurfaces": None,
    "su2RestartFile": "restart_flow.csv",
}

mesh = USMesh(options=opts, comm=MPI.COMM_WORLD)

xs = mesh.getSurfaceCoordinates()
# Modify xs with your geometry or FFD workflow.
mesh.setSurfaceCoordinates(xs)
mesh.warpMesh()
mesh.writeGrid("mesh_deformed.su2")
mesh.writeSU2Restart("restart_warped.csv")
```

## SU2-Specific Options

| Option | Purpose |
| --- | --- |
| `fileType` | Must be set to `"SU2"`. |
| `su2ReadMode` | Reader mode: `auto`, `fast`, or `robust`. |
| `specifiedSurfaces` | SU2 `MARKER_TAG` names to use as the warping surface. |
| `symmetryPlanes` | Explicit point-normal symmetry definition. |
| `symmetrySurfaces` | SU2 `MARKER_TAG` names used to derive symmetry planes. |
| `su2RestartFile` | Input SU2 restart/solution file used by `writeSU2Restart`. |

All standard IDWarp warping parameters such as `aExp`, `bExp`, `LdefFact`, `alpha`, `errTol`, `evalMode`, `useRotations`, and `bucketSize` still apply.

## Symmetry Behavior

SU2 markers do not include a boundary-condition type, so IDWarp cannot automatically detect symmetry boundaries.

- Use `symmetryPlanes` when you already know the symmetry plane.
- Use `symmetrySurfaces` when you want IDWarp to derive a plane from one or more marker surfaces.
- If both are set, `symmetryPlanes` takes precedence.

## Read Modes

`su2ReadMode` controls how the SU2 mesh is parsed.

- `auto`: tries the fast parser first, then falls back to the robust parser
- `fast`: uses only the vectorized fast path and errors on non-canonical layout
- `robust`: uses the more tolerant line-by-line parser

For unknown meshes, `auto` is the safest default.

## Restart Update

`writeSU2Restart(restartOut)` updates the coordinates in an ASCII SU2 restart or solution file while keeping the field values attached to the same `PointID`.

- The input file comes from the `su2RestartFile` option.
- The output path is the `restartOut` argument.
- This is separate from IDWarp's `restartFile` option, which is the internal coefficient cache.

## Notes

- This support targets SU2 ASCII mesh files.
- The output mesh rewrites only the node coordinate block; the rest of the file is streamed through unchanged.
- Explicitly setting `specifiedSurfaces` is recommended instead of relying on all markers.
