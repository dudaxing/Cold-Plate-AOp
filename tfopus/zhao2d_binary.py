"""Thresholded designs, and the acceptance of post-processed states.

Fixed-design checks compare designs after thresholding. The rules they apply
live here, so that every check applies them the same way and each can be
tested without solving anything:

- `threshold`: solid where the solid fraction s >= t, the fixed tabs fluid by
  definition; no repair of any kind.
- `gate_passed`: a state is accepted only if EVERY relative residual is a
  finite number no larger than the tolerance. A NaN must fail -- not slip past
  `max()`, which returns the finite entry when the NaN comes second.
- `cell_usable`: a cell counts only if it was analysed, passed the gate and,
  if it is an anchor, reproduced its record.
- `comparison_kind`: two usable cells form a `qualified_comparison` only if
  both designs satisfy the fluid-fraction bound. Otherwise the numbers are a
  `threshold_diagnostic` of a post-processing rule, not a ranking of designs.
  PDE acceptance and design qualification are recorded separately.
"""

from __future__ import annotations

import numpy as np

from tfopus import zhao2d as _z
from tfopus.mesh import Face

# Area-weighted fractions of equal cells need not sum to the bound exactly:
# 2000 of 5000 cells can come out an ulp above 0.4.
VOLUME_RTOL = 1e-12


def threshold(s, design_mask, t: float) -> np.ndarray:
    """1 (solid) where s >= t, 0 elsewhere; the non-design tabs always 0."""
    s_bin = (np.asarray(s) >= t).astype(float)
    s_bin[~np.asarray(design_mask, dtype=bool)] = 0.0
    return s_bin


def volume_threshold(s, design_mask, areas, max_fluid_fraction: float) -> dict:
    """R1l's export rule: one threshold t for the whole design, no repair.

    Fluid is s < t. t is chosen so that the design domain's fluid volume is
    at most `max_fluid_fraction` of it and as close to that as possible, cells
    of equal s moving together (t sits exactly at a value of s, so every cell
    at that value is solid). The tabs stay fluid. Returns t, the thresholded
    s and what it changed against s = 0.5. This t is an export threshold, not
    a projection's eta.
    """
    s = np.asarray(s, dtype=float)
    design = np.asarray(design_mask, dtype=bool)
    a = np.asarray(areas, dtype=float)[design]
    sd = s[design]
    budget = max_fluid_fraction * a.sum() * (1.0 + VOLUME_RTOL)
    values = np.unique(sd)                                   # ascending
    below = np.concatenate([[0.0], np.cumsum(np.bincount(
        np.searchsorted(values, sd), weights=a, minlength=len(values)))])
    k = int(np.flatnonzero(below <= budget).max())           # fluid area if t = values[k]
    t = float(values[k]) if k < len(values) else float(np.nextafter(values[-1], np.inf))
    s_bin = threshold(s, design, t)
    fluid = design & (s_bin < 0.5)
    return {
        "t": t,
        "s_binary": s_bin,
        "fluid_cells": int(fluid.sum()),
        "fluid_fraction": float(a[(s_bin < 0.5)[design]].sum() / a.sum()),
        "cells_changed_from_0.5": int(np.count_nonzero(design & ((s < t) != (s < 0.5)))),
    }


def port_elements(planar, tag) -> np.ndarray:
    return np.unique([e for e, _ in planar.elem_faces[tag]])


def connectivity(planar, s, thresholds) -> dict:
    """Fluid components and whether inlet and outlet join, per threshold probed."""
    inlet = port_elements(planar, Face.INLET)
    outlet = port_elements(planar, Face.OUTLET)
    design = np.asarray(planar.design_mask, dtype=bool)
    out = {}
    for t in thresholds:
        fluid = threshold(s, design, t) < 0.5
        n, sizes, joined, _, _ = _z.fluid_connectivity(planar, fluid, inlet, outlet)
        out[f"{t:g}"] = {"fluid_components": int(n), "inlet_outlet_connected": bool(joined),
                         "largest_components": [int(v) for v in np.sort(sizes)[::-1][:3]]}
    return out


def gate_passed(norms: dict, tol: float) -> bool:
    """True only if there are residuals and every one is finite and <= tol."""
    values = [float(v) for v in norms.values()]
    return bool(values) and all(np.isfinite(v) and v <= tol for v in values)


def volume_feasible(v_f_design: float, bound: float, rtol: float = VOLUME_RTOL) -> bool:
    """The fluid-fraction constraint v_f <= bound, to rounding."""
    v = float(v_f_design)
    return bool(np.isfinite(v) and v <= bound * (1.0 + rtol))


def anchor_status(values: dict, anchor: dict, tol: float) -> dict:
    """Relative differences of Psi and C from an anchor record, and the verdict."""
    out = {"source": anchor.get("source")}
    rel = {q: values[q] / anchor[q] - 1.0 for q in ("psi", "compliance")}
    out.update({f"{q}_relative": r for q, r in rel.items()})
    out["reproduced"] = all(np.isfinite(r) and abs(r) <= tol for r in rel.values())
    return out


def cell_usable(cell: dict | None) -> bool:
    if not cell or not cell.get("analysed") or not cell.get("gate_passed"):
        return False
    anchor = cell.get("anchor")
    return anchor is None or bool(anchor.get("reproduced"))


def comparison_kind(a: dict | None, b: dict | None) -> str | None:
    """'qualified_comparison', 'threshold_diagnostic', or None if either is unusable."""
    if not (cell_usable(a) and cell_usable(b)):
        return None
    if a.get("volume_feasible") and b.get("volume_feasible"):
        return "qualified_comparison"
    return "threshold_diagnostic"


def check_failures(cells: dict) -> dict:
    """Anchors that did not reproduce and analysed cells that failed the gate."""
    return {
        "anchors_not_reproduced": [k for k, c in cells.items()
                                   if "anchor" in c and not c["anchor"].get("reproduced")],
        "cells_failing_gate": [k for k, c in cells.items()
                               if c.get("analysed") and not c.get("gate_passed")],
    }
