"""Uniform mesh refinement carrying the SAME design, for fixed-design studies.

The point of a mesh check is to change one thing. So the design field is
transferred, never rebuilt: each fine element inherits the physical solid
fraction of the coarse element containing it. No re-filtering, no re-projection,
no re-thresholding on the fine mesh.

That last one matters. A binary design must be thresholded on the PARENT mesh
and then replicated, not thresholded after interpolation -- otherwise the fine
result differs from the coarse one both by discretisation and by a different
polygonal boundary, and the two cannot be separated.

Cell lookup uses `floor(centre / h)`. Centres sit at (i + 1/2) h, so
`round` does banker's rounding on every key and collapses i with i+1 for odd i
-- the defect that made the connectivity diagnostic report a connected design
as 65 components.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import jax.numpy as jnp

from tfopus import zhao2d as _z


def refine_spec(spec: _z.Zhao2DSpec, factor: int = 2) -> _z.Zhao2DSpec:
    """Same geometry and physics, element size divided by `factor`."""
    if factor < 1 or factor != int(factor):
        raise ValueError(f"factor must be a positive integer, got {factor}")
    return dataclasses.replace(spec, element_size=spec.element_size / factor)


def _cell_index(planar: _z.PlanarMesh) -> tuple[dict, float]:
    centres = np.asarray(planar.elem_centres)
    h = float(np.sqrt(np.asarray(planar.elem_area)[0]))
    index = {
        (int(np.floor(x / h)), int(np.floor(y / h))): i
        for i, (x, y) in enumerate(centres)
    }
    if len(index) != len(centres):
        raise RuntimeError("cell index collided; see the floor/round note above")
    return index, h


def parent_of_each_fine_element(
    coarse: _z.PlanarMesh, fine: _z.PlanarMesh
) -> np.ndarray:
    """(fine.num_elems,) index of the coarse element containing each fine one."""
    index, h_coarse = _cell_index(coarse)
    fine_centres = np.asarray(fine.elem_centres)
    parents = np.empty(fine.num_elems, dtype=int)
    for k, (x, y) in enumerate(fine_centres):
        key = (int(np.floor(x / h_coarse)), int(np.floor(y / h_coarse)))
        if key not in index:
            raise RuntimeError(
                f"fine element {k} at {(x, y)} has no parent coarse cell; the "
                "two meshes do not describe the same domain"
            )
        parents[k] = index[key]
    return parents


def refine_design(coarse: _z.PlanarMesh, fine: _z.PlanarMesh, s_coarse) -> jnp.ndarray:
    """Physical solid fraction transferred onto the fine mesh, value for value."""
    parents = parent_of_each_fine_element(coarse, fine)
    return jnp.asarray(np.asarray(s_coarse)[parents])


def check_transfer(
    coarse: _z.PlanarMesh, fine: _z.PlanarMesh, s_coarse, s_fine, rtol: float = 1e-12
) -> dict:
    """Everything that must be invariant under a pure refinement.

    Raises if any of it is not, rather than reporting a difference that would
    then have to be disentangled from the discretisation effect being measured.
    """
    parents = parent_of_each_fine_element(coarse, fine)
    ratio = round((fine.num_elems / coarse.num_elems) ** 0.5)

    areas_c = np.asarray(coarse.elem_area)
    areas_f = np.asarray(fine.elem_area)
    counts = np.bincount(parents, minlength=coarse.num_elems)

    report = {
        "coarse_elements": int(coarse.num_elems),
        "fine_elements": int(fine.num_elems),
        "children_per_parent": sorted(set(counts.tolist())),
        "total_area_coarse": float(areas_c.sum()),
        "total_area_fine": float(areas_f.sum()),
        "design_cells_coarse": int(coarse.design_mask.sum()),
        "design_cells_fine": int(fine.design_mask.sum()),
    }

    if report["children_per_parent"] != [ratio * ratio]:
        raise RuntimeError(
            f"each coarse cell should have {ratio * ratio} children, got "
            f"{report['children_per_parent']}"
        )
    if not np.isclose(areas_c.sum(), areas_f.sum(), rtol=rtol):
        raise RuntimeError(
            f"domain area changed: {areas_c.sum()} -> {areas_f.sum()}"
        )
    # the design/tab partition must transfer exactly
    if not np.array_equal(fine.design_mask, coarse.design_mask[parents]):
        raise RuntimeError("the design/tab partition is not preserved by refinement")
    # and the tabs must still be pure fluid
    if not np.allclose(np.asarray(s_fine)[~fine.design_mask], 0.0):
        raise RuntimeError("tabs picked up material during transfer")

    fc = _z.fluid_fractions(coarse, s_coarse)
    ff = _z.fluid_fractions(fine, s_fine)
    for key in fc:
        if not np.isclose(fc[key], ff[key], rtol=rtol):
            raise RuntimeError(f"{key} changed: {fc[key]} -> {ff[key]}")
    report.update({f"coarse_{k}": v for k, v in fc.items()})
    report.update({f"fine_{k}": v for k, v in ff.items()})
    return report


def threshold_design(planar: _z.PlanarMesh, s, level: float = 0.5) -> jnp.ndarray:
    """Solid where s >= level; the fixed tabs stay fluid by definition."""
    out = (np.asarray(s) >= level).astype(float)
    out[~planar.design_mask] = 0.0
    return jnp.asarray(out)


def undershoot(temperature, inlet_temperature: float, tol: float = 1e-10) -> dict:
    """Where and how far the temperature falls below the inlet value.

    With a positive source everywhere and the inlet as the only cold boundary,
    the continuous problem admits no temperature below the inlet value, so any
    such node is a discrete undershoot -- a statement about the discretisation,
    not about cooling.
    """
    t = np.asarray(temperature)
    below = t < inlet_temperature - tol
    return {
        "nodes_below_inlet": int(below.sum()),
        "fraction_below_inlet": float(below.mean()),
        "min_temperature": float(t.min()),
        "undershoot_depth": float(inlet_temperature - t.min()),
    }
