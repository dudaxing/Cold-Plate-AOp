"""Separating what a mesh refinement actually changes in the thermal solution.

Refining changes three things at once: the temperature approximation space, the
stabilisation parameter (tau shrinks with h), and -- if the flow is re-solved --
the velocity driving the convection. A single coarse-to-fine comparison sums all
three and attributes none of them.

The four analyses below change one at a time, along a chosen path:

    A   thermal mesh h,   velocity u_h,          tau from the formula at h
    B   thermal mesh h/2, u_h extended exactly,  tau FROZEN at the parent value
    C   thermal mesh h/2, u_h extended exactly,  tau from the formula at h/2
    D   thermal mesh h/2, u_{h/2} re-solved,     tau from the formula at h/2

    A -> B   the temperature space alone
    B -> C   the stabilisation coefficient alone
    C -> D   the velocity input alone

B is a counterfactual for attribution, not a scheme anyone should run: its tau
is not what the formula gives on that mesh. The differences sum to the total
along THIS path; they are not a path-independent error budget, and calling them
one would overclaim.

The velocity extension is exact rather than interpolated onto a new solve: the
coarse Q1 field is evaluated at the fine nodes, so B and C see the same
continuous velocity function that A does, with the same boundary trace. Nothing
is re-imposed from boundary conditions.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from tfopus import zhao2d as _z


def _cell_lookup(planar: _z.PlanarMesh):
    centres = np.asarray(planar.elem_centres)
    h = float(np.sqrt(np.asarray(planar.elem_area)[0]))
    index = {
        (int(np.floor(x / h)), int(np.floor(y / h))): i
        for i, (x, y) in enumerate(centres)
    }
    if len(index) != len(centres):
        raise RuntimeError("cell index collided")
    return index, h, centres


def extend_velocity(
    coarse: _z.PlanarMesh, fine: _z.PlanarMesh, nodal_velocity_coarse
) -> np.ndarray:
    """Evaluate the coarse Q1 velocity field at the fine mesh's nodes.

    This is the same function, sampled more finely -- not a projection and not a
    re-solve. On nodes the two meshes share it reproduces the coarse values
    exactly, which `check_extension` asserts.
    """
    index, h, _ = _cell_lookup(coarse)
    coarse_nodes = np.asarray(coarse.mesh.elem_nodes)
    coarse_coords = np.asarray(coarse.mesh.nodes.coords)
    u_coarse = np.asarray(nodal_velocity_coarse)

    fine_coords = np.asarray(fine.mesh.nodes.coords)
    out = np.empty((fine_coords.shape[0], u_coarse.shape[1]))

    for n, (x, y) in enumerate(fine_coords):
        i0, j0 = int(np.floor(x / h)), int(np.floor(y / h))
        cell = None
        # a node on a cell boundary floors into either neighbour; by continuity
        # any containing cell gives the same value, so take the first that exists
        for di in (0, -1):
            for dj in (0, -1):
                if (i0 + di, j0 + dj) in index:
                    cand = index[(i0 + di, j0 + dj)]
                    nodes = coarse_coords[coarse_nodes[cand]]
                    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
                    if (x >= lo[0] - 1e-12 * h and x <= hi[0] + 1e-12 * h
                            and y >= lo[1] - 1e-12 * h and y <= hi[1] + 1e-12 * h):
                        cell = cand
                        break
            if cell is not None:
                break
        if cell is None:
            raise RuntimeError(f"fine node {n} at {(x, y)} is in no coarse cell")

        nodes = coarse_coords[coarse_nodes[cell]]
        lo = nodes.min(axis=0)
        xi = 2.0 * (x - lo[0]) / h - 1.0
        eta = 2.0 * (y - lo[1]) / h - 1.0
        shp = np.asarray(
            coarse.mesh.elem_template.shape_functions(jnp.array([xi, eta]))
        )
        out[n] = shp @ u_coarse[coarse_nodes[cell]]
    return out


def check_extension(
    coarse: _z.PlanarMesh, fine: _z.PlanarMesh, u_coarse, u_fine, tol: float = 1e-12
) -> dict:
    """The extension must reproduce the coarse field on every shared node."""
    cc = np.asarray(coarse.mesh.nodes.coords)
    fc = np.asarray(fine.mesh.nodes.coords)
    h = float(np.sqrt(np.asarray(fine.elem_area)[0]))

    key = {(round(x / h), round(y / h)): i for i, (x, y) in enumerate(fc)}
    matched, worst = 0, 0.0
    for i, (x, y) in enumerate(cc):
        k = (round(x / h), round(y / h))
        if k in key:
            matched += 1
            worst = max(
                worst,
                float(np.max(np.abs(np.asarray(u_fine)[key[k]] - np.asarray(u_coarse)[i]))),
            )
    if matched == 0:
        raise RuntimeError("no shared nodes found; the meshes are not nested")
    if worst > tol:
        raise RuntimeError(
            f"extension disagrees with the coarse field by {worst:g} on "
            f"{matched} shared nodes"
        )
    return {"shared_nodes": matched, "worst_shared_node_difference": worst}


def element_velocity_from_nodal(planar: _z.PlanarMesh, nodal_velocity):
    """(num_elems, nodes_per_elem * dim), the layout the thermal solver wants."""
    nodes = np.asarray(planar.mesh.elem_nodes)
    return jnp.asarray(np.asarray(nodal_velocity)[nodes].reshape(len(nodes), -1))


def element_tau(solver, elem_velocity, k) -> jnp.ndarray:
    """(num_elems,) tau as the solver's own formula produces it."""
    return jax.vmap(solver._tau)(
        jnp.asarray(elem_velocity).reshape(
            solver.mesh.num_elems, -1, solver.dim
        ),
        jnp.asarray(k),
        solver.elem_length,
    )


def freeze_tau_from_parent(parents: np.ndarray, tau_coarse) -> jnp.ndarray:
    """Parent tau copied to each child, for the B counterfactual."""
    return jnp.asarray(np.asarray(tau_coarse)[parents])


def peclet_statistics(
    planar: _z.PlanarMesh, elem_velocity, k, b_f: float, h: float, s=None
) -> dict:
    """Element Peclet number, with the mask and velocity point stated.

    Pe_e = b_f |u_c| h / (2 k), with u_c the velocity at the ELEMENT CENTRE.
    The fluid-mask median and the whole-domain median differ by four orders of
    magnitude here, so a bare "median Pe" is not a number anyone can check.
    """
    vel = np.asarray(elem_velocity).reshape(planar.num_elems, -1, 2)
    speed = np.linalg.norm(vel.mean(axis=1), axis=1)
    pe = b_f * speed * h / (2.0 * np.asarray(k))

    out = {"definition": "b_f * |u at element centre| * h / (2 k)", "h": h}
    masks = {"whole_domain": np.ones(planar.num_elems, dtype=bool)}
    if s is not None:
        fluid = np.asarray(s) < 0.5
        masks["fluid_s_lt_0.5_incl_tabs"] = fluid
        masks["fluid_s_lt_0.5_design_only"] = fluid & planar.design_mask
    for name, m in masks.items():
        if not m.any():
            continue
        out[name] = {
            "elements": int(m.sum()),
            "median": float(np.median(pe[m])),
            "p90": float(np.percentile(pe[m], 90)),
            "max": float(pe[m].max()),
            "fraction_above_1": float(np.mean(pe[m] > 1)),
        }
    return out
