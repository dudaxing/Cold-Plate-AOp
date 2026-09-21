"""Plane-channel Navier-Stokes-Brinkman problems built on upstream TOFLUX.

A plane channel is the smallest setup that has a closed-form solution (Poiseuille)
and can be turned into a genuinely two-dimensional flow by blocking part of it
with the Brinkman term. Every test in this suite is built from these helpers.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import toflux.src.bc as tf_bc
import toflux.src.fe_fluid as tf_fluid
import toflux.src.material as tf_material
import toflux.src.mesher as tf_mesher
import toflux.src.solver as tf_solver
import toflux.src.utils as tf_utils

Field = tf_fluid.FluidField

# scipy's sparse direct solve keeps the suite free of PETSc and PARDISO; the
# tolerances below assume a direct solve, not an iterative one.
SOLVER_SETTINGS = {
    "linear": {"solver": tf_solver.LinearSolvers.SCIPY_SPARSE, "rtol": 1e-10},
    "nonlinear": {"max_iter": 40, "threshold": 1e-11},
}


def channel_mesh(nx: int, ny: int, length: float, height: float):
    """A structured Q1 grid over [0, length] x [0, height], 3 dofs/node (p, u, v)."""
    box = tf_mesher.BoundingBox(
        x=tf_utils.Extent(min=0.0, max=length),
        y=tf_utils.Extent(min=0.0, max=height),
    )
    return tf_mesher.GridMesh(
        nel=(nx, ny), bounding_box=box, dofs_per_node=3, gauss_order=2
    )


def _boundary_faces(mesh, predicate):
    """(elem, face) pairs on the mesh boundary whose node coords satisfy predicate."""
    face_conn = np.asarray(mesh.elem_template.face_connectivity)
    elems, faces = np.nonzero(mesh.boundary_faces)
    out = []
    for elem, face in zip(elems, faces):
        coords = np.asarray(mesh.elem_node_coords[elem])[face_conn[face]]
        if predicate(coords):
            out.append((int(elem), int(face)))
    return out


def _per_node_values(mesh, elem_faces, fn):
    """Per-face 2-vectors evaluated at that face's own nodes.

    DirichletBC broadcasts `value[i] * ones(nodes_per_face)`, so handing it a
    2-vector per face assigns each node its own value. Upstream's notebooks use
    the same trick to impose a parabolic inlet.
    """
    face_conn = np.asarray(mesh.elem_template.face_connectivity)
    vals = []
    for elem, face in elem_faces:
        coords = np.asarray(mesh.elem_node_coords[elem])[face_conn[face]]
        vals.append([fn(coords[0]), fn(coords[1])])
    return jnp.array(vals)


def poiseuille_exact(height: float, umax: float):
    """u(y) = umax * 4y(H - y) / H^2, the parabola with peak umax at y = H/2."""
    return lambda xy: umax * 4.0 * xy[1] * (height - xy[1]) / height**2


def poiseuille_bcs(mesh, length: float, height: float, umax: float, tol: float):
    """Parabolic inlet, no-slip walls, zero-pressure outlet."""
    u_exact = poiseuille_exact(height, umax)
    inlet = _boundary_faces(mesh, lambda c: np.all(np.abs(c[:, 0]) < tol))
    outlet = _boundary_faces(mesh, lambda c: np.all(np.abs(c[:, 0] - length) < tol))
    walls = _boundary_faces(
        mesh,
        lambda c: np.all(np.abs(c[:, 1]) < tol)
        or np.all(np.abs(c[:, 1] - height) < tol),
    )
    bcs = [
        tf_bc.DirichletBC(
            walls,
            [(Field.U_VEL, np.zeros(len(walls))), (Field.V_VEL, np.zeros(len(walls)))],
            "wall",
        ),
        tf_bc.DirichletBC(
            inlet,
            [
                (Field.U_VEL, _per_node_values(mesh, inlet, u_exact)),
                (Field.V_VEL, jnp.zeros((len(inlet), 2))),
            ],
            "inlet",
        ),
        tf_bc.DirichletBC(
            outlet,
            [
                (Field.V_VEL, np.zeros(len(outlet))),
                (Field.PRESSURE, np.zeros(len(outlet))),
            ],
            "outlet",
        ),
    ]
    return tf_bc.process_boundary_conditions(bcs, mesh), u_exact


def solve(solver, bc, brinkman_penalty):
    """Damped Newton-Raphson from an initial guess that already satisfies the BCs."""
    x0 = jnp.zeros((solver.mesh.num_dofs,)).at[bc["fixed_dofs"]].set(
        bc["dirichlet_values"]
    )
    return tf_solver.modified_newton_raphson_solve(solver, x0, brinkman_penalty)


def split_fields(press_vel):
    """(p, u, v) at nodes, from the interleaved (p1, u1, v1, p2, ...) dof vector."""
    return press_vel[0::3], press_vel[1::3], press_vel[2::3]


def dissipated_power(solver, mesh, brinkman_penalty, press_vel) -> float:
    return float(
        jnp.sum(
            jax.vmap(solver.compute_elem_dissipated_power)(
                brinkman_penalty, press_vel[mesh.elem_dof_mat], mesh.elem_node_coords
            )
        )
    )


def brinkman_bounds(dynamic_viscosity: float, length: float):
    """Upstream's thickness-bracketed inverse-permeability range."""
    return (
        tf_material.brinkman_bound(dynamic_viscosity, 100.0 * length),
        tf_material.brinkman_bound(dynamic_viscosity, 1.0e-2 * length),
    )
