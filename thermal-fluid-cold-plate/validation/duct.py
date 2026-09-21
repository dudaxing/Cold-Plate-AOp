"""3D plane-Poiseuille duct harness, shared by the stabilisation tests.

Box L x W x H. The two y faces are symmetry planes (u_y = 0) so the flow is
effectively plane; the z faces are no-slip walls; the inlet carries the analytic
parabola; the outlet is left free.

The outlet is a zero external pressure traction, which is NOT consistent with
holding the analytic profile up to the outlet plane -- fully developed flow has
outlet traction (-p, mu U_y, mu U_z) and the tangential part does not vanish.
The comparison is therefore made on an interior plane. The measured error is
uniform along the duct (within 10% between x = 10%, 50% and 90% of the length),
which is what rules the outlet out as the source of the error these tests are
about.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import toflux.src.bc as tbc
import toflux.src.solver as tsolver

from tfopus import elements as E
from tfopus import fe_flow
from tfopus import mesh as M

L, W, H = 0.02, 0.004, 0.001
RHO, MU = 1000.0, 0.001

SETTINGS = {
    "linear": {"solver": tsolver.LinearSolvers.SCIPY_SPARSE, "rtol": 1e-12},
    "nonlinear": {"max_iter": 30, "threshold": 1e-10},
}


def exact_dpdx(umax: float) -> float:
    return -8.0 * MU * umax / H**2


def _face_fn(axis, side, phys, param):
    if axis == 0:
        return M.Face.INLET if side == 0 else M.Face.OUTLET
    return M.Face.SYMMETRY if axis == 1 else M.Face.WALL


def solve_duct(nx: int, ny: int, nz: int, umax: float, h_mode="min_edge", tau_scale=1.0):
    """Returns (relative velocity error on the mid plane, fitted dp/dx)."""
    shell = M.build_shell_mesh(
        lambda x, y, z: np.stack([x, y, z * H], axis=1),
        np.linspace(0, L, nx + 1),
        np.linspace(0, W, ny + 1),
        nz,
        4,
        lambda a, b: np.zeros_like(a, dtype=int),
        _face_fn,
    )
    mesh = shell.mesh
    coords = np.asarray(mesh.nodes.coords)
    u_exact = lambda z: umax * 4.0 * z * (H - z) / H**2  # noqa: E731

    conn = np.asarray(mesh.elem_template.face_connectivity)
    elem_nodes = np.asarray(mesh.elem_nodes)
    nodes_of = lambda tag: np.unique(  # noqa: E731
        np.concatenate([elem_nodes[e][conn[f]] for e, f in shell.elem_faces[tag]])
    )

    vals = {}
    for n in nodes_of(M.Face.SYMMETRY):
        vals[4 * n + 2] = 0.0
    for n in nodes_of(M.Face.WALL):
        vals[4 * n + 1] = vals[4 * n + 2] = vals[4 * n + 3] = 0.0
    for n in nodes_of(M.Face.INLET):  # inlet wins on the rim
        vals[4 * n + 1] = float(u_exact(coords[n, 2]))
        vals[4 * n + 2] = vals[4 * n + 3] = 0.0

    fixed = np.array(sorted(vals), dtype=int)
    assert not (fixed % 4 == 0).any(), "a pressure dof got constrained"
    bc = tbc.BCDict(
        elem_forces=jnp.zeros((mesh.num_elems, mesh.num_dofs_per_elem)),
        fixed_dofs=fixed,
        free_dofs=np.setdiff1d(np.arange(mesh.num_dofs), fixed),
        dirichlet_values=np.array([vals[d] for d in fixed]),
    )

    solver = fe_flow.FlowSolver(
        mesh, bc, RHO, MU, SETTINGS,
        elem_length=E.element_lengths(mesh, h_mode),
        tau_scale=tau_scale,
    )
    x0 = jnp.zeros((mesh.num_dofs,)).at[fixed].set(bc["dirichlet_values"])
    pv = np.asarray(
        tsolver.modified_newton_raphson_solve(solver, x0, jnp.zeros((mesh.num_elems,)))
    )
    p, ux = pv[0::4], pv[1::4]

    xs = np.unique(coords[:, 0])
    plane = np.abs(coords[:, 0] - xs[len(xs) // 2]) < 1e-12
    ref = u_exact(coords[plane, 2])
    err = np.linalg.norm(ux[plane] - ref) / np.linalg.norm(ref)

    interior = (
        (np.abs(coords[:, 2] - H / 2) < 1e-12)
        & (coords[:, 0] > 0.2 * L)
        & (coords[:, 0] < 0.8 * L)
    )
    dpdx = float(np.polyfit(coords[interior, 0], p[interior], 1)[0])
    return err, dpdx
