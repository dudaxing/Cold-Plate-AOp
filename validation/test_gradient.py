"""The implicit-AD sensitivity against central finite differences.

This is the paper's headline claim: differentiating through a converged
Newton-Raphson solve via the implicit function theorem gives the exact gradient
at a cost independent of the iteration count. If this test fails, every
optimisation result built on the framework is suspect.
"""

import numpy as np
import jax
import jax.numpy as jnp

from validation import harness

LENGTH, HEIGHT, UMAX, MU, RHO = 2.0, 1.0, 1.0, 1.0, 1.0
NX, NY = 24, 12
RAMP_PENALTY = 1.0
FD_STEP = 1e-5
NUM_PROBES = 12


def _objective_fn():
    mesh = harness.channel_mesh(NX, NY, LENGTH, HEIGHT)
    tol = 0.25 * min(LENGTH / NX, HEIGHT / NY)
    bc, _ = harness.poiseuille_bcs(mesh, LENGTH, HEIGHT, UMAX, tol)
    material = harness.tf_material.FluidMaterial(
        mass_density=RHO, dynamic_viscosity=MU
    )
    solver = harness.tf_fluid.FluidSolver(
        mesh, bc, material, harness.SOLVER_SETTINGS
    )
    a_min, a_max = harness.brinkman_bounds(MU, LENGTH)
    extent = harness.tf_utils.Extent(min=a_min, max=a_max)
    x0 = jnp.zeros((mesh.num_dofs,)).at[bc["fixed_dofs"]].set(bc["dirichlet_values"])

    def objective(gamma):
        alpha = harness.tf_material.compute_ramp_interpolation(
            gamma, RAMP_PENALTY, extent, mode="convex"
        )
        press_vel = harness.tf_solver.modified_newton_raphson_solve(solver, x0, alpha)
        return jnp.sum(
            jax.vmap(solver.compute_elem_dissipated_power)(
                alpha, press_vel[mesh.elem_dof_mat], mesh.elem_node_coords
            )
        )

    return objective, mesh.num_elems


def test_ad_matches_central_differences():
    objective, num_elems = _objective_fn()
    rng = np.random.default_rng(0)
    gamma = jnp.array(rng.uniform(0.2, 0.8, size=(num_elems,)))

    grad_ad = jax.grad(objective)(gamma)

    worst = 0.0
    for i in rng.choice(num_elems, size=NUM_PROBES, replace=False):
        plus = float(objective(gamma.at[i].add(FD_STEP)))
        minus = float(objective(gamma.at[i].add(-FD_STEP)))
        fd = (plus - minus) / (2 * FD_STEP)
        rel = abs(float(grad_ad[i]) - fd) / max(abs(fd), 1e-30)
        worst = max(worst, rel)

    # Central differences at h = 1e-5 bottom out around 1e-9; anything above
    # 1e-6 means the sensitivity itself is wrong, not the finite difference.
    assert worst < 1e-6, f"max relative AD/FD difference {worst:.3e}"
