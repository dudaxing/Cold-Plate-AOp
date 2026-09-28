"""The state of a problem whose residual is affine in its unknown, by one linear solve.

For a given flow and density the thermal residual is affine in T:

    R_T(T; u, s) = K_T(u, s) T - f_T(u, s)

tau depends on u, kappa and h, not on T (`ThermalSolver._tau`); the Galerkin
and SUPG terms are linear in grad T, plus the source; and the boundary values
are fixed. Upstream solves it with `modified_newton_raphson_solve` all the same.
Each of that loop's iterations assembles the residual and tangent three times
(the current point, a half step and a full step) and solves once. On an exactly
linear problem its damped step divides by a round-off quantity: the full step
leaves only round-off, so the fitted curvature is round-off too. At the
threshold 1e-11, every h/8 temperature of R1q and R1r ran to the 40-iteration
cap.

`affine_solve` takes the state in one step from `x0`, which carries the
Dirichlet values:

    x = x0 - K^{-1} R(x0)

For an affine residual that is the solution, up to the linear solve's round-off,
whatever x0's free values are. Its derivative is upstream's implicit-function
rule for its Newton solvers (`toflux.src.solver.solve_jvp`), unchanged and taken
at the state this returns: dx = -K^{-1} (dR/dp) dp, through the same `solve` and
its transpose solve. Nothing is frozen: the tangents of every parameter --
velocity, conductivity, source -- pass through.

It does not check on each call that the residual is affine. The tests check it
for the thermal residual, and the caller's residual gate checks every state it
returns.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

import toflux.src.solver as _solver

# the solve whose host callback never calls JAX; see tfopus/_callback_solve.py
from tfopus import _callback_solve  # noqa: F401


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def affine_solve(problem, x0: jnp.ndarray, *params) -> jnp.ndarray:
    """x with R(x; *params) = 0, for a residual affine in x: one assembly, one solve."""
    residual, jacobian = problem.get_residual_and_tangent_stiffness(x0, *params)
    return x0 - _solver.solve(jacobian, residual, params=problem.solver_settings["linear"])


@affine_solve.defjvp
def _affine_solve_jvp(problem, primals, tangents):
    """Upstream's implicit-function rule, at the state `affine_solve` returns."""
    x0, *params = primals
    _, *dparams = tangents
    x = affine_solve(problem, x0, *params)
    _, dr_dp, jacobian = jax.jvp(
        problem.get_residual_and_tangent_stiffness,
        (x, *params),
        (jnp.zeros_like(x), *dparams),
        has_aux=True,
    )
    return x, _solver.solve(jacobian, -dr_dp, params=problem.solver_settings["linear"])
