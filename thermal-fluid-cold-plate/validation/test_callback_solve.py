"""The linear-solve callback fix: installed, and the same numbers bit for bit.

Upstream's `solve` calls JAX from inside its host callback, which can deadlock
an eager backward pass (see tfopus/_callback_solve.py). The replacement must be
the function the Newton solvers actually call, and must change nothing
numerically: the value, the transposed solve reverse mode uses, and a full
Newton-solved design gradient are compared with upstream's own function.
"""

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import pytest

jax.config.update("jax_enable_x64", True)

import toflux.src.solver as upstream  # noqa: E402

from tfopus import _callback_solve as cb  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

SETTINGS = {"solver": upstream.LinearSolvers.SCIPY_SPARSE, "rtol": 1e-10}


def _system(n=40, seed=0):
    """A sparse, non-symmetric, well-conditioned system, like an FE tangent."""
    rng = np.random.default_rng(seed)
    dense = np.diag(rng.uniform(4.0, 6.0, n))
    dense += 0.3 * rng.normal(size=(n, n)) * (rng.uniform(size=(n, n)) < 0.2)
    return jsparse.BCOO.fromdense(jnp.asarray(dense)), jnp.asarray(rng.normal(size=n))


def test_the_newton_solvers_call_the_replacement():
    assert upstream.solve is cb.solve
    assert cb.UPSTREAM_SOLVE is not None and cb.UPSTREAM_SOLVE is not cb.solve


def test_value_and_transposed_solve_are_bit_identical_to_upstreams():
    A, b = _system()
    assert np.array_equal(np.asarray(cb.solve(A, b, SETTINGS)),
                          np.asarray(cb.UPSTREAM_SOLVE(A, b, SETTINGS)))

    # reverse mode through custom_linear_solve runs the TRANSPOSED solve
    c = jnp.asarray(np.random.default_rng(1).normal(size=b.shape[0]))

    def loss(fn):
        return lambda rhs: jnp.dot(c, fn(A, rhs, SETTINGS))

    assert np.array_equal(np.asarray(jax.grad(loss(cb.solve))(b)),
                          np.asarray(jax.grad(loss(cb.UPSTREAM_SOLVE))(b)))


def test_a_newton_solved_design_gradient_is_bit_identical_to_upstreams():
    spec = dataclasses.replace(z.Zhao2DSpec(), element_size=5.0e-4)
    problem = dual.Zhao2DDualProblem(spec, r1.R1Config(), 2, 3)
    x = jnp.asarray(np.random.default_rng(5).uniform(0.15, 0.85, problem.num_design))

    def c_of(v):
        return problem.metrics(problem.solid_fraction(v, 8.0), 1.0e7)[1]

    new_value, new_grad = jax.value_and_grad(c_of)(x)
    upstream.solve = cb.UPSTREAM_SOLVE
    try:
        old_value, old_grad = jax.value_and_grad(c_of)(x)
    finally:
        upstream.solve = cb.solve
    assert float(new_value) == float(old_value)
    assert np.array_equal(np.asarray(new_grad), np.asarray(old_grad))


def test_other_solvers_are_refused_not_half_ported():
    A, b = _system(8)
    with pytest.raises(Exception, match="not ported"):
        jax.block_until_ready(
            cb.solve(A, b, {"solver": upstream.LinearSolvers.AMG_CG, "rtol": 1e-8}))
