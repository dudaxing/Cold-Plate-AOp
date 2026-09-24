"""Upstream's linear solve, without calling JAX from inside its host callback.

`toflux.src.solver.solve` hands the assembled matrix to SciPy through
`jax.pure_callback`, and the callback's first lines -- `jax.lax.stop_gradient
(A.data)`, `A.indices[:, 0]`, `jax.lax.stop_gradient(b)` -- are JAX operations.
In the JAX this project runs, the callback receives jax.Arrays, so each of them
dispatches a new computation from inside the one that is running the callback.

Forward solves have never hung: there the Newton loop is one compiled
computation and the main thread only waits for it. An eager backward pass is
different. The main thread keeps dispatching the pass's next operation, which
needs the callback's result, while the callback dispatches its own, and the two
can end up waiting on each other with no CPU in use. R1j's check hung twice,
both times in a reverse pass after its first; in the second, faulthandler
caught the two threads blocked in dispatch together -- the callback at
`A.indices[:, 0]`, the main thread at the next transposed dot
(results/zhao2d_r1j_check_attempt2_stacks.log). It is a race: every reverse
pass of the earlier stages completed, R1d's 300 among them, and why R1j's later
ones lost it is not established.

JAX's rule for host callbacks is to stay out of JAX inside them. `solve` below
converts the callback's inputs to NumPy before touching them and is otherwise
upstream's function, for the one solver this project uses (SciPy's sparse
direct solve; the others are refused rather than half-ported). It is installed
over `toflux.src.solver.solve`, which the Newton solvers look up when called,
and only if upstream's function is byte for byte the version it was written
against: a changed upstream must be read again, not silently overridden.
"""

from __future__ import annotations

import hashlib
import inspect

import numpy as np
import scipy.sparse as _sp
import jax

import toflux.src.solver as _upstream

# sha256 of inspect.getsource(toflux.src.solver.solve) as extracted from the
# TOFLUX supplementary archive by scripts/setup_toflux.py
UPSTREAM_SOLVE_SHA256 = "cefeea58f12a88d6da5938b347cd3cef0663d9b94b6b829d301205f5bd7e544f"

UPSTREAM_SOLVE = None  # upstream's own function, kept for comparison tests


def solve(A, b, params, u0=None):
    """`toflux.src.solver.solve`, with a callback that never calls JAX."""

    def mv(u):
        return A @ u

    def solver_wrapper(A, b):
        # NumPy first: no JAX operation may run inside the callback.
        data = np.asarray(A.data)
        indices = np.asarray(A.indices)
        rhs = np.asarray(b)
        if params["solver"] != _upstream.LinearSolvers.SCIPY_SPARSE:
            raise NotImplementedError(
                f"{params['solver']} is not ported: this replacement covers "
                "SCIPY_SPARSE, the solver this project uses"
            )
        A_sp = _sp.coo_matrix((data, (indices[:, 0], indices[:, 1])), shape=A.shape)
        x = _upstream.spy_linalg.spsolve(A_sp.tocsr(), rhs)
        return np.asarray(x).astype(rhs.dtype).reshape(rhs.shape)

    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)

    def cust_fwd_solver(mv, b):
        return jax.pure_callback(solver_wrapper, result_shape, A, b)

    def cust_bwd_solver(mv, b):
        return jax.pure_callback(solver_wrapper, result_shape, A.T, b)

    sol = jax.lax.custom_linear_solve(
        mv, b, solve=cust_fwd_solver, transpose_solve=cust_bwd_solver
    )
    return sol.reshape(-1)


def install() -> None:
    """Put `solve` in place of upstream's, once; refuse a changed upstream."""
    global UPSTREAM_SOLVE
    if _upstream.solve is solve:
        return
    source = inspect.getsource(_upstream.solve)
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != UPSTREAM_SOLVE_SHA256:
        raise RuntimeError(
            "toflux.src.solver.solve is not the version tfopus/_callback_solve.py "
            f"was written against (sha256 {digest}); re-read it before replacing it"
        )
    UPSTREAM_SOLVE = _upstream.solve
    _upstream.solve = solve


install()
