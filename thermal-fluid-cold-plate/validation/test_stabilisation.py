"""The stabilisation is the accuracy-limiting choice, and h_e is why.

These pin down a result that no error message would ever surface: with the
characteristic length taken as the mean body diagonal -- upstream's 2D
convention -- the 3D duct solution is only first-order accurate and, at
Re = 1500, one mesh in the sequence blows up outright. Taking the shortest edge
instead fixes both. Nothing in the solver complains either way; only a measured
comparison separates them, so it lives here.

Measured, 40 x 4 x 8 duct, relative velocity error on the mid plane:

    Re        diagonal     cube root vol   min edge
    0.15      5.625e-01    1.379e-01       1.562e-02
    150       9.591e-02    3.220e-02       7.780e-03
    1500      1.252e+01 (diverged)         1.032e-03
"""

import numpy as np
import pytest

from validation import duct

RE_LOW, RE_MID = 0.15, 150.0
U_LOW = RE_LOW * duct.MU / (duct.RHO * duct.H)
U_MID = RE_MID * duct.MU / (duct.RHO * duct.H)


def test_error_is_the_stabilisation_not_the_galerkin_terms():
    """Scaling tau down collapses the error, so the volume terms are right.

    This is the test that says a large duct error is NOT a bug in the residual,
    the boundary conditions or the assembly.
    """
    errors = {}
    for scale in (1.0, 0.01, 0.001):
        err, dpdx = duct.solve_duct(40, 4, 8, U_LOW, h_mode="diagonal", tau_scale=scale)
        errors[scale] = (err, dpdx)

    assert errors[1.0][0] > 0.3, "the diagonal-h error should be large here"
    assert errors[0.001][0] < 2e-3, f"tau -> 0 should recover the exact solution, got {errors[0.001][0]:.2e}"
    # and the pressure gradient converges to the analytic value as tau -> 0
    exact = duct.exact_dpdx(U_LOW)
    assert abs(errors[0.001][1] - exact) / abs(exact) < 5e-3


@pytest.mark.parametrize(
    "umax, factor", [(U_LOW, 10.0), (U_MID, 5.0)], ids=["Re=0.15", "Re=150"]
)
def test_min_edge_beats_the_body_diagonal(umax, factor):
    """Shortest edge is uniformly more accurate, at both tau regimes."""
    err_diag, _ = duct.solve_duct(40, 4, 8, umax, h_mode="diagonal")
    err_min, dpdx = duct.solve_duct(40, 4, 8, umax, h_mode="min_edge")
    assert err_min * factor < err_diag, (
        f"min_edge {err_min:.3e} vs diagonal {err_diag:.3e}"
    )
    exact = duct.exact_dpdx(umax)
    assert abs(dpdx - exact) / abs(exact) < 0.05


def test_min_edge_is_second_order_in_the_diffusive_regime():
    """At Re = 0.15 tau is the diffusive limit and the scheme recovers O(h^2)."""
    errs = [duct.solve_duct(n, n // 10, n // 5, U_LOW)[0] for n in (20, 40, 80)]
    rates = [np.log2(a / b) for a, b in zip(errs, errs[1:])]
    assert all(1.8 < r < 2.2 for r in rates), f"rates {rates} from {errs}"


def test_body_diagonal_is_only_first_order():
    """The same sequence with the diagonal: rate ~0.93, and it does not improve."""
    errs = [
        duct.solve_duct(n, n // 10, n // 5, U_LOW, h_mode="diagonal")[0]
        for n in (20, 40, 80)
    ]
    rates = [np.log2(a / b) for a, b in zip(errs, errs[1:])]
    assert all(r < 1.5 for r in rates), f"expected first order, got rates {rates}"


@pytest.mark.slow
def test_convection_dominated_case_refines_and_stays_bounded():
    """Accuracy and non-degradation only -- the ORDER is recorded, not asserted.

    At Re = 150 the shortest-edge scheme converges at roughly 1.3-1.5 rather
    than 2. Asserting that band would be backwards: a future fix that restores
    second order would fail the test. So the observed rate is printed for the
    record and the assertions cover what should hold under any correct scheme --
    the error falls monotonically under refinement and stays small.

    The rate has NOT been attributed. The truncated strong residual is the
    leading suspect, since tau is the convective limit h/(2|u|) = O(h) here,
    but a non-zero strong residual does not by itself imply a non-zero discrete
    defect, let alone a particular global order. That attribution needs the
    manufactured-solution A/B, not this duct.
    """
    errs = [duct.solve_duct(n, n // 10, n // 5, U_MID)[0] for n in (20, 40, 80)]
    rates = [float(np.log2(a / b)) for a, b in zip(errs, errs[1:])]
    print(f"observed rates at Re=150: {rates} from errors {errs}")
    assert all(b < a for a, b in zip(errs, errs[1:])), f"not refining: {errs}"
    assert errs[-1] < 5e-3, f"final error {errs[-1]:.2e}"
