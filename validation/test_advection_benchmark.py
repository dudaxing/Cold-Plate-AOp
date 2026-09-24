"""The analytic benchmark's measuring stick, checked before it measures anything.

An earlier version reported relative L2 errors whose DENOMINATOR -- the norm of
the exact solution, integrated with a fixed 10-point rule -- was 51% low at
nx = 10 and 9% low at nx = 20. Nothing looked wrong; the numbers were just not
comparable across meshes.
"""

import numpy as np
import pytest
from scipy import integrate

from tfopus import advection_benchmark as ab

PE = 1000.0


@pytest.mark.parametrize("nx, ny", [(10, 8), (20, 8), (40, 8), (8, 2), (32, 8)])
def test_the_exact_norm_does_not_depend_on_the_mesh(nx, ny):
    """Integrating T_exact itself must return the closed form on every mesh."""
    mesh, _, _, _ = ab.build(nx, ny, PE)
    out = ab.l2_error(mesh, np.zeros(mesh.num_nodes), PE)  # ||0 - T|| = ||T||
    assert out["error_sq"] == pytest.approx(ab.exact_norm_sq(PE), rel=1e-10)


@pytest.mark.parametrize("pe", [0.5, 5.0, 40.0])
def test_closed_form_norm_against_adaptive_quadrature(pe):
    ref, _ = integrate.quad(lambda x: ab.exact(x, pe) ** 2, 0.0, ab.L,
                            epsabs=0.0, epsrel=1e-13, limit=200)
    assert ab.exact_norm_sq(pe) == pytest.approx(ab.H * ref, rel=1e-11)


def test_closed_form_norm_at_the_benchmark_peclet_number():
    assert ab.exact_norm_sq(PE) == pytest.approx(ab.H * ab.L / (2 * PE), rel=1e-12)


def test_an_unconverged_error_integral_is_refused_not_reported():
    mesh, _, _, _ = ab.build(10, 8, PE)
    with pytest.raises(RuntimeError, match="not converged"):
        ab.l2_error(mesh, np.zeros(mesh.num_nodes), PE, max_subdivision=2)


def test_lengths_are_reported_separately():
    """nx = 10, ny = 8: tau sees hy, not hx -- the anisotropic series' trap."""
    geo = ab.lengths(ab.build(10, 8, PE)[0])
    assert geo["hx"] == pytest.approx(0.1)
    assert geo["hy"] == pytest.approx(0.03125)
    assert geo["h_tau"] == pytest.approx(0.03125)

    geo = ab.lengths(ab.build(20, 5, PE)[0])
    assert geo["hx"] == pytest.approx(geo["hy"]) == pytest.approx(geo["h_tau"])
