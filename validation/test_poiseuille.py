"""The forward Navier-Stokes-Brinkman solver against the plane-Poiseuille solution.

With alpha -> 0 the channel carries pure fluid, and u(y) = umax * 4y(H-y)/H^2 with
dp/dx = -8*mu*umax/H^2 solves the steady Navier-Stokes equations exactly (the
convective term vanishes because u depends on y alone). Q1 elements cannot
represent the parabola exactly, so the test asserts the discretisation error and
its convergence rate rather than an exact match.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from validation import harness

LENGTH, HEIGHT, UMAX, MU, RHO = 4.0, 1.0, 1.0, 1.0, 1.0
EXACT_DPDX = -8.0 * MU * UMAX / HEIGHT**2
ALPHA_FLUID = 1e-9  # effectively zero Brinkman penalty


def _solve(nx, ny):
    mesh = harness.channel_mesh(nx, ny, LENGTH, HEIGHT)
    tol = 0.25 * min(LENGTH / nx, HEIGHT / ny)
    bc, u_exact = harness.poiseuille_bcs(mesh, LENGTH, HEIGHT, UMAX, tol)
    material = harness.tf_material.FluidMaterial(
        mass_density=RHO, dynamic_viscosity=MU
    )
    solver = harness.tf_fluid.FluidSolver(
        mesh, bc, material, harness.SOLVER_SETTINGS
    )
    alpha = ALPHA_FLUID * jnp.ones((mesh.num_elems,))
    press_vel = harness.solve(solver, bc, alpha)
    return mesh, press_vel, u_exact


def _errors(nx, ny):
    mesh, press_vel, u_exact = _solve(nx, ny)
    p, u, v = harness.split_fields(press_vel)
    coords = np.asarray(mesh.nodes.coords)

    u_ref = np.array([u_exact(c) for c in coords])
    err_u = np.linalg.norm(np.asarray(u) - u_ref) / np.linalg.norm(u_ref)
    err_v = np.linalg.norm(np.asarray(v)) / np.linalg.norm(u_ref)

    centreline = np.abs(coords[:, 1] - HEIGHT / 2) < 1e-9
    dpdx = np.polyfit(coords[centreline, 0], np.asarray(p)[centreline], 1)[0]
    return err_u, err_v, dpdx


def test_velocity_profile_matches_analytic():
    err_u, err_v, _ = _errors(40, 20)
    assert err_u < 2e-2, f"velocity error {err_u:.3e} too large"
    # v should be identically zero; only discretisation error makes it non-zero.
    assert err_v < 1e-3, f"spurious transverse velocity {err_v:.3e}"


def test_pressure_gradient_matches_analytic():
    _, _, dpdx = _errors(40, 20)
    rel = abs(dpdx - EXACT_DPDX) / abs(EXACT_DPDX)
    assert rel < 2e-2, f"dp/dx = {dpdx:.5f}, expected {EXACT_DPDX} (rel {rel:.2e})"


def test_second_order_convergence():
    """Halving h should cut the error by ~4 in both velocity and pressure gradient."""
    meshes = [(20, 10), (40, 20), (80, 40)]
    err_u, err_p = [], []
    for nx, ny in meshes:
        eu, _, dpdx = _errors(nx, ny)
        err_u.append(eu)
        err_p.append(abs(dpdx - EXACT_DPDX) / abs(EXACT_DPDX))

    for name, errs in (("velocity", err_u), ("dp/dx", err_p)):
        rates = [np.log2(a / b) for a, b in zip(errs, errs[1:])]
        assert all(1.8 < r < 2.2 for r in rates), (
            f"{name} convergence rates {rates} are not second order "
            f"(errors {errs})"
        )
