"""How much the two stabilisation defects change the answer.

See validation/variants.py for what v0-v3 are. The flow is a plane channel with
the lower half blocked by a Brinkman plug, which makes it genuinely
two-dimensional -- plain Poiseuille does not discriminate, because there both
the published and the corrected forms of the two SUPG terms happen to coincide.

These tests pin down three facts:

  1. At the Reynolds numbers used in the paper (0.167 - 3) the defects are
     invisible, so the published designs stand.
  2. The SUPG index mix-up, not tau_1, is the dominant contributor.
  3. The discrepancy is weighted by tau, so it vanishes under mesh refinement:
     this is a loss of consistency order, not convergence to a different PDE.

Fact 1 is the one that matters when reusing this code at higher Reynolds number.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from validation import harness, variants

LENGTH, HEIGHT, MU, RHO = 4.0, 1.0, 1.0, 1.0
PLUG_X = (1.4, 1.6)
PLUG_Y_MAX = 0.5


def _run(variant: str, nx: int, ny: int, reynolds: float):
    umax = reynolds * MU / (RHO * HEIGHT)
    mesh = harness.channel_mesh(nx, ny, LENGTH, HEIGHT)
    tol = 0.25 * min(LENGTH / nx, HEIGHT / ny)
    bc, _ = harness.poiseuille_bcs(mesh, LENGTH, HEIGHT, umax, tol)
    material = harness.tf_material.FluidMaterial(
        mass_density=RHO, dynamic_viscosity=MU
    )
    solver = variants.build(variant).FluidSolver(
        mesh, bc, material, harness.SOLVER_SETTINGS
    )

    centres = np.asarray(mesh.elem_centers)
    plugged = (
        (centres[:, 0] > PLUG_X[0])
        & (centres[:, 0] < PLUG_X[1])
        & (centres[:, 1] < PLUG_Y_MAX)
    )
    a_min, a_max = harness.brinkman_bounds(MU, LENGTH)
    alpha = jnp.where(jnp.array(plugged), a_max, a_min)

    press_vel = harness.solve(solver, bc, alpha)
    objective = harness.dissipated_power(solver, mesh, alpha, press_vel)
    return np.asarray(press_vel), objective


def _relative_gap(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_variants_agree_at_the_papers_reynolds_number():
    """At Re = 1 all four variants give the same answer to within 1e-3."""
    reference, obj_ref = _run("v3", 64, 16, reynolds=1.0)
    for name in ("v0", "v1", "v2"):
        field, obj = _run(name, 64, 16, reynolds=1.0)
        gap = _relative_gap(field, reference)
        assert gap < 1e-3, f"{variants.LABELS[name]} differs by {gap:.3e} at Re=1"
        assert abs(obj - obj_ref) / abs(obj_ref) < 1e-4


def test_supg_indices_dominate_over_tau1_at_high_reynolds():
    """At Re = 50 fixing the SUPG indices alone closes most of the gap."""
    reference, _ = _run("v3", 64, 16, reynolds=50.0)
    gap_upstream = _relative_gap(_run("v0", 64, 16, 50.0)[0], reference)
    gap_tau1_only = _relative_gap(_run("v1", 64, 16, 50.0)[0], reference)
    gap_supg_only = _relative_gap(_run("v2", 64, 16, 50.0)[0], reference)

    assert gap_upstream > 1e-3, (
        f"expected a material gap at Re=50, got {gap_upstream:.3e}. "
        "If upstream fixed these defects, retire this test."
    )
    assert gap_supg_only < gap_tau1_only, (
        f"SUPG-only residual gap {gap_supg_only:.3e} should be smaller than "
        f"tau_1-only {gap_tau1_only:.3e}"
    )


@pytest.mark.slow
def test_discrepancy_vanishes_under_mesh_refinement():
    """The gap is tau-weighted, so refining the mesh must shrink it."""
    gaps = []
    for nx, ny in [(32, 8), (64, 16), (128, 32)]:
        reference, _ = _run("v3", nx, ny, reynolds=10.0)
        gaps.append(_relative_gap(_run("v0", nx, ny, 10.0)[0], reference))

    assert all(a > b for a, b in zip(gaps, gaps[1:])), (
        f"gap did not shrink monotonically under refinement: {gaps}"
    )
    assert gaps[-1] < gaps[0] / 4, (
        f"gap shrank only from {gaps[0]:.3e} to {gaps[-1]:.3e}; expected at "
        "least a factor of 4 over two refinements"
    )
