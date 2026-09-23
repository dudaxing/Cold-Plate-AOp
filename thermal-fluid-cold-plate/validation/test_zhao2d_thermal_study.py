"""The R1f separation machinery: velocity extension, frozen tau, the C identity.

Each of these is a way the separation could be silently wrong: an extension that
is really a projection, a "frozen" tau that quietly recomputed, or a
decomposition that does not actually add up to the equation being solved.
"""

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import toflux.src.solver as tf_solver  # noqa: E402

from tfopus import elements as elements  # noqa: E402
from tfopus import fe_thermal as fe_thermal  # noqa: E402
from tfopus import materials as materials  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402
from tfopus import zhao2d_refine as ref  # noqa: E402
from tfopus import zhao2d_thermal_study as ts  # noqa: E402

SPEC = dataclasses.replace(z.Zhao2DSpec(), element_size=5.0e-4)


@pytest.fixture(scope="module")
def meshes():
    fine_spec = ref.refine_spec(SPEC, 2)
    return (
        z.build_mesh(SPEC, dofs_per_node=1, gauss_order=3),
        z.build_mesh(fine_spec, dofs_per_node=1, gauss_order=3),
        fine_spec,
    )


# -- velocity extension -----------------------------------------------------


def test_extension_reproduces_a_linear_field_exactly(meshes):
    """Q1 represents linear fields exactly, so the extension must be exact."""
    coarse, fine, _ = meshes
    cc = np.asarray(coarse.mesh.nodes.coords)
    u = np.stack([3.0 + 2.0 * cc[:, 0] - cc[:, 1], -1.0 + 0.5 * cc[:, 1]], axis=1)
    out = ts.extend_velocity(coarse, fine, u)

    fc = np.asarray(fine.mesh.nodes.coords)
    expect = np.stack(
        [3.0 + 2.0 * fc[:, 0] - fc[:, 1], -1.0 + 0.5 * fc[:, 1]], axis=1
    )
    assert np.allclose(out, expect, atol=1e-12)


def test_extension_is_exact_on_shared_nodes(meshes):
    """The check that separates an extension from a projection."""
    coarse, fine, _ = meshes
    rng = np.random.default_rng(0)
    u = rng.normal(size=(coarse.mesh.num_nodes, 2))
    out = ts.extend_velocity(coarse, fine, u)
    report = ts.check_extension(coarse, fine, u, out)
    assert report["shared_nodes"] == coarse.mesh.num_nodes
    assert report["worst_shared_node_difference"] < 1e-13


def test_check_extension_rejects_a_perturbed_field(meshes):
    coarse, fine, _ = meshes
    rng = np.random.default_rng(1)
    u = rng.normal(size=(coarse.mesh.num_nodes, 2))
    out = np.array(ts.extend_velocity(coarse, fine, u))
    out[0] += 1.0
    with pytest.raises(RuntimeError, match="disagrees"):
        ts.check_extension(coarse, fine, u, out)


# -- frozen tau -------------------------------------------------------------


def _thermal(mesh, spec, tau_elem=None):
    bc = z.build_thermal_bc(mesh, spec)
    return fe_thermal.ThermalSolver(
        mesh.mesh,
        bc,
        b_f=spec.b_f,
        solver_settings=za.default_solver_settings(),
        elem_length=elements.element_lengths(mesh.mesh, "min_edge"),
        form=r1.R1_THERMAL_FORM,
        tau_elem=tau_elem,
    ), bc


def test_frozen_tau_is_the_value_supplied(meshes):
    """A 'frozen' tau that silently recomputed would make B meaningless."""
    coarse, fine, fine_spec = meshes
    parents = ref.parent_of_each_fine_element(coarse, fine)
    rng = np.random.default_rng(2)
    tau_coarse = rng.uniform(1e-6, 1e-4, coarse.num_elems)
    tau_fine = ts.freeze_tau_from_parent(parents, tau_coarse)

    assert np.array_equal(np.asarray(tau_fine), tau_coarse[parents])
    solver, _ = _thermal(fine, fine_spec, tau_elem=tau_fine)
    assert solver.tau_elem is not None
    assert np.allclose(np.asarray(solver.tau_elem), tau_coarse[parents])


def test_tau_formula_scales_as_h_squared_in_the_diffusive_limit(meshes):
    """tau_3 = b_f h^2 / (4k): halving h must divide tau by four at rest.

    This is the internal check that B -> C really changed tau and by the
    expected amount, rather than by some other mesh-dependent quantity.
    """
    coarse, fine, fine_spec = meshes
    for mesh, spec in ((coarse, SPEC), (fine, fine_spec)):
        solver, _ = _thermal(mesh, spec)
        vel = jnp.zeros((mesh.num_elems, 4 * 2))
        k = jnp.full(mesh.num_elems, 47.888)
        tau = np.asarray(ts.element_tau(solver, vel, k))
        expected = spec.b_f * spec.element_size**2 / (4.0 * 47.888)
        assert np.allclose(tau, expected, rtol=1e-10)


@pytest.mark.parametrize("speed", [0.0, 1e-5, 1e-3, 0.2])
def test_tau_ratio_under_halving_follows_the_peclet_formula(meshes, speed):
    """tau(h/2)/tau(h) = 1/2 sqrt(Pe^2 + 1)/sqrt(Pe^2 + 4), Pe the parent's.

    1/4 only where diffusion dominates, 1/2 where convection does. So a
    whole-domain median of 1/4 says the median cell is diffusion-dominated --
    on the cold plate, solid -- and nothing about the channels, where R1f's
    paired child/parent ratio has a median near 1/2.
    """
    coarse, fine, fine_spec = meshes
    k = 0.61
    taus = []
    for mesh, spec in ((coarse, SPEC), (fine, fine_spec)):
        solver, _ = _thermal(mesh, spec)
        vel = jnp.tile(jnp.array([speed, 0.0]), (mesh.num_elems, 4))
        taus.append(np.asarray(
            ts.element_tau(solver, vel, jnp.full(mesh.num_elems, k))
        ))
    pe = SPEC.b_f * speed * SPEC.element_size / (2.0 * k)
    expected = 0.5 * np.sqrt(pe**2 + 1.0) / np.sqrt(pe**2 + 4.0)
    assert np.allclose(taus[1] / taus[0][0], expected, rtol=1e-12)


def test_supplying_tau_changes_the_answer(meshes):
    """If tau_elem were ignored, A/B/C would be indistinguishable."""
    coarse, _, _ = meshes
    solver_a, bc = _thermal(coarse, SPEC)
    vel = jnp.tile(jnp.array([0.2, 0.0]), (coarse.num_elems, 4))
    k = jnp.full(coarse.num_elems, 47.888)
    q = z.heat_source_field(coarse, SPEC, z.SourceRegion.WHOLE_DOMAIN)
    t0 = jnp.zeros((coarse.mesh.num_dofs,)).at[bc["fixed_dofs"]].set(
        bc["dirichlet_values"]
    )
    t_a = tf_solver.modified_newton_raphson_solve(solver_a, t0, vel, k, q)

    tau_a = ts.element_tau(solver_a, vel, k)
    solver_b, _ = _thermal(coarse, SPEC, tau_elem=tau_a)
    t_b = tf_solver.modified_newton_raphson_solve(solver_b, t0, vel, k, q)
    assert np.allclose(np.asarray(t_a), np.asarray(t_b), rtol=1e-10), (
        "supplying the formula's own tau must reproduce the formula path"
    )

    solver_c, _ = _thermal(coarse, SPEC, tau_elem=tau_a * 10.0)
    t_c = tf_solver.modified_newton_raphson_solve(solver_c, t0, vel, k, q)
    assert not np.allclose(np.asarray(t_a), np.asarray(t_c), rtol=1e-6)


# -- the compliance identity ------------------------------------------------


def test_compliance_identity_closes_on_a_real_solve(meshes):
    """C + D_SUPG - F_SUPG = L_Q, exactly, whenever T_h is a valid test function.

    It holds because the inlet Dirichlet value is zero, so T_h vanishes there.
    """
    coarse, _, _ = meshes
    solver, bc = _thermal(coarse, SPEC)
    vel = jnp.tile(jnp.array([0.2, 0.05]), (coarse.num_elems, 4))
    k = jnp.full(coarse.num_elems, 12.0)
    q = z.heat_source_field(coarse, SPEC, z.SourceRegion.WHOLE_DOMAIN)
    t0 = jnp.zeros((coarse.mesh.num_dofs,)).at[bc["fixed_dofs"]].set(
        bc["dirichlet_values"]
    )
    T = tf_solver.modified_newton_raphson_solve(solver, t0, vel, k, q)

    # The identity is exact for the EXACT discrete solution, so it closes only
    # as well as the state is solved -- tie the tolerance to the achieved
    # residual rather than to a fixed number, or the test measures the linear
    # solver's luck instead of the algebra.
    res, _ = solver.get_residual_and_tangent_stiffness(T, vel, k, q)
    res0, _ = solver.get_residual_and_tangent_stiffness(t0, vel, k, q)
    residual_relative = float(
        jnp.linalg.norm(res) / jnp.maximum(jnp.linalg.norm(res0), 1e-300)
    )

    d = solver.compliance_decomposition(T, vel, k, q)
    assert d["closure_relative"] <= max(1e-13, 100.0 * residual_relative), (
        f"closure {d['closure_relative']:.2e} against a state residual of "
        f"{residual_relative:.2e}"
    )
    assert d["compliance"] == pytest.approx(
        d["c_advective"] + d["c_diffusive"], rel=1e-12
    )
    # and it agrees with the objective the optimiser actually used
    assert d["compliance"] == pytest.approx(
        float(solver.thermal_compliance(T, vel, k)), rel=1e-10
    )


def test_identity_does_not_close_if_the_state_is_wrong(meshes):
    """The closure is a real check, not an algebraic tautology."""
    coarse, _, _ = meshes
    solver, bc = _thermal(coarse, SPEC)
    vel = jnp.tile(jnp.array([0.2, 0.0]), (coarse.num_elems, 4))
    k = jnp.full(coarse.num_elems, 12.0)
    q = z.heat_source_field(coarse, SPEC, z.SourceRegion.WHOLE_DOMAIN)
    bogus = jnp.asarray(
        np.random.default_rng(3).normal(size=coarse.mesh.num_dofs)
    )
    d = solver.compliance_decomposition(bogus, vel, k, q)
    assert d["closure_relative"] > 1e-3


# -- Peclet reporting -------------------------------------------------------


def test_peclet_reports_every_mask_separately(meshes):
    """A bare 'median Pe' is unusable: the two masks differ by 10^4 here."""
    coarse, _, _ = meshes
    rng = np.random.default_rng(4)
    s = jnp.asarray(rng.uniform(0.0, 1.0, coarse.num_elems))
    vel = jnp.tile(jnp.array([0.2, 0.0]), (coarse.num_elems, 4))
    k = jnp.full(coarse.num_elems, 47.888)
    stats = ts.peclet_statistics(
        coarse, vel, k, SPEC.b_f, SPEC.element_size, s
    )
    assert "element centre" in stats["definition"]
    for mask in ("whole_domain", "fluid_s_lt_0.5_incl_tabs",
                 "fluid_s_lt_0.5_design_only"):
        assert set(stats[mask]) == {
            "elements", "median", "p90", "max", "fraction_above_1"
        }
    assert (
        stats["fluid_s_lt_0.5_design_only"]["elements"]
        <= stats["fluid_s_lt_0.5_incl_tabs"]["elements"]
        <= stats["whole_domain"]["elements"]
    )

