"""R1 plumbing: the frozen configuration, the guards, and the design map.

These are the things that make a result trustworthy rather than merely
reproducible-looking. The physics is covered by test_zhao2d.py; what is at
stake here is that a run cannot silently be a different discretisation, cannot
consume an unconverged state, and cannot reuse a reference value frozen under
different settings.
"""

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tfopus import fe_flow, fe_thermal, zhao2d as z, zhao2d_r1 as r1  # noqa: E402

# Coarse everywhere: these test wiring, not discretisation accuracy. h must
# divide the tab (0.001) and the design domain (0.005 x 0.01), so 5e-4 is the
# coarsest conforming size -- 10 x 20 design plus 2 x (2 x 2) tabs = 208 cells.
SPEC = dataclasses.replace(z.Zhao2DSpec(), element_size=5.0e-4)


@pytest.fixture(scope="module")
def problem():
    return r1.Zhao2DProblem(SPEC, r1.R1Config())


@pytest.fixture(scope="module")
def frozen(problem):
    values, _ = r1.freeze_reference(SPEC, r1.R1Config())
    return values


# -- the frozen configuration ----------------------------------------------


def test_r1_main_line_is_what_was_decided():
    """The decided settings, asserted so a default change cannot move them."""
    c = r1.R1Config()
    assert c.flow_form == fe_flow.FlowForm(
        brinkman_in_tau=True,
        brinkman_in_supg_residual=True,
        viscous_form="symmetric",
    )
    assert c.thermal_form.tau == "two_limit"
    assert c.thermal_form.stabilise_source is True
    assert c.thermal_form.supg_heat_capacity is True
    assert c.reference == z.ReferenceField.TABS_FLUID.value
    assert c.alpha_max_reference == 1.0e6
    assert c.source == z.SourceRegion.WHOLE_DOMAIN.value


def test_r1_config_does_not_inherit_the_analysis_defaults():
    """zhao2d_analysis still defaults to ZHAO_FORM; R1 must not pick that up."""
    from tfopus import zhao2d_analysis as za

    assert za.CaseOptions().flow_form == fe_flow.ZHAO_FORM
    assert za.CaseOptions().thermal_form == fe_thermal.ZHAO_FORM
    assert r1.R1Config().flow_form != fe_flow.ZHAO_FORM
    assert r1.R1Config().thermal_form != fe_thermal.ZHAO_FORM


@pytest.mark.parametrize(
    "change",
    [
        {"reference": z.ReferenceField.UNIFORM_ALL.value},
        {"alpha_max_reference": 1.0e7},
        {"source": z.SourceRegion.DESIGN_ONLY.value},
        {"outlet": z.OutletKind.PINNED.value},
        {"element_length_mode": "diagonal"},
        {"volume_domain": r1.VolumeDomain.WHOLE},
        {"filter_radius_elements": 3.0},
        {"projection_beta": 2.0},
        {"flow_form": fe_flow.ZHAO_FORM},
        {"thermal_form": fe_thermal.ZHAO_FORM},
    ],
)
def test_every_switch_reaches_the_fingerprint(change):
    """A fingerprint that missed a switch would let a reference be misapplied."""
    base = r1.R1Config()
    assert dataclasses.replace(base, **change).fingerprint() != base.fingerprint()


def test_reference_values_reject_a_different_configuration(frozen):
    frozen.check(r1.R1Config(), SPEC)  # the one it was frozen under
    with pytest.raises(ValueError, match="different"):
        frozen.check(
            dataclasses.replace(r1.R1Config(), alpha_max_reference=1.0e7), SPEC
        )


def test_reference_values_reject_a_different_mesh(frozen):
    with pytest.raises(ValueError, match="frozen at"):
        frozen.check(r1.R1Config(), dataclasses.replace(SPEC, element_size=2.0e-4))


def test_reference_values_round_trip_through_json(frozen):
    assert r1.ReferenceValues.from_json(frozen.to_json()) == frozen


# -- the design map ---------------------------------------------------------


def test_tabs_stay_fluid_whatever_the_design_says(problem):
    """The filter must not bleed material into the fixed inlet/outlet tabs."""
    for value in (0.0, 0.5, 1.0):
        s = problem.solid_fraction(jnp.full(problem.num_design, value))
        tabs = np.asarray(s)[~problem.flow_mesh.design_mask]
        assert np.allclose(tabs, 0.0), f"tabs picked up material at x = {value}"


def test_filter_preserves_a_uniform_field(problem):
    """A row-normalised filter must leave a constant field unchanged."""
    s = problem.solid_fraction(jnp.full(problem.num_design, 0.6))
    design = np.asarray(s)[problem.flow_mesh.design_mask]
    assert np.allclose(design, 0.6, atol=1e-12)


def test_fluid_fraction_matches_the_analytic_reference_value(problem):
    """x = 0.6 is gamma = 0.4 on the design domain, exactly the bound."""
    x = jnp.full(problem.num_design, 1.0 - SPEC.reference_gamma)
    assert float(problem.fluid_fraction(x)) == pytest.approx(0.4, rel=1e-12)

    whole = dataclasses.replace(r1.R1Config(), volume_domain=r1.VolumeDomain.WHOLE)
    p_whole = r1.Zhao2DProblem(SPEC, whole)
    n_design = int(p_whole.flow_mesh.design_mask.sum())
    n_all = p_whole.flow_mesh.num_elems
    expected = (0.4 * n_design + (n_all - n_design)) / n_all
    assert float(
        p_whole.fluid_fraction(jnp.full(p_whole.num_design, 0.6))
    ) == pytest.approx(expected, rel=1e-12)


# -- the guards -------------------------------------------------------------


def test_unconverged_states_are_refused():
    """A one-iteration Newton must not be allowed to feed MMA.

    The implicit-function-theorem gradient assumes R = 0. At a state that never
    reached it the returned array is not the sensitivity of anything, and
    nothing else in the stack would notice.
    """
    settings = {
        "linear": {"solver": r1._solver.LinearSolvers.SCIPY_SPARSE, "rtol": 1e-10},
        "nonlinear": {"max_iter": 1, "threshold": 1e-14},
    }
    problem = r1.Zhao2DProblem(SPEC, r1.R1Config(), solver_settings=settings)
    s = problem.solid_fraction(jnp.full(problem.num_design, 0.6))
    with pytest.raises(r1.NotConverged, match="residual above"):
        problem.require_converged(s, 1.0e6)


def test_converged_states_pass_the_gate(problem):
    s = problem.solid_fraction(jnp.full(problem.num_design, 0.6))
    norms = problem.require_converged(s, 1.0e6)
    assert norms["flow"] < 1e-8 and norms["thermal"] < 1e-8


# -- the objective ----------------------------------------------------------


def test_objective_is_exactly_one_at_the_frozen_reference(problem, frozen):
    """J = w + (1-w) = 1 there, by construction -- so it checks the binding.

    If the stored denominators came from any other state or configuration this
    is not 1, which is the cheapest possible detector of a stale reference.
    """
    x = jnp.full(problem.num_design, 1.0 - SPEC.reference_gamma)
    j, g = problem.objective_and_constraint(x, frozen, 1.0e6)
    assert float(j) == pytest.approx(1.0, rel=1e-10)
    assert float(g) == pytest.approx(0.0, abs=1e-12)


def test_psi_0_does_not_depend_on_the_thermal_configuration():
    """One-way coupling: the thermal switch cannot move the flow answer.

    If this ever fails, the flow solve has picked up a thermal dependency and
    the A/B/C attribution in the freeze study is invalid.
    """
    base = r1.R1Config()
    psis = set()
    for form in (
        fe_thermal.ZHAO_FORM,
        fe_thermal.ZHOU_FORM,
        r1.R1_THERMAL_FORM,
    ):
        values, _ = r1.freeze_reference(
            SPEC, dataclasses.replace(base, thermal_form=form)
        )
        psis.add(values.psi_0)
    assert len(psis) == 1, f"Psi_0 moved with the thermal setting: {psis}"


def test_objective_is_differentiable_end_to_end(problem, frozen):
    """A finite, non-zero gradient through both implicit solves.

    test_zhao2d_gradient_check covers correctness against finite differences;
    this only guarantees the chain is not silently broken (all-zero or NaN),
    which is cheap enough to keep in the fast suite.
    """
    x = jnp.full(problem.num_design, 0.6)
    grad = jax.grad(
        lambda v: problem.objective_and_constraint(v, frozen, 1.0e6)[0]
    )(x)
    grad = np.asarray(grad)
    assert np.all(np.isfinite(grad))
    assert np.linalg.norm(grad) > 0.0
