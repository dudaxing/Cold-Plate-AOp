"""The optimisation driver: schedule shape, terminal pairing, stop reasons.

These are the things a silent bug would turn into a plausible-looking result.
"""

import dataclasses

import numpy as np
import jax
import pytest

jax.config.update("jax_enable_x64", True)

from tfopus import zhao2d as z, zhao2d_driver as drv, zhao2d_r1 as r1  # noqa: E402

SPEC = dataclasses.replace(z.Zhao2DSpec(), element_size=5.0e-4)


# -- schedule ---------------------------------------------------------------


def test_full_schedule_reaches_the_cap_before_any_projection():
    """Zhao's 1.03 ramp hits 1e7 at n = 78, inside the 100-step alpha phase."""
    spec = z.Zhao2DSpec()
    phases = drv.r1d_schedule(spec, drv.FULL_BUDGET)
    assert sum(p.iterations for p in phases) == drv.FULL_BUDGET

    alpha = phases[0]
    assert alpha.beta == 0.0
    assert alpha.alpha_at(78) == pytest.approx(spec.alpha_max_final)
    assert alpha.alpha_at(77) < spec.alpha_max_final
    # and it is still at the cap when the phase ends
    assert alpha.alpha_at(alpha.iterations - 1) == pytest.approx(
        spec.alpha_max_final
    )
    assert [p.beta for p in phases[1:]] == [1.0, 2.0, 4.0, 8.0]
    assert all(p.alpha_max == spec.alpha_max_final for p in phases[1:])


def test_full_schedule_uses_the_papers_growth_rate_unmodified():
    spec = z.Zhao2DSpec()
    phases = drv.r1d_schedule(spec, drv.FULL_BUDGET)
    assert "COMPRESSED" not in phases[0].note
    for n in (0, 10, 50, 78):
        assert phases[0].alpha_at(n) == pytest.approx(spec.alpha_max(n))


def test_short_budget_compresses_the_ramp_and_says_so():
    """A rehearsal must still reach the cap before beta moves, and admit it."""
    spec = z.Zhao2DSpec()
    phases = drv.r1d_schedule(spec, 25)
    assert sum(p.iterations for p in phases) == 25
    assert "COMPRESSED" in phases[0].note
    assert phases[0].alpha_at(phases[0].iterations - 1) == pytest.approx(
        spec.alpha_max_final
    )
    drv.validate_schedule(phases, spec)


def test_schedule_never_produces_negative_or_zero_phases():
    spec = z.Zhao2DSpec()
    for budget in (5, 7, 13, 25, 60, 137, 300):
        phases = drv.r1d_schedule(spec, budget)
        assert all(p.iterations >= 1 for p in phases), budget
        assert sum(p.iterations for p in phases) == budget
    for budget in (1, 4):
        with pytest.raises(ValueError):
            drv.r1d_schedule(spec, budget)


def test_validate_schedule_rejects_a_boundary_that_moves_both():
    """The guard that makes a jump in J attributable to one cause."""
    spec = z.Zhao2DSpec()
    bad = [
        drv.Phase("ramp", 3, beta=0.0, alpha_max=spec.alpha_max),  # ends at 1.09e6
        drv.Phase("beta-1", 3, beta=1.0, alpha_max=spec.alpha_max_final),
    ]
    with pytest.raises(ValueError, match="moves both"):
        drv.validate_schedule(bad, spec)


# -- the loop ---------------------------------------------------------------


@pytest.fixture(scope="module")
def setup():
    config = r1.R1Config()
    problem = r1.Zhao2DProblem(SPEC, config)
    reference, _ = r1.freeze_reference(SPEC, config)
    return problem, reference


def test_terminal_record_is_paired_with_the_saved_design(setup):
    """The saved design must have metrics from ITSELF, not from one step back.

    The loop evaluates x_n then asks MMA for x_{n+1}; saving state.x after the
    loop saves a design that was never solved. The driver re-evaluates it.
    """
    problem, reference = setup
    phases = drv.r1d_schedule(SPEC, 6)
    result = drv.run(problem, reference, phases, move_limit=0.1)

    assert result.terminal["iteration"] == len(result.history)
    assert result.terminal.get("terminal") is True

    # recomputing from the saved design reproduces the terminal record exactly
    import jax.numpy as jnp

    rec, _ = drv._evaluate(
        problem, reference, jnp.asarray(result.design),
        result.final_alpha_max, result.final_beta,
    )
    assert rec["J_self"] == pytest.approx(result.terminal["J_self"], rel=1e-12)
    assert rec["constraint_g"] == pytest.approx(
        result.terminal["constraint_g"], rel=1e-12
    )


def test_terminal_uses_the_model_the_run_actually_ended_in(setup):
    """Not phases[-1]: a budget-stopped run never reaches the last phase.

    Evaluating the final design under a phase the run never entered compares
    two different models and produces a meaningless jump.
    """
    problem, reference = setup
    phases = drv.r1d_schedule(SPEC, 25)
    result = drv.run(problem, reference, phases, move_limit=0.1, budget=4)

    assert result.stop_reason == "budget_exhausted"
    assert result.final_phase == result.history[-1]["phase"]
    assert result.final_alpha_max == result.history[-1]["alpha_max"]
    assert result.final_beta == result.history[-1]["beta"]
    assert result.terminal["beta"] == result.history[-1]["beta"]
    # the run stopped inside the first phase, so it is NOT the last phase
    assert result.final_phase != phases[-1].name


def test_budget_exhaustion_is_not_reported_as_convergence(setup):
    """Upstream MMA sets is_converged on max_iter; the driver must not echo it."""
    problem, reference = setup
    phases = drv.r1d_schedule(SPEC, 25)
    result = drv.run(problem, reference, phases, move_limit=0.1, budget=3)
    assert result.stop_reason == "budget_exhausted"
    assert len(result.history) == 3


def test_every_iteration_records_its_model_and_residuals(setup):
    problem, reference = setup
    phases = drv.r1d_schedule(SPEC, 6)
    result = drv.run(problem, reference, phases, move_limit=0.1)
    for rec in result.history:
        assert rec["flow_residual_relative"] < 1e-8
        assert rec["thermal_residual_relative"] < 1e-8
        assert {"alpha_max", "beta", "phase", "model_changed"} <= set(rec)
    # the first iteration has nothing to compare against
    assert result.history[0]["model_changed"] is False


def test_both_normalisations_are_reported(setup):
    problem, reference = setup
    result = drv.run(problem, reference, drv.r1d_schedule(SPEC, 6), move_limit=0.1)
    rec = result.history[0]
    assert rec["psi_over_psi0_self"] == pytest.approx(
        rec["psi"] / reference.psi_0, rel=1e-12
    )
    assert rec["psi_over_paper"] == pytest.approx(
        rec["psi"] / drv.PAPER_PSI_0_INTERPRETED, rel=1e-12
    )
    assert rec["c_over_paper"] == pytest.approx(
        rec["compliance"] / drv.PAPER_C_0_INTERPRETED, rel=1e-12
    )
