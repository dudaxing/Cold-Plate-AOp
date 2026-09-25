"""The optimisation driver: schedule, terminal pairing, stop reasons, warm starts.

These are the things a silent bug would turn into a plausible-looking result.
"""

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tfopus import zhao2d as z, zhao2d_driver as drv, zhao2d_r1 as r1  # noqa: E402
from tfopus import materials, zhao2d_analysis as za  # noqa: E402

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


# -- warm start and stop reasons (the prerequisites of R1k) --------------------


def _fixed(iterations):
    return [drv.Phase("fixed", iterations, beta=8.0, alpha_max=1.0e7)]


def test_run_starts_from_the_given_initial_design(setup, monkeypatch):
    """The first evaluation must receive exactly the design passed in.

    Without the argument run() starts MMA from 1 - gamma_ref everywhere, so a
    script that loads a saved design and then calls run() would not start there.
    """
    problem, reference = setup
    x0 = np.random.default_rng(3).uniform(0.2, 0.8, problem.num_design)
    seen = []
    original = drv.evaluate

    def spy(problem_, reference_, x, *args, **kwargs):
        seen.append(np.array(x))
        return original(problem_, reference_, x, *args, **kwargs)

    monkeypatch.setattr(drv, "evaluate", spy)
    result = drv.run(problem, reference, _fixed(2), move_limit=0.1, initial_design=x0)

    assert np.array_equal(seen[0], x0)
    assert np.array_equal(result.initial_design, x0)
    # every record is paired with the design it was evaluated at
    assert len(result.designs) == len(result.history) == 2
    assert np.array_equal(result.designs[0], x0)
    assert np.array_equal(result.designs[1], seen[1])
    # and the initial state is the one history[0] was computed on: its metrics
    # recompute from it exactly as the record states them
    material = za.build_material(SPEC, 1.0e7)
    s0 = jnp.asarray(result.initial_solid_fraction)
    pv0, t0 = jnp.asarray(result.initial_press_vel), jnp.asarray(result.initial_temperature)
    assert result.history[0]["psi"] == pytest.approx(float(problem.flow.dissipated_power(
        pv0, materials.brinkman_penalty(s0, material))), rel=1e-13)
    assert result.history[0]["compliance"] == pytest.approx(float(
        problem.thermal.thermal_compliance(t0, problem.thermal_velocity(pv0),
                                           materials.conductivity(s0, material))), rel=1e-13)
    # A re-solve agrees only to rounding: s traced under value_and_grad and s
    # from a plain forward pass differ by an ulp in some entries (measured:
    # 1.1e-16 in 103 of 208), and the states and J by ~1e-15 relative.
    rec, (s, press_vel, temperature) = drv._evaluate(
        problem, reference, jnp.asarray(x0), 1.0e7, 8.0)
    assert rec["J_self"] == pytest.approx(result.history[0]["J_self"], rel=1e-12)
    assert np.allclose(result.initial_solid_fraction, s, rtol=1e-12, atol=1e-15)
    assert np.allclose(result.initial_press_vel, press_vel, rtol=1e-12, atol=1e-12)
    assert np.allclose(result.initial_temperature, temperature, rtol=1e-12, atol=1e-12)


def test_run_refuses_a_bad_initial_design_before_any_solve(setup, monkeypatch):
    """Refused, never clipped or resampled -- and before anything is solved."""
    problem, reference = setup
    monkeypatch.setattr(problem, "solve_states",
                        lambda *a, **k: pytest.fail("solved before refusing"))
    n = problem.num_design
    for bad in (np.full(problem.flow_mesh.num_elems, 0.5),  # s's length, not x's
                np.full((n, 2), 0.5),
                np.where(np.arange(n) == 7, np.nan, 0.5),
                np.full(n, 1.0 + 1e-12),
                np.full(n, -1e-12)):
        with pytest.raises(ValueError, match="initial_design"):
            drv.run(problem, reference, _fixed(1), initial_design=bad)


def test_a_proxy_criterion_is_not_reported_as_convergence(setup, monkeypatch):
    """Upstream's step_tol and mixed-point KKT are proxies, and the run says so."""
    problem, reference = setup
    monkeypatch.setattr(drv, "_mma_proxy_criterion", lambda state, params: "kkt_tol")
    result = drv.run(problem, reference, _fixed(3), move_limit=0.1)

    assert result.stop_reason == "proxy_criterion"
    assert result.proxy_criterion == "kkt_tol"
    assert result.proxy_fired_at_final_stage is True
    assert len(result.history) == 1
    assert result.history[0]["proxy_criterion"] == "kkt_tol"
    assert result.terminal["iteration"] == 1


def test_a_failed_gate_keeps_the_run_so_far(setup, monkeypatch):
    """A state that fails the gate stops the run -- with its history, not without."""
    problem, reference = setup
    norms = problem.residual_norms_at
    calls = []

    def second_state_fails(*args, **kwargs):
        calls.append(1)
        out = norms(*args, **kwargs)
        return {k: 1.0 for k in out} if len(calls) == 2 else out

    monkeypatch.setattr(problem, "residual_norms_at", second_state_fails)
    with pytest.raises(r1.NotConverged) as info:
        drv.run(problem, reference, _fixed(3), move_limit=0.1)

    partial = info.value.partial
    assert len(partial["history"]) == 1
    assert partial["designs"].shape == (1, problem.num_design)
    assert partial["failed_iteration"] == 1
    assert partial["failed_design"].shape == (problem.num_design,)


def test_no_gradient_at_a_degenerate_projection_root(setup, monkeypatch):
    """Uniform x = 0 filters to all zeros: the volume-preserving root is
    degenerate and the design map has no derivative. The driver refuses to
    hand MMA a gradient there, before solving anything; a value still works."""
    problem, reference = setup
    x = jnp.zeros(problem.num_design)
    root = problem.projection_root(x, 8.0)
    assert root["nondegenerate"] is False and root["slope"] == 0.0

    solve = problem.solve_states
    monkeypatch.setattr(problem, "solve_states",
                        lambda *a, **k: pytest.fail("solved before refusing"))
    with pytest.raises(r1.DegenerateProjection):
        drv.evaluate(problem, reference, x, 1.0e7, 8.0)

    monkeypatch.setattr(problem, "solve_states", solve)
    record, _, dj, dg = drv.evaluate(problem, reference, x, 1.0e7, 8.0, gradient=False)
    assert dj is None and dg is None
    assert record["projection_root_slope"] == 0.0


def test_records_carry_the_projection_and_its_root(setup):
    problem, reference = setup
    x = jnp.asarray(np.random.default_rng(6).uniform(0.2, 0.8, problem.num_design))
    record, _, _, _ = drv.evaluate(problem, reference, x, 1.0e7, 8.0, gradient=False)
    assert record["projection"] == r1.Projection.VOLUME_PRESERVING
    assert 0.0 < record["projection_eta"] < 1.0
    assert record["projection_root_slope"] < 0.0
