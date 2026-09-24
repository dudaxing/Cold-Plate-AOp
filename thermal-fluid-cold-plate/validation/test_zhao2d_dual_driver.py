"""R1j: the dual-mesh model's own reference, and the driver running that model.

Each test targets one way the optimisation chain could describe two models at
once without anything looking wrong:

  - a reference whose identity leaves out the thermal mesh, accepted for it;
  - a refusal that comes only after the states were solved;
  - the driver forming C with the flow mesh's velocity, gating one solve while
    reporting another, or returning a gradient of a different function;
  - a stale denominator, which J = 1 at the reference state exposes at once.
"""

import dataclasses
import json

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tfopus import materials  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_driver as drv  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

# 10 x 20 design + 2 x (2 x 2) tabs = 208 flow cells, 832 thermal; wiring only.
SPEC = dataclasses.replace(z.Zhao2DSpec(), element_size=5.0e-4)
CONFIG = r1.R1Config()
STEPS = (1e-4, 1e-5, 1e-6)
TOL = 1e-5  # the existing gradient-check threshold, not re-tuned here


@pytest.fixture(scope="module")
def frozen():
    values, report, problem = dual.freeze_reference(
        SPEC, CONFIG, thermal_refinement=2, thermal_quadrature=3)
    return values, report, problem


def _grey(problem, seed=5):
    return jnp.asarray(np.random.default_rng(seed).uniform(0.15, 0.85, problem.num_design))


def _with_identity(values, **thermal_mesh):
    identity = json.loads(values.identity)
    identity["thermal_mesh"].update(thermal_mesh)
    return dataclasses.replace(values, identity=json.dumps(identity, sort_keys=True))


# -- the reference ----------------------------------------------------------------


def test_the_reference_is_this_models_and_reproduces_itself(frozen):
    values, report, problem = frozen
    assert json.loads(values.identity)["thermal_mesh"] == {
        "refinement": 2, "quadrature": 3, "element_length_mode": "min_edge"}
    assert max(report["residual_flow"], report["residual_thermal"]) < 1e-8
    problem.check_reference(values)

    # both ratios, separately, exactly 1 on a re-solve of the reference state
    s = z.reference_solid_fraction(problem.flow_mesh, SPEC, CONFIG.reference_field)
    psi, c = problem.metrics(s, CONFIG.alpha_max_reference)
    assert float(psi) / values.psi_0 == pytest.approx(1.0, rel=1e-12)
    assert float(c) / values.c_0 == pytest.approx(1.0, rel=1e-12)

    # the flow side is the single-mesh model's, so Psi_0 is too; C_0 is not
    single, _ = r1.freeze_reference(SPEC, CONFIG)
    assert values.psi_0 == pytest.approx(single.psi_0, rel=1e-13)
    assert abs(values.c_0 / single.c_0 - 1.0) > 1e-3


def test_the_reference_round_trips_and_loads_only_for_its_own_model(frozen, tmp_path):
    values, _, problem = frozen
    path = tmp_path / "reference.json"
    path.write_text(values.to_json(), encoding="utf-8")
    assert dual.load_reference(problem, path) == values

    for other in (dual.Zhao2DDualProblem(SPEC, CONFIG, 4, 3),
                  dual.Zhao2DDualProblem(SPEC, CONFIG, 2, 2),
                  dual.Zhao2DDualProblem(dataclasses.replace(SPEC, heat_source=2.0e8),
                                         CONFIG, 2, 3)):
        with pytest.raises(ValueError, match="not frozen for the dual-mesh"):
            dual.load_reference(other, path)


def test_the_driver_refuses_another_models_reference_before_any_solve(frozen):
    values, _, _ = frozen
    problem = dual.Zhao2DDualProblem(SPEC, CONFIG, 2, 3)
    problem.solve_states = lambda *a, **k: pytest.fail("solved before refusing")
    single, _ = r1.freeze_reference(SPEC, CONFIG)
    x = jnp.full(problem.num_design, 0.6)
    for wrong in (single, _with_identity(values, refinement=4),
                  _with_identity(values, quadrature=2)):
        with pytest.raises(ValueError, match="not frozen for the dual-mesh"):
            drv.evaluate(problem, wrong, x, 1.0e6, 0.0)
        with pytest.raises(ValueError, match="not frozen for the dual-mesh"):
            drv.run(problem, wrong, drv.r1d_schedule(SPEC, 6))


# -- the driver's one evaluation ------------------------------------------------------


def test_the_driver_reports_the_dual_model_from_one_gated_solve(frozen):
    values, _, problem = frozen
    x = _grey(problem)
    beta, alpha_max = 8.0, 1.0e7
    record, (s, press_vel, temperature), dj, dg = drv.evaluate(
        problem, values, x, alpha_max, beta)

    # C lives on the thermal mesh, and the record is recomputable from the states
    assert temperature.shape == (problem.thermal_mesh.mesh.num_dofs,)
    alpha = materials.brinkman_penalty(jnp.asarray(s), za.build_material(SPEC, alpha_max))
    kappa_t = problem.thermal_conductivity(jnp.asarray(s), alpha_max)
    pv, temp = jnp.asarray(press_vel), jnp.asarray(temperature)
    assert record["psi"] == pytest.approx(
        float(problem.flow.dissipated_power(pv, alpha)), rel=1e-13)
    assert record["compliance"] == pytest.approx(float(problem.thermal.thermal_compliance(
        temp, problem.thermal_velocity(pv), kappa_t)), rel=1e-13)
    # the gate ran on THESE states
    norms = problem.residual_norms_at(pv, temp, alpha, kappa_t)
    assert record["flow_residual_relative"] == pytest.approx(norms["flow"], rel=1e-12)
    assert record["thermal_residual_relative"] == pytest.approx(norms["thermal"], rel=1e-12)

    # the direct API on the same design: the same numbers and the same gradient
    psi, c = problem.metrics(problem.solid_fraction(x, beta), alpha_max)
    assert record["psi"] == pytest.approx(float(psi), rel=1e-12)
    assert record["compliance"] == pytest.approx(float(c), rel=1e-12)
    w = CONFIG.weight

    def j_direct(v):
        p, cc = problem.metrics(problem.solid_fraction(v, beta), alpha_max)
        return w * p / values.psi_0 + (1.0 - w) * cc / values.c_0

    g_direct = np.asarray(jax.grad(j_direct)(x))
    assert np.allclose(np.asarray(dj), g_direct, rtol=1e-10,
                       atol=1e-12 * np.abs(g_direct).max())

    # and against central differences taken through the driver itself
    direction = jnp.asarray(np.random.default_rng(13).choice([-1.0, 1.0], problem.num_design))
    ad = float(jnp.dot(dj, direction))
    best = np.inf
    for h in STEPS:
        jp = drv.evaluate(problem, values, x + h * direction, alpha_max, beta,
                          gradient=False)[0]["J_self"]
        jm = drv.evaluate(problem, values, x - h * direction, alpha_max, beta,
                          gradient=False)[0]["J_self"]
        fd = (jp - jm) / (2 * h)
        best = min(best, abs(ad - fd) / abs(fd))
    assert best < TOL


def test_a_short_dual_run_starts_at_one_and_pairs_its_terminal_record(frozen):
    """The loop on the dual model, six updates on a 208-cell mesh: wiring only.

    Six is the smallest budget whose alpha phase reaches the cap before beta
    moves. MMA starts from x = 1 - gamma_ref with beta = 0 and alpha_max = 1e6,
    which IS the reference state, so the first J must be exactly 1 -- the
    cheapest detector of a denominator frozen for some other model.
    """
    values, _, problem = frozen
    result = drv.run(problem, values, drv.r1d_schedule(SPEC, 6), move_limit=0.1)
    assert result.history[0]["J_self"] == pytest.approx(1.0, rel=1e-12)
    assert len(result.history) == 6
    assert all(r["thermal_residual_relative"] < 1e-8 for r in result.history)
    assert result.temperature.shape == (problem.thermal_mesh.mesh.num_dofs,)

    rec, _ = drv._evaluate(problem, values, jnp.asarray(result.design),
                           result.final_alpha_max, result.final_beta)
    assert rec["J_self"] == pytest.approx(result.terminal["J_self"], rel=1e-12)
