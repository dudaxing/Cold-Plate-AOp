"""R1s: the temperature by one linear solve, against upstream's Newton loop.

`thermal_path="linear"` must give the same discrete model's state, objectives
and design gradient as the default path. Each test targets one way it could be
silently wrong:

  - the thermal residual not affine in T after all, or its tangent depending
    on T, so that one solve would not be the solution;
  - a state or an objective that differs from the Newton path's by more than
    round-off;
  - a transpose solve that is not the forward solve's transpose (the thermal
    matrix is not symmetric, so a symmetric shortcut would fail here, not
    pass);
  - a design gradient that drops the flow's or tau's dependence, or disagrees
    with finite differences;
  - a bad linear state getting past the gate, or the default path changing;
  - R1s's script building the model after its inputs fail to reconcile.
"""

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import scipy.sparse as sp

jax.config.update("jax_enable_x64", True)

import toflux.src.solver as tf_solver  # noqa: E402

from tfopus import affine_solve as affine  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_fineflow as ff  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

# 208 design-mesh cells (200 variables), flow 832 cells, thermal 3328.
SPEC = dataclasses.replace(z.Zhao2DSpec(), element_size=5.0e-4)
CONFIG = r1.R1Config(projection=r1.Projection.VOLUME_PRESERVING)
STEPS = (1e-4, 1e-5, 1e-6)
TOL = 1e-5  # the existing gradient-check threshold, not re-tuned here
GATE = 1e-8
ALPHA_MAX, BETA = 1.0e7, 8.0


@pytest.fixture(scope="module")
def newton():
    return ff.Zhao2DFineFlowProblem(SPEC, CONFIG, 2, 2, 3)


@pytest.fixture(scope="module")
def linear():
    return ff.Zhao2DFineFlowProblem(SPEC, CONFIG, 2, 2, 3, thermal_path="linear")


@pytest.fixture(scope="module")
def scale(linear, tmp_path_factory):
    """A development-model reference for this spec, as a file; its values only scale J."""
    values = r1.ReferenceValues(
        psi_0=0.02, c_0=3.0e3, identity=ff.development_identity(SPEC, CONFIG, 4, 3),
        run_fingerprint="test", spec_element_size=SPEC.element_size,
        flow_residual_relative=0.0, thermal_residual_relative=0.0)
    path = tmp_path_factory.mktemp("reference") / "development.json"
    path.write_text(values.to_json(), encoding="utf-8")
    return ff.common_scale(linear, path, 4, 3)


def _grey(n, seed):
    return jnp.asarray(np.random.default_rng(seed).uniform(0.15, 0.85, n))


def _flow(problem, seed=9, beta=BETA):
    """A design's density, its flow on the flow mesh, and the thermal mesh's inputs."""
    s = problem.solid_fraction(_grey(problem.num_design, seed), beta)
    alpha = problem.flow_material(s, ALPHA_MAX)
    pv = tf_solver.modified_newton_raphson_solve(problem.flow, problem.flow_x0, alpha)
    return s, pv, problem.thermal_velocity(pv), problem.thermal_conductivity(s, ALPHA_MAX)


def test_the_default_is_newton_and_an_unknown_path_is_refused(newton, linear, scale):
    assert newton.thermal_path == "newton" and linear.thermal_path == "linear"
    with pytest.raises(ValueError, match="thermal_path"):
        ff.Zhao2DFineFlowProblem(SPEC, CONFIG, 2, 2, 3, thermal_path="lu")
    # a solver choice, not a model change: one identity, one common scale
    assert linear.model_identity() == newton.model_identity()
    newton.check_common_scale(scale)


def test_the_thermal_residual_is_affine_in_t_with_a_tangent_that_does_not_depend_on_it(linear):
    _, _, vel, kappa = _flow(linear)
    x0 = linear.thermal_x0
    free = np.ones(x0.shape, bool)
    free[np.asarray(linear.thermal_bc["fixed_dofs"])] = False
    rng = np.random.default_rng(1)
    t1, t2 = (x0.at[free].set(rng.uniform(0.0, 20.0, int(free.sum()))) for _ in range(2))
    a = 0.3
    r_1, k_1 = linear.thermal.get_residual_and_tangent_stiffness(t1, vel, kappa, linear.q_source)
    r_2, k_2 = linear.thermal.get_residual_and_tangent_stiffness(t2, vel, kappa, linear.q_source)
    r_m, _ = linear.thermal.get_residual_and_tangent_stiffness(a * t1 + (1 - a) * t2, vel, kappa,
                                                              linear.q_source)
    assert float(jnp.max(jnp.abs(r_m - (a * r_1 + (1 - a) * r_2)))) <= 1e-13 * float(
        jnp.max(jnp.abs(r_1)))
    assert np.array_equal(k_1.data, k_2.data) and np.array_equal(k_1.indices, k_2.indices)


@pytest.mark.parametrize("beta", [8.0, 32.0])
def test_the_linear_path_gives_the_newton_paths_state_and_objectives(newton, linear, beta):
    s = newton.solid_fraction(_grey(newton.num_design, 3), beta)
    pv_n, t_n, alpha_n, kappa_n = newton.solve_states(s, ALPHA_MAX)
    pv_l, t_l, alpha_l, kappa_l = linear.solve_states(s, ALPHA_MAX)

    assert np.array_equal(pv_n, pv_l)  # the flow's path is unchanged
    assert float(jnp.max(jnp.abs(t_l - t_n)) / jnp.max(jnp.abs(t_n))) <= 1e-12
    for problem, state in ((newton, (pv_n, t_n, alpha_n, kappa_n)),
                           (linear, (pv_l, t_l, alpha_l, kappa_l))):
        assert all(v <= GATE for v in problem.residual_norms_at(*state).values())
    c_n = float(newton.thermal.thermal_compliance(t_n, newton.thermal_velocity(pv_n), kappa_n))
    c_l = float(linear.thermal.thermal_compliance(t_l, linear.thermal_velocity(pv_l), kappa_l))
    assert c_l == pytest.approx(c_n, rel=1e-12)
    assert float(linear.flow.dissipated_power(pv_l, alpha_l)) == float(
        newton.flow.dissipated_power(pv_n, alpha_n))


def test_the_transpose_solve_is_the_forward_solves_transpose(newton, linear):
    """<w, J v> = <J^T w, v> through the velocity and the conductivity together."""
    _, _, vel, kappa = _flow(linear)
    rng = np.random.default_rng(4)
    v = (jnp.asarray(rng.standard_normal(vel.shape)), jnp.asarray(rng.standard_normal(kappa.shape)))
    w = jnp.asarray(rng.standard_normal(linear.thermal_x0.shape))

    def temperature(solve, vel_, kappa_):
        return solve(linear.thermal, linear.thermal_x0, vel_, kappa_, linear.q_source)

    lin = lambda a, b: temperature(affine.affine_solve, a, b)  # noqa: E731
    _, jv = jax.jvp(lin, (vel, kappa), v)
    _, vjp = jax.vjp(lin, vel, kappa)
    jtw = vjp(w)
    lhs = float(jnp.dot(w, jv))
    rhs = float(jnp.dot(jtw[0].ravel(), v[0].ravel()) + jnp.dot(jtw[1], v[1]))
    assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), abs(rhs))

    # the same directional derivative as upstream's rule on the Newton path
    _, jv_newton = jax.jvp(lambda a, b: temperature(tf_solver.modified_newton_raphson_solve, a, b),
                           (vel, kappa), v)
    assert float(jnp.max(jnp.abs(jv - jv_newton))) <= 1e-10 * float(jnp.max(jnp.abs(jv_newton)))

    # and the matrix is far from symmetric, so a symmetric shortcut would show
    _, k = linear.thermal.get_residual_and_tangent_stiffness(linear.thermal_x0, vel, kappa,
                                                            linear.q_source)
    m = sp.coo_matrix((np.asarray(k.data), (np.asarray(k.indices[:, 0]), np.asarray(k.indices[:, 1]))),
                      shape=k.shape).tocsr()
    assert sp.linalg.norm(m - m.T) / sp.linalg.norm(m) > 1e-2


@pytest.mark.parametrize("seed", [11, 12])
def test_the_linear_paths_design_gradient_matches_newton_and_finite_differences(
        newton, linear, scale, seed):
    """Psi, C, g and J through E_DF, the fine flow, P_FT, E_FT, tau(k, u) and the one solve."""
    x0 = _grey(linear.num_design, seed)
    direction = jnp.asarray(np.random.default_rng(seed).choice([-1.0, 1.0], linear.num_design))
    rec_n, _, g_n = ff.evaluate(newton, scale, x0, ALPHA_MAX, BETA)
    rec_l, _, g_l = ff.evaluate(linear, scale, x0, ALPHA_MAX, BETA)
    assert rec_n["thermal_path"] == "newton" and rec_l["thermal_path"] == "linear"
    for key in g_n:
        assert np.allclose(g_l[key], g_n[key], rtol=1e-9, atol=1e-12 * np.abs(g_n[key]).max()), key

    keys = {"psi": "psi", "c": "compliance", "g": "constraint_g", "J": "J_common_scale"}
    ad = {q: float(np.dot(g_l[q], np.asarray(direction))) for q in keys}
    best = {q: np.inf for q in keys}
    for h in STEPS:
        plus = ff.evaluate(linear, scale, x0 + h * direction, ALPHA_MAX, BETA, gradient=False)[0]
        minus = ff.evaluate(linear, scale, x0 - h * direction, ALPHA_MAX, BETA, gradient=False)[0]
        for q, key in keys.items():
            fd = (plus[key] - minus[key]) / (2 * h)
            best[q] = min(best[q], abs(ad[q] - fd) / max(abs(fd), 1e-300))
    assert max(best.values()) < TOL, best


def test_the_linear_path_keeps_the_flows_dependence_on_the_design(linear, scale):
    """Freezing the flow moves dC, and finite differences side with the full chain."""
    x0 = _grey(linear.num_design, 21)
    direction = jnp.asarray(np.random.default_rng(22).choice([-1.0, 1.0], linear.num_design))

    def c_full(v):
        return linear.metrics(linear.solid_fraction(v, BETA), ALPHA_MAX)[1]

    def c_frozen_flow(v):
        s = linear.solid_fraction(v, BETA)
        pv = jax.lax.stop_gradient(linear.solve_states(s, ALPHA_MAX)[0])
        temp = linear.solve_thermal(pv, s, ALPHA_MAX)
        return linear.thermal.thermal_compliance(
            temp, linear.thermal_velocity(pv), linear.thermal_conductivity(s, ALPHA_MAX))

    full = float(jnp.dot(jax.grad(c_full)(x0), direction))
    frozen = float(jnp.dot(jax.grad(c_frozen_flow)(x0), direction))
    h = 1e-5
    fd = (ff.evaluate(linear, scale, x0 + h * direction, ALPHA_MAX, BETA, gradient=False)[0]["compliance"]
          - ff.evaluate(linear, scale, x0 - h * direction, ALPHA_MAX, BETA,
                        gradient=False)[0]["compliance"]) / (2 * h)
    assert abs(full - fd) / abs(fd) < TOL
    assert abs(frozen - fd) / abs(fd) > 100 * TOL


def test_a_bad_linear_state_is_refused_by_the_gate(linear, scale, monkeypatch):
    """The gate judges the state the one solve returned, with no second solve in its place."""
    original = linear._thermal_state
    calls = []

    def spoiled(vel, kappa):
        calls.append(1)
        return original(vel, kappa) * 1.001

    monkeypatch.setattr(linear, "_thermal_state", spoiled)
    with pytest.raises(r1.NotConverged, match="thermal"):
        ff.evaluate(linear, scale, _grey(linear.num_design, 5), ALPHA_MAX, BETA)
    assert len(calls) == 1


# -- the main-mesh script ------------------------------------------------------------


def test_r1s_stops_at_a_failed_input_before_building_the_model(tmp_path, monkeypatch):
    """R1q's saved main point and R1r's saved start must be the same arrays; one
    value off by 1e-9 stops the run, with its record written, before any build."""
    import json
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    import zhao2d_r1s_linear_thermal as rs

    original = rs.load_fields

    def tampered(path):
        fields = original(path)
        if pathlib.Path(path).name == "zhao2d_r1r_fields.npz":
            t = fields["initial_temperature"].copy()
            t[123] += 1e-9
            fields["initial_temperature"] = t
        return fields

    monkeypatch.setattr(rs.sys, "argv", ["r1s", "--out", str(tmp_path)])
    monkeypatch.setattr(rs.faulthandler, "dump_traceback_later", lambda *a, **k: None)
    monkeypatch.setattr(rs, "load_fields", tampered)
    monkeypatch.setattr(rs.ff, "Zhao2DFineFlowProblem",
                        lambda *a, **k: pytest.fail("built the model after a failed input check"))
    with pytest.raises(SystemExit, match="CHECKS FAILED at inputs"):
        rs.main()
    record = json.loads((tmp_path / rs.RECORD).read_text(encoding="utf-8"))
    assert record["stopped"].startswith("inputs")
    assert [c["stage"] for c in record["checkpoints"]] == ["inputs"]
    assert any("initial_temperature" in f for f in record["checkpoints"][0]["failures"])
    assert not (tmp_path / rs.FIELDS).exists()
