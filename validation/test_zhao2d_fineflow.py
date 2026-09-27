"""R1q: the check layer as a differentiable model of the coarse design.

Each test targets one way the new chain could be silently wrong:

  - the design side moved onto the flow mesh -- four times the variables and
    half the filter's physical radius;
  - a density that does not reach the flow and thermal meshes by parents, or a
    transpose that averages children instead of summing them;
  - a chain that, with the flow and design meshes coinciding, is not the
    existing dual-mesh model;
  - a model that, for a given design, does not give R1m's check-layer states;
  - the development reference accepted as this model's normalisation, or a
    common scale accepted on a model it was not made for;
  - a gradient that freezes the flow, splits wrongly into Psi and C, or
    disagrees with finite differences.
"""

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_fineflow as ff  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402
from tfopus import zhao2d_refine as ref  # noqa: E402

# 10 x 20 design + 2 x (2 x 2) tabs = 208 design cells; flow 832, thermal 3328.
SPEC = dataclasses.replace(z.Zhao2DSpec(), element_size=5.0e-4)
STEPS = (1e-4, 1e-5, 1e-6)
TOL = 1e-5  # the existing gradient-check threshold, not re-tuned here
ALPHA_MAX, BETA = 1.0e7, 8.0


@pytest.fixture(scope="module")
def problem():
    return ff.Zhao2DFineFlowProblem(SPEC, r1.R1Config(), flow_refinement=2,
                                    thermal_refinement=2, thermal_quadrature=3)


@pytest.fixture(scope="module")
def coarse():
    return r1.Zhao2DProblem(SPEC, r1.R1Config())


@pytest.fixture(scope="module")
def development_reference(tmp_path_factory):
    """A development-model reference for this spec, as a file; its values only scale J."""
    values = r1.ReferenceValues(
        psi_0=0.02, c_0=3.0e3,
        identity=ff.development_identity(SPEC, r1.R1Config(), 4, 3),
        run_fingerprint="test", spec_element_size=SPEC.element_size,
        flow_residual_relative=0.0, thermal_residual_relative=0.0)
    path = tmp_path_factory.mktemp("reference") / "development.json"
    path.write_text(values.to_json(), encoding="utf-8")
    return values, path


@pytest.fixture(scope="module")
def scale(problem, development_reference):
    return ff.common_scale(problem, development_reference[1], 4, 3)


def _grey(n, seed=5):
    return jnp.asarray(np.random.default_rng(seed).uniform(0.15, 0.85, n))


# -- the design side -----------------------------------------------------------


def test_the_design_side_stays_on_the_design_mesh(problem, coarse):
    """The same variables, filter, projection and constraint as the model on h."""
    assert problem.num_design == coarse.num_design
    assert problem.design_mesh.num_elems == coarse.flow_mesh.num_elems
    assert problem.flow_mesh.num_elems == 4 * coarse.flow_mesh.num_elems
    assert problem.thermal_mesh.num_elems == 16 * coarse.flow_mesh.num_elems
    x = _grey(problem.num_design)
    assert np.array_equal(np.asarray(problem.filter_matrix @ x),
                          np.asarray(coarse.filter_matrix @ x))
    assert np.array_equal(np.asarray(problem.solid_fraction(x, BETA)),
                          np.asarray(coarse.solid_fraction(x, BETA)))
    assert float(problem.fluid_fraction(x, BETA)) == float(coarse.fluid_fraction(x, BETA))
    assert problem.projection_root(x, BETA) == coarse.projection_root(x, BETA)

    # the trap the review named: the existing model on h/2 moves the design side
    trap = r1.Zhao2DProblem(ref.refine_spec(SPEC, 2), r1.R1Config())
    assert trap.num_design == 4 * problem.num_design


def test_density_reaches_the_flow_and_thermal_meshes_by_parents(problem):
    s = problem.solid_fraction(_grey(problem.num_design), BETA)
    s_f = np.asarray(problem.flow_density(s))
    s_t = np.asarray(problem.thermal_density(s))
    to_flow = ref.parent_of_each_fine_element(problem.design_mesh, problem.flow_mesh)
    to_thermal = ref.parent_of_each_fine_element(problem.design_mesh, problem.thermal_mesh)
    assert np.array_equal(s_f, np.asarray(s)[to_flow])
    assert np.array_equal(s_t, np.asarray(s)[to_thermal])

    on_design = z.fluid_fractions(problem.design_mesh, s)
    for mesh, dens in ((problem.flow_mesh, s_f), (problem.thermal_mesh, s_t)):
        on_fine = z.fluid_fractions(mesh, dens)
        for k, v in on_design.items():
            assert on_fine[k] == pytest.approx(v, rel=1e-13)
        assert np.all(dens[~mesh.design_mask] == 0.0), "the tabs picked up material"
    n = problem.nesting
    assert n["heat_source_thermal_mesh"] == pytest.approx(n["heat_source_design_mesh"], rel=1e-13)
    assert n["design_cells_flow_mesh"] == 4 * n["design_cells"]


def test_the_transposes_sum_children(problem):
    """E^T must ACCUMULATE: averaging would scale the density gradient by 1/r^2."""
    rng = np.random.default_rng(3)
    s = jnp.asarray(rng.uniform(size=problem.design_mesh.num_elems))
    _, to_flow = jax.vjp(problem.flow_density, s)
    _, to_thermal = jax.vjp(problem.thermal_density, s)
    assert np.all(np.asarray(to_flow(jnp.ones(problem.flow_mesh.num_elems))[0]) == 4)
    assert np.all(np.asarray(to_thermal(jnp.ones(problem.thermal_mesh.num_elems))[0]) == 16)

    e_mat, _ = problem.design_to_flow.matrices()
    ct = rng.normal(size=problem.flow_mesh.num_elems)
    back = np.asarray(to_flow(jnp.asarray(ct))[0])
    assert np.allclose(back, e_mat.T @ ct, rtol=1e-13, atol=1e-13)
    a = rng.normal(size=problem.design_mesh.num_elems)
    assert float(np.dot(np.asarray(problem.flow_density(a)), ct)) == pytest.approx(
        float(np.dot(a, back)), rel=1e-12)


# -- the chain against the existing models ---------------------------------------


def test_flow_refinement_one_is_the_dual_model():
    """With the design and flow meshes coinciding, values and gradients are the old ones."""
    old = dual.Zhao2DDualProblem(SPEC, r1.R1Config(), thermal_refinement=2, thermal_quadrature=3)
    new = ff.Zhao2DFineFlowProblem(SPEC, r1.R1Config(), flow_refinement=1,
                                   thermal_refinement=2, thermal_quadrature=3)
    x = _grey(old.num_design)
    s = old.solid_fraction(x, BETA)
    assert np.array_equal(np.asarray(new.solid_fraction(x, BETA)), np.asarray(s))
    psi_o, c_o = old.metrics(s, ALPHA_MAX)
    psi_n, c_n = new.metrics(s, ALPHA_MAX)
    assert float(psi_n) == pytest.approx(float(psi_o), rel=1e-13)
    assert float(c_n) == pytest.approx(float(c_o), rel=1e-12)

    def c_of(p):
        return lambda v: p.metrics(p.solid_fraction(v, BETA), ALPHA_MAX)[1]

    g_o = np.asarray(jax.grad(c_of(old))(x))
    g_n = np.asarray(jax.grad(c_of(new))(x))
    assert np.allclose(g_n, g_o, rtol=1e-10, atol=1e-10 * np.abs(g_o).max())


def test_a_given_design_gives_r1ms_check_layer_states(problem):
    """R1m solved a given design on a dual model built on h/2, the material copied
    down from the parent cell; this model must give the same states."""
    fine = dual.Zhao2DDualProblem(ref.refine_spec(SPEC, 2), r1.R1Config(),
                                  thermal_refinement=2, thermal_quadrature=3)
    s_d = problem.solid_fraction(_grey(problem.num_design, 7), BETA)
    s_f = ref.refine_design(problem.design_mesh, fine.flow_mesh, s_d)
    pv_new, t_new, _, _ = problem.solve_states(s_d, ALPHA_MAX)
    pv_old, t_old, _, _ = fine.solve_states(s_f, ALPHA_MAX)
    assert np.allclose(np.asarray(pv_new), np.asarray(pv_old), rtol=1e-13, atol=1e-15)
    assert np.allclose(np.asarray(t_new), np.asarray(t_old), rtol=1e-13, atol=1e-15)


# -- the scale -------------------------------------------------------------------


def test_the_development_reference_is_a_common_scale_not_a_normalisation(
        problem, development_reference, scale, tmp_path):
    values, _ = development_reference
    dev = dual.Zhao2DDualProblem(SPEC, r1.R1Config(), thermal_refinement=4, thermal_quadrature=3)
    assert ff.development_identity(SPEC, r1.R1Config(), 4, 3) == dev.reference_identity()

    problem.check_common_scale(scale)
    assert scale.target_identity == problem.model_identity() != scale.source_identity

    # check_reference stays strict: the development reference is not this model's
    with pytest.raises(ValueError, match="not frozen for the fine-flow model"):
        problem.check_reference(values)
    with pytest.raises(ValueError, match="not frozen for the fine-flow model"):
        problem.objective_and_constraint(jnp.full(problem.num_design, 0.6), values, 1.0e6)
    problem.check_reference(dataclasses.replace(values, identity=problem.reference_identity()))

    # a scale made for another model is refused, before anything is solved
    other = ff.Zhao2DFineFlowProblem(SPEC, r1.R1Config(), flow_refinement=1,
                                     thermal_refinement=2, thermal_quadrature=3)
    with pytest.raises(ValueError, match="made for another model"):
        ff.evaluate(other, scale, _grey(other.num_design), ALPHA_MAX, BETA, gradient=False)

    # a file that is not the development model's reference cannot become a scale
    wrong = dataclasses.replace(values, identity=ff.development_identity(SPEC, r1.R1Config(), 2, 3))
    path = tmp_path / "wrong.json"
    path.write_text(wrong.to_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="not the development model's reference"):
        ff.common_scale(problem, path, 4, 3)

    # nor can a scale whose source identity was edited afterwards
    with pytest.raises(ValueError, match="source is not the development model"):
        problem.check_common_scale(dataclasses.replace(scale, source_identity=wrong.identity))


# -- the gradient ------------------------------------------------------------------


def test_the_entry_uses_one_solve_and_splits_the_gradient_correctly(problem, scale):
    x = _grey(problem.num_design, 9)
    rec, (s, pv, temp), grads = ff.evaluate(problem, scale, x, ALPHA_MAX, BETA)

    # the record is the returned states' own evaluation
    s, pv, temp = jnp.asarray(s), jnp.asarray(pv), jnp.asarray(temp)
    alpha = problem.flow_material(s, ALPHA_MAX)
    kappa = problem.thermal_conductivity(s, ALPHA_MAX)
    assert float(problem.flow.dissipated_power(pv, alpha)) == pytest.approx(rec["psi"], rel=1e-13)
    assert float(problem.thermal.thermal_compliance(
        temp, problem.thermal_velocity(pv), kappa)) == pytest.approx(rec["compliance"], rel=1e-13)
    w = problem.config.weight
    assert rec["J_common_scale"] == pytest.approx(
        w * rec["psi"] / scale.psi_0 + (1 - w) * rec["compliance"] / scale.c_0, rel=1e-14)

    # the two reverse passes are the metrics' own gradients, and dJ is their sum
    def metric(i):
        return lambda v: problem.metrics(problem.solid_fraction(v, BETA), ALPHA_MAX)[i]

    for key, i in (("psi", 0), ("c", 1)):
        separate = np.asarray(jax.grad(metric(i))(x))
        assert np.allclose(grads[key], separate, rtol=1e-10, atol=1e-10 * np.abs(separate).max())
    assert np.allclose(grads["J"], w * grads["psi"] / scale.psi_0 + (1 - w) * grads["c"] / scale.c_0,
                       rtol=1e-14, atol=0.0)
    g_direct = np.asarray(jax.grad(
        lambda v: problem.fluid_fraction(v, BETA) / problem.config.max_fluid_fraction - 1.0)(x))
    assert np.array_equal(grads["g"], g_direct)


@pytest.mark.parametrize("seed", [11, 12])
def test_the_total_gradient_matches_finite_differences(problem, scale, seed):
    """Psi, C, g and J, through E_DF, the fine flow, P_FT, E_FT and tau(k, u)."""
    rng = np.random.default_rng(seed)
    x0 = _grey(problem.num_design, seed)
    direction = jnp.asarray(rng.choice([-1.0, 1.0], problem.num_design))
    _, _, grads = ff.evaluate(problem, scale, x0, ALPHA_MAX, BETA)
    keys = {"psi": "psi", "c": "compliance", "g": "constraint_g", "J": "J_common_scale"}
    ad = {q: float(np.dot(grads[q], np.asarray(direction))) for q in keys}
    best = {q: np.inf for q in keys}
    for h in STEPS:
        plus = ff.evaluate(problem, scale, x0 + h * direction, ALPHA_MAX, BETA, gradient=False)[0]
        minus = ff.evaluate(problem, scale, x0 - h * direction, ALPHA_MAX, BETA, gradient=False)[0]
        for q, key in keys.items():
            fd = (plus[key] - minus[key]) / (2 * h)
            best[q] = min(best[q], abs(ad[q] - fd) / max(abs(fd), 1e-300))
    assert max(best.values()) < TOL, best


def test_the_gradient_keeps_the_flows_dependence_on_the_design(problem, scale):
    """Freezing the flow changes dC, and finite differences side with the full chain.

    Were a saved or stopped flow standing in for the solved one, the reverse
    pass would give the frozen derivative and disagree with finite differences.
    """
    x0 = _grey(problem.num_design, 21)
    direction = jnp.asarray(np.random.default_rng(22).choice([-1.0, 1.0], problem.num_design))

    def c_full(v):
        return problem.metrics(problem.solid_fraction(v, BETA), ALPHA_MAX)[1]

    def c_frozen_flow(v):
        s = problem.solid_fraction(v, BETA)
        pv = jax.lax.stop_gradient(problem.solve_states(s, ALPHA_MAX)[0])
        temp = problem.solve_thermal(pv, s, ALPHA_MAX)
        return problem.thermal.thermal_compliance(
            temp, problem.thermal_velocity(pv), problem.thermal_conductivity(s, ALPHA_MAX))

    full = float(jnp.dot(jax.grad(c_full)(x0), direction))
    frozen = float(jnp.dot(jax.grad(c_frozen_flow)(x0), direction))
    h = 1e-5
    fd = (ff.evaluate(problem, scale, x0 + h * direction, ALPHA_MAX, BETA, gradient=False)[0]["compliance"]
          - ff.evaluate(problem, scale, x0 - h * direction, ALPHA_MAX, BETA,
                        gradient=False)[0]["compliance"]) / (2 * h)
    assert abs(full - fd) / abs(fd) < TOL
    assert abs(frozen - fd) / abs(fd) > 100 * TOL, (
        "freezing the flow barely moved the derivative, so this test cannot see "
        "whether the flow's dependence is present")


# -- the main-mesh check script ------------------------------------------------------


def test_r1q_stops_at_a_failed_anchor_before_evaluating_the_main_point(tmp_path, monkeypatch):
    """A given binary design that does not reproduce its record stops the run
    before the main point is evaluated -- the run's first solve."""
    import json
    import pathlib
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    import zhao2d_r1q_fineflow_check as rq

    root = pathlib.Path(__file__).resolve().parent.parent / "results"
    rows = json.loads((root / "zhao2d_r1m_flow_check.json").read_text(encoding="utf-8"))["rows"]
    recs = {name: json.loads((root / file).read_text(encoding="utf-8"))
            for name, file in (("r1n", "zhao2d_r1n_beta32.json"), ("r1o", "zhao2d_r1o.json"))}
    fields = {name: np.load(root / file)
              for name, file in (("r1n", "zhao2d_r1n_fields.npz"), ("r1o", "zhao2d_r1o_fields.npz"))}
    by_design = {fields[n]["solid_fraction_binary"].tobytes(): n for n in fields}
    by_flow = {fields[n]["new_press_vel_check"].tobytes(): n for n in fields}

    def which_flow(pv):
        return by_flow[np.asarray(pv).tobytes()]

    def psi(pv, alpha):
        n = which_flow(pv)
        return recs[n]["cells"]["new/check"]["psi"] * (1.0 + 1e-9 if n == "r1o" else 1.0)

    flow_mesh = SimpleNamespace(num_elems=20800)
    fake = SimpleNamespace(
        design_mesh=z.build_mesh(z.Zhao2DSpec(), dofs_per_node=3), flow_mesh=flow_mesh,
        thermal_mesh=SimpleNamespace(num_elems=332800), num_design=5000, nesting={},
        model_identity=lambda: "{}",
        flow_density=lambda s: fields[by_design[np.asarray(s).tobytes()]]["new_solid_fraction_flow_h2"],
        flow_material=lambda s, a: None, thermal_conductivity=lambda s, a: None,
        thermal_velocity=which_flow,
        flow=SimpleNamespace(dissipated_power=psi),
        thermal=SimpleNamespace(thermal_compliance=lambda t, n, k: recs[n]["cells"]["new/check"]["compliance"]),
        residual_norms_at=lambda *a: {"flow": 1e-14, "thermal": 2e-11},
        projection_root=lambda *a: pytest.fail("reached the main point after an anchor failed"))
    yard = recs["r1n"]["yardstick"]

    monkeypatch.setattr(rq.sys, "argv", ["r1q", "--out", str(tmp_path)])
    monkeypatch.setattr(rq.faulthandler, "dump_traceback_later", lambda *a, **k: None)
    monkeypatch.setattr(rq.ff, "Zhao2DFineFlowProblem", lambda *a, **k: fake)
    monkeypatch.setattr(rq, "mesh_identity", lambda planar: (
        rows["flow_h2"]["flow_mesh"] if planar is flow_mesh else rows["flow_h"]["flow_mesh"]))
    monkeypatch.setattr(rq, "thermal_identity", lambda problem: rows["flow_h2"]["thermal_mesh"])
    monkeypatch.setattr(rq.ff, "common_scale", lambda *a, **k: SimpleNamespace(
        psi_0=yard["psi_0"], c_0=yard["c_0"], source_file="f", source_sha256="h", source_model={}))
    monkeypatch.setattr(rq.fs, "verify_flow_state",
                        lambda problem, pv, s, alpha_max, tol: {"residual_relative": 0.0})
    monkeypatch.setattr(rq.ff, "evaluate",
                        lambda *a, **k: pytest.fail("evaluated the main point after an anchor failed"))
    with pytest.raises(SystemExit) as stop:
        rq.main()
    assert "nothing after it was solved" in str(stop.value.code)
    record = json.loads((tmp_path / rq.RECORD).read_text(encoding="utf-8"))
    assert [c["stage"] for c in record["checkpoints"]] == [
        "inputs", "identities and the common scale",
        "anchors: the given binary designs, nothing solved"]
    assert [f.split(":")[0] for f in record["checkpoints"][-1]["failures"]] == ["r1o"]
    assert record["anchors"]["r1n"]["anchor"]["reproduced"]
    assert not (tmp_path / rq.FIELDS).exists()
