"""R1r's prerequisite: the optimisation entry on the check layer, on a small mesh.

`zhao2d_fineflow.run` is the driver's MMA loop with `zhao2d_fineflow.evaluate`
on a declared common scale. Each test targets one way that wiring could be
silently wrong:

  - MMA handed a design of the wrong length -- the flow or the design mesh's
    cell count instead of the coarse design variables -- or a J, gradient or
    constraint that is not the evaluation's own;
  - a state that fails the gate reaching MMA, or the gate run on a different
    solve from the one that is used;
  - a wrong scale, a wrong start, or a failed zero-step anchor getting past
    the first update;
  - a terminal that is not evaluated at the saved design, or a proxy stop
    reported as convergence;
  - the old entry quietly accepting a common scale;
  - R1r's script going past a zero step that does not reproduce R1q's main
    point.
"""

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_driver as drv  # noqa: E402
from tfopus import zhao2d_fineflow as ff  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

# 10 x 20 design + 2 x (2 x 2) tabs = 208 design-mesh cells, 200 of them design
# variables; flow 832 cells, thermal 3328.
SPEC = dataclasses.replace(z.Zhao2DSpec(), element_size=5.0e-4)
CONFIG = r1.R1Config(projection=r1.Projection.VOLUME_PRESERVING)
ALPHA_MAX, BETA = 1.0e7, 32.0  # R1r's


@pytest.fixture(scope="module")
def problem():
    return ff.Zhao2DFineFlowProblem(SPEC, CONFIG, flow_refinement=2, thermal_refinement=2,
                                    thermal_quadrature=3)


@pytest.fixture(scope="module")
def development_reference(tmp_path_factory):
    """A development-model reference for this spec, as a file; its values only scale J."""
    values = r1.ReferenceValues(
        psi_0=0.02, c_0=3.0e3,
        identity=ff.development_identity(SPEC, CONFIG, 4, 3),
        run_fingerprint="test", spec_element_size=SPEC.element_size,
        flow_residual_relative=0.0, thermal_residual_relative=0.0)
    path = tmp_path_factory.mktemp("reference") / "development.json"
    path.write_text(values.to_json(), encoding="utf-8")
    return values, path


@pytest.fixture(scope="module")
def scale(problem, development_reference):
    return ff.common_scale(problem, development_reference[1], 4, 3)


def _grey(n, seed):
    return np.random.default_rng(seed).uniform(0.15, 0.85, n)


def _fixed(iterations):
    return [drv.Phase("fixed", iterations, beta=BETA, alpha_max=ALPHA_MAX)]


def _spy_on_updates(monkeypatch):
    """Every MMA update's inputs, in order; upstream's update still runs."""
    seen = []
    update = drv._mma.update_mma

    def spy(state, params, j, dj, g, dg):
        seen.append({"n": params.num_design_var, "j": j, "dj": np.array(dj).ravel(),
                     "g": np.array(g).ravel(), "dg": np.array(dg).ravel()})
        return update(state, params, j, dj, g, dg)

    monkeypatch.setattr(drv._mma, "update_mma", spy)
    return seen


def test_mma_gets_the_coarse_design_and_each_evaluations_own_j_and_gradients(
        problem, scale, monkeypatch):
    n = problem.num_design
    assert n == int(problem.design_mesh.design_mask.sum()) == 200
    assert n < problem.design_mesh.num_elems < problem.flow_mesh.num_elems
    x0 = _grey(n, 3)
    seen = _spy_on_updates(monkeypatch)
    result = ff.run(problem, scale, _fixed(2), move_limit=0.1, initial_design=x0)

    # the design vector is the coarse design's; the states are on their own meshes
    assert [u["n"] for u in seen] == [n, n]
    assert result.designs.shape == (2, n) and result.design.shape == (n,)
    assert np.array_equal(result.designs[0], x0) and np.array_equal(result.initial_design, x0)
    assert result.solid_fraction.shape == (problem.design_mesh.num_elems,)
    assert result.press_vel.shape == (problem.flow_mesh.mesh.num_dofs,)
    assert result.temperature.shape == (problem.thermal_mesh.mesh.num_dofs,)

    # MMA is handed each record's own J on the common scale and its own g
    for update, rec in zip(seen, result.history):
        assert "J_self" not in rec
        assert update["j"] == rec["J_common_scale"]
        assert update["g"].tolist() == [rec["constraint_g"]]
    # ... and, at the start, exactly the gradients the entry gives
    rec0, _, grads = ff.evaluate(problem, scale, jnp.asarray(x0), ALPHA_MAX, BETA)
    assert rec0["J_common_scale"] == pytest.approx(result.history[0]["J_common_scale"], rel=1e-12)
    for key, got in (("J", seen[0]["dj"]), ("g", seen[0]["dg"])):
        assert np.allclose(got, grads[key], rtol=1e-10, atol=1e-12 * np.abs(grads[key]).max())


def test_a_real_proxy_stop_still_evaluates_the_terminal_at_the_same_point(problem, scale):
    """A move limit of 1e-9 keeps the second step under upstream's step_tol."""
    result = ff.run(problem, scale, _fixed(5), move_limit=1e-9, initial_design=_grey(problem.num_design, 4))

    assert result.stop_reason == "proxy_criterion"
    assert result.proxy_criterion == "step_tol"
    assert result.proxy_fired_at_final_stage is True
    # upstream measures a step only from its second update on
    assert len(result.history) == 2
    assert result.history[-1]["proxy_criterion"] == "step_tol"
    assert result.terminal.get("terminal") is True and result.terminal["iteration"] == 2

    rec, (s, pv, temp), grads = ff.evaluate(problem, scale, jnp.asarray(result.design),
                                            result.final_alpha_max, result.final_beta,
                                            gradient=False)
    assert grads is None
    assert rec["J_common_scale"] == pytest.approx(result.terminal["J_common_scale"], rel=1e-12)
    assert rec["constraint_g"] == pytest.approx(result.terminal["constraint_g"], rel=1e-12, abs=1e-15)
    assert np.allclose(result.solid_fraction, s, rtol=1e-12, atol=1e-15)
    assert np.allclose(result.temperature, temp, rtol=1e-12, atol=1e-12)


def test_a_wrong_scale_or_start_is_refused_before_any_solve(
        problem, scale, development_reference, monkeypatch):
    monkeypatch.setattr(problem, "solve_states", lambda *a, **k: pytest.fail("solved before refusing"))
    n = problem.num_design
    x0 = _grey(n, 5)
    other_model = dataclasses.replace(scale, target_identity=ff.development_identity(SPEC, CONFIG, 4, 3))
    edited_source = dataclasses.replace(scale, source_identity=scale.source_identity.replace(
        '"refinement": 4', '"refinement": 2'))
    assert edited_source.source_identity != scale.source_identity
    for bad, match in ((other_model, "another model"), (edited_source, "source")):
        with pytest.raises(ValueError, match=match):
            ff.run(problem, bad, _fixed(1), initial_design=x0)
    for bad_x in (np.full(problem.design_mesh.num_elems, 0.5),  # s's length: the tabs as well
                  np.full(problem.flow_mesh.num_elems, 0.5),    # the flow mesh's cells
                  np.full(n, 1.0 + 1e-12)):
        with pytest.raises(ValueError, match="initial_design"):
            ff.run(problem, scale, _fixed(1), initial_design=bad_x)

    # the old entry takes neither the development reference nor the common scale
    with pytest.raises(ValueError, match="not frozen for the fine-flow model"):
        drv.run(problem, development_reference[0], _fixed(1), initial_design=x0)
    with pytest.raises(TypeError, match="CommonScale"):
        drv.run(problem, scale, _fixed(1), initial_design=x0)


def test_a_failed_zero_step_anchor_stops_before_any_update(problem, scale, monkeypatch):
    """How R1r checks its zero step: `on_iteration` raises, and MMA never moves."""
    seen = _spy_on_updates(monkeypatch)

    class Mismatch(RuntimeError):
        pass

    def anchor(record):
        if record["iteration"] == 0:
            raise Mismatch(f"J {record['J_common_scale']} is not the anchor's")

    with pytest.raises(Mismatch):
        ff.run(problem, scale, _fixed(3), move_limit=0.1, on_iteration=anchor,
               initial_design=_grey(problem.num_design, 7))
    assert seen == []


def test_the_gate_runs_on_the_same_solve_and_a_failed_state_never_reaches_mma(
        problem, scale, monkeypatch):
    """The second iterate's returned temperature is spoiled by 1%: the gate on
    the states that solve returned refuses it, with no second solve in its place."""
    solve = problem.solve_states
    calls = []

    def second_temperature_spoiled(s, alpha_max):
        calls.append(1)
        pv, temp, alpha, kappa = solve(s, alpha_max)
        return (pv, temp * 1.01, alpha, kappa) if len(calls) == 2 else (pv, temp, alpha, kappa)

    monkeypatch.setattr(problem, "solve_states", second_temperature_spoiled)
    seen = _spy_on_updates(monkeypatch)
    n = problem.num_design
    with pytest.raises(r1.NotConverged, match="thermal") as info:
        ff.run(problem, scale, _fixed(3), move_limit=0.1, initial_design=_grey(n, 6))

    assert len(calls) == 2  # one solve per iterate, and none after the refusal
    assert len(seen) == 1  # MMA moved once, on the first iterate only
    partial = info.value.partial
    assert len(partial["history"]) == 1 and partial["failed_iteration"] == 1
    assert partial["designs"].shape == (1, n) and partial["failed_design"].shape == (n,)


# -- the main-mesh script ------------------------------------------------------------


def test_r1r_stops_at_a_failed_zero_step_before_any_update(tmp_path, monkeypatch):
    """A zero step that does not reproduce R1q's main point stops the run with
    its record written, before MMA's first update."""
    import json
    import pathlib
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    import zhao2d_r1r_d_optimise as rr

    root = pathlib.Path(__file__).resolve().parent.parent / "results"
    rows = json.loads((root / "zhao2d_r1m_flow_check.json").read_text(encoding="utf-8"))["rows"]
    q = json.loads((root / "zhao2d_r1q_fineflow_check.json").read_text(encoding="utf-8"))
    main = q["main_point"]["evaluation"]

    flow_mesh, thermal_mesh = SimpleNamespace(num_elems=20800), SimpleNamespace(num_elems=332800)
    fake = SimpleNamespace(
        design_mesh=z.build_mesh(z.Zhao2DSpec(), dofs_per_node=3), flow_mesh=flow_mesh,
        thermal_mesh=thermal_mesh, num_design=5000, nesting={}, model_identity=lambda: "{}",
        projection_root=lambda x, beta: {"eta": main["projection_eta"], "slope": -4.0e-5,
                                         "nondegenerate": True},
        fluid_fraction=lambda x, beta: (main["constraint_g"] + 1.0) * CONFIG.max_fluid_fraction)
    updates = []

    def fake_run(problem, scale, phases, move_limit, on_iteration, initial_design):
        assert problem is fake and phases[0].iterations == 20 and phases[0].beta == 32.0
        on_iteration({**main, "psi": main["psi"] * (1.0 + 1e-9), "iteration": 0, "seconds": 0.0})
        updates.append(1)
        pytest.fail("the run went on past a zero step that does not reproduce R1q's main point")

    monkeypatch.setattr(rr.sys, "argv", ["r1r", "--out", str(tmp_path)])
    monkeypatch.setattr(rr.faulthandler, "dump_traceback_later", lambda *a, **k: None)
    monkeypatch.setattr(rr.ff, "Zhao2DFineFlowProblem", lambda *a, **k: fake)
    monkeypatch.setattr(rr, "mesh_identity", lambda planar: (
        rows["flow_h2"]["flow_mesh"] if planar is flow_mesh else rows["flow_h"]["flow_mesh"]))
    monkeypatch.setattr(rr, "thermal_identity", lambda problem: rows["flow_h2"]["thermal_mesh"])
    monkeypatch.setattr(rr.ff, "common_scale", lambda *a, **k: SimpleNamespace(
        **{key: q["scale"][key] for key in ("psi_0", "c_0", "source_file", "source_sha256",
                                            "source_model")}))
    monkeypatch.setattr(rr.ff, "run", fake_run)

    with pytest.raises(SystemExit, match="CHECKS FAILED at the zero step"):
        rr.main()
    assert updates == []
    record = json.loads((tmp_path / rr.RECORD).read_text(encoding="utf-8"))
    assert record["stopped"].startswith("the zero step against R1q's main point")
    assert record["zero_step_anchor"]["reproduced"] is False
    assert [c["stage"] for c in record["checkpoints"]] == [
        "inputs", "the model and the common scale", "the zero step's map",
        "the zero step against R1q's main point"]
    assert len(record["history"]) == 1
    assert not (tmp_path / rr.FIELDS).exists()
