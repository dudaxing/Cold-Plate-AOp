"""R1t's prerequisite: MMA's own state, saved and restored, on a small mesh.

Before R1t every warm start kept only the design and reinitialised MMA. Each
test targets one way keeping MMA's state could be silently wrong:

  - a resumed run that does not make the uninterrupted run's updates -- a
    pause turned into a reinitialisation;
  - the KKT residual MMA actually writes (`kktnorm`, not the declared
    `kkt_norm` that upstream's `to_array` serialises) lost on the way;
  - a checkpoint restored into another run: another thermal path, scale,
    move limit, schedule or budget -- or another optimisation problem on the
    same reference identity: weight, volume bound or domain, projection,
    filter;
  - a paused run evaluating, and passing off as a state, a design MMA has
    not yet been given back;
  - R1t's script going past a zero step that does not reproduce R1r's
    terminal.
"""

import dataclasses

import numpy as np
import jax
import pytest

jax.config.update("jax_enable_x64", True)

import toflux.src.mma as tf_mma  # noqa: E402

from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_driver as drv  # noqa: E402
from tfopus import zhao2d_fineflow as ff  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

# 5 x 10 design + 2 tabs = 52 design-mesh cells, 50 of them variables; flow
# 208 cells, thermal 832.
SPEC = dataclasses.replace(z.Zhao2DSpec(), element_size=1.0e-3)
CONFIG = r1.R1Config(projection=r1.Projection.VOLUME_PRESERVING)
ALPHA_MAX, BETA = 1.0e7, 32.0  # R1t's


@pytest.fixture(scope="module")
def linear():
    return ff.Zhao2DFineFlowProblem(SPEC, CONFIG, 2, 2, 3, thermal_path="linear")


@pytest.fixture(scope="module")
def newton():
    return ff.Zhao2DFineFlowProblem(SPEC, CONFIG, 2, 2, 3)


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
    return np.random.default_rng(seed).uniform(0.15, 0.85, n)


def _fixed(iterations, beta=BETA):
    return [drv.Phase("fixed", iterations, beta=beta, alpha_max=ALPHA_MAX)]


def test_four_updates_equal_two_then_a_saved_and_loaded_state_then_two(linear, scale, tmp_path):
    assert linear.num_design == 50
    x0 = _grey(linear.num_design, 3)
    whole = ff.run(linear, scale, _fixed(4), move_limit=0.1, initial_design=x0)

    first = ff.run(linear, scale, _fixed(4), move_limit=0.1, initial_design=x0, stop_after=2)
    assert first.stop_reason == "paused" and first.terminal is None
    assert first.mma.updates_done == 2 and len(first.history) == 2

    path = tmp_path / "mma.npz"
    np.savez(path, **first.mma.to_npz())
    with np.load(path) as data:
        loaded = drv.MMACheckpoint.from_npz(data)
    assert np.array_equal(loaded.state_array, first.mma.state_array)
    assert (loaded.kktnorm, loaded.updates_done, loaded.num_design_var, loaded.binding) == (
        first.mma.kktnorm, first.mma.updates_done, first.mma.num_design_var, first.mma.binding)

    second = ff.run(linear, scale, _fixed(4), move_limit=0.1, resume=loaded)
    assert np.array_equal(second.initial_design, first.design)

    # the uninterrupted run's designs, records, terminal and MMA state, bit for bit
    assert np.array_equal(np.vstack([first.designs, second.designs]), whole.designs)
    assert np.array_equal(second.design, whole.design)
    joined = first.history + second.history
    assert [r["iteration"] for r in joined] == [0, 1, 2, 3]
    for a, b in zip(joined, whole.history):
        for key in ("J_common_scale", "constraint_g", "kkt_proxy", "design_step_norm"):
            assert a[key] == b[key], key
    assert second.terminal["iteration"] == whole.terminal["iteration"] == 4
    assert second.terminal["J_common_scale"] == whole.terminal["J_common_scale"]
    assert np.array_equal(second.mma.state_array, whole.mma.state_array)
    assert second.mma.kktnorm == whole.mma.kktnorm and second.mma.updates_done == 4

    # and the test can tell: MMA reinitialised at the same design goes elsewhere
    restart = ff.run(linear, scale, _fixed(2), move_limit=0.1, initial_design=first.design)
    assert not np.array_equal(restart.design, whole.design)


def test_the_kkt_residual_mma_writes_is_kept_beside_the_declared_fields(linear, scale):
    run = ff.run(linear, scale, _fixed(3), move_limit=0.1,
                 initial_design=_grey(linear.num_design, 4), stop_after=2)
    cp = run.mma
    assert np.isfinite(cp.kktnorm) and cp.kktnorm == run.history[-1]["kkt_proxy"]
    # upstream's array alone keeps the declared kkt_norm, which update_mma never writes
    bare = tf_mma.MMAState.from_array(cp.state_array.copy(), cp.num_design_var)
    assert drv._kkt_norm(bare) == 1000.0 != cp.kktnorm
    restored = cp.restore(cp.binding)
    assert drv._kkt_norm(restored) == cp.kktnorm and restored.epoch == 2


def test_a_checkpoint_is_refused_under_any_other_binding_before_any_solve(
        linear, newton, scale, monkeypatch):
    x0 = _grey(linear.num_design, 5)
    cp = ff.run(linear, scale, _fixed(3), move_limit=0.1, initial_design=x0, stop_after=1).mma
    for problem in (linear, newton):
        monkeypatch.setattr(problem, "solve_states",
                            lambda *a, **k: pytest.fail("solved before refusing"))
    cases = {
        "another thermal path": (newton, scale, _fixed(3), 0.1),
        "another scale": (linear, dataclasses.replace(scale, psi_0=1.5 * scale.psi_0), _fixed(3), 0.1),
        "another move limit": (linear, scale, _fixed(3), 0.2),
        "another budget": (linear, scale, _fixed(4), 0.1),
        "another beta": (linear, scale, _fixed(3, beta=16.0), 0.1),
    }
    for what, (problem, sc, phases, move) in cases.items():
        with pytest.raises(ValueError, match="another run"):
            ff.run(problem, sc, phases, move_limit=move, resume=cp)
    with pytest.raises(ValueError, match="initial_design"):
        ff.run(linear, scale, _fixed(3), move_limit=0.1, resume=cp, initial_design=x0)
    with pytest.raises(ValueError, match="binding"):
        drv.run_loop(linear, lambda *a, **k: pytest.fail("evaluated"), _fixed(3), resume=cp)



def test_a_checkpoint_is_refused_when_the_optimisation_problem_changes(linear, scale, monkeypatch):
    """The weight, the volume bound and domain, the projection and the filter are
    left out of the reference's identity on purpose; the checkpoint's binding
    carries them (the review of 0650e7b found them missing)."""
    cp = ff.run(linear, scale, _fixed(3), move_limit=0.1,
                initial_design=_grey(linear.num_design, 7), stop_after=1).mma
    variants = {
        "weight": dataclasses.replace(CONFIG, weight=0.6),
        "volume bound": dataclasses.replace(CONFIG, max_fluid_fraction=0.35),
        "volume domain": dataclasses.replace(CONFIG, volume_domain=r1.VolumeDomain.WHOLE),
        "projection": dataclasses.replace(CONFIG, projection=r1.Projection.TANH),
        "filter radius": dataclasses.replace(CONFIG, filter_radius_elements=3.0),
    }
    for what, config in variants.items():
        variant = ff.Zhao2DFineFlowProblem(SPEC, config, 2, 2, 3, thermal_path="linear")
        # the reference's identity cannot see the change, so the scale still serves ...
        assert variant.model_identity() == linear.model_identity(), what
        variant.check_common_scale(scale)
        # ... and the binding must
        assert ff.run_binding(variant, scale) != ff.run_binding(linear, scale), what
        monkeypatch.setattr(variant, "solve_states",
                            lambda *a, _what=what, **k: pytest.fail(f"{_what}: solved before refusing"))
        with pytest.raises(ValueError, match="another run"):
            ff.run(variant, scale, _fixed(3), move_limit=0.1, resume=cp)

def test_a_pause_solves_nothing_past_its_last_update(linear, scale, monkeypatch):
    solve, calls = linear.solve_states, []

    def counted(s, alpha_max):
        calls.append(1)
        return solve(s, alpha_max)

    monkeypatch.setattr(linear, "solve_states", counted)
    paused = ff.run(linear, scale, _fixed(4), move_limit=0.1,
                    initial_design=_grey(linear.num_design, 6), stop_after=2)
    assert len(calls) == 2
    assert paused.stop_reason == "paused" and paused.terminal is None
    assert paused.solid_fraction is None and paused.temperature is None

    calls.clear()
    done = ff.run(linear, scale, _fixed(2), move_limit=0.1, initial_design=_grey(linear.num_design, 6))
    assert len(calls) == 3  # two iterates and the terminal
    assert done.terminal["terminal"] is True and done.mma.updates_done == 2
    assert np.array_equal(done.mma.state_array[:done.mma.num_design_var], done.design)


# -- the main-mesh script ------------------------------------------------------------


def test_r1t_stops_at_a_failed_zero_step_before_any_update(tmp_path, monkeypatch):
    """A zero step whose C is off R1r's terminal by 1e-7 -- inside a looser
    criterion, outside R1s's 1e-8 across the thermal paths -- stops the run,
    with its record written, before MMA's first update."""
    import json
    import pathlib
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    import zhao2d_r1t_d_continue as rt

    root = pathlib.Path(__file__).resolve().parent.parent / "results"
    rows = json.loads((root / "zhao2d_r1m_flow_check.json").read_text(encoding="utf-8"))["rows"]
    q = json.loads((root / "zhao2d_r1q_fineflow_check.json").read_text(encoding="utf-8"))
    term = json.loads((root / "zhao2d_r1r.json").read_text(encoding="utf-8"))["terminal"]

    flow_mesh, thermal_mesh = SimpleNamespace(num_elems=20800), SimpleNamespace(num_elems=332800)
    fake = SimpleNamespace(
        design_mesh=z.build_mesh(z.Zhao2DSpec(), dofs_per_node=3), flow_mesh=flow_mesh,
        thermal_mesh=thermal_mesh, num_design=5000, nesting={}, model_identity=lambda: "{}",
        thermal_path="linear",
        projection_root=lambda x, beta: {"eta": term["projection_eta"], "slope": -4.0e-5,
                                         "nondegenerate": True},
        fluid_fraction=lambda x, beta: (term["constraint_g"] + 1.0) * CONFIG.max_fluid_fraction)
    updates = []

    def fake_run(problem, scale, phases, move_limit, on_iteration, initial_design):
        assert problem is fake and phases[0].iterations == 20 and phases[0].beta == 32.0
        on_iteration({**term, "compliance": term["compliance"] * (1.0 + 1e-7),
                      "iteration": 0, "seconds": 0.0})
        updates.append(1)
        pytest.fail("the run went on past a zero step that does not reproduce R1r's terminal")

    monkeypatch.setattr(rt.sys, "argv", ["r1t", "--out", str(tmp_path)])
    monkeypatch.setattr(rt.faulthandler, "dump_traceback_later", lambda *a, **k: None)
    monkeypatch.setattr(rt.ff, "Zhao2DFineFlowProblem", lambda *a, **k: fake)
    monkeypatch.setattr(rt, "mesh_identity", lambda planar: (
        rows["flow_h2"]["flow_mesh"] if planar is flow_mesh else rows["flow_h"]["flow_mesh"]))
    monkeypatch.setattr(rt, "thermal_identity", lambda problem: rows["flow_h2"]["thermal_mesh"])
    monkeypatch.setattr(rt.ff, "common_scale", lambda *a, **k: SimpleNamespace(
        **{key: q["scale"][key] for key in ("psi_0", "c_0", "source_file", "source_sha256",
                                            "source_model")}))
    monkeypatch.setattr(rt.ff, "run", fake_run)

    with pytest.raises(SystemExit, match="CHECKS FAILED at the zero step"):
        rt.main()
    assert updates == []
    record = json.loads((tmp_path / rt.RECORD).read_text(encoding="utf-8"))
    assert record["stopped"].startswith("the zero step against R1r's terminal")
    anchor = record["zero_step_anchor"]
    assert anchor["reproduced"] is False
    assert anchor["compliance_relative"] == pytest.approx(1e-7, rel=1e-6)
    assert [c["stage"] for c in record["checkpoints"]] == [
        "inputs", "the model and the common scale", "the zero step's map",
        "the zero step against R1r's terminal"]
    assert not (tmp_path / rt.FIELDS).exists()
