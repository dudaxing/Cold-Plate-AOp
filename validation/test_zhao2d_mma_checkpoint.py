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

And R1v's two operations, on the same small mesh:

  - budget appended to a finished run that does not make the longer run's
    updates -- or that changes a step already scheduled, the run it names, or
    any number of MMA's state but the budget's flag; a proxy stop mistaken
    for the budget's;
  - a legacy checkpoint migrated into anything but today's binding for the
    run it came from, or with its state touched; and R1t's own migration
    writing over R1t's files.
"""

import dataclasses
import hashlib
import json
import pathlib
import sys

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


# -- R1v: budget appended to a finished run ------------------------------------------


@pytest.fixture(scope="module")
def start(linear):
    return _grey(linear.num_design, 8)


@pytest.fixture(scope="module")
def whole(linear, scale, start):
    """Four updates at once: what a finished two with two appended must be."""
    return ff.run(linear, scale, _fixed(4), move_limit=0.1, initial_design=start)


@pytest.fixture(scope="module")
def finished(linear, scale, start):
    """Two updates of a budget of two, from the same start, terminal evaluated."""
    return ff.run(linear, scale, _fixed(2), move_limit=0.1, initial_design=start)


def _state(checkpoint):
    return tf_mma.MMAState.from_array(checkpoint.state_array.copy(), checkpoint.num_design_var)


def _same_updates(first, second, whole):
    """first + second made whole's updates, bit for bit, and ended in its state."""
    assert np.array_equal(np.vstack([first.designs, second.designs]), whole.designs)
    joined = first.history + second.history
    assert [r["iteration"] for r in joined] == [r["iteration"] for r in whole.history]
    for a, b in zip(joined, whole.history):
        for key in ("J_common_scale", "constraint_g", "kkt_proxy", "design_step_norm"):
            assert a[key] == b[key], key
    assert np.array_equal(second.design, whole.design)
    assert second.terminal["J_common_scale"] == whole.terminal["J_common_scale"]
    assert np.array_equal(second.mma.state_array, whole.mma.state_array)
    assert second.mma.kktnorm == whole.mma.kktnorm
    assert second.mma.binding == whole.mma.binding


def test_a_finished_run_with_two_appended_makes_the_four_updates_of_one_run(
        linear, scale, finished, whole):
    assert finished.stop_reason == "phase_end" and finished.terminal is not None
    done = finished.mma
    assert done.updates_done == 2 and _state(done).is_converged  # the budget's flag, epoch 2
    before = (done.state_array.copy(), done.kktnorm, done.binding)

    # R1t's position before R1v: its own binding has nothing left, a larger one is another run
    with pytest.raises(RuntimeError, match="no iteration completed"):
        ff.run(linear, scale, _fixed(2), move_limit=0.1, resume=done)
    with pytest.raises(ValueError, match="another run"):
        ff.run(linear, scale, _fixed(4), move_limit=0.1, resume=done)

    more = done.append_budget(_fixed(4), 4)
    assert np.array_equal(done.state_array, before[0])  # the old checkpoint is left as it was
    assert (done.kktnorm, done.binding) == before[1:]
    changed = np.flatnonzero(more.state_array != done.state_array)
    assert changed.size == 1 and not _state(more).is_converged  # the flag, and nothing else
    assert (more.kktnorm, more.updates_done) == (done.kktnorm, 2) and _state(more).epoch == 2
    grown, old = json.loads(more.binding), json.loads(done.binding)
    assert (grown["budget"], grown["mma"]["max_iter"], len(grown["schedule"])) == (4, 4, 4)
    assert grown["schedule"][:2] == old["schedule"] and grown["entry"] == old["entry"]

    cont = ff.run(linear, scale, _fixed(4), move_limit=0.1, resume=more)
    assert np.array_equal(cont.initial_design, finished.design)
    # its first iterate is the finished run's terminal, now with a gradient
    assert cont.history[0]["J_common_scale"] == pytest.approx(
        finished.terminal["J_common_scale"], rel=1e-12, abs=0.0)
    assert cont.terminal["iteration"] == 4 and cont.mma.updates_done == 4
    _same_updates(finished, cont, whole)


def test_a_pause_inside_the_appended_budget_is_the_uninterrupted_runs_pause(
        linear, scale, start, finished, whole):
    more = finished.mma.append_budget(_fixed(4), 4)
    paused = ff.run(linear, scale, _fixed(4), move_limit=0.1, resume=more, stop_after=1)
    straight = ff.run(linear, scale, _fixed(4), move_limit=0.1, initial_design=start, stop_after=3)
    assert paused.stop_reason == straight.stop_reason == "paused"
    assert paused.mma.updates_done == straight.mma.updates_done == 3
    # the flag included: the appended run did not carry the old budget's flag along
    assert np.array_equal(paused.mma.state_array, straight.mma.state_array)
    assert paused.mma.kktnorm == straight.mma.kktnorm and paused.mma.binding == straight.mma.binding
    rest = ff.run(linear, scale, _fixed(4), move_limit=0.1, resume=paused.mma)
    assert np.array_equal(rest.design, whole.design)
    assert np.array_equal(rest.mma.state_array, whole.mma.state_array)


def test_appending_keeps_what_is_scheduled_and_the_run_is_refused_if_changed_before_a_solve(
        linear, newton, scale, finished, monkeypatch):
    done = finished.mma
    step = dict(beta=BETA, alpha_max=ALPHA_MAX)
    kept = "does not keep the old one"
    refused = {
        "a step already scheduled at another beta":
            ([drv.Phase("fixed", 4, beta=16.0, alpha_max=ALPHA_MAX)], 4, kept),
        "a step already scheduled at another alpha_max":
            ([drv.Phase("fixed", 4, beta=BETA, alpha_max=1e6)], 4, kept),
        "a step already scheduled under another name": ([drv.Phase("other", 4, **step)], 4, kept),
        "a schedule shorter than the old one": (_fixed(1), 4, kept),
        "a budget that adds nothing": (_fixed(4), 2, "adds nothing"),
    }
    for what, (phases, budget, match) in refused.items():
        with pytest.raises(ValueError, match=match):
            done.append_budget(phases, budget)
    more = done.append_budget([drv.Phase("fixed", 2, **step), drv.Phase("appended", 2, **step)], 4)

    for problem in (linear, newton):
        monkeypatch.setattr(problem, "solve_states",
                            lambda *a, **k: pytest.fail("solved before refusing"))
    appended = [drv.Phase("fixed", 2, **step), drv.Phase("appended", 2, **step)]
    runs = {
        "another thermal path": (newton, scale, appended, 0.1, None),
        "another scale": (linear, dataclasses.replace(scale, c_0=2.0 * scale.c_0), appended, 0.1, None),
        "another move limit": (linear, scale, appended, 0.2, None),
        "another appended step": (linear, scale, [drv.Phase("fixed", 2, **step),
                                                  drv.Phase("appended", 2, beta=16.0,
                                                            alpha_max=ALPHA_MAX)], 0.1, None),
        "another budget": (linear, scale, appended, 0.1, 5),
    }
    for what, (problem, sc, phases, move, budget) in runs.items():
        with pytest.raises(ValueError, match="another run"):
            ff.run(problem, sc, phases, move_limit=move, budget=budget, resume=more)
    for what, config in {
        "weight": dataclasses.replace(CONFIG, weight=0.6),
        "volume bound": dataclasses.replace(CONFIG, max_fluid_fraction=0.35),
        "projection": dataclasses.replace(CONFIG, projection=r1.Projection.TANH),
        "filter radius": dataclasses.replace(CONFIG, filter_radius_elements=3.0),
    }.items():
        variant = ff.Zhao2DFineFlowProblem(SPEC, config, 2, 2, 3, thermal_path="linear")
        monkeypatch.setattr(variant, "solve_states",
                            lambda *a, _what=what, **k: pytest.fail(f"{_what}: solved before refusing"))
        with pytest.raises(ValueError, match="another run"):
            ff.run(variant, scale, appended, move_limit=0.1, resume=more)


def test_appending_tells_the_budgets_flag_from_a_proxy_stop(linear, scale, start, finished):
    """The states here are made up (marked), to reach what twenty updates on a
    small mesh do not: upstream's proxies firing."""
    done, state = finished.mma, _state(finished.mma)
    made_up = {
        "step_tol": dataclasses.replace(done, state_array=dataclasses.replace(
            state, change_design_var=1e-9).to_array()),
        "kkt_tol": dataclasses.replace(done, kktnorm=1e-9),
    }
    for proxy, checkpoint in made_up.items():
        with pytest.raises(ValueError, match=f"{proxy} proxy fired"):
            checkpoint.append_budget(_fixed(4), 4)
    unset = dataclasses.replace(done, state_array=dataclasses.replace(
        state, is_converged=False).to_array())  # made up: at the budget, no flag
    with pytest.raises(ValueError, match="neither the budget nor a proxy"):
        unset.append_budget(_fixed(4), 4)

    paused = ff.run(linear, scale, _fixed(3), move_limit=0.1, initial_design=start,
                    stop_after=1).mma
    early = dataclasses.replace(paused, state_array=dataclasses.replace(
        _state(paused), is_converged=True).to_array())  # made up: a flag before the budget
    with pytest.raises(ValueError, match="neither the budget nor a proxy"):
        early.append_budget(_fixed(4), 4)
    # a paused run has no flag to clear: appending changes no number of its state
    more = paused.append_budget(_fixed(4), 4)
    assert np.array_equal(more.state_array, paused.state_array)


# -- R1v: a legacy checkpoint migrated -------------------------------------------------


def _legacy(checkpoint):
    """The same checkpoint as it would have been signed from R1t until c31b3c2.

    `run_binding` then named the model, the thermal path and the scale; c31b3c2
    added the optimisation block and changed nothing else, so the legacy entry
    is today's without that block."""
    old = json.loads(checkpoint.binding)
    entry = {k: v for k, v in old["entry"].items() if k != "optimisation"}
    assert sorted(entry) == sorted(ff.LEGACY_ENTRY_KEYS)
    return dataclasses.replace(checkpoint, binding=json.dumps(dict(old, entry=entry), sort_keys=True))


def test_a_legacy_checkpoint_migrates_with_its_state_untouched_and_then_resumes(
        linear, scale, finished, whole):
    legacy = _legacy(finished.mma)
    with pytest.raises(ValueError, match="another run"):  # today's runs refuse it, as R1t's
        ff.run(linear, scale, _fixed(4), move_limit=0.1, resume=legacy.append_budget(_fixed(4), 4))

    design = r1.Zhao2DProblem(SPEC, CONFIG)  # the design side, as D builds its own
    migrated = ff.migrate_legacy_checkpoint(legacy, design, CONFIG.fingerprint())
    assert migrated.binding == finished.mma.binding  # exactly what today's run signs
    assert np.array_equal(migrated.state_array, legacy.state_array)
    assert (migrated.kktnorm, migrated.updates_done, migrated.num_design_var) == (
        legacy.kktnorm, legacy.updates_done, legacy.num_design_var)

    cont = ff.run(linear, scale, _fixed(4), move_limit=0.1,
                  resume=migrated.append_budget(_fixed(4), 4))
    _same_updates(finished, cont, whole)


def test_a_migration_is_refused_unless_rebuilt_from_the_runs_own_record(
        linear, scale, finished, monkeypatch):
    legacy = _legacy(finished.mma)
    design = r1.Zhao2DProblem(SPEC, CONFIG)
    with pytest.raises(ValueError, match="not a legacy binding"):
        ff.migrate_legacy_checkpoint(finished.mma, design, CONFIG.fingerprint())
    heavier = dataclasses.replace(CONFIG, weight=0.6)
    with pytest.raises(ValueError, match="not the one the run recorded"):
        ff.migrate_legacy_checkpoint(legacy, r1.Zhao2DProblem(SPEC, heavier), CONFIG.fingerprint())
    other_spec = dataclasses.replace(SPEC, q_alpha=0.25)
    with pytest.raises(ValueError, match="model identity"):
        ff.migrate_legacy_checkpoint(legacy, r1.Zhao2DProblem(other_spec, CONFIG),
                                     CONFIG.fingerprint())

    # A record that misstates a setting the model identity leaves out signs another
    # optimisation problem -- and the run it came from then refuses it, before a solve.
    wrong = ff.migrate_legacy_checkpoint(legacy, r1.Zhao2DProblem(SPEC, heavier),
                                         heavier.fingerprint())
    assert wrong.binding != finished.mma.binding
    monkeypatch.setattr(linear, "solve_states", lambda *a, **k: pytest.fail("solved before refusing"))
    with pytest.raises(ValueError, match="another run"):
        ff.run(linear, scale, _fixed(4), move_limit=0.1, resume=wrong.append_budget(_fixed(4), 4))


# -- the main-mesh scripts -----------------------------------------------------------


def _scripts():
    path = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
    if path not in sys.path:
        sys.path.insert(0, path)


def test_the_locked_source_matches_its_bytes_or_its_text_up_to_line_endings(tmp_path):
    _scripts()
    import zhao2d_r1v_migrate as mv

    crlf = b"a = 1\r\nb = 2\r\n"
    lf = crlf.replace(b"\r\n", b"\n")
    recorded = hashlib.sha256(crlf).hexdigest()
    path = tmp_path / "source.py"
    path.write_bytes(crlf)
    assert mv.locked_match(path, recorded) == "bytes"
    path.write_bytes(lf)
    assert mv.locked_match(path, recorded) == "up to line endings: recorded with CRLF"
    assert mv.locked_match(path, hashlib.sha256(lf).hexdigest()) == "bytes"
    path.write_bytes(crlf)
    assert mv.locked_match(path, hashlib.sha256(lf).hexdigest()) == "up to line endings: recorded with LF"
    path.write_bytes(b"a = 1\r\nb = 3\r\n")
    assert mv.locked_match(path, recorded) is None  # a changed character is a changed file


def test_r1v_migrates_r1ts_checkpoint_and_leaves_r1ts_files_alone(tmp_path, monkeypatch):
    """The real migration: R1t's record and fields, the design side on h, no solve.

    Its last stage also needs the reference file's recorded bytes, which on an
    LF checkout have to be restored first (as for R1u's test)."""
    _scripts()
    import zhao2d_r1v_migrate as mv

    root = pathlib.Path(__file__).resolve().parent.parent / "results"
    before = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in mv.INPUTS}
    monkeypatch.setattr(mv.sys, "argv", ["r1v", "--out", str(tmp_path)])
    mv.main()
    assert {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in mv.INPUTS} == before

    record = json.loads((tmp_path / mv.RECORD).read_text(encoding="utf-8"))
    assert [c["stage"] for c in record["checkpoints"]] == list(mv.STAGES)
    assert not any(c["failures"] for c in record["checkpoints"])
    assert all(f["matches"] for f in record["locked_source"]["files"].values())
    with np.load(root / "zhao2d_r1t_fields.npz") as data:
        legacy = drv.MMACheckpoint.from_npz(data)
    with np.load(tmp_path / mv.FIELDS) as data:
        migrated = drv.MMACheckpoint.from_npz(data)
        assert str(data["legacy_binding"]) == legacy.binding
    assert np.array_equal(migrated.state_array, legacy.state_array)
    assert (migrated.kktnorm, migrated.updates_done, migrated.num_design_var) == (
        legacy.kktnorm, 20, 5000)
    old, new = json.loads(legacy.binding), json.loads(migrated.binding)
    assert {k: v for k, v in new["entry"].items() if k != "optimisation"} == old["entry"]
    assert {k: v for k, v in new.items() if k != "entry"} == {k: v for k, v in old.items() if k != "entry"}
    block = new["entry"]["optimisation"]
    assert (block["weight"], block["max_fluid_fraction"], block["volume_domain"], block["projection"]) == (
        0.5, 0.4, "design", "volume_preserving")
    assert block["filter"]["radius_m"] == pytest.approx(2e-4, rel=1e-15)
    assert block["design_map"]["num_design"] == 5000
    dry = record["dry_part_2"]
    assert dry["resume"]["accepted_by_run_loop"] and dry["append"]["is_converged"] == [True, False]
    assert dry["resume"]["first_evaluation"]["design_is_r1t_raw_terminal"]

    with pytest.raises(SystemExit, match="already holds"):  # its own outputs are not overwritten
        mv.main()


def test_r1t_stops_at_a_failed_zero_step_before_any_update(tmp_path, monkeypatch):
    """A zero step whose C is off R1r's terminal by 1e-7 -- inside a looser
    criterion, outside R1s's 1e-8 across the thermal paths -- stops the run,
    with its record written, before MMA's first update."""
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
