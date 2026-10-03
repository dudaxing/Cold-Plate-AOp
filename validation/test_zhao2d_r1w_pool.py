"""R1w: the binary designs along R1v's trajectory, screened once on D.

The contract's new logic, and its ways to go silently wrong:

  - a design solved twice, or a duplicate paid for as a new one;
  - a reused state that is not the design it stands for;
  - a failed solve dropped, retried, or ranked as if it were a worse design;
  - a ranking that claims the whole pool when a design failed;
  - anything built or solved before the inputs and the manifest are checked.

The main-mesh solves are not run here. The last two tests run the script on
the real records up to the manifest, which is geometry only.
"""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

import zhao2d_r1w_pool_screen as rw  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent / "results"


def test_the_pool_groups_by_digest_labels_by_earliest_iterate_and_reuses_known_designs():
    iterations = [20, 21, 22, 23, 24, 25, 26, 27]
    digests = ["T", "a", "b", "a", "c", "V", "V", "U"]
    groups = rw.plan_pool(iterations, digests, known={"T": "r1t", "V": "r1v", "U": "r1u"},
                          unqualified={"c", "U"})
    assert [g["label"] for g in groups] == [20, 21, 22, 24, 25, 27]
    assert [g["iterations"] for g in groups] == [[20], [21, 23], [22], [24], [25, 26], [27]]
    assert [g["action"] for g in groups] == ["reuse", "solve", "solve", "not qualified", "reuse",
                                             "not qualified"]
    # a design the rule does not qualify is never reused, even if it is a candidate's
    assert [g["reuse_of"] for g in groups] == ["r1t", None, None, None, "r1v", None]
    # the label is the earliest iterate; the identity is the digest
    assert {g["sha256"] for g in groups} == set(digests)


def test_each_design_is_solved_once_and_a_failure_is_recorded_not_retried():
    groups = [{"label": k} for k in (21, 22, 23, 24)]
    calls, seen = [], []

    def solve_one(group):
        calls.append(group["label"])
        if group["label"] == 22:
            raise rw.SolveFailed("a residual fails the gate", {"residual_flow": 1.0})
        if group["label"] == 23:
            raise ZeroDivisionError("an error inside the solve")
        return {"J": 1.0 + group["label"] / 100}

    def report(result):
        seen.append(result)
        if result["label"] == 21:
            raise PermissionError("the progress file is locked")  # reporting must not stop it

    results = rw.screen(groups, solve_one, on_result=report)
    assert calls == [21, 22, 23, 24]  # each once, in order: no retry, and the run goes on
    assert seen == results
    assert results[0]["reporting_error"] == "PermissionError: the progress file is locked"
    assert [r["valid"] for r in results] == [True, False, False, True]
    assert results[1]["failure"] == "SolveFailed: a residual fails the gate"
    assert results[1]["partial"] == {"residual_flow": 1.0}
    assert results[2]["failure"].startswith("ZeroDivisionError")
    assert "cell" not in results[1] and "cell" not in results[2]


def test_the_ranking_lists_failures_apart_and_narrows_its_claim():
    entries = [{"label": 20, "valid": True, "J": 1.30}, {"label": 21, "valid": True, "J": 1.29},
               {"label": 22, "valid": False, "J": None}, {"label": 23, "valid": True, "J": 1.31}]
    ranking = rw.rank_pool(entries)
    assert ranking["order"] == [21, 20, 23]  # the failed design is not ranked, as worse or at all
    assert ranking["lowest"] == 21
    assert ranking["no_valid_performance"] == [22]
    assert ranking["complete"] is False
    assert "3 of 4" in ranking["scope"] and "does not rule out" in ranking["scope"]

    whole = rw.rank_pool([e for e in entries if e["valid"]])
    assert whole["complete"] is True and whole["scope"] == "all 3 designs of the pool"

    # a design the rule does not qualify is listed apart too, and counted in the pool
    with_unqualified = rw.rank_pool([*[e for e in entries if e["valid"]],
                                     {"label": 24, "valid": False, "not_qualified": True, "J": None}])
    assert with_unqualified["order"] == [21, 20, 23] and with_unqualified["not_qualified"] == [24]
    assert with_unqualified["complete"] is False and with_unqualified["no_valid_performance"] == []
    assert with_unqualified["scope"] == "the 3 qualified designs of the 4"


def test_r1w_is_refused_at_its_inputs_before_anything_is_built(tmp_path, monkeypatch):
    real = rw.load_json

    def tampered(path):
        rec = real(path)
        if pathlib.Path(path).name == "zhao2d_r1t.json":
            rec["export"]["binary_sha256"] = "0" * 64  # R1t's recorded binary design, changed
        return rec

    def never(*a, **k):
        pytest.fail("built before the inputs were checked")

    monkeypatch.setattr(rw, "load_json", tampered)
    monkeypatch.setattr(rw.r1, "Zhao2DProblem", never)
    monkeypatch.setattr(rw.dual, "Zhao2DDualProblem", never)
    monkeypatch.setattr(rw.faulthandler, "dump_traceback_later", lambda *a, **k: None)
    monkeypatch.setattr(rw.sys, "argv", ["r1w", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="CHECKS FAILED at inputs"):
        rw.main()
    record = json.loads((tmp_path / rw.RECORD).read_text(encoding="utf-8"))
    assert record["stopped"].startswith("inputs")
    assert not (tmp_path / rw.MANIFEST).exists()


def test_r1w_writes_its_manifest_before_the_route_is_built(tmp_path, monkeypatch):
    """The real pool, geometry only: 21 iterates, 18 distinct binary designs,
    R1t's and R1v's reused, 16 to solve. Stopped where the route would be built."""

    class RouteNotBuilt(Exception):
        pass

    def stop(*a, **k):
        assert (tmp_path / rw.MANIFEST).exists(), "the route was reached before the manifest"
        raise RouteNotBuilt

    monkeypatch.setattr(rw.dual, "Zhao2DDualProblem", stop)
    monkeypatch.setattr(rw.faulthandler, "dump_traceback_later", lambda *a, **k: None)
    monkeypatch.setattr(rw.sys, "argv", ["r1w", "--out", str(tmp_path)])
    with pytest.raises(RouteNotBuilt):
        rw.main()

    manifest = json.loads((tmp_path / rw.MANIFEST).read_text(encoding="utf-8"))
    groups = manifest["groups"]
    assert manifest["counts"] == {"iterates": 21, "distinct": 18, "reuse": 2, "solve": 16,
                                  "not_qualified": 0}
    assert [g["label"] for g in groups] == [20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33,
                                            34, 35, 36, 37, 38]
    by_label = {g["label"]: g for g in groups}
    assert by_label[27]["iterations"] == [27, 29]
    assert by_label[37]["iterations"] == [37, 39, 40]
    assert (by_label[20]["action"], by_label[20]["reuse_of"]) == ("reuse", "r1t")
    assert (by_label[37]["action"], by_label[37]["reuse_of"]) == ("reuse", "r1v")
    # each iterate carries its group's label and action
    per_iterate = {i["iteration"]: i for i in manifest["iterates"]}
    assert (per_iterate[29]["label"], per_iterate[29]["action"]) == (27, "solve")
    assert (per_iterate[39]["label"], per_iterate[39]["action"],
            per_iterate[39]["reuse_of"]) == (37, "reuse", "r1v")
    t_rec = json.loads((ROOT / "zhao2d_r1t.json").read_text(encoding="utf-8"))
    v_rec = json.loads((ROOT / "zhao2d_r1v.json").read_text(encoding="utf-8"))
    assert by_label[20]["sha256"] == t_rec["export"]["binary_sha256"]
    assert by_label[37]["sha256"] == v_rec["export"]["binary_sha256"]
    # the review of 03ccf6c's catalogue of the same pool, by digest
    review = {"a1134eb5": 21, "a084393b": 27, "dedf4c2a": 38, "98d7e4d2": 36}
    for prefix, label in review.items():
        assert by_label[label]["sha256"].startswith(prefix)


def test_r1w_refuses_a_second_run_in_the_same_folder_and_cleans_up_under_overwrite(
        tmp_path, monkeypatch):
    (tmp_path / rw.LOCK).write_text("", encoding="utf-8")
    monkeypatch.setattr(rw.sys, "argv", ["r1w", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="another R1w run"):
        rw.main()
    (tmp_path / rw.LOCK).unlink()

    # under --overwrite, no file of an earlier run survives next to a new one
    for name in (rw.RECORD, rw.FIELDS, rw.MANIFEST, rw.PROGRESS):
        (tmp_path / name).write_text("an earlier run", encoding="utf-8")
    real = rw.load_json

    def tampered(path):
        rec = real(path)
        if pathlib.Path(path).name == "zhao2d_r1t.json":
            rec["export"]["binary_sha256"] = "0" * 64
        return rec

    monkeypatch.setattr(rw, "load_json", tampered)
    monkeypatch.setattr(rw.faulthandler, "dump_traceback_later", lambda *a, **k: None)
    monkeypatch.setattr(rw.sys, "argv", ["r1w", "--out", str(tmp_path), "--overwrite"])
    with pytest.raises(SystemExit, match="CHECKS FAILED at inputs"):
        rw.main()
    assert not (tmp_path / rw.FIELDS).exists() and not (tmp_path / rw.MANIFEST).exists()
    assert json.loads((tmp_path / rw.RECORD).read_text(encoding="utf-8"))["stopped"].startswith("inputs")
    assert not (tmp_path / rw.LOCK).exists()  # released when the record is written


def test_a_recomputed_density_is_compared_to_a_stated_tolerance_not_bit_for_bit():
    """A continuous density recomputed elsewhere may differ in its last bits; a changed map may not."""
    saved = np.linspace(0.0, 1.0, 11)
    same = rw.density_match(saved, saved)
    assert same == {"bitwise": True, "max_abs_diff": 0.0, "atol": rw.DENSITY_ATOL, "match": True}

    roundoff = rw.density_match(saved + 1e-14 * (-1.0) ** np.arange(11), saved)
    assert roundoff["match"] and not roundoff["bitwise"]
    assert 0.0 < roundoff["max_abs_diff"] < 1e-13

    assert not rw.density_match(saved + 1e-6, saved)["match"]  # a real change of the map
    assert not rw.density_match(saved[:-1], saved)["match"]  # another mesh
    with_nan = saved.copy()
    with_nan[3] = np.nan
    assert not rw.density_match(with_nan, saved)["match"]


def _r1t_density_shifted(monkeypatch, shift):
    """np.load as the script sees it, with R1t's saved continuous density moved by `shift`."""
    real = np.load

    class Fields(dict):
        @property
        def files(self):
            return list(self)

    def load(path, *args, **kwargs):
        fields = real(path, *args, **kwargs)
        if pathlib.Path(path).name != "zhao2d_r1t_fields.npz":
            return fields
        out = Fields({key: fields[key] for key in fields.files})
        out["solid_fraction"] = out["solid_fraction"] + shift
        return out

    monkeypatch.setattr(rw.np, "load", load)


def test_r1w_accepts_a_saved_density_off_by_round_off(tmp_path, monkeypatch):
    """As on another platform: the pool and its manifest go through, and the record says how."""

    class RouteNotBuilt(Exception):
        pass

    def stop(*a, **k):
        raise RouteNotBuilt

    _r1t_density_shifted(monkeypatch, 1e-13 * (-1.0) ** np.arange(5200))
    monkeypatch.setattr(rw.dual, "Zhao2DDualProblem", stop)
    monkeypatch.setattr(rw.faulthandler, "dump_traceback_later", lambda *a, **k: None)
    monkeypatch.setattr(rw.sys, "argv", ["r1w", "--out", str(tmp_path)])
    with pytest.raises(RouteNotBuilt):
        rw.main()
    assert (tmp_path / rw.MANIFEST).exists()
    record = json.loads((tmp_path / rw.RECORD).read_text(encoding="utf-8"))
    r1t = record["density_checks"]["r1t"]
    assert r1t["match"] and not r1t["bitwise"] and 0.0 < r1t["max_abs_diff"] < 1e-12
    assert record["density_checks"]["r1v"]["match"]  # untouched; bit for bit is not required
    # the binary designs are still compared exactly: iterate 20 exports R1t's
    assert record["manifest"]["groups"][0]["reuse_of"] == "r1t"


def test_r1w_refuses_a_saved_density_that_is_really_different(tmp_path, monkeypatch):
    """A difference beyond round-off stops the run at the pool, before the manifest or the route."""

    def never(*a, **k):
        pytest.fail("the route was built after the density check failed")

    _r1t_density_shifted(monkeypatch, 1e-6)
    monkeypatch.setattr(rw.dual, "Zhao2DDualProblem", never)
    monkeypatch.setattr(rw.faulthandler, "dump_traceback_later", lambda *a, **k: None)
    monkeypatch.setattr(rw.sys, "argv", ["r1w", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="CHECKS FAILED at the pool and the manifest"):
        rw.main()
    assert not (tmp_path / rw.MANIFEST).exists()
    record = json.loads((tmp_path / rw.RECORD).read_text(encoding="utf-8"))
    assert not record["density_checks"]["r1t"]["match"]
    assert record["density_checks"]["r1t"]["max_abs_diff"] == pytest.approx(1e-6)
