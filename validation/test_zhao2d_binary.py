"""The rules for thresholded designs and post-processed states. Nothing is solved.

Each test targets a way a check could count a state or a comparison it should
not: a NaN residual accepted because `max()` returned the finite one, an anchor
that failed but left its cell counted, a volume-infeasible design ranked as if
it qualified. The two reproducers from the review of 320ea73 are the first two
gate and usability cases.
"""

import dataclasses
import hashlib
import pathlib
import sys

import numpy as np
import jax.numpy as jnp
import pytest

from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_binary as zb  # noqa: E402

SPEC = dataclasses.replace(z.Zhao2DSpec(), element_size=1.0e-3)  # 50 design cells + 2 tabs


@pytest.fixture(scope="module")
def mesh():
    pm = z.build_mesh(SPEC, dofs_per_node=3)
    assert pm.num_elems == 52
    return pm


def test_threshold_is_solid_at_the_tie_keeps_tabs_fluid_and_leaves_s_alone():
    s = np.array([0.0, 0.499, 0.5, 0.8, 1.0])
    design = np.array([True, True, True, False, False])
    before = s.copy()
    assert np.array_equal(zb.threshold(s, design, 0.5), [0, 0, 1, 0, 0])
    assert np.array_equal(s, before)


def test_connectivity_open_and_closed(mesh):
    design = np.asarray(mesh.design_mask)
    for solid, joined, components in ((0.0, True, 1), (1.0, False, 2)):
        s = np.where(design, solid, 0.0)
        result = zb.connectivity(mesh, s, (0.3, 0.5, 0.7))
        assert set(result) == {"0.3", "0.5", "0.7"}
        for r in result.values():
            assert r["inlet_outlet_connected"] is joined
            assert r["fluid_components"] == components


@pytest.mark.parametrize("norms", [
    {"flow": 1e-14, "thermal": float("nan")},   # the review's reproducer: max() kept 1e-14
    {"flow": float("nan"), "thermal": 1e-14},
    {"flow": 1e-14, "thermal": float("inf")},
    {"flow": 1e-14, "thermal": 2e-8},
    {},
])
def test_the_gate_refuses_any_non_finite_or_large_residual(norms):
    assert zb.gate_passed(norms, 1e-8) is False


def test_the_gate_accepts_finite_residuals_within_tolerance():
    assert zb.gate_passed({"flow": 1.3e-14, "thermal": 2.6e-11}, 1e-8) is True
    assert zb.gate_passed({"flow": np.float64(1e-8), "thermal": np.float32(1e-9)}, 1e-8) is True


def test_a_failed_anchor_makes_its_cell_unusable():
    # the review's second reproducer: usable() used to ignore the anchor
    good = {"analysed": True, "gate_passed": True}
    assert zb.cell_usable(good) is True
    assert zb.cell_usable({**good, "anchor": {"reproduced": False}}) is False
    assert zb.cell_usable({**good, "anchor": {"reproduced": True}}) is True
    assert zb.cell_usable({**good, "gate_passed": False}) is False
    assert zb.cell_usable({"analysed": False}) is False
    assert zb.cell_usable(None) is False


def test_anchor_status_needs_finite_small_differences():
    anchor = {"psi": 0.0143, "compliance": 36180.0, "source": "test"}
    ok = zb.anchor_status({"psi": 0.0143 * (1 + 1e-12), "compliance": 36180.0}, anchor, 1e-9)
    assert ok["reproduced"] is True and ok["source"] == "test"
    assert zb.anchor_status({"psi": 0.0143 * 1.001, "compliance": 36180.0},
                            anchor, 1e-9)["reproduced"] is False
    assert zb.anchor_status({"psi": float("nan"), "compliance": 36180.0},
                            anchor, 1e-9)["reproduced"] is False


def test_the_volume_bound_holds_to_rounding_and_no_further(mesh):
    design = np.asarray(mesh.design_mask)
    cells = np.flatnonzero(design)
    bound = 0.4
    for fluid_cells, feasible in ((20, True), (21, False)):  # 40% of 50 design cells
        s = np.where(design, 1.0, 0.0)
        s[cells[:fluid_cells]] = 0.0
        v_f = z.fluid_fractions(mesh, jnp.asarray(s))["v_f_design_domain"]
        assert zb.volume_feasible(v_f, bound) is feasible
    assert zb.volume_feasible(float("nan"), bound) is False


def test_only_two_feasible_usable_cells_make_a_qualified_comparison():
    usable = {"analysed": True, "gate_passed": True}
    feasible, infeasible = {**usable, "volume_feasible": True}, {**usable, "volume_feasible": False}
    assert zb.comparison_kind(feasible, feasible) == "qualified_comparison"
    assert zb.comparison_kind(feasible, infeasible) == "threshold_diagnostic"
    assert zb.comparison_kind(feasible, {**feasible, "anchor": {"reproduced": False}}) is None
    assert zb.comparison_kind(feasible, None) is None


def test_check_failures_names_every_failing_cell():
    cells = {
        "a": {"analysed": True, "gate_passed": True, "anchor": {"reproduced": True}},
        "b": {"analysed": True, "gate_passed": False},
        "c": {"analysed": True, "gate_passed": True, "anchor": {"reproduced": False}},
        "d": {"analysed": False},
    }
    assert zb.check_failures(cells) == {"anchors_not_reproduced": ["c"],
                                        "cells_failing_gate": ["b"]}


def test_the_volume_threshold_fills_the_budget_and_no_more(mesh):
    design = np.asarray(mesh.design_mask)
    areas = np.asarray(mesh.elem_area)
    n = int(design.sum())  # 50 design cells; 40% is 20
    s = np.zeros(mesh.num_elems)
    s[design] = np.linspace(0.05, 0.95, n)  # distinct values
    out = zb.volume_threshold(s, design, areas, 0.4)
    assert out["fluid_cells"] == 20 and out["fluid_fraction"] == pytest.approx(0.4, rel=1e-12)
    assert np.count_nonzero(design & (s < out["t"])) == 20
    assert np.all(out["s_binary"][~design] == 0.0)  # tabs fluid


def test_equal_densities_move_together(mesh):
    """A tie straddling the budget goes solid as a group, never split."""
    design = np.asarray(mesh.design_mask)
    areas = np.asarray(mesh.elem_area)
    idx = np.flatnonzero(design)
    s = np.zeros(mesh.num_elems)
    s[idx] = 0.9
    s[idx[:18]] = 0.1      # 18 cells clearly fluid
    s[idx[18:23]] = 0.5    # 5 tied cells: 18 + 5 = 23 would overshoot 20
    out = zb.volume_threshold(s, design, areas, 0.4)
    assert out["fluid_cells"] == 18 and out["t"] == 0.5


def test_the_volume_threshold_reproduces_the_reviewed_geometry():
    """The review of 320ea73 found t = 0.5288802660 (x300) and 0.4014602995 (x30)."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "results"
    pm = z.build_mesh(z.Zhao2DSpec(), dofs_per_node=3)
    design, areas = np.asarray(pm.design_mask), np.asarray(pm.elem_area)
    for path, key, t, changed in (("zhao2d_r1d_main_fields.npz", "solid_fraction", 0.5288802660, 6),
                                  ("zhao2d_r1k_fields.npz", "solid_fraction", 0.4014602995, 26)):
        s = np.load(root / path)[key]
        out = zb.volume_threshold(s, design, areas, 0.4)
        assert out["t"] == pytest.approx(t, abs=5e-11)
        assert out["fluid_cells"] == 2000 and out["cells_changed_from_0.5"] == changed


def test_the_terminal_check_refuses_to_overwrite_its_record_before_building(tmp_path, monkeypatch):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    import zhao2d_r1k_terminal_check as tc

    target = tmp_path / tc.RECORD
    target.write_text("cited evidence", encoding="utf-8")
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    monkeypatch.setattr(tc.sys, "argv", ["terminal_check", "--out", str(tmp_path)])
    monkeypatch.setattr(tc.dual, "Zhao2DDualProblem",
                        lambda *a, **k: pytest.fail("built a problem before refusing"))
    with pytest.raises(SystemExit):
        tc.main()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before
    assert not (tmp_path / tc.STACKS).exists()
