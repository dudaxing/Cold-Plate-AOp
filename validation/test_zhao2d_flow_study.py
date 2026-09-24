"""R1h: reusing a flow state across meshes, and the identities every cell reports.

Each test targets one way the fixed-design comparison could be silently wrong:

  - a saved flow state reused by a problem it does not solve (R1f's cache
    carried no identity at all);
  - a flow state that is not converged, or does not carry its own Dirichlet
    values, accepted because nobody recomputed its residual;
  - a rerun that re-solves the fine flow, or overwrites the saved state and the
    cited records, instead of loading what is already there;
  - an inflow that changes with the flow mesh, so "replacing the flow" also
    changes the load;
  - the h/4 thermal mesh seeing a different density depending on which flow
    mesh it was reached from;
  - an identity that is quoted but does not close, so its terms mean nothing.
"""

import dataclasses
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402
from tfopus import zhao2d_refine as ref  # noqa: E402

# 10 x 20 design + 2 x (2 x 2) tabs = 208 flow cells; wiring, not accuracy.
SPEC = dataclasses.replace(z.Zhao2DSpec(), element_size=5.0e-4)
CONFIG = r1.R1Config()
ALPHA_MAX = 1.0e7


@pytest.fixture(scope="module")
def coarse():
    return dual.Zhao2DDualProblem(SPEC, CONFIG, thermal_refinement=2, thermal_quadrature=3)


def _grey(problem, seed=5):
    x = np.random.default_rng(seed).uniform(0.15, 0.85, problem.num_design)
    return problem.solid_fraction(jnp.asarray(x), 8.0)


@pytest.fixture(scope="module")
def solved(coarse):
    s = _grey(coarse)
    press_vel, temperature, _, _ = coarse.solve_states(s, ALPHA_MAX)
    return s, press_vel, temperature


# -- a flow state and what it belongs to ------------------------------------


def test_identity_binds_what_the_flow_depends_on_and_nothing_else(coarse, solved):
    s = solved[0]
    base = fs.flow_state_identity(coarse, s, ALPHA_MAX)
    assert base == fs.flow_state_identity(coarse, s, ALPHA_MAX)

    moved = np.asarray(s).copy()
    moved[coarse.design_elements[0]] += 1e-3
    assert fs.flow_state_identity(coarse, moved, ALPHA_MAX) != base
    assert fs.flow_state_identity(coarse, s, 2 * ALPHA_MAX) != base

    for field, value, changes in (("inlet_speed", 0.3, True),
                                  ("fluid_viscosity", 0.002, True),
                                  ("heat_source", 2.0e8, False),
                                  ("solid_conductivity", 100.0, False)):
        other = dual.Zhao2DDualProblem(dataclasses.replace(SPEC, **{field: value}), CONFIG,
                                       thermal_refinement=2, thermal_quadrature=3)
        assert (fs.flow_state_identity(other, s, ALPHA_MAX) != base) == changes, field


def test_a_converged_state_passes_and_other_problems_do_not(coarse, solved):
    s, press_vel, _ = solved
    report = fs.verify_flow_state(coarse, press_vel, s, ALPHA_MAX)
    assert report["residual_relative"] < 1e-8
    assert report["dirichlet_max_deviation"] == 0.0

    with pytest.raises(r1.NotConverged):
        fs.verify_flow_state(coarse, press_vel, s, 2 * ALPHA_MAX)
    with pytest.raises(r1.NotConverged):
        fs.verify_flow_state(coarse, press_vel, jnp.clip(s + 0.05, 0.0, 1.0), ALPHA_MAX)

    broken = np.asarray(press_vel).copy()
    broken[coarse.flow_bc.bc["fixed_dofs"][0]] += 1e-9
    with pytest.raises(ValueError, match="Dirichlet"):
        fs.verify_flow_state(coarse, broken, s, ALPHA_MAX)
    with pytest.raises(ValueError, match="shape"):
        fs.verify_flow_state(coarse, np.asarray(press_vel)[:-3], s, ALPHA_MAX)


def test_a_saved_state_reloads_only_for_its_own_problem(coarse, solved, tmp_path):
    s, press_vel, _ = solved
    path = tmp_path / "flow.npz"
    identity = fs.flow_state_identity(coarse, s, ALPHA_MAX)
    fs.save_flow_state(path, press_vel, s, identity, {"stage": "test"})

    back, record = fs.load_flow_state(path, coarse, s, ALPHA_MAX)
    assert np.array_equal(np.asarray(back), np.asarray(press_vel))
    assert record["verification"]["residual_relative"] < 1e-8
    assert record["provenance"] == {"stage": "test"}

    with pytest.raises(ValueError, match="material.alpha_max"):
        fs.load_flow_state(path, coarse, s, 2 * ALPHA_MAX)

    fine = dual.Zhao2DDualProblem(ref.refine_spec(SPEC, 2), CONFIG, 1, 3)
    s_fine = ref.refine_design(coarse.flow_mesh, fine.flow_mesh, s)
    with pytest.raises(ValueError, match="different problem"):
        fs.load_flow_state(path, fine, s_fine, ALPHA_MAX)


# -- where a flow state comes from ------------------------------------------


class _ForbiddenSolve:
    """Stands in for the flow solve where a test forbids it; counts any call."""

    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        raise AssertionError("the flow solve must not be called here")


def test_a_qualified_identity_cache_is_loaded_and_the_solve_never_called(
        coarse, solved, tmp_path):
    """Only the R1h-style file, no bare cache: load it, never solve, never rewrite it."""
    s, press_vel, _ = solved
    cache = tmp_path / "flow.npz"
    fs.save_flow_state(cache, press_vel, s, fs.flow_state_identity(coarse, s, ALPHA_MAX),
                       {"stage": "test"})
    before = cache.read_bytes()
    solve = _ForbiddenSolve()

    got, record = fs.resolve_flow_state(coarse, s, ALPHA_MAX, identity_cache=cache,
                                        bare_cache=tmp_path / "absent.npz", solve=solve)
    assert solve.calls == 0
    assert record["source"] == "identity cache" and record["rejected"] == []
    assert np.array_equal(np.asarray(got), np.asarray(press_vel))
    assert cache.read_bytes() == before


def test_a_foreign_identity_cache_falls_back_to_a_verified_bare_cache(
        coarse, solved, tmp_path):
    s, press_vel, _ = solved
    foreign = tmp_path / "foreign.npz"
    fs.save_flow_state(foreign, press_vel, s,
                       fs.flow_state_identity(coarse, s, 2 * ALPHA_MAX), {})
    bare = tmp_path / "bare.npz"
    np.savez_compressed(bare, press_vel=np.asarray(press_vel))
    solve = _ForbiddenSolve()

    _, record = fs.resolve_flow_state(coarse, s, ALPHA_MAX, identity_cache=foreign,
                                      bare_cache=bare, solve=solve)
    assert solve.calls == 0 and record["source"] == "bare cache"
    assert "different problem" in record["rejected"][0]["reason"]

    # a bare cache is only as good as the checks it passes: a stated Psi it does
    # not reproduce rejects it too
    with pytest.raises(RuntimeError, match="not the reference"):
        fs.resolve_flow_state(coarse, s, ALPHA_MAX, bare_cache=bare,
                              reference_psi=1.0, psi_rtol=1e-10)


def test_the_flow_is_solved_once_only_when_no_cache_qualifies(coarse, solved, tmp_path):
    s, press_vel, _ = solved
    bare = tmp_path / "bare.npz"
    np.savez_compressed(bare, press_vel=np.asarray(press_vel) * 1.001)  # not a solution
    calls = []

    def solve():
        calls.append(1)
        return press_vel, 0.0

    _, record = fs.resolve_flow_state(coarse, s, ALPHA_MAX,
                                      identity_cache=tmp_path / "absent.npz",
                                      bare_cache=bare, solve=solve)
    assert len(calls) == 1 and record["source"] == "solved"
    assert record["rejected"] and record["rejected"][0]["file"] == str(bare)

    with pytest.raises(RuntimeError, match="no trusted flow state"):
        fs.resolve_flow_state(coarse, s, ALPHA_MAX, identity_cache=tmp_path / "absent.npz")


# -- the load does not change with the flow mesh ----------------------------


def test_inlet_trace_and_inflow_are_the_same_on_both_flow_meshes(coarse):
    fine = dual.Zhao2DDualProblem(ref.refine_spec(SPEC, 2), CONFIG, 1, 3)
    traces = []
    for problem in (coarse, fine):
        h = problem.spec.element_size
        trace = fs.inlet_trace(problem.flow_mesh, SPEC, problem.flow_x0)
        assert trace["nodes"] == round(SPEC.inlet_half_width / h) + 1
        assert trace["max_deviation_from_imposed"] == 0.0
        assert trace["inflow_over_nominal"] == pytest.approx(1.0, rel=1e-14)
        # the inlet wins on both rims ...
        assert [n["v"] for n in trace["rim_wall"]] == [-SPEC.inlet_speed]
        assert [n["v"] for n in trace["rim_symmetry"]] == [-SPEC.inlet_speed]
        # ... so the wall slips over exactly one flow element next to the rim
        (slip,) = trace["wall_slip_next_to_rim"]
        assert slip["next_wall_node"]["v"] == 0.0
        assert slip["slip_length"] == pytest.approx(h, rel=1e-12)
        traces.append(trace)
    assert fs.compare_inlet_traces(*traces) == 0.0


def test_the_finest_density_is_the_same_by_either_route():
    """h -> h/4 directly, or h -> h/2 (a flow mesh) -> h/4: one density, one mesh."""
    direct = dual.Zhao2DDualProblem(SPEC, CONFIG, thermal_refinement=4, thermal_quadrature=3)
    via = dual.Zhao2DDualProblem(ref.refine_spec(SPEC, 2), CONFIG, 2, 3)
    for a, b in ((direct.thermal_mesh.mesh.nodes.coords, via.thermal_mesh.mesh.nodes.coords),
                 (direct.thermal_mesh.mesh.elem_nodes, via.thermal_mesh.mesh.elem_nodes),
                 (direct.q_source, via.q_source),
                 (direct.thermal_bc["fixed_dofs"], via.thermal_bc["fixed_dofs"])):
        assert np.array_equal(np.asarray(a), np.asarray(b))

    s = np.random.default_rng(3).uniform(size=direct.flow_mesh.num_elems)
    s[~direct.flow_mesh.design_mask] = 0.0
    s_half = ref.refine_design(direct.flow_mesh, via.flow_mesh, s)
    assert np.array_equal(np.asarray(direct.maps.density(s)),
                          np.asarray(via.maps.density(s_half)))


# -- the identities ------------------------------------------------------------


def test_divergence_theorem_holds_for_arbitrary_q1_fields(coarse):
    """Exact for any nodal values: T u is continuous, so interior edges cancel."""
    rng = np.random.default_rng(7)
    planar = coarse.thermal_mesh
    n, n_el = planar.mesh.num_nodes, planar.num_elems
    u = rng.normal(size=(n, 2))
    t = rng.normal(size=n)

    vol = fs.volume_terms(planar.mesh, u, t, b_f=SPEC.b_f)
    bnd = fs.boundary_terms(planar, u, t, b_f=SPEC.b_f)
    assert bnd["total"]["volume_flux"] == pytest.approx(vol["integral_div_u"],
                                                        rel=1e-10, abs=1e-16)
    assert bnd["total"]["enthalpy_flux"] == pytest.approx(
        vol["b_f_u_grad_T"] + vol["D_T"], rel=1e-10)

    elem_vel = jnp.asarray(u[np.asarray(planar.mesh.elem_nodes)].reshape(n_el, -1))
    c_adv = coarse.thermal.compliance_decomposition(
        jnp.asarray(t), elem_vel, jnp.full(n_el, 1.0), jnp.zeros(n_el)
    )["c_advective"]
    assert c_adv == pytest.approx(bnd["total"]["half_t2_flux"] - 0.5 * vol["D_T2"], rel=1e-10)


def test_volume_terms_on_fields_with_known_divergence(coarse):
    planar = coarse.thermal_mesh
    xy = np.asarray(planar.mesh.nodes.coords)
    area = float(np.sum(planar.elem_area))
    const = fs.volume_terms(planar.mesh, np.tile([0.3, -1.2], (len(xy), 1)),
                            np.ones(len(xy)), b_f=2.0)
    # zero up to the rounding of shape-function gradients that sum to zero
    assert abs(const["integral_div_u"]) < 1e-14 and abs(const["D_T"]) < 1e-12
    assert abs(const["b_f_u_grad_T"]) < 1e-12

    stretch = np.stack([3.0 * xy[:, 0], -1.0 * xy[:, 1]], axis=1)  # div u = 2
    out = fs.volume_terms(planar.mesh, stretch, np.ones(len(xy)), b_f=2.0)
    assert out["integral_div_u"] == pytest.approx(2.0 * area, rel=1e-12)
    assert out["integral_div_u_squared"] == pytest.approx(4.0 * area, rel=1e-12)
    assert out["D_T"] == pytest.approx(2.0 * 2.0 * area, rel=1e-12)


def test_every_identity_closes_on_a_solved_state(coarse, solved):
    s, press_vel, temperature = solved
    rep = fs.cell_report(coarse, s, press_vel, temperature, ALPHA_MAX)
    ident = rep["identities"]

    assert max(rep["residual_relative"].values()) < 1e-8
    assert ident["enthalpy"]["closure_relative"] < 1e-12
    assert ident["advective_half"]["closure_relative"] < 1e-12
    assert ident["stabilisation"]["closure_relative"] < 1e-12
    # the bracket is the Dirichlet reaction, to solver precision -- and is not zero
    source = ident["source"]
    assert abs(source["bracket_minus_reaction"]) < 1e-9 * rep["heat_in"]
    assert abs(source["bracket"]) > 100 * abs(source["bracket_minus_reaction"])

    # a second implementation of the same sums (conservation's per-face loops)
    heat = rep["heat"]
    assert rep["boundary"]["total"]["enthalpy_flux"] == pytest.approx(
        heat["enthalpy_net_out"], rel=1e-12)
    assert rep["boundary"]["total"]["conduction_out"] == pytest.approx(
        heat["conduction_net_out"], rel=1e-10, abs=1e-10)
    assert rep["divergence"]["D_T"] == pytest.approx(
        heat["temperature_weighted_divergence"], rel=1e-12)

    # P embeds the flow exactly, so sampling it finely cannot change its divergence
    div = rep["divergence"]
    assert div["thermal_mesh"]["integral_div_u_squared"] == pytest.approx(
        div["flow_mesh"]["integral_div_u_squared"], rel=1e-12)
    assert rep["mass"]["imbalance_relative"] < 1e-12
    assert rep["mass"]["wall_and_symmetry_flux"] == 0.0


# -- the committed evidence ------------------------------------------------------


def test_the_script_will_not_overwrite_records_already_in_its_output_dir(tmp_path):
    """A rerun stops before any work rather than replace a cited R1h record."""
    import subprocess
    import sys

    script = (pathlib.Path(__file__).resolve().parent.parent
              / "scripts" / "zhao2d_flow_mesh_check.py")
    record = tmp_path / "zhao2d_r1h_matrix.json"
    record.write_text("cited", encoding="utf-8")
    run = subprocess.run([sys.executable, str(script), "--out", str(tmp_path)],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode != 0
    assert "--overwrite" in run.stderr
    assert record.read_text(encoding="utf-8") == "cited"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["zhao2d_r1h_matrix.json"]


# -- the anchor ------------------------------------------------------------------


@pytest.mark.slow
def test_r1h_fine_flow_reloads_and_reproduces_r1f_row_d():
    """The saved h/2 flow, taken by the dispatch without a solve, gives R1f's D row."""
    results = pathlib.Path(__file__).resolve().parent.parent / "results"
    flow_file = results / "zhao2d_r1h_fine_flow.npz"
    fields = results / "zhao2d_r1d_main_fields.npz"
    if not (flow_file.is_file() and fields.is_file()):
        pytest.skip("R1h fine flow or R1d fields not present")

    spec = z.Zhao2DSpec()
    base = dual.Zhao2DDualProblem(spec, CONFIG, 1, 3)
    fine = dual.Zhao2DDualProblem(ref.refine_spec(spec, 2), CONFIG, 1, 3)
    s_fine = ref.refine_design(base.flow_mesh, fine.flow_mesh,
                               np.load(fields)["solid_fraction"])
    solve = _ForbiddenSolve()
    press_vel, record = fs.resolve_flow_state(fine, s_fine, ALPHA_MAX,
                                              identity_cache=flow_file, solve=solve)
    assert record["source"] == "identity cache" and solve.calls == 0
    c = fine.thermal.thermal_compliance(
        fine.solve_thermal(press_vel, s_fine, ALPHA_MAX),
        fine.thermal_velocity(press_vel),
        fine.thermal_conductivity(s_fine, ALPHA_MAX),
    )
    assert float(c) == pytest.approx(32400.115711650746, rel=1e-10)
