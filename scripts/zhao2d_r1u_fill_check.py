"""R1u: R1t's binary design with its isolated fluid cell filled, on D: one flow and one thermal solve.

The contract proposed in the review of 0650e7b. R1t's exported design has one
fluid cell, 3750, isolated by the shared-edge criterion the export rule uses:
its four edge neighbours are solid, and it touches the main channel only at a
corner. This builds a diagnostic geometry with an identity of its own: R1t's
binary s with that cell made solid and nothing else. There is no
re-projection or re-thresholding, no fluid is added elsewhere, and no channel
is opened. That leaves 1999 fluid cells in the design domain, within the bound
of 2000.

Filling a cell changes both the Brinkman resistance and the conductivity, so
both states are solved afresh on D (flow h/2, thermal h/8). The route is R1m's
check layer (a dual model built on h/2, the design copied down), as R1t's
binary design was solved: the flow on the Newton path, the temperature by one
linear solve. At most 1F + 1T; no MMA, no AD.

Checked in stages. A failed gate writes the record and exits non-zero before
the next build or solve:

* inputs: R1t's binary design and its D state, and R1r's D values, are the
  recorded ones; the scale is R1q's;
* the geometry: by shared edges, R1t's fluid has exactly two components, and
  the smaller is exactly cell 3750. Filling it changes that one design cell,
  leaves 1999 fluid cells in one component, and inlet and outlet still
  connect;
* the check-layer route: R1m's meshes. The copy to h/2 is a pure refinement
  and differs from R1t's saved one in the filled cell's four children only;
  the h/8 material is the parents';
* one flow solve, gated; one thermal solve, gated.

The filled design is compared, on D at the common scale, with R1t's design and
R1r's, from their saved values; neither is re-solved. The result stands
whatever it shows.

    python scripts/zhao2d_r1u_fill_check.py [--inputs results] [--out DIR] [--overwrite]
"""

from __future__ import annotations

import pathlib
import sys

# Default the BLAS thread count BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import faulthandler
import json
import time

import numpy as np
import scipy.sparse.csgraph as csgraph
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))
sys.path.insert(0, str(REPO / "scripts"))

import toflux.src.solver as tf_solver  # noqa: E402

from tfopus import affine_solve as affine  # noqa: E402
from tfopus import materials  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_binary as zb  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402
from tfopus import zhao2d_refine as ref  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb, timed  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file  # noqa: E402
# R1m's report, so the filled design on D is summarised exactly as the candidates were
from zhao2d_r1m_flow_check import flat, mesh_identity, thermal_identity  # noqa: E402

ALPHA_MAX = 1.0e7
FLOW_REFINEMENT, THERMAL_REFINEMENT, QUADRATURE = 2, 4, 3
THERMAL_PATH = "linear"
GATE = 1e-8
FILLED_CELL = 3750  # as the review of 0650e7b named it; the script also finds it
FLUID_CELLS_AFTER = 1999
RECORD = "zhao2d_r1u.json"
FIELDS = "zhao2d_r1u_fields.npz"
STACKS = "zhao2d_r1u_stacks.log"
INPUTS = ("zhao2d_r1m_flow_check.json", "zhao2d_r1q_fineflow_check.json", "zhao2d_r1r.json",
          "zhao2d_r1t.json", "zhao2d_r1t_fields.npz")
NAMES = ("r1t", "r1r")
COMPARED = ("J", "J_star", "psi", "compliance", "T_max", "D_T_over_Q")


def rel(b, a) -> float:
    return b / a - 1.0


def load_json(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def load_fields(path: pathlib.Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def fluid_components(planar, s) -> list[np.ndarray]:
    """The fluid's components by shared edges -- the export rule's criterion --
    each as its cell indices, the largest first."""
    fluid = np.asarray(s) < 0.5
    num, labels = csgraph.connected_components(z.element_adjacency(planar)[fluid][:, fluid],
                                               directed=False)
    cells = np.nonzero(fluid)[0]
    return sorted((cells[labels == k] for k in range(num)), key=len, reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=pathlib.Path, default=REPO / "results")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    inputs, out = args.inputs, args.out
    existing = [n for n in (RECORD, FIELDS) if (out / n).exists()]
    if existing and not args.overwrite:
        sys.exit(f"{out} already holds {', '.join(existing)}, which is cited evidence; write "
                 "elsewhere with --out DIR, or pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    stacks = open(out / STACKS, "w", encoding="utf-8")
    faulthandler.dump_traceback_later(3600, repeat=True, file=stacks)  # longer than a healthy run
    t_start = time.perf_counter()

    spec = z.Zhao2DSpec()
    config = r1.R1Config(projection=r1.Projection.VOLUME_PRESERVING)
    w = config.weight
    material = za.build_material(spec, ALPHA_MAX)
    single = r1.load_reference(spec, config)  # J* only
    m_rec = load_json(inputs / "zhao2d_r1m_flow_check.json")
    q_rec = load_json(inputs / "zhao2d_r1q_fineflow_check.json")
    r_rec = load_json(inputs / "zhao2d_r1r.json")
    t_rec = load_json(inputs / "zhao2d_r1t.json")
    ft = load_fields(inputs / "zhao2d_r1t_fields.npz")
    rows = m_rec["rows"]
    base = {"r1t": t_rec["cells"]["new/check"], "r1r": r_rec["cells"]["new/check"]}
    s_t = np.asarray(ft["solid_fraction_binary"], dtype=np.float64)

    record = {
        "note": ("R1u: R1t's qualified binary design with its one fluid cell isolated by shared "
                 "edges (3750) made solid, nothing else changed: a diagnostic geometry with its "
                 "own identity, solved afresh on the check layer D (flow h/2, thermal h/8; the "
                 "flow on the Newton path, the temperature by one linear solve) and compared with "
                 "R1t's and R1r's saved D values at the common scale. No MMA, no AD."),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in INPUTS},
        "contract": {"geometry": f"R1t's qualified binary design with cell {FILLED_CELL} filled, "
                                 "nothing else: no re-projection, no re-threshold, no fluid "
                                 "added, no channel opened",
                     "model": {"flow_refinement": FLOW_REFINEMENT,
                               "thermal_refinement": THERMAL_REFINEMENT,
                               "thermal_quadrature": QUADRATURE, "thermal_path": THERMAL_PATH},
                     "alpha_max": ALPHA_MAX, "gate": GATE, "weight": w,
                     "solves_at_most": {"flow": 1, "thermal": 1}, "mma_updates": 0,
                     "authorised": "by the user, on the local CPU, as the review of 0650e7b states it"},
    }
    for script in (pathlib.Path(__file__).resolve(), REPO / "scripts" / "zhao2d_r1m_flow_check.py"):
        record["provenance"]["source_sha256"][script.relative_to(REPO).as_posix()] = sha256_file(script)
    failures, checkpoints, cost, cells, saved = [], [], {}, {}, {}

    def require(ok, what: str) -> bool:
        if not ok:
            failures.append(what)
        return bool(ok)

    def finish() -> None:
        record.update(cells=cells, checkpoints=checkpoints)
        if saved:
            np.savez_compressed(out / FIELDS, **saved)
            record["fields"] = {"file": FIELDS, "arrays": sorted(saved)}
        record["cost"] = {**cost, "peak_working_set_mb": peak_working_set_mb(),
                          "peak_working_set_note": "the whole process's, cumulative",
                          "wall_clock_total_s": time.perf_counter() - t_start}
        faulthandler.cancel_dump_traceback_later()
        stacks.close()
        if (out / STACKS).stat().st_size == 0:
            (out / STACKS).unlink()
        (out / RECORD).write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")

    def checkpoint(stage: str) -> None:
        """Stop here, with the record written, if anything so far has failed."""
        checkpoints.append({"stage": stage, "failures": list(failures)})
        if failures:
            record["stopped"] = f"{stage}: checks failed; nothing after it was built or solved"
            finish()
            sys.exit(f"CHECKS FAILED at {stage}; nothing after it was built or solved: {failures}")
        print(f"checkpoint passed: {stage}", flush=True)

    # -- inputs ---------------------------------------------------------------------------
    require(fs.digest(s_t) == t_rec["export"]["binary_sha256"],
            "R1t's saved binary design is not the one it exported")
    require(t_rec["export"]["qualified"], "R1t's export is not qualified in its record")
    for name, rec_ in (("R1m", m_rec), ("R1q", q_rec), ("R1r", r_rec), ("R1t", t_rec)):
        require(not any(c["failures"] for c in rec_["checkpoints"]), f"{name}'s record has failed checks")
    sha = base["r1t"]["state_sha256"]
    for mine, theirs in (("new_solid_fraction_flow_h2", "s"), ("new_press_vel_check", "press_vel"),
                         ("new_temperature_check", "temperature")):
        require(fs.digest(ft[mine]) == sha[theirs], f"R1t's saved {mine} is not its D state")
    for name in NAMES:
        require(zb.cell_usable(base[name]) and base[name]["volume_feasible"],
                f"{name}: its D cell is not a usable, gated, feasible state in its record")
    ref_path = dual.reference_file(4, QUADRATURE)
    values = r1.ReferenceValues.from_json(ref_path.read_text(encoding="utf-8"))
    yard = {"psi_0": values.psi_0, "c_0": values.c_0, "file": ref_path.relative_to(REPO).as_posix(),
            "file_sha256": sha256_file(ref_path), "declared": "the common scale"}
    require(all(yard[k] == q_rec["scale"][k] for k in ("psi_0", "c_0"))
            and yard["file_sha256"] == q_rec["scale"]["source_sha256"],
            "the development reference is not the common scale R1q, R1r and R1t used")
    for name in NAMES:
        j = w * base[name]["psi"] / yard["psi_0"] + (1 - w) * base[name]["compliance"] / yard["c_0"]
        require(base[name]["J"] == j, f"{name}: its recorded D value of J is not on this scale")
    record["scale"] = yard
    checkpoint("inputs")

    # -- the geometry: find the isolated cell, fill it, check what that leaves -------------------
    planar = z.build_mesh(spec, dofs_per_node=3)
    design = np.asarray(planar.design_mask, dtype=bool)
    areas = np.asarray(planar.elem_area)
    inlet, outlet = zb.port_elements(planar, z.Face.INLET), zb.port_elements(planar, z.Face.OUTLET)
    before = fluid_components(planar, s_t)
    geometry = {"components_before": [int(len(c)) for c in before]}
    record["geometry"] = geometry
    require(len(before) == 2 and before[1].tolist() == [FILLED_CELL],
            f"R1t's fluid is not one main component plus cell {FILLED_CELL} alone: "
            f"{[c.tolist() if len(c) < 5 else len(c) for c in before]}")
    checkpoint("the geometry: R1t's isolated cell")
    neighbours = np.unique(z.element_adjacency(planar)[FILLED_CELL].nonzero()[1])
    neighbours = neighbours[neighbours != FILLED_CELL]
    s_fill = s_t.copy()
    s_fill[FILLED_CELL] = 1.0
    after = fluid_components(planar, s_fill)
    n_fluid, _, joined, _, _ = z.fluid_connectivity(planar, s_fill < 0.5, inlet, outlet)
    fractions = z.fluid_fractions(planar, jnp.asarray(s_fill))
    geometry.update(
        filled_cell=FILLED_CELL, filled_cell_is_design=bool(design[FILLED_CELL]),
        centre_mm=(np.asarray(planar.elem_centres)[FILLED_CELL] * 1e3).tolist(),
        continuous_s_in_r1t=float(ft["solid_fraction"][FILLED_CELL]),
        edge_neighbours=neighbours.tolist(),
        edge_neighbours_solid=bool(np.all(s_t[neighbours] >= 0.5)),
        cells_changed=int(np.sum(s_fill != s_t)),
        fluid_cells_design_domain=int(np.sum((s_fill < 0.5) & design)),
        fluid_fractions=fractions,
        volume_feasible=zb.volume_feasible(fractions["v_f_design_domain"], config.max_fluid_fraction),
        components_after=[int(len(c)) for c in after], inlet_outlet_connected=bool(joined),
        derived_from={"r1t_binary_sha256": t_rec["export"]["binary_sha256"]},
        sha256=fs.digest(s_fill),
        area_check=float(np.sum(areas[design & (s_fill < 0.5)]) / np.sum(areas[design])))
    require(geometry["filled_cell_is_design"], f"cell {FILLED_CELL} is not a design cell")
    require(s_t[FILLED_CELL] == 0.0, f"cell {FILLED_CELL} is not fluid in R1t's design")
    require(geometry["edge_neighbours_solid"] and len(neighbours) == 4,
            f"cell {FILLED_CELL}'s edge neighbours are not four solid cells")
    require(geometry["cells_changed"] == 1, f"{geometry['cells_changed']} cells changed, not one")
    require(geometry["fluid_cells_design_domain"] == FLUID_CELLS_AFTER,
            f"{geometry['fluid_cells_design_domain']} fluid cells, not {FLUID_CELLS_AFTER}")
    require(geometry["volume_feasible"], f"over the fluid bound: {fractions}")
    require(len(after) == 1 and joined, "the filled design's fluid is not one component joining "
                                        "inlet and outlet")
    saved["solid_fraction_filled"] = s_fill
    print(f"geometry: cell {FILLED_CELL} at {np.round(geometry['centre_mm'], 3).tolist()} mm (s in R1t's "
          f"continuous design {geometry['continuous_s_in_r1t']:.4f}) filled; components "
          f"{geometry['components_before']} -> {geometry['components_after']}, "
          f"{geometry['fluid_cells_design_domain']} fluid cells, v_f "
          f"{fractions['v_f_design_domain']:.6f}, inlet-outlet {joined}", flush=True)
    checkpoint("the geometry: the filled design")

    # -- the check-layer route, before its solves ---------------------------------------------
    t0 = time.perf_counter()
    fine = dual.Zhao2DDualProblem(ref.refine_spec(spec, FLOW_REFINEMENT), config,
                                  thermal_refinement=THERMAL_REFINEMENT,
                                  thermal_quadrature=QUADRATURE)
    cost["build_check_s"] = time.perf_counter() - t0
    print(f"built the check-layer route: flow {fine.flow_mesh.num_elems} el, thermal "
          f"{fine.thermal_mesh.num_elems} el ({cost['build_check_s']:.1f} s)", flush=True)
    record["check_layer"] = {"flow_mesh": mesh_identity(fine.flow_mesh),
                             "thermal_mesh": thermal_identity(fine)}
    require(record["check_layer"]["flow_mesh"] == rows["flow_h2"]["flow_mesh"],
            "the h/2 flow mesh is not R1m's")
    require(record["check_layer"]["thermal_mesh"] == rows["flow_h2"]["thermal_mesh"],
            "the h/8 thermal side is not R1m's")
    s_f = np.asarray(ref.refine_design(planar, fine.flow_mesh, s_fill))
    try:
        record["check_layer"]["transfer"] = ref.check_transfer(planar, fine.flow_mesh, s_fill, s_f)
    except RuntimeError as exc:
        require(False, f"the copy to h/2 is not a pure refinement: {exc}")
    parents_f = ref.parent_of_each_fine_element(planar, fine.flow_mesh)
    changed = np.nonzero(s_f != ft["new_solid_fraction_flow_h2"])[0]
    record["check_layer"]["flow_cells_changed_from_r1t"] = changed.tolist()
    require(len(changed) == 4 and np.all(np.asarray(parents_f)[changed] == FILLED_CELL),
            "the copy to h/2 differs from R1t's elsewhere than the filled cell's four children")
    parents_t = ref.parent_of_each_fine_element(planar, fine.thermal_mesh)
    same_material = bool(np.array_equal(np.asarray(fine.maps.density(jnp.asarray(s_f))),
                                        s_fill[parents_t]))
    record["check_layer"]["thermal_material_is_the_parents"] = same_material
    require(same_material, "the h/8 material is not the parents' material")
    checkpoint("the check-layer route: before its solves")

    # -- one flow solve and one thermal solve ------------------------------------------------
    key = "filled/check"
    s = jnp.asarray(s_f)
    fluid = z.fluid_fractions(fine.flow_mesh, s)
    cell = {"design": "filled", "layer": "check", "state": "solved here", "fluid_fractions": fluid,
            "thermal_path": THERMAL_PATH,
            "volume_feasible": zb.volume_feasible(fluid["v_f_design_domain"],
                                                  config.max_fluid_fraction)}
    cells[key] = cell
    alpha = materials.brinkman_penalty(s, material)
    pv, t_flow = timed(tf_solver.modified_newton_raphson_solve, fine.flow, fine.flow_x0, alpha)
    cell["t_flow_solve_s"] = t_flow
    saved.update(solid_fraction_flow_h2=s_f, press_vel=np.asarray(pv))
    try:
        cell["flow_verification"] = fs.verify_flow_state(fine, pv, s, ALPHA_MAX, GATE)
    except (ValueError, r1.NotConverged) as exc:
        require(False, f"{key}: the flow solve fails its gate: {exc}")
    checkpoint(f"the flow solve ({t_flow:.1f} s)")
    cell["flow_identity"] = fs.flow_state_identity(fine, s, ALPHA_MAX)
    temp, t_thermal = timed(
        lambda: affine.affine_solve(fine.thermal, fine.thermal_x0, fine.thermal_velocity(pv),
                                    fine.thermal_conductivity(s, ALPHA_MAX), fine.q_source))
    cell["t_thermal_solve_s"] = t_thermal
    saved["temperature"] = np.asarray(temp)
    t0 = time.perf_counter()
    rep = fs.cell_report(fine, s, pv, temp, ALPHA_MAX, single)
    cell.update(analysed=True, **flat(rep, yard, w))
    cell["gate_passed"] = zb.gate_passed(rep["residual_relative"], GATE)
    cell["t_report_s"] = time.perf_counter() - t0
    cell["report"] = rep
    cell["state_sha256"] = {"s": fs.digest(np.asarray(s)), "press_vel": fs.digest(np.asarray(pv)),
                            "temperature": fs.digest(np.asarray(temp))}
    print(f"{key}: Psi {cell['psi']:.9e}  C {cell['compliance']:.6f}  J {cell['J']:.9f}  "
          f"D_T/Q {cell['D_T_over_Q']:.4%}  T_max {cell['T_max']:.4f}  |R| "
          f"{cell['residual_flow']:.1e}/{cell['residual_thermal']:.1e}  flow {t_flow:.1f} s, "
          f"thermal {t_thermal:.1f} s", flush=True)
    require(cell["volume_feasible"], f"{key}: over the fluid bound, {fluid}")
    require(cell["gate_passed"], f"{key}: a residual fails the gate {rep['residual_relative']}")
    checkpoint(f"the thermal solve ({t_thermal:.1f} s)")

    # -- against R1t's and R1r's saved D values ------------------------------------------------
    record["candidates"] = {n: {k: base[n].get(k) for k in COMPARED} for n in NAMES}
    record["against"] = {}
    for n in NAMES:
        kind = zb.comparison_kind(cell, base[n])
        record["against"][n] = None if kind is None else {
            "kind": kind, **{q: rel(cell[q], base[n][q]) for q in COMPARED},
            "terms": {"dissipation": 0.5 * (cell["psi"] - base[n]["psi"]) / yard["psi_0"],
                      "thermal_compliance": 0.5 * (cell["compliance"] - base[n]["compliance"])
                      / yard["c_0"],
                      "dJ": cell["J"] - base[n]["J"]}}
    cost["solves"] = {"flow": 1, "thermal": 1, "mma": 0, "reverse_passes": 0}
    finish()

    for n, v in record["against"].items():
        print(f"filled design against {n:3s} on D: " + (
            "not usable" if v is None else
            f"J {v['J']:+.3%}  Psi {v['psi']:+.3%}  C {v['compliance']:+.3%}  T_max "
            f"{v['T_max']:+.3%}  (terms {v['terms']['dissipation']:+.8f} "
            f"{v['terms']['thermal_compliance']:+.8f}) [{v['kind']}]"))
    print(f"\nwrote {out / RECORD} and {out / FIELDS}; total "
          f"{record['cost']['wall_clock_total_s']:.0f} s")


if __name__ == "__main__":
    main()
