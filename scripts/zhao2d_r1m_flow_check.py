"""R1m: two qualified binary designs with the flow on h or h/2, on one thermal h/8.

    thermal h/8            flow h                           flow h/2
    x300's baseline        the h/8 check's states, reused   new: one flow, one thermal solve
    the pilot's terminal   likewise                         new: likewise

The designs are R1l's qualified binary designs, each a given s with 2000 fluid
cells in the design domain -- never filtered, projected or re-thresholded. On
h/2 the material is copied from the parent element, with no new design
variable, and the h/8 thermal mesh must receive the same material field from
either flow mesh.

Checked in stages, and any failure writes the record and exits non-zero before
the next build or solve:

* inputs: each s, flow and h/8 temperature is the state R1l and the h/8 check
  recorded (hashes), each design qualified, and the yardstick the h/8 check's;
* flow h, nothing solved: each saved flow is re-verified against its s, and
  each saved h/8 temperature re-evaluated must reproduce the h/8 check's Psi
  and C;
* flow h/2, before anything is solved: the thermal mesh, heat source and
  thermal Dirichlet nodes are stage one's, and each design's copy to h/2 keeps
  its fluid volume, keeps the tabs fluid and gives the same h/8 material;
* then per design one flow solve on h/2, gated as a given state (Dirichlet
  values exact, relative residual <= 1e-8), and one thermal solve on h/8 with
  that flow, every residual finite and within the gate.

Every cell is reported the same way (`zhao2d_flow_study.cell_report`), from its
states. J uses the h/4 model's frozen constants -- the yardstick of R1l and the
h/8 check -- as a common scale, not a reference for a fine-flow model; J* the
single-mesh ones. The heat balance is R1h's, H - Q - r_D = D_T, with r_D the
thermal residual at the Dirichlet nodes. Replacing the flow is one complete
response: the inlet-wins slip on the tab wall shortens with the flow mesh too,
so not every difference is the divergence's. Two flow meshes are not an exact
solution.

    python scripts/zhao2d_r1m_flow_check.py [--inputs results] [--out DIR] [--overwrite]
"""

from __future__ import annotations

import pathlib
import sys

# Default the BLAS thread count BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import faulthandler
import gc
import json
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))
sys.path.insert(0, str(REPO / "scripts"))

import toflux.src.solver as tf_solver  # noqa: E402

from tfopus import materials  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_binary as zb  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402
from tfopus import zhao2d_refine as ref  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb, timed  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file, thermal_inlet_velocity  # noqa: E402

ALPHA_MAX, QUADRATURE = 1.0e7, 3
GATE, ANCHOR_TOL = 1e-8, 1e-12
FLUID_BOUND = 0.4
RECORD = "zhao2d_r1m_flow_check.json"
FIELDS = "zhao2d_r1m_fields.npz"
STACKS = "zhao2d_r1m_stacks.log"
INPUTS = ("zhao2d_r1l_baselines.json", "zhao2d_r1l_baselines_fields.npz",
          "zhao2d_r1l_vp_pilot.json", "zhao2d_r1l_vp_pilot_fields.npz",
          "zhao2d_r1l_h8_check.json", "zhao2d_r1l_h8_fields.npz", "zhao2d_r1h_fine_flow.npz")
# (flow refinement against h, thermal refinement against THAT flow mesh): both reach h/8
ROWS = {"flow_h": (1, 8), "flow_h2": (2, 4)}
THERMAL_KEYS = ("elements", "nodes", "node_coords_sha256", "geometry_sha256", "region_sha256",
                "q_source_sha256", "dirichlet_sha256", "quadrature")


def load_json(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def mesh_identity(planar) -> dict:
    return {"elements": int(planar.num_elems), "nodes": int(planar.mesh.num_nodes),
            "node_coords_sha256": fs.digest(np.asarray(planar.mesh.nodes.coords)),
            "geometry_sha256": fs.digest(np.asarray(planar.mesh.nodes.coords),
                                         np.asarray(planar.mesh.elem_nodes)),
            "region_sha256": fs.digest(np.asarray(planar.region))}


def thermal_identity(problem) -> dict:
    """What both rows must share on the thermal side, by hash."""
    bc = problem.thermal_bc
    return {**mesh_identity(problem.thermal_mesh),
            "q_source_sha256": fs.digest(np.asarray(problem.q_source)),
            "dirichlet_sha256": fs.digest(np.asarray(bc["fixed_dofs"], dtype=np.int64),
                                          np.asarray(bc["dirichlet_values"], dtype=np.float64)),
            "quadrature": int(problem.thermal_quadrature)}


def flat(rep: dict, yard: dict, w: float) -> dict:
    """The headline numbers of one cell, and R1h's heat balance, from its report."""
    h, q = rep["identities"]["enthalpy"]["H"], rep["heat_in"]
    r_d, d_t = rep["reaction"]["reaction_at_dirichlet_nodes"], rep["divergence"]["D_T"]
    return {
        "psi": rep["psi"], "compliance": rep["compliance"],
        "J": w * rep["psi"] / yard["psi_0"] + (1 - w) * rep["compliance"] / yard["c_0"],
        "J_star": rep["j_star"],
        "c_advective": rep["c_advective"], "c_diffusive": rep["c_diffusive"],
        "T_max": rep["t_max"], "T_min": rep["min_temperature"],
        "nodes_below_inlet": rep["nodes_below_inlet"],
        "H": h, "Q": q, "r_D": r_d, "D_T": d_t, "D_T_over_Q": d_t / q,
        "balance_closure_relative": abs(h - q - r_d - d_t) / q,
        "integral_div_u_squared_flow_mesh": rep["divergence"]["flow_mesh"]["integral_div_u_squared"],
        "mass_imbalance_relative": rep["mass"]["imbalance_relative"],
        "residual_flow": rep["residual_relative"]["flow"],
        "residual_thermal": rep["residual_relative"]["thermal"],
    }


def public(trace: dict) -> dict:
    return {k: v for k, v in trace.items() if not k.startswith("_")}


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

    spec, config = z.Zhao2DSpec(), r1.R1Config()
    w = config.weight
    material = za.build_material(spec, ALPHA_MAX)
    single = r1.load_reference(spec, config)  # J* only
    a_rec = load_json(inputs / "zhao2d_r1l_baselines.json")
    b_rec = load_json(inputs / "zhao2d_r1l_vp_pilot.json")
    h8_rec = load_json(inputs / "zhao2d_r1l_h8_check.json")
    fa = np.load(inputs / "zhao2d_r1l_baselines_fields.npz")
    fb = np.load(inputs / "zhao2d_r1l_vp_pilot_fields.npz")
    f8 = np.load(inputs / "zhao2d_r1l_h8_fields.npz")
    designs = {
        "x300": {"s": fa["x300_solid_fraction_binary"], "press_vel": fa["x300_press_vel"],
                 "temperature_h8": f8["x300_temperature_h8"], "r1l": a_rec["designs"]["x300"],
                 "h8": h8_rec["cells"]["x300/h8"],
                 "source": "R1l A, x_300's qualified binary baseline"},
        "pilot": {"s": fb["solid_fraction_binary"], "press_vel": fb["binary_press_vel"],
                  "temperature_h8": f8["pilot_temperature_h8"], "r1l": b_rec["export"],
                  "h8": h8_rec["cells"]["pilot/h8"],
                  "source": "R1l B, the pilot's qualified binary terminal"},
    }

    record = {
        "note": ("R1m: x300's qualified binary baseline and the pilot's qualified binary "
                 "terminal, each a given s, with the flow on h (R1l's saved states) or h/2 "
                 "(solved here, the material copied from the parent element), on one thermal "
                 "h/8 mesh. J uses the h/4 model's constants as the common yardstick of R1l "
                 "and the h/8 check, J* the single-mesh ones; neither is a reference for a "
                 "fine-flow model. Two flow meshes are not an exact solution. Nothing "
                 "optimised, no gradient, no reference frozen."),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in INPUTS if (inputs / n).is_file()},
        "alpha_max": ALPHA_MAX, "thermal_quadrature": QUADRATURE, "gate": GATE,
        "anchor_tolerance": ANCHOR_TOL,
    }
    here = pathlib.Path(__file__).resolve()
    record["provenance"]["source_sha256"][here.relative_to(REPO).as_posix()] = sha256_file(here)
    failures, checkpoints, cost = [], [], {}
    cells, saved, rows, density, traces = {}, {}, {}, {}, {}

    def require(ok, what: str) -> bool:
        if not ok:
            failures.append(what)
        return bool(ok)

    def finish() -> None:
        record.update(rows=rows, density=density, cells=cells, checkpoints=checkpoints)
        if saved:
            np.savez_compressed(out / FIELDS, **saved)
            record["fields"] = {"file": FIELDS, "arrays": sorted(saved)}
        record["cost"] = {**cost, "peak_working_set_mb": peak_working_set_mb(),
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

    # -- inputs: the recorded states, and the yardstick ---------------------------
    for name, d in designs.items():
        s = np.asarray(d["s"])
        require(fs.digest(s) == d["r1l"]["binary_sha256"], f"{name}: s is not R1l's binary design")
        require(fs.digest(s) == d["h8"]["s_sha256"], f"{name}: s is not the h/8 check's")
        require(fs.digest(d["press_vel"]) == d["r1l"]["state_sha256"]["press_vel"],
                f"{name}: the flow is not R1l's saved state")
        require(fs.digest(d["temperature_h8"]) == d["h8"]["temperature_sha256"],
                f"{name}: the h/8 temperature is not the h/8 check's")
        require(d["r1l"].get("qualified", True) and d["r1l"].get("volume_feasible"),
                f"{name}: not a qualified design in R1l's record")
        require(int(d["r1l"]["fluid_cells"]) == 2000, f"{name}: not 2000 fluid cells in R1l's record")
        require(np.all((s == 0.0) | (s == 1.0)), f"{name}: s is not 0/1")
        require(zb.cell_usable(d["h8"]), f"{name}: the h/8 check's cell is not usable")
    ref_file = dual.reference_file(4, QUADRATURE)
    ref_values = load_json(ref_file)
    yard = {"psi_0": float(ref_values["psi_0"]), "c_0": float(ref_values["c_0"]),
            "file": ref_file.relative_to(REPO).as_posix(), "file_sha256": sha256_file(ref_file)}
    require(all(yard[k] == h8_rec["yardstick"][k] for k in ("psi_0", "c_0", "file")),
            "the yardstick is not the one R1l and the h/8 check used")
    record["yardstick"] = yard
    record["reporting_scale"] = {"psi_0": single.psi_0, "c_0": single.c_0,
                                 "identity": "single-mesh h reference; J* only"}
    record["designs"] = {name: {"source": d["source"], "s_sha256": fs.digest(d["s"]),
                                "press_vel_flow_h_sha256": fs.digest(d["press_vel"]),
                                "temperature_flow_h_thermal_h8_sha256": fs.digest(d["temperature_h8"])}
                         for name, d in designs.items()}
    checkpoint("inputs")

    # -- flow h: the saved states on thermal h/8, evaluated, nothing solved --------
    t0 = time.perf_counter()
    problem = dual.Zhao2DDualProblem(spec, config, thermal_refinement=ROWS["flow_h"][1],
                                     thermal_quadrature=QUADRATURE)
    cost["build_flow_h_s"] = time.perf_counter() - t0
    thermal_h = thermal_identity(problem)
    rows["flow_h"] = {"flow_mesh": mesh_identity(problem.flow_mesh), "thermal_mesh": thermal_h,
                      "flow_quadrature": int(problem.flow_mesh.mesh.gauss_order),
                      "nesting": problem.nesting, "t_build_s": cost["build_flow_h_s"]}
    print(f"built flow h ({problem.flow_mesh.num_elems} el) / thermal h/8 "
          f"({problem.thermal_mesh.num_elems} el): {cost['build_flow_h_s']:.1f} s", flush=True)
    require(thermal_h["node_coords_sha256"] == h8_rec["thermal_meshes"]["h8"]["node_coords_sha256"],
            "the h/8 thermal mesh is not the h/8 check's")
    flow_mesh_h = problem.flow_mesh
    for name, d in designs.items():
        s, pv, temp = jnp.asarray(d["s"]), jnp.asarray(d["press_vel"]), jnp.asarray(d["temperature_h8"])
        key = f"{name}/flow_h"
        fluid = z.fluid_fractions(flow_mesh_h, s)
        cell = {"design": name, "source": d["source"], "flow": "h: R1l's saved state",
                "temperature": "h/8: the h/8 check's saved state", "fluid_fractions": fluid,
                "volume_feasible": zb.volume_feasible(fluid["v_f_design_domain"], FLUID_BOUND)}
        cells[key] = cell
        require(cell["volume_feasible"], f"{key}: over the fluid bound, {fluid}")
        try:
            cell["flow_verification"] = fs.verify_flow_state(problem, pv, s, ALPHA_MAX, GATE)
        except (ValueError, r1.NotConverged) as exc:
            require(False, f"{key}: the saved flow does not verify: {exc}")
            continue
        t0 = time.perf_counter()
        rep = fs.cell_report(problem, s, pv, temp, ALPHA_MAX, single)
        cell.update(analysed=True, **flat(rep, yard, w))
        cell["gate_passed"] = zb.gate_passed(rep["residual_relative"], GATE)
        cell["anchor"] = zb.anchor_status(
            {"psi": rep["psi"], "compliance": rep["compliance"]},
            {"psi": d["h8"]["psi"], "compliance": d["h8"]["compliance"],
             "source": f"the h/8 check, {name}/h8"}, ANCHOR_TOL)
        cell["thermal_inlet"] = fs.thermal_inlet(problem.thermal_mesh, spec, temp)
        cell["thermal_inlet_velocity_max_deviation"] = thermal_inlet_velocity(problem, pv, spec)
        cell["t_report_s"] = time.perf_counter() - t0
        cell["report"] = rep
        require(cell["gate_passed"], f"{key}: a residual fails the gate {rep['residual_relative']}")
        require(cell["anchor"]["reproduced"], f"{key}: does not reproduce the h/8 check: {cell['anchor']}")
        traces[name] = {"flow_h": fs.inlet_trace(flow_mesh_h, spec, pv)}
        density[name] = {"s_flow_h_sha256": fs.digest(d["s"]),
                         "s_thermal_h8_sha256": fs.digest(np.asarray(problem.maps.density(s)))}
        print(f"{key}: Psi {cell['psi']:.9e}  C {cell['compliance']:.6f}  J {cell['J']:.9f}  "
              f"D_T/Q {cell['D_T_over_Q']:.4%}  |R| {cell['residual_flow']:.1e}/"
              f"{cell['residual_thermal']:.1e}  anchor "
              f"{'ok' if cell['anchor']['reproduced'] else 'MISMATCH'}  "
              f"({cell['t_report_s']:.1f} s)", flush=True)
    del problem
    gc.collect()
    cost["peak_working_set_mb_after_flow_h"] = peak_working_set_mb()
    checkpoint("flow h: the saved states on thermal h/8, re-evaluated")

    # -- flow h/2: everything checked before anything is solved --------------------
    t0 = time.perf_counter()
    problem = dual.Zhao2DDualProblem(ref.refine_spec(spec, ROWS["flow_h2"][0]), config,
                                     thermal_refinement=ROWS["flow_h2"][1],
                                     thermal_quadrature=QUADRATURE)
    cost["build_flow_h2_s"] = time.perf_counter() - t0
    thermal_f = thermal_identity(problem)
    rows["flow_h2"] = {"flow_mesh": mesh_identity(problem.flow_mesh), "thermal_mesh": thermal_f,
                       "flow_quadrature": int(problem.flow_mesh.mesh.gauss_order),
                       "nesting": problem.nesting, "t_build_s": cost["build_flow_h2_s"]}
    print(f"built flow h/2 ({problem.flow_mesh.num_elems} el) / thermal h/8 "
          f"({problem.thermal_mesh.num_elems} el): {cost['build_flow_h2_s']:.1f} s", flush=True)
    for k in THERMAL_KEYS:
        require(thermal_f[k] == thermal_h[k], f"the thermal side reached from flow h/2 differs in {k}")
    require(rows["flow_h2"]["flow_quadrature"] == rows["flow_h"]["flow_quadrature"],
            "the flow quadrature differs between the rows")
    r1h_flow = inputs / "zhao2d_r1h_fine_flow.npz"
    if r1h_flow.is_file():
        with np.load(r1h_flow) as f:
            r1h_mesh = json.loads(str(f["identity"]))["mesh"]
        mine = {"elements": rows["flow_h2"]["flow_mesh"]["elements"],
                "nodes": rows["flow_h2"]["flow_mesh"]["nodes"],
                "geometry_sha256": rows["flow_h2"]["flow_mesh"]["geometry_sha256"],
                "region_sha256": rows["flow_h2"]["flow_mesh"]["region_sha256"]}
        rows["flow_h2"]["same_flow_mesh_as_r1h"] = all(r1h_mesh[k] == v for k, v in mine.items())
    fine_s = {}
    for name, d in designs.items():
        try:
            s_f = np.asarray(ref.refine_design(flow_mesh_h, problem.flow_mesh, d["s"]))
            transfer = ref.check_transfer(flow_mesh_h, problem.flow_mesh, d["s"], s_f)
        except RuntimeError as exc:
            require(False, f"{name}: the copy to h/2 is not a pure refinement: {exc}")
            continue
        same = (fs.digest(np.asarray(problem.maps.density(jnp.asarray(s_f))))
                == density[name]["s_thermal_h8_sha256"])
        require(same, f"{name}: the h/8 material field depends on the flow mesh it came through")
        density[name].update(transfer=transfer, s_flow_h2_sha256=fs.digest(s_f),
                             same_thermal_material_via_either_flow_mesh=same)
        fine_s[name] = s_f
    checkpoint("flow h/2: meshes, material and source, before anything is solved")

    for name, d in designs.items():
        key = f"{name}/flow_h2"
        s_f = jnp.asarray(fine_s[name])
        alpha = materials.brinkman_penalty(s_f, material)
        fluid = z.fluid_fractions(problem.flow_mesh, s_f)
        cell = {"design": name, "source": d["source"],
                "flow": "h/2: solved here, the material copied from the parent element",
                "temperature": "h/8: solved here on that flow", "fluid_fractions": fluid,
                "volume_feasible": zb.volume_feasible(fluid["v_f_design_domain"], FLUID_BOUND)}
        cells[key] = cell
        pv, t_flow = timed(tf_solver.modified_newton_raphson_solve, problem.flow, problem.flow_x0, alpha)
        cell["t_flow_solve_s"] = t_flow
        saved[f"{name}_press_vel_flow_h2"] = np.asarray(pv)
        try:
            cell["flow_verification"] = fs.verify_flow_state(problem, pv, s_f, ALPHA_MAX, GATE)
        except (ValueError, r1.NotConverged) as exc:
            require(False, f"{key}: the flow solve fails its gate: {exc}")
        checkpoint(f"flow h/2: {name}'s flow solve ({t_flow:.1f} s)")
        cell["flow_identity"] = fs.flow_state_identity(problem, s_f, ALPHA_MAX)
        temp, t_thermal = timed(problem.solve_thermal, pv, s_f, ALPHA_MAX)
        cell["t_thermal_solve_s"] = t_thermal
        saved[f"{name}_temperature_flow_h2_thermal_h8"] = np.asarray(temp)
        t0 = time.perf_counter()
        rep = fs.cell_report(problem, s_f, pv, temp, ALPHA_MAX, single)
        cell.update(analysed=True, **flat(rep, yard, w))
        cell["gate_passed"] = zb.gate_passed(rep["residual_relative"], GATE)
        cell["thermal_inlet"] = fs.thermal_inlet(problem.thermal_mesh, spec, temp)
        cell["thermal_inlet_velocity_max_deviation"] = thermal_inlet_velocity(problem, pv, spec)
        cell["t_report_s"] = time.perf_counter() - t0
        cell["report"] = rep
        cell["state_sha256"] = {"s_flow_h2": fs.digest(np.asarray(s_f)),
                                "press_vel": fs.digest(np.asarray(pv)),
                                "temperature": fs.digest(np.asarray(temp))}
        traces[name]["flow_h2"] = fs.inlet_trace(problem.flow_mesh, spec, pv)
        saved[f"{name}_solid_fraction_flow_h2"] = np.asarray(s_f)
        print(f"{key}: Psi {cell['psi']:.9e}  C {cell['compliance']:.6f}  J {cell['J']:.9f}  "
              f"D_T/Q {cell['D_T_over_Q']:.4%}  |R| {cell['residual_flow']:.1e}/"
              f"{cell['residual_thermal']:.1e}  flow {t_flow:.1f} s, thermal {t_thermal:.1f} s",
              flush=True)
        require(cell["volume_feasible"], f"{key}: over the fluid bound, {fluid}")
        require(cell["gate_passed"], f"{key}: a residual fails the gate {rep['residual_relative']}")
        checkpoint(f"flow h/2: {name}'s thermal solve ({t_thermal:.1f} s)")
    saved["flow_node_coords_h2"] = np.asarray(problem.flow_mesh.mesh.nodes.coords)
    del problem
    gc.collect()

    # -- what the stage exists for ---------------------------------------------------
    replacement = {}
    for name in designs:
        a, b = cells[f"{name}/flow_h"], cells[f"{name}/flow_h2"]
        replacement[name] = {
            **{q: b[q] / a[q] - 1.0 for q in ("psi", "compliance", "J", "J_star", "c_advective",
                                               "c_diffusive", "T_max")},
            "D_T_over_Q": {"flow_h": a["D_T_over_Q"], "flow_h2": b["D_T_over_Q"]},
        }
    ranking = {}
    for row in ROWS:
        ca, cb = cells[f"pilot/{row}"], cells[f"x300/{row}"]
        kind = zb.comparison_kind(ca, cb)
        ranking[row] = None if kind is None else {
            "kind": kind, **{q: ca[q] / cb[q] - 1.0 for q in ("J", "J_star", "psi", "compliance")}}
    record["flow_replacement"] = replacement
    record["ranking"] = {"pilot_against_x300": ranking}
    record["inlet"] = {name: {"flow_h": public(t["flow_h"]), "flow_h2": public(t["flow_h2"]),
                              "max_difference_between_traces": fs.compare_inlet_traces(
                                  t["flow_h"], t["flow_h2"])}
                       for name, t in traces.items()}
    finish()

    print("\n" + " " * 28 + "".join(f"{k:>18s}" for k in cells))
    for label, q, fmt in (("Psi", "psi", "{:18.9e}"), ("C", "compliance", "{:18.6f}"),
                          ("J (h/4 yardstick)", "J", "{:18.9f}"), ("J* (single mesh)", "J_star", "{:18.9f}"),
                          ("T_max", "T_max", "{:18.6f}"), ("T_min", "T_min", "{:18.6f}"),
                          ("D_T/Q", "D_T_over_Q", "{:18.4%}"), ("H", "H", "{:18.6f}"),
                          ("r_D", "r_D", "{:18.6f}"), ("balance closure / Q", "balance_closure_relative", "{:18.1e}"),
                          ("|R| flow", "residual_flow", "{:18.1e}"), ("|R| thermal", "residual_thermal", "{:18.1e}")):
        print(f"  {label:26s}" + "".join(fmt.format(c[q]) for c in cells.values()))
    for name, v in replacement.items():
        print(f"\n  {name}, flow h -> h/2: Psi {v['psi']:+.3%}  C {v['compliance']:+.3%}  J {v['J']:+.3%}  "
              f"T_max {v['T_max']:+.3%}  D_T/Q {v['D_T_over_Q']['flow_h']:.2%} -> {v['D_T_over_Q']['flow_h2']:.2%}")
    for row, r in ranking.items():
        print(f"  pilot against x300, {row:7s}: " + ("not usable" if r is None else
              f"J {r['J']:+.3%} (Psi {r['psi']:+.3%}, C {r['compliance']:+.3%}) [{r['kind']}]"))
    print(f"\nwrote {out / RECORD} and {out / FIELDS}; total {record['cost']['wall_clock_total_s']:.0f} s")


if __name__ == "__main__":
    main()
