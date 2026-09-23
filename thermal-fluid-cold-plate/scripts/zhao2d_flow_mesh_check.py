"""R1h: the R1d design with the flow at h or h/2, on a common thermal mesh.

    flow \\ thermal    h/2                        h/4
    h                 R1g level 2 (states reused)  R1g level 4 (states reused)
    h/2               R1f row D (re-analysed)      new

The coarse-flow row is R1g's saved states. Nothing there is re-solved; every
number is re-evaluated on them, starting with the residual gate.

The fine flow is R1f's cached h/2 solve, which was saved as a bare array with
no record of what it solves. It is checked against the problem it must belong
to -- its shape, the Dirichlet values it carries, its residual under this
density, alpha, boundary conditions and properties, and its Psi against R1e's
separate solve of the same problem -- and only then reused, and saved again
WITH that identity. If any check fails, the flow is solved once instead. The
two thermal solves then both use that one fine flow.

The physical density is R1d's, copied from the parent element down to every
finer mesh: no filter, no projection, no design variable on a fine mesh. Flow
2x2, thermal 3x3; the physics, the forms and the boundary conditions are the
frozen ones.

J* divides by the single-mesh frozen Psi_0 and C_0 as a common reporting scale
only. Nothing here is a reference, an objective or a gradient.

    python scripts/zhao2d_flow_mesh_check.py
"""

from __future__ import annotations

import pathlib
import sys

# Default the BLAS thread count BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads as threads  # noqa: E402

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time

import numpy as np
import scipy
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))
sys.path.insert(0, str(REPO / "scripts"))

import toflux.src.solver as tf_solver  # noqa: E402

from tfopus import materials as materials  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402
from tfopus import zhao2d_refine as ref  # noqa: E402

# R1g's measurement helpers, so memory and time mean what they meant there
from zhao2d_dual_check import peak_working_set_mb, timed  # noqa: E402

QUAD = 3

# Recorded values this matrix must reproduce, at full precision.
R1F_C = 33233.13278907864  # zhao2d_r1f_separation.json, row C
R1F_D = 32400.115711650746  # row D: flow h/2, thermal h/2, 3x3
R1E_PSI_FINE = 0.014067841749016196  # zhao2d_r1e_refine.json, continuous/fine
PSI_SAME_STATE_RTOL = 1e-10

R1F_CACHE = "zhao2d_r1f_fine_flow.npz"
FINE_FLOW = "zhao2d_r1h_fine_flow.npz"
INPUTS = (
    "zhao2d_r1d_main.json",
    "zhao2d_r1d_main_fields.npz",
    "zhao2d_r1g_dual.json",
    "zhao2d_r1g_fields.npz",
    "zhao2d_r1f_separation.json",
    "zhao2d_r1e_refine.json",
    R1F_CACHE,
)

# (flow refinement against h, thermal refinement against THAT flow mesh)
CELLS = {
    "flow_h__thermal_h2": (1, 2),
    "flow_h__thermal_h4": (1, 4),
    "flow_h2__thermal_h2": (2, 1),
    "flow_h2__thermal_h4": (2, 2),
}
R1G_LEVEL = {"flow_h__thermal_h2": "2", "flow_h__thermal_h4": "4"}


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def provenance() -> dict:
    """The code and environment this record came from."""

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception as exc:  # noqa: BLE001 -- recorded, not fatal
            return f"unavailable: {exc}"

    here = pathlib.Path(__file__).resolve()
    sources = sorted(p.relative_to(REPO).as_posix() for p in (REPO / "tfopus").glob("*.py"))
    sources += [here.relative_to(REPO).as_posix(), "scripts/zhao2d_dual_check.py"]
    toflux_src = REPO / "external" / "TOFLUX" / "toflux" / "src"
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain").splitlines(),
        "source_sha256": {s: sha256_file(REPO / s) for s in sources},
        "toflux_src_sha256": {p.name: sha256_file(p) for p in sorted(toflux_src.glob("*.py"))},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "x64": bool(jax.config.jax_enable_x64),
            "threads": {v: os.environ.get(v) for v in threads.VARIABLES},
            "cpu_count": os.cpu_count(),
        },
    }


def same_mesh(a: z.PlanarMesh, b: z.PlanarMesh) -> bool:
    return (np.array_equal(np.asarray(a.mesh.nodes.coords), np.asarray(b.mesh.nodes.coords))
            and np.array_equal(np.asarray(a.mesh.elem_nodes), np.asarray(b.mesh.elem_nodes))
            and np.array_equal(a.region, b.region))


def thermal_inlet_velocity(problem, press_vel, spec) -> float:
    """Largest departure of u_T from (0, -U) on the THERMAL mesh's inlet nodes."""
    nodes = z.face_nodes(problem.thermal_mesh, z.Face.INLET)
    u_t = np.asarray(problem.thermal_nodal_velocity(press_vel))[nodes]
    return float(np.max(np.abs(u_t - np.array([0.0, -spec.inlet_speed]))))


def summary(rep: dict) -> dict:
    """The headline numbers of one cell, flat, for the table and the differences."""
    dv = rep["divergence"]
    return {
        "compliance": rep["compliance"],
        "c_advective": rep["c_advective"],
        "c_diffusive": rep["c_diffusive"],
        "l_q": rep["l_q"],
        "net_supg": rep["d_supg"] - rep["f_supg"],
        "net_supg_over_c": rep["net_supg_over_c"],
        "t_max": rep["t_max"],
        "min_temperature": rep["min_temperature"],
        "nodes_below_inlet": rep["nodes_below_inlet"],
        "psi": rep["psi"],
        "j_star": rep["j_star"],
        "D_T": dv["D_T"],
        "D_T_over_heat_in": dv["D_T_over_heat_in"],
        "D_T2": dv["D_T2"],
        "minus_half_D_T2": dv["minus_half_D_T2"],
        "minus_half_D_T2_over_C": dv["minus_half_D_T2_over_C"],
        "integral_div_u_squared": dv["flow_mesh"]["integral_div_u_squared"],
        "H": rep["identities"]["enthalpy"]["H"],
        "bracket": rep["identities"]["source"]["bracket"],
        "dirichlet_reaction": rep["identities"]["source"]["dirichlet_reaction"],
        "residual_flow": rep["residual_relative"]["flow"],
        "residual_thermal": rep["residual_relative"]["thermal"],
    }


DIFF_KEYS = ("compliance", "c_advective", "c_diffusive", "l_q", "net_supg", "t_max",
             "psi", "j_star", "D_T", "minus_half_D_T2", "H", "bracket")


def difference(a: dict, b: dict) -> dict:
    return {k: {"abs": b[k] - a[k], "rel": (b[k] / a[k] - 1.0) if a[k] != 0 else None}
            for k in DIFF_KEYS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results")
    args = ap.parse_args()
    out = args.out
    t_start = time.perf_counter()
    memory = {}

    record = {
        "note": (
            "R1h fixed-design check: the R1d continuous design with the flow solved "
            "at h or h/2, each on the thermal meshes h/2 and h/4. Two meshes are "
            "not an exact solution; differences between cells are mesh "
            "differences, not errors. D_T and -1/2 D_T2 are algebraic shares of "
            "each solution. J* uses the single-mesh frozen Psi_0 and C_0 as a "
            "common reporting scale only."
        ),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(out / n) for n in INPUTS if (out / n).is_file()},
        "reference_file_sha256": sha256_file(r1.REFERENCE_FILE),
        "thermal_quadrature": QUAD,
    }
    record["inputs_sha256"]["missing"] = [n for n in INPUTS if not (out / n).is_file()]

    # -- inputs -------------------------------------------------------------
    meta = json.loads((out / "zhao2d_r1d_main.json").read_text(encoding="utf-8"))
    r1g_meta = json.loads((out / "zhao2d_r1g_dual.json").read_text(encoding="utf-8"))
    alpha_max, beta = meta["final_alpha_max"], meta["final_beta"]
    r1d = np.load(out / "zhao2d_r1d_main_fields.npz")
    r1g = np.load(out / "zhao2d_r1g_fields.npz")
    s_d = np.asarray(r1d["solid_fraction"])
    pv_h = jnp.asarray(r1d["press_vel"])
    if not (np.array_equal(r1g["solid_fraction"], s_d)
            and np.array_equal(r1g["press_vel_coarse"], np.asarray(pv_h))):
        sys.exit("R1g's saved states are not the R1d design and flow; stopping")
    if r1g_meta["alpha_max"] != alpha_max or r1g_meta["beta"] != beta:
        sys.exit("R1g and R1d disagree on alpha_max or beta; stopping")

    spec, config = z.Zhao2DSpec(), r1.R1Config()
    fine_spec = ref.refine_spec(spec, 2)
    scale = r1.load_reference(spec, config)  # single-mesh; a reporting scale here
    record.update(alpha_max=alpha_max, beta=beta, run_fingerprint=config.fingerprint(),
                  reporting_scale={"psi_0": scale.psi_0, "c_0": scale.c_0,
                                   "identity": "single-mesh h = 1e-4 reference"})
    print(f"R1d design, alpha_max {alpha_max:.3e}, beta {beta:g}; flow 2x2, thermal "
          f"{QUAD}x{QUAD}; J* on the single-mesh reference as a reporting scale\n")

    # -- the four problems ----------------------------------------------------
    problems, build_s = {}, {}
    for key, (rf, rt) in CELLS.items():
        t0 = time.perf_counter()
        problems[key] = dual.Zhao2DDualProblem(
            spec if rf == 1 else fine_spec, config,
            thermal_refinement=rt, thermal_quadrature=QUAD,
        )
        build_s[key] = time.perf_counter() - t0
        print(f"built {key}: flow {problems[key].flow_mesh.num_elems} el, thermal "
              f"{problems[key].thermal_mesh.num_elems} el ({build_s[key]:.1f} s)", flush=True)
    memory["after_builds"] = peak_working_set_mb()
    p2, p4 = problems["flow_h__thermal_h2"], problems["flow_h__thermal_h4"]
    f2, f4 = problems["flow_h2__thermal_h2"], problems["flow_h2__thermal_h4"]

    # -- one physical density, copied down -------------------------------------
    s_f = np.asarray(ref.refine_design(p2.flow_mesh, f2.flow_mesh, s_d))
    transfer = ref.check_transfer(p2.flow_mesh, f2.flow_mesh, s_d, s_f)
    parents_flow = ref.parent_of_each_fine_element(p2.flow_mesh, f2.flow_mesh)
    density = {
        "flow_h_to_h2": transfer,
        "flow_h_to_h2_parents_sha256": fs.digest(parents_flow.astype(np.int64)),
        "s_flow_h_sha256": fs.digest(s_d),
        "s_flow_h2_sha256": fs.digest(s_f),
    }
    for col, (a, b) in {"h2": (p2, f2), "h4": (p4, f4)}.items():
        if not same_mesh(a.thermal_mesh, b.thermal_mesh):
            sys.exit(f"the two thermal meshes of column {col} differ; stopping")
        via_coarse = np.asarray(a.maps.density(s_d))
        via_fine = np.asarray(b.maps.density(s_f))
        if not np.array_equal(via_coarse, via_fine):
            sys.exit(f"thermal density in column {col} depends on the route; stopping")
        if not np.array_equal(np.asarray(a.q_source), np.asarray(b.q_source)):
            sys.exit(f"heat source differs in column {col}; stopping")
        if not np.array_equal(np.asarray(a.thermal_bc["fixed_dofs"]),
                              np.asarray(b.thermal_bc["fixed_dofs"])):
            sys.exit(f"thermal Dirichlet nodes differ in column {col}; stopping")
        density[f"thermal_{col}"] = {
            "same_mesh_in_both_rows": True,
            "density_same_via_either_flow_mesh": True,
            "elements": int(a.thermal_mesh.num_elems),
            "s_thermal_sha256": fs.digest(via_coarse),
            "geometry_sha256": fs.digest(np.asarray(a.thermal_mesh.mesh.nodes.coords),
                                         np.asarray(a.thermal_mesh.mesh.elem_nodes)),
        }
    record["density"] = density
    print(f"density: h -> h/2 copied, fluid fractions kept "
          f"(design {transfer['fine_v_f_design_domain']:.10f}); both thermal columns "
          f"reach the same mesh and the same density from either flow mesh")

    # -- the coarse flow: R1g's state, gated ----------------------------------
    flows = {}
    check_h = fs.verify_flow_state(p2, pv_h, s_d, alpha_max)
    flows["h"] = {
        "source": "R1d/R1g saved state (results/zhao2d_r1d_main_fields.npz, press_vel)",
        "identity": fs.flow_state_identity(p2, s_d, alpha_max),
        "verification": check_h,
    }
    print(f"flow h:   reused R1d/R1g state, |R|/|R0| {check_h['residual_relative']:.1e}")

    # -- the fine flow: R1f's cache if it proves to be this problem's ---------
    material = za.build_material(spec, alpha_max)
    alpha_f = materials.brinkman_penalty(jnp.asarray(s_f), material)
    pv_f, fine = None, {}
    cache = out / R1F_CACHE
    if cache.is_file():
        cached = np.asarray(np.load(cache)["press_vel"])
        fine["cache"] = {"file": f"results/{R1F_CACHE}", "sha256": sha256_file(cache),
                         "stored_arrays": ["press_vel"], "stored_identity": None}
        try:
            check = fs.verify_flow_state(f2, cached, s_f, alpha_max)
            psi_cached = float(f2.flow.dissipated_power(jnp.asarray(cached), alpha_f))
            psi_rel = psi_cached / R1E_PSI_FINE - 1.0
            fine["cache"].update(verification=check, psi=psi_cached,
                                 psi_vs_r1e_fine_relative=psi_rel)
            if abs(psi_rel) > PSI_SAME_STATE_RTOL:
                raise ValueError(f"Psi {psi_cached!r} is not R1e's {R1E_PSI_FINE!r}")
            pv_f = jnp.asarray(cached)
            fine["source"] = "R1f cache, verified here against this problem; not re-solved"
            fine["verification"] = check
        except (ValueError, r1.NotConverged) as exc:
            fine["cache"]["rejected"] = str(exc)
            print(f"fine flow cache rejected: {exc}")
    if pv_f is None:
        pv_f, t_solve = timed(tf_solver.modified_newton_raphson_solve,
                              f2.flow, f2.flow_x0, alpha_f)
        fine["source"] = "solved here, once"
        fine["t_flow_solve_first_s"] = t_solve
        fine["verification"] = fs.verify_flow_state(f2, pv_f, s_f, alpha_max)
    fine["identity"] = fs.flow_state_identity(f2, s_f, alpha_max)
    fs.save_flow_state(
        out / FINE_FLOW, pv_f, s_f, fine["identity"],
        {"stage": "R1h", "source": fine["source"],
         "verification": fine["verification"],
         "density": "R1d solid_fraction copied from the parent element, h -> h/2",
         "script": "scripts/zhao2d_flow_mesh_check.py"},
    )
    # read back through the identity check: the saved file must be reusable as is
    pv_back, back = fs.load_flow_state(out / FINE_FLOW, f2, s_f, alpha_max)
    if not np.array_equal(np.asarray(pv_back), np.asarray(pv_f)):
        sys.exit("the saved fine flow does not read back bit for bit; stopping")
    fine["saved"] = {"file": f"results/{FINE_FLOW}", "sha256": sha256_file(out / FINE_FLOW),
                     "reloaded_through_identity_check": True,
                     "reload_residual_relative": back["verification"]["residual_relative"]}
    flows["h2"] = fine
    print(f"flow h/2: {fine['source']}; |R|/|R0| "
          f"{fine['verification']['residual_relative']:.1e}"
          + (f", Psi vs R1e {fine['cache']['psi_vs_r1e_fine_relative']:+.1e}"
             if "psi_vs_r1e_fine_relative" in fine.get("cache", {}) else ""))
    memory["after_flows"] = peak_working_set_mb()

    # -- inlet traces --------------------------------------------------------
    trace_h = fs.inlet_trace(p2.flow_mesh, spec, pv_h)
    trace_h2 = fs.inlet_trace(f2.flow_mesh, spec, pv_f)
    inlet = {
        "flow_h": {k: v for k, v in trace_h.items() if not k.startswith("_")},
        "flow_h2": {k: v for k, v in trace_h2.items() if not k.startswith("_")},
        "max_difference_between_traces": fs.compare_inlet_traces(trace_h, trace_h2),
        "thermal_inlet_velocity_max_deviation": {
            key: thermal_inlet_velocity(problems[key], pv_h if CELLS[key][0] == 1 else pv_f,
                                        spec)
            for key in CELLS
        },
    }
    record["flows"], record["inlet"] = flows, inlet
    print(f"inlet: {trace_h['nodes']} / {trace_h2['nodes']} nodes, inflow/nominal "
          f"{trace_h['inflow_over_nominal']:.15f} / {trace_h2['inflow_over_nominal']:.15f}, "
          f"traces differ by {inlet['max_difference_between_traces']:.1e}; wall slip next "
          f"to the rim {trace_h['wall_slip_next_to_rim'][0]['slip_length']:.1e} / "
          f"{trace_h2['wall_slip_next_to_rim'][0]['slip_length']:.1e}\n")

    # -- temperatures ----------------------------------------------------------
    temperatures, cost = {}, {}
    for key in ("flow_h__thermal_h2", "flow_h__thermal_h4"):
        level = R1G_LEVEL[key]
        coords = np.asarray(problems[key].thermal_mesh.mesh.nodes.coords)
        if not np.array_equal(coords, r1g[f"thermal_node_coords_level{level}"]):
            sys.exit(f"R1g level {level} was saved on a different thermal mesh; stopping")
        temperatures[key] = jnp.asarray(r1g[f"temperature_level{level}"])
        cost[key] = {"thermal": f"reused R1g level {level}; not re-solved"}
    for key in ("flow_h2__thermal_h2", "flow_h2__thermal_h4"):
        problem = problems[key]
        temp, t_first = timed(problem.solve_thermal, pv_f, jnp.asarray(s_f), alpha_max)
        again, t_repeat = timed(problem.solve_thermal, pv_f, jnp.asarray(s_f), alpha_max)
        if not np.array_equal(np.asarray(temp), np.asarray(again)):
            sys.exit(f"{key}: the repeated thermal solve differs; stopping")
        temperatures[key] = temp
        cost[key] = {"t_thermal_first_s": t_first, "t_thermal_repeat_s": t_repeat,
                     "repeat_bitwise_equal": True}
        print(f"solved {key}: thermal {t_first:.1f} s first, {t_repeat:.1f} s repeat",
              flush=True)
    memory["after_thermal_solves"] = peak_working_set_mb()

    # -- the four cells, reported the same way ---------------------------------
    cells, flat = {}, {}
    for key, (rf, _) in CELLS.items():
        problem = problems[key]
        s_flow, pv = (s_d, pv_h) if rf == 1 else (s_f, pv_f)
        t0 = time.perf_counter()
        rep = fs.cell_report(problem, s_flow, pv, temperatures[key], alpha_max, scale)
        rep["gate"] = {k: v <= 1e-8 for k, v in rep["residual_relative"].items()}
        if not all(rep["gate"].values()):
            sys.exit(f"{key}: a state fails the convergence gate "
                     f"{rep['residual_relative']}; stopping")
        rep["thermal_inlet"] = fs.thermal_inlet(problem.thermal_mesh, spec, temperatures[key])
        rep["state"] = {
            "flow": "h (R1d/R1g)" if rf == 1 else "h/2 (results/" + FINE_FLOW + ")",
            "temperature": cost[key].get("thermal") or "solved in this run",
            "temperature_sha256": fs.digest(np.asarray(temperatures[key])),
        }
        rep["cost"] = {**cost[key], "t_build_s": build_s[key],
                       "t_report_s": time.perf_counter() - t0}
        cells[key] = rep
        flat[key] = summary(rep)
        print(f"reported {key} ({rep['cost']['t_report_s']:.1f} s)", flush=True)
    memory["after_reports"] = peak_working_set_mb()
    record["cells"] = cells
    record["summary"] = flat

    # -- anchors -------------------------------------------------------------
    lv = r1g_meta["levels"]
    anchors = {
        "flow_h__thermal_h2_vs_r1g_level2": flat["flow_h__thermal_h2"]["compliance"]
        / lv["2"]["compliance"] - 1.0,
        "flow_h__thermal_h2_vs_r1f_C": flat["flow_h__thermal_h2"]["compliance"] / R1F_C - 1.0,
        "flow_h__thermal_h4_vs_r1g_level4": flat["flow_h__thermal_h4"]["compliance"]
        / lv["4"]["compliance"] - 1.0,
        "flow_h2__thermal_h2_vs_r1f_D": flat["flow_h2__thermal_h2"]["compliance"] / R1F_D - 1.0,
        "psi_flow_h_vs_r1g": flat["flow_h__thermal_h2"]["psi"] / r1g_meta["flow"]["psi"] - 1.0,
        "psi_flow_h2_vs_r1e_fine": flat["flow_h2__thermal_h2"]["psi"] / R1E_PSI_FINE - 1.0,
    }
    record["anchors"] = anchors

    # -- the comparisons the stage exists for -----------------------------------
    comparisons = {
        "flow_replacement_at_thermal_h2": difference(flat["flow_h__thermal_h2"],
                                                     flat["flow_h2__thermal_h2"]),
        "flow_replacement_at_thermal_h4": difference(flat["flow_h__thermal_h4"],
                                                     flat["flow_h2__thermal_h4"]),
        "thermal_refinement_on_flow_h": difference(flat["flow_h__thermal_h2"],
                                                   flat["flow_h__thermal_h4"]),
        "thermal_refinement_on_flow_h2": difference(flat["flow_h2__thermal_h2"],
                                                    flat["flow_h2__thermal_h4"]),
    }
    comparisons["interaction"] = {
        k: (comparisons["thermal_refinement_on_flow_h2"][k]["abs"]
            - comparisons["thermal_refinement_on_flow_h"][k]["abs"])
        for k in DIFF_KEYS
    }
    comparisons["note"] = (
        "Each difference is the COMPLETE response to the one mesh change named, with "
        "everything else held: a flow replacement includes every change in the "
        "velocity field, not only its divergence. interaction = (h/2->h/4 on flow "
        "h/2) - (h/2->h/4 on flow h) = (flow replacement at h/4) - (at h/2)."
    )
    record["comparisons"] = comparisons

    record["cost"] = {
        "build_s": build_s,
        "thermal_solves": {k: v for k, v in cost.items()},
        "fine_flow": ({"t_flow_solve_first_s": fine["t_flow_solve_first_s"]}
                      if "t_flow_solve_first_s" in fine else
                      "not solved in this run: the verified R1f cache was reused"),
        "peak_working_set_mb_cumulative": memory,
        "peak_working_set_note": "this process's peak so far, cumulative across every "
                                 "step before it; not a per-model figure",
        "wall_clock_total_s": time.perf_counter() - t_start,
    }

    # -- save, before anything else can fail ----------------------------------
    path = out / "zhao2d_r1h_matrix.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    fields = out / "zhao2d_r1h_fields.npz"
    np.savez_compressed(
        fields,
        temperature_flow_h2_thermal_h2=np.asarray(temperatures["flow_h2__thermal_h2"]),
        temperature_flow_h2_thermal_h4=np.asarray(temperatures["flow_h2__thermal_h4"]),
        thermal_node_coords_h2=np.asarray(f2.thermal_mesh.mesh.nodes.coords),
        thermal_node_coords_h4=np.asarray(f4.thermal_mesh.mesh.nodes.coords),
        flow_node_coords_h2=np.asarray(f2.flow_mesh.mesh.nodes.coords),
        solid_fraction_flow_h2=s_f,
    )

    # -- print ---------------------------------------------------------------
    print("\nanchors (relative difference):")
    for k, v in anchors.items():
        print(f"  {k:36s} {v:+.3e}")

    cols = list(CELLS)
    print("\n" + " " * 26 + "".join(f"{c:>22s}" for c in cols))
    rows = [("C", "compliance", "{:22.6f}"), ("c_advective", "c_advective", "{:22.4f}"),
            ("c_diffusive", "c_diffusive", "{:22.4f}"), ("L_Q", "l_q", "{:22.4f}"),
            ("D_SUPG - F_SUPG", "net_supg", "{:22.4f}"),
            ("(D - F)/C", "net_supg_over_c", "{:22.4%}"),
            ("T_max", "t_max", "{:22.6f}"), ("T_min", "min_temperature", "{:22.6f}"),
            ("nodes below inlet", "nodes_below_inlet", "{:22d}"),
            ("Psi", "psi", "{:22.12f}"), ("J* (single-mesh scale)", "j_star", "{:22.6f}"),
            ("D_T / Q", "D_T_over_heat_in", "{:22.4%}"),
            ("-1/2 D_T2 / C", "minus_half_D_T2_over_C", "{:22.4%}"),
            ("int (div u)^2, flow mesh", "integral_div_u_squared", "{:22.10f}"),
            ("int b_f u.grad T - Q", "bracket", "{:22.6f}"),
            ("Dirichlet reaction", "dirichlet_reaction", "{:22.6f}"),
            ("|R|/|R0| flow", "residual_flow", "{:22.1e}"),
            ("|R|/|R0| thermal", "residual_thermal", "{:22.1e}")]
    for label, key, fmt in rows:
        print(f"  {label:24s}" + "".join(fmt.format(flat[c][key]) for c in cols))

    print("\nmesh changes, complete response (relative):")
    for name in ("flow_replacement_at_thermal_h2", "flow_replacement_at_thermal_h4",
                 "thermal_refinement_on_flow_h", "thermal_refinement_on_flow_h2"):
        d = comparisons[name]
        print(f"  {name:32s} C {d['compliance']['abs']:+10.3f} ({d['compliance']['rel']:+.3%})  "
              f"c_adv {d['c_advective']['rel']:+.2%}  c_diff {d['c_diffusive']['rel']:+.2%}  "
              f"T_max {d['t_max']['rel']:+.2%}  J* {d['j_star']['rel']:+.2%}")
    print(f"  interaction in C: {comparisons['interaction']['compliance']:+.3f}")

    print(f"\nwrote {path}\nwrote {fields} (the two new temperatures)\n"
          f"wrote {out / FINE_FLOW} (the fine flow, with its identity)\n"
          f"total {record['cost']['wall_clock_total_s']:.0f} s")


if __name__ == "__main__":
    main()
