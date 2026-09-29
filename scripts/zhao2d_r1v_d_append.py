"""R1v, part 2: up to twenty updates appended to R1t's run on D, MMA's history carried over.

The contract proposed in the review of 7786ce7, its second part, authorised by
the user after part 1. It starts from R1t's raw continuous terminal design and
MMA's state after R1t's twenty updates, as part 1 re-signed it
(`results/zhao2d_r1v_mma_migrated.npz`). The budget is appended from 20 to at
most 40 (`MMACheckpoint.append_budget`), and the run resumes: MMA's epoch,
asymptotes and design history are carried over, not reinitialised. For the
first time a run here continues rather than restarts.

Everything else is R1t's: D (design h, flow h/2, thermal h/8), the linear
thermal path, the volume-preserving projection at beta = 32, alpha_max = 1e7,
q = 0.2 for both, the filter at 2e-4, the common scale with w = 0.5, and the
move limit 0.1.

Checked in stages. A failed gate writes the record and exits non-zero before
the next build, update or solve:

* inputs: part 1's record passed every stage, and its checkpoint is the one it
  describes and holds R1t's raw terminal design; R1t's terminal and binary
  design, R1u's filled design and the older candidates' D values are the
  recorded ones;
* the model and the scale: R1m's check layer, 5000 variables, the thermal path
  linear, R1q's scale;
* the append: R1t's twenty steps kept, twenty appended, the budget 40; MMA's
  state changed in the budget's flag only, as part 1 checked it dry;
* the first resumed evaluation's map, nothing solved: at R1t's raw terminal
  design, eta and g as R1t's terminal recorded them, the root non-degenerate;
* at most twenty updates. The run checks its own binding before anything is
  solved. The first resumed evaluation is then checked against R1t's
  continuous terminal -- Psi to 1e-12 relative, g and eta to 1e-12 absolute,
  C and J to 1e-8 relative, the gates R1t's zero step had, neither looser nor
  tighter -- before the next update is made. Every state is gated at 1e-8, and
  the terminal is evaluated at the same point;
* the export by R1l's rule if the terminal is feasible, with every fluid
  component by shared edges recorded. An infeasible terminal, or an export
  that does not qualify or is not connected, ends the run there: a result,
  not a failure. An isolated component is reported; no cell is filled and no
  volume made up;
* the new binary design on D, once: one flow and one thermal solve on R1m's
  check-layer route, the temperature by one linear solve, each gated -- unless
  its geometry is a candidate's, whose recorded values are then reused.

The new qualified binary design is compared, on D and at the common scale,
with the saved values of R1t's, R1u's (R1t's, filled), R1r's, R1o's, R1n's and
the R1l pilot's designs; none is re-solved or deleted. No improvement closes
the stage normally.

    python scripts/zhao2d_r1v_d_append.py [--inputs results] [--out DIR] [--overwrite]
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
import hashlib
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

import toflux.src.mma as tf_mma  # noqa: E402
import toflux.src.solver as tf_solver  # noqa: E402

from tfopus import affine_solve as affine  # noqa: E402
from tfopus import materials  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_binary as zb  # noqa: E402
from tfopus import zhao2d_driver as drv  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_fineflow as ff  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402
from tfopus import zhao2d_refine as ref  # noqa: E402
from tfopus.mesh import Face  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb, timed  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file  # noqa: E402
# R1m's report, so the new design on D is summarised exactly as the candidates were
from zhao2d_r1m_flow_check import flat, mesh_identity, thermal_identity  # noqa: E402

ALPHA_MAX, BETA = 1.0e7, 32.0
FLOW_REFINEMENT, THERMAL_REFINEMENT, QUADRATURE = 2, 4, 3
THERMAL_PATH = "linear"
R1T_UPDATES, APPENDED = 20, 20
BUDGET = R1T_UPDATES + APPENDED
R1T_PHASE, APPENDED_PHASE = "r1t-d-vp32-linear", "r1v-d-vp32-linear"
MOVE_LIMIT, NEAR_LIMIT = 0.1, 0.0999
GATE, ANCHOR_TOL, ACROSS_PATHS_TOL = 1e-8, 1e-12, 1e-8
RECORD = "zhao2d_r1v.json"
FIELDS = "zhao2d_r1v_fields.npz"
PROGRESS = "zhao2d_r1v_progress.json"
STACKS = "zhao2d_r1v_stacks.log"
INPUTS = ("zhao2d_r1l_vp_pilot.json", "zhao2d_r1l_vp_pilot_fields.npz",
          "zhao2d_r1m_flow_check.json", "zhao2d_r1n_beta32.json", "zhao2d_r1n_fields.npz",
          "zhao2d_r1o.json", "zhao2d_r1o_fields.npz", "zhao2d_r1q_fineflow_check.json",
          "zhao2d_r1r.json", "zhao2d_r1r_fields.npz", "zhao2d_r1s.json",
          "zhao2d_r1t.json", "zhao2d_r1t_fields.npz", "zhao2d_r1u.json", "zhao2d_r1u_fields.npz",
          "zhao2d_r1v_migration.json", "zhao2d_r1v_mma_migrated.npz")
KEYS = ("J_common_scale", "psi", "compliance", "constraint_g", "v_f_design_domain",
        "grey_fraction", "projection_eta", "projection_root_slope",
        "flow_residual_relative", "thermal_residual_relative", "thermal_path")
NAMES = ("r1t", "r1u", "r1r", "r1o", "r1n", "pilot")
COMPARED = ("J", "J_star", "psi", "compliance")


class ResumeMismatch(RuntimeError):
    """The first resumed evaluation does not reproduce R1t's continuous terminal."""


def rel(b, a) -> float:
    return b / a - 1.0


def load_json(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def step_norms(designs: np.ndarray, first_update: int) -> list[dict]:
    """Each update's design change, numbered as the run numbers its updates."""
    out = []
    for k, d in enumerate(np.diff(np.asarray(designs), axis=0), start=first_update):
        out.append({"update": k, "l2": float(np.linalg.norm(d)), "linf": float(np.abs(d).max()),
                    "rms": float(np.sqrt(np.mean(d * d)))})
    return out


def resume_anchor(first: dict, terminal: dict) -> dict:
    """The first resumed evaluation against R1t's continuous terminal.

    The same x, model and thermal path; the terminal was a value only, this
    evaluation also takes the gradient. R1t's zero step's gates, neither
    looser nor tighter: Psi to 1e-12 relative, g and eta to 1e-12 absolute,
    C and J to 1e-8 relative.
    """
    out = {"source": "R1t's continuous terminal record (linear thermal path, value only)",
           "psi_relative": rel(first["psi"], terminal["psi"]),
           "compliance_relative": rel(first["compliance"], terminal["compliance"]),
           "J_common_scale_relative": rel(first["J_common_scale"], terminal["J_common_scale"]),
           "constraint_g_difference": first["constraint_g"] - terminal["constraint_g"],
           "eta_difference": first["projection_eta"] - terminal["projection_eta"],
           "v_f_design_domain_difference": first["v_f_design_domain"] - terminal["v_f_design_domain"],
           "tolerances": {"psi_relative": ANCHOR_TOL, "constraint_g_absolute": ANCHOR_TOL,
                          "eta_absolute": ANCHOR_TOL, "C_and_J_relative": ACROSS_PATHS_TOL}}
    out["reproduced"] = bool(abs(out["psi_relative"]) <= ANCHOR_TOL
                             and abs(out["constraint_g_difference"]) <= ANCHOR_TOL
                             and abs(out["eta_difference"]) <= ANCHOR_TOL
                             and abs(out["compliance_relative"]) <= ACROSS_PATHS_TOL
                             and abs(out["J_common_scale_relative"]) <= ACROSS_PATHS_TOL)
    return out


def fluid_components(planar, s_binary) -> list[dict]:
    """Every fluid component by shared edges -- the export rule's criterion --
    the largest first, with whether it touches the inlet or the outlet."""
    fluid = np.asarray(s_binary) < 0.5
    num, labels = csgraph.connected_components(z.element_adjacency(planar)[fluid][:, fluid],
                                               directed=False)
    cells = np.nonzero(fluid)[0]
    inlet = set(zb.port_elements(planar, Face.INLET).tolist())
    outlet = set(zb.port_elements(planar, Face.OUTLET).tolist())
    centres = np.asarray(planar.elem_centres)
    out = []
    for k in range(num):
        members = cells[labels == k]
        found = set(members.tolist())
        out.append({"size": int(members.size), "touches_inlet": bool(inlet & found),
                    "touches_outlet": bool(outlet & found), "cells": members.tolist(),
                    "centres_mm": (centres[members] * 1e3).tolist()})
    out.sort(key=lambda c: -c["size"])
    return out


def component_summary(components: list[dict]) -> dict:
    isolated = [c for c in components if not (c["touches_inlet"] or c["touches_outlet"])]
    return {"sizes": [c["size"] for c in components], "isolated": isolated,
            "single_component": len(components) == 1}


def lowest_by_weight(points: dict) -> list[dict]:
    """Which design has the lowest J(w) = w Psi/Psi_0 + (1 - w) C/C_0, w in [0, 1].

    `points` maps a name to (Psi/Psi_0, C/C_0) on the common scale, so w = 0.5
    is the J the run minimises. A fixed set of designs reweighted: not a Pareto
    front that was optimised."""
    names = list(points)
    cuts = {0.0, 1.0}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (pa, ca), (pb, cb) = points[a], points[b]
            den = (pa - ca) - (pb - cb)
            if den != 0.0 and 0.0 < (cb - ca) / den < 1.0:
                cuts.add((cb - ca) / den)
    cuts = sorted(cuts)
    out = []
    for lo, hi in zip(cuts, cuts[1:]):
        mid = 0.5 * (lo + hi)
        best = min(names, key=lambda n: mid * points[n][0] + (1.0 - mid) * points[n][1])
        if out and out[-1]["lowest"] == best:
            out[-1]["to"] = hi
        else:
            out.append({"from": lo, "to": hi, "lowest": best})
    return out


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
    faulthandler.dump_traceback_later(7200, repeat=True, file=stacks)  # longer than a healthy run
    t_start = time.perf_counter()

    spec = z.Zhao2DSpec()
    config = r1.R1Config(projection=r1.Projection.VOLUME_PRESERVING)
    w = config.weight
    material = za.build_material(spec, ALPHA_MAX)
    single = r1.load_reference(spec, config)  # J* only
    p_rec = load_json(inputs / "zhao2d_r1l_vp_pilot.json")
    m_rec = load_json(inputs / "zhao2d_r1m_flow_check.json")
    n_rec = load_json(inputs / "zhao2d_r1n_beta32.json")
    o_rec = load_json(inputs / "zhao2d_r1o.json")
    q_rec = load_json(inputs / "zhao2d_r1q_fineflow_check.json")
    r_rec = load_json(inputs / "zhao2d_r1r.json")
    s_rec = load_json(inputs / "zhao2d_r1s.json")
    t_rec = load_json(inputs / "zhao2d_r1t.json")
    u_rec = load_json(inputs / "zhao2d_r1u.json")
    v_rec = load_json(inputs / "zhao2d_r1v_migration.json")
    fp = np.load(inputs / "zhao2d_r1l_vp_pilot_fields.npz")
    fn = np.load(inputs / "zhao2d_r1n_fields.npz")
    fo = np.load(inputs / "zhao2d_r1o_fields.npz")
    fr = np.load(inputs / "zhao2d_r1r_fields.npz")
    ft = np.load(inputs / "zhao2d_r1t_fields.npz")
    fu = np.load(inputs / "zhao2d_r1u_fields.npz")
    with np.load(inputs / "zhao2d_r1v_mma_migrated.npz") as f:
        migrated = drv.MMACheckpoint.from_npz(f)
        legacy_binding = str(f["legacy_binding"])
    x_start = np.asarray(ft["design"], dtype=np.float64)  # R1t's raw terminal design
    t_terminal = t_rec["terminal"]
    rows = m_rec["rows"]
    # the candidates on D, as recorded
    base = {"r1t": t_rec["cells"]["new/check"], "r1u": u_rec["cells"]["filled/check"],
            "r1r": r_rec["cells"]["new/check"], "r1o": o_rec["cells"]["new/check"],
            "r1n": n_rec["cells"]["new/check"], "pilot": m_rec["cells"]["pilot/flow_h2"]}
    binaries = {"r1t": ft["solid_fraction_binary"], "r1u": fu["solid_fraction_filled"],
                "r1r": fr["solid_fraction_binary"], "r1o": fo["solid_fraction_binary"],
                "r1n": fn["solid_fraction_binary"], "pilot": fp["solid_fraction_binary"]}
    binary_sha = {"r1t": t_rec["export"]["binary_sha256"], "r1u": u_rec["geometry"]["sha256"],
                  "r1r": r_rec["export"]["binary_sha256"], "r1o": o_rec["export"]["binary_sha256"],
                  "r1n": n_rec["export"]["binary_sha256"], "pilot": p_rec["export"]["binary_sha256"]}

    record = {
        "note": ("R1v, part 2: up to 20 MMA updates appended to R1t's run on the check layer D "
                 "(design h, flow h/2, thermal h/8), from R1t's raw terminal design with MMA's "
                 "state after its 20 updates carried over (as part 1 re-signed it), the budget "
                 "appended from 20 to 40; the linear thermal path, beta 32, J on the development "
                 "model's Psi_0 and C_0 as a declared common scale. The terminal's qualified "
                 "binary design is evaluated once on D and compared with R1t's, R1u's, R1r's, "
                 "R1o's, R1n's and the R1l pilot's saved D values. Analysis on one model; a real "
                 "continuation, not a restart; nothing here is a convergence claim."),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in INPUTS},
        "contract": {"model": {"design": "h, 5000 raw variables", "flow_refinement": FLOW_REFINEMENT,
                               "thermal_refinement": THERMAL_REFINEMENT,
                               "thermal_quadrature": QUADRATURE, "thermal_path": THERMAL_PATH},
                     "projection": config.projection, "beta": BETA, "alpha_max": ALPHA_MAX,
                     "updates_appended_at_most": APPENDED, "budget": BUDGET,
                     "move_limit": MOVE_LIMIT, "gate": GATE, "weight": w,
                     "continuation": "none", "internal_restarts": "none",
                     "mma_reinitialised": False,
                     "resumed_from": "results/zhao2d_r1v_mma_migrated.npz (part 1)",
                     "fingerprint": config.fingerprint(),
                     "resume_anchor": {"source": "results/zhao2d_r1t.json, terminal",
                                       "J_common_scale": t_terminal["J_common_scale"],
                                       "tolerances": {"psi_relative": ANCHOR_TOL,
                                                      "g_absolute": ANCHOR_TOL,
                                                      "eta_absolute": ANCHOR_TOL,
                                                      "C_and_J_relative": ACROSS_PATHS_TOL}},
                     "binary_solves_at_most": {"flow": 1, "thermal": 1, "layer": "D"},
                     "isolated_components": "reported; no cell filled, no volume made up",
                     "near_limit_band": f"largest single change >= {NEAR_LIMIT}",
                     "authorised": ("by the user, part 2, on the local CPU, after part 1, as the "
                                    "review of 7786ce7 states it")},
        "initial_design": {"file": "results/zhao2d_r1t_fields.npz", "key": "design",
                           "sha256_float64": fs.digest(x_start),
                           "also": "the migrated checkpoint's x, bit for bit"},
    }
    for script in (pathlib.Path(__file__).resolve(), REPO / "scripts" / "zhao2d_r1m_flow_check.py"):
        record["provenance"]["source_sha256"][script.relative_to(REPO).as_posix()] = sha256_file(script)
    failures, checkpoints, cost, cells, saved, progress = [], [], {}, {}, {}, []

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
        if not failures:
            (out / PROGRESS).unlink(missing_ok=True)

    def checkpoint(stage: str) -> None:
        """Stop here, with the record written, if anything so far has failed."""
        checkpoints.append({"stage": stage, "failures": list(failures)})
        if failures:
            record["stopped"] = f"{stage}: checks failed; nothing after it was built, updated or solved"
            finish()
            sys.exit(f"CHECKS FAILED at {stage}; nothing after it was built, updated or solved: "
                     f"{failures}")
        print(f"checkpoint passed: {stage}", flush=True)

    def end(reason: str) -> None:
        """An outcome the contract stops at: recorded, and not a failure."""
        record["stopped"] = reason
        finish()
        print(f"\nstopped, as the contract says: {reason}\nwrote {out / RECORD}")
        sys.exit(0)

    def common_j(psi: float, c: float) -> float:
        return w * psi / scale.psi_0 + (1.0 - w) * c / scale.c_0

    def solve_new(problem, s, yard: dict) -> None:
        """The new binary design on D: one flow solve and one linear thermal solve, each gated."""
        key = "new/check"
        s = jnp.asarray(s)
        fluid = z.fluid_fractions(problem.flow_mesh, s)
        cell = {"design": "new", "layer": "check", "state": "solved here", "fluid_fractions": fluid,
                "thermal_path": THERMAL_PATH,
                "volume_feasible": zb.volume_feasible(fluid["v_f_design_domain"],
                                                      config.max_fluid_fraction)}
        cells[key] = cell
        alpha = materials.brinkman_penalty(s, material)
        pv, t_flow = timed(tf_solver.modified_newton_raphson_solve, problem.flow,
                           problem.flow_x0, alpha)
        cell["t_flow_solve_s"] = t_flow
        saved["new_press_vel_check"] = np.asarray(pv)
        try:
            cell["flow_verification"] = fs.verify_flow_state(problem, pv, s, ALPHA_MAX, GATE)
        except (ValueError, r1.NotConverged) as exc:
            require(False, f"{key}: the flow solve fails its gate: {exc}")
        checkpoint(f"check layer: the new design's flow solve ({t_flow:.1f} s)")
        cell["flow_identity"] = fs.flow_state_identity(problem, s, ALPHA_MAX)
        temp, t_thermal = timed(
            lambda: jax.block_until_ready(affine.affine_solve(
                problem.thermal, problem.thermal_x0, problem.thermal_velocity(pv),
                problem.thermal_conductivity(s, ALPHA_MAX), problem.q_source)))
        cell["t_thermal_solve_s"] = t_thermal
        saved["new_temperature_check"] = np.asarray(temp)
        t0 = time.perf_counter()
        rep = fs.cell_report(problem, s, pv, temp, ALPHA_MAX, single)
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
        checkpoint(f"check layer: the new design's thermal solve ({t_thermal:.1f} s)")

    # -- inputs ---------------------------------------------------------------------------
    for name, rec_ in (("R1m", m_rec), ("R1n", n_rec), ("R1o", o_rec), ("R1q", q_rec),
                       ("R1r", r_rec), ("R1s", s_rec), ("R1t", t_rec), ("R1u", u_rec),
                       ("R1v part 1", v_rec)):
        require(not any(c["failures"] for c in rec_["checkpoints"]), f"{name}'s record has failed checks")
    # R1l's record predates checkpoints: its checks, and its export's own verdict
    require(not any(p_rec["checks"].values()) and p_rec["export"]["qualified"],
            "the R1l pilot's record has failed checks or an unqualified export")
    require(s_rec["verdict"]["equivalent"], "R1s did not find the two thermal paths equivalent")
    require(t_rec["stop"]["mma_updates"] == R1T_UPDATES and t_terminal.get("terminal") is True
            and t_terminal["iteration"] == R1T_UPDATES,
            "R1t's record does not hold its terminal after its 20 updates")
    require(len(v_rec["checkpoints"]) == 7, "part 1's record does not hold its seven stages")
    require(fs.digest(migrated.state_array) == v_rec["migration"]["unchanged"]["state_array_sha256"]
            == t_rec["mma_checkpoint"]["state_array_sha256"],
            "the migrated checkpoint's state is not R1t's, as part 1 recorded it")
    require(hashlib.sha256(migrated.binding.encode()).hexdigest()
            == v_rec["migration"]["to"]["binding_sha256"],
            "the migrated checkpoint's binding is not the one part 1 signed")
    require(hashlib.sha256(legacy_binding.encode()).hexdigest()
            == v_rec["legacy"]["binding_sha256"], "the legacy binding beside it is not R1t's")
    require((migrated.kktnorm, migrated.updates_done, migrated.num_design_var)
            == (t_rec["mma_checkpoint"]["kktnorm"], R1T_UPDATES, 5000),
            "the migrated checkpoint is not after R1t's 20 updates of 5000 variables")
    require(np.array_equal(migrated.state_array[:5000], x_start),
            "the migrated checkpoint's design is not R1t's raw terminal design")
    for name in NAMES:
        require(fs.digest(binaries[name]) == binary_sha[name],
                f"{name}: the saved binary design is not the one its record exported")
        require(zb.cell_usable(base[name]) and base[name]["volume_feasible"],
                f"{name}: its D cell is not a usable, gated, feasible state in its record")
    checkpoint("inputs")

    # -- the model and the scale ---------------------------------------------------------------
    t0 = time.perf_counter()
    problem = ff.Zhao2DFineFlowProblem(spec, config, FLOW_REFINEMENT, THERMAL_REFINEMENT, QUADRATURE,
                                       thermal_path=THERMAL_PATH)
    cost["build_s"] = time.perf_counter() - t0
    print(f"built D, thermal path {problem.thermal_path}: design {problem.design_mesh.num_elems} el "
          f"({problem.num_design} variables), flow {problem.flow_mesh.num_elems} el, thermal "
          f"{problem.thermal_mesh.num_elems} el ({cost['build_s']:.1f} s)", flush=True)
    record["identities"] = {"flow_mesh": mesh_identity(problem.flow_mesh),
                            "thermal_mesh": thermal_identity(problem),
                            "design_mesh": mesh_identity(problem.design_mesh),
                            "num_design": problem.num_design, "thermal_path": problem.thermal_path,
                            "model_identity": json.loads(problem.model_identity()),
                            "nesting": problem.nesting}
    require(record["identities"]["flow_mesh"] == rows["flow_h2"]["flow_mesh"],
            "the flow side is not R1m's check layer")
    require(record["identities"]["thermal_mesh"] == rows["flow_h2"]["thermal_mesh"],
            "the thermal side is not R1m's check layer")
    require(record["identities"]["design_mesh"] == rows["flow_h"]["flow_mesh"],
            "the design mesh is not the development model's h mesh")
    require(record["identities"]["model_identity"] == t_rec["identities"]["model_identity"],
            "the model is not R1t's")
    require(problem.num_design == 5000, f"{problem.num_design} design variables, not 5000")
    require(problem.thermal_path == THERMAL_PATH, f"the thermal path is {problem.thermal_path}")
    try:
        scale = ff.common_scale(problem, dual.reference_file(4, QUADRATURE), 4, QUADRATURE)
    except ValueError as exc:
        require(False, f"the common scale: {exc}")
        checkpoint("the model and the common scale")
    record["scale"] = {"psi_0": scale.psi_0, "c_0": scale.c_0, "source_file": scale.source_file,
                       "source_sha256": scale.source_sha256, "source_model": scale.source_model,
                       "declared": "the development model's constants, used as a common scale; "
                                   "not a reference of this model"}
    require(all(value == q_rec["scale"][k] for k, value in
                (("psi_0", scale.psi_0), ("c_0", scale.c_0), ("source_sha256", scale.source_sha256))),
            "the scale is not the one R1q, R1r and R1t used")
    for name in NAMES:
        require(base[name]["J"] == common_j(base[name]["psi"], base[name]["compliance"]),
                f"{name}: its recorded D value of J is not on this scale")
    yard = {"psi_0": scale.psi_0, "c_0": scale.c_0, "file": "tfopus/zhao2d_reference_dual_r4q3_v1.json",
            "file_sha256": scale.source_sha256, "declared": "the common scale"}
    checkpoint("the model and the common scale")

    # -- the append -----------------------------------------------------------------------------
    phases = [drv.Phase(R1T_PHASE, R1T_UPDATES, beta=BETA, alpha_max=ALPHA_MAX,
                        note="R1t's twenty updates, already made; kept as scheduled"),
              drv.Phase(APPENDED_PHASE, APPENDED, beta=BETA, alpha_max=ALPHA_MAX,
                        note="R1v: appended, MMA's history carried over")]
    try:
        appended = migrated.append_budget(phases, BUDGET)
    except ValueError as exc:
        require(False, f"the append was refused: {exc}")
        checkpoint("the append")
    old, grown = json.loads(migrated.binding), json.loads(appended.binding)
    changed = np.flatnonzero(appended.state_array != migrated.state_array)
    before = tf_mma.MMAState.from_array(migrated.state_array.copy(), 5000)
    after = tf_mma.MMAState.from_array(appended.state_array.copy(), 5000)
    require(changed.size == 1 and before.is_converged and not after.is_converged
            and after.epoch == before.epoch == R1T_UPDATES,
            "the append changed more of MMA's state than the budget's flag")
    require(grown["schedule"][:R1T_UPDATES] == old["schedule"] and len(grown["schedule"]) == BUDGET
            and grown["budget"] == grown["mma"]["max_iter"] == BUDGET,
            "the append did not keep R1t's twenty steps and add twenty")
    appended_sha = hashlib.sha256(appended.binding.encode()).hexdigest()
    require(appended_sha == v_rec["dry_part_2"]["append"]["binding_sha256"],
            "the append is not the one part 1 checked dry")
    record["append"] = {"from_binding_sha256": v_rec["migration"]["to"]["binding_sha256"],
                        "to_binding_sha256": appended_sha, "state_entries_changed": changed.tolist(),
                        "is_converged": [bool(before.is_converged), bool(after.is_converged)],
                        "epoch": int(after.epoch), "kktnorm": appended.kktnorm,
                        "budget": [old["budget"], grown["budget"]],
                        "schedule_steps": [len(old["schedule"]), len(grown["schedule"])]}
    checkpoint("the append")

    # -- the first resumed evaluation's map, nothing solved ----------------------------------------
    xj = jnp.asarray(x_start)
    root = problem.projection_root(xj, BETA)
    g0 = float(problem.fluid_fraction(xj, BETA)) / config.max_fluid_fraction - 1.0
    record["resume_map"] = {"root": root, "g0": g0,
                            "eta_vs_r1t_terminal": root["eta"] - t_terminal["projection_eta"],
                            "g_vs_r1t_terminal": g0 - t_terminal["constraint_g"]}
    print(f"resume map: eta {root['eta']:.10f} (R1t's terminal {t_terminal['projection_eta']:.10f}), "
          f"root slope {root['slope']:.3e} (non-degenerate {root['nondegenerate']}), g0 {g0:+.6e}",
          flush=True)
    require(root["nondegenerate"], f"the projection root at R1t's terminal is degenerate: {root}")
    require(abs(root["eta"] - t_terminal["projection_eta"]) <= ANCHOR_TOL,
            f"eta {root['eta']} is not R1t's terminal's {t_terminal['projection_eta']}")
    require(abs(g0 - t_terminal["constraint_g"]) <= ANCHOR_TOL,
            f"the constraint {g0} is not R1t's terminal's {t_terminal['constraint_g']}")
    checkpoint("the first resumed evaluation's map")

    # -- at most twenty updates, the first resumed evaluation checked first --------------------------
    def on_iteration(rec) -> None:
        progress.append({k: rec.get(k) for k in (*KEYS, "iteration", "phase", "seconds", "timing_s")})
        print(f"{rec['iteration']:3d}  J {rec['J_common_scale']:.8f}  Psi {rec['psi']:.6e}  "
              f"C {rec['compliance']:.3f}  g {rec['constraint_g']:+.3e}  grey "
              f"{rec['grey_fraction']:.4f}  eta {rec['projection_eta']:.5f}  |R| "
              f"{rec['flow_residual_relative']:.1e}/{rec['thermal_residual_relative']:.1e}  "
              f"{rec['seconds']:.1f} s", flush=True)
        (out / PROGRESS).write_text(json.dumps(progress, indent=1), encoding="utf-8")
        if rec["iteration"] == R1T_UPDATES:
            anchor = resume_anchor(rec, t_terminal)
            record["resume_anchor"] = anchor
            print(f"first resumed evaluation against R1t's terminal: Psi {anchor['psi_relative']:+.1e}, "
                  f"C {anchor['compliance_relative']:+.1e}, J "
                  f"{anchor['J_common_scale_relative']:+.1e}, g "
                  f"{anchor['constraint_g_difference']:+.1e}, eta {anchor['eta_difference']:+.1e}",
                  flush=True)
            if not anchor["reproduced"]:
                raise ResumeMismatch(f"the first resumed evaluation does not reproduce R1t's "
                                     f"terminal: {anchor}")

    t0 = time.perf_counter()
    try:
        result = ff.run(problem, scale, phases, move_limit=MOVE_LIMIT, budget=BUDGET,
                        on_iteration=on_iteration, resume=appended)
    except ResumeMismatch as exc:
        record["history"] = progress
        require(False, str(exc))
        checkpoint("the first resumed evaluation against R1t's terminal")
    except ValueError as exc:
        record["history"] = progress
        require(False, f"the run refused the appended checkpoint ({len(progress)} evaluations "
                       f"made): {exc}")
        checkpoint("the run's own binding check")
    except (r1.NotConverged, r1.DegenerateProjection) as exc:
        partial = getattr(exc, "partial", {})
        record["stop"] = {"stop_reason": type(exc).__name__, "message": str(exc),
                          "failed_iteration": partial.get("failed_iteration")}
        record["history"] = partial.get("history", progress)
        if "failed_design" in partial:
            saved.update(designs=partial["designs"], failed_design=partial["failed_design"])
        if partial.get("mma") is not None:
            saved.update(partial["mma"].to_npz())
        require(False, f"the updates stopped: {type(exc).__name__}: {exc}")
        checkpoint("the updates")
    cost["t_updates_including_terminal_s"] = time.perf_counter() - t0
    checkpoints.append({"stage": "the run's own binding check", "failures": []})
    checkpoints.append({"stage": "the first resumed evaluation against R1t's terminal", "failures": []})

    history, terminal = result.history, result.terminal
    first = history[0]
    evaluated = [*history, terminal]
    for rec_ in evaluated:
        rec_["J_star"] = float(dual.reporting_objective(rec_["psi"], rec_["compliance"], single, w))
        rec_["volume_feasible"] = zb.volume_feasible(rec_["v_f_design_domain"],
                                                     config.max_fluid_fraction)
    feasible = [r_ for r_ in evaluated if r_["volume_feasible"]]
    best = min(feasible, key=lambda r_: r_["J_common_scale"]) if feasible else None
    designs = np.vstack([result.designs, np.asarray(result.design)[None, :]])
    steps = step_norms(designs, first_update=R1T_UPDATES + 1)
    record["history"], record["terminal"], record["steps"] = history, terminal, steps
    record["near_limit_band_updates"] = [s_["update"] for s_ in steps if s_["linf"] >= NEAR_LIMIT]
    record["stop"] = {"stop_reason": result.stop_reason, "proxy_criterion": result.proxy_criterion,
                      "proxy_fired_at_final_stage": result.proxy_fired_at_final_stage,
                      "mma_updates_this_run": len(history),
                      "mma_updates_in_all": result.mma.updates_done, "convergence_verified": False,
                      "first_feasible_iteration": next(
                          (r_["iteration"] for r_ in evaluated if r_["volume_feasible"]), None)}
    r1t_zero = t_rec["history"][0]
    record["responses"] = {
        "optimisation": {
            "from": "the first resumed evaluation (R1t's continuous terminal, evaluated again)",
            "to": "the terminal",
            **{q: rel(terminal[k], first[k]) for q, k in (("J", "J_common_scale"), ("psi", "psi"),
                                                          ("compliance", "compliance"))},
            "g": [first["constraint_g"], terminal["constraint_g"]],
            "grey": [first["grey_fraction"], terminal["grey_fraction"]]},
        "over_r1t_and_r1v": {
            "from": "R1t's zero step (R1r's continuous terminal, on the linear path)",
            "to": "the terminal",
            **{q: rel(terminal[k], r1t_zero[k]) for q, k in (("J", "J_common_scale"), ("psi", "psi"),
                                                             ("compliance", "compliance"))}},
    }
    record["best_feasible_evaluated"] = (
        None if best is None else {"iteration": best["iteration"],
                                   "design_row": best["iteration"] - R1T_UPDATES,
                                   "is_terminal": bool(best.get("terminal")),
                                   "J_common_scale": best["J_common_scale"], "psi": best["psi"],
                                   "compliance": best["compliance"],
                                   "constraint_g": best["constraint_g"]})
    cp = result.mma
    record["mma_checkpoint"] = {
        "updates_done": cp.updates_done, "kktnorm": cp.kktnorm,
        "state_array_sha256": fs.digest(cp.state_array),
        "binding_sha256": hashlib.sha256(cp.binding.encode()).hexdigest(),
        "binding": json.loads(cp.binding),
        "note": ("MMA's state after the last update, under the appended binding; the fields "
                 "file holds it under mma_*")}
    saved.update(initial_design=result.initial_design, designs=designs, design=result.design,
                 initial_solid_fraction=result.initial_solid_fraction,
                 initial_press_vel=result.initial_press_vel,
                 initial_temperature=result.initial_temperature,
                 solid_fraction=result.solid_fraction, press_vel=result.press_vel,
                 temperature=result.temperature,
                 design_elem_centres=np.asarray(problem.design_mesh.elem_centres), **cp.to_npz())
    if best is not None:
        saved["best_feasible_design"] = designs[best["iteration"] - R1T_UPDATES]
    print(f"terminal: J {terminal['J_common_scale']:.8f}  g {terminal['constraint_g']:+.3e}  grey "
          f"{terminal['grey_fraction']:.4f}  stop {result.stop_reason}; last step L2 "
          f"{steps[-1]['l2']:.4f}, Linf {steps[-1]['linf']:.4f}; near-limit band "
          f"{record['near_limit_band_updates']}; MMA state after {cp.updates_done} updates in all",
          flush=True)
    checkpoints.append({"stage": f"the updates ({len(history)}, {result.stop_reason})", "failures": []})

    # -- the export -------------------------------------------------------------------------
    design_mesh = problem.design_mesh
    design_mask = np.asarray(design_mesh.design_mask)
    areas = np.asarray(design_mesh.elem_area)
    export = {"terminal_volume_feasible": bool(terminal["volume_feasible"])}
    record["export"] = export
    if not terminal["volume_feasible"]:
        end("the terminal is not volume-feasible, so nothing is exported")
    s_t = np.asarray(result.solid_fraction)
    out_ = zb.volume_threshold(s_t, design_mask, areas, config.max_fluid_fraction)
    s_bin = out_.pop("s_binary")
    conn = zb.connectivity(design_mesh, s_t, (out_["t"],))[f"{out_['t']:g}"]
    export.update(out_, connected=bool(conn["inlet_outlet_connected"]),
                  fluid_components=conn["fluid_components"],
                  volume_feasible=zb.volume_feasible(out_["fluid_fraction"],
                                                     config.max_fluid_fraction),
                  binary_sha256=fs.digest(s_bin),
                  fluid_cells_at_s_0_5=int(np.sum((s_t < 0.5) & design_mask)))
    export["qualified"] = bool(export["volume_feasible"] and export["connected"])
    export["components"] = component_summary(fluid_components(design_mesh, s_bin))
    export["same_geometry_as"] = [n for n in NAMES if fs.digest(s_bin) == binary_sha[n]]
    export["cells_differing_from"] = {n: int(np.sum(s_bin != np.asarray(binaries[n]))) for n in NAMES}
    saved["solid_fraction_binary"] = s_bin
    print(f"export: t {export['t']:.10f}, {export['fluid_cells']} fluid cells, "
          f"{export['cells_changed_from_0.5']} changed from s = 0.5, connected "
          f"{export['connected']}, components {export['components']['sizes']}, isolated "
          f"{[c['cells'] for c in export['components']['isolated']]}, cells differing from the "
          f"candidates {export['cells_differing_from']}", flush=True)
    if not export["qualified"]:
        end("the export does not qualify or is not connected")

    # -- the new binary design on D, once ------------------------------------------------------
    same = export["same_geometry_as"][0] if export["same_geometry_as"] else None
    s_flow_model = np.asarray(problem.flow_density(jnp.asarray(s_bin)))  # the model's own E_DF copy
    cost["peak_working_set_mb_after_updates"] = peak_working_set_mb()
    del problem
    gc.collect()
    if same:
        cells["new/check"] = {**base[same], "design": "new",
                              "state": f"the same geometry as {same}: its recorded D values reused"}
    else:
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
        s_f = np.asarray(ref.refine_design(design_mesh, fine.flow_mesh, s_bin))
        try:
            record["check_layer"]["transfer"] = ref.check_transfer(design_mesh, fine.flow_mesh, s_bin, s_f)
        except RuntimeError as exc:
            require(False, f"the new design's copy to h/2 is not a pure refinement: {exc}")
        record["check_layer"]["copy_is_the_models_E_DF"] = bool(np.array_equal(s_f, s_flow_model))
        require(record["check_layer"]["copy_is_the_models_E_DF"],
                "the copy to h/2 is not the optimised model's own E_DF copy")
        parents = ref.parent_of_each_fine_element(design_mesh, fine.thermal_mesh)
        same_material = bool(np.array_equal(np.asarray(fine.maps.density(jnp.asarray(s_f))),
                                            np.asarray(s_bin)[parents]))
        record["check_layer"]["thermal_material_is_the_parents"] = same_material
        require(same_material, "the new design's h/8 material is not its parents' material")
        checkpoint("check layer: before its solves")
        saved["new_solid_fraction_flow_h2"] = s_f
        solve_new(fine, s_f, yard)
        del fine
        gc.collect()

    new = cells["new/check"]
    record["responses"]["export_gap"] = {
        "J": rel(new["J"], terminal["J_common_scale"]), "psi": rel(new["psi"], terminal["psi"]),
        "compliance": rel(new["compliance"], terminal["compliance"]),
        "note": "both on D, at the common scale"}
    record["candidates"] = {n: {k: base[n].get(k) for k in (*COMPARED, "D_T_over_Q", "T_max")}
                            for n in NAMES}
    record["candidate_components"] = {
        n: component_summary(fluid_components(design_mesh, binaries[n]))["sizes"] for n in NAMES}
    record["against"] = {}
    for n in NAMES:
        kind = zb.comparison_kind(new, base[n])
        record["against"][n] = None if kind is None else {
            "kind": kind, **{q: rel(new[q], base[n][q]) for q in COMPARED},
            "T_max": rel(new["T_max"], base[n]["T_max"]),
            "terms": {"dissipation": w * (new["psi"] - base[n]["psi"]) / scale.psi_0,
                      "thermal_compliance": (1.0 - w) * (new["compliance"] - base[n]["compliance"])
                      / scale.c_0,
                      "dJ": new["J"] - base[n]["J"]}}
    points = {n: (base[n]["psi"] / scale.psi_0, base[n]["compliance"] / scale.c_0) for n in NAMES}
    points["new"] = (new["psi"] / scale.psi_0, new["compliance"] / scale.c_0)
    single_component = [n for n in NAMES if len(record["candidate_components"][n]) == 1]
    if export["components"]["single_component"]:
        single_component.append("new")
    record["lowest_by_weight"] = {
        "note": "a fixed set of designs reweighted on the common scale; not an optimised front",
        "all": lowest_by_weight(points),
        "single_component": lowest_by_weight({n: points[n] for n in single_component}),
        "single_component_designs": single_component}
    finish()

    r = record["responses"]
    print(f"\noptimisation J {r['optimisation']['J']:+.3%}  Psi {r['optimisation']['psi']:+.3%}  "
          f"C {r['optimisation']['compliance']:+.3%}  grey {r['optimisation']['grey'][0]:.4f} -> "
          f"{r['optimisation']['grey'][1]:.4f}  (over R1t and R1v: J "
          f"{r['over_r1t_and_r1v']['J']:+.3%})")
    print(f"export gap   J {r['export_gap']['J']:+.3%}  Psi {r['export_gap']['psi']:+.3%}  "
          f"C {r['export_gap']['compliance']:+.3%}")
    for n, v in record["against"].items():
        print(f"new binary against {n:5s} on D: " + (
            "not usable" if v is None else
            f"J {v['J']:+.3%}  Psi {v['psi']:+.3%}  C {v['compliance']:+.3%}  T_max "
            f"{v['T_max']:+.3%} [{v['kind']}]"))
    for key, intervals in (("all", record["lowest_by_weight"]["all"]),
                           ("single component", record["lowest_by_weight"]["single_component"])):
        print(f"lowest by weight ({key}): " + ", ".join(
            f"{i['lowest']} {i['from']:.4f}-{i['to']:.4f}" for i in intervals))
    print(f"stop: {result.stop_reason}; convergence not verified")
    print(f"\nwrote {out / RECORD} and {out / FIELDS}; total "
          f"{record['cost']['wall_clock_total_s']:.0f} s")


if __name__ == "__main__":
    main()
