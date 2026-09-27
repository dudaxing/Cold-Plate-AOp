"""R1r: a budgeted optimisation on the check layer D, at the common scale.

The contract proposed in the review of b57cd61. The model is R1q's
`zhao2d_fineflow.Zhao2DFineFlowProblem`: the design on h (5000 raw variables,
the filter at 2e-4, the volume-preserving projection at beta = 32 with eta
solved on the coarse design volume), the flow on h/2 and the temperature on
h/8. J is on the development model's Psi_0 and C_0 as a declared common scale,
w = 0.5.

From R1o's raw continuous terminal design, which is R1q's main point, with
MMA's history reinitialised -- a warm start, not an exact resume -- at most 20
MMA updates at alpha_max = 1e7 and beta = 32, move limit 0.1, no internal
restarts, every state gated at 1e-8 by the entry. The zero step must reproduce
R1q's main point (J_D = 1.00092334514); that same evaluation is the first
update's basis, with no separate forward for the check.

Checked in stages. A failed gate writes the record and exits non-zero before
the next build, update or solve:

* inputs: the start design is R1o's last saved design row and R1q's main
  point; the candidates' D values and binary designs are the recorded ones;
* the model and the scale: the flow and thermal sides are R1m's check layer,
  the design side the development model's h mesh with 5000 variables; the
  scale is the one R1q used;
* the zero step's map, nothing solved: a non-degenerate root, and eta and g as
  at R1q's main point;
* at most 20 updates, the zero step checked against R1q's main point before
  the first update, and the terminal evaluated at the same point;
* the export by R1l's rule if the terminal is feasible; an infeasible
  terminal, or an export that does not qualify or is not connected, ends the
  run there -- a result, not a failure;
* the new binary design on D, once: one flow and one thermal solve, each
  gated, on R1m's check-layer route (a dual model built on h/2, the design
  copied down), which R1q showed gives this model's states for a given design.

The new qualified binary design is compared, on D and at the common scale,
with the saved values of R1o's design (the lead candidate), R1n's (the
control) and the R1l pilot's (the lower-dissipation alternative); none of them
is re-solved. A geometry identical to a candidate's reuses its recorded values.
No development-model run, no finer layer, no second binary check.

    python scripts/zhao2d_r1r_d_optimise.py [--inputs results] [--out DIR] [--overwrite]
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
from tfopus import zhao2d_driver as drv  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_fineflow as ff  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402
from tfopus import zhao2d_refine as ref  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb, timed  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file  # noqa: E402
# R1m's report, so the new design on D is summarised exactly as the candidates were
from zhao2d_r1m_flow_check import flat, mesh_identity, thermal_identity  # noqa: E402

ALPHA_MAX, BETA = 1.0e7, 32.0
FLOW_REFINEMENT, THERMAL_REFINEMENT, QUADRATURE = 2, 4, 3
BUDGET, MOVE_LIMIT, NEAR_LIMIT = 20, 0.1, 0.0999
GATE, ANCHOR_TOL = 1e-8, 1e-12
RECORD = "zhao2d_r1r.json"
FIELDS = "zhao2d_r1r_fields.npz"
PROGRESS = "zhao2d_r1r_progress.json"
STACKS = "zhao2d_r1r_stacks.log"
INPUTS = ("zhao2d_r1l_vp_pilot.json", "zhao2d_r1l_vp_pilot_fields.npz",
          "zhao2d_r1m_flow_check.json", "zhao2d_r1n_beta32.json", "zhao2d_r1n_fields.npz",
          "zhao2d_r1o.json", "zhao2d_r1o_fields.npz", "zhao2d_r1q_fineflow_check.json")
KEYS = ("J_common_scale", "psi", "compliance", "constraint_g", "v_f_design_domain",
        "grey_fraction", "projection_eta", "projection_root_slope",
        "flow_residual_relative", "thermal_residual_relative")
NAMES = ("r1o", "r1n", "pilot")
COMPARED = ("J", "J_star", "psi", "compliance")


class ZeroStepMismatch(RuntimeError):
    """The zero step does not reproduce R1q's main point."""


def rel(b, a) -> float:
    return b / a - 1.0


def load_json(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def step_norms(designs: np.ndarray) -> list[dict]:
    """Each update's design change, with its norm named."""
    out = []
    for k, d in enumerate(np.diff(np.asarray(designs), axis=0), start=1):
        out.append({"update": k, "l2": float(np.linalg.norm(d)), "linf": float(np.abs(d).max()),
                    "rms": float(np.sqrt(np.mean(d * d)))})
    return out


def zero_step_anchor(zero: dict, main: dict, tol: float = ANCHOR_TOL) -> dict:
    """The zero step against R1q's main point: the same x, the same model, the same scale."""
    out = zb.anchor_status({"psi": zero["psi"], "compliance": zero["compliance"]},
                           {"psi": main["psi"], "compliance": main["compliance"],
                            "source": "R1q's main point, R1o's raw terminal design on D"}, tol)
    out["J_common_scale_relative"] = rel(zero["J_common_scale"], main["J_common_scale"])
    out["constraint_g_difference"] = zero["constraint_g"] - main["constraint_g"]
    out["eta_difference"] = zero["projection_eta"] - main["projection_eta"]
    out["reproduced"] = bool(out["reproduced"]
                             and abs(out["J_common_scale_relative"]) <= tol
                             and abs(out["constraint_g_difference"]) <= tol
                             and abs(out["eta_difference"]) <= tol)
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
    faulthandler.dump_traceback_later(10800, repeat=True, file=stacks)  # longer than a healthy run
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
    fp = np.load(inputs / "zhao2d_r1l_vp_pilot_fields.npz")
    fn = np.load(inputs / "zhao2d_r1n_fields.npz")
    fo = np.load(inputs / "zhao2d_r1o_fields.npz")
    x_start = np.asarray(fo["design"], dtype=np.float64)
    main_point = q_rec["main_point"]["evaluation"]
    rows = m_rec["rows"]
    # the candidates on D, as recorded: R1o's and R1n's check-layer cells, and R1m's for the pilot
    base = {"r1o": o_rec["cells"]["new/check"], "r1n": n_rec["cells"]["new/check"],
            "pilot": m_rec["cells"]["pilot/flow_h2"]}
    binaries = {"r1o": fo["solid_fraction_binary"], "r1n": fn["solid_fraction_binary"],
                "pilot": fp["solid_fraction_binary"]}
    binary_sha = {"r1o": o_rec["export"]["binary_sha256"], "r1n": n_rec["export"]["binary_sha256"],
                  "pilot": p_rec["export"]["binary_sha256"]}

    record = {
        "note": ("R1r: up to 20 MMA updates on the check layer D (design h, flow h/2, thermal "
                 "h/8) from R1o's raw terminal design, at beta 32, MMA reinitialised, J on the "
                 "development model's Psi_0 and C_0 as a declared common scale; the zero step "
                 "must reproduce R1q's main point. The terminal's qualified binary design is "
                 "evaluated once on D and compared with R1o's, R1n's and the R1l pilot's saved "
                 "D values. With D generating and ranking, this is analysis on one model; a warm "
                 "start, not an exact resume; nothing here is a convergence claim."),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in INPUTS},
        "contract": {"model": {"design": "h, 5000 raw variables", "flow_refinement": FLOW_REFINEMENT,
                               "thermal_refinement": THERMAL_REFINEMENT,
                               "thermal_quadrature": QUADRATURE},
                     "projection": config.projection, "beta": BETA, "alpha_max": ALPHA_MAX,
                     "max_mma_updates": BUDGET, "move_limit": MOVE_LIMIT, "gate": GATE,
                     "weight": w, "continuation": "none", "internal_restarts": "none",
                     "fingerprint": config.fingerprint(),
                     "zero_step_anchor": {"J_common_scale": main_point["J_common_scale"],
                                          "source": "results/zhao2d_r1q_fineflow_check.json",
                                          "tolerance": ANCHOR_TOL},
                     "binary_solves_at_most": {"flow": 1, "thermal": 1, "layer": "D"},
                     "near_limit_band": f"largest single change >= {NEAR_LIMIT}",
                     "authorised": ("by the user, on the local CPU, as the review file of b57cd61 "
                                    "states it: at most 20 updates, the binary design on D only "
                                    "(not the pasted variant of 30 updates with 2F + 2T on D and "
                                    "A)")},
        "initial_design": {"file": "results/zhao2d_r1o_fields.npz", "key": "design",
                           "sha256_float64": fs.digest(x_start)},
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
        """The new binary design on D: one flow and one thermal solve, each gated."""
        key = "new/check"
        s = jnp.asarray(s)
        fluid = z.fluid_fractions(problem.flow_mesh, s)
        cell = {"design": "new", "layer": "check", "state": "solved here", "fluid_fractions": fluid,
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
        temp, t_thermal = timed(problem.solve_thermal, pv, s, ALPHA_MAX)
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
              f"D_T/Q {cell['D_T_over_Q']:.4%}  |R| {cell['residual_flow']:.1e}/"
              f"{cell['residual_thermal']:.1e}  flow {t_flow:.1f} s, thermal {t_thermal:.1f} s",
              flush=True)
        require(cell["volume_feasible"], f"{key}: over the fluid bound, {fluid}")
        require(cell["gate_passed"], f"{key}: a residual fails the gate {rep['residual_relative']}")
        checkpoint(f"check layer: the new design's thermal solve ({t_thermal:.1f} s)")

    # -- inputs ---------------------------------------------------------------------------
    require(np.array_equal(fo["design"], fo["designs"][-1]),
            "the start design is not R1o's last saved design row")
    require(fs.digest(x_start) == q_rec["main_point"]["sha256_float64"],
            "the start design is not R1q's main point")
    for name, rec_ in (("R1m", m_rec), ("R1n", n_rec), ("R1o", o_rec), ("R1q", q_rec)):
        require(not any(c["failures"] for c in rec_["checkpoints"]), f"{name}'s record has failed checks")
    # R1l's record predates checkpoints: its checks, and its export's own verdict
    require(not any(p_rec["checks"].values()) and p_rec["export"]["qualified"],
            "the R1l pilot's record has failed checks or an unqualified export")
    require(q_rec["verdict"]["all_pass"], "R1q's gradient check did not pass")
    for name in NAMES:
        require(fs.digest(binaries[name]) == binary_sha[name],
                f"{name}: the saved binary design is not the one its record exported")
        require(zb.cell_usable(base[name]) and base[name]["volume_feasible"],
                f"{name}: its D cell is not a usable, gated, feasible state in its record")
    for name in ("r1o", "r1n"):
        a = q_rec["anchors"][name]
        require(a["anchor"]["reproduced"] and a["psi"] == base[name]["psi"]
                and a["compliance"] == base[name]["compliance"],
                f"{name}: R1q's anchor does not carry its recorded D values")
    checkpoint("inputs")

    # -- the model and the scale ---------------------------------------------------------------
    t0 = time.perf_counter()
    problem = ff.Zhao2DFineFlowProblem(spec, config, FLOW_REFINEMENT, THERMAL_REFINEMENT, QUADRATURE)
    cost["build_s"] = time.perf_counter() - t0
    print(f"built D: design {problem.design_mesh.num_elems} el ({problem.num_design} variables), "
          f"flow {problem.flow_mesh.num_elems} el, thermal {problem.thermal_mesh.num_elems} el "
          f"({cost['build_s']:.1f} s)", flush=True)
    record["identities"] = {"flow_mesh": mesh_identity(problem.flow_mesh),
                            "thermal_mesh": thermal_identity(problem),
                            "design_mesh": mesh_identity(problem.design_mesh),
                            "num_design": problem.num_design,
                            "model_identity": json.loads(problem.model_identity()),
                            "nesting": problem.nesting}
    require(record["identities"]["flow_mesh"] == rows["flow_h2"]["flow_mesh"],
            "the flow side is not R1m's check layer")
    require(record["identities"]["thermal_mesh"] == rows["flow_h2"]["thermal_mesh"],
            "the thermal side is not R1m's check layer")
    require(record["identities"]["design_mesh"] == rows["flow_h"]["flow_mesh"],
            "the design mesh is not the development model's h mesh")
    require(problem.num_design == 5000, f"{problem.num_design} design variables, not 5000")
    try:
        scale = ff.common_scale(problem, dual.reference_file(4, QUADRATURE), 4, QUADRATURE)
    except ValueError as exc:
        require(False, f"the common scale: {exc}")
        checkpoint("the model and the common scale")
    record["scale"] = {"psi_0": scale.psi_0, "c_0": scale.c_0, "source_file": scale.source_file,
                       "source_sha256": scale.source_sha256, "source_model": scale.source_model,
                       "declared": "the development model's constants, used as a common scale; "
                                   "not a reference of this model"}
    require(all(scale_value == q_rec["scale"][k] for k, scale_value in
                (("psi_0", scale.psi_0), ("c_0", scale.c_0), ("source_sha256", scale.source_sha256))),
            "the scale is not the one R1q used")
    for name in NAMES:
        require(base[name]["J"] == common_j(base[name]["psi"], base[name]["compliance"]),
                f"{name}: its recorded D value of J is not on this scale")
    yard = {"psi_0": scale.psi_0, "c_0": scale.c_0, "file": "tfopus/zhao2d_reference_dual_r4q3_v1.json",
            "file_sha256": scale.source_sha256, "declared": "the common scale"}
    checkpoint("the model and the common scale")

    # -- the zero step's map, nothing solved ------------------------------------------------------
    xj = jnp.asarray(x_start)
    root = problem.projection_root(xj, BETA)
    g0 = float(problem.fluid_fraction(xj, BETA)) / config.max_fluid_fraction - 1.0
    record["zero_step_map"] = {"root": root, "g0": g0,
                               "eta_vs_r1q_main_point": root["eta"] - main_point["projection_eta"],
                               "g_vs_r1q_main_point": g0 - main_point["constraint_g"]}
    print(f"zero-step map: eta {root['eta']:.10f} (R1q's main point "
          f"{main_point['projection_eta']:.10f}), root slope {root['slope']:.3e} (non-degenerate "
          f"{root['nondegenerate']}), g0 {g0:+.6e}", flush=True)
    require(root["nondegenerate"], f"the zero step's projection root is degenerate: {root}")
    require(abs(root["eta"] - main_point["projection_eta"]) <= ANCHOR_TOL,
            f"eta {root['eta']} is not R1q's main point's {main_point['projection_eta']}")
    require(abs(g0 - main_point["constraint_g"]) <= ANCHOR_TOL,
            f"the constraint {g0} is not R1q's main point's {main_point['constraint_g']}")
    checkpoint("the zero step's map")

    # -- at most twenty updates, the zero step checked against R1q's main point first ------------
    phases = [drv.Phase("r1r-d-vp32", BUDGET, beta=BETA, alpha_max=ALPHA_MAX,
                        note="the check layer D at the common scale, beta 32 held, warm start "
                             "from R1o's raw terminal design")]

    def on_iteration(rec) -> None:
        progress.append({k: rec.get(k) for k in (*KEYS, "iteration", "seconds")})
        print(f"{rec['iteration']:3d}  J {rec['J_common_scale']:.8f}  Psi {rec['psi']:.6e}  "
              f"C {rec['compliance']:.3f}  g {rec['constraint_g']:+.3e}  grey "
              f"{rec['grey_fraction']:.4f}  eta {rec['projection_eta']:.5f}  |R| "
              f"{rec['flow_residual_relative']:.1e}/{rec['thermal_residual_relative']:.1e}  "
              f"{rec['seconds']:.1f} s", flush=True)
        (out / PROGRESS).write_text(json.dumps(progress, indent=1), encoding="utf-8")
        if rec["iteration"] == 0:
            anchor = zero_step_anchor(rec, main_point)
            record["zero_step_anchor"] = anchor
            print(f"zero step against R1q's main point: Psi {anchor['psi_relative']:+.1e}, "
                  f"C {anchor['compliance_relative']:+.1e}, J "
                  f"{anchor['J_common_scale_relative']:+.1e}, g "
                  f"{anchor['constraint_g_difference']:+.1e}, eta {anchor['eta_difference']:+.1e}",
                  flush=True)
            if not anchor["reproduced"]:
                raise ZeroStepMismatch(f"the zero step does not reproduce R1q's main point: {anchor}")

    t0 = time.perf_counter()
    try:
        result = ff.run(problem, scale, phases, move_limit=MOVE_LIMIT, on_iteration=on_iteration,
                        initial_design=x_start)
    except ZeroStepMismatch as exc:
        record["history"] = progress
        require(False, str(exc))
        checkpoint("the zero step against R1q's main point")
    except (r1.NotConverged, r1.DegenerateProjection) as exc:
        partial = getattr(exc, "partial", {})
        record["stop"] = {"stop_reason": type(exc).__name__, "message": str(exc),
                          "failed_iteration": partial.get("failed_iteration")}
        record["history"] = partial.get("history", progress)
        if "failed_design" in partial:
            saved.update(designs=partial["designs"], failed_design=partial["failed_design"])
        require(False, f"the updates stopped: {type(exc).__name__}: {exc}")
        checkpoint("the updates")
    cost["t_updates_including_terminal_s"] = time.perf_counter() - t0
    checkpoints.append({"stage": "the zero step against R1q's main point", "failures": []})

    history, terminal = result.history, result.terminal
    zero = history[0]
    evaluated = [*history, terminal]
    for rec_ in evaluated:
        rec_["J_star"] = float(dual.reporting_objective(rec_["psi"], rec_["compliance"], single, w))
        rec_["volume_feasible"] = zb.volume_feasible(rec_["v_f_design_domain"],
                                                     config.max_fluid_fraction)
    feasible = [r_ for r_ in evaluated if r_["volume_feasible"]]
    best = min(feasible, key=lambda r_: r_["J_common_scale"]) if feasible else None
    designs = np.vstack([result.designs, np.asarray(result.design)[None, :]])
    steps = step_norms(designs)
    record["history"], record["terminal"], record["steps"] = history, terminal, steps
    record["near_limit_updates"] = [s_["update"] for s_ in steps if s_["linf"] >= NEAR_LIMIT]
    record["stop"] = {"stop_reason": result.stop_reason, "proxy_criterion": result.proxy_criterion,
                      "proxy_fired_at_final_stage": result.proxy_fired_at_final_stage,
                      "mma_updates": len(history), "convergence_verified": False,
                      "first_feasible_iteration": next(
                          (r_["iteration"] for r_ in evaluated if r_["volume_feasible"]), None)}
    record["responses"] = {
        "optimisation": {
            "from": "the zero step (R1q's main point, reproduced)", "to": "the terminal",
            **{q: rel(terminal[k], zero[k]) for q, k in (("J", "J_common_scale"), ("psi", "psi"),
                                                         ("compliance", "compliance"))},
            "g": [zero["constraint_g"], terminal["constraint_g"]],
            "grey": [zero["grey_fraction"], terminal["grey_fraction"]]},
    }
    record["best_feasible_evaluated"] = (
        None if best is None else {"iteration": best["iteration"], "design_row": best["iteration"],
                                   "is_terminal": bool(best.get("terminal")),
                                   "J_common_scale": best["J_common_scale"], "psi": best["psi"],
                                   "compliance": best["compliance"],
                                   "constraint_g": best["constraint_g"]})
    saved.update(initial_design=result.initial_design, designs=designs, design=result.design,
                 initial_solid_fraction=result.initial_solid_fraction,
                 initial_press_vel=result.initial_press_vel,
                 initial_temperature=result.initial_temperature,
                 solid_fraction=result.solid_fraction, press_vel=result.press_vel,
                 temperature=result.temperature,
                 design_elem_centres=np.asarray(problem.design_mesh.elem_centres))
    if best is not None:
        saved["best_feasible_design"] = designs[best["iteration"]]
    print(f"terminal: J {terminal['J_common_scale']:.8f}  g {terminal['constraint_g']:+.3e}  grey "
          f"{terminal['grey_fraction']:.4f}  stop {result.stop_reason}; last step L2 "
          f"{steps[-1]['l2']:.4f}, Linf {steps[-1]['linf']:.4f}; near-limit updates "
          f"{record['near_limit_updates']}", flush=True)
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
    export["same_geometry_as"] = [n for n in NAMES if fs.digest(s_bin) == binary_sha[n]]
    export["cells_differing_from"] = {n: int(np.sum(s_bin != np.asarray(binaries[n]))) for n in NAMES}
    saved["solid_fraction_binary"] = s_bin
    print(f"export: t {export['t']:.10f}, {export['fluid_cells']} fluid cells, "
          f"{export['cells_changed_from_0.5']} changed from s = 0.5, connected "
          f"{export['connected']}, cells differing from the candidates "
          f"{export['cells_differing_from']}", flush=True)
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
    record["against"] = {}
    for n in NAMES:
        kind = zb.comparison_kind(new, base[n])
        record["against"][n] = None if kind is None else {
            "kind": kind, **{q: rel(new[q], base[n][q]) for q in COMPARED}}
    finish()

    r = record["responses"]
    print(f"\noptimisation J {r['optimisation']['J']:+.3%}  Psi {r['optimisation']['psi']:+.3%}  "
          f"C {r['optimisation']['compliance']:+.3%}  grey {r['optimisation']['grey'][0]:.4f} -> "
          f"{r['optimisation']['grey'][1]:.4f}")
    print(f"export gap   J {r['export_gap']['J']:+.3%}  Psi {r['export_gap']['psi']:+.3%}  "
          f"C {r['export_gap']['compliance']:+.3%}")
    for n, v in record["against"].items():
        print(f"new binary against {n:5s} on D: " + (
            "not usable" if v is None else
            f"J {v['J']:+.3%}  Psi {v['psi']:+.3%}  C {v['compliance']:+.3%} [{v['kind']}]"))
    print(f"stop: {result.stop_reason}; convergence not verified")
    print(f"\nwrote {out / RECORD} and {out / FIELDS}; total "
          f"{record['cost']['wall_clock_total_s']:.0f} s")


if __name__ == "__main__":
    main()
