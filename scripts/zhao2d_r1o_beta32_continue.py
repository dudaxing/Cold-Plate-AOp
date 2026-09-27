"""R1o: thirty more updates at beta = 32 from R1n's design, its binary design on two layers.

The contract proposed in the review of 35abba9. From R1n's raw terminal design
(results/zhao2d_r1n_fields.npz, "design"), with MMA's history reinitialised --
a warm start, not an exact resume -- the volume-preserving projection at the
same beta = 32, eta solved at every evaluation. Everything else is R1n's: flow
h, thermal h/4 (3x3), alpha_max = 1e7, q_alpha = q_kappa = 0.2, the filter, the
thermal residual, the dual-mesh reference, w = 0.5, move limit 0.1.

The map does not change, so the zero step must reproduce R1n's continuous
terminal: its Psi and C are compared with R1n's record as soon as the zero step
is evaluated, before any update. Early changes after that come from restarting
MMA, not from beta.

Checked in stages, as in R1n. A failed gate writes the record and exits
non-zero before the next build, update or solve:

* inputs: the start design is R1n's last saved design row; the two candidates'
  states -- R1n's qualified binary design and the R1l pilot's, on both layers --
  are the ones R1n, R1l and R1m recorded;
* development layer, before any update: the reference is R1n's yardstick; both
  candidates' saved states re-verify and reproduce their records' Psi and C; the
  zero step's root is non-degenerate, the volume preserved and the constraint
  gradient affine; then the zero step reproduces R1n's terminal;
* at most 30 MMA updates, every state gated by the driver, and the terminal
  evaluated at the same point;
* the export by R1l's rule if the terminal is feasible; an infeasible terminal,
  or an export that does not qualify or is not connected, ends the run there --
  a result, not a failure;
* the new binary design: one flow and one thermal solve on each layer, each
  gated; on the check layer after its meshes, both candidates' states and the
  new design's copy to h/2 are checked.

The new qualified binary design is compared with R1n's (the preferred
candidate at w = 0.5) and with the pilot's, on each layer, on the h/4 model's
constants. A geometry identical to a candidate's reuses that candidate's
states. No improvement closes the stage normally, with R1n's design still
preferred.

    python scripts/zhao2d_r1o_beta32_continue.py [--inputs results] [--out DIR] [--overwrite]
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
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402
from tfopus import zhao2d_refine as ref  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb, timed  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file  # noqa: E402
# R1m's report, so every layer is summarised exactly as R1m and R1n summarised it
from zhao2d_r1m_flow_check import flat, mesh_identity, thermal_identity  # noqa: E402

REFINEMENT, QUADRATURE = 4, 3
ALPHA_MAX, BETA = 1.0e7, 32.0
BUDGET, MOVE_LIMIT, NEAR_LIMIT = 30, 0.1, 0.0999
GATE, ANCHOR_TOL, AFFINE_RTOL = 1e-8, 1e-12, 1e-12
RECORD = "zhao2d_r1o.json"
FIELDS = "zhao2d_r1o_fields.npz"
PROGRESS = "zhao2d_r1o_progress.json"
STACKS = "zhao2d_r1o_stacks.log"
INPUTS = ("zhao2d_r1l_vp_pilot.json", "zhao2d_r1l_vp_pilot_fields.npz",
          "zhao2d_r1m_flow_check.json", "zhao2d_r1m_fields.npz",
          "zhao2d_r1n_beta32.json", "zhao2d_r1n_fields.npz")
KEYS = ("J_self", "psi", "compliance", "constraint_g", "v_f_design_domain", "grey_fraction",
        "projection_eta", "projection_root_slope", "design_step_norm",
        "flow_residual_relative", "thermal_residual_relative")
NAMES = ("r1n", "pilot")
COMPARED = ("J", "J_star", "psi", "compliance")


class ZeroStepMismatch(RuntimeError):
    """The zero step does not reproduce R1n's continuous terminal."""


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


def zero_step_anchor(zero: dict, terminal: dict, tol: float = ANCHOR_TOL) -> dict:
    """The zero step against R1n's terminal record: the same x and the same map."""
    out = zb.anchor_status({"psi": zero["psi"], "compliance": zero["compliance"]},
                           {"psi": terminal["psi"], "compliance": terminal["compliance"],
                            "source": "R1n's continuous terminal record"}, tol)
    out["J_self_relative"] = zero["J_self"] / terminal["J_self"] - 1.0
    out["constraint_g_difference"] = zero["constraint_g"] - terminal["constraint_g"]
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
    faulthandler.dump_traceback_later(4800, repeat=True, file=stacks)  # longer than a healthy run
    t_start = time.perf_counter()

    spec = z.Zhao2DSpec()
    config = r1.R1Config(projection=r1.Projection.VOLUME_PRESERVING)
    w = config.weight
    material = za.build_material(spec, ALPHA_MAX)
    single = r1.load_reference(spec, config)  # J* only
    b_rec = load_json(inputs / "zhao2d_r1l_vp_pilot.json")
    m_rec = load_json(inputs / "zhao2d_r1m_flow_check.json")
    n_rec = load_json(inputs / "zhao2d_r1n_beta32.json")
    fb = np.load(inputs / "zhao2d_r1l_vp_pilot_fields.npz")
    fm = np.load(inputs / "zhao2d_r1m_fields.npz")
    fn = np.load(inputs / "zhao2d_r1n_fields.npz")
    x_start = np.asarray(fn["design"], dtype=np.float64)
    nd, nc, bx = n_rec["cells"]["new/development"], n_rec["cells"]["new/check"], b_rec["export"]
    pc = m_rec["cells"]["pilot/flow_h2"]
    # the two candidates' saved states: R1n's on both layers, the pilot's from R1l and R1m
    dev = {"r1n": {"s": fn["solid_fraction_binary"], "press_vel": fn["new_press_vel_development"],
                   "temperature": fn["new_temperature_development"], "rec": nd,
                   "sha": nd["state_sha256"]},
           "pilot": {"s": fb["solid_fraction_binary"], "press_vel": fb["binary_press_vel"],
                     "temperature": fb["binary_temperature_h4"], "rec": bx,
                     "sha": {"s": bx["binary_sha256"], **bx["state_sha256"]}}}
    chk = {"r1n": {"s": fn["new_solid_fraction_flow_h2"], "press_vel": fn["new_press_vel_check"],
                   "temperature": fn["new_temperature_check"], "rec": nc,
                   "sha": nc["state_sha256"]},
           "pilot": {"s": fm["pilot_solid_fraction_flow_h2"], "press_vel": fm["pilot_press_vel_flow_h2"],
                     "temperature": fm["pilot_temperature_flow_h2_thermal_h8"], "rec": pc,
                     "sha": {"s": pc["state_sha256"]["s_flow_h2"],
                             "press_vel": pc["state_sha256"]["press_vel"],
                             "temperature": pc["state_sha256"]["temperature"]}}}

    record = {
        "note": ("R1o: up to 30 more updates at the same beta 32 from R1n's raw terminal design, "
                 "MMA reinitialised, on the development model (flow h, thermal h/4); the zero "
                 "step must reproduce R1n's continuous terminal. The new qualified binary design "
                 "is evaluated there and on the check layer (flow h/2, thermal h/8) against "
                 "R1n's and the R1l pilot's, on the h/4 model's constants. A warm start, not an "
                 "exact resume; nothing here is a convergence claim."),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in INPUTS},
        "contract": {"projection": config.projection, "beta": BETA, "alpha_max": ALPHA_MAX,
                     "max_mma_updates": BUDGET, "move_limit": MOVE_LIMIT, "gate": GATE,
                     "weight": w, "continuation": "none", "fingerprint": config.fingerprint(),
                     "binary_solves_at_most": {"flow": 2, "thermal": 2},
                     "near_limit_band": f"largest single change >= {NEAR_LIMIT}"},
        "initial_design": {"file": "results/zhao2d_r1n_fields.npz", "key": "design",
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

    def reuse(problem, layer: str, name: str, d: dict, anchor_source: str, yard: dict) -> None:
        """A candidate's saved state on this layer: re-verified and re-evaluated, never re-solved."""
        key = f"{name}/{layer}"
        s, pv, temp = jnp.asarray(d["s"]), jnp.asarray(d["press_vel"]), jnp.asarray(d["temperature"])
        fluid = z.fluid_fractions(problem.flow_mesh, s)
        cell = {"design": name, "layer": layer, "state": f"reused: {anchor_source}",
                "fluid_fractions": fluid,
                "volume_feasible": zb.volume_feasible(fluid["v_f_design_domain"],
                                                      config.max_fluid_fraction)}
        cells[key] = cell
        require(cell["volume_feasible"], f"{key}: over the fluid bound, {fluid}")
        try:
            cell["flow_verification"] = fs.verify_flow_state(problem, pv, s, ALPHA_MAX, GATE)
        except (ValueError, r1.NotConverged) as exc:
            require(False, f"{key}: the saved flow does not verify: {exc}")
            return
        t0 = time.perf_counter()
        rep = fs.cell_report(problem, s, pv, temp, ALPHA_MAX, single)
        cell.update(analysed=True, **flat(rep, yard, w))
        cell["gate_passed"] = zb.gate_passed(rep["residual_relative"], GATE)
        cell["anchor"] = zb.anchor_status(
            {"psi": rep["psi"], "compliance": rep["compliance"]},
            {"psi": d["rec"]["psi"], "compliance": d["rec"]["compliance"], "source": anchor_source},
            ANCHOR_TOL)
        cell["t_report_s"] = time.perf_counter() - t0
        cell["report"] = rep
        require(cell["gate_passed"], f"{key}: a residual fails the gate {rep['residual_relative']}")
        require(cell["anchor"]["reproduced"], f"{key}: does not reproduce {anchor_source}: "
                                              f"{cell['anchor']}")
        print(f"{key}: Psi {cell['psi']:.9e}  C {cell['compliance']:.6f}  J {cell['J']:.9f}  "
              f"D_T/Q {cell['D_T_over_Q']:.4%}  anchor "
              f"{'ok' if cell['anchor']['reproduced'] else 'MISMATCH'}  ({cell['t_report_s']:.1f} s)",
              flush=True)

    def solve_new(problem, layer: str, s, yard: dict) -> None:
        """The new binary design on this layer: one flow and one thermal solve, each gated."""
        key = f"new/{layer}"
        s = jnp.asarray(s)
        fluid = z.fluid_fractions(problem.flow_mesh, s)
        cell = {"design": "new", "layer": layer, "state": "solved here", "fluid_fractions": fluid,
                "volume_feasible": zb.volume_feasible(fluid["v_f_design_domain"],
                                                      config.max_fluid_fraction)}
        cells[key] = cell
        alpha = materials.brinkman_penalty(s, material)
        pv, t_flow = timed(tf_solver.modified_newton_raphson_solve, problem.flow,
                           problem.flow_x0, alpha)
        cell["t_flow_solve_s"] = t_flow
        saved[f"new_press_vel_{layer}"] = np.asarray(pv)
        try:
            cell["flow_verification"] = fs.verify_flow_state(problem, pv, s, ALPHA_MAX, GATE)
        except (ValueError, r1.NotConverged) as exc:
            require(False, f"{key}: the flow solve fails its gate: {exc}")
        checkpoint(f"{layer}: the new design's flow solve ({t_flow:.1f} s)")
        cell["flow_identity"] = fs.flow_state_identity(problem, s, ALPHA_MAX)
        temp, t_thermal = timed(problem.solve_thermal, pv, s, ALPHA_MAX)
        cell["t_thermal_solve_s"] = t_thermal
        saved[f"new_temperature_{layer}"] = np.asarray(temp)
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
        checkpoint(f"{layer}: the new design's thermal solve ({t_thermal:.1f} s)")

    def compare(layer: str) -> dict:
        new = cells[f"new/{layer}"]
        out_ = {}
        for name in NAMES:
            base = cells[f"{name}/{layer}"]
            kind = zb.comparison_kind(new, base)
            out_[name] = None if kind is None else {"kind": kind,
                                                    **{q: rel(new[q], base[q]) for q in COMPARED}}
        return out_

    # -- inputs ---------------------------------------------------------------------------
    require(np.array_equal(fn["design"], fn["designs"][-1]),
            "the start design is not R1n's last saved design row")
    require(not any(c["failures"] for c in n_rec["checkpoints"]), "R1n's record has failed checks")
    require(fs.digest(fn["solid_fraction_binary"]) == n_rec["export"]["binary_sha256"],
            "R1n's binary design is not the one its record exported")
    for layer, group in (("development", dev), ("check", chk)):
        for name, d in group.items():
            for key in ("s", "press_vel", "temperature"):
                require(fs.digest(d[key]) == d["sha"][key],
                        f"{name}/{layer}: the saved {key} is not the recorded one")
            require(zb.cell_usable(d["rec"]) and d["rec"].get("qualified", True),
                    f"{name}/{layer}: not a usable, gated state in its record")
    checkpoint("inputs")

    # -- development layer: the candidates, then the zero step's map -----------------------
    t0 = time.perf_counter()
    problem = dual.Zhao2DDualProblem(spec, config, REFINEMENT, QUADRATURE)
    cost["build_development_s"] = time.perf_counter() - t0
    print(f"built the development layer: flow {problem.flow_mesh.num_elems} el, thermal "
          f"{problem.thermal_mesh.num_elems} el ({cost['build_development_s']:.1f} s)", flush=True)
    ref_path = dual.reference_file(REFINEMENT, QUADRATURE)
    try:
        reference = dual.load_reference(problem, ref_path)
    except (ValueError, FileNotFoundError) as exc:
        require(False, f"the reference does not belong to the development layer: {exc}")
        checkpoint("development layer: the reference")
    yard = {"psi_0": reference.psi_0, "c_0": reference.c_0,
            "file": ref_path.relative_to(REPO).as_posix(), "file_sha256": sha256_file(ref_path)}
    require(all(yard[k] == n_rec["yardstick"][k] for k in ("psi_0", "c_0", "file")),
            "the reference is not R1n's yardstick")
    record["yardstick"] = yard
    record["reporting_scale"] = {"psi_0": single.psi_0, "c_0": single.c_0,
                                 "identity": "single-mesh h reference; J* only"}
    for name, source in (("r1n", "R1n's record, its new design at flow h, thermal h/4"),
                         ("pilot", "R1l's record, the pilot's binary design at flow h, thermal h/4")):
        reuse(problem, "development", name, dev[name], source, yard)
    checkpoint("development layer: the candidates' states")

    xj = jnp.asarray(x_start)
    design_mask = np.asarray(problem.flow_mesh.design_mask)
    areas = np.asarray(problem.flow_mesh.elem_area)
    a_d = jnp.asarray(areas[problem.design_elements])

    def constraint(v):
        return problem.fluid_fraction(v, BETA) / config.max_fluid_fraction - 1.0

    g0, dg0 = jax.value_and_grad(constraint)(xj)
    affine = -(problem.filter_matrix.T @ a_d) / (config.max_fluid_fraction * float(a_d.sum()))
    root = problem.projection_root(xj, BETA)
    drift = float(problem.fluid_fraction(xj, BETA) - problem.fluid_fraction(xj, 0.0))
    gap = float(jnp.max(jnp.abs(dg0 - affine)))
    scale = float(jnp.max(jnp.abs(affine)))
    record["zero_step_map"] = {"root": root, "g0": float(g0),
                               "volume_projected_minus_filtered": drift,
                               "constraint_gradient_vs_affine_max_abs": gap,
                               "constraint_gradient_affine_max_abs": scale,
                               "eta_vs_r1n_terminal": root["eta"] - n_rec["terminal"]["projection_eta"]}
    print(f"zero-step map: eta {root['eta']:.8f} (R1n's terminal "
          f"{n_rec['terminal']['projection_eta']:.8f}), root slope {root['slope']:.3e} "
          f"(non-degenerate {root['nondegenerate']}), g0 {float(g0):+.8f}, volume drift "
          f"{drift:+.1e}, dg vs affine {gap:.1e}", flush=True)
    require(root["nondegenerate"], f"the zero step's projection root is degenerate: {root}")
    require(abs(drift) <= 1e-12, f"the projection does not preserve the volume: {drift:+.3e}")
    require(gap <= AFFINE_RTOL * scale, f"the constraint gradient is not affine: {gap:.3e}")
    checkpoint("development layer: the zero step's map")

    # -- at most thirty updates, the zero step checked against R1n's terminal first --------
    phases = [drv.Phase("r1o-vp32", BUDGET, beta=BETA, alpha_max=ALPHA_MAX,
                        note="volume-preserving projection, beta 32 held, warm start from "
                             "R1n's raw terminal design")]

    def on_iteration(rec) -> None:
        progress.append({k: rec.get(k) for k in (*KEYS, "iteration", "seconds")})
        print(f"{rec['iteration']:3d}  J {rec['J_self']:.8f}  Psi {rec['psi']:.6e}  "
              f"C {rec['compliance']:.3f}  g {rec['constraint_g']:+.3e}  grey "
              f"{rec['grey_fraction']:.4f}  eta {rec['projection_eta']:.5f}  |R| "
              f"{rec['flow_residual_relative']:.1e}/{rec['thermal_residual_relative']:.1e}  "
              f"{rec['seconds']:.1f} s", flush=True)
        (out / PROGRESS).write_text(json.dumps(progress, indent=1), encoding="utf-8")
        if rec["iteration"] == 0:
            anchor = zero_step_anchor(rec, n_rec["terminal"])
            record["zero_step_anchor"] = anchor
            print(f"zero step against R1n's terminal: Psi {anchor['psi_relative']:+.1e}, "
                  f"C {anchor['compliance_relative']:+.1e}, J {anchor['J_self_relative']:+.1e}",
                  flush=True)
            if not anchor["reproduced"]:
                raise ZeroStepMismatch(f"the zero step does not reproduce R1n's terminal: {anchor}")

    t0 = time.perf_counter()
    try:
        result = drv.run(problem, reference, phases, move_limit=MOVE_LIMIT,
                         on_iteration=on_iteration, initial_design=x_start)
    except ZeroStepMismatch as exc:
        record["history"] = progress
        require(False, str(exc))
        checkpoint("the zero step against R1n's terminal")
    except (r1.NotConverged, r1.DegenerateProjection) as exc:
        partial = getattr(exc, "partial", {})
        record["stop"] = {"stop_reason": type(exc).__name__, "message": str(exc),
                          "failed_iteration": partial.get("failed_iteration")}
        record["history"] = partial.get("history", progress)
        require(False, f"the updates stopped: {type(exc).__name__}: {exc}")
        checkpoint("the thirty updates")
    cost["t_updates_including_terminal_s"] = time.perf_counter() - t0
    checkpoints.append({"stage": "the zero step against R1n's terminal", "failures": []})

    history, terminal = result.history, result.terminal
    zero = history[0]
    evaluated = [*history, terminal]
    for rec_ in evaluated:
        rec_["J_star"] = float(dual.reporting_objective(rec_["psi"], rec_["compliance"], single, w))
        rec_["volume_feasible"] = zb.volume_feasible(rec_["v_f_design_domain"],
                                                     config.max_fluid_fraction)
    feasible = [r_ for r_ in evaluated if r_["volume_feasible"]]
    best = min(feasible, key=lambda r_: r_["J_self"]) if feasible else None
    designs = np.vstack([result.designs, np.asarray(result.design)[None, :]])
    steps = step_norms(designs)
    record["history"], record["terminal"], record["steps"] = history, terminal, steps
    record["near_limit_updates"] = [s_["update"] for s_ in steps if s_["linf"] >= NEAR_LIMIT]
    record["stop"] = {"stop_reason": result.stop_reason, "proxy_criterion": result.proxy_criterion,
                      "mma_updates": len(history), "convergence_verified": False,
                      "first_feasible_iteration": next(
                          (r_["iteration"] for r_ in evaluated if r_["volume_feasible"]), None)}
    record["responses"] = {
        "optimisation": {
            "from": "the zero step (R1n's continuous terminal, reproduced)", "to": "the terminal",
            **{q: rel(terminal[k], zero[k]) for q, k in (("J", "J_self"), ("psi", "psi"),
                                                         ("compliance", "compliance"))},
            "g": [zero["constraint_g"], terminal["constraint_g"]],
            "grey": [zero["grey_fraction"], terminal["grey_fraction"]]},
    }
    record["best_feasible_evaluated"] = (
        None if best is None else {"iteration": best["iteration"],
                                   "is_terminal": bool(best.get("terminal")),
                                   "J_self": best["J_self"], "psi": best["psi"],
                                   "compliance": best["compliance"],
                                   "constraint_g": best["constraint_g"]})
    saved.update(initial_design=result.initial_design, designs=designs, design=result.design,
                 solid_fraction=result.solid_fraction, press_vel=result.press_vel,
                 temperature=result.temperature,
                 elem_centres=np.asarray(problem.flow_mesh.elem_centres))
    print(f"terminal: J {terminal['J_self']:.8f}  g {terminal['constraint_g']:+.3e}  grey "
          f"{terminal['grey_fraction']:.4f}  stop {result.stop_reason}; last step L2 "
          f"{steps[-1]['l2']:.4f}, Linf {steps[-1]['linf']:.4f}; near-limit updates "
          f"{record['near_limit_updates']}", flush=True)

    # -- the export -------------------------------------------------------------------------
    export = {"terminal_volume_feasible": bool(terminal["volume_feasible"])}
    record["export"] = export
    if not terminal["volume_feasible"]:
        end("the terminal is not volume-feasible, so nothing is exported")
    s_t = np.asarray(result.solid_fraction)
    out_ = zb.volume_threshold(s_t, design_mask, areas, config.max_fluid_fraction)
    s_bin = out_.pop("s_binary")
    conn = zb.connectivity(problem.flow_mesh, s_t, (out_["t"],))[f"{out_['t']:g}"]
    export.update(out_, connected=bool(conn["inlet_outlet_connected"]),
                  fluid_components=conn["fluid_components"],
                  volume_feasible=zb.volume_feasible(out_["fluid_fraction"],
                                                     config.max_fluid_fraction),
                  binary_sha256=fs.digest(s_bin),
                  fluid_cells_at_s_0_5=int(np.sum((s_t < 0.5) & design_mask)))
    export["qualified"] = bool(export["volume_feasible"] and export["connected"])
    export["same_geometry_as"] = [n for n in NAMES if fs.digest(s_bin) == fs.digest(dev[n]["s"])]
    export["cells_differing_from"] = {n: int(np.sum(s_bin != np.asarray(dev[n]["s"]))) for n in NAMES}
    saved["solid_fraction_binary"] = s_bin
    print(f"export: t {export['t']:.10f}, {export['fluid_cells']} fluid cells, "
          f"{export['cells_changed_from_0.5']} changed from s = 0.5, connected "
          f"{export['connected']}, cells differing from the candidates "
          f"{export['cells_differing_from']}", flush=True)
    if not export["qualified"]:
        end("the export does not qualify or is not connected")

    # -- the new binary design on the development layer ------------------------------------
    same = export["same_geometry_as"][0] if export["same_geometry_as"] else None
    if same:
        cells["new/development"] = {**cells[f"{same}/development"], "design": "new",
                                    "state": f"the same geometry as {same}: its state reused"}
    else:
        solve_new(problem, "development", s_bin, yard)
    new_dev = cells["new/development"]
    record["responses"]["export_gap"] = {
        "J": rel(new_dev["J"], terminal["J_self"]), "psi": rel(new_dev["psi"], terminal["psi"]),
        "compliance": rel(new_dev["compliance"], terminal["compliance"])}
    record["against"] = {"development": compare("development")}
    flow_mesh_h = problem.flow_mesh
    del problem
    gc.collect()
    cost["peak_working_set_mb_after_development"] = peak_working_set_mb()

    # -- the check layer: identities and the candidates' states before its solves ----------
    t0 = time.perf_counter()
    fine = dual.Zhao2DDualProblem(ref.refine_spec(spec, 2), config, thermal_refinement=4,
                                  thermal_quadrature=QUADRATURE)
    cost["build_check_s"] = time.perf_counter() - t0
    print(f"built the check layer: flow {fine.flow_mesh.num_elems} el, thermal "
          f"{fine.thermal_mesh.num_elems} el ({cost['build_check_s']:.1f} s)", flush=True)
    rows = m_rec["rows"]["flow_h2"]
    record["check_layer"] = {"flow_mesh": mesh_identity(fine.flow_mesh),
                             "thermal_mesh": thermal_identity(fine)}
    require(record["check_layer"]["flow_mesh"] == rows["flow_mesh"], "the h/2 flow mesh is not R1m's")
    require(record["check_layer"]["thermal_mesh"] == rows["thermal_mesh"],
            "the h/8 thermal side is not R1m's")
    for name, source in (("r1n", "R1n's record, its new design at flow h/2, thermal h/8"),
                         ("pilot", "R1m's record, the pilot at flow h/2, thermal h/8")):
        reuse(fine, "check", name, chk[name], source, yard)
    s_f = np.asarray(ref.refine_design(flow_mesh_h, fine.flow_mesh, s_bin))
    try:
        record["check_layer"]["transfer"] = ref.check_transfer(flow_mesh_h, fine.flow_mesh, s_bin, s_f)
    except RuntimeError as exc:
        require(False, f"the new design's copy to h/2 is not a pure refinement: {exc}")
    parents = ref.parent_of_each_fine_element(flow_mesh_h, fine.thermal_mesh)
    same_material = bool(np.array_equal(np.asarray(fine.maps.density(jnp.asarray(s_f))),
                                        np.asarray(s_bin)[parents]))
    record["check_layer"]["thermal_material_is_the_parents"] = same_material
    require(same_material, "the new design's h/8 material is not its parents' material")
    checkpoint("check layer: before its solves")

    if same:
        cells["new/check"] = {**cells[f"{same}/check"], "design": "new",
                              "state": f"the same geometry as {same}: its state reused"}
    else:
        saved["new_solid_fraction_flow_h2"] = s_f
        solve_new(fine, "check", s_f, yard)
    record["against"]["check"] = compare("check")
    saved["flow_node_coords_h2"] = np.asarray(fine.flow_mesh.mesh.nodes.coords)
    del fine
    gc.collect()
    finish()

    r = record["responses"]
    print(f"\noptimisation J {r['optimisation']['J']:+.3%}  Psi {r['optimisation']['psi']:+.3%}  "
          f"C {r['optimisation']['compliance']:+.3%}  grey {r['optimisation']['grey'][0]:.4f} -> "
          f"{r['optimisation']['grey'][1]:.4f}")
    print(f"export gap   J {r['export_gap']['J']:+.3%}  Psi {r['export_gap']['psi']:+.3%}  "
          f"C {r['export_gap']['compliance']:+.3%}")
    for layer, against in record["against"].items():
        for name, v in against.items():
            print(f"new binary against {name:5s} on the {layer:11s} layer: " + (
                "not usable" if v is None else
                f"J {v['J']:+.3%}  Psi {v['psi']:+.3%}  C {v['compliance']:+.3%} [{v['kind']}]"))
    print(f"stop: {result.stop_reason}; convergence not verified")
    print(f"\nwrote {out / RECORD} and {out / FIELDS}; total "
          f"{record['cost']['wall_clock_total_s']:.0f} s")


if __name__ == "__main__":
    main()
