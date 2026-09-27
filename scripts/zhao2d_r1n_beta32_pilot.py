"""R1n: one Xu beta = 32 stage from the pilot's raw design, its binary design on two layers.

The contract proposed in the review of d56d6ab. From the R1l pilot's raw
terminal design (results/zhao2d_r1l_vp_pilot_fields.npz, "design"), with MMA's
history reinitialised -- not an exact resume -- the volume-preserving
projection, explicit, at beta = 32, eta solved at every evaluation. Everything
else is the development model's and R1l's: flow h, thermal h/4 (3x3),
alpha_max = 1e7, q_alpha = q_kappa = 0.2, the filter, the thermal residual, the
dual-mesh reference, w = 0.5, move limit 0.1.

Checked in stages. A failed gate writes the record and exits non-zero before
the next build, update or solve:

* inputs: the start design comes from the pilot's fields file as R1m hashed it,
  and is its last saved design row; the reused binary states are the ones R1l
  and R1m recorded;
* development layer (flow h, thermal h/4), before any update: the reference
  belongs to the problem and is R1m's yardstick; x300's and the pilot's saved
  binary states re-verify and reproduce R1l's Psi and C; the zero step's root
  is non-degenerate, the volume preserved and the constraint gradient affine;
* at most 30 MMA updates, every state gated by the driver, and the terminal
  evaluated at the same point;
* if the terminal is feasible, its export by R1l's rule; a terminal that is
  infeasible, or an export that does not qualify or is not connected, ends the
  run there -- a result, not a failure;
* the new binary design: one flow and one thermal solve on the development
  layer, each gated;
* check layer (flow h/2, thermal h/8), before its solves: the meshes are R1m's,
  x300's and the pilot's R1m states re-verify and reproduce R1m's Psi and C,
  and the new design's copy to h/2 keeps its fluid fraction and its tabs and
  gives the h/8 mesh its parents' material; then one flow and one thermal
  solve, each gated.

Reported apart: the map switch (the pilot's beta = 16 terminal record to the
beta = 32 zero step, the same x) and the optimisation (zero step to terminal);
then the export gap. The new qualified binary design is compared with the
pilot's and x300's on each layer, on the h/4 model's constants. If it is the
same geometry as one of them, that geometry's states are reused, not solved
again. The pilot stays the verified candidate whatever this finds.

    python scripts/zhao2d_r1n_beta32_pilot.py [--inputs results] [--out DIR] [--overwrite]
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
# R1m's report, so both layers are summarised exactly as R1m summarised them
from zhao2d_r1m_flow_check import flat, mesh_identity, thermal_identity  # noqa: E402

REFINEMENT, QUADRATURE = 4, 3
ALPHA_MAX, BETA = 1.0e7, 32.0
BUDGET, MOVE_LIMIT = 30, 0.1
GATE, ANCHOR_TOL, AFFINE_RTOL = 1e-8, 1e-12, 1e-12
RECORD = "zhao2d_r1n_beta32.json"
FIELDS = "zhao2d_r1n_fields.npz"
PROGRESS = "zhao2d_r1n_progress.json"
STACKS = "zhao2d_r1n_stacks.log"
INPUTS = ("zhao2d_r1l_baselines.json", "zhao2d_r1l_baselines_fields.npz",
          "zhao2d_r1l_vp_pilot.json", "zhao2d_r1l_vp_pilot_fields.npz",
          "zhao2d_r1m_flow_check.json", "zhao2d_r1m_fields.npz")
KEYS = ("J_self", "psi", "compliance", "constraint_g", "v_f_design_domain", "grey_fraction",
        "projection_eta", "projection_root_slope", "design_step_norm",
        "flow_residual_relative", "thermal_residual_relative")
NAMES = ("x300", "pilot")
COMPARED = ("J", "J_star", "psi", "compliance")


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
    a_rec = load_json(inputs / "zhao2d_r1l_baselines.json")
    b_rec = load_json(inputs / "zhao2d_r1l_vp_pilot.json")
    m_rec = load_json(inputs / "zhao2d_r1m_flow_check.json")
    fa = np.load(inputs / "zhao2d_r1l_baselines_fields.npz")
    fb = np.load(inputs / "zhao2d_r1l_vp_pilot_fields.npz")
    fm = np.load(inputs / "zhao2d_r1m_fields.npz")
    x_start = np.asarray(fb["design"], dtype=np.float64)
    # the reused qualified binary states: on the development layer R1l's, on the check layer R1m's
    dev = {"x300": {"s": fa["x300_solid_fraction_binary"], "press_vel": fa["x300_press_vel"],
                    "temperature": fa["x300_temperature_h4"], "rec": a_rec["designs"]["x300"]},
           "pilot": {"s": fb["solid_fraction_binary"], "press_vel": fb["binary_press_vel"],
                     "temperature": fb["binary_temperature_h4"], "rec": b_rec["export"]}}
    chk = {name: {"s": fm[f"{name}_solid_fraction_flow_h2"],
                  "press_vel": fm[f"{name}_press_vel_flow_h2"],
                  "temperature": fm[f"{name}_temperature_flow_h2_thermal_h8"],
                  "rec": m_rec["cells"][f"{name}/flow_h2"]} for name in NAMES}

    record = {
        "note": ("R1n: one stage of the volume-preserving projection at beta 32 from the R1l "
                 "pilot's raw terminal design, MMA reinitialised, on the development model "
                 "(flow h, thermal h/4); its qualified binary design evaluated there and on the "
                 "check layer (flow h/2, thermal h/8) against the pilot's and x300's, on the "
                 "h/4 model's constants. Not an exact resume, not a finding that beta 32 beats "
                 "16, not an equal-budget projection ablation."),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in INPUTS},
        "contract": {"projection": config.projection, "beta": BETA, "alpha_max": ALPHA_MAX,
                     "max_mma_updates": BUDGET, "move_limit": MOVE_LIMIT, "gate": GATE,
                     "weight": w, "continuation": "none", "fingerprint": config.fingerprint(),
                     "binary_solves_at_most": {"flow": 2, "thermal": 2}},
        "initial_design": {"file": "results/zhao2d_r1l_vp_pilot_fields.npz", "key": "design",
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
        """A saved state on this layer: re-verified and re-evaluated, never re-solved."""
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
    require(sha256_file(inputs / "zhao2d_r1l_vp_pilot_fields.npz")
            == m_rec["inputs_sha256"]["zhao2d_r1l_vp_pilot_fields.npz"],
            "the pilot's fields file is not the one R1m hashed")
    require(np.array_equal(fb["design"], fb["designs"][-1]),
            "the start design is not the pilot's last saved design row")
    for name, d in dev.items():
        rec = d["rec"]
        require(fs.digest(d["s"]) == rec["binary_sha256"], f"{name}: s is not R1l's binary design")
        require(fs.digest(d["press_vel"]) == rec["state_sha256"]["press_vel"],
                f"{name}: the h flow is not R1l's saved state")
        require(fs.digest(d["temperature"]) == rec["state_sha256"]["temperature"],
                f"{name}: the h/4 temperature is not R1l's saved state")
        require(rec.get("qualified", True) and rec.get("volume_feasible") and rec.get("gate_passed"),
                f"{name}: not a qualified, gated design in R1l's record")
    for name, d in chk.items():
        sha = d["rec"]["state_sha256"]
        require(fs.digest(d["s"]) == sha["s_flow_h2"], f"{name}: the h/2 material is not R1m's")
        require(fs.digest(d["press_vel"]) == sha["press_vel"], f"{name}: the h/2 flow is not R1m's")
        require(fs.digest(d["temperature"]) == sha["temperature"],
                f"{name}: the h/8 temperature is not R1m's")
        require(zb.cell_usable(d["rec"]), f"{name}: R1m's check-layer cell is not usable")
    checkpoint("inputs")

    # -- development layer: the reused states, then the zero step's map --------------------
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
    require(all(yard[k] == m_rec["yardstick"][k] for k in ("psi_0", "c_0", "file")),
            "the reference is not R1m's yardstick")
    record["yardstick"] = yard
    record["reporting_scale"] = {"psi_0": single.psi_0, "c_0": single.c_0,
                                 "identity": "single-mesh h reference; J* only"}
    for name in NAMES:
        reuse(problem, "development", name, dev[name],
              f"R1l's record, {name}'s qualified binary design at flow h, thermal h/4", yard)
    checkpoint("development layer: the reused states")

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
                               "constraint_gradient_affine_max_abs": scale}
    print(f"zero-step map: eta {root['eta']:.8f}, root slope {root['slope']:.3e} "
          f"(non-degenerate {root['nondegenerate']}), g0 {float(g0):+.8f}, volume drift "
          f"{drift:+.1e}, dg vs affine {gap:.1e}", flush=True)
    require(root["nondegenerate"], f"the zero step's projection root is degenerate: {root}")
    require(abs(drift) <= 1e-12, f"the projection does not preserve the volume: {drift:+.3e}")
    require(gap <= AFFINE_RTOL * scale, f"the constraint gradient is not affine: {gap:.3e}")
    checkpoint("development layer: the zero step's map")

    # -- at most thirty updates -----------------------------------------------------------
    phases = [drv.Phase("r1n-vp32", BUDGET, beta=BETA, alpha_max=ALPHA_MAX,
                        note="volume-preserving projection, beta 32, warm start from the "
                             "R1l pilot's raw terminal design")]

    def on_iteration(rec) -> None:
        progress.append({k: rec.get(k) for k in (*KEYS, "iteration", "seconds")})
        print(f"{rec['iteration']:3d}  J {rec['J_self']:.8f}  Psi {rec['psi']:.6e}  "
              f"C {rec['compliance']:.3f}  g {rec['constraint_g']:+.3e}  grey "
              f"{rec['grey_fraction']:.4f}  eta {rec['projection_eta']:.5f}  |R| "
              f"{rec['flow_residual_relative']:.1e}/{rec['thermal_residual_relative']:.1e}  "
              f"{rec['seconds']:.1f} s", flush=True)
        (out / PROGRESS).write_text(json.dumps(progress, indent=1), encoding="utf-8")

    t0 = time.perf_counter()
    try:
        result = drv.run(problem, reference, phases, move_limit=MOVE_LIMIT,
                         on_iteration=on_iteration, initial_design=x_start)
    except (r1.NotConverged, r1.DegenerateProjection) as exc:
        partial = getattr(exc, "partial", {})
        record["stop"] = {"stop_reason": type(exc).__name__, "message": str(exc),
                          "failed_iteration": partial.get("failed_iteration")}
        record["history"] = partial.get("history", progress)
        require(False, f"the updates stopped: {type(exc).__name__}: {exc}")
        checkpoint("the thirty updates")
    cost["t_updates_including_terminal_s"] = time.perf_counter() - t0

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
    record["history"], record["terminal"] = history, terminal
    record["steps"] = step_norms(designs)
    record["stop"] = {"stop_reason": result.stop_reason, "proxy_criterion": result.proxy_criterion,
                      "mma_updates": len(history), "convergence_verified": False,
                      "first_feasible_iteration": next(
                          (r_["iteration"] for r_ in evaluated if r_["volume_feasible"]), None)}
    bt = b_rec["terminal"]
    record["responses"] = {
        "map_switch": {
            "from": "the R1l pilot's terminal record, volume-preserving beta 16",
            "to": "the zero step, volume-preserving beta 32, the same x",
            **{q: rel(zero[k], bt[k]) for q, k in (("J", "J_self"), ("psi", "psi"),
                                                   ("compliance", "compliance"))},
            "g": [bt["constraint_g"], zero["constraint_g"]],
            "grey": [bt["grey_fraction"], zero["grey_fraction"]],
            "eta": [bt["projection_eta"], zero["projection_eta"]]},
        "optimisation": {
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
          f"{terminal['grey_fraction']:.4f}  stop {result.stop_reason}; last step "
          f"L2 {record['steps'][-1]['l2']:.4f}, Linf {record['steps'][-1]['linf']:.4f}", flush=True)

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
                  binary_sha256=fs.digest(s_bin))
    export["qualified"] = bool(export["volume_feasible"] and export["connected"])
    export["same_geometry_as"] = [n for n in NAMES if fs.digest(s_bin) == fs.digest(dev[n]["s"])]
    saved["solid_fraction_binary"] = s_bin
    print(f"export: t {export['t']:.10f}, {export['fluid_cells']} fluid cells, "
          f"{export['cells_changed_from_0.5']} changed from s = 0.5, connected "
          f"{export['connected']}, same geometry as {export['same_geometry_as'] or 'none'}",
          flush=True)
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

    # -- the check layer: identities and reused states before its solves -------------------
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
    for name in NAMES:
        reuse(fine, "check", name, chk[name],
              f"R1m's record, {name} at flow h/2, thermal h/8", yard)
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
    print(f"\nmap switch   J {r['map_switch']['J']:+.3%}  Psi {r['map_switch']['psi']:+.3%}  "
          f"C {r['map_switch']['compliance']:+.3%}  grey {r['map_switch']['grey'][0]:.4f} -> "
          f"{r['map_switch']['grey'][1]:.4f}")
    print(f"optimisation J {r['optimisation']['J']:+.3%}  Psi {r['optimisation']['psi']:+.3%}  "
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
