"""R1l B: thirty updates from x_30 on the volume-preserving projection at beta 16.

The contract frozen in the review of a1b4af9. From R1k's raw terminal design
x_30, with MMA's history reinitialised: the volume-preserving projection of
Xu, Cai & Cheng (2010), explicit, at beta = 16, eta solved at every
evaluation. Everything else is R1k's: alpha_max = 1e7, q_alpha = q_kappa =
0.2, the filter, the thermal residual, flow h and thermal h/4 (3x3), the
dual-mesh reference (both projections share it), w = 0.5, move limit 0.1.

The zero step is evaluated under the new map. It is infeasible: the
constraint now sees the filtered volume, which for x_30 is 0.7% over the
budget. Before any update, and without solving anything, the script checks
the root is non-degenerate, the volume is preserved, and the constraint
gradient equals its affine expression -F^T v / (0.4 V_D). At most 30 updates
follow, then the driver's same-point evaluation of the terminal design. A
proxy criterion is not convergence, and a state that breaks the bound never
counts as the best feasible design.

Three responses are reported apart:
  model switch   R1k's tanh terminal -> the new map's zero step (same x_30)
  optimisation   zero step -> terminal, both under the new map
  export gap     terminal -> its qualified binary design

If the terminal is feasible, it is exported by R1l A's rule (one threshold,
at most 40% fluid, ties together, no repair); if that design qualifies and is
connected it is solved once as a given s and compared with A's two
baselines. If not, that is reported and the script stops there.

    python scripts/zhao2d_r1l_vp_pilot.py [--inputs results] [--out DIR] [--overwrite]
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
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))
sys.path.insert(0, str(REPO / "scripts"))

from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_binary as zb  # noqa: E402
from tfopus import zhao2d_driver as drv  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file  # noqa: E402

REFINEMENT, QUADRATURE = 4, 3
ALPHA_MAX, BETA = 1.0e7, 16.0
BUDGET, MOVE_LIMIT, GATE = 30, 0.1, 1e-8
RECORD = "zhao2d_r1l_vp_pilot.json"
FIELDS = "zhao2d_r1l_vp_pilot_fields.npz"
PROGRESS = "zhao2d_r1l_vp_pilot_progress.json"
STACKS = "zhao2d_r1l_vp_pilot_stacks.log"
INPUTS = ("zhao2d_r1k_fields.npz", "zhao2d_r1k_warm_start.json", "zhao2d_r1l_baselines.json")
KEYS = ("J_self", "psi", "compliance", "constraint_g", "v_f_design_domain", "grey_fraction",
        "projection_eta", "projection_root_slope", "flow_residual_relative",
        "thermal_residual_relative")


def rel(b, a) -> float:
    return b / a - 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=pathlib.Path, default=REPO / "results")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    inputs, out = args.inputs, args.out
    if (out / RECORD).exists() and not args.overwrite:
        sys.exit(f"{out / RECORD} exists and is cited evidence; write elsewhere with "
                 "--out DIR, or pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    stacks = open(out / STACKS, "w", encoding="utf-8")
    faulthandler.dump_traceback_later(2400, repeat=True, file=stacks)  # longer than a healthy run
    t_start = time.perf_counter()

    spec = z.Zhao2DSpec()
    config = r1.R1Config(projection=r1.Projection.VOLUME_PRESERVING)
    w = config.weight
    fields = np.load(inputs / "zhao2d_r1k_fields.npz")
    r1k = json.loads((inputs / "zhao2d_r1k_warm_start.json").read_text(encoding="utf-8"))
    base = json.loads((inputs / "zhao2d_r1l_baselines.json").read_text(encoding="utf-8"))
    x30 = np.asarray(fields["design"], dtype=np.float64)

    record = {
        "note": ("R1l B: 30 updates from R1k's raw x_30 on the volume-preserving projection "
                 "(Xu, Cai & Cheng 2010) at beta 16, eta solved every evaluation; MMA "
                 "reinitialised; everything else R1k's. Judged on the qualified binary design "
                 "against R1l A's baselines, not on the continuous J or the grey fraction."),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in INPUTS},
        "contract": {"projection": config.projection, "beta": BETA, "alpha_max": ALPHA_MAX,
                     "max_mma_updates": BUDGET, "move_limit": MOVE_LIMIT, "gate": GATE,
                     "weight": w, "continuation": "none", "fingerprint": config.fingerprint()},
        "initial_design": {"file": "results/zhao2d_r1k_fields.npz", "key": "design",
                           "sha256_float64": fs.digest(x30)},
    }
    here = pathlib.Path(__file__).resolve()
    record["provenance"]["source_sha256"][here.relative_to(REPO).as_posix()] = sha256_file(here)

    t0 = time.perf_counter()
    problem = dual.Zhao2DDualProblem(spec, config, REFINEMENT, QUADRATURE)
    t_build = time.perf_counter() - t0
    print(f"built: flow {problem.flow_mesh.num_elems} el, thermal "
          f"{problem.thermal_mesh.num_elems} el ({t_build:.1f} s)", flush=True)
    ref_path = dual.reference_file(REFINEMENT, QUADRATURE)
    reference = dual.load_reference(problem, ref_path)
    single = r1.load_reference(spec, config)
    record["reference"] = {"file": ref_path.relative_to(REPO).as_posix(),
                           "sha256": sha256_file(ref_path),
                           "psi_0": reference.psi_0, "c_0": reference.c_0}

    # -- the zero step's map, checked without solving anything --------------------------
    xj = jnp.asarray(x30)
    design_mask = np.asarray(problem.flow_mesh.design_mask)
    areas = np.asarray(problem.flow_mesh.elem_area)
    a_d = jnp.asarray(areas[problem.design_elements])

    def constraint(v):
        return problem.fluid_fraction(v, BETA) / config.max_fluid_fraction - 1.0

    g0, dg0 = jax.value_and_grad(constraint)(xj)
    affine = -(problem.filter_matrix.T @ a_d) / (config.max_fluid_fraction * float(a_d.sum()))
    root = problem.projection_root(xj, BETA)
    record["zero_step_map"] = {
        "root": root,
        "g0": float(g0),
        "volume_projected_minus_filtered": float(problem.fluid_fraction(xj, BETA)
                                                 - problem.fluid_fraction(xj, 0.0)),
        "constraint_gradient_vs_affine_max_abs": float(jnp.max(jnp.abs(dg0 - affine))),
        "constraint_gradient_affine_max_abs": float(jnp.max(jnp.abs(affine))),
    }
    print(f"zero-step map: eta {root['eta']:.8f}, root slope {root['slope']:.3e} "
          f"(non-degenerate {root['nondegenerate']}), g0 {float(g0):+.8f}, "
          f"volume drift {record['zero_step_map']['volume_projected_minus_filtered']:+.1e}, "
          f"dg vs affine {record['zero_step_map']['constraint_gradient_vs_affine_max_abs']:.1e}",
          flush=True)
    if not root["nondegenerate"]:
        (out / RECORD).write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
        sys.exit("the zero step's projection root is degenerate; nothing optimised")

    # -- the thirty updates ---------------------------------------------------------------
    phases = [drv.Phase("r1l-vp16", BUDGET, beta=BETA, alpha_max=ALPHA_MAX,
                        note="volume-preserving projection, beta 16, warm start from x_30")]
    progress = []

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
                         on_iteration=on_iteration, initial_design=x30)
    except (r1.NotConverged, r1.DegenerateProjection) as exc:
        partial = getattr(exc, "partial", {})
        record["stop"] = {"stop_reason": type(exc).__name__, "message": str(exc),
                          "failed_iteration": partial.get("failed_iteration")}
        record["history"] = partial.get("history", progress)
        (out / RECORD).write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
        faulthandler.cancel_dump_traceback_later()
        stacks.close()
        sys.exit(f"stopped: {type(exc).__name__}: {exc}")
    t_run = time.perf_counter() - t0

    history, terminal = result.history, result.terminal
    zero = history[0]
    evaluated = [*history, terminal]
    for rec_ in evaluated:
        rec_["J_star"] = float(dual.reporting_objective(rec_["psi"], rec_["compliance"], single, w))
        rec_["volume_feasible"] = zb.volume_feasible(rec_["v_f_design_domain"],
                                                     config.max_fluid_fraction)
    feasible = [r_ for r_ in evaluated if r_["volume_feasible"]]
    best = min(feasible, key=lambda r_: r_["J_self"]) if feasible else None
    first_feasible = next((r_["iteration"] for r_ in evaluated if r_["volume_feasible"]), None)
    record["history"] = history
    record["terminal"] = terminal
    record["stop"] = {"stop_reason": result.stop_reason, "proxy_criterion": result.proxy_criterion,
                      "mma_updates": len(history), "convergence_verified": False,
                      "first_feasible_iteration": first_feasible}
    rt = r1k["terminal"]
    record["responses"] = {
        "model_switch": {
            "from": "R1k tanh beta 8 terminal record", "to": "zero step, volume-preserving beta 16",
            "J": rel(zero["J_self"], rt["J_self"]), "psi": rel(zero["psi"], rt["psi"]),
            "compliance": rel(zero["compliance"], rt["compliance"]),
            "g": [rt["constraint_g"], zero["constraint_g"]],
            "grey": [rt["grey_fraction"], zero["grey_fraction"]]},
        "optimisation": {
            "J": rel(terminal["J_self"], zero["J_self"]), "psi": rel(terminal["psi"], zero["psi"]),
            "compliance": rel(terminal["compliance"], zero["compliance"]),
            "g": [zero["constraint_g"], terminal["constraint_g"]],
            "grey": [zero["grey_fraction"], terminal["grey_fraction"]]},
    }
    record["best_feasible_evaluated"] = (
        None if best is None else {"iteration": best["iteration"],
                                   "is_terminal": bool(best.get("terminal")),
                                   "J_self": best["J_self"], "psi": best["psi"],
                                   "compliance": best["compliance"],
                                   "constraint_g": best["constraint_g"]})

    designs = np.vstack([result.designs, np.asarray(result.design)[None, :]])
    saved = dict(initial_design=result.initial_design,
                 initial_solid_fraction=result.initial_solid_fraction,
                 initial_press_vel=result.initial_press_vel,
                 initial_temperature=result.initial_temperature,
                 designs=designs, design=result.design, solid_fraction=result.solid_fraction,
                 press_vel=result.press_vel, temperature=result.temperature,
                 elem_centres=np.asarray(problem.flow_mesh.elem_centres))

    # -- export the terminal, if it is feasible ------------------------------------------
    export = {"terminal_volume_feasible": bool(terminal["volume_feasible"])}
    if terminal["volume_feasible"]:
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
        saved["solid_fraction_binary"] = s_bin
        if export["qualified"]:
            t0 = time.perf_counter()
            pv, temp, alpha, kappa = problem.solve_states(jnp.asarray(s_bin), ALPHA_MAX)
            norms = problem.residual_norms_at(pv, temp, alpha, kappa)
            psi = float(problem.flow.dissipated_power(pv, alpha))
            c = float(problem.thermal.thermal_compliance(temp, problem.thermal_velocity(pv), kappa))
            t_arr = np.asarray(temp)
            export.update(
                analysed=True, seconds=time.perf_counter() - t0, psi=psi, compliance=c,
                J=w * psi / reference.psi_0 + (1 - w) * c / reference.c_0,
                J_star=float(dual.reporting_objective(psi, c, single, w)),
                T_max=float(t_arr.max()), negative_nodes=int(np.count_nonzero(t_arr < 0.0)),
                residual_flow=norms["flow"], residual_thermal=norms["thermal"],
                gate_passed=zb.gate_passed(norms, GATE),
                state_sha256={"press_vel": fs.digest(pv), "temperature": fs.digest(t_arr)})
            saved["binary_press_vel"] = np.asarray(pv)
            saved["binary_temperature_h4"] = t_arr
            record["responses"]["export_gap"] = {
                "J": rel(export["J"], terminal["J_self"]), "psi": rel(psi, terminal["psi"]),
                "compliance": rel(c, terminal["compliance"])}
            against = {}
            for name in ("x300", "x30"):
                b = base["designs"][name]
                kind = zb.comparison_kind({**export, "volume_feasible": export["volume_feasible"]}, b)
                against[name] = None if kind is None else {
                    "kind": kind, **{q: rel(export[q], b[q]) for q in ("J", "J_star", "psi", "compliance")}}
            record["binary_against_baselines"] = against
    record["export"] = export
    record["checks"] = zb.check_failures({"terminal_binary": export} if export.get("analysed") else {})
    record["cost"] = {"t_build_s": t_build, "t_run_including_terminal_s": t_run,
                      "t_iteration_s": [r_["seconds"] for r_ in history],
                      "peak_working_set_mb": peak_working_set_mb(),
                      "wall_clock_total_s": time.perf_counter() - t_start}

    np.savez_compressed(out / FIELDS, **saved)
    faulthandler.cancel_dump_traceback_later()
    stacks.close()
    if (out / STACKS).stat().st_size == 0:
        (out / STACKS).unlink()
    (out / RECORD).write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
    (out / PROGRESS).unlink(missing_ok=True)

    r = record["responses"]
    print(f"\nterminal: J {terminal['J_self']:.8f}  Psi {terminal['psi']:.6e}  C "
          f"{terminal['compliance']:.3f}  g {terminal['constraint_g']:+.3e}  grey "
          f"{terminal['grey_fraction']:.4f}  (first feasible iteration: {first_feasible})")
    print(f"model switch  J {r['model_switch']['J']:+.3%}  Psi {r['model_switch']['psi']:+.3%}  "
          f"C {r['model_switch']['compliance']:+.3%}")
    print(f"optimisation  J {r['optimisation']['J']:+.3%}  Psi {r['optimisation']['psi']:+.3%}  "
          f"C {r['optimisation']['compliance']:+.3%}")
    if "export_gap" in r:
        print(f"export gap    J {r['export_gap']['J']:+.3%}  Psi {r['export_gap']['psi']:+.3%}  "
              f"C {r['export_gap']['compliance']:+.3%}")
        for name, v in record["binary_against_baselines"].items():
            if v:
                print(f"binary terminal against {name}'s baseline ({v['kind']}): J {v['J']:+.3%}  "
                      f"Psi {v['psi']:+.3%}  C {v['compliance']:+.3%}")
    else:
        print(f"no qualified binary terminal: {export}")
    print(f"stop: {result.stop_reason}; convergence not verified")
    print(f"\nwrote {out / RECORD} and {out / FIELDS}; total "
          f"{record['cost']['wall_clock_total_s']:.0f} s")
    if any(record["checks"].values()):
        sys.exit(f"CHECKS FAILED: {record['checks']}")


if __name__ == "__main__":
    main()
