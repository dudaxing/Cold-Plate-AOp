"""R1k: a bounded warm start on the development model -- not a resumed R1d.

Model, exactly R1j's: design and flow on h = 1e-4 (2x2), temperature on h/4
(3x3), normalised by its own frozen reference
(tfopus/zhao2d_reference_dual_r4q3_v1.json), w = 0.5. alpha_max = 1e7 and
beta = 8 are held fixed -- no continuation is restarted -- so a change in J is
the optimiser's, not a change of model.

Start: R1d's saved raw design x_300 (the 5000 design variables, not the
5200-element solid fraction), handed to the driver as `initial_design`. MMA's
history starts fresh, so this is a new run from a saved design, not R1d
resumed; and R1d optimised a different objective (thermal mesh h, the
single-mesh reference), so its J is not comparable with this one's.

Budget: at most 30 MMA updates at move_limit 0.1, every state through the
driver's 1e-8 residual gate, then the same-point evaluation of the terminal
design that the driver always makes. The run stops at the budget, at a failed
gate, or when an upstream proxy criterion fires -- step_tol, or upstream's
mixed-point KKT residual -- and a proxy firing is reported as that: no
same-point convergence check is made here, so nothing is called converged.
Nothing is added automatically, whatever J does.

Saved: the initial and terminal x, s, u/p and T; the design of every iterate,
so any iterate quoted later has its own design; per iterate J, J*, raw Psi and
C, g, v_f, residuals and step sizes; the actual stop reason.

    python scripts/zhao2d_r1k_warm_start.py [--inputs results] [--out DIR]
        [--overwrite]
"""

from __future__ import annotations

import pathlib
import sys

# Default the BLAS thread count BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import faulthandler
import hashlib
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
from tfopus import zhao2d_driver as drv  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file  # noqa: E402

REFINEMENT, QUADRATURE = 4, 3
ALPHA_MAX, BETA = 1.0e7, 8.0
BUDGET, MOVE_LIMIT = 30, 0.1
RECORD = "zhao2d_r1k_warm_start.json"
FIELDS = "zhao2d_r1k_fields.npz"
PROGRESS = "zhao2d_r1k_progress.json"
STACKS = "zhao2d_r1k_stacks.log"
# R1j's baseline at x_300 through the same driver, model and reference
R1J_BASELINE = {"J_self": 1.116472423534761, "psi": 0.014351143070216229,
                "compliance": 36180.16029596453,
                "constraint_g": -5.3831532892845146e-05}
KEYS = ("J_self", "psi", "compliance", "constraint_g", "v_f_design_domain",
        "grey_fraction", "flow_residual_relative", "thermal_residual_relative")


def digest(a) -> str:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def row(rec, j_star) -> str:
    return (f"{rec['iteration']:3d}  J {rec['J_self']:.8f}  J* {j_star:.8f}  "
            f"Psi {rec['psi']:.6e}  C {rec['compliance']:.3f}  "
            f"g {rec['constraint_g']:+.2e}  grey {rec['grey_fraction']:.4f}  "
            f"|R| {max(rec['flow_residual_relative'], rec['thermal_residual_relative']):.1e}"
            + (f"  {rec['seconds']:.1f} s" if "seconds" in rec else ""))


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
    # A hang leaves every thread's Python stack behind, every 10 minutes.
    stacks = open(out / STACKS, "w", encoding="utf-8")
    faulthandler.dump_traceback_later(600, repeat=True, file=stacks)
    t_start = time.perf_counter()
    memory = {}

    spec, config = z.Zhao2DSpec(), r1.R1Config()
    w = config.weight
    meta = json.loads((inputs / "zhao2d_r1d_main.json").read_text(encoding="utf-8"))
    fields = np.load(inputs / "zhao2d_r1d_main_fields.npz")
    if (meta["final_alpha_max"], meta["final_beta"]) != (ALPHA_MAX, BETA):
        sys.exit(f"R1d ended at alpha_max {meta['final_alpha_max']}, beta "
                 f"{meta['final_beta']}, not the {ALPHA_MAX:g} / {BETA:g} this run holds")
    x300 = np.asarray(fields["design"], dtype=np.float64)

    record = {
        "note": (
            "R1k: a warm start from R1d's x_300 on the flow h / thermal h/4 "
            "development model, alpha_max = 1e7 and beta = 8 held fixed, at most 30 "
            "MMA updates. MMA's history starts fresh: a new run from a saved design, "
            "not R1d resumed. J uses this model's own frozen reference; J* the "
            "single-mesh one, as a reporting scale only. No same-point convergence "
            "check is made, so nothing here is called converged."
        ),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in
                          ("zhao2d_r1d_main.json", "zhao2d_r1d_main_fields.npz")},
        "contract": {"alpha_max": ALPHA_MAX, "beta": BETA, "weight": w,
                     "max_mma_updates": BUDGET, "move_limit": MOVE_LIMIT,
                     "residual_gate": 1e-8, "continuation": "none"},
        "model": {"h_flow": spec.element_size, "flow_quadrature": 2,
                  "thermal_refinement": REFINEMENT, "thermal_quadrature": QUADRATURE},
        "initial_design": {"file": "results/zhao2d_r1d_main_fields.npz",
                           "key": "design", "shape": list(x300.shape),
                           "sha256_float64": digest(x300),
                           "min": float(x300.min()), "max": float(x300.max())},
    }
    here = pathlib.Path(__file__).resolve()
    record["provenance"]["source_sha256"][here.relative_to(REPO).as_posix()] = sha256_file(here)

    t0 = time.perf_counter()
    problem = dual.Zhao2DDualProblem(spec, config, REFINEMENT, QUADRATURE)
    t_build = time.perf_counter() - t0
    memory["after_build"] = peak_working_set_mb()
    print(f"built: flow {problem.flow_mesh.num_elems} el, thermal "
          f"{problem.thermal_mesh.num_elems} el ({t_build:.1f} s)", flush=True)

    ref_path = dual.reference_file(REFINEMENT, QUADRATURE)
    reference = dual.load_reference(problem, ref_path)
    single = r1.load_reference(spec, config)
    record["reference"] = {"file": ref_path.relative_to(REPO).as_posix(),
                           "sha256": sha256_file(ref_path),
                           "psi_0": reference.psi_0, "c_0": reference.c_0,
                           "single_mesh_scale": {"psi_0": single.psi_0,
                                                 "c_0": single.c_0}}

    def j_star(rec) -> float:
        return float(dual.reporting_objective(rec["psi"], rec["compliance"], single, w))

    s_map = np.asarray(problem.solid_fraction(jnp.asarray(x300), BETA))
    record["design_map_vs_saved_s_max_abs"] = float(
        np.max(np.abs(s_map - np.asarray(fields["solid_fraction"]))))

    phases = [drv.Phase("r1k-fixed", BUDGET, beta=BETA, alpha_max=ALPHA_MAX,
                        note="alpha_max and beta held fixed; warm start from x_300")]
    progress = []

    def on_iteration(rec) -> None:
        progress.append({k: rec[k] for k in (*KEYS, "iteration", "seconds")})
        print(row(rec, j_star(rec)), flush=True)
        (out / PROGRESS).write_text(json.dumps(progress, indent=1), encoding="utf-8")

    t0 = time.perf_counter()
    try:
        result = drv.run(problem, reference, phases, move_limit=MOVE_LIMIT,
                         on_iteration=on_iteration, initial_design=x300)
    except r1.NotConverged as exc:
        partial = getattr(exc, "partial", {})
        record["stop"] = {"stop_reason": "gate_failed", "message": str(exc),
                          "failed_iteration": partial.get("failed_iteration")}
        record["history"] = partial.get("history", [])
        np.savez_compressed(out / FIELDS, designs=partial.get("designs"),
                            failed_design=partial.get("failed_design"),
                            initial_design=x300)
        (out / RECORD).write_text(json.dumps(record, indent=2, default=float),
                                  encoding="utf-8")
        faulthandler.cancel_dump_traceback_later()
        stacks.close()
        sys.exit(f"a state failed the residual gate at iteration "
                 f"{partial.get('failed_iteration')}: {exc}; the run so far is in {out / RECORD}")
    t_run = time.perf_counter() - t0
    memory["after_run"] = peak_working_set_mb()

    history, terminal = result.history, result.terminal
    first = history[0]
    record["zero_step_vs_r1j_baseline"] = {
        "J_self_relative": first["J_self"] / R1J_BASELINE["J_self"] - 1.0,
        "psi_relative": first["psi"] / R1J_BASELINE["psi"] - 1.0,
        "compliance_relative": first["compliance"] / R1J_BASELINE["compliance"] - 1.0,
        "constraint_g_absolute": first["constraint_g"] - R1J_BASELINE["constraint_g"],
        "initial_design_is_x300": bool(np.array_equal(result.initial_design, x300)),
        "first_evaluated_design_is_x300": bool(np.array_equal(result.designs[0], x300)),
    }

    designs = np.vstack([result.designs, np.asarray(result.design)[None, :]])
    steps = np.diff(designs, axis=0)
    for rec in history:
        rec["J_star"] = j_star(rec)
    terminal["J_star"] = j_star(terminal)
    record["history"] = history
    record["terminal"] = terminal
    record["steps"] = {
        "note": ("step k is designs[k+1] - designs[k]; the last is to the terminal "
                 "design. Upstream's change_design_var (design_step_norm in the "
                 "history) is computed from the second update on; before that it "
                 "keeps its initial 1000.0."),
        "two_norm": np.linalg.norm(steps, axis=1).tolist(),
        "max_abs": np.max(np.abs(steps), axis=1).tolist(),
        "terminal_minus_x300_two_norm": float(np.linalg.norm(designs[-1] - x300)),
        "terminal_minus_x300_max_abs": float(np.max(np.abs(designs[-1] - x300))),
    }

    evaluated = [*history, terminal]
    feasible = [r for r in evaluated if r["constraint_g"] <= 0.0]
    best = min(feasible, key=lambda r: r["J_self"]) if feasible else None

    def temps(t):
        t = np.asarray(t)
        return {"T_max": float(t.max()), "T_min": float(t.min()),
                "negative_nodes": int(np.count_nonzero(t < 0.0))}

    record["comparison"] = {
        "initial": {**{k: first[k] for k in KEYS}, "J_star": first["J_star"],
                    **temps(result.initial_temperature)},
        "terminal": {**{k: terminal[k] for k in KEYS}, "J_star": terminal["J_star"],
                     **temps(result.temperature)},
        "terminal_over_initial_minus_1": {
            k: terminal[k] / first[k] - 1.0 for k in ("J_self", "psi", "compliance")},
        "lowest_J_among_feasible_evaluated": (
            None if best is None else
            {"iteration": best["iteration"], "is_terminal": bool(best.get("terminal")),
             "J_self": best["J_self"], "psi": best["psi"],
             "compliance": best["compliance"], "constraint_g": best["constraint_g"],
             "design": f"{FIELDS}: designs[{best['iteration']}]"}),
        "J_self_monotone_non_increasing": bool(all(
            b["J_self"] <= a["J_self"] for a, b in zip(evaluated, evaluated[1:]))),
    }
    record["stop"] = {
        "stop_reason": result.stop_reason,
        "proxy_criterion": result.proxy_criterion,
        "proxy_fired_at_final_stage": result.proxy_fired_at_final_stage,
        "mma_updates": len(history),
        "convergence_verified": False,
        "note": ("phase_end with one fixed phase of 30 means the budget was used; "
                 "proxy_criterion names an upstream proxy, not a convergence check; "
                 "no same-point convergence check is made in R1k."),
    }
    record["cost"] = {
        "t_build_s": t_build,
        "t_iteration_s": [r["seconds"] for r in history],
        "t_run_including_terminal_s": t_run,
        "peak_working_set_mb_cumulative": memory,
        "wall_clock_total_s": time.perf_counter() - t_start,
    }

    np.savez_compressed(
        out / FIELDS,
        initial_design=result.initial_design,
        initial_solid_fraction=result.initial_solid_fraction,
        initial_press_vel=result.initial_press_vel,
        initial_temperature=result.initial_temperature,
        designs=designs,
        design=result.design,
        solid_fraction=result.solid_fraction,
        press_vel=result.press_vel,
        temperature=result.temperature,
        elem_centres=np.asarray(problem.flow_mesh.elem_centres),
    )
    faulthandler.cancel_dump_traceback_later()
    stacks.close()
    if (out / STACKS).stat().st_size == 0:
        (out / STACKS).unlink()
    (out / RECORD).write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
    (out / PROGRESS).unlink(missing_ok=True)

    print(f"\nterminal (same-point evaluation of the design MMA last produced):")
    print(row(terminal, terminal["J_star"]))
    c = record["comparison"]
    print(f"terminal / initial - 1: J {c['terminal_over_initial_minus_1']['J_self']:+.4%}, "
          f"Psi {c['terminal_over_initial_minus_1']['psi']:+.4%}, "
          f"C {c['terminal_over_initial_minus_1']['compliance']:+.4%}")
    print(f"stop: {result.stop_reason}"
          + (f" ({result.proxy_criterion}, a proxy)" if result.proxy_criterion else "")
          + f" after {len(history)} updates; convergence not verified")
    print(f"zero step vs R1j: J {record['zero_step_vs_r1j_baseline']['J_self_relative']:+.1e}, "
          f"C {record['zero_step_vs_r1j_baseline']['compliance_relative']:+.1e}")
    print(f"\nwrote {out / RECORD} and {out / FIELDS}; total "
          f"{record['cost']['wall_clock_total_s']:.0f} s")


if __name__ == "__main__":
    main()
