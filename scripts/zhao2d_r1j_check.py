"""R1j: the dual-mesh development model through the driver -- checked, not optimised.

Model: design and flow on h = 1e-4 (2x2), temperature on h/4 (3x3), normalised
by its own frozen reference (tfopus/zhao2d_reference_dual_r4q3_v1.json). At the
R1d design x_300, alpha_max = 1e7, beta = 8:

  1. references: this model's loads through its identity check; the
     single-mesh one and another thermal mesh's are refused at the driver's
     entry -- `evaluate` and `run` -- before any solve;
  2. one baseline evaluation through the driver -- J, g, their gradients, the
     states and the residual gate from one solve -- checked against the direct
     API and the reporter on the SAME states, and against R1g's level-4 record
     of the same design, flow and thermal mesh;
  3. directional derivatives of Psi, C, g and J along two directions fixed
     before any result was seen, against central differences: 2 directions x
     3 steps x 2 signs = 12 perturbed evaluations through the driver, each
     gated on the state it returned.

The criterion is fixed here, before the run, and is the small-mesh one: for
each direction and quantity, the best of the three steps agrees to 1e-5
relative. Absolute differences are reported beside it. Each direction is random
signs (seeds 11 and 12) on the design variables at least the largest step away
from both bounds, zero on the rest, so every perturbed design stays in [0, 1]
without clipping anything along the difference.

Psi and C need their own derivatives; they come from two scalar reverse passes
through the direct API at the baseline (two more forward solves and two
adjoints), which also cross-check the driver's gradient of J. No MMA update, no
other mesh, no gradient at h/8.

    python scripts/zhao2d_r1j_check.py [--out DIR] [--inputs results]
"""

from __future__ import annotations

import pathlib
import sys

# Default the BLAS thread count BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import dataclasses
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

from tfopus import materials  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_driver as drv  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file  # noqa: E402

REFINEMENT, QUADRATURE = 4, 3
STEPS = (1e-4, 1e-5, 1e-6)
SEEDS = (11, 12)
TOL = 1e-5
RECORD = "zhao2d_r1j_check.json"
# zhao2d_r1g_dual.json: the same design, flow and thermal mesh, level 4
R1G_C4 = 36180.16029596454
R1G_PSI = 0.014351143070216225
QUANTITIES = ("psi", "compliance", "constraint_g", "J_self")


def with_identity(values, **thermal_mesh):
    identity = json.loads(values.identity)
    identity["thermal_mesh"].update(thermal_mesh)
    return dataclasses.replace(values, identity=json.dumps(identity, sort_keys=True))


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
    # If this run hangs as attempt 1 did, leave the Python stacks of every thread
    # behind, every 10 minutes, instead of a silent log. A normal run takes ~6.
    stacks = open(out / "zhao2d_r1j_check_stacks.log", "w", encoding="utf-8")
    faulthandler.dump_traceback_later(600, repeat=True, file=stacks)
    t_start = time.perf_counter()
    memory = {}

    spec, config = z.Zhao2DSpec(), r1.R1Config(projection=r1.Projection.TANH)  # the projection its record used
    meta = json.loads((inputs / "zhao2d_r1d_main.json").read_text(encoding="utf-8"))
    fields = np.load(inputs / "zhao2d_r1d_main_fields.npz")
    alpha_max, beta = meta["final_alpha_max"], meta["final_beta"]
    x = jnp.asarray(fields["design"])
    w = config.weight

    record = {
        "note": (
            "R1j: the flow h / thermal h/4 development model through the driver, at "
            "x_300. A check of wiring, normalisation and gradient -- no MMA update. "
            "J uses this model's own frozen reference; J* the single-mesh one, as "
            "a reporting scale only."
        ),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in
                          ("zhao2d_r1d_main.json", "zhao2d_r1d_main_fields.npz")},
        "alpha_max": alpha_max,
        "beta": beta,
        "model": {"h_flow": spec.element_size, "flow_quadrature": 2,
                  "thermal_refinement": REFINEMENT, "thermal_quadrature": QUADRATURE},
        "criterion": {"steps": list(STEPS), "best_step_relative_tolerance": TOL,
                      "seeds": list(SEEDS), "fixed_before_the_run": True},
    }
    here = pathlib.Path(__file__).resolve()
    record["provenance"]["source_sha256"][here.relative_to(REPO).as_posix()] = sha256_file(here)

    t0 = time.perf_counter()
    problem = dual.Zhao2DDualProblem(spec, config, REFINEMENT, QUADRATURE)
    t_build = time.perf_counter() - t0
    memory["after_build"] = peak_working_set_mb()
    print(f"built: flow {problem.flow_mesh.num_elems} el, thermal "
          f"{problem.thermal_mesh.num_elems} el ({t_build:.1f} s)", flush=True)

    # -- 1. references -------------------------------------------------------------
    ref_path = dual.reference_file(REFINEMENT, QUADRATURE)
    reference = dual.load_reference(problem, ref_path)
    single = r1.load_reference(spec, config)
    record["reference"] = {
        "file": ref_path.relative_to(REPO).as_posix(),
        "sha256": sha256_file(ref_path),
        "psi_0": reference.psi_0,
        "c_0": reference.c_0,
        "identity": json.loads(reference.identity),
        "single_mesh_scale": {"psi_0": single.psi_0, "c_0": single.c_0},
    }

    calls = []
    solve = problem.solve_states
    problem.solve_states = lambda *a, **k: (calls.append(1), solve(*a, **k))[1]
    refusals = {}
    wrong = {"single-mesh reference": single,
             "thermal refinement 2": with_identity(reference, refinement=2),
             "thermal quadrature 2": with_identity(reference, quadrature=2)}
    for name, bad in wrong.items():
        for entry in ("evaluate", "run"):
            try:
                if entry == "evaluate":
                    drv.evaluate(problem, bad, x, alpha_max, beta)
                else:
                    drv.run(problem, bad, drv.r1d_schedule(spec, 6))
            except ValueError as exc:
                refusals[f"{name} / {entry}"] = str(exc).splitlines()[0]
            else:
                sys.exit(f"{name} was ACCEPTED by drv.{entry}; stopping")
    del problem.solve_states
    record["refusals"] = {"refused": refusals, "solves_before_refusing": len(calls)}
    if calls:
        sys.exit("a refused reference got as far as a solve; stopping")
    print(f"references: this model's loaded; {len(refusals)} refusals at the driver "
          "entry, no solve before any of them", flush=True)

    # -- 2. the baseline, through the driver ----------------------------------------
    s_map = np.asarray(problem.solid_fraction(x, beta))
    t0 = time.perf_counter()
    base, (s, press_vel, temperature), dj, dg = drv.evaluate(
        problem, reference, x, alpha_max, beta)
    t_base = time.perf_counter() - t0
    memory["after_baseline"] = peak_working_set_mb()

    pv, temp = jnp.asarray(press_vel), jnp.asarray(temperature)
    alpha = materials.brinkman_penalty(jnp.asarray(s), za.build_material(spec, alpha_max))
    kappa_t = problem.thermal_conductivity(jnp.asarray(s), alpha_max)
    psi_state = float(problem.flow.dissipated_power(pv, alpha))
    c_state = float(problem.thermal.thermal_compliance(
        temp, problem.thermal_velocity(pv), kappa_t))
    report = fs.cell_report(problem, s, pv, temp, alpha_max, single)
    g_direct = float(problem.fluid_fraction(x, beta) / config.max_fluid_fraction - 1.0)
    consistency = {
        "design_map_vs_saved_s_max_abs": float(np.max(np.abs(
            s_map - np.asarray(fields["solid_fraction"])))),
        "psi_record_vs_state": base["psi"] / psi_state - 1.0,
        "c_record_vs_state": base["compliance"] / c_state - 1.0,
        "psi_record_vs_reporter": base["psi"] / report["psi"] - 1.0,
        "c_record_vs_reporter": base["compliance"] / report["compliance"] - 1.0,
        "g_record_vs_direct": base["constraint_g"] - g_direct,
        "residuals_record": {"flow": base["flow_residual_relative"],
                             "thermal": base["thermal_residual_relative"]},
        "residuals_reporter": report["residual_relative"],
        "c_vs_r1g_level4": base["compliance"] / R1G_C4 - 1.0,
        "psi_vs_r1g": base["psi"] / R1G_PSI - 1.0,
    }
    record["baseline"] = {
        "record": base,
        "j_star_single_mesh_scale": float(dual.reporting_objective(
            base["psi"], base["compliance"], single, w)),
        "consistency": consistency,
        "reporter_identities": {k: v.get("closure_relative")
                                for k, v in report["identities"].items()},
        "t_value_and_gradient_s": t_base,
        "gradient_norm_J": float(jnp.linalg.norm(dj)),
        "gradient_norm_g": float(jnp.linalg.norm(dg)),
    }
    print(f"baseline: J {base['J_self']:.10f}, Psi {base['psi']:.12f}, C "
          f"{base['compliance']:.6f}, g {base['constraint_g']:+.3e}; |R| "
          f"{base['flow_residual_relative']:.1e} / {base['thermal_residual_relative']:.1e}; "
          f"C vs R1g level 4 {consistency['c_vs_r1g_level4']:+.1e} ({t_base:.1f} s)",
          flush=True)

    # Psi and C each, through the direct API: two scalar reverse passes, the
    # pattern the baseline itself uses. (Attempt 1 linearised (Psi, C) once with
    # jax.vjp and hung in the adjoint pass with no CPU for 1 h 50 min; see
    # results/zhao2d_r1j_check_attempt1_hang.log. The cause was not identified.)
    def component(i):
        def f(v):
            return problem.metrics(problem.solid_fraction(v, beta), alpha_max)[i]
        return f

    t0 = time.perf_counter()
    grads_pc = []
    for i, name in enumerate(("Psi", "C")):
        print(f"  gradient of {name} alone ...", flush=True)
        value, grad = jax.value_and_grad(component(i))(x)
        grads_pc.append((float(value), grad))
    t_vjp = time.perf_counter() - t0
    (psi_v, d_psi), (c_v, d_c) = grads_pc
    composed = w * d_psi / reference.psi_0 + (1.0 - w) * d_c / reference.c_0
    record["component_gradients"] = {
        "method": "two scalar reverse passes through problem.metrics",
        "t_two_value_and_gradient_s": t_vjp,
        "psi_c_vs_baseline": [psi_v / base["psi"] - 1.0, c_v / base["compliance"] - 1.0],
        "driver_dJ_vs_composed_relative": float(
            jnp.linalg.norm(dj - composed) / jnp.linalg.norm(composed)),
    }
    memory["after_linearisation"] = peak_working_set_mb()
    print(f"components: driver dJ vs w dPsi/Psi0 + (1-w) dC/C0: relative "
          f"{record['component_gradients']['driver_dJ_vs_composed_relative']:.1e} "
          f"({t_vjp:.1f} s)", flush=True)

    # -- 3. directional derivatives against central differences ----------------------
    x_np = np.asarray(x)
    free = (x_np >= STEPS[0]) & (x_np <= 1.0 - STEPS[0])
    grads = {"psi": d_psi, "compliance": d_c, "constraint_g": dg, "J_self": dj}
    directions = []
    for seed in SEEDS:
        d = np.random.default_rng(seed).choice([-1.0, 1.0], x_np.size) * free
        directions.append({
            "seed": seed,
            "active_components": int(np.count_nonzero(d)),
            "sha256": fs.digest(d),
            "ad": {q: float(jnp.dot(grads[q], jnp.asarray(d))) for q in QUANTITIES},
            "_d": d,
        })
    record["design_variables"] = {"total": int(x_np.size),
                                  "at_least_largest_step_from_both_bounds": int(free.sum())}

    evaluations = []
    for k, direction in enumerate(directions):
        d = jnp.asarray(direction["_d"])
        direction["fd"] = []
        for h in STEPS:
            sides = {}
            for sign in (1.0, -1.0):
                t0 = time.perf_counter()
                rec = drv.evaluate(problem, reference, x + sign * h * d, alpha_max, beta,
                                   gradient=False)[0]
                sides[sign] = rec
                evaluations.append({
                    "direction": k, "step": h, "sign": sign,
                    "seconds": time.perf_counter() - t0,
                    "residual_flow": rec["flow_residual_relative"],
                    "residual_thermal": rec["thermal_residual_relative"],
                    **{q: rec[q] for q in QUANTITIES},
                })
            row = {"step": h}
            for q in QUANTITIES:
                fd = (sides[1.0][q] - sides[-1.0][q]) / (2.0 * h)
                ad = direction["ad"][q]
                row[q] = {"fd": fd, "abs_error": abs(ad - fd),
                          "rel_error": abs(ad - fd) / abs(fd) if fd != 0 else None}
            direction["fd"].append(row)
            print(f"  direction {k} step {h:.0e}: "
                  + "  ".join(f"{q} {row[q]['rel_error']:.1e}" for q in QUANTITIES),
                  flush=True)
        direction["best_relative_error"] = {
            q: min(r[q]["rel_error"] for r in direction["fd"]) for q in QUANTITIES}
        direction["pass"] = {q: direction["best_relative_error"][q] <= TOL
                             for q in QUANTITIES}
    memory["after_differences"] = peak_working_set_mb()
    for direction in directions:
        del direction["_d"]
    record["directions"] = directions
    record["evaluations"] = evaluations
    record["verdict"] = {
        "all_pass": all(all(d["pass"].values()) for d in directions),
        "perturbed_evaluations": len(evaluations),
        "worst_best_step_relative_error": max(
            max(d["best_relative_error"].values()) for d in directions),
    }
    record["cost"] = {
        "t_build_s": t_build,
        "t_baseline_value_and_gradient_s": t_base,
        "t_component_linearisation_s": t_vjp,
        "t_perturbed_evaluation_s": [e["seconds"] for e in evaluations],
        "peak_working_set_mb_cumulative": memory,
        "wall_clock_total_s": time.perf_counter() - t_start,
    }

    faulthandler.cancel_dump_traceback_later()
    stacks.close()
    (out / RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("\nbest-step relative error, per direction:")
    for d in directions:
        print(f"  seed {d['seed']} ({d['active_components']} active): "
              + "  ".join(f"{q} {d['best_relative_error'][q]:.1e}" for q in QUANTITIES))
    print(f"verdict: {'PASS' if record['verdict']['all_pass'] else 'FAIL'} at {TOL:g} "
          f"({len(evaluations)} perturbed evaluations)")
    print(f"\nwrote {out / RECORD}; total {record['cost']['wall_clock_total_s']:.0f} s")


if __name__ == "__main__":
    main()
