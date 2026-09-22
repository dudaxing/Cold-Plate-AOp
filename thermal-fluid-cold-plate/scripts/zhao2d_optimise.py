"""R1d: the single 2D main case.

Loads the frozen reference from disk (it does NOT refreeze), verifies both
normalisation ratios separately against a re-solved physical reference state,
runs the phased continuation, and saves a design paired with metrics from the
same iterate.

Finishes with a binary diagnostic: threshold at s = 0.5 with NO morphological
repair, re-analyse, and report what changed. If the thresholded design is
infeasible or disconnected, that is the result; it is not repaired into
something that looks like the paper.

    python scripts/zhao2d_optimise.py [--budget 300] [--coarse] [--out DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))

from tfopus import zhao2d as z, zhao2d_driver as drv, zhao2d_r1 as r1  # noqa: E402
from tfopus.mesh import Face  # noqa: E402


def element_adjacency(planar) -> sp.csr_matrix:
    """Face-neighbour graph of a structured masked grid, from element centres."""
    centres = np.asarray(planar.elem_centres)
    h = float(np.sqrt(np.asarray(planar.elem_area)[0]))
    index = {(round(x / h), round(y / h)): i for i, (x, y) in enumerate(centres)}
    rows, cols = [], []
    for (i, j), e in index.items():
        for nb in ((i + 1, j), (i, j + 1)):
            if nb in index:
                rows += [e, index[nb]]
                cols += [index[nb], e]
    n = len(centres)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def port_elements(planar, tag: Face) -> np.ndarray:
    return np.unique([e for e, _ in planar.elem_faces[tag]])


def binary_diagnostic(problem, reference, x, alpha_max, beta) -> dict:
    """Threshold at 0.5, re-analyse, report. No repair of any kind."""
    s_grey = np.asarray(problem.solid_fraction(x, beta))
    s_bin = (s_grey >= 0.5).astype(float)
    # the fixed tabs are fluid by definition, not by the threshold
    s_bin[~problem.flow_mesh.design_mask] = 0.0

    fluid = s_bin < 0.5
    graph = element_adjacency(problem.flow_mesh)
    n_comp, labels = csgraph.connected_components(
        graph[fluid][:, fluid], directed=False
    )
    fluid_idx = np.nonzero(fluid)[0]
    lookup = {e: labels[k] for k, e in enumerate(fluid_idx)}
    inlet = [lookup[e] for e in port_elements(problem.flow_mesh, Face.INLET)
             if e in lookup]
    outlet = [lookup[e] for e in port_elements(problem.flow_mesh, Face.OUTLET)
              if e in lookup]
    connected = bool(inlet and outlet and set(inlet) & set(outlet))

    out = {
        "grey_fraction_before": float(np.mean((s_grey > 0.05) & (s_grey < 0.95))),
        "fluid_components": int(n_comp),
        "inlet_outlet_connected": connected,
        **{f"binary_{k}": v for k, v in z.fluid_fractions(
            problem.flow_mesh, jnp.asarray(s_bin)).items()},
    }
    if not connected:
        out["note"] = (
            "inlet and outlet are NOT connected through fluid after "
            "thresholding; the binary design is not a usable channel and was "
            "not repaired"
        )
        return out
    try:
        norms = problem.require_converged(jnp.asarray(s_bin), alpha_max)
        psi, c = problem.metrics(jnp.asarray(s_bin), alpha_max)
        w = problem.config.weight
        out.update(
            binary_psi=float(psi),
            binary_compliance=float(c),
            binary_J_self=float(
                w * psi / reference.psi_0 + (1 - w) * c / reference.c_0
            ),
            binary_flow_residual=norms["flow"],
            binary_thermal_residual=norms["thermal"],
        )
    except r1.NotConverged as exc:
        out["binary_state"] = f"did not converge: {exc}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--move-limit", type=float, default=0.1)
    ap.add_argument("--coarse", action="store_true",
                    help="h = 2e-4 for a rehearsal; the main case is h = 1e-4")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results")
    args = ap.parse_args()

    spec = z.Zhao2DSpec()
    if args.coarse:
        spec = dataclasses.replace(spec, element_size=spec.element_size * 2)
    config = r1.R1Config()

    # r is an absolute coordinate length, fixed at 2e-4. On the main mesh that
    # is 2h; on any other mesh it is NOT, and the ratio is reported so a mesh
    # study cannot silently change the regularisation along with the mesh.
    radius = 2.0e-4
    config = dataclasses.replace(
        config, filter_radius_elements=radius / spec.element_size
    )

    problem = r1.Zhao2DProblem(spec, config)
    print(f"mesh h = {spec.element_size:g}, {problem.flow_mesh.num_elems} elements, "
          f"{problem.num_design} design variables")
    print(f"filter radius r = {radius:g} (= {radius / spec.element_size:g} h)")

    if args.coarse:
        reference, _ = r1.freeze_reference(spec, config)
        print("REHEARSAL on a coarse mesh: reference refrozen, not the stored one")
    else:
        reference = r1.load_reference(spec, config)
        print("loaded the frozen reference from disk")
    ratios = r1.verify_reference_against_state(problem, reference)
    print(f"  Psi/Psi_0 = {ratios['psi_over_psi_0']:.12f}")
    print(f"  C/C_0     = {ratios['c_over_c_0']:.12f}   (checked separately)")

    phases = drv.r1d_schedule(spec, args.budget)
    print(f"\nschedule, budget {args.budget}:")
    step = 0
    for p in phases:
        print(f"  {p.name:12s} n={p.iterations:4d} beta={p.beta:<4g} "
              f"alpha_max {p.alpha_at(step):.3e} -> "
              f"{p.alpha_at(step + p.iterations - 1):.3e}"
              + (f"   {p.note}" if p.note else ""))
        step += p.iterations

    head = (f"{'it':>4} {'phase':>11} {'a_max':>8} {'b':>3} {'J':>9} {'Psi/Psi0':>9} "
            f"{'C/C0':>8} {'v_f':>7} {'g':>9} {'grey':>6} {'|R|':>8} {'s':>5}")
    print("\n" + head)
    print("-" * len(head))

    def show(rec):
        mark = "*" if rec["model_changed"] else " "
        print(f"{rec['iteration']:4d}{mark}{rec['phase']:>10} {rec['alpha_max']:8.1e} "
              f"{rec['beta']:3g} {rec['J_self']:9.5f} {rec['psi_over_psi0_self']:9.4f} "
              f"{rec['c_over_c0_self']:8.4f} {rec['v_f_design_domain']:7.4f} "
              f"{rec['constraint_g']:9.2e} {rec['grey_fraction']:6.3f} "
              f"{max(rec['flow_residual_relative'], rec['thermal_residual_relative']):8.1e} "
              f"{rec['seconds']:5.1f}")

    t0 = time.time()
    result = drv.run(problem, reference, phases, args.move_limit, args.budget, show)
    elapsed = time.time() - t0

    print(f"\nstop reason: {result.stop_reason}")
    print(f"ended in phase {result.final_phase} at alpha_max "
          f"{result.final_alpha_max:.3e}, beta {result.final_beta:g}")
    print(f"terminal design re-evaluated: J = {result.terminal['J_self']:.6f} "
          f"(the loop's last record was {result.history[-1]['J_self']:.6f} for the "
          f"PREVIOUS iterate, same model)")

    diag = binary_diagnostic(
        problem, reference, jnp.asarray(result.design),
        result.final_alpha_max, result.final_beta
    )
    print("\nbinary diagnostic at s = 0.5, no repair:")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    args.out.mkdir(parents=True, exist_ok=True)
    tag = "coarse" if args.coarse else "main"
    np.savez_compressed(
        args.out / f"zhao2d_r1d_{tag}_fields.npz",
        design=result.design,
        solid_fraction=result.solid_fraction,
        press_vel=result.press_vel,
        temperature=result.temperature,
        elem_centres=np.asarray(problem.flow_mesh.elem_centres),
    )
    meta = {
        "stop_reason": result.stop_reason,
        "converged": result.stop_reason == "converged",
        "elapsed_seconds": elapsed,
        "element_size": spec.element_size,
        "num_elements": int(problem.flow_mesh.num_elems),
        "filter_radius": radius,
        "filter_radius_over_h": radius / spec.element_size,
        "reference": dataclasses.asdict(reference),
        "reference_check": ratios,
        "run_fingerprint": config.fingerprint(),
        "schedule": [
            {"name": p.name, "iterations": p.iterations, "beta": p.beta}
            for p in phases
        ],
        "history": result.history,
        "terminal": result.terminal,
        "final_alpha_max": result.final_alpha_max,
        "final_beta": result.final_beta,
        "final_phase": result.final_phase,
        "binary_diagnostic": diag,
    }
    (args.out / f"zhao2d_r1d_{tag}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # Read back: recompute the terminal metrics from the SAVED design.
    saved = np.load(args.out / f"zhao2d_r1d_{tag}_fields.npz")["design"]
    rec, _ = drv._evaluate(
        problem, reference, jnp.asarray(saved), result.final_alpha_max,
        result.final_beta
    )
    drift = abs(rec["J_self"] - result.terminal["J_self"])
    print(f"\nread-back: J from the saved design = {rec['J_self']:.12f}, "
          f"drift {drift:.2e}")
    if drift > 1e-10:
        sys.exit("read-back mismatch: the saved design does not reproduce the "
                 "reported terminal metrics")
    print(f"saved to {(args.out / f'zhao2d_r1d_{tag}.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
