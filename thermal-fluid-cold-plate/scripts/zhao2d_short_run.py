"""A SHORT MMA run: does the optimisation loop hold together end to end?

This is a mechanism check, not a result. It stops at a fixed iteration count and
does not continue to convergence, so the design it produces is not an optimum
and must not be reported as one.

What it checks:
  * MMA accepts the objective, constraint and their gradients and moves x;
  * every state is verified converged BEFORE its gradient reaches MMA;
  * the volume constraint is driven toward and held at the bound;
  * the metrics, both normalisations and the design are saved consistently.

alpha_max is held FIXED by default. Continuation changes the model itself, so
the objective is not comparable across a continuation step -- with `--continue`
the two phases are logged separately and the step where alpha_max moves is
marked, so a rise in J there is not mistaken for an optimiser failure, and
neither is it allowed to hide an unconverged state: the residuals are printed on
every iteration regardless.

    python scripts/zhao2d_short_run.py [--iterations N] [--full] [--continue]
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

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))

import toflux.src.mma as _mma  # noqa: E402

from tfopus import zhao2d as z, zhao2d_r1 as r1  # noqa: E402

# Zhao's transposed constants, for the parallel report only. Never the main
# objective; see docs/zhao_reproduction.md.
PAPER_PSI_0_INTERPRETED = 0.0456
PAPER_C_0_INTERPRETED = 20816.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--continue", dest="continuation", action="store_true",
                    help="let alpha_max follow 1e6 * 1.03^iter after half the run")
    ap.add_argument("--move-limit", type=float, default=0.1)
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "results" / "zhao2d_short_run.json")
    args = ap.parse_args()

    spec = z.Zhao2DSpec()
    if not args.full:
        spec = dataclasses.replace(spec, element_size=spec.element_size * 2)
    config = r1.R1Config()
    problem = r1.Zhao2DProblem(spec, config)

    reference, _ = r1.freeze_reference(spec, config)
    print(f"mesh h = {spec.element_size:g}, {problem.flow_mesh.num_elems} elements, "
          f"{problem.num_design} design variables")
    print(f"Psi_0 = {reference.psi_0:.10g}   C_0 = {reference.c_0:.10g}")
    print(f"move limit {args.move_limit}, {args.iterations} iterations, "
          f"continuation {'on after half' if args.continuation else 'OFF'}\n")

    n = problem.num_design
    params = _mma.MMAParams(
        max_iter=args.iterations,
        kkt_tol=1e-6,
        step_tol=1e-6,
        move_limit=args.move_limit,
        num_design_var=n,
        num_cons=1,
        lower_bound=np.zeros((n, 1)),
        upper_bound=np.ones((n, 1)),
    )
    # Start from the reference material fraction: gamma = 0.4 means s = 0.6.
    state = _mma.init_mma(np.full((n, 1), 1.0 - spec.reference_gamma), params)

    def payload(x, alpha_max):
        s = problem.solid_fraction(x)
        psi, c = problem.metrics(s, alpha_max)
        g = problem.fluid_fraction(x) / config.max_fluid_fraction - 1.0
        w = config.weight
        j = w * psi / reference.psi_0 + (1.0 - w) * c / reference.c_0
        return j, (psi, c, g)

    value_and_grad = jax.value_and_grad(lambda x, a: payload(x, a)[0], has_aux=False)
    grad_g = jax.grad(
        lambda x: problem.fluid_fraction(x) / config.max_fluid_fraction - 1.0
    )

    history = []
    header = (f"{'it':>3} {'alpha_max':>9} {'J':>10} {'Psi/Psi0':>9} {'C/C0':>9} "
              f"{'v_f des':>8} {'v_f all':>8} {'g':>9} {'|Rf|':>8} {'|Rt|':>8} "
              f"{'s':>5}")
    print(header)
    print("-" * len(header))

    for it in range(args.iterations):
        alpha_max = (
            spec.alpha_max(it) if args.continuation and it >= args.iterations // 2
            else config.alpha_max_reference
        )
        x = jnp.asarray(state.x.reshape(-1))
        t0 = time.time()

        # Gate first: an unconverged state must never reach MMA.
        s_field = problem.solid_fraction(x)
        norms = problem.require_converged(s_field, alpha_max)

        j, dj = value_and_grad(x, alpha_max)
        _, (psi, c, g) = payload(x, alpha_max)
        dg = grad_g(x)

        fractions = z.fluid_fractions(problem.flow_mesh, s_field)
        row = {
            "iteration": it,
            "alpha_max": alpha_max,
            "continuation_step": bool(
                history and alpha_max != history[-1]["alpha_max"]
            ),
            "J_self": float(j),
            "psi": float(psi),
            "compliance": float(c),
            "psi_over_psi0_self": float(psi) / reference.psi_0,
            "c_over_c0_self": float(c) / reference.c_0,
            "J_paper_interpreted": (
                config.weight * float(psi) / PAPER_PSI_0_INTERPRETED
                + (1 - config.weight) * float(c) / PAPER_C_0_INTERPRETED
            ),
            "constraint_g": float(g),
            **fractions,
            "flow_residual_relative": norms["flow"],
            "thermal_residual_relative": norms["thermal"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        mark = "*" if row["continuation_step"] else " "
        print(f"{it:3d}{mark}{alpha_max:9.1e} {row['J_self']:10.5f} "
              f"{row['psi_over_psi0_self']:9.4f} {row['c_over_c0_self']:9.4f} "
              f"{fractions['v_f_design_domain']:8.4f} "
              f"{fractions['v_f_whole_domain']:8.4f} {row['constraint_g']:9.2e} "
              f"{norms['flow']:8.1e} {norms['thermal']:8.1e} {row['seconds']:5.1f}")

        state = _mma.update_mma(
            state,
            params,
            float(j),
            np.asarray(dj).reshape((-1, 1)),
            np.array([float(g)]),
            np.asarray(dg).reshape((1, -1)),
        )

    _summarise(history, args.continuation)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "note": "SHORT mechanism check, stopped at a fixed iteration "
                        "count. Not a converged design.",
                "config_fingerprint": config.fingerprint(),
                "element_size": spec.element_size,
                "reference": dataclasses.asdict(reference),
                "paper_interpreted": {
                    "psi_0": PAPER_PSI_0_INTERPRETED,
                    "c_0": PAPER_C_0_INTERPRETED,
                },
                "history": history,
                "final_design": np.asarray(state.x.reshape(-1)).tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved to {args.out.relative_to(REPO)}")


def _summarise(history, continuation: bool) -> None:
    j = [h["J_self"] for h in history]
    print(f"\nJ: {j[0]:.5f} -> {j[-1]:.5f}   ({(j[-1] / j[0] - 1) * 100:+.2f}%)")
    print(f"constraint g: {history[0]['constraint_g']:.3e} -> "
          f"{history[-1]['constraint_g']:.3e}   (<= 0 is feasible)")

    worst_r = max(
        max(h["flow_residual_relative"], h["thermal_residual_relative"])
        for h in history
    )
    print(f"worst relative residual over the run: {worst_r:.2e} "
          f"(every iteration passed the gate, or the run would have stopped)")

    rises = [
        (h["iteration"], prev, h["J_self"], h["continuation_step"])
        for prev, h in zip(j, history[1:])
        if h["J_self"] > prev
    ]
    if not rises:
        print("J decreased monotonically.")
        return
    model_changes = [r for r in rises if r[3]]
    within_model = [r for r in rises if not r[3]]
    print(f"J rose on {len(rises)} step(s): {len(model_changes)} at a "
          f"continuation step (the model changed, so J is not comparable "
          f"across it), {len(within_model)} at fixed alpha_max.")
    for it, before, after, _ in within_model:
        print(f"    iteration {it}: {before:.6f} -> {after:.6f} at fixed "
              f"alpha_max -- an MMA step, not a model change")


if __name__ == "__main__":
    main()
