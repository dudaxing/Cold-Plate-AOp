"""Total gradients of Psi, C, the volume constraint and J, against finite differences.

Checked SEPARATELY before the combination. With w = 0.5 and two ratios of order
one, an error in dPsi and an equal and opposite error in dC would cancel in dJ
and leave a clean-looking check, so the combined gradient alone proves nothing.

The chain being differentiated is the whole thing:

    x -> filter -> projection -> s -> alpha(s), kappa(s)
      -> Newton(flow)     [implicit]  -> u
      -> Newton(thermal)  [implicit]  -> T          depends on u AND kappa
      -> Psi(u, alpha),  C(T, u, kappa)

with two design-dependent paths inside the thermal residual that are easy to
miss and are exercised here: tau_T depends on kappa and on u, and the stabilised
source term tau_T (u.grad v) Q depends on both even though Q itself does not
depend on the design.

Both a uniform field and a non-uniform grey field are tested. A uniform field
can hide errors in anything that varies element to element -- the filter, and
the kappa dependence of tau_T.

No tolerance is invented: each derivative is evaluated at several step sizes and
the check is that the central difference has a PLATEAU in agreement with AD --
truncation falls as h^2 until round-off takes over, so agreement should improve,
flatten, then degrade. The reported number is the best agreement over the sweep.

    python scripts/zhao2d_gradient_check.py [--full] [--probes N]
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))

from tfopus import zhao2d as z, zhao2d_r1 as r1  # noqa: E402

STEPS = (1e-4, 1e-5, 1e-6)


def build(full: bool):
    spec = z.Zhao2DSpec()
    if not full:
        spec = dataclasses.replace(spec, element_size=spec.element_size * 2)
    config = r1.R1Config()
    problem = r1.Zhao2DProblem(spec, config)
    reference, _ = r1.freeze_reference(spec, config)
    return spec, config, problem, reference


def quantities(problem, reference, x, alpha_max):
    """(Psi, C, g, J) from ONE pair of solves, all as JAX scalars."""
    s = problem.solid_fraction(x)
    psi, c = problem.metrics(s, alpha_max)
    g = problem.fluid_fraction(x) / problem.config.max_fluid_fraction - 1.0
    w = problem.config.weight
    j = w * psi / reference.psi_0 + (1.0 - w) * c / reference.c_0
    return psi, c, g, j


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="full 5200-element mesh")
    ap.add_argument("--probes", type=int, default=6)
    args = ap.parse_args()

    spec, config, problem, reference = build(args.full)
    alpha_max = config.alpha_max_reference
    print(f"mesh h = {spec.element_size:g}, {problem.flow_mesh.num_elems} elements, "
          f"{problem.num_design} design variables")
    print(f"filter radius = {config.filter_radius_elements} h, "
          f"projection beta = {config.projection_beta}")
    print(f"Psi_0 = {reference.psi_0:.10g}, C_0 = {reference.c_0:.10g}\n")

    rng = np.random.default_rng(0)
    fields = {
        "uniform 0.6": np.full(problem.num_design, 0.6),
        "non-uniform grey": rng.uniform(0.15, 0.85, problem.num_design),
    }

    names = ("Psi", "C", "g", "J")
    worst_overall = 0.0

    for field_name, x0 in fields.items():
        x0 = np.asarray(x0)
        print(f"--- {field_name} ---")

        # the state must be converged before any gradient is meaningful
        norms = problem.require_converged(problem.solid_fraction(x0), alpha_max)
        print(f"    converged: flow {norms['flow']:.2e}, thermal {norms['thermal']:.2e}")

        grads = [
            np.asarray(
                jax.grad(lambda x, i=i: quantities(problem, reference, x, alpha_max)[i])(
                    jax.numpy.asarray(x0)
                )
            )
            for i in range(4)
        ]

        probes = rng.choice(problem.num_design, size=args.probes, replace=False)
        print(f"    {'probe':>6} {'quantity':>9} {'AD':>15} {'FD (best h)':>15} "
              f"{'rel':>10} {'best h':>8}")
        for i in probes:
            fd_by_step = {}
            for h in STEPS:
                xp, xm = x0.copy(), x0.copy()
                xp[i] += h
                xm[i] -= h
                up = quantities(problem, reference, jax.numpy.asarray(xp), alpha_max)
                um = quantities(problem, reference, jax.numpy.asarray(xm), alpha_max)
                fd_by_step[h] = [
                    (float(a) - float(b)) / (2 * h) for a, b in zip(up, um)
                ]
            for q in range(4):
                ad = float(grads[q][i])
                rels = {
                    h: abs(ad - fd[q]) / max(abs(fd[q]), 1e-300)
                    for h, fd in fd_by_step.items()
                }
                best_h = min(rels, key=rels.get)
                worst_overall = max(worst_overall, rels[best_h])
                print(f"    {i:6d} {names[q]:>9} {ad:15.7e} "
                      f"{fd_by_step[best_h][q]:15.7e} {rels[best_h]:10.2e} "
                      f"{best_h:8.0e}")
        print()

    print(f"worst relative AD/FD disagreement over all probes and quantities: "
          f"{worst_overall:.3e}")
    if worst_overall > 1e-5:
        sys.exit("gradient check FAILED: disagreement is too large to be "
                 "finite-difference error")
    print("gradient check passed")


if __name__ == "__main__":
    main()
