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

**What the headline number is, precisely.** Each derivative is evaluated at
three step sizes and ALL THREE are printed. The summary figure is

    max over (field, probe, quantity) of min over step of relative error

i.e. the largest disagreement after each derivative is allowed its own best
step. That is good positive evidence, but it is NOT a demonstration that the
differences sit on a plateau -- an earlier report called it one. The full
per-step table is printed so the trend can be read, and the perturbed states
x +- delta e_i are gated for convergence too, not only the base state: a
finite difference taken across an unconverged solve is not a derivative of
anything.

**Dual-mesh mode (R1g).** `--thermal-refinement r` runs the same protocol on
`Zhao2DDualProblem`: the temperature on a nested mesh r times finer, reached
through the density map E and the velocity extension P. Everything above still
applies, and two things are added. `--directions N` random +-1 directions are
checked as well as coordinate probes: a directional derivative sums every
component, so it exercises every E^T and P^T accumulation at once, which a
single coordinate probe cannot. And each perturbed state is solved once and
gated from that same solve (`evaluate`), rather than solved twice. The
threshold is unchanged. J uses the check mesh's single-mesh reference as fixed
constants -- a reporting scale, which is all a derivative check needs.

    python scripts/zhao2d_gradient_check.py [--full] [--probes N] [--stage NAME]
        [--thermal-refinement R] [--thermal-quadrature Q] [--directions N]
"""

from __future__ import annotations

import pathlib
import sys

# Default the BLAS thread count BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import dataclasses

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))

from tfopus import zhao2d as z, zhao2d_dual as dual, zhao2d_r1 as r1  # noqa: E402

STEPS = (1e-4, 1e-5, 1e-6)


def build(full: bool, refinement: int = 1, quadrature: int | None = None):
    spec = z.Zhao2DSpec()
    if not full:
        spec = dataclasses.replace(spec, element_size=spec.element_size * 2)
    config = r1.R1Config(projection=r1.Projection.TANH)  # the projection its record used
    if refinement == 1 and quadrature is None:
        problem = r1.Zhao2DProblem(spec, config)
    else:
        problem = dual.Zhao2DDualProblem(
            spec, config, thermal_refinement=refinement,
            thermal_quadrature=3 if quadrature is None else quadrature,
        )
    # single-mesh reference on the check mesh: fixed constants for J
    reference, _ = r1.freeze_reference(spec, config)
    return spec, config, problem, reference


def perturbed(problem, reference, x, alpha_max, beta):
    """(Psi, C, g, J) as floats at a perturbed design, gated for convergence.

    The dual-mesh problem gates from the same solve it reports; the single-mesh
    problem keeps the original two-step path, so R1b's protocol is unchanged.
    """
    if isinstance(problem, dual.Zhao2DDualProblem):
        e = problem.evaluate(x, alpha_max, beta)
        bad = {k: v for k, v in e["residuals"].items() if not (v <= 1e-8)}
        if bad:
            raise r1.NotConverged(f"perturbed state not converged: {bad}")
        w = problem.config.weight
        j = w * e["psi"] / reference.psi_0 + (1.0 - w) * e["c"] / reference.c_0
        return e["psi"], e["c"], e["g"], j
    problem.require_converged(problem.solid_fraction(x, beta), alpha_max)
    return tuple(float(v) for v in quantities(problem, reference, x, alpha_max, beta))


def quantities(problem, reference, x, alpha_max, beta):
    """(Psi, C, g, J) from ONE pair of solves, all as JAX scalars."""
    s = problem.solid_fraction(x, beta)
    psi, c = problem.metrics(s, alpha_max)
    g = problem.fluid_fraction(x, beta) / problem.config.max_fluid_fraction - 1.0
    w = problem.config.weight
    j = w * psi / reference.psi_0 + (1.0 - w) * c / reference.c_0
    return psi, c, g, j


# The stages whose derivative paths R1d actually uses. The projection is the
# genuinely new one: beta > 0 inserts a tanh between the filter and the physics,
# and alpha_max = 1e7 is where the Brinkman term is stiffest.
STAGES = {
    "reference": (1.0e6, 0.0),
    "final-alpha": (1.0e7, 0.0),
    "projected": (1.0e7, 8.0),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="full 5200-element mesh")
    ap.add_argument("--probes", type=int, default=6)
    ap.add_argument("--stage", choices=tuple(STAGES) + ("all",), default="all")
    ap.add_argument("--thermal-refinement", type=int, default=1,
                    help="R1g: temperature on a nested mesh this many times finer")
    ap.add_argument("--thermal-quadrature", type=int, default=None,
                    help="thermal Gauss order (dual mode defaults to 3)")
    ap.add_argument("--directions", type=int, default=None,
                    help="random +-1 directional probes (dual mode defaults to 2)")
    args = ap.parse_args()

    spec, config, problem, reference = build(
        args.full, args.thermal_refinement, args.thermal_quadrature
    )
    is_dual = isinstance(problem, dual.Zhao2DDualProblem)
    n_dirs = args.directions if args.directions is not None else (2 if is_dual else 0)
    if is_dual:
        print(f"DUAL MESH: thermal refinement {problem.thermal_refinement}, "
              f"{problem.thermal_mesh.num_elems} thermal elements, quadrature "
              f"{problem.thermal_quadrature}")
    stages = STAGES if args.stage == "all" else {args.stage: STAGES[args.stage]}
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

    for stage_name, (alpha_max, beta) in stages.items():
        for field_name, x0 in fields.items():
            x0 = np.asarray(x0)
            print(f"--- stage {stage_name} (alpha_max {alpha_max:.0e}, "
                  f"beta {beta:g}) | {field_name} ---")

            norms = problem.require_converged(
                problem.solid_fraction(x0, beta), alpha_max
            )
            print(f"    base converged: flow {norms['flow']:.2e}, "
                  f"thermal {norms['thermal']:.2e}")

            grads = [
                np.asarray(
                    jax.grad(
                        lambda x, i=i: quantities(
                            problem, reference, x, alpha_max, beta
                        )[i]
                    )(jax.numpy.asarray(x0))
                )
                for i in range(4)
            ]

            probes = rng.choice(problem.num_design, size=args.probes, replace=False)
            directions = [rng.choice([-1.0, 1.0], problem.num_design)
                          for _ in range(n_dirs)]
            print(f"    {'probe':>6} {'qty':>4} {'AD':>14} "
                  + "".join(f"{'rel @ ' + f'{h:.0e}':>13}" for h in STEPS)
                  + f"{'best':>10}")
            probe_list = [("e", i, None) for i in probes] + [
                ("d", k, d) for k, d in enumerate(directions)
            ]
            for kind, i, d in probe_list:
                if kind == "e":
                    step_dir = np.zeros(problem.num_design)
                    step_dir[i] = 1.0
                else:
                    step_dir = d
                fd_by_step = {}
                for h in STEPS:
                    # gate the PERTURBED state too: a difference taken across
                    # an unconverged solve is not a derivative
                    row = [
                        perturbed(problem, reference,
                                  jax.numpy.asarray(x0 + sign * h * step_dir),
                                  alpha_max, beta)
                        for sign in (+1, -1)
                    ]
                    fd_by_step[h] = [
                        (float(a) - float(b)) / (2 * h) for a, b in zip(*row)
                    ]
                label = f"{i:6d}" if kind == "e" else f"{'dir' + str(i):>6}"
                for q in range(4):
                    ad = float(np.dot(grads[q], step_dir))
                    rels = {
                        h: abs(ad - fd[q]) / max(abs(fd[q]), 1e-300)
                        for h, fd in fd_by_step.items()
                    }
                    best = min(rels.values())
                    worst_overall = max(worst_overall, best)
                    print(f"    {label} {names[q]:>4} {ad:14.6e} "
                          + "".join(f"{rels[h]:13.2e}" for h in STEPS)
                          + f"{best:10.2e}")
            print()

    print("summary figure = max over (stage, field, probe, quantity) of "
          "min over step of relative error")
    print(f"  = {worst_overall:.3e}")
    print("  This is the best-step disagreement, NOT a plateau demonstration; "
          "read the per-step columns above for the trend.")
    if worst_overall > 1e-5:
        sys.exit("gradient check FAILED: disagreement is too large to be "
                 "finite-difference error")
    print("gradient check passed")


if __name__ == "__main__":
    main()
