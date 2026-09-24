"""Which of the three momentum-form switches actually moves the answer?

`ZHAO_FORM` differs from `ZHOU_FORM` in three places at once, so a difference
between them attributes to nothing. This varies each switch independently at
the figure-7 scale and the selected reference state.

Motivated by a correction: the reactive limit in tau was previously credited
with the bulk of the Zhao/Zhou difference, on the strength of a ratio computed
at mu = 1.0 rather than Zhao's 0.001. `scripts/tau_decomposition.py` shows the
real factor there is 1.4% (edge h) to 5.4% (diagonal h), which cannot account
for a 17% change in Psi. So the cause is one of the other two switches, and
this measures which.

    python scripts/zhao2d_form_attribution.py [--coarse]
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))

from tfopus import fe_flow, zhao2d as z, zhao2d_analysis as za  # noqa: E402

# The state R0 selected: gamma = 0.4 over the whole domain, continuation start.
REFERENCE = z.ReferenceField.UNIFORM_ALL
ALPHA_MAX = 1.0e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coarse", action="store_true")
    args = ap.parse_args()

    import dataclasses

    spec = z.Zhao2DSpec()
    if args.coarse:
        spec = dataclasses.replace(spec, element_size=spec.element_size * 2)

    head = (f"{'tau reactive':>13} {'alpha in SUPG':>14} {'viscous':>10} "
            f"{'Psi':>12} {'C':>12} {'dPsi vs Zhou':>13} {'dC vs Zhou':>11}")
    print(f"reference = {REFERENCE.value}, alpha_max = {ALPHA_MAX:.0e}, "
          f"h = {spec.element_size:g}\n")
    print(head)
    print("-" * len(head))

    results = {}
    for react, supg_alpha, visc in itertools.product(
        (True, False), (True, False), ("symmetric", "laplacian")
    ):
        form = fe_flow.FlowForm(
            brinkman_in_tau=react,
            brinkman_in_supg_residual=supg_alpha,
            viscous_form=visc,
        )
        r = za.analyse(
            spec,
            za.CaseOptions(
                reference=REFERENCE, alpha_max=ALPHA_MAX, flow_form=form
            ),
        )
        results[(react, supg_alpha, visc)] = (r["psi"], r["compliance"])

    base = results[(True, True, "symmetric")]  # ZHOU_FORM
    for key, (psi, c) in results.items():
        react, supg_alpha, visc = key
        print(f"{str(react):>13} {str(supg_alpha):>14} {visc:>10} "
              f"{psi:12.6g} {c:12.6g} "
              f"{(psi / base[0] - 1) * 100:12.2f}% {(c / base[1] - 1) * 100:10.2f}%")

    print("\none-at-a-time from ZHOU_FORM (True, True, symmetric):")
    for key, label in (
        ((False, True, "symmetric"), "drop the reactive limit in tau"),
        ((True, False, "symmetric"), "drop alpha*u from the SUPG residual"),
        ((True, True, "laplacian"), "Laplacian instead of symmetric viscous"),
    ):
        psi, c = results[key]
        print(f"  {label:<40} dPsi {(psi / base[0] - 1) * 100:+7.2f}%   "
              f"dC {(c / base[1] - 1) * 100:+7.2f}%")


if __name__ == "__main__":
    main()
