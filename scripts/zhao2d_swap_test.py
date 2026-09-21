"""Are Zhao's reported Psi_0 and C_0 swapped?

Section 4.1 reports Psi_0 = 20,816 and C_0 = 0.0456 for a uniform gamma = 0.4
reference. At the geometry figure 7 prints, this implementation gets values of
those two magnitudes -- but the other way round. This script sweeps the readings
section 4.1 leaves open and scores each one BOTH ways, so the comparison is a
measurement rather than an assertion.

Nothing is relabelled anywhere in the library. `Zhao2DSpec.reported_psi_0` and
`reported_c_0` keep the paper's own labels.
"""

from __future__ import annotations

import itertools
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))

from tfopus import fe_flow, fe_thermal, zhao2d as z, zhao2d_analysis as za  # noqa: E402

PAPER_PSI_0 = 20816.0
PAPER_C_0 = 0.0456


def score(value, target):
    """Ratio, always >= 1, so 1.00 is perfect and bigger is worse either way."""
    if value <= 0 or target <= 0:
        return float("inf")
    r = value / target
    return r if r >= 1.0 else 1.0 / r


def main() -> None:
    spec = z.Zhao2DSpec()
    print(f"geometry as figure 7 prints it: design "
          f"{spec.design_half_width} x {spec.design_height}, h = {spec.element_size}")
    print(f"paper: Psi_0 = {PAPER_PSI_0}, C_0 = {PAPER_C_0}\n")

    head = (f"{'reference':>12} {'amax':>7} {'flow':>5} {'source':>13} "
            f"{'Psi':>11} {'C':>11} | {'as printed':>11} {'if swapped':>11} "
            f"{'T_max':>7}")
    print(head)
    print("-" * len(head))

    rows = []
    for ref, amax, (fname, fform), src in itertools.product(
        list(z.ReferenceField),
        (1.0e6, 1.0e7),
        (("zhou", fe_flow.ZHOU_FORM), ("zhao", fe_flow.ZHAO_FORM)),
        list(z.SourceRegion),
    ):
        opts = za.CaseOptions(
            reference=ref, alpha_max=amax, flow_form=fform, source=src,
            thermal_form=fe_thermal.ZHAO_FORM,
        )
        r = za.analyse(spec, opts)
        psi, c = r["psi"], r["compliance"]
        printed = max(score(psi, PAPER_PSI_0), score(c, PAPER_C_0))
        swapped = max(score(psi, PAPER_C_0), score(c, PAPER_PSI_0))
        rows.append((printed, swapped, ref.value, amax, fname, src.value, psi, c,
                     r["temperature_max"]))
        print(f"{ref.value:>12} {amax:7.0e} {fname:>5} {src.value:>13} "
              f"{psi:11.5g} {c:11.5g} | {printed:11.3g} {swapped:11.3g} "
              f"{r['temperature_max']:7.3f}")

    best_printed = min(rows, key=lambda t: t[0])
    best_swapped = min(rows, key=lambda t: t[1])
    print("\nbest agreement with the labels AS PRINTED: "
          f"worst-of-two ratio {best_printed[0]:.4g}  ({best_printed[2]}, "
          f"amax={best_printed[3]:.0e}, {best_printed[4]}, {best_printed[5]})")
    print("best agreement with the labels SWAPPED:    "
          f"worst-of-two ratio {best_swapped[1]:.4g}  ({best_swapped[2]}, "
          f"amax={best_swapped[3]:.0e}, {best_swapped[4]}, {best_swapped[5]})")


if __name__ == "__main__":
    main()
