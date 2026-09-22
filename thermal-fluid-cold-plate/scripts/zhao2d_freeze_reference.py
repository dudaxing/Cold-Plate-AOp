"""Freeze the R1 normalisation, and measure the thermal switch one change at a time.

Psi does not depend on the thermal configuration at all: the coupling is one
way, so with the flow settings fixed the same converged flow field feeds every
thermal analysis. That makes Psi_0 a cross-check (it must be identical across
A/B/C) and isolates what the thermal switch actually does to C.

    A   tau = convective, stabilise_source = False   the R0 thermal setting
    B   tau = two_limit,  stabilise_source = False   changes tau alone
    C   tau = two_limit,  stabilise_source = True    the R1 main line -> C_0

All three keep supg_heat_capacity = True. Nothing here picks a configuration by
closeness to the paper's numbers; C is the decided setting, and A and B exist so
that the difference between R0's and R1's C_0 attributes to one change each.

    python scripts/zhao2d_freeze_reference.py [--write] [--coarse]
"""

from __future__ import annotations

import pathlib
import sys

# Cap BLAS threads BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import dataclasses

import jax

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))

from tfopus import fe_thermal, zhao2d as z, zhao2d_r1 as r1  # noqa: E402

OUT = REPO / "tfopus" / "zhao2d_reference.json"

VARIANTS = {
    "A  R0 setting     ": fe_thermal.ThermalForm(
        tau="convective", stabilise_source=False, supg_heat_capacity=True
    ),
    "B  tau only       ": fe_thermal.ThermalForm(
        tau="two_limit", stabilise_source=False, supg_heat_capacity=True
    ),
    "C  R1 MAIN LINE   ": fe_thermal.ThermalForm(
        tau="two_limit", stabilise_source=True, supg_heat_capacity=True
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help=f"write the frozen values to {OUT.name}")
    ap.add_argument("--coarse", action="store_true")
    args = ap.parse_args()

    spec = z.Zhao2DSpec()
    if args.coarse:
        spec = dataclasses.replace(spec, element_size=spec.element_size * 2)

    base = r1.R1Config()
    print("R1 frozen configuration")
    print(f"  reference       {base.reference}")
    print(f"  alpha_max_ref   {base.alpha_max_reference:.0e}")
    print(f"  source          {base.source}")
    print(f"  outlet          {base.outlet}")
    print(f"  h_e mode        {base.element_length_mode}")
    print(f"  volume domain   {base.volume_domain}")
    print(f"  flow form       {base.flow_form}")
    print(f"  thermal form    {base.thermal_form}")
    print(f"  mesh h          {spec.element_size:g}\n")

    head = (f"{'thermal variant':>20} {'Psi_0':>16} {'C_0':>16} "
            f"{'|Rf|/|Rf0|':>11} {'|Rt|/|Rt0|':>11}")
    print(head)
    print("-" * len(head))

    results = {}
    for label, form in VARIANTS.items():
        config = dataclasses.replace(base, thermal_form=form)
        values, report = r1.freeze_reference(spec, config)
        results[label.strip()[0]] = (values, report)
        print(f"{label:>20} {values.psi_0:16.10g} {values.c_0:16.10g} "
              f"{values.flow_residual_relative:11.2e} "
              f"{values.thermal_residual_relative:11.2e}")

    psi_seen = [v.psi_0 for v, _ in results.values()]
    spread = max(psi_seen) - min(psi_seen)
    print(f"\nPsi_0 spread across the three thermal variants: {spread:.3e} "
          f"(must be 0 -- the coupling is one way)")
    assert spread == 0.0, "Psi_0 changed with the thermal setting; coupling is not one way"

    c0 = {k: v.c_0 for k, (v, _) in results.items()}
    print(f"\nC_0 change, A -> B (tau only)           "
          f"{(c0['B'] / c0['A'] - 1) * 100:+7.2f}%")
    print(f"C_0 change, B -> C (stabilised source)  "
          f"{(c0['C'] / c0['B'] - 1) * 100:+7.2f}%")
    print(f"C_0 change, A -> C (both)               "
          f"{(c0['C'] / c0['A'] - 1) * 100:+7.2f}%")

    frozen, frozen_report = results["C"]
    print("\nfrozen R1 reference:")
    print(f"  Psi_0 = {frozen.psi_0!r}")
    print(f"  C_0   = {frozen.c_0!r}")
    print(f"  v_f   design {frozen_report['v_f_design_domain']:.6f}  "
          f"whole {frozen_report['v_f_whole_domain']:.6f}")

    if args.write:
        if args.coarse:
            sys.exit("refusing to write a reference frozen on the coarse mesh")
        OUT.write_text(frozen.to_json(), encoding="utf-8")
        print(f"\nwritten to {OUT.relative_to(REPO)}")
    else:
        print("\n(not written; pass --write)")


if __name__ == "__main__":
    main()
