"""R0: can Zhao's reported Psi_0 = 20,816 and C_0 = 0.0456 be reproduced?

Sweeps the choices section 4.1 leaves undetermined and reports what each one
gives. Nothing is tuned toward the reported numbers; they are printed as a
ratio so the reader can see the gap rather than a pass/fail.

Psi is computed first and alone, because it does not involve the thermal model:
if no flow-side reading reproduces Psi_0, no thermal convention can explain it.

    python scripts/zhao2d_reference_study.py [--stage flow|full] [--coarse]
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import sys
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))

from tfopus import fe_flow as fe_flow  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402


def scaled_spec(spec: z.Zhao2DSpec, factor: float) -> z.Zhao2DSpec:
    """Multiply every length by `factor`, leaving physics untouched.

    Figure 7 prints 0.001 and 0.01. Those give Re = 200 when Re = rho U D / mu
    with D the inlet half width. A domain 1000x larger gives the SAME 5200
    elements (the count is a ratio) and also gives Re = 200 if the reported
    Reynolds number instead used mu as a kinematic viscosity. The element count
    cannot separate the two readings; Psi can, because the Brinkman term
    integrates over area while the viscous term is scale invariant.
    """
    return dataclasses.replace(
        spec,
        inlet_half_width=spec.inlet_half_width * factor,
        tab_length=spec.tab_length * factor,
        design_half_width=spec.design_half_width * factor,
        design_height=spec.design_height * factor,
        element_size=spec.element_size * factor,
    )


def flow_sweep(base: z.Zhao2DSpec, scales) -> None:
    print("\n" + "=" * 100)
    print("STAGE 1 -- Psi only (independent of every thermal choice)")
    print("=" * 100)
    header = (
        f"{'length x':>9} {'reference':>13} {'outlet':>9} {'alpha_max':>10} "
        f"{'form':>5} {'Psi':>13} {'Psi/20816':>11} {'|R|/|R0|':>10} {'u_max':>8}"
    )
    print(header)
    print("-" * len(header))

    for scale, ref, outlet, amax, (fname, form) in itertools.product(
        scales,
        list(z.ReferenceField),
        list(z.OutletKind),
        (base.alpha_max_initial, base.alpha_max_final),
        (("zhao", fe_flow.ZHAO_FORM), ("zhou", fe_flow.ZHOU_FORM)),
    ):
        spec = scaled_spec(base, scale)
        opts = za.CaseOptions(
            reference=ref, outlet=outlet, alpha_max=amax, flow_form=form
        )
        try:
            r = za.analyse(spec, opts, with_thermal=False)
        except Exception as exc:  # noqa: BLE001 - a failed reading is a result
            print(f"{scale:9.0f} {ref.value:>13} {outlet.value:>9} "
                  f"{amax:10.0e} {fname:>5}  FAILED: {type(exc).__name__}: {exc}")
            continue
        print(
            f"{scale:9.0f} {ref.value:>13} {outlet.value:>9} {amax:10.0e} "
            f"{fname:>5} {r['psi']:13.5g} {r['psi_over_reported']:11.4g} "
            f"{r['flow_residual_relative']:10.1e} {r['speed_max']:8.4f}"
        )


def full_sweep(base: z.Zhao2DSpec, scale: float) -> None:
    print("\n" + "=" * 100)
    print(f"STAGE 2 -- C as well, at length scale x{scale:.0f}")
    print("=" * 100)
    header = (
        f"{'reference':>13} {'source':>13} {'amax':>8} {'thermal tau':>12} "
        f"{'Psi':>12} {'C':>12} {'C/0.0456':>10} {'T_max':>10} {'dE/E':>9}"
    )
    print(header)
    print("-" * len(header))

    from tfopus import fe_thermal as fe_thermal

    spec = scaled_spec(base, scale)
    thermal_forms = (
        ("zhao conv", fe_thermal.ZHAO_FORM),
        ("zhou 2lim", fe_thermal.ZHOU_FORM),
    )
    for ref, src, amax, (tname, tform) in itertools.product(
        list(z.ReferenceField),
        list(z.SourceRegion),
        (base.alpha_max_initial, base.alpha_max_final),
        thermal_forms,
    ):
        opts = za.CaseOptions(
            reference=ref, source=src, alpha_max=amax, thermal_form=tform
        )
        try:
            r = za.analyse(spec, opts)
        except Exception as exc:  # noqa: BLE001
            print(f"{ref.value:>13} {src.value:>13} {amax:8.0e} {tname:>12}  "
                  f"FAILED: {type(exc).__name__}: {exc}")
            continue
        print(
            f"{ref.value:>13} {src.value:>13} {amax:8.0e} {tname:>12} "
            f"{r['psi']:12.5g} {r['compliance']:12.5g} "
            f"{r['compliance_over_reported']:10.4g} {r['temperature_max']:10.4g} "
            f"{r['energy_imbalance_rel']:9.2e}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provenance", action="store_true",
                    help="print the per-field provenance table first")
    ap.add_argument("--stage", choices=("flow", "full", "both"), default="both")
    ap.add_argument(
        "--coarse",
        action="store_true",
        help="2x coarser mesh (1300 elements) for a fast pass",
    )
    ap.add_argument("--scale", type=float, default=1000.0,
                    help="length scale for stage 2")
    args = ap.parse_args()

    base = z.Zhao2DSpec()
    if args.coarse:
        base = dataclasses.replace(base, element_size=base.element_size * 2)

    if args.provenance:
        print(base.provenance())
    print(f"\nRe at the figure-7 scale, inlet half width as L: {base.reynolds():.1f}")

    if args.stage in ("flow", "both"):
        flow_sweep(base, scales=(1.0, 1000.0))
    if args.stage in ("full", "both"):
        full_sweep(base, scale=args.scale)


if __name__ == "__main__":
    main()
