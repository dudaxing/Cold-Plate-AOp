"""Freeze Psi_0 and C_0 for a dual-mesh model, into a versioned file of its own.

The single-mesh reference, tfopus/zhao2d_reference.json, is left as it is: it
stays the common reporting scale J*. This is the normalisation of the dual-mesh
model named on the command line -- by default R1j's development model, flow on
the design mesh h and temperature on h/4 at 3x3 -- and only that model accepts
it (`zhao2d_dual.load_reference` checks an identity that includes the thermal
mesh).

The reference state is the configured physical density set directly -- gamma =
0.4 in the design domain, the tabs fluid -- at alpha_max_reference = 1e6, never
routed through the filter or the projection. It is solved once and gated on the
states that solve returned. A file that exists is never overwritten: a
refrozen reference is a new version.

With w = 0.5 fixed, a new C_0 changes the weight C carries against Psi in J; it
is a different objective, not the old one in new units. The shift is printed.

    python scripts/zhao2d_freeze_dual_reference.py [--thermal-refinement 4]
        [--thermal-quadrature 3] [--version 1] [--write]
"""

from __future__ import annotations

import pathlib
import sys

# Default the BLAS thread count BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))

from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thermal-refinement", type=int, default=4)
    ap.add_argument("--thermal-quadrature", type=int, default=3)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    spec, config = z.Zhao2DSpec(), r1.R1Config()
    r, q = args.thermal_refinement, args.thermal_quadrature
    print(f"dual-mesh reference: flow on h = {spec.element_size:g} (2x2), temperature "
          f"on h/{r} ({q}x{q}); reference state {config.reference}, alpha_max "
          f"{config.alpha_max_reference:.0e}\n")

    t0 = time.perf_counter()
    values, report, problem = dual.freeze_reference(spec, config, r, q)
    seconds = time.perf_counter() - t0
    print(f"  Psi_0 = {values.psi_0!r}")
    print(f"  C_0   = {values.c_0!r}")
    print(f"  |R|/|R0| flow {values.flow_residual_relative:.2e}, thermal "
          f"{values.thermal_residual_relative:.2e}  (gate 1e-8)")
    print(f"  v_f design {report['v_f_design_domain']:.6f}, whole "
          f"{report['v_f_whole_domain']:.6f}; T_max {report['t_max']:.6f}; "
          f"{report['thermal_elements']} thermal elements; {seconds:.1f} s")

    single = r1.load_reference(spec, config)
    print(f"\nagainst the single-mesh reference (h, 2x2), which stays the J* scale:")
    print(f"  Psi_0 ratio {values.psi_0 / single.psi_0:.15f}  (same flow problem)")
    print(f"  C_0 ratio   {values.c_0 / single.c_0:.6f}")
    print(f"  so at w = 0.5 the C term of J weighs {single.c_0 / values.c_0:.4f} of what it "
          "did against Psi: a different objective, not a change of units")

    if not args.write:
        print("\n(not written; pass --write)")
        return
    path = dual.reference_file(r, q, args.version)
    if path.exists():
        sys.exit(f"{path.name} exists; a refrozen reference is a new --version, "
                 "never an overwrite")
    path.write_text(values.to_json(), encoding="utf-8")
    back = dual.load_reference(problem, path)
    if back != values:
        sys.exit("the written reference does not read back identically")
    print(f"\nwritten to {path.relative_to(REPO)}; read back through the identity "
          "check")


if __name__ == "__main__":
    main()
