"""A high-Peclet convection-diffusion benchmark with a known exact solution.

The problem, the exact solution and the measurement rules live in
`tfopus/advection_benchmark.py`; this script runs two mesh series and records
them.

    anisotropic  ny = 8 fixed, nx = 10 ... 320. The original series, kept as it
                 was run. Its first refinement does NOT change h_tau (min edge =
                 hy = 0.03125 at nx = 10 and 20) but does change the aspect
                 ratio, so it cannot be read as a clean convergence study.
    square       ny = nx H / L, so hx = hy = h_tau. The series to compare with
                 the cold plate's square elements.

Reported per mesh: hx, hy, h_tau and the Peclet number on both lengths; the L2
error of T against the closed-form norm; the error in integral k |grad T|^2
(the diffusive half of the compliance); the over/undershoot.

This checks the method against a known answer. It does not say how fine the
cold plate's thermal mesh must be -- see the module docstring.

    python scripts/zhao2d_advection_benchmark.py [--pe 1000]
"""

from __future__ import annotations

import pathlib
import sys

# Default the BLAS thread count BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import dataclasses
import json

import jax

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))

from tfopus import advection_benchmark as ab  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

SERIES = {
    "anisotropic": [(nx, 8) for nx in (10, 20, 40, 80, 160, 320)],
    "square": [(nx, nx // 4) for nx in (8, 16, 32, 64, 128, 256, 512)],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pe", type=float, default=1000.0)
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results",
                    help="directory for the JSON record")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    form = r1.R1_THERMAL_FORM
    norm = ab.exact_norm_sq(args.pe) ** 0.5
    print(f"global Pe = {args.pe:g}, layer width ~ L/Pe = {ab.L / args.pe:.3e}")
    print(f"||T_exact|| = {norm:.15g} (closed form, mesh independent)")
    print(f"thermal form: tau={form.tau}, stabilise_source={form.stabilise_source}, "
          f"supg_heat_capacity={form.supg_heat_capacity}\n")

    record = {
        "note": (
            "Analytic convection-diffusion reference for the SUPG element the "
            "cold plate uses. L2 errors are relative to the CLOSED-FORM norm and "
            "integrated by composite Gauss verified at two subdivision levels. "
            "hx, hy and h_tau are reported separately: the anisotropic series "
            "keeps h_tau fixed over its first refinement. This checks the method "
            "on a known answer; it does not size the cold plate's thermal mesh."
        ),
        "global_pe": args.pe,
        "layer_width": ab.L / args.pe,
        "exact_norm": norm,
        "thermal_form": dataclasses.asdict(form),
        "series": {},
    }

    head = (f"{'nx':>4} {'ny':>4} {'hx':>9} {'hy':>9} {'h_tau':>9} {'Pe_x':>7} "
            f"{'Pe_tau':>7} {'L2 rel':>9} {'int k|gT|^2 err':>15} {'under':>9} "
            f"{'m':>4}")
    for name, meshes in SERIES.items():
        print(f"=== {name} ===")
        print(head)
        print("-" * len(head))
        rows = []
        for nx, ny in meshes:
            r = ab.run(nx, ny, args.pe, form)
            rows.append(r)
            print(f"{r['nx']:4d} {r['ny']:4d} {r['hx']:9.5f} {r['hy']:9.5f} "
                  f"{r['h_tau']:9.5f} {r['pe_x']:7.2f} {r['pe_tau']:7.2f} "
                  f"{r['l2_error_rel']:9.5f} {r['diffusive_rel_error']:15.4%} "
                  f"{r['undershoot']:+9.2e} {r['l2_quadrature']['subintervals']:4d}")
        orders = ab.observed_orders(rows, "hx")
        print("  observed order against hx between successive meshes:")
        for o in orders:
            print(f"    {o['from_nx']:4d} -> {o['to_nx']:4d}   L2 {o['l2']:+.4f}   "
                  f"integral {o['diffusive']:+.4f}")
        print()
        record["series"][name] = {"rows": rows, "observed_orders_vs_hx": orders}

    if not args.no_write:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "zhao2d_r1f_benchmark.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
