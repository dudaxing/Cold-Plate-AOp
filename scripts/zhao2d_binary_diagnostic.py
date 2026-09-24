"""Re-run the binary diagnostic on a saved design, without redoing the optimisation.

The optimisation result is the design field; the diagnostic is a post-process.
When the diagnostic itself is wrong -- as the first adjacency graph was -- it can
be corrected and re-applied to the stored design at the cost of two solves
rather than the whole run.

    python scripts/zhao2d_binary_diagnostic.py [--fields PATH] [--threshold 0.5]
"""

from __future__ import annotations

import pathlib
import sys

# Default the BLAS thread count BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import json
import pathlib
import sys

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))

from tfopus import zhao2d as z, zhao2d_r1 as r1  # noqa: E402
from tfopus.mesh import Face  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields", type=pathlib.Path,
                    default=REPO / "results" / "zhao2d_r1d_main_fields.npz")
    ap.add_argument("--meta", type=pathlib.Path,
                    default=REPO / "results" / "zhao2d_r1d_main.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--write", action="store_true",
                    help="replace binary_diagnostic in the results JSON")
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    s_grey = np.load(args.fields)["solid_fraction"]

    spec = z.Zhao2DSpec()
    config = r1.R1Config()
    problem = r1.Zhao2DProblem(spec, config)
    reference = r1.load_reference(spec, config)
    alpha_max = meta["final_alpha_max"]

    pm = problem.flow_mesh
    inlet = np.unique([e for e, _ in pm.elem_faces[Face.INLET]])
    outlet = np.unique([e for e, _ in pm.elem_faces[Face.OUTLET]])

    print(f"design from {args.fields.name}, alpha_max {alpha_max:.3e}, "
          f"beta {meta['final_beta']:g}")
    print(f"grey (0.05-0.95) = {np.mean((s_grey > 0.05) & (s_grey < 0.95)):.4f}\n")

    print(f"{'threshold':>10} {'fluid frac':>11} {'comps':>6} "
          f"{'largest':>8} {'inlet comp':>11} {'outlet comp':>12} {'joined':>7}")
    print("-" * 70)
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
        s_bin = (s_grey >= tau).astype(float)
        s_bin[~pm.design_mask] = 0.0
        fluid = s_bin < 0.5
        num, sizes, joined, inl, outl = z.fluid_connectivity(
            pm, fluid, inlet, outlet
        )
        frac = float(fluid[pm.design_mask].mean())
        print(f"{tau:10.2f} {frac:11.4f} {num:6d} {int(sizes.max()):8d} "
              f"{str([int(sizes[i]) for i in inl]):>11} "
              f"{str([int(sizes[i]) for i in outl]):>12} {str(joined):>7}")

    # the reported design, at the nominal threshold
    tau = args.threshold
    s_bin = (s_grey >= tau).astype(float)
    s_bin[~pm.design_mask] = 0.0
    num, sizes, joined, inl, outl = z.fluid_connectivity(
        pm, s_bin < 0.5, inlet, outlet
    )
    fractions = z.fluid_fractions(pm, jnp.asarray(s_bin))
    print(f"\nat threshold {tau}: {num} fluid components, "
          f"sizes {np.sort(sizes)[::-1][:5].tolist()}, joined = {joined}")
    print(f"  v_f design {fractions['v_f_design_domain']:.4f}  "
          f"whole {fractions['v_f_whole_domain']:.4f}")

    result = {
        "threshold": tau,
        "grey_fraction_before": float(np.mean((s_grey > 0.05) & (s_grey < 0.95))),
        "fluid_components": int(num),
        "largest_components": np.sort(sizes)[::-1][:5].tolist(),
        "inlet_outlet_connected": bool(joined),
        **{f"binary_{k}": v for k, v in fractions.items()},
    }

    if not joined:
        result["note"] = (
            "inlet and outlet are NOT connected through fluid after "
            "thresholding; not repaired"
        )
        print("\ninlet and outlet are NOT connected; not repaired.")
        _write(args, meta, result)
        return

    norms = problem.require_converged(jnp.asarray(s_bin), alpha_max)
    psi, c = problem.metrics(jnp.asarray(s_bin), alpha_max)
    w = config.weight
    j = float(w * psi / reference.psi_0 + (1 - w) * c / reference.c_0)
    t = meta["terminal"]
    result.update(
        binary_psi=float(psi),
        binary_compliance=float(c),
        binary_J_self=j,
        binary_flow_residual=norms["flow"],
        binary_thermal_residual=norms["thermal"],
        delta_psi_percent=(float(psi) / t["psi"] - 1) * 100,
        delta_c_percent=(float(c) / t["compliance"] - 1) * 100,
        delta_J_percent=(j / t["J_self"] - 1) * 100,
    )
    print(f"\n{'':22} {'continuous':>14} {'binary':>14} {'change':>10}")
    for name, cont, binv in (
        ("Psi", t["psi"], float(psi)),
        ("C", t["compliance"], float(c)),
        ("J (self scale)", t["J_self"], j),
    ):
        print(f"{name:>22} {cont:14.6g} {binv:14.6g} "
              f"{(binv / cont - 1) * 100:9.2f}%")
    print(f"{'residual flow':>22} {'':>14} {norms['flow']:14.2e}")
    print(f"{'residual thermal':>22} {'':>14} {norms['thermal']:14.2e}")
    _write(args, meta, result)


def _write(args, meta, result) -> None:
    if not args.write:
        print("\n(not written; pass --write)")
        return
    superseded = meta.get("binary_diagnostic")
    meta["binary_diagnostic"] = result
    meta["binary_diagnostic_superseded"] = {
        "reason": (
            "recomputed: the original adjacency graph indexed cells with "
            "round(centre/h). Centres sit at half-integer multiples of h, so "
            "banker's rounding collapsed i and i+1 for odd i and dropped 160 "
            "of 5200 elements from the graph, reporting a connected design as "
            "65 disconnected components. The optimisation itself is unaffected "
            "-- the diagnostic is a post-process on the stored design."
        ),
        "previous": superseded,
    }
    args.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.meta.name} (previous value kept under "
          "binary_diagnostic_superseded)")


if __name__ == "__main__":
    main()
