"""R1l A: qualified binary baselines for x_300 and x_30, by one export rule.

The designs are the physical densities R1d and R1k SAVED, which the tanh
projection made: R1d's s (x_300, results/zhao2d_r1d_main_fields.npz) and R1k's
terminal s (x_30, results/zhao2d_r1k_fields.npz). Their raw x is never
re-projected by the new default -- that would be a different design, not the
baseline.

One export rule for both (tfopus/zhao2d_binary.py, `volume_threshold`): a
single threshold t per design, fluid where s < t, chosen so the design domain's
fluid volume is at most 40% and as close to it as possible, cells of equal s
moving together, the tabs fluid, no repair. t is an export threshold, not a
projection's eta. A design qualifies if it meets the bound and joins inlet and
outlet through fluid; each qualified design is then solved once as a GIVEN s
-- never filtered or projected again -- with the flow on h (2x2) and the
temperature on h/4 (3x3), alpha_max = 1e7, gated at 1e-8, every state saved.
J uses this model's frozen reference; J* the single-mesh one.

    python scripts/zhao2d_r1l_baselines.py [--inputs results] [--out DIR] [--overwrite]
"""

from __future__ import annotations

import pathlib
import sys

# Default the BLAS thread count BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import json
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))
sys.path.insert(0, str(REPO / "scripts"))

from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_binary as zb  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file  # noqa: E402

ALPHA_MAX = 1.0e7
REFINEMENT, QUADRATURE = 4, 3
GATE = 1e-8
PROBES = (0.3, 0.4, 0.5, 0.6, 0.7)
RECORD = "zhao2d_r1l_baselines.json"
FIELDS = "zhao2d_r1l_baselines_fields.npz"
INPUTS = ("zhao2d_r1d_main_fields.npz", "zhao2d_r1k_fields.npz",
          "zhao2d_r1k_warm_start.json", "zhao2d_r1k_terminal_check.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=pathlib.Path, default=REPO / "results")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    inputs, out = args.inputs, args.out
    if (out / RECORD).exists() and not args.overwrite:
        sys.exit(f"{out / RECORD} exists and is cited evidence; write elsewhere with "
                 "--out DIR, or pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    spec, config = z.Zhao2DSpec(), r1.R1Config()
    w = config.weight
    r1k = json.loads((inputs / "zhao2d_r1k_warm_start.json").read_text(encoding="utf-8"))
    term = json.loads((inputs / "zhao2d_r1k_terminal_check.json").read_text(encoding="utf-8"))
    designs = {
        "x300": {"s": np.asarray(np.load(inputs / "zhao2d_r1d_main_fields.npz")["solid_fraction"]),
                 "source": "results/zhao2d_r1d_main_fields.npz: solid_fraction (R1d, tanh beta 8)",
                 "continuous_h4": {"psi": 0.014351143070216229, "compliance": 36180.16029596453,
                                   "source": "R1j baseline"},
                 "threshold_0.5_h4": term["cells"]["x300/binary/h4"]},
        "x30": {"s": np.asarray(np.load(inputs / "zhao2d_r1k_fields.npz")["solid_fraction"]),
                "source": "results/zhao2d_r1k_fields.npz: solid_fraction (R1k terminal, tanh beta 8)",
                "continuous_h4": {"psi": r1k["terminal"]["psi"],
                                  "compliance": r1k["terminal"]["compliance"],
                                  "source": "R1k terminal record"},
                "threshold_0.5_h4": term["cells"]["x30/binary/h4"]},
    }

    record = {
        "note": ("R1l A: qualified binary baselines. The saved tanh physical densities of "
                 "x_300 (R1d) and x_30 (R1k terminal), each exported by one volume-threshold "
                 "rule and, if qualified and connected, solved once as a given s on flow h / "
                 "thermal h/4. No projection is applied to the binary designs."),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in INPUTS},
        "rule": {"name": "volume_threshold", "max_fluid_fraction": config.max_fluid_fraction,
                 "fluid_where": "s < t", "ties": "cells of equal s move together (solid)",
                 "tabs": "fluid", "repair": "none"},
        "alpha_max": ALPHA_MAX,
        "model": {"h_flow": spec.element_size, "thermal_refinement": REFINEMENT,
                  "thermal_quadrature": QUADRATURE},
    }
    here = pathlib.Path(__file__).resolve()
    record["provenance"]["source_sha256"][here.relative_to(REPO).as_posix()] = sha256_file(here)

    t0 = time.perf_counter()
    problem = dual.Zhao2DDualProblem(spec, config, REFINEMENT, QUADRATURE)
    t_build = time.perf_counter() - t0
    print(f"built: flow {problem.flow_mesh.num_elems} el, thermal "
          f"{problem.thermal_mesh.num_elems} el ({t_build:.1f} s)", flush=True)
    ref_path = dual.reference_file(REFINEMENT, QUADRATURE)
    reference = dual.load_reference(problem, ref_path)
    single = r1.load_reference(spec, config)
    record["reference"] = {"file": ref_path.relative_to(REPO).as_posix(),
                           "sha256": sha256_file(ref_path),
                           "psi_0": reference.psi_0, "c_0": reference.c_0}
    design_mask = np.asarray(problem.flow_mesh.design_mask)
    areas = np.asarray(problem.flow_mesh.elem_area)

    cells, saved = {}, {}
    for name, d in designs.items():
        export = zb.volume_threshold(d["s"], design_mask, areas, config.max_fluid_fraction)
        s_bin = export.pop("s_binary")
        conn = zb.connectivity(problem.flow_mesh, d["s"], (export["t"],))[f"{export['t']:g}"]
        cell = {
            "source": d["source"],
            "s_sha256": fs.digest(d["s"]),
            **export,
            "binary_sha256": fs.digest(s_bin),
            "volume_feasible": zb.volume_feasible(export["fluid_fraction"],
                                                  config.max_fluid_fraction),
            "connected": bool(conn["inlet_outlet_connected"]),
            "fluid_components": conn["fluid_components"],
            "probes_on_saved_s": zb.connectivity(problem.flow_mesh, d["s"], PROBES),
        }
        cell["qualified"] = bool(cell["volume_feasible"] and cell["connected"])
        saved[f"{name}_solid_fraction_binary"] = s_bin
        if not cell["qualified"]:
            cell["analysed"] = False
            cells[name] = cell
            print(f"{name}: t {export['t']:.10f}, not qualified "
                  f"(feasible {cell['volume_feasible']}, connected {cell['connected']})", flush=True)
            continue

        t0 = time.perf_counter()
        pv, temperature, alpha, kappa = problem.solve_states(jnp.asarray(s_bin), ALPHA_MAX)
        seconds = time.perf_counter() - t0
        norms = problem.residual_norms_at(pv, temperature, alpha, kappa)
        psi = float(problem.flow.dissipated_power(pv, alpha))
        c = float(problem.thermal.thermal_compliance(
            temperature, problem.thermal_velocity(pv), kappa))
        t_arr = np.asarray(temperature)
        cell.update(
            analysed=True, seconds=seconds, psi=psi, compliance=c,
            J=w * psi / reference.psi_0 + (1 - w) * c / reference.c_0,
            J_star=float(dual.reporting_objective(psi, c, single, w)),
            T_max=float(t_arr.max()), T_min=float(t_arr.min()),
            negative_nodes=int(np.count_nonzero(t_arr < 0.0)),
            residual_flow=norms["flow"], residual_thermal=norms["thermal"],
            gate_passed=zb.gate_passed(norms, GATE),
            state_sha256={"press_vel": fs.digest(pv), "temperature": fs.digest(t_arr)},
        )
        cont, diag = d["continuous_h4"], d["threshold_0.5_h4"]
        cont_j = w * cont["psi"] / reference.psi_0 + (1 - w) * cont["compliance"] / reference.c_0
        cell["against_continuous_h4"] = {
            "source": cont["source"],
            "J": cell["J"] / cont_j - 1.0,
            "psi": psi / cont["psi"] - 1.0,
            "compliance": c / cont["compliance"] - 1.0}
        cell["against_threshold_0.5_h4"] = {
            "note": "the terminal check's s = 0.5 diagnostic of the same saved s",
            "J": cell["J"] / diag["J"] - 1.0,
            "psi": psi / diag["psi"] - 1.0,
            "compliance": c / diag["compliance"] - 1.0}
        saved[f"{name}_press_vel"] = np.asarray(pv)
        saved[f"{name}_temperature_h4"] = t_arr
        cells[name] = cell
        print(f"{name}: t {export['t']:.10f}  fluid {export['fluid_cells']}  changed "
              f"{export['cells_changed_from_0.5']}  Psi {psi:.6e}  C {c:.3f}  J {cell['J']:.6f}  "
              f"|R| {norms['flow']:.1e}/{norms['thermal']:.1e}  {seconds:.1f} s", flush=True)

    record["designs"] = cells
    comparison = {"kind": zb.comparison_kind(cells.get("x300"), cells.get("x30"))}
    if comparison["kind"] is not None:
        comparison.update({q: cells["x30"][q] / cells["x300"][q] - 1.0
                           for q in ("J", "J_star", "psi", "compliance")})
    record["x30_against_x300"] = comparison
    record["checks"] = zb.check_failures(cells)
    record["cost"] = {"t_build_s": t_build, "peak_working_set_mb": peak_working_set_mb(),
                      "wall_clock_total_s": time.perf_counter() - t_start}
    np.savez_compressed(out / FIELDS, elem_centres=np.asarray(problem.flow_mesh.elem_centres),
                        thermal_node_coords_h4=np.asarray(problem.thermal_mesh.mesh.nodes.coords),
                        **saved)
    (out / RECORD).write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")

    if comparison["kind"] is not None:
        print(f"\nx_30 against x_300, qualified binary designs ({comparison['kind']}): "
              f"J {comparison['J']:+.3%}, Psi {comparison['psi']:+.3%}, "
              f"C {comparison['compliance']:+.3%}")
    print(f"wrote {out / RECORD} and {out / FIELDS}; total "
          f"{record['cost']['wall_clock_total_s']:.0f} s")
    if any(record["checks"].values()):
        sys.exit(f"CHECKS FAILED: {record['checks']}")


if __name__ == "__main__":
    main()
