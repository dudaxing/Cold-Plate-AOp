"""R1l's three qualified binary designs on thermal h/8: does the ranking hold?

The designs are exactly the ones R1l evaluated, taken from its fields files:
x_300's and x_30's qualified binary baselines (R1l A) and the pilot's
qualified binary terminal (R1l B). Each is a given s -- never filtered or
projected -- with the flow R1l solved for it on h.

Before anything is solved, the saved states are checked: each flow is
re-verified against its s (Dirichlet values exact, relative residual within
the gate), and C, Psi and the residuals at h/4 are recomputed from the saved
h/4 temperature and must reproduce R1l's records. If any of that fails, the
record is written with the failures and the script exits non-zero before h/8
is built. Then, per design, one temperature solve on h/8 (3x3) with that same
flow, gated at 1e-8 with every residual required finite, and every h/8 state
saved.

J uses the h/4 model's frozen constants as one fixed yardstick on both meshes
-- the same weights, not a new normalisation; J* the single-mesh ones. The
ranking at h/8 is compared with the ranking at h/4. Differences between the
two meshes are neighbouring-mesh differences, not errors against an exact
solution, and the flow stays on h.

    python scripts/zhao2d_r1l_h8_check.py [--inputs results] [--out DIR] [--overwrite]
"""

from __future__ import annotations

import pathlib
import sys

# Default the BLAS thread count BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import faulthandler
import gc
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

from tfopus import materials  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_binary as zb  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file  # noqa: E402

ALPHA_MAX, QUADRATURE = 1.0e7, 3
GATE, ANCHOR_TOL = 1e-8, 1e-12
RECORD = "zhao2d_r1l_h8_check.json"
FIELDS = "zhao2d_r1l_h8_fields.npz"
STACKS = "zhao2d_r1l_h8_stacks.log"
INPUTS = ("zhao2d_r1l_baselines.json", "zhao2d_r1l_baselines_fields.npz",
          "zhao2d_r1l_vp_pilot.json", "zhao2d_r1l_vp_pilot_fields.npz")
PAIRS = (("pilot", "x300"), ("pilot", "x30"), ("x30", "x300"))


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
    stacks = open(out / STACKS, "w", encoding="utf-8")
    faulthandler.dump_traceback_later(2400, repeat=True, file=stacks)  # longer than a healthy run
    t_start = time.perf_counter()

    spec, config = z.Zhao2DSpec(), r1.R1Config()
    w = config.weight
    a_rec = json.loads((inputs / "zhao2d_r1l_baselines.json").read_text(encoding="utf-8"))
    b_rec = json.loads((inputs / "zhao2d_r1l_vp_pilot.json").read_text(encoding="utf-8"))
    fa = np.load(inputs / "zhao2d_r1l_baselines_fields.npz")
    fb = np.load(inputs / "zhao2d_r1l_vp_pilot_fields.npz")
    designs = {
        "x300": {"s": fa["x300_solid_fraction_binary"], "press_vel": fa["x300_press_vel"],
                 "temperature_h4": fa["x300_temperature_h4"], "h4": a_rec["designs"]["x300"],
                 "source": "R1l A, x_300's qualified binary baseline"},
        "x30": {"s": fa["x30_solid_fraction_binary"], "press_vel": fa["x30_press_vel"],
                "temperature_h4": fa["x30_temperature_h4"], "h4": a_rec["designs"]["x30"],
                "source": "R1l A, x_30's qualified binary baseline"},
        "pilot": {"s": fb["solid_fraction_binary"], "press_vel": fb["binary_press_vel"],
                  "temperature_h4": fb["binary_temperature_h4"], "h4": b_rec["export"],
                  "source": "R1l B, the pilot's qualified binary terminal"},
    }
    for name, d in designs.items():
        if not (d["h4"].get("qualified", True) and d["h4"].get("volume_feasible")):
            sys.exit(f"{name} is not a qualified design in its record; stopping")

    record = {
        "note": ("R1l's three qualified binary designs, given s with their saved flows on h, "
                 "re-checked at h/4 from the saved states and solved once on thermal h/8 "
                 "(3x3). J uses the h/4 model's constants as one fixed yardstick; J* the "
                 "single-mesh ones. Nothing optimised, no reference frozen."),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in INPUTS},
        "alpha_max": ALPHA_MAX, "thermal_quadrature": QUADRATURE,
    }
    here = pathlib.Path(__file__).resolve()
    record["provenance"]["source_sha256"][here.relative_to(REPO).as_posix()] = sha256_file(here)
    single = r1.load_reference(spec, config)
    material = za.build_material(spec, ALPHA_MAX)

    cells, saved, meshes = {}, {}, {}
    yard = {}

    def finish() -> None:
        record["cost"] = {"peak_working_set_mb": peak_working_set_mb(),
                          "wall_clock_total_s": time.perf_counter() - t_start}
        faulthandler.cancel_dump_traceback_later()
        stacks.close()
        if (out / STACKS).stat().st_size == 0:
            (out / STACKS).unlink()
        (out / RECORD).write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")

    for level in (4, 8):
        t0 = time.perf_counter()
        problem = dual.Zhao2DDualProblem(spec, config, level, QUADRATURE)
        coords = np.asarray(problem.thermal_mesh.mesh.nodes.coords)
        meshes[f"h{level}"] = {"num_elems": int(problem.thermal_mesh.num_elems),
                               "num_nodes": int(problem.thermal_mesh.mesh.num_nodes),
                               "node_coords_sha256": fs.digest(coords),
                               "t_build_s": time.perf_counter() - t0}
        print(f"built thermal h/{level}: {problem.thermal_mesh.num_elems} el "
              f"({meshes[f'h{level}']['t_build_s']:.1f} s)", flush=True)
        if not yard:
            ref = dual.load_reference(problem, dual.reference_file(4, QUADRATURE))
            yard.update(psi_0=ref.psi_0, c_0=ref.c_0,
                        file=dual.reference_file(4, QUADRATURE).relative_to(REPO).as_posix())
            record["yardstick"] = dict(yard)

        for name, d in designs.items():
            s = jnp.asarray(d["s"])
            pv = jnp.asarray(d["press_vel"])
            alpha = materials.brinkman_penalty(s, material)
            kappa = problem.thermal_conductivity(s, ALPHA_MAX)
            cell = {"design": name, "source": d["source"], "thermal_refinement": level,
                    "s_sha256": fs.digest(d["s"]), "volume_feasible": True,
                    "fluid_cells": int(d["h4"]["fluid_cells"])}
            if level == 4:
                # the saved states, checked as given: nothing is solved here
                cell["flow_verification"] = fs.verify_flow_state(problem, pv, s, ALPHA_MAX)
                temperature = jnp.asarray(d["temperature_h4"])
                cell["t_thermal_s"] = 0.0
            else:
                t0 = time.perf_counter()
                temperature = problem.solve_thermal(pv, s, ALPHA_MAX)
                cell["t_thermal_s"] = time.perf_counter() - t0
            norms = problem.residual_norms_at(pv, temperature, alpha, kappa)
            psi = float(problem.flow.dissipated_power(pv, alpha))
            c = float(problem.thermal.thermal_compliance(
                temperature, problem.thermal_velocity(pv), kappa))
            t_arr = np.asarray(temperature)
            cell.update(
                analysed=True, psi=psi, compliance=c,
                J=w * psi / yard["psi_0"] + (1 - w) * c / yard["c_0"],
                J_star=float(dual.reporting_objective(psi, c, single, w)),
                T_max=float(t_arr.max()), T_min=float(t_arr.min()),
                negative_nodes=int(np.count_nonzero(t_arr < 0.0)),
                residual_flow=norms["flow"], residual_thermal=norms["thermal"],
                gate_passed=zb.gate_passed(norms, GATE),
                temperature_sha256=fs.digest(t_arr))
            if level == 4:
                cell["anchor"] = zb.anchor_status(
                    {"psi": psi, "compliance": c},
                    {"psi": d["h4"]["psi"], "compliance": d["h4"]["compliance"],
                     "source": f"R1l record, {name} at h/4"}, ANCHOR_TOL)
            else:
                saved[f"{name}_temperature_h8"] = t_arr
            cells[f"{name}/h{level}"] = cell
            print(f"{name}/h{level}: Psi {psi:.6e}  C {c:.3f}  J {cell['J']:.6f}  T_max "
                  f"{cell['T_max']:.2f}  T_min {cell['T_min']:+.3f}  |R| {norms['flow']:.1e}/"
                  f"{norms['thermal']:.1e}  thermal {cell['t_thermal_s']:.1f} s"
                  + (f"  anchor {'ok' if cell['anchor']['reproduced'] else 'MISMATCH'}"
                     if level == 4 else ""), flush=True)
        del problem
        gc.collect()
        if level == 4:
            # the saved states must check out before anything is solved on h/8
            checks = zb.check_failures(cells)
            if any(checks.values()):
                record.update(thermal_meshes=meshes, cells=cells, checks=checks,
                              stopped="the h/4 checks failed; nothing was built or solved on h/8")
                finish()
                sys.exit(f"CHECKS FAILED at h/4, before anything was solved on h/8: {checks}")

    record["thermal_meshes"] = meshes
    record["cells"] = cells
    (out / RECORD).write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
    checks = zb.check_failures(cells)

    ranking = {}
    for a, b in PAIRS:
        row = {}
        for level in (4, 8):
            ca, cb = cells[f"{a}/h{level}"], cells[f"{b}/h{level}"]
            kind = zb.comparison_kind(ca, cb)
            row[f"h{level}"] = None if kind is None else {
                "kind": kind, **{q: ca[q] / cb[q] - 1.0 for q in ("J", "J_star", "psi", "compliance")}}
        ranking[f"{a}_against_{b}"] = row
    record["ranking"] = ranking
    record["thermal_step_h4_to_h8"] = {
        name: {q: cells[f"{name}/h8"][q] / cells[f"{name}/h4"][q] - 1.0 for q in ("J", "compliance")}
        for name in designs}
    record["checks"] = checks

    np.savez_compressed(out / FIELDS, **saved)
    finish()

    print("\nranking, qualified binary designs (J, same yardstick):")
    for k, row in ranking.items():
        if row["h4"] is None or row["h8"] is None:
            print(f"  {k:22s} not a usable comparison: {row}")
            continue
        print(f"  {k:22s} h/4 {row['h4']['J']:+.3%}   h/8 {row['h8']['J']:+.3%}   "
              f"(C h/4 {row['h4']['compliance']:+.3%}, h/8 {row['h8']['compliance']:+.3%})")
    for name, v in record["thermal_step_h4_to_h8"].items():
        print(f"  {name:6s} h/4 -> h/8: C {v['compliance']:+.3%}, J {v['J']:+.3%}")
    print(f"\nwrote {out / RECORD} and {out / FIELDS}; total "
          f"{record['cost']['wall_clock_total_s']:.0f} s")
    if any(checks.values()):
        sys.exit(f"CHECKS FAILED: {checks}")


if __name__ == "__main__":
    main()
