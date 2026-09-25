"""R1k terminal check: does the warm start's gain survive h/8 and thresholding?

Two designs, both from R1k's record: the start x_300 (R1d's design) and the
terminal x_30. Each is evaluated four ways, all at alpha_max = 1e7 and beta = 8
with the flow on h (2x2):

                  continuous s             binary s (threshold 0.5)
  thermal h/4     R1k's model              the flow re-solved for the binary s
  thermal h/8     R1i's level (3x3)        the same binary flow

Each flow is solved once and shared by both thermal meshes. Three continuous
cells are anchors, recomputed here: x_300 at h/4 must reproduce R1j's
baseline, x_30 at h/4 R1k's terminal record, and x_300 at h/8 R1i's cell.

The binary design is the thresholded solid fraction, the fixed tabs kept fluid,
with no repair; if inlet and outlet are not joined through fluid it is reported
as such and not analysed. Fluid connectivity is also recorded for thresholds
0.3 to 0.7. Every state is gated at 1e-8; a cell that fails is recorded and not
used.

J uses R1k's constants -- this model's Psi_0 and C_0 at w = 0.5 -- as one fixed
yardstick on every cell: the same weights, not a new normalisation. J* uses
the single-mesh ones. A cell's gain is x_30 against x_300 under the same
evaluation. Nothing is optimised and no reference is frozen.

    python scripts/zhao2d_r1k_terminal_check.py [--inputs results] [--out DIR]
        [--overwrite]
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

import toflux.src.solver as _solver  # noqa: E402

from tfopus import materials  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402
from tfopus.mesh import Face  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file  # noqa: E402

ALPHA_MAX, BETA = 1.0e7, 8.0
REFINEMENTS, QUADRATURE = (4, 8), 3
THRESHOLD = 0.5
CONNECTIVITY_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
GATE, ANCHOR_TOL = 1e-8, 1e-9
RECORD = "zhao2d_r1k_terminal_check.json"
FIELDS = "zhao2d_r1k_terminal_fields.npz"
STACKS = "zhao2d_r1k_terminal_stacks.log"
INPUTS = ("zhao2d_r1k_warm_start.json", "zhao2d_r1k_fields.npz", "zhao2d_r1i_h8.json")


def port_elements(planar, tag) -> np.ndarray:
    return np.unique([e for e, _ in planar.elem_faces[tag]])


def binary(s: np.ndarray, design_mask: np.ndarray, threshold: float) -> np.ndarray:
    """Solid where s >= threshold; the fixed tabs fluid by definition. No repair."""
    s_bin = (s >= threshold).astype(float)
    s_bin[~design_mask] = 0.0
    return s_bin


def connectivity(problem, s: np.ndarray) -> dict:
    pm = problem.flow_mesh
    inlet, outlet = port_elements(pm, Face.INLET), port_elements(pm, Face.OUTLET)
    out = {}
    for t in CONNECTIVITY_THRESHOLDS:
        fluid = binary(s, np.asarray(pm.design_mask), t) < 0.5
        n, sizes, joined, ids_in, ids_out = z.fluid_connectivity(pm, fluid, inlet, outlet)
        out[f"{t:g}"] = {"fluid_components": int(n), "inlet_outlet_connected": joined,
                         "largest_components": [int(v) for v in np.sort(sizes)[::-1][:3]]}
    return out


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
    # A hang leaves every thread's stack behind; the interval is longer than a
    # healthy run (~20 min), so a dump means something stopped.
    stacks = open(out / STACKS, "w", encoding="utf-8")
    faulthandler.dump_traceback_later(1500, repeat=True, file=stacks)
    t_start = time.perf_counter()
    memory = {}

    spec, config = z.Zhao2DSpec(), r1.R1Config()
    w = config.weight
    r1k = json.loads((inputs / "zhao2d_r1k_warm_start.json").read_text(encoding="utf-8"))
    f = np.load(inputs / "zhao2d_r1k_fields.npz")
    r1i = json.loads((inputs / "zhao2d_r1i_h8.json").read_text(encoding="utf-8"))
    if (r1k["contract"]["alpha_max"], r1k["contract"]["beta"]) != (ALPHA_MAX, BETA):
        sys.exit("R1k's record is not at alpha_max 1e7, beta 8; stopping")
    designs = {"x300": np.asarray(f["initial_design"]), "x30": np.asarray(f["design"])}
    anchors = {
        ("x300", "continuous", 4): {"psi": 0.014351143070216229,
                                    "compliance": 36180.16029596453,
                                    "source": "R1j baseline (results/zhao2d_r1j_check.json)"},
        ("x30", "continuous", 4): {"psi": r1k["terminal"]["psi"],
                                   "compliance": r1k["terminal"]["compliance"],
                                   "source": "R1k terminal record"},
        ("x300", "continuous", 8): {"psi": r1i["cell"]["psi"],
                                    "compliance": r1i["cell"]["compliance"],
                                    "source": "R1i cell (results/zhao2d_r1i_h8.json)"},
    }

    single = r1.load_reference(spec, config)
    yardstick = {}  # R1k's reference, loaded through its identity check at h/4

    record = {
        "note": (
            "R1k terminal check: x_300 and R1k's terminal x_30, each continuous and "
            "thresholded at 0.5, each on thermal h/4 and h/8; flow on h, alpha_max "
            "1e7, beta 8. J uses R1k's constants as one fixed yardstick on every cell; "
            "J* the single-mesh ones. Nothing optimised, no reference frozen."
        ),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in INPUTS},
        "single_mesh_scale": {"psi_0": single.psi_0, "c_0": single.c_0},
        "alpha_max": ALPHA_MAX, "beta": BETA, "threshold": THRESHOLD,
        "thermal_quadrature": QUADRATURE,
    }
    here = pathlib.Path(__file__).resolve()
    record["provenance"]["source_sha256"][here.relative_to(REPO).as_posix()] = sha256_file(here)

    def objective(psi, c):
        return w * psi / yardstick["psi_0"] + (1.0 - w) * c / yardstick["c_0"]

    cells, flows, saved, build = {}, {}, {}, {}
    for level in REFINEMENTS:
        t0 = time.perf_counter()
        problem = dual.Zhao2DDualProblem(spec, config, level, QUADRATURE)
        build[f"h{level}"] = time.perf_counter() - t0
        memory[f"after_build_h{level}"] = peak_working_set_mb()
        print(f"built thermal h/{level}: {problem.thermal_mesh.num_elems} el, "
              f"{problem.thermal_mesh.mesh.num_nodes} nodes ({build[f'h{level}']:.1f} s)",
              flush=True)
        if not yardstick:
            path = dual.reference_file(4, QUADRATURE)
            ref = dual.load_reference(problem, path)
            yardstick.update(psi_0=ref.psi_0, c_0=ref.c_0, weight=w,
                             file=path.relative_to(REPO).as_posix(),
                             sha256=sha256_file(path))
            record["yardstick"] = dict(yardstick)
        material = za.build_material(spec, ALPHA_MAX)
        design_mask = np.asarray(problem.flow_mesh.design_mask)

        for name, x in designs.items():
            s = np.asarray(problem.solid_fraction(jnp.asarray(x), BETA))
            if level == REFINEMENTS[0]:
                record.setdefault("designs", {})[name] = {
                    "sha256_float64": fs.digest(x),
                    "connectivity": connectivity(problem, s),
                }
            for version in ("continuous", "binary"):
                key = f"{name}/{version}/h{level}"
                s_v = s if version == "continuous" else binary(s, design_mask, THRESHOLD)
                cell = {"design": name, "version": version, "thermal_refinement": level,
                        **z.fluid_fractions(problem.flow_mesh, jnp.asarray(s_v)),
                        "grey_fraction": float(np.mean((s_v > 0.05) & (s_v < 0.95)))}
                cell["constraint_g"] = cell["v_f_design_domain"] / config.max_fluid_fraction - 1.0
                if version == "binary" and not record["designs"][name]["connectivity"][
                        f"{THRESHOLD:g}"]["inlet_outlet_connected"]:
                    cell["analysed"] = False
                    cell["note"] = "inlet and outlet not joined through fluid; not repaired"
                    cells[key] = cell
                    print(f"{key}: disconnected, not analysed", flush=True)
                    continue

                alpha = materials.brinkman_penalty(jnp.asarray(s_v), material)
                if (name, version) not in flows:  # each flow once, shared by h/4 and h/8
                    t0 = time.perf_counter()
                    flows[(name, version)] = np.asarray(_solver.modified_newton_raphson_solve(
                        problem.flow, problem.flow_x0, alpha))
                    cell["t_flow_s"] = time.perf_counter() - t0
                pv = jnp.asarray(flows[(name, version)])
                t0 = time.perf_counter()
                temperature = problem.solve_thermal(pv, jnp.asarray(s_v), ALPHA_MAX)
                cell["t_thermal_s"] = time.perf_counter() - t0
                kappa = problem.thermal_conductivity(jnp.asarray(s_v), ALPHA_MAX)
                norms = problem.residual_norms_at(pv, temperature, alpha, kappa)
                psi = float(problem.flow.dissipated_power(pv, alpha))
                c = float(problem.thermal.thermal_compliance(
                    temperature, problem.thermal_velocity(pv), kappa))
                t = np.asarray(temperature)
                cell.update(
                    analysed=True, psi=psi, compliance=c, J=objective(psi, c),
                    J_star=float(dual.reporting_objective(psi, c, single, w)),
                    T_max=float(t.max()), T_min=float(t.min()),
                    negative_nodes=int(np.count_nonzero(t < 0.0)),
                    residual_flow=norms["flow"], residual_thermal=norms["thermal"],
                    gate_passed=bool(max(norms.values()) <= GATE),
                )
                anchor = anchors.get((name, version, level))
                if anchor is not None:
                    cell["anchor"] = {
                        "source": anchor["source"],
                        "psi_relative": psi / anchor["psi"] - 1.0,
                        "compliance_relative": c / anchor["compliance"] - 1.0,
                    }
                    cell["anchor"]["reproduced"] = bool(
                        abs(cell["anchor"]["psi_relative"]) <= ANCHOR_TOL
                        and abs(cell["anchor"]["compliance_relative"]) <= ANCHOR_TOL)
                cells[key] = cell
                if level == 8 and name == "x30":
                    saved[f"temperature_h8_{version}"] = t
                print(f"{key}: Psi {psi:.6e}  C {c:.3f}  J {cell['J']:.6f}  "
                      f"T_max {cell['T_max']:.2f}  |R| {max(norms.values()):.1e}  "
                      f"thermal {cell['t_thermal_s']:.1f} s"
                      + (f"  anchor {'ok' if cell['anchor']['reproduced'] else 'MISMATCH'}"
                         if anchor is not None else ""), flush=True)
            if level == REFINEMENTS[0]:
                saved[f"solid_fraction_binary_{name}"] = binary(s, design_mask, THRESHOLD)
        memory[f"after_h{level}"] = peak_working_set_mb()
        # the h/8 node coordinates are R1i's (results/zhao2d_r1i_fields.npz): same mesh
        del problem
        gc.collect()

    record["cells"] = cells
    # the solves are the expensive part: keep them even if what follows fails
    (out / RECORD).write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
    bad_anchor = [k for k, c in cells.items() if "anchor" in c and not c["anchor"]["reproduced"]]
    failed_gate = [k for k, c in cells.items() if c.get("analysed") and not c["gate_passed"]]

    def usable(key):
        c = cells.get(key)
        return c is not None and c.get("analysed") and c["gate_passed"]

    gains, penalties = {}, {}
    for version in ("continuous", "binary"):
        for level in REFINEMENTS:
            a, b = f"x300/{version}/h{level}", f"x30/{version}/h{level}"
            if usable(a) and usable(b):
                gains[f"{version}/h{level}"] = {
                    q: cells[b][q] / cells[a][q] - 1.0
                    for q in ("J", "J_star", "psi", "compliance")}
    for name in designs:
        for level in REFINEMENTS:
            a, b = f"{name}/continuous/h{level}", f"{name}/binary/h{level}"
            if usable(a) and usable(b):
                penalties[f"{name}/h{level}"] = {
                    q: cells[b][q] / cells[a][q] - 1.0 for q in ("J", "psi", "compliance")}
    for name in designs:
        a, b = f"{name}/continuous/h4", f"{name}/continuous/h8"
        if usable(a) and usable(b):
            record.setdefault("thermal_step_h4_to_h8", {})[name] = {
                "compliance": cells[b]["compliance"] / cells[a]["compliance"] - 1.0,
                "J": cells[b]["J"] / cells[a]["J"] - 1.0}
    record["gain_x30_over_x300"] = gains
    record["thresholding_penalty"] = penalties
    record["checks"] = {"anchors_not_reproduced": bad_anchor, "cells_failing_gate": failed_gate}
    record["cost"] = {"t_build_s": build,
                      "peak_working_set_mb_cumulative": memory,
                      "wall_clock_total_s": time.perf_counter() - t_start}

    np.savez_compressed(out / FIELDS, **saved)
    faulthandler.cancel_dump_traceback_later()
    stacks.close()
    if (out / STACKS).stat().st_size == 0:
        (out / STACKS).unlink()
    (out / RECORD).write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")

    print("\ngain of x_30 over x_300, same evaluation:")
    for k, v in gains.items():
        print(f"  {k:16s} J {v['J']:+.3%}  J* {v['J_star']:+.3%}  Psi {v['psi']:+.3%}  "
              f"C {v['compliance']:+.3%}")
    print("thresholding penalty (binary over continuous):")
    for k, v in penalties.items():
        print(f"  {k:10s} J {v['J']:+.3%}  Psi {v['psi']:+.3%}  C {v['compliance']:+.3%}")
    if bad_anchor or failed_gate:
        print(f"\nCHECKS FAILED: anchors {bad_anchor}, gate {failed_gate}")
    print(f"\nwrote {out / RECORD} and {out / FIELDS}; total "
          f"{record['cost']['wall_clock_total_s']:.0f} s")


if __name__ == "__main__":
    main()
