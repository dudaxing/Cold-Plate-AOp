"""R1i: one more thermal level, h_T = h/8, on the saved coarse flow of the R1d design.

    thermal mesh   elements   state
    h/2              20,800   R1h cell flow_h__thermal_h2 (= R1g level 2), reused
    h/4              83,200   R1h cell flow_h__thermal_h4 (= R1g level 4), reused
    h/8             332,800   solved here, once

Everything else is held: the R1d physical density copied from the parent
element, the saved coarse flow u_h (re-gated, not re-solved), alpha_max 1e7,
materials, source, boundaries, the thermal residual, tau_T by its rule on the
h/8 mesh, thermal 3x3. The one question: does the step shrink from
Delta_24 = C_h/4 - C_h/2 to Delta_48 = C_h/8 - C_h/4, and by how much.
|Delta_48| / |Delta_24| is an observation -- not an error estimate, and not a
pass mark.

One thermal solve: no repeated timing (its time includes compilation), no flow
solve, no gradient, no other level. The same per-cell report as R1h, so the
three levels are compared field for field.

    python scripts/zhao2d_thermal_h8_check.py [--out DIR] [--inputs results]
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
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

# the earlier stages' helpers, so every figure is defined as it was there
from zhao2d_dual_check import peak_working_set_mb, timed  # noqa: E402
from zhao2d_flow_mesh_check import (  # noqa: E402
    provenance, sha256_file, summary, thermal_inlet_velocity,
)

QUAD = 3
LEVEL = 8
RECORD = "zhao2d_r1i_h8.json"
FIELDS = "zhao2d_r1i_fields.npz"
INPUTS = (
    "zhao2d_r1d_main.json",
    "zhao2d_r1d_main_fields.npz",
    "zhao2d_r1g_dual.json",
    "zhao2d_r1h_matrix.json",
)
# the fields compared across levels, as `summary` names them
KEYS = ("compliance", "c_advective", "c_diffusive", "l_q", "net_supg", "t_max",
        "j_star", "D_T", "minus_half_D_T2", "H", "bracket")


def steps(a: dict, b: dict, c: dict) -> dict:
    """Delta_24 = b - a, Delta_48 = c - b and |Delta_48| / |Delta_24|, per field."""
    out = {}
    for k in KEYS:
        d24, d48 = b[k] - a[k], c[k] - b[k]
        out[k] = {
            "delta_24": d24,
            "delta_48": d48,
            "rel_24": d24 / a[k] if a[k] != 0 else None,
            "rel_48": d48 / b[k] if b[k] != 0 else None,
            "abs_ratio_48_over_24": abs(d48) / abs(d24) if d24 != 0 else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=pathlib.Path, default=REPO / "results",
                    help="where the R1d, R1g and R1h records are read")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results",
                    help="where this run's records are written")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing R1i records already in --out")
    args = ap.parse_args()
    inputs, out = args.inputs, args.out
    existing = [n for n in (RECORD, FIELDS) if (out / n).exists()]
    if existing and not args.overwrite:
        sys.exit(f"{out} already holds {', '.join(existing)}, which the docs cite as "
                 "evidence; write this run elsewhere with --out DIR, or pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    memory = {}

    record = {
        "note": (
            "R1i fixed-design check: h_T = h/8 on the saved coarse flow of the R1d "
            "continuous design, one thermal state. The h/2 and h/4 levels are R1h's "
            "reports of R1g's states. The shrink ratio |Delta_48|/|Delta_24| is an "
            "observation, not an error estimate or a pass mark. J* uses the "
            "single-mesh frozen Psi_0 and C_0 as a common reporting scale only."
        ),
        "provenance": provenance(),
        "inputs_dir": str(inputs),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in INPUTS},
        "reference_file_sha256": sha256_file(r1.REFERENCE_FILE),
        "thermal_quadrature": QUAD,
        "thermal_refinement": LEVEL,
    }
    here = pathlib.Path(__file__).resolve()
    record["provenance"]["source_sha256"][here.relative_to(REPO).as_posix()] = (
        sha256_file(here)
    )

    # -- inputs, and that they are the states R1h reported on ------------------
    meta = json.loads((inputs / "zhao2d_r1d_main.json").read_text(encoding="utf-8"))
    r1g = json.loads((inputs / "zhao2d_r1g_dual.json").read_text(encoding="utf-8"))
    r1h = json.loads((inputs / "zhao2d_r1h_matrix.json").read_text(encoding="utf-8"))
    alpha_max, beta = meta["final_alpha_max"], meta["final_beta"]
    fields = np.load(inputs / "zhao2d_r1d_main_fields.npz")
    s_d = np.asarray(fields["solid_fraction"])
    pv_h = jnp.asarray(fields["press_vel"])
    if (r1h["alpha_max"], r1h["beta"]) != (alpha_max, beta):
        sys.exit("R1h and R1d disagree on alpha_max or beta; stopping")
    if fs.digest(s_d) != r1h["density"]["s_flow_h_sha256"]:
        sys.exit("the R1d density is not the one R1h reported on; stopping")

    spec, config = z.Zhao2DSpec(), r1.R1Config()
    scale = r1.load_reference(spec, config)  # single-mesh; a reporting scale here
    record.update(alpha_max=alpha_max, beta=beta, run_fingerprint=config.fingerprint(),
                  reporting_scale={"psi_0": scale.psi_0, "c_0": scale.c_0,
                                   "identity": "single-mesh h = 1e-4 reference"})
    print(f"R1d design, alpha_max {alpha_max:.3e}, beta {beta:g}; saved coarse flow; "
          f"thermal h/{LEVEL}, {QUAD}x{QUAD}\n", flush=True)

    # -- the problem ------------------------------------------------------------
    t0 = time.perf_counter()
    problem = dual.Zhao2DDualProblem(spec, config, thermal_refinement=LEVEL,
                                     thermal_quadrature=QUAD)
    t_build = time.perf_counter() - t0
    memory["after_build"] = peak_working_set_mb()
    print(f"built: flow {problem.flow_mesh.num_elems} el, thermal "
          f"{problem.thermal_mesh.num_elems} el, {problem.thermal_mesh.mesh.num_nodes} "
          f"nodes ({t_build:.1f} s)", flush=True)

    # the saved coarse flow, gated as R1h gated it
    flow_check = fs.verify_flow_state(problem, pv_h, s_d, alpha_max)
    identity = fs.flow_state_identity(problem, s_d, alpha_max)
    if identity != r1h["flows"]["h"]["identity"]:
        sys.exit("the coarse flow's identity is not the one R1h recorded; stopping")
    record["flow"] = {
        "source": "R1d/R1g saved state (results/zhao2d_r1d_main_fields.npz, press_vel)",
        "identity_matches_r1h": True,
        "verification": flow_check,
    }
    print(f"coarse flow: |R|/|R0| {flow_check['residual_relative']:.1e}, identity as "
          "in R1h", flush=True)

    # -- the one thermal solve -----------------------------------------------------
    temperature, t_solve = timed(problem.solve_thermal, pv_h, jnp.asarray(s_d), alpha_max)
    memory["after_thermal_solve"] = peak_working_set_mb()
    print(f"thermal h/{LEVEL}: solved once, {t_solve:.1f} s (includes compilation)",
          flush=True)

    t0 = time.perf_counter()
    rep = fs.cell_report(problem, s_d, pv_h, temperature, alpha_max, scale)
    rep["gate"] = {k: v <= 1e-8 for k, v in rep["residual_relative"].items()}
    if not all(rep["gate"].values()):
        # recorded, not hidden: the state is not counted, and the stage stops here
        record["cell"] = {"residual_relative": rep["residual_relative"],
                          "gate": rep["gate"]}
        (out / RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
        sys.exit(f"the h/{LEVEL} state fails the convergence gate "
                 f"{rep['residual_relative']}; recorded and stopping")
    rep["thermal_inlet"] = fs.thermal_inlet(problem.thermal_mesh, spec, temperature)
    rep["thermal_inlet_velocity_max_deviation"] = thermal_inlet_velocity(
        problem, pv_h, spec)
    rep["state"] = {
        "flow": "h (R1d/R1g)",
        "temperature": "solved in this run",
        "temperature_sha256": fs.digest(np.asarray(temperature)),
        "solid_fraction_thermal_sha256": fs.digest(
            np.asarray(problem.maps.density(jnp.asarray(s_d)))),
    }
    t_report = time.perf_counter() - t0
    memory["after_report"] = peak_working_set_mb()
    rep["cost"] = {"t_build_s": t_build, "t_thermal_solve_s": t_solve,
                   "t_thermal_solve_note": "one call, first of its shape: includes "
                                           "compilation; not repeated",
                   "t_report_s": t_report}
    record["cell"] = rep
    print(f"reported ({t_report:.1f} s)", flush=True)

    # -- the three levels ------------------------------------------------------------
    cells = r1h["cells"]
    levels = {
        "h2": summary(cells["flow_h__thermal_h2"]),
        "h4": summary(cells["flow_h__thermal_h4"]),
        "h8": summary(rep),
    }
    record["levels"] = levels
    record["steps"] = steps(levels["h2"], levels["h4"], levels["h8"])
    c1 = r1g["levels"]["1"]["compliance"]  # h_T = h, for context only
    record["context_h"] = {
        "compliance_h": c1,
        "delta_12": levels["h2"]["compliance"] - c1,
        "abs_ratio_24_over_12": abs(record["steps"]["compliance"]["delta_24"])
        / abs(levels["h2"]["compliance"] - c1),
        "note": "h_T = h is R1g's level 1 (3x3); context for the sequence, not "
                "part of the R1i question",
    }
    div = rep["divergence"]
    record["anchors"] = {
        "h2_vs_r1g_level2": levels["h2"]["compliance"] / r1g["levels"]["2"]["compliance"] - 1,
        "h4_vs_r1g_level4": levels["h4"]["compliance"] / r1g["levels"]["4"]["compliance"] - 1,
        "psi_vs_r1g": rep["psi"] / r1g["flow"]["psi"] - 1,
        # P is an exact embedding, so the coarse flow's divergence cannot change
        "div_u_squared_thermal_vs_flow_mesh": div["thermal_mesh"]["integral_div_u_squared"]
        / div["flow_mesh"]["integral_div_u_squared"] - 1,
    }
    record["cost"] = {
        "t_build_s": t_build,
        "t_thermal_solve_s": t_solve,
        "t_report_s": t_report,
        "peak_working_set_mb_cumulative": memory,
        "peak_working_set_note": "this process's peak so far; the process built only "
                                 "the h/8 problem (with its coarse flow side)",
        "wall_clock_total_s": time.perf_counter() - t_start,
    }

    # -- save, before anything else can fail ------------------------------------------
    (out / RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / FIELDS,
        temperature_thermal_h8=np.asarray(temperature),
        thermal_node_coords_h8=np.asarray(problem.thermal_mesh.mesh.nodes.coords),
    )

    # -- print --------------------------------------------------------------------------
    print("\nanchors (relative):")
    for k, v in record["anchors"].items():
        print(f"  {k:36s} {v:+.3e}")
    print("\n" + " " * 26 + "".join(f"{c:>18s}" for c in ("T h/2", "T h/4", "T h/8")))
    for label, key, fmt in (("C", "compliance", "{:18.6f}"),
                            ("c_advective", "c_advective", "{:18.4f}"),
                            ("c_diffusive", "c_diffusive", "{:18.4f}"),
                            ("L_Q", "l_q", "{:18.4f}"),
                            ("D_SUPG - F_SUPG", "net_supg", "{:18.4f}"),
                            ("(D - F)/C", "net_supg_over_c", "{:18.4%}"),
                            ("T_max", "t_max", "{:18.6f}"),
                            ("T_min", "min_temperature", "{:18.6f}"),
                            ("nodes below inlet", "nodes_below_inlet", "{:18d}"),
                            ("J* (single-mesh scale)", "j_star", "{:18.6f}"),
                            ("D_T / Q", "D_T_over_heat_in", "{:18.4%}"),
                            ("-1/2 D_T2 / C", "minus_half_D_T2_over_C", "{:18.4%}"),
                            ("Dirichlet reaction", "dirichlet_reaction", "{:18.6f}"),
                            ("|R|/|R0| thermal", "residual_thermal", "{:18.1e}")):
        print(f"  {label:24s}" + "".join(fmt.format(levels[c][key]) for c in levels))
    print("\nsteps (Delta_24 = h/2 -> h/4, Delta_48 = h/4 -> h/8):")
    for k in KEYS:
        s = record["steps"][k]
        ratio = s["abs_ratio_48_over_24"]
        print(f"  {k:18s} {s['delta_24']:+14.4f} -> {s['delta_48']:+14.4f}"
              + (f"   |ratio| {ratio:.4f}" if ratio is not None else ""))
    ctx = record["context_h"]
    print(f"  (C from h: Delta_12 {ctx['delta_12']:+.4f}; |Delta_24|/|Delta_12| "
          f"{ctx['abs_ratio_24_over_12']:.4f})")
    print(f"\nwrote {out / RECORD}\nwrote {out / FIELDS}\n"
          f"total {record['cost']['wall_clock_total_s']:.0f} s")


if __name__ == "__main__":
    main()
