"""R1f: is the thermal mesh sensitivity the temperature space, tau, or the velocity?

Four thermal analyses on the SAME continuous design, changing one thing at a
time. See tfopus/zhao2d_thermal_study.py for why B exists and why the
differences are path-dependent rather than an error budget.

3x3 thermal quadrature throughout, because 2x2 is not exact for the SUPG term
(the product of two streamline derivatives reaches fourth order per direction
on a Q1 element). The production baseline stays 2x2; this does not overwrite it.

    python scripts/zhao2d_thermal_separation.py [--binary] [--reuse-fine-flow]
"""

from __future__ import annotations

import pathlib
import sys

# Cap BLAS threads BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import dataclasses
import json
import pathlib
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))

import toflux.src.solver as tf_solver  # noqa: E402

from tfopus import fe_thermal as fe_thermal  # noqa: E402
from tfopus import materials as materials  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402
from tfopus import zhao2d_refine as ref  # noqa: E402
from tfopus import zhao2d_thermal_study as ts  # noqa: E402

QUAD = 3
FINE_FLOW = REPO / "results" / "zhao2d_r1f_fine_flow.npz"


def thermal_only(spec, config, s, elem_vel, kappa, tau_elem, quad=QUAD) -> dict:
    """Solve temperature alone on a given velocity, and report the full split."""
    mesh = z.build_mesh(spec, dofs_per_node=1, gauss_order=quad)
    bc = z.build_thermal_bc(mesh, spec)
    q_source = z.heat_source_field(mesh, spec, config.source_region)
    from tfopus import elements as elements

    solver = fe_thermal.ThermalSolver(
        mesh.mesh,
        bc,
        b_f=spec.b_f,
        solver_settings=za.default_solver_settings(),
        elem_length=elements.element_lengths(mesh.mesh, config.element_length_mode),
        form=config.thermal_form,
        tau_elem=tau_elem,
    )
    t0 = jnp.zeros((mesh.mesh.num_dofs,)).at[bc["fixed_dofs"]].set(
        bc["dirichlet_values"]
    )
    temperature = tf_solver.modified_newton_raphson_solve(
        solver, t0, elem_vel, kappa, q_source
    )
    res, _ = solver.get_residual_and_tangent_stiffness(
        temperature, elem_vel, kappa, q_source
    )
    res0, _ = solver.get_residual_and_tangent_stiffness(
        t0, elem_vel, kappa, q_source
    )
    split = solver.compliance_decomposition(temperature, elem_vel, kappa, q_source)

    tau_used = (
        np.asarray(tau_elem) if tau_elem is not None
        else np.asarray(ts.element_tau(solver, elem_vel, kappa))
    )
    return {
        **split,
        "residual_relative": float(
            jnp.linalg.norm(res) / jnp.maximum(jnp.linalg.norm(res0), 1e-300)
        ),
        "t_max": float(jnp.max(temperature)),
        **ref.undershoot(temperature, spec.inlet_temperature),
        "tau_median": float(np.median(tau_used)),
        "quadrature": quad,
        "num_elements": int(mesh.num_elems),
        "_temperature": np.asarray(temperature),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", action="store_true",
                    help="also run the two most informative binary controls")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results")
    args = ap.parse_args()

    meta = json.loads(
        (args.out / "zhao2d_r1d_main.json").read_text(encoding="utf-8")
    )
    s_coarse = jnp.asarray(
        np.load(args.out / "zhao2d_r1d_main_fields.npz")["solid_fraction"]
    )
    press_vel_coarse = np.load(
        args.out / "zhao2d_r1d_main_fields.npz"
    )["press_vel"]
    alpha_max = meta["final_alpha_max"]

    spec = z.Zhao2DSpec()
    fine_spec = ref.refine_spec(spec, 2)
    config = r1.R1Config()

    coarse = z.build_mesh(spec, dofs_per_node=1, gauss_order=QUAD)
    fine = z.build_mesh(fine_spec, dofs_per_node=1, gauss_order=QUAD)
    parents = ref.parent_of_each_fine_element(coarse, fine)

    material_c = za.build_material(spec, alpha_max)
    kappa_c = materials.conductivity(s_coarse, material_c)
    s_fine = ref.refine_design(coarse, fine, s_coarse)
    kappa_f = materials.conductivity(s_fine, material_c)

    print(f"design from R1d, alpha_max {alpha_max:.3e}, beta {meta['final_beta']:g}")
    print(f"thermal quadrature {QUAD}x{QUAD} (production baseline is 2x2)\n")

    # velocity: coarse solve (saved), extended exactly, and the fine re-solve
    u_nodal_coarse = press_vel_coarse.reshape(-1, 3)[:, 1:]
    u_nodal_fine_ext = ts.extend_velocity(coarse, fine, u_nodal_coarse)
    chk = ts.check_extension(coarse, fine, u_nodal_coarse, u_nodal_fine_ext)
    print(f"velocity extension: exact on {chk['shared_nodes']} shared nodes "
          f"(worst {chk['worst_shared_node_difference']:.2e})")

    ev_coarse = ts.element_velocity_from_nodal(coarse, u_nodal_coarse)
    ev_fine_ext = ts.element_velocity_from_nodal(fine, u_nodal_fine_ext)

    if FINE_FLOW.is_file():
        u_nodal_fine_solved = np.load(FINE_FLOW)["press_vel"].reshape(-1, 3)[:, 1:]
        print("fine-mesh flow: reused from disk")
    else:
        t0 = time.time()
        problem_f = r1.Zhao2DProblem(fine_spec, config)
        pv_f, _, _, _ = problem_f.solve_states(s_fine, alpha_max)
        np.savez_compressed(FINE_FLOW, press_vel=np.asarray(pv_f))
        u_nodal_fine_solved = np.asarray(pv_f).reshape(-1, 3)[:, 1:]
        print(f"fine-mesh flow: solved once and saved ({time.time() - t0:.0f} s)")
    ev_fine_solved = ts.element_velocity_from_nodal(fine, u_nodal_fine_solved)

    # tau at A, frozen onto the children for B
    from tfopus import elements as elements
    probe = fe_thermal.ThermalSolver(
        coarse.mesh,
        z.build_thermal_bc(coarse, spec),
        b_f=spec.b_f,
        solver_settings=za.default_solver_settings(),
        elem_length=elements.element_lengths(coarse.mesh, config.element_length_mode),
        form=config.thermal_form,
    )
    tau_coarse = ts.element_tau(probe, ev_coarse, kappa_c)
    tau_frozen = ts.freeze_tau_from_parent(parents, tau_coarse)

    cases = {
        "A  h,   u_h,      tau_h": (spec, coarse, ev_coarse, kappa_c, None),
        "B  h/2, u_h ext,  tau_h frozen": (fine_spec, fine, ev_fine_ext, kappa_f,
                                           tau_frozen),
        "C  h/2, u_h ext,  tau_h/2": (fine_spec, fine, ev_fine_ext, kappa_f, None),
        "D  h/2, u_h/2,    tau_h/2": (fine_spec, fine, ev_fine_solved, kappa_f, None),
    }

    results = {}
    for label, (sp, mesh, ev, ka, tau) in cases.items():
        r = thermal_only(sp, config, None, ev, ka, tau)
        results[label] = r
        print(f"  {label:34s} C {r['compliance']:12.4f}  "
              f"tau_med {r['tau_median']:.3e}  Tmax {r['t_max']:8.4f}  "
              f"Tmin {r['min_temperature']:+9.6f}  under {r['nodes_below_inlet']:4d}  "
              f"|R| {r['residual_relative']:.1e}")

    keys = list(cases)
    print("\nstep-by-step along this path (each changes ONE thing):")
    total = results[keys[-1]]["compliance"] - results[keys[0]]["compliance"]
    for a, b, what in (
        (0, 1, "temperature space (A -> B)"),
        (1, 2, "stabilisation coefficient (B -> C)"),
        (2, 3, "velocity input (C -> D)"),
    ):
        d = results[keys[b]]["compliance"] - results[keys[a]]["compliance"]
        print(f"  {what:34s} dC {d:+11.2f}   {d / total * 100:+6.1f}% of the total")
    print(f"  {'total (A -> D)':34s} dC {total:+11.2f}")

    print("\nidentity C + D_SUPG - F_SUPG = L_Q, and what the stabilisation carries:")
    print(f"  {'case':34s} {'L_Q':>12} {'D_SUPG':>11} {'F_SUPG':>9} "
          f"{'(D-F)/C':>9} {'closure':>10}")
    for label in keys:
        r = results[label]
        print(f"  {label:34s} {r['l_q']:12.2f} {r['d_supg']:11.2f} "
              f"{r['f_supg']:9.2f} {r['net_supg_over_c'] * 100:8.2f}% "
              f"{r['closure_relative']:10.1e}")

    pe = ts.peclet_statistics(
        coarse, ev_coarse, kappa_c, spec.b_f, spec.element_size, s_coarse
    )
    print(f"\nPeclet, {pe['definition']}, h = {pe['h']:g}:")
    for mask, stat in pe.items():
        if isinstance(stat, dict):
            print(f"  {mask:30s} n={stat['elements']:5d} median {stat['median']:9.4f} "
                  f"p90 {stat['p90']:9.4f} max {stat['max']:9.4f} "
                  f">1: {stat['fraction_above_1'] * 100:5.1f}%")

    payload = {
        "note": "R1f thermal separation. B freezes tau at the parent value and "
                "is a counterfactual for attribution, not a scheme to run. The "
                "differences sum along this path only.",
        "thermal_quadrature": QUAD,
        "alpha_max": alpha_max,
        "velocity_extension_check": chk,
        "peclet_coarse": pe,
        "cases": {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in results.items()
        },
    }

    if args.binary:
        s_bin_c = ref.threshold_design(coarse, s_coarse, 0.5)
        s_bin_f = ref.refine_design(coarse, fine, s_bin_c)
        ka_bc = materials.conductivity(s_bin_c, material_c)
        ka_bf = materials.conductivity(s_bin_f, material_c)
        print("\nbinary controls (the two that bound the same question):")
        binary = {}
        for label, (sp, ev, ka, tau) in {
            "A' h,   binary, u_h,     tau_h": (spec, ev_coarse, ka_bc, None),
            "C' h/2, binary, u_h ext, tau_h/2": (fine_spec, ev_fine_ext, ka_bf, None),
        }.items():
            r = thermal_only(sp, config, None, ev, ka, tau)
            binary[label] = {k: v for k, v in r.items() if not k.startswith("_")}
            print(f"  {label:34s} C {r['compliance']:12.4f}  "
                  f"Tmax {r['t_max']:8.4f}  Tmin {r['min_temperature']:+9.6f}")
        payload["binary_controls"] = binary

    (args.out / "zhao2d_r1f_separation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nsaved to {(args.out / 'zhao2d_r1f_separation.json').name}")


if __name__ == "__main__":
    main()
