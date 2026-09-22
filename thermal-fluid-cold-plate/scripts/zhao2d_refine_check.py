"""R1e: fixed-design mesh check. No optimisation, no re-filtering, no reprojection.

Takes the R1d design exactly as saved and re-analyses it on h and h/2, for both
the continuous field and its 0.5 threshold. The binary design is thresholded on
the PARENT mesh and then transferred, so the two meshes describe the same
polygon and the only thing that changes is the discretisation.

Psi and C are reported raw. J is shown against the ORIGINAL h = 1e-4 frozen
denominators, used purely as a common reporting scale -- it is not a fine-mesh
reference, `ReferenceValues.check` is not relaxed to pretend otherwise, and the
fine-mesh J is therefore not a normalised objective in the sense R1d's was.

No pass threshold is set in advance. Two levels measure a trend and a
magnitude; they do not establish mesh independence.

    python scripts/zhao2d_refine_check.py [--factor 2] [--no-figures]
"""

from __future__ import annotations

import pathlib
import sys

# Cap BLAS threads BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
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

from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402
from tfopus import zhao2d_refine as ref  # noqa: E402
from tfopus.mesh import Face  # noqa: E402


def analyse(spec, config, s, alpha_max, scale) -> dict:
    """One fixed-design flow+thermal solve plus the full diagnostic set."""
    problem = r1.Zhao2DProblem(spec, config)
    norms = problem.require_converged(s, alpha_max)
    press_vel, temperature, alpha, kappa = problem.solve_states(s, alpha_max)

    psi = float(problem.flow.dissipated_power(press_vel, alpha))
    c = float(
        problem.thermal.thermal_compliance(
            temperature, problem.flow.element_velocities(press_vel), kappa
        )
    )
    thermal_mesh = z.build_mesh(spec, dofs_per_node=1)
    q_source = z.heat_source_field(thermal_mesh, spec, config.source_region)
    cons = za.conservation(
        spec, thermal_mesh, press_vel, temperature, q_source, kappa
    )

    inlet = np.unique([e for e, _ in problem.flow_mesh.elem_faces[Face.INLET]])
    outlet = np.unique([e for e, _ in problem.flow_mesh.elem_faces[Face.OUTLET]])
    num, sizes, joined, _, _ = z.fluid_connectivity(
        problem.flow_mesh, np.asarray(s) < 0.5, inlet, outlet
    )

    out = {
        "element_size": spec.element_size,
        "num_elements": int(problem.flow_mesh.num_elems),
        "psi": psi,
        "compliance": c,
        "J_on_original_scale": (
            config.weight * psi / scale.psi_0
            + (1 - config.weight) * c / scale.c_0
        ),
        "t_max": float(jnp.max(temperature)),
        "t_mean": float(jnp.mean(temperature)),
        **ref.undershoot(temperature, spec.inlet_temperature),
        "fluid_components": int(num),
        "inlet_outlet_connected": bool(joined),
        **z.fluid_fractions(problem.flow_mesh, s),
        "flow_residual_relative": norms["flow"],
        "thermal_residual_relative": norms["thermal"],
        **{k: cons[k] for k in (
            "heat_in", "enthalpy_net_out", "conduction_net_out",
            "energy_imbalance_rel", "temperature_weighted_divergence",
            "temperature_weighted_divergence_rel", "mass_imbalance_rel",
        )},
    }
    return out, (problem, np.asarray(temperature), np.asarray(s))


def figures(panels, out_dir, spec) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_all = np.concatenate([p["temp"] for p in panels])
    tmin, tmax = float(t_all.min()), float(t_all.max())

    fig, axes = plt.subplots(3, len(panels), figsize=(4.2 * len(panels), 11))
    if len(panels) == 1:
        axes = axes.reshape(3, 1)

    for col, panel in enumerate(panels):
        pm, s, temp = panel["mesh"], panel["s"], panel["temp"]
        c = np.asarray(pm.elem_centres)
        h = float(np.sqrt(np.asarray(pm.elem_area)[0]))
        nodes = np.asarray(pm.mesh.nodes.coords)
        t_el = temp[np.asarray(pm.mesh.elem_nodes)].mean(axis=1)

        # A cell whose NODAL MEAN is negative is a much rarer thing than a cell
        # touching a negative node: averaging four nodes hides a single cold
        # one. The earlier version plotted the mean and so showed no undershoot
        # at all while 18 nodes were below the inlet value. Mark any cell with
        # at least one negative node, which is the region being discussed.
        below_node = temp < spec.inlet_temperature - 1e-10
        touches = below_node[np.asarray(pm.mesh.elem_nodes)].any(axis=1)

        for row, (field, cmap, label, rng) in enumerate((
            (s, "gray_r", "solid fraction s", (0.0, 1.0)),
            (t_el, "inferno", "temperature", (tmin, tmax)),
            (np.where(touches, 1.0, np.nan),
             "cool", "cells touching a node below inlet T", (0.0, 1.0)),
        )):
            ax = axes[row, col]
            ax.scatter(c[:, 0], c[:, 1], c=field, s=(h / 1e-4) ** 2 * 1.1,
                       marker="s", cmap=cmap, vmin=rng[0], vmax=rng[1],
                       linewidths=0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f"{panel['name']}\n{pm.num_elems} elements")
            if row == 2:
                ax.set_xlabel(
                    f"{int(below_node.sum())} nodes below inlet T "
                    f"in {int(touches.sum())} cells"
                )

    fig.suptitle("R1e fixed-design mesh check: same design, h and h/2",
                 fontsize=12)
    fig.tight_layout()
    path = out_dir / "zhao2d_r1e_fields.png"
    fig.savefig(path, dpi=130)
    print(f"figures -> {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results")
    args = ap.parse_args()

    meta = json.loads(
        (args.out / "zhao2d_r1d_main.json").read_text(encoding="utf-8")
    )
    s_coarse = jnp.asarray(
        np.load(args.out / "zhao2d_r1d_main_fields.npz")["solid_fraction"]
    )
    alpha_max = meta["final_alpha_max"]

    spec = z.Zhao2DSpec()
    config = r1.R1Config()
    scale = r1.load_reference(spec, config)
    fine_spec = ref.refine_spec(spec, args.factor)

    coarse_mesh = z.build_mesh(spec, dofs_per_node=1)
    fine_mesh = z.build_mesh(fine_spec, dofs_per_node=1)

    print(f"design from R1d, alpha_max {alpha_max:.3e}, beta {meta['final_beta']:g}")
    print(f"coarse h = {spec.element_size:g} ({coarse_mesh.num_elems} elements)")
    print(f"fine   h = {fine_spec.element_size:g} ({fine_mesh.num_elems} elements)")
    print(f"J shown against the ORIGINAL frozen scale Psi_0 = {scale.psi_0:.10g}, "
          f"C_0 = {scale.c_0:.10g}\n")

    designs = {
        "continuous": s_coarse,
        "binary-0.5": ref.threshold_design(coarse_mesh, s_coarse, 0.5),
    }

    results, panels = {}, []
    for name, s_c in designs.items():
        s_f = ref.refine_design(coarse_mesh, fine_mesh, s_c)
        check = ref.check_transfer(coarse_mesh, fine_mesh, s_c, s_f)
        print(f"--- {name} --- transfer check: "
              f"{check['children_per_parent']} children/parent, "
              f"v_f design {check['coarse_v_f_design_domain']:.8f} -> "
              f"{check['fine_v_f_design_domain']:.8f}")

        for label, sp, sd in (("coarse", spec, s_c), ("fine", fine_spec, s_f)):
            t0 = time.time()
            res, (problem, temp, sv) = analyse(sp, config, sd, alpha_max, scale)
            res["seconds"] = time.time() - t0
            results[f"{name}/{label}"] = res
            panels.append({
                "name": f"{name}\n{label}", "mesh": problem.flow_mesh,
                "s": sv, "temp": temp,
            })
            print(f"    {label:6s} {res['num_elements']:6d} el  "
                  f"Psi {res['psi']:.8g}  C {res['compliance']:.8g}  "
                  f"Tmax {res['t_max']:.5f}  Tmin {res['min_temperature']:+.6f}  "
                  f"({res['seconds']:.0f} s)")

    print("\n" + "=" * 96)
    head = (f"{'design / mesh':>20} {'Psi':>13} {'C':>13} {'J*':>9} {'Tmax':>9} "
            f"{'Tmin':>10} {'under':>6} {'dE/E':>9} {'TdivU/Q':>9}")
    print(head)
    print("-" * len(head))
    for key, r in results.items():
        print(f"{key:>20} {r['psi']:13.7g} {r['compliance']:13.7g} "
              f"{r['J_on_original_scale']:9.5f} {r['t_max']:9.4f} "
              f"{r['min_temperature']:10.6f} {r['nodes_below_inlet']:6d} "
              f"{r['energy_imbalance_rel']:9.3e} "
              f"{r['temperature_weighted_divergence_rel']:9.3e}")

    print("\nchange from h to h/2 (same design, discretisation only):")
    for name in designs:
        a, b = results[f"{name}/coarse"], results[f"{name}/fine"]
        print(f"  {name:>12}: Psi {(b['psi']/a['psi']-1)*100:+7.2f}%   "
              f"C {(b['compliance']/a['compliance']-1)*100:+7.2f}%   "
              f"Tmax {(b['t_max']/a['t_max']-1)*100:+7.2f}%   "
              f"undershoot depth {a['undershoot_depth']:.4f} -> "
              f"{b['undershoot_depth']:.4f}")

    print("\ncontinuous -> binary on each mesh (the R1d conclusion, per mesh):")
    for label in ("coarse", "fine"):
        a = results[f"continuous/{label}"]
        b = results[f"binary-0.5/{label}"]
        print(f"  {label:>12}: Psi {(b['psi']/a['psi']-1)*100:+7.2f}%   "
              f"C {(b['compliance']/a['compliance']-1)*100:+7.2f}%   "
              f"J* {(b['J_on_original_scale']/a['J_on_original_scale']-1)*100:+7.2f}%   "
              f"Tmax {(b['t_max']/a['t_max']-1)*100:+7.2f}%")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "zhao2d_r1e_refine.json").write_text(
        json.dumps(
            {
                "note": "R1e fixed-design mesh check. J is shown against the "
                        "ORIGINAL h=1e-4 frozen denominators as a common "
                        "reporting scale only; it is not a fine-mesh reference.",
                "source_run": "results/zhao2d_r1d_main.json",
                "alpha_max": alpha_max,
                "beta": meta["final_beta"],
                "reporting_scale": {"psi_0": scale.psi_0, "c_0": scale.c_0},
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved to {(args.out / 'zhao2d_r1e_refine.json').name}")

    if not args.no_figures:
        figures(panels, args.out, spec)


if __name__ == "__main__":
    main()
