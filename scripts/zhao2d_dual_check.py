"""R1g: the dual-mesh thermal model on the R1d design -- anchors, h/2 vs h/4, cost.

Fixed design, no optimisation. The coarse flow is solved ONCE and reused by
every thermal level, so the levels differ only in how the temperature is
discretised:

    level   thermal mesh   elements   quadrature
    1       h                5200     3x3    must reproduce R1f's A row
    2       h/2             20800     3x3    must reproduce R1f's C row, NOT D
    4       h/4             83200     3x3    the one new resolution check

R1f computed A and C from the SAVED coarse velocity. Here the flow is re-solved
from the saved design, so matching them also confirms the re-solve. D used a
fine-mesh flow; matching D instead of C would mean a fine flow had crept back
into a model whose whole point is to keep the coarse one.

Per level: C and its parts, the stabilisation identity, temperature extremes
and undershoot, the heat diagnostics, tau statistics (including the paired
child/parent ratio over fluid and over the whole domain), element Peclet
numbers with their masks, and cost -- thermal forward, and value-and-gradient
of J* through the WHOLE chain, first call (with compilation) and repeat, plus
the process's peak working set. The states -- the coarse flow and each level's
temperature -- are saved beside the record.

J* uses the single-mesh frozen denominators as a reporting scale only. It is
not this model's normalised objective; there is none yet.

    python scripts/zhao2d_dual_check.py [--levels 1 2 4] [--no-gradient]
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
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))

import toflux.src.solver as tf_solver  # noqa: E402

from tfopus import materials as materials  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402
from tfopus import zhao2d_refine as ref  # noqa: E402
from tfopus import zhao2d_thermal_study as ts  # noqa: E402

QUAD = 3
R1F = {  # zhao2d_r1f_separation.json, full precision
    "A": 26938.332194702118,
    "C": 33233.13278907864,
    "D": 32400.115711650746,
}


def peak_working_set_mb() -> float | None:
    """This process's peak resident memory so far, in MiB. Cumulative."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD
            ]
            c = Counters()
            c.cb = ctypes.sizeof(Counters)
            if not psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb
            ):
                return None
            return c.PeakWorkingSetSize / 2**20
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    return out, time.perf_counter() - t0


def quantiles(a) -> dict:
    a = np.asarray(a)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def divergence_integrals(problem, press_vel, temperature, b_f) -> dict:
    """integral b_f T div u and integral b_f T^2 div u on the thermal mesh.

    With u_T = P u_F these are properties of the COARSE velocity's discrete
    divergence, sampled by the temperature. What holds exactly, for these
    polynomial fields at 3x3 quadrature, is the divergence theorem:

        H = boundary integral b_f T u.n = integral b_f u.grad T + D_T,
        H - Q = D_T + [integral b_f u.grad T - Q],

    with D_T the first integral here. The bracket is the discrete equation
    tested with v = 1, i.e. the reaction at the Dirichlet inlet: small (-0.036,
    -0.068, -0.072 at h, h/2, h/4) but not zero, so D_T is close to H - Q
    without being it. The second integral enters C: element by element,

        integral b_f T u.grad T = 1/2 boundary integral b_f T^2 u.n
                                  - 1/2 integral b_f T^2 div u,

    so -1/2 D_T2 is the part of C's advective half written as a volume term in
    div u. That is an algebraic split of ONE solution. It is not what C would
    change by if the velocity were divergence-free: T, the boundary term and
    the SUPG terms all move with the flow.
    """
    mesh = problem.thermal_mesh.mesh
    shp = jax.vmap(mesh.elem_template.shape_functions)(mesh.gauss_pts)
    vel = problem.thermal_nodal_velocity(press_vel)
    t = jnp.asarray(temperature)

    def per_element(node_ids, node_coords):
        grad_n = jax.vmap(
            mesh.elem_template.get_gradient_shape_function_physical, in_axes=(0, None)
        )(mesh.gauss_pts, node_coords)
        _, det = jax.vmap(
            mesh.elem_template.compute_jacobian_and_determinant, in_axes=(0, None)
        )(mesh.gauss_pts, node_coords)
        div_u = jnp.einsum("gnd, nd -> g", grad_n, vel[node_ids])
        t_g = jnp.einsum("gn, n -> g", shp, t[node_ids])
        wdet = mesh.gauss_weights * det
        return (jnp.einsum("g, g, g -> ", t_g, div_u, wdet),
                jnp.einsum("g, g, g -> ", t_g**2, div_u, wdet))

    first, second = jax.vmap(per_element)(jnp.asarray(mesh.elem_nodes),
                                          mesh.elem_node_coords)
    return {
        "b_f_T_div_u": float(b_f * jnp.sum(first)),
        "b_f_T2_div_u": float(b_f * jnp.sum(second)),
    }


def boundary_t2_flux(problem, press_vel, temperature, b_f) -> float:
    """1/2 boundary integral b_f T^2 u.n, the other half of the split above.

    Two-point Gauss on each edge is exact here: u.n is linear along an edge and
    T^2 quadratic, so the integrand is cubic. T is interpolated first and then
    squared -- interpolating nodal T^2 would be a different function.
    """
    from tfopus.mesh import Face

    vel = np.asarray(problem.thermal_nodal_velocity(press_vel))
    t = np.asarray(temperature)

    def per_face(nodes, normal, length):
        total = 0.0
        for tt, ww in zip(za._GAUSS_T, za._GAUSS_W):
            u = (1.0 - tt) * vel[nodes[0]] + tt * vel[nodes[1]]
            temp = (1.0 - tt) * t[nodes[0]] + tt * t[nodes[1]]
            total += ww * float(np.dot(u, normal)) * temp * temp
        return total * length

    return 0.5 * b_f * sum(
        za._edge_integral(problem.thermal_mesh, tag, per_face)
        for tag in (Face.INLET, Face.OUTLET, Face.WALL, Face.SYMMETRY)
    )


def level_report(problem, spec, s, press_vel, alpha_max, tau_level1, s_level1) -> dict:
    """Everything reported for one thermal level, from one thermal solve."""
    out = {}
    temperature, out["t_thermal_first_s"] = timed(
        problem.solve_thermal, press_vel, s, alpha_max
    )
    _, out["t_thermal_repeat_s"] = timed(problem.solve_thermal, press_vel, s, alpha_max)

    vel_t = problem.thermal_velocity(press_vel)
    kappa_t = problem.thermal_conductivity(s, alpha_max)
    q = problem.q_source

    res, _ = problem.thermal.get_residual_and_tangent_stiffness(
        temperature, vel_t, kappa_t, q
    )
    res0, _ = problem.thermal.get_residual_and_tangent_stiffness(
        problem.thermal_x0, vel_t, kappa_t, q
    )
    out["thermal_residual_relative"] = float(
        jnp.linalg.norm(res) / jnp.maximum(jnp.linalg.norm(res0), 1e-300)
    )

    split = problem.thermal.compliance_decomposition(temperature, vel_t, kappa_t, q)
    out.update(split)
    out["t_max"] = float(jnp.max(temperature))
    out.update(ref.undershoot(temperature, spec.inlet_temperature))

    # conservation() reads velocity from a press_vel layout; pressure is unused
    nodal = np.asarray(problem.thermal_nodal_velocity(press_vel))
    pv_t = np.zeros((nodal.shape[0], 3))
    pv_t[:, 1:] = nodal
    cons = za.conservation(
        spec, problem.thermal_mesh, pv_t.ravel(), temperature, q, kappa_t
    )
    out["heat"] = {k: float(v) for k, v in cons.items()}
    div = divergence_integrals(problem, press_vel, temperature, spec.b_f)
    out["divergence"] = {
        **div,
        "enthalpy_excess": cons["enthalpy_net_out"] - cons["heat_in"],
        "enthalpy_excess_minus_T_div_u": (
            cons["enthalpy_net_out"] - cons["heat_in"] - div["b_f_T_div_u"]
        ),
        "advective_part_from_div_u": -0.5 * div["b_f_T2_div_u"],
        "advective_part_from_div_u_over_c": -0.5 * div["b_f_T2_div_u"]
        / split["compliance"],
    }
    boundary = boundary_t2_flux(problem, press_vel, temperature, spec.b_f)
    out["divergence"]["advective_part_from_boundary"] = boundary
    # the split must close on c_advective, or the attribution means nothing
    out["divergence"]["split_closure_relative"] = abs(
        boundary + out["divergence"]["advective_part_from_div_u"] - split["c_advective"]
    ) / abs(split["c_advective"])

    tau = np.asarray(ts.element_tau(problem.thermal, vel_t, kappa_t))
    s_t = np.asarray(problem.maps.density(s))
    fluid_t = s_t < 0.5
    out["tau"] = {
        "whole_domain": quantiles(tau),
        "fluid_s_lt_0.5": quantiles(tau[fluid_t]),
    }
    if tau_level1 is not None:
        parents = problem.maps.parents
        ratio = tau / tau_level1[parents]
        fluid_parent = np.asarray(s_level1)[parents] < 0.5
        out["tau"]["child_over_parent"] = {
            "definition": "tau on this thermal mesh / tau of the parent at h, "
                          "same velocity function, paired element by element",
            "whole_domain": quantiles(ratio),
            "fluid_parent_s_lt_0.5": quantiles(ratio[fluid_parent]),
        }

    h_t = spec.element_size / problem.thermal_refinement
    out["peclet"] = ts.peclet_statistics(
        problem.thermal_mesh, vel_t, kappa_t, spec.b_f, h_t, s_t
    )
    return out, tau, np.asarray(temperature)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--no-gradient", action="store_true",
                    help="skip the value-and-gradient timing")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results")
    args = ap.parse_args()
    if args.levels[0] != 1:
        sys.exit("level 1 must come first: it anchors R1f's A row and the tau pairing")

    fields = np.load(args.out / "zhao2d_r1d_main_fields.npz")
    meta = json.loads((args.out / "zhao2d_r1d_main.json").read_text(encoding="utf-8"))
    alpha_max, beta = meta["final_alpha_max"], meta["final_beta"]
    spec, config = z.Zhao2DSpec(), r1.R1Config(projection=r1.Projection.TANH)  # the projection its record used
    scale = r1.load_reference(spec, config)  # single-mesh; a reporting scale here
    w = config.weight

    x = jnp.asarray(fields["design"])
    s = jnp.asarray(fields["solid_fraction"])
    print(f"R1d design: alpha_max {alpha_max:.3e}, beta {beta:g}; thermal quadrature "
          f"{QUAD}x{QUAD}; J* on the single-mesh reference as a reporting scale\n")

    record = {
        "note": (
            "R1g fixed-design check of the dual-mesh thermal model. The coarse "
            "flow is solved once and reused by every level. J* divides by the "
            "single-mesh frozen Psi_0 and C_0 as a common reporting scale; it "
            "is not a normalised objective of this model."
        ),
        "alpha_max": alpha_max,
        "beta": beta,
        "thermal_quadrature": QUAD,
        "run_fingerprint": config.fingerprint(),
        "reporting_scale": {"psi_0": scale.psi_0, "c_0": scale.c_0,
                            "identity": "single-mesh h = 1e-4 reference"},
        "r1f_rows": R1F,
        "levels": {},
    }

    # -- the flow, once -------------------------------------------------------
    base = dual.Zhao2DDualProblem(spec, config, thermal_refinement=1,
                                  thermal_quadrature=QUAD)
    s_map = np.asarray(base.solid_fraction(x, beta))
    record["design_map_max_abs_diff"] = float(np.max(np.abs(s_map - np.asarray(s))))
    print(f"design map reproduces the saved s: max |diff| "
          f"{record['design_map_max_abs_diff']:.2e}")

    material = za.build_material(spec, alpha_max)
    alpha = materials.brinkman_penalty(s, material)
    press_vel, t_flow = timed(
        tf_solver.modified_newton_raphson_solve, base.flow, base.flow_x0, alpha
    )
    _, t_flow_repeat = timed(
        tf_solver.modified_newton_raphson_solve, base.flow, base.flow_x0, alpha
    )
    fr, _ = base.flow.get_residual_and_tangent_stiffness(press_vel, alpha)
    f0, _ = base.flow.get_residual_and_tangent_stiffness(base.flow_x0, alpha)
    saved = fields["press_vel"]
    psi = float(base.flow.dissipated_power(press_vel, alpha))
    record["flow"] = {
        "elements": base.flow_mesh.num_elems,
        "t_flow_solve_s": t_flow,
        "t_flow_solve_repeat_s": t_flow_repeat,
        "residual_relative": float(jnp.linalg.norm(fr) / jnp.linalg.norm(f0)),
        "resolved_vs_saved_rel": float(
            np.linalg.norm(np.asarray(press_vel) - saved) / np.linalg.norm(saved)
        ),
        "psi": psi,
    }
    print(f"coarse flow: {t_flow:.1f} s, |R|/|R0| {record['flow']['residual_relative']:.1e}, "
          f"re-solved vs saved {record['flow']['resolved_vs_saved_rel']:.1e}, "
          f"Psi {psi:.8g}\n")

    tau_1 = None
    gradients = {}
    states = {
        "press_vel_coarse": np.asarray(press_vel),
        "solid_fraction": np.asarray(s),
        "design": np.asarray(x),
    }
    for r in args.levels:
        t0 = time.perf_counter()
        problem = (base if r == 1 else
                   dual.Zhao2DDualProblem(spec, config, thermal_refinement=r,
                                          thermal_quadrature=QUAD))
        t_build = time.perf_counter() - t0
        rep, tau, temperature = level_report(problem, spec, s, press_vel, alpha_max,
                                             tau_1, np.asarray(s))
        states[f"temperature_level{r}"] = temperature
        states[f"thermal_node_coords_level{r}"] = np.asarray(
            problem.thermal_mesh.mesh.nodes.coords
        )
        if r == 1:
            tau_1 = tau
        rep["t_build_s"] = t_build
        rep["maps"] = problem.maps.identity()
        rep["nesting"] = problem.nesting
        rep["psi"] = psi
        rep["j_star"] = float(dual.reporting_objective(psi, rep["compliance"], scale, w))

        if not args.no_gradient:
            def j_star(v, problem=problem):
                p, c = problem.metrics(problem.solid_fraction(v, beta), alpha_max)
                return dual.reporting_objective(p, c, scale, w)

            vg = jax.value_and_grad(j_star)
            (j1, g1), t_first = timed(vg, x)
            (j2, g2), t_repeat = timed(vg, x)
            gradients[r] = np.asarray(g1)
            states[f"gradient_j_star_level{r}"] = np.asarray(g1)
            rep["cost"] = {
                "value_and_grad_first_s": t_first,
                "value_and_grad_repeat_s": t_repeat,
                "gradient_norm": float(jnp.linalg.norm(g1)),
                "repeat_reproduces_gradient": float(
                    jnp.linalg.norm(g2 - g1) / jnp.linalg.norm(g1)
                ),
                "j_star_through_full_chain": float(j1),
            }
        rep["peak_working_set_mb_so_far"] = peak_working_set_mb()
        record["levels"][str(r)] = rep

        line = (f"h/{r}: {problem.thermal_mesh.num_elems:6d} el  C {rep['compliance']:12.4f}  "
                f"(D-F)/C {rep['net_supg_over_c'] * 100:6.2f}%  Tmax {rep['t_max']:8.4f}  "
                f"Tmin {rep['min_temperature']:+.4f} ({rep['nodes_below_inlet']} nodes)  "
                f"|R| {rep['thermal_residual_relative']:.1e}  thermal {rep['t_thermal_repeat_s']:.1f} s")
        if "cost" in rep:
            line += (f"  value+grad {rep['cost']['value_and_grad_first_s']:.1f} / "
                     f"{rep['cost']['value_and_grad_repeat_s']:.1f} s")
        print(line, flush=True)

    # -- anchors and the comparison ----------------------------------------
    lv = record["levels"]
    anchors = {}
    if "1" in lv:
        anchors["level1_vs_A"] = lv["1"]["compliance"] / R1F["A"] - 1.0
    if "2" in lv:
        anchors["level2_vs_C"] = lv["2"]["compliance"] / R1F["C"] - 1.0
        anchors["level2_vs_D"] = lv["2"]["compliance"] / R1F["D"] - 1.0
    record["anchors"] = anchors
    print("\nanchors (relative difference):")
    for k, v in anchors.items():
        print(f"  {k:14s} {v:+.3e}")

    keys = ["compliance", "c_advective", "c_diffusive", "l_q", "d_supg", "f_supg",
            "t_max", "j_star"]
    levels = [k for k in ("1", "2", "4") if k in lv]
    steps = {}
    for a, b in zip(levels, levels[1:]):
        steps[f"{a}->{b}"] = {
            k: {"abs": lv[b][k] - lv[a][k], "rel": lv[b][k] / lv[a][k] - 1.0}
            for k in keys
        }
    record["steps"] = steps
    if len(gradients) > 1:
        # Local sensitivity only: says which way J* would move from THIS
        # design, not where an optimisation on each model would end.
        record["gradient_cosines"] = {
            f"{a}-{b}": float(
                np.dot(gradients[a], gradients[b])
                / (np.linalg.norm(gradients[a]) * np.linalg.norm(gradients[b]))
            )
            for i, a in enumerate(sorted(gradients))
            for b in sorted(gradients)[i + 1:]
        }
    if len(levels) == 3:
        d1 = lv["2"]["compliance"] - lv["1"]["compliance"]
        d2 = lv["4"]["compliance"] - lv["2"]["compliance"]
        record["successive_change_ratio"] = d1 / d2 if d2 != 0 else None
    print("\nstep changes:")
    for name, st in steps.items():
        print(f"  {name}: " + "  ".join(
            f"{k} {v['rel']:+.2%}" for k, v in st.items() if k in
            ("compliance", "c_diffusive", "c_advective", "t_max", "j_star")))
    if "successive_change_ratio" in record:
        print(f"  (C_h/2 - C_h) / (C_h/4 - C_h/2) = {record['successive_change_ratio']:.3f}")
    print("\nthe coarse velocity's divergence, as the temperature samples it:")
    for k in levels:
        dv = lv[k]["divergence"]
        print(f"  h/{k}: enthalpy excess {dv['enthalpy_excess']:9.3f} vs int b_f T div u "
              f"{dv['b_f_T_div_u']:9.3f}  ({dv['b_f_T_div_u'] / lv[k]['heat']['heat_in']:+.2%} of the source);"
              f"  -1/2 int b_f T^2 div u = {dv['advective_part_from_div_u']:9.2f} "
              f"({dv['advective_part_from_div_u_over_c']:+.2%} of C); split closes to "
              f"{dv['split_closure_relative']:.1e}")
    if "gradient_cosines" in record:
        print("\ngradient of J* -- norms and cosines between levels (local only):")
        for k in levels:
            if "cost" in lv[k]:
                print(f"  h/{k}: |grad| {lv[k]['cost']['gradient_norm']:.4e}")
        for k, v in record["gradient_cosines"].items():
            print(f"  cos({k}) = {v:+.4f}")

    path = args.out / "zhao2d_r1g_dual.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    fields_path = args.out / "zhao2d_r1g_fields.npz"
    np.savez_compressed(fields_path, **states)
    print(f"\nwrote {path}\nwrote {fields_path} (the dual-mesh states)")


if __name__ == "__main__":
    main()
