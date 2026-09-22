"""A high-Peclet convection-diffusion benchmark with a known exact solution.

Why this exists: on the cold plate there is no exact answer, so comparing
stabilisation variants there can only rank them by which gives a smaller C --
which is selection bias, not accuracy. This gives an independent reference.

The problem, using the SAME thermal element and residual the cold plate uses:

    b_f u dT/dx - k d2T/dx2 = 0   on (0, L) x (0, H),  u = (U, 0), k constant
    T(0) = 0,  T(L) = 1,  adiabatic top and bottom (the natural condition)

    T(x) = (exp(Pe x / L) - 1) / (exp(Pe) - 1),     Pe = b_f U L / k

which is flat over most of the domain and rises through an exponential layer of
width ~ L/Pe at the outlet -- the same situation the cold plate's thermal field
is in, and the reason C moves so much under refinement there.

Reported per mesh: the L2 error of T, the error in the integral metric
integral k |grad T|^2 (the diffusive half of the compliance, which is what the
cold plate's C is sensitive to), and the over/undershoot. Nodal values and
linear-field patch tests are deliberately NOT the criterion -- they pass for
schemes that are badly wrong in a layer.

    python scripts/zhao2d_advection_benchmark.py [--pe 1000]
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

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))

import toflux.src.bc as tf_bc  # noqa: E402
import toflux.src.mesher as tf_mesher  # noqa: E402
import toflux.src.solver as tf_solver  # noqa: E402

from tfopus import elements as elements  # noqa: E402
from tfopus import fe_thermal as fe_thermal  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

B_F = 4.18e6  # Zhao table 1
L, H = 1.0, 0.25
U = 0.2


def exact(x, pe):
    """T(x) = (exp(Pe x/L) - 1)/(exp(Pe) - 1), written to avoid overflow."""
    return np.exp(pe * (x / L - 1.0)) * (1.0 - np.exp(-pe * x / L)) / (
        1.0 - np.exp(-pe)
    )


def exact_diffusive_integral(k, pe):
    """integral_Omega k |grad T|^2 dOmega for the exact solution.

    grad T = (Pe/L) exp(Pe x/L)/(exp(Pe)-1), so the integral is
    H k Pe^2 / L^2 * integral_0^L exp(2 Pe x/L) dx / (exp(Pe)-1)^2
      = H k Pe / (2 L) * (exp(2 Pe) - 1) / (exp(Pe) - 1)^2
      = H k Pe / (2 L) * coth(Pe / 2) ... written stably below.
    """
    return H * k * pe / (2.0 * L) / np.tanh(pe / 2.0)


def build(nx, ny, pe):
    k = B_F * U * L / pe
    # a plain structured grid, one dof per node
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, H, ny + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel()], axis=1)

    def nid(i, j):
        return i * (ny + 1) + j

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    elems = np.stack(
        [nid(ii, jj), nid(ii + 1, jj), nid(ii + 1, jj + 1), nid(ii, jj + 1)], axis=1
    )
    mesh = tf_mesher.Mesh(
        nodes=tf_mesher.Nodes(coords=jnp.asarray(coords), dof_per_node=1),
        elem_nodes=elems,
        elem_template=elements.Quad4(),
        gauss_order=3,
    )

    left = np.nonzero(np.abs(coords[:, 0]) < 1e-12)[0]
    right = np.nonzero(np.abs(coords[:, 0] - L) < 1e-12)[0]
    fixed = np.concatenate([left, right])
    values = np.concatenate([np.zeros(len(left)), np.ones(len(right))])
    bc = tf_bc.BCDict(
        elem_forces=jnp.zeros((mesh.num_elems, mesh.num_dofs_per_elem)),
        fixed_dofs=fixed,
        free_dofs=np.setdiff1d(np.arange(mesh.num_dofs), fixed),
        dirichlet_values=values,
    )
    return mesh, bc, k, coords


def _l2_against_exact(mesh, nodal, pe) -> tuple[float, float]:
    """(||T_h - T_exact||, ||T_exact||), both true integrals over the domain.

    T_exact is evaluated ANALYTICALLY at the quadrature points, not interpolated
    from its nodal values. That distinction is the whole measurement here: the
    exact solution varies by O(1) inside the last element on every mesh below
    nx ~ 1000, so its nodal interpolant is a different function, and comparing
    T_h against that interpolant hides exactly the error being looked for. It
    also makes the norm mesh-dependent, which is why an earlier version of this
    script reported an error that grew under refinement.

    A higher-order rule is used than the element's own, because the integrand
    contains exp(Pe x) and the solution rule is chosen for polynomials.
    """
    import numpy.polynomial.legendre as leg

    pts, wts = leg.leggauss(10)
    shp = jax.vmap(mesh.elem_template.shape_functions)(
        jnp.asarray([[a, b] for a in pts for b in pts])
    )
    w2 = jnp.asarray([wa * wb for wa in wts for wb in wts])
    values = jnp.asarray(nodal)[jnp.asarray(mesh.elem_nodes)]
    coords = mesh.elem_node_coords

    # physical coordinates of those quadrature points, per element
    x_q = jnp.einsum("gn, end -> egd", shp, coords)[..., 0]
    t_h = jnp.einsum("gn, en -> eg", shp, values)
    t_ex = jnp.asarray(exact(np.asarray(x_q), pe))

    def det_of(node_coords):
        _, det = jax.vmap(
            mesh.elem_template.compute_jacobian_and_determinant, in_axes=(0, None)
        )(jnp.asarray([[a, b] for a in pts for b in pts]), node_coords)
        return det

    det = jax.vmap(det_of)(coords)
    err = float(jnp.sum(jnp.einsum("eg, g, eg -> ", (t_h - t_ex) ** 2, w2, det)))
    ref = float(jnp.sum(jnp.einsum("eg, g, eg -> ", t_ex**2, w2, det)))
    return float(np.sqrt(err)), float(np.sqrt(ref))


def run(nx, ny, pe, form):
    mesh, bc, k, coords = build(nx, ny, pe)
    solver = fe_thermal.ThermalSolver(
        mesh,
        bc,
        b_f=B_F,
        solver_settings=za.default_solver_settings(),
        elem_length=elements.element_lengths(mesh, "min_edge"),
        form=form,
    )
    n_el = mesh.num_elems
    vel = jnp.tile(jnp.array([U, 0.0]), (n_el, 4))
    kappa = jnp.full(n_el, k)
    q = jnp.zeros(n_el)

    t0 = jnp.zeros((mesh.num_dofs,)).at[bc["fixed_dofs"]].set(bc["dirichlet_values"])
    T = tf_solver.modified_newton_raphson_solve(solver, t0, vel, kappa, q)
    T = np.asarray(T)

    t_exact = exact(coords[:, 0], pe)
    max_err = float(np.max(np.abs(T - t_exact)))
    l2, l2_exact = _l2_against_exact(mesh, T, pe)

    split = solver.compliance_decomposition(jnp.asarray(T), vel, kappa, q)
    diff_exact = exact_diffusive_integral(k, pe)
    h = L / nx
    pe_elem = B_F * U * h / (2.0 * k)
    return {
        "nx": nx,
        "h": h,
        "pe_element": pe_elem,
        "l2_error": l2,
        "l2_error_rel": l2 / l2_exact,
        "max_nodal_error": max_err,
        "diffusive_integral": split["c_diffusive"],
        "diffusive_exact": diff_exact,
        "diffusive_rel_error": abs(split["c_diffusive"] - diff_exact) / diff_exact,
        "overshoot": float(T.max() - 1.0),
        "undershoot": float(-T.min()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pe", type=float, default=1000.0)
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results",
                    help="directory for the JSON record")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    form = r1.R1_THERMAL_FORM
    print(f"global Pe = {args.pe:g}, layer width ~ L/Pe = {L / args.pe:.3e}")
    print(f"thermal form: tau={form.tau}, stabilise_source={form.stabilise_source}, "
          f"supg_heat_capacity={form.supg_heat_capacity}\n")

    head = (f"{'nx':>5} {'h':>10} {'Pe_elem':>9} {'L2 rel':>11} {'max err':>9} "
            f"{'int k|gradT|^2':>15} {'exact':>13} {'rel err':>10} "
            f"{'over':>9} {'under':>9}")
    print(head)
    print("-" * len(head))
    rows = []
    for nx in (10, 20, 40, 80, 160, 320):
        r = run(nx, 8, args.pe, form)
        rows.append(r)
        print(f"{r['nx']:5d} {r['h']:10.4g} {r['pe_element']:9.2f} "
              f"{r['l2_error_rel']:11.3e} {r['max_nodal_error']:9.3e} "
              f"{r['diffusive_integral']:15.6g} "
              f"{r['diffusive_exact']:13.6g} {r['diffusive_rel_error']:10.3e} "
              f"{r['overshoot']:+9.2e} {r['undershoot']:+9.2e}")

    print("\nobserved order between successive meshes:")
    for a, b in zip(rows, rows[1:]):
        for key in ("l2_error_rel", "diffusive_rel_error"):
            if a[key] > 0 and b[key] > 0:
                order = np.log2(a[key] / b[key])
                print(f"  {a['nx']:3d} -> {b['nx']:3d}  {key:22s} {order:+.2f}")

    print("\nThe integral metric is the one that matters for C: a scheme can be "
          "visually smooth and still get it badly wrong at these Peclet numbers.")

    if not args.no_write:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "zhao2d_r1f_benchmark.json"
        path.write_text(json.dumps({
            "note": (
                "Analytic convection-diffusion reference for the SUPG element "
                "the cold plate uses. The exact solution is evaluated "
                "analytically at the quadrature points, NOT interpolated from "
                "its nodal values -- it varies by O(1) inside the last element "
                "on every mesh here, so its interpolant is a different "
                "function and comparing against it hides the error."
            ),
            "global_pe": args.pe,
            "layer_width": L / args.pe,
            "thermal_form": dataclasses.asdict(form),
            "rows": rows,
        }, indent=2), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
