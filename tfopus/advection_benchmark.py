"""A high-Peclet convection-diffusion problem with a known exact solution.

Why this exists: on the cold plate there is no exact answer, so comparing
stabilisation variants there by which gives the smaller C would be selection
bias. This problem uses the SAME thermal element and residual and has one:

    b_f u dT/dx - k d2T/dx2 = 0   on (0, L) x (0, H),  u = (U, 0), k constant
    T(0) = 0,  T(L) = 1,  adiabatic top and bottom (the natural condition)

    T(x) = (exp(Pe x / L) - 1) / (exp(Pe) - 1),     Pe = b_f U L / k

flat over most of the domain, rising through an exponential layer of width
~ L/Pe at the Dirichlet outlet.

What it answers is whether the method gets a known answer right at a given
element Peclet number. What it does NOT answer is how fine the cold plate's
thermal mesh must be: an outlet layer at a Dirichlet boundary is not the cold
plate's interior channel-and-solid problem, and its resolution requirement does
not convert into a mesh size there. That question belongs to a refinement study
of the cold plate itself, with physics, design and scheme held fixed.

**Three measurement rules, each learnt from getting it wrong.**

1.  The exact solution is evaluated analytically, never through its nodal
    interpolant. It varies by O(1) inside the last element on every mesh used
    here, so the interpolant is a different function; comparing against it
    made an earlier version report an error that grew under refinement.

2.  Evaluating it analytically is not enough either. A Gauss rule is exact for
    polynomials up to its degree, which says nothing about an exponential layer
    thinner than the element. With a fixed 10-point rule the reference norm
    ITSELF came out 51% low at nx = 10 and 9% low at nx = 20, so the "relative
    error" had a mesh-dependent denominator. The denominator is now the
    closed-form norm, and the numerator is integrated by composite Gauss along
    x, accepted only once doubling the subdivision stops changing it.

3.  Lengths are recorded separately rather than merged into one "h": hx = L/nx
    is the streamwise size, hy = H/ny the cross-stream size, and h_tau the
    length the stabilisation parameter actually uses (min edge). With ny fixed
    at 8, h_tau = hy = 0.03125 at both nx = 10 and nx = 20, so that first
    refinement changes the streamwise resolution and the aspect ratio but NOT
    tau -- which is why the square-element series exists alongside it.
"""

from __future__ import annotations

import numpy as np
import numpy.polynomial.legendre as _leg
import jax
import jax.numpy as jnp

import toflux.src.bc as _bc
import toflux.src.mesher as _mesher
import toflux.src.solver as _solver

from tfopus import elements as _elements
from tfopus import fe_thermal as _fe_thermal
from tfopus import zhao2d_analysis as _za

B_F = 4.18e6  # Zhao table 1, rho * c of the fluid
L, H = 1.0, 0.25
U = 0.2


def conductivity(pe: float) -> float:
    """k giving global Peclet number `pe` = b_f U L / k."""
    return B_F * U * L / pe


def exact(x, pe: float):
    """T(x), written as (exp(k(x-L)) - a)/(1 - a), a = exp(-Pe), to avoid overflow."""
    x = np.asarray(x, dtype=float)
    return np.exp(pe * (x / L - 1.0)) * (-np.expm1(-pe * x / L)) / (-np.expm1(-pe))


def exact_norm_sq(pe: float) -> float:
    """||T||^2 over the domain, in closed form.

    With c = Pe/L and a = exp(-Pe),
        integral_0^L T^2 dx = [(1 - a^2)/(2c) - 2a(1 - a)/c + a^2 L] / (1 - a)^2
    and the y-direction contributes a factor H. At Pe = 1000 this is
    H L / (2 Pe) = 1.25e-4 to all printed digits.
    """
    c = pe / L
    a = np.exp(-pe)
    one_minus_a = -np.expm1(-pe)
    inner = ((1.0 - a * a) / (2.0 * c) - 2.0 * a * one_minus_a / c + a * a * L) / (
        one_minus_a**2
    )
    return float(H * inner)


def exact_diffusive_integral(k: float, pe: float) -> float:
    """integral k |grad T|^2 over the domain = H k Pe / (2 L) coth(Pe / 2)."""
    return float(H * k * pe / (2.0 * L) / np.tanh(pe / 2.0))


def build(nx: int, ny: int, pe: float):
    """Structured nx x ny Quad4 mesh with 3x3 quadrature, T = 0 left, T = 1 right."""
    k = conductivity(pe)
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
    mesh = _mesher.Mesh(
        nodes=_mesher.Nodes(coords=jnp.asarray(coords), dof_per_node=1),
        elem_nodes=elems,
        elem_template=_elements.Quad4(),
        gauss_order=3,
    )

    left = np.nonzero(np.abs(coords[:, 0]) < 1e-12)[0]
    right = np.nonzero(np.abs(coords[:, 0] - L) < 1e-12)[0]
    fixed = np.concatenate([left, right])
    values = np.concatenate([np.zeros(len(left)), np.ones(len(right))])
    bc = _bc.BCDict(
        elem_forces=jnp.zeros((mesh.num_elems, mesh.num_dofs_per_elem)),
        fixed_dofs=fixed,
        free_dofs=np.setdiff1d(np.arange(mesh.num_dofs), fixed),
        dirichlet_values=values,
    )
    return mesh, bc, k, coords


def lengths(mesh) -> dict:
    """hx, hy and the length tau actually uses, reported separately.

    Raises unless every element is the same axis-aligned rectangle, which the
    error integration below relies on (constant det J = hx hy / 4).
    """
    c = np.asarray(mesh.elem_node_coords)  # (e, 4, 2)
    lo, hi = c.min(axis=1), c.max(axis=1)
    hx, hy = hi[:, 0] - lo[:, 0], hi[:, 1] - lo[:, 1]
    if not (np.allclose(hx, hx[0]) and np.allclose(hy, hy[0])):
        raise ValueError("benchmark mesh is not uniform")
    # axis-aligned: each node sits on a corner of its bounding box
    on_corner = (np.isclose(c[..., 0], lo[:, None, 0]) | np.isclose(c[..., 0], hi[:, None, 0])) & (
        np.isclose(c[..., 1], lo[:, None, 1]) | np.isclose(c[..., 1], hi[:, None, 1])
    )
    if not on_corner.all():
        raise ValueError("benchmark mesh is not axis-aligned")
    h_tau = np.asarray(_elements.element_lengths(mesh, "min_edge"))
    return {
        "hx": float(hx[0]),
        "hy": float(hy[0]),
        "h_tau": float(h_tau.max()),
        "h_tau_min": float(h_tau.min()),
        "aspect_hx_over_hy": float(hx[0] / hy[0]),
    }


def l2_error(mesh, nodal, pe: float, rtol: float = 1e-11, points: int = 10,
             max_subdivision: int = 1024) -> dict:
    """||T_h - T|| with T analytic, by composite Gauss verified at two levels.

    Each element is split into m equal sub-intervals along x, each carrying a
    `points`-point Gauss rule; across y a 3-point rule is exact, because T_h is
    linear in y within an element and T does not depend on y. m doubles until
    the integral changes by less than `rtol` relative, and the result is
    refused -- not returned with a caveat -- if that never happens.
    """
    geom = lengths(mesh)
    det = geom["hx"] * geom["hy"] / 4.0
    template = mesh.elem_template
    values = np.asarray(nodal)[np.asarray(mesh.elem_nodes)]  # (e, 4)
    coords = np.asarray(mesh.elem_node_coords)
    z_x, w_x = _leg.leggauss(points)
    z_y, w_y = _leg.leggauss(3)

    def integral(m: int) -> float:
        offsets = -1.0 + (2.0 * np.arange(m) + 1.0) / m  # sub-interval midpoints
        xi = (offsets[:, None] + z_x[None, :] / m).ravel()
        wxi = np.tile(w_x / m, m)
        pts = np.array([[a, b] for a in xi for b in z_y])
        wts = np.array([wa * wb for wa in wxi for wb in w_y])
        shp = np.asarray(jax.vmap(template.shape_functions)(jnp.asarray(pts)))  # (g, 4)
        x_q = np.einsum("gn, en -> eg", shp, coords[..., 0])
        t_h = np.einsum("gn, en -> eg", shp, values)
        err = t_h - exact(x_q, pe)
        return float(np.einsum("eg, g -> ", err * err, wts) * det)

    m = 1
    prev = integral(m)
    while True:
        m *= 2
        if m > max_subdivision:
            raise RuntimeError(
                f"L2 error integral not converged at {m // 2} sub-intervals per "
                f"element (rtol {rtol:g}); refusing to report it"
            )
        cur = integral(m)
        change = abs(cur - prev) / max(abs(cur), 1e-300)
        if change < rtol:
            return {
                "error_sq": cur,
                "subintervals": m,
                "points_per_subinterval": points,
                "last_relative_change": change,
            }
        prev = cur


def run(nx: int, ny: int, pe: float, form: _fe_thermal.ThermalForm) -> dict:
    """Solve on an nx x ny mesh and report the errors, with every length named."""
    mesh, bc, k, coords = build(nx, ny, pe)
    geom = lengths(mesh)
    solver = _fe_thermal.ThermalSolver(
        mesh,
        bc,
        b_f=B_F,
        solver_settings=_za.default_solver_settings(),
        elem_length=_elements.element_lengths(mesh, "min_edge"),
        form=form,
    )
    n_el = mesh.num_elems
    vel = jnp.tile(jnp.array([U, 0.0]), (n_el, 4))
    kappa = jnp.full(n_el, k)
    q = jnp.zeros(n_el)

    t0 = jnp.zeros((mesh.num_dofs,)).at[bc["fixed_dofs"]].set(bc["dirichlet_values"])
    T = np.asarray(_solver.modified_newton_raphson_solve(solver, t0, vel, kappa, q))

    err = l2_error(mesh, T, pe)
    norm_sq = exact_norm_sq(pe)
    split = solver.compliance_decomposition(jnp.asarray(T), vel, kappa, q)
    diff_exact = exact_diffusive_integral(k, pe)
    return {
        "nx": nx,
        "ny": ny,
        **geom,
        "pe_x": B_F * U * geom["hx"] / (2.0 * k),
        "pe_tau": B_F * U * geom["h_tau"] / (2.0 * k),
        "l2_error": float(np.sqrt(err["error_sq"])),
        "l2_error_rel": float(np.sqrt(err["error_sq"] / norm_sq)),
        "l2_quadrature": {k_: v for k_, v in err.items() if k_ != "error_sq"},
        "max_nodal_error": float(np.max(np.abs(T - exact(coords[:, 0], pe)))),
        "diffusive_integral": float(split["c_diffusive"]),
        "diffusive_exact": diff_exact,
        "diffusive_rel_error": abs(float(split["c_diffusive"]) - diff_exact) / diff_exact,
        "overshoot": float(T.max() - 1.0),
        "undershoot": float(-T.min()),
    }


def observed_orders(rows: list[dict], length: str = "hx") -> list[dict]:
    """log(e_a/e_b)/log(h_a/h_b) between successive rows, for both error measures."""
    out = []
    for a, b in zip(rows, rows[1:]):
        ratio = np.log(a[length] / b[length])
        out.append({
            "from_nx": a["nx"],
            "to_nx": b["nx"],
            "l2": float(np.log(a["l2_error_rel"] / b["l2_error_rel"]) / ratio),
            "diffusive": float(
                np.log(a["diffusive_rel_error"] / b["diffusive_rel_error"]) / ratio
            ),
        })
    return out
