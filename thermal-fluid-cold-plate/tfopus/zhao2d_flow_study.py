"""R1h: one fixed design, a coarse or a fine flow, on a common thermal mesh.

R1g kept the flow on the design mesh h and gave the temperature a nested mesh
of its own. On the R1d design, C still moved +8.87% from h_T = h/2 to h/4 on
that coarse flow. R1h asks the complementary question: at a FIXED thermal mesh,
what does the flow solved at h/2 instead of h do to the complete thermal
response, and on the fine flow, does h_T = h/2 -> h/4 still drift?

                     thermal h/2            thermal h/4
    flow h           R1g level 2            R1g level 4
    flow h/2         R1f row D, re-solved   new

This is not a new model and not an optimisation chain. It composes parts that
exist: the flow solver on the refined mesh, `zhao2d_dual`'s nested maps from
that flow mesh to a thermal mesh, and the thermal solver. The physical density
is copied from the parent element down every level -- no filter, no
projection, no design variable on a fine mesh. A production fine-flow model
would still need the design -> fine flow -> fine temperature gradient, which is
not here.

What this module adds is what a cross-mesh comparison needs and the earlier
stages did not have:

* an identity for a flow STATE, stored beside it, so a saved field is reused
  only by the problem it solves -- R1f's fine-flow cache carried none;
* the inlet trace as each flow mesh actually carries it;
* the volume and per-boundary terms of three exact identities, evaluated the
  same way for every cell:

      H     = integral b_f u.grad T + D_T                   divergence theorem
      C_adv = 1/2 boundary integral b_f T^2 u.n - 1/2 D_T2  the same, for T^2 u
      C + D_SUPG - F_SUPG = L_Q                             T as its own test fn

  with D_T = integral b_f T div u and D_T2 = integral b_f T^2 div u, and H the
  boundary enthalpy flux.

These are algebraic statements about ONE solution. D_T and -1/2 D_T2 are shares
of that solution, not errors, and not what C would become if the velocity were
replaced by a divergence-free one: every term moves when the flow does.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import jax
import jax.numpy as jnp

from tfopus import materials as _materials
from tfopus import zhao2d as _z
from tfopus import zhao2d_analysis as _za
from tfopus import zhao2d_dual as _dual
from tfopus import zhao2d_r1 as _r1
from tfopus import zhao2d_refine as _refine
from tfopus import zhao2d_thermal_study as _ts
from tfopus.mesh import Face

# The four tagged boundaries, in reporting order. Every boundary face is one.
BOUNDARIES = (Face.INLET, Face.OUTLET, Face.WALL, Face.SYMMETRY)


def digest(*arrays) -> str:
    """sha256 over dtype, shape and bytes, so a reshaped array cannot collide."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# A flow state and what it belongs to
# --------------------------------------------------------------------------

# What a flow state depends on. The heat source, the conductivities and the
# thermal form are absent on purpose: the coupling is one-way, so none of them
# can move the flow, and binding them would invalidate a state for a change
# that cannot touch it.
_FLOW_SPEC_FIELDS = (
    "inlet_half_width",
    "tab_length",
    "design_half_width",
    "design_height",
    "element_size",
    "inlet_speed",
    "fluid_density",
    "fluid_viscosity",
    "q_alpha",
)


def _canonical(payload: dict) -> dict:
    """The dict as it will read back from JSON (tuples become lists)."""
    return json.loads(json.dumps(payload, sort_keys=True))


def _flatten(payload: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in payload.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def flow_state_identity(problem, s, alpha_max: float) -> dict:
    """Everything the flow state of `problem` at density `s` depends on.

    `problem` is a `Zhao2DProblem` or its dual subclass; only the flow side is
    read. `s` is the physical solid fraction ON THE FLOW MESH -- for a refined
    flow mesh, the parent values copied down, which this records by hash rather
    than trusting a file name.
    """
    mesh = problem.flow_mesh.mesh
    bc = problem.flow_bc.bc
    s = np.asarray(s, dtype=np.float64)
    if s.shape != (mesh.num_elems,):
        raise ValueError(
            f"density has shape {s.shape}; the flow mesh has {mesh.num_elems} elements"
        )
    material = _za.build_material(problem.spec, alpha_max)
    linear = problem.settings["linear"]
    newton = problem.settings["nonlinear"]
    return _canonical({
        "kind": "zhao2d flow state: press_vel, (p, u, v) per node",
        "spec": {f: getattr(problem.spec, f) for f in _FLOW_SPEC_FIELDS},
        "config": {
            "outlet": problem.config.outlet,
            "element_length_mode": problem.config.element_length_mode,
            "flow_form": dataclasses.asdict(problem.config.flow_form),
        },
        "material": {
            "alpha_min": material.alpha_min,
            "alpha_max": float(alpha_max),
            "q_alpha": material.q_alpha_zhou,
        },
        "mesh": {
            "elements": int(mesh.num_elems),
            "nodes": int(mesh.num_nodes),
            "dofs": int(mesh.num_dofs),
            "quadrature": int(mesh.gauss_order),
            "geometry_sha256": digest(
                np.asarray(mesh.nodes.coords), np.asarray(mesh.elem_nodes)
            ),
            "region_sha256": digest(np.asarray(problem.flow_mesh.region)),
        },
        "solid_fraction_sha256": digest(s),
        "dirichlet_sha256": digest(
            np.asarray(bc["fixed_dofs"], dtype=np.int64),
            np.asarray(bc["dirichlet_values"], dtype=np.float64),
        ),
        "solver": {
            "linear": getattr(linear["solver"], "name", str(linear["solver"])),
            "linear_rtol": linear["rtol"],
            "newton_threshold": newton["threshold"],
            "newton_max_iter": newton["max_iter"],
        },
    })


def verify_flow_state(problem, press_vel, s, alpha_max: float, tol: float = 1e-8) -> dict:
    """Check a flow state against the problem it claims to solve; raise if it fails.

    The gate is the relative residual `Zhao2DProblem.require_converged` uses,
    recomputed on the state AS GIVEN rather than on a fresh solve -- so it
    verifies a saved field, which a re-solve would not. The Dirichlet values
    must be carried exactly: Newton never moves a constrained dof.
    """
    x0 = problem.flow_x0
    pv = jnp.asarray(press_vel)
    if pv.shape != x0.shape:
        raise ValueError(f"flow state has shape {pv.shape}; the problem expects {x0.shape}")
    bc = problem.flow_bc.bc
    fixed = np.asarray(bc["fixed_dofs"])
    deviation = float(
        np.max(np.abs(np.asarray(pv)[fixed] - np.asarray(bc["dirichlet_values"])))
    )
    alpha = _materials.brinkman_penalty(
        jnp.asarray(s), _za.build_material(problem.spec, alpha_max)
    )
    res, _ = problem.flow.get_residual_and_tangent_stiffness(pv, alpha)
    res0, _ = problem.flow.get_residual_and_tangent_stiffness(x0, alpha)
    rel = float(jnp.linalg.norm(res) / jnp.maximum(jnp.linalg.norm(res0), 1e-300))
    report = {
        "dofs": int(pv.shape[0]),
        "dirichlet_max_deviation": deviation,
        "residual_relative": rel,
        "gate": tol,
    }
    if deviation != 0.0:
        raise ValueError(
            f"the state does not carry this problem's Dirichlet values "
            f"(max deviation {deviation:g})"
        )
    if not rel <= tol:
        raise _r1.NotConverged(f"flow state: relative residual {rel:.3e} above {tol:g}")
    return report


def save_flow_state(path, press_vel, s, identity: dict, provenance: dict) -> None:
    """The state, the density it was solved for, and what it belongs to -- together."""
    np.savez_compressed(
        path,
        press_vel=np.asarray(press_vel),
        solid_fraction=np.asarray(s),
        identity=np.array(json.dumps(identity, sort_keys=True)),
        provenance=np.array(json.dumps(provenance, sort_keys=True)),
    )


def load_flow_state(path, problem, s, alpha_max: float, tol: float = 1e-8):
    """A saved flow state, only if it belongs to this problem; gated again on load.

    Returns (press_vel, record). Refuses, with the differing fields, a state
    whose stored identity is not this problem's -- the check a bare cached
    array cannot support.
    """
    with np.load(path) as f:
        stored = json.loads(str(f["identity"]))
        provenance = json.loads(str(f["provenance"]))
        press_vel = np.asarray(f["press_vel"])
    current = flow_state_identity(problem, s, alpha_max)
    if stored != current:
        a, b = _flatten(stored), _flatten(current)
        diffs = [
            f"{k}: {a.get(k)!r} -> {b.get(k)!r}"
            for k in sorted(set(a) | set(b))
            if a.get(k) != b.get(k)
        ]
        raise ValueError(
            "the saved flow state belongs to a different problem:\n  " + "\n  ".join(diffs)
        )
    report = verify_flow_state(problem, press_vel, s, alpha_max, tol)
    return jnp.asarray(press_vel), {
        "identity": stored,
        "provenance": provenance,
        "verification": report,
    }


# --------------------------------------------------------------------------
# Inlet traces
# --------------------------------------------------------------------------


def inlet_trace(planar: _z.PlanarMesh, spec: _z.Zhao2DSpec, press_vel) -> dict:
    """The inlet velocity as this flow mesh carries it, and the inflow it gives.

    Read off the STATE, not the boundary-condition table. The two rim nodes are
    shared with the tab wall (x = inlet half width) and the symmetry plane
    (x = 0); `build_flow_bc` lets the inlet win on both, which keeps the inflow
    at exactly U times the inlet width on every mesh. The price is on the wall,
    not the inlet: the tangential velocity falls from -U at the rim to zero at
    the next wall node, a slip one flow element long, so it shortens with the
    flow mesh. Both are recorded.

    Keys starting with "_" hold arrays for `compare_inlet_traces`; drop them
    before writing JSON.
    """
    coords = np.asarray(planar.mesh.nodes.coords)
    vel = np.asarray(press_vel).reshape(-1, 3)[:, 1:]
    nodes = _z.face_nodes(planar, Face.INLET)
    nodes = nodes[np.argsort(coords[nodes, 0])]
    imposed = np.array([0.0, -spec.inlet_speed])
    wall = _z.face_nodes(planar, Face.WALL)
    symmetry = set(_z.face_nodes(planar, Face.SYMMETRY).tolist())

    def node_record(nd):
        return {"x": float(coords[nd, 0]), "y": float(coords[nd, 1]),
                "u": float(vel[nd, 0]), "v": float(vel[nd, 1])}

    wall_set = set(wall.tolist())
    rim_wall = [nd for nd in nodes if nd in wall_set]
    rim_symmetry = [nd for nd in nodes if nd in symmetry]
    slip = []
    for nd in rim_wall:
        # the next wall node along the same wall line, away from the inlet
        same_line = wall[(np.abs(coords[wall, 0] - coords[nd, 0]) < 1e-12)
                         & (wall != nd)]
        nearest = same_line[np.argmin(np.abs(coords[same_line, 1] - coords[nd, 1]))]
        slip.append({
            "rim": node_record(nd),
            "next_wall_node": node_record(nearest),
            "slip_length": float(abs(coords[nearest, 1] - coords[nd, 1])),
        })

    inflow = -boundary_terms(planar, vel)["inlet"]["volume_flux"]
    nominal = spec.inlet_speed * spec.inlet_half_width
    return {
        "nodes": int(len(nodes)),
        "x_range": [float(coords[nodes, 0].min()), float(coords[nodes, 0].max())],
        "y": sorted({float(y) for y in coords[nodes, 1]}),
        "imposed": imposed.tolist(),
        "max_deviation_from_imposed": float(np.max(np.abs(vel[nodes] - imposed))),
        "rim_wall": [node_record(nd) for nd in rim_wall],
        "rim_symmetry": [node_record(nd) for nd in rim_symmetry],
        "wall_slip_next_to_rim": slip,
        "inflow": inflow,
        "nominal": nominal,
        "inflow_over_nominal": inflow / nominal,
        "_x": coords[nodes, 0],
        "_u": vel[nodes],
    }


def compare_inlet_traces(a: dict, b: dict) -> float:
    """Largest difference between two piecewise-linear inlet traces.

    Evaluated at the union of both node sets, where either trace can have a
    kink, so a difference anywhere along the inlet shows up here.
    """
    x = np.union1d(a["_x"], b["_x"])
    return max(
        float(np.max(np.abs(np.interp(x, a["_x"], a["_u"][:, d])
                            - np.interp(x, b["_x"], b["_u"][:, d]))))
        for d in range(a["_u"].shape[1])
    )


def thermal_inlet(planar: _z.PlanarMesh, spec: _z.Zhao2DSpec, temperature) -> dict:
    """The inlet temperature as the thermal state carries it."""
    nodes = _z.face_nodes(planar, Face.INLET)
    t = np.asarray(temperature)[nodes]
    return {
        "nodes": int(len(nodes)),
        "max_deviation_from_inlet_temperature": float(
            np.max(np.abs(t - spec.inlet_temperature))
        ),
    }


# --------------------------------------------------------------------------
# The terms of the identities
# --------------------------------------------------------------------------


def volume_terms(mesh, nodal_velocity, temperature=None, b_f: float = 1.0) -> dict:
    """Divergence integrals; with a temperature, also D_T, D_T2 and int b_f u.grad T.

    At the mesh's own quadrature. For Q1 fields on these axis-aligned
    rectangles div u is linear in each coordinate, and every integrand below is
    a polynomial of degree at most three in each, so 2x2 and 3x3 Gauss are both
    exact: the numbers are properties of the fields, not of the rule.
    """
    shp = jax.vmap(mesh.elem_template.shape_functions)(mesh.gauss_pts)
    vel = jnp.asarray(nodal_velocity)
    temp = None if temperature is None else jnp.asarray(temperature)

    def per_element(node_ids, node_coords):
        grad_n = jax.vmap(
            mesh.elem_template.get_gradient_shape_function_physical, in_axes=(0, None)
        )(mesh.gauss_pts, node_coords)  # (g, n, d)
        _, det = jax.vmap(
            mesh.elem_template.compute_jacobian_and_determinant, in_axes=(0, None)
        )(mesh.gauss_pts, node_coords)
        w = mesh.gauss_weights * det
        u_e = vel[node_ids]
        div_u = jnp.einsum("gnd, nd -> g", grad_n, u_e)
        out = [jnp.sum(div_u * w), jnp.sum(div_u * div_u * w)]
        if temp is not None:
            t_e = temp[node_ids]
            t_g = shp @ t_e
            u_g = shp @ u_e
            grad_t = jnp.einsum("gnd, n -> gd", grad_n, t_e)
            u_grad_t = jnp.einsum("gd, gd -> g", u_g, grad_t)
            out += [jnp.sum(t_g * div_u * w), jnp.sum(t_g * t_g * div_u * w),
                    jnp.sum(u_grad_t * w)]
        return tuple(out)

    parts = jax.vmap(per_element)(jnp.asarray(mesh.elem_nodes), mesh.elem_node_coords)
    out = {
        "integral_div_u": float(jnp.sum(parts[0])),
        "integral_div_u_squared": float(jnp.sum(parts[1])),
    }
    if temp is not None:
        out.update(
            D_T=float(b_f * jnp.sum(parts[2])),
            D_T2=float(b_f * jnp.sum(parts[3])),
            b_f_u_grad_T=float(b_f * jnp.sum(parts[4])),
        )
    return out


def boundary_terms(planar: _z.PlanarMesh, nodal_velocity, temperature=None,
                   kappa=None, b_f: float = 1.0) -> dict:
    """Per boundary tag: volume flux, enthalpy flux, half the T^2 flux, conduction.

    Outward normals throughout. Two-point Gauss along each edge is exact for the
    first three: u.n and T are linear along an edge, so the integrands are at
    most cubic. T is interpolated and then squared -- interpolating nodal T^2
    would be a different function.

    `conduction_out` is -k grad T . n from the element-interior gradient at the
    face midpoint, the same ONE-SIDED estimate `zhao2d_analysis.conservation`
    makes: exact for that one-sided gradient, which is linear along the edge,
    but not the flux the weak form balances. On the adiabatic walls and the
    outlet the weak form's flux is zero and this is not; at the inlet,
    `dirichlet_reaction` gives the weak form's own value.
    """
    if kappa is not None and temperature is None:
        raise ValueError("conduction needs a temperature")
    template = planar.mesh.elem_template
    conn = np.asarray(template.face_connectivity)
    elem_nodes = np.asarray(planar.mesh.elem_nodes)
    coords = np.asarray(planar.mesh.elem_node_coords)
    vel = np.asarray(nodal_velocity)
    temp = None if temperature is None else np.asarray(temperature)
    mid_local = _za.face_midpoints_local(template)
    gradient_at = jax.vmap(template.get_gradient_shape_function_physical)

    keys = ["volume_flux"]
    if temp is not None:
        keys += ["enthalpy_flux", "half_t2_flux"]
    if kappa is not None:
        keys += ["conduction_out"]

    out = {}
    for tag in BOUNDARIES:
        pairs = planar.elem_faces[tag]
        entry = {"faces": len(pairs), **{k: 0.0 for k in keys}}
        if pairs:
            e = np.array([p[0] for p in pairs])
            f = np.array([p[1] for p in pairs])
            ends = elem_nodes[e[:, None], conn[f]]  # (faces, 2) global nodes
            pts = coords[e[:, None], conn[f]]  # (faces, 2, dim)
            edge = pts[:, 1] - pts[:, 0]
            length = np.linalg.norm(edge, axis=1)
            normal = np.stack([edge[:, 1], -edge[:, 0]], axis=1) / length[:, None]
            inward = np.einsum(
                "fd, fd -> f", normal, pts.mean(axis=1) - coords[e].mean(axis=1)
            ) < 0
            normal[inward] *= -1.0

            for tq, wq in zip(_za._GAUSS_T, _za._GAUSS_W):
                u = (1.0 - tq) * vel[ends[:, 0]] + tq * vel[ends[:, 1]]
                un = np.einsum("fd, fd -> f", u, normal) * length * wq
                entry["volume_flux"] += float(np.sum(un))
                if temp is not None:
                    t = (1.0 - tq) * temp[ends[:, 0]] + tq * temp[ends[:, 1]]
                    entry["enthalpy_flux"] += b_f * float(np.sum(t * un))
                    entry["half_t2_flux"] += 0.5 * b_f * float(np.sum(t * t * un))
            if kappa is not None:
                grad_n = np.asarray(
                    gradient_at(jnp.asarray(mid_local[f]), jnp.asarray(coords[e]))
                )  # (faces, nodes, dim)
                grad_t = np.einsum("fnd, fn -> fd", grad_n, temp[elem_nodes[e]])
                entry["conduction_out"] = float(np.sum(
                    -np.asarray(kappa)[e] * np.einsum("fd, fd -> f", grad_t, normal)
                    * length
                ))
        out[tag.name.lower()] = entry
    out["total"] = {k: sum(out[b.name.lower()][k] for b in BOUNDARIES) for k in keys}
    return out


def dirichlet_reaction(solver, temperature, elem_velocity, kappa, q_source) -> dict:
    """The thermal residual at the Dirichlet dofs, before they are zeroed.

    The Q1 shape functions sum to one, so summing the assembled residual over
    ALL nodes tests the discrete equation with v = 1, for which the conduction
    and the SUPG terms vanish:

        sum_i R_i = integral b_f u.grad T - integral Q.

    At a converged state the free-node residuals are zero to solver precision,
    so this is the sum over the Dirichlet (inlet) nodes alone: the flux the
    discrete equations need there to hold T fixed -- the weak form's own,
    consistent account of the heat conducted through the inlet, INTO the
    domain when positive. It is the bracket in

        H - Q = D_T + [integral b_f u.grad T - Q],

    small, and not zero.
    """
    mesh = solver.mesh
    args = (
        jnp.asarray(temperature)[mesh.elem_dof_mat],
        elem_velocity,
        kappa,
        q_source,
        mesh.elem_node_coords,
        solver.elem_length,
        solver.tau_elem if solver.tau_elem is not None else jnp.zeros(mesh.num_elems),
    )
    elem_res = jax.vmap(solver.element_residual)(*args)
    full = np.asarray(jnp.zeros(mesh.num_dofs).at[mesh.elem_dof_mat].add(elem_res))
    fixed = np.asarray(solver.bc["fixed_dofs"])
    free = np.asarray(solver.bc["free_dofs"])
    return {
        "dirichlet_nodes": int(len(fixed)),
        "reaction_at_dirichlet_nodes": float(full[fixed].sum()),
        "sum_over_free_nodes": float(full[free].sum()),
        "sum_over_all_nodes": float(full.sum()),
    }


# --------------------------------------------------------------------------
# One cell of the matrix
# --------------------------------------------------------------------------


def _quantiles(a) -> dict:
    a = np.asarray(a)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def cell_report(problem: _dual.Zhao2DDualProblem, s_flow, press_vel, temperature,
                alpha_max: float, scale: _r1.ReferenceValues | None = None) -> dict:
    """Everything reported for one (flow mesh, thermal mesh) cell, from its states.

    Nothing is solved: the states come in and every number is evaluated on
    them, including the residual gate. `s_flow` is the physical density on
    `problem`'s flow mesh. `scale`, if given, is only the stated reporting
    scale for J*, never a normalisation of this model.
    """
    spec = problem.spec
    s_flow = jnp.asarray(s_flow)
    material = _za.build_material(spec, alpha_max)
    alpha = _materials.brinkman_penalty(s_flow, material)
    kappa_t = problem.thermal_conductivity(s_flow, alpha_max)
    vel_t = problem.thermal_velocity(press_vel)
    nodal_t = problem.thermal_nodal_velocity(press_vel)
    q = problem.q_source
    temp = jnp.asarray(temperature)

    residuals = problem.residual_norms_at(press_vel, temp, alpha, kappa_t)
    split = problem.thermal.compliance_decomposition(temp, vel_t, kappa_t, q)
    psi = float(problem.flow.dissipated_power(press_vel, alpha))
    heat_in = float(np.sum(np.asarray(q) * np.asarray(problem.thermal_mesh.elem_area)))

    u_flow = np.asarray(press_vel).reshape(-1, problem.flow.num_fields)[:, 1:]
    vol = volume_terms(problem.thermal_mesh.mesh, nodal_t, temp, spec.b_f)
    vol_flow = volume_terms(problem.flow_mesh.mesh, u_flow)
    bnd = boundary_terms(problem.thermal_mesh, nodal_t, temp, kappa_t, spec.b_f)
    bnd_flow = boundary_terms(problem.flow_mesh, u_flow)
    reaction = dirichlet_reaction(problem.thermal, temp, vel_t, kappa_t, q)

    c = split["compliance"]
    h_flux = bnd["total"]["enthalpy_flux"]
    adv = vol["b_f_u_grad_T"]
    d_t, d_t2 = vol["D_T"], vol["D_T2"]
    half_t2 = bnd["total"]["half_t2_flux"]
    bracket = adv - heat_in

    # conservation() reads velocity from a press_vel layout; pressure is unused.
    # Kept so these cells carry the same "heat" block R1g recorded.
    pv_t = np.zeros((nodal_t.shape[0], 3))
    pv_t[:, 1:] = np.asarray(nodal_t)
    heat = {k: float(v) for k, v in _za.conservation(
        spec, problem.thermal_mesh, pv_t.ravel(), temp, q, kappa_t
    ).items()}

    s_t = np.asarray(problem.maps.density(s_flow))
    tau = np.asarray(_ts.element_tau(problem.thermal, vel_t, kappa_t))
    h_t = spec.element_size / problem.thermal_refinement
    inflow = -bnd_flow["inlet"]["volume_flux"]
    outflow = bnd_flow["outlet"]["volume_flux"]

    report = {
        "mesh": {
            "h_flow": spec.element_size,
            "flow_elements": int(problem.flow_mesh.num_elems),
            "flow_quadrature": int(problem.flow_mesh.mesh.gauss_order),
            "h_thermal": h_t,
            "thermal_elements": int(problem.thermal_mesh.num_elems),
            "thermal_nodes": int(problem.thermal_mesh.mesh.num_nodes),
            "thermal_quadrature": int(problem.thermal_quadrature),
            "maps_flow_to_thermal": problem.maps.identity(),
            "nesting": problem.nesting,
        },
        "residual_relative": residuals,
        "psi": psi,
        **split,
        "t_max": float(jnp.max(temp)),
        **_refine.undershoot(temp, spec.inlet_temperature),
        "heat_in": heat_in,
        "mass": {
            "inflow": inflow,
            "outflow": outflow,
            "imbalance_relative": abs(inflow - outflow) / abs(inflow),
            "wall_and_symmetry_flux": bnd_flow["wall"]["volume_flux"]
            + bnd_flow["symmetry"]["volume_flux"],
            "note": "on the flow mesh, from the flow state itself",
        },
        "divergence": {
            "flow_mesh": vol_flow,
            "thermal_mesh": {k: vol[k] for k in ("integral_div_u", "integral_div_u_squared")},
            "D_T": d_t,
            "D_T2": d_t2,
            "D_T_over_heat_in": d_t / heat_in,
            "minus_half_D_T2": -0.5 * d_t2,
            "minus_half_D_T2_over_C": -0.5 * d_t2 / c,
            "note": "algebraic shares of this solution, not errors and not the "
                    "change a divergence-free velocity would produce",
        },
        "identities": {
            "enthalpy": {
                "statement": "H = int b_f u.grad T + D_T",
                "H": h_flux,
                "b_f_u_grad_T": adv,
                "D_T": d_t,
                "closure": h_flux - adv - d_t,
                "closure_relative": abs(h_flux - adv - d_t) / abs(h_flux),
            },
            "advective_half": {
                "statement": "C_adv = 1/2 boundary int b_f T^2 u.n - 1/2 D_T2",
                "c_advective": split["c_advective"],
                "half_boundary_t2_flux": half_t2,
                "minus_half_D_T2": -0.5 * d_t2,
                "closure": split["c_advective"] - half_t2 + 0.5 * d_t2,
                "closure_relative": abs(split["c_advective"] - half_t2 + 0.5 * d_t2)
                / abs(split["c_advective"]),
            },
            "source": {
                "statement": "H - Q = D_T + [int b_f u.grad T - Q]; the bracket is "
                             "the Dirichlet reaction",
                "H_minus_heat_in": h_flux - heat_in,
                "D_T": d_t,
                "bracket": bracket,
                "dirichlet_reaction": reaction["reaction_at_dirichlet_nodes"],
                "bracket_minus_reaction": bracket - reaction["reaction_at_dirichlet_nodes"],
            },
            "stabilisation": {
                "statement": "C + D_SUPG - F_SUPG = L_Q",
                "closure": split["closure"],
                "closure_relative": split["closure_relative"],
            },
        },
        "boundary": bnd,
        "reaction": reaction,
        "heat": heat,
        "tau": {"whole_domain": _quantiles(tau), "fluid_s_lt_0.5": _quantiles(tau[s_t < 0.5])},
        "peclet": _ts.peclet_statistics(
            problem.thermal_mesh, vel_t, kappa_t, spec.b_f, h_t, s_t
        ),
    }
    if scale is not None:
        report["j_star"] = float(
            _dual.reporting_objective(psi, c, scale, problem.config.weight)
        )
    return report
