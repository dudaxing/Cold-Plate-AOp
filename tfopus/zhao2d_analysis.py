"""Fixed-design analysis of Zhao's 2D heat sink. No optimisation happens here.

R0's job: given an explicit reading of figure 7 and section 4.1, solve the flow
and the temperature on a FIXED pseudo-density field, compute Zhao's Psi (eq 24)
and C (eq 23), and report the conservation and convergence diagnostics that
decide whether those numbers mean anything.

The reported Psi_0 = 20,816 and C_0 = 0.0456 are carried through as comparison
values only. Nothing here is tuned to reach them.

Two facts make this a useful discriminator, in this order:

*   Psi depends only on the geometry, the flow model, the reference field and
    alpha_max. It does not involve the thermal model at all.
*   C then depends on the temperature, given that same velocity -- so it tests
    the thermal stabilisation convention with the flow interpretation already
    pinned.

So a sweep over the undetermined choices separates cleanly: if no flow-side
combination reproduces Psi_0, the thermal conventions cannot be the explanation.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp

import toflux.src.solver as _solver

from tfopus import elements as _elements
from tfopus import fe_flow as _fe_flow
from tfopus import fe_thermal as _fe_thermal
from tfopus import materials as _materials
from tfopus import zhao2d as _z
from tfopus.mesh import Face


@dataclasses.dataclass(frozen=True)
class CaseOptions:
    """Every undetermined choice in one place, so a run names its own reading."""

    reference: _z.ReferenceField = _z.ReferenceField.TABS_FLUID
    outlet: _z.OutletKind = _z.OutletKind.TRACTION
    source: _z.SourceRegion = _z.SourceRegion.WHOLE_DOMAIN
    alpha_max: float = 1.0e6
    flow_form: _fe_flow.FlowForm = _fe_flow.ZHAO_FORM
    thermal_form: _fe_thermal.ThermalForm = _fe_thermal.ZHAO_FORM
    element_length_mode: str = "min_edge"

    def label(self) -> str:
        return (
            f"ref={self.reference.value} out={self.outlet.value} "
            f"src={self.source.value} amax={self.alpha_max:.0e}"
        )


def default_solver_settings(max_iter: int = 40) -> dict:
    """scipy sparse direct: no PETSc needed, and bit-reproducible."""
    return {
        "linear": {"solver": _solver.LinearSolvers.SCIPY_SPARSE, "rtol": 1e-10},
        "nonlinear": {"max_iter": max_iter, "threshold": 1e-11},
    }


def build_material(spec: _z.Zhao2DSpec, alpha_max: float):
    """Zhao table 1 in `tfopus.materials`' container.

    The solid phase carries only its conductivity: section 2 puts the fluid
    rho*c over the whole domain, so table 1's solid density 2700 and heat
    capacity 900 are never used here.

    They are left at zero. Note what that does and does not buy: zero is a
    marker, NOT a guard. `Phase.volumetric_heat_capacity` returns 0.0 for this
    phase without raising, so a code path that started using a two-phase rho*c
    would silently get zero rather than failing. The protection against that is
    that every consumer takes the single `b_f` off the material object; there is
    no separate defensive mechanism, and this docstring previously claimed one.
    """
    return _materials.ThermoFluidMaterial(
        fluid=_materials.Phase(
            "fluid",
            spec.fluid_density,
            spec.fluid_conductivity,
            spec.fluid_heat_capacity,
            spec.fluid_viscosity,
        ),
        solid=_materials.Phase("solid", 0.0, spec.solid_conductivity, 0.0),
        alpha_min=0.0,
        alpha_max=alpha_max,
        q_alpha_zhou=spec.q_alpha,
        q_k_zhou=spec.q_kappa,
    )


def _residual_norm(problem, state, *params) -> float:
    """Recompute the residual at `state`.

    Upstream's Newton loop stores the residual from BEFORE the step it then
    takes, and prints "converged" whether or not it converged, so its own report
    cannot decide whether a solution is acceptable. This measures it.
    """
    res, _ = problem.get_residual_and_tangent_stiffness(state, *params)
    return float(jnp.linalg.norm(res))


def solve_flow(spec, planar, s, options, settings):
    """Navier-Stokes-Brinkman on a fixed design. Returns (state, alpha, report)."""
    material = build_material(spec, options.alpha_max)
    alpha = _materials.brinkman_penalty(s, material)

    flow_bc = _z.build_flow_bc(planar, spec, options.outlet)
    solver = _fe_flow.FlowSolver(
        planar.mesh,
        flow_bc.bc,
        density=spec.fluid_density,
        viscosity=spec.fluid_viscosity,
        solver_settings=settings,
        elem_length=_elements.element_lengths(planar.mesh, options.element_length_mode),
        form=options.flow_form,
    )
    x0 = (
        jnp.zeros((planar.mesh.num_dofs,))
        .at[flow_bc.bc["fixed_dofs"]]
        .set(flow_bc.bc["dirichlet_values"])
    )
    state = _solver.modified_newton_raphson_solve(solver, x0, alpha)
    report = {
        "flow_residual": _residual_norm(solver, state, alpha),
        "flow_residual_initial": _residual_norm(solver, x0, alpha),
        "num_fixed_flow_dofs": int(len(flow_bc.bc["fixed_dofs"])),
    }
    report["flow_residual_relative"] = report["flow_residual"] / max(
        report["flow_residual_initial"], 1e-300
    )
    return solver, state, alpha, report


def solve_thermal(spec, planar, s, elem_vel, options, settings):
    """Convection-conduction with the volumetric source, on the given velocity."""
    material = build_material(spec, options.alpha_max)
    kappa = _materials.conductivity(s, material)

    bc = _z.build_thermal_bc(planar, spec)
    q_source = _z.heat_source_field(planar, spec, options.source)
    solver = _fe_thermal.ThermalSolver(
        planar.mesh,
        bc,
        b_f=spec.b_f,
        solver_settings=settings,
        elem_length=_elements.element_lengths(planar.mesh, options.element_length_mode),
        form=options.thermal_form,
    )
    t0 = jnp.zeros((planar.mesh.num_dofs,)).at[bc["fixed_dofs"]].set(
        bc["dirichlet_values"]
    )
    state = _solver.modified_newton_raphson_solve(
        solver, t0, elem_vel, kappa, q_source
    )
    report = {
        "thermal_residual": _residual_norm(solver, state, elem_vel, kappa, q_source),
        "thermal_residual_initial": _residual_norm(
            solver, t0, elem_vel, kappa, q_source
        ),
    }
    report["thermal_residual_relative"] = report["thermal_residual"] / max(
        report["thermal_residual_initial"], 1e-300
    )
    return solver, state, kappa, q_source, report


# Two-point Gauss on a straight edge, in the parameter t in [0, 1]. Exact for
# quadratics, which is what a product of two linear traces is.
_GAUSS_T = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
_GAUSS_W = np.array([0.5, 0.5])


def _edge_integral(planar, tag: Face, per_face):
    """Sum per_face(nodes, outward_normal, length) over the tagged edges.

    `per_face` receives the two end-node indices and is expected to return an
    already-integrated quantity for that edge; use `_edge_quadrature` for
    integrands that are not linear in the nodal values.
    """
    conn = np.asarray(planar.mesh.elem_template.face_connectivity)
    elem_nodes = np.asarray(planar.mesh.elem_nodes)
    coords = np.asarray(planar.mesh.elem_node_coords)
    total = 0.0
    for e, f in planar.elem_faces[tag]:
        pts = coords[e][conn[f]]
        edge = pts[1] - pts[0]
        length = float(np.linalg.norm(edge))
        normal = np.array([edge[1], -edge[0]]) / length
        if np.dot(normal, pts.mean(axis=0) - coords[e].mean(axis=0)) < 0:
            normal = -normal
        total += per_face(elem_nodes[e][conn[f]], normal, length)
    return total


def _edge_quadrature(nodal_a, nodal_b, nodes, normal, length):
    """integral over the edge of (a.n) * b, with a and b linear in t.

    The trapezoid rule -- mean(a).n * mean(b) * length -- is NOT this integral:
    the product of two linear traces is quadratic, and the two differ by
    (a_1 - a_0).n (b_1 - b_0) * length / 12. On the inlet, where u_n is uniform
    but T varies, they happen to agree; on the outlet, where both vary, they do
    not.
    """
    a0, a1 = nodal_a[nodes[0]], nodal_a[nodes[1]]
    b0, b1 = nodal_b[nodes[0]], nodal_b[nodes[1]]
    total = 0.0
    for t, w in zip(_GAUSS_T, _GAUSS_W):
        a = (1.0 - t) * a0 + t * a1
        b = (1.0 - t) * b0 + t * b1
        total += w * float(np.dot(a, normal)) * float(b)
    return total * length


def conservation(spec, planar, press_vel, temperature, q_source, kappa=None) -> dict:
    """Mass and energy budgets measured on the discrete solution.

    The energy budget closes only if BOTH boundary transports are counted:

        integral_Omega Q dOmega = integral_Gamma b_f (u.n) T dGamma
                                - integral_Gamma k (grad T).n dGamma

    The conductive term is not optional. The inlet carries a Dirichlet
    temperature, which does not make it adiabatic -- there is generally a
    non-zero conductive flux there, and at the large kappa the reference field
    produces it is not small. `energy_imbalance_rel` counts both; the two
    transports are also reported separately so a failure can be attributed.

    The conductive flux is evaluated from the element-interior temperature
    gradient at the face midpoint, which is a ONE-SIDED estimate, not the
    consistent nodal flux the weak form would recover. It is a diagnostic at
    discretisation accuracy, so a residual imbalance of the order of the
    discretisation error is expected and does not by itself indicate a defect.
    """
    vel = np.asarray(press_vel).reshape(-1, 3)[:, 1:]
    temp = np.asarray(temperature)

    def flux(nodes, normal, length):
        return _edge_quadrature(vel, np.ones(len(temp)), nodes, normal, length)

    def enthalpy(nodes, normal, length):
        return spec.b_f * _edge_quadrature(vel, temp, nodes, normal, length)

    inflow = -_edge_integral(planar, Face.INLET, flux)
    outflow = _edge_integral(planar, Face.OUTLET, flux)
    wall_leak = _edge_integral(planar, Face.WALL, flux) + _edge_integral(
        planar, Face.SYMMETRY, flux
    )
    nominal = spec.inlet_speed * spec.inlet_half_width

    heat_in = float(jnp.sum(q_source * jnp.asarray(planar.elem_area)))
    net_enthalpy = sum(
        _edge_integral(planar, tag, enthalpy)
        for tag in (Face.INLET, Face.OUTLET, Face.WALL, Face.SYMMETRY)
    )
    net_conduction = (
        _conductive_outflow(planar, temp, kappa) if kappa is not None else None
    )

    out = {
        "mass_in": inflow,
        "mass_out": outflow,
        "mass_imbalance_rel": abs(inflow - outflow) / max(abs(inflow), 1e-300),
        "wall_leak_rel": abs(wall_leak) / max(abs(inflow), 1e-300),
        "inlet_flux_over_nominal": inflow / nominal,
        "heat_in": heat_in,
        "enthalpy_net_out": net_enthalpy,
    }
    if net_conduction is None:
        # Without kappa only the advective half can be formed. Name it for what
        # it is rather than calling it an energy balance.
        out["enthalpy_minus_source_rel"] = abs(net_enthalpy - heat_in) / max(
            abs(heat_in), 1e-300
        )
        return out

    out["conduction_net_out"] = net_conduction
    out["energy_imbalance_rel"] = abs(net_enthalpy + net_conduction - heat_in) / max(
        abs(heat_in), 1e-300
    )
    out["temperature_weighted_divergence"] = temperature_weighted_divergence(
        planar, press_vel, temperature, spec.b_f
    )
    out["temperature_weighted_divergence_rel"] = abs(
        out["temperature_weighted_divergence"]
    ) / max(abs(heat_in), 1e-300)
    return out


# Local coordinates of the Quad4 reference nodes, in isoparametric order.
_REF_NODES_2D = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def face_midpoints_local(template) -> np.ndarray:
    """(num_faces, 2) isoparametric coordinates of each face midpoint.

    Read off `face_connectivity` rather than recovered by inverting the
    isoparametric map: the midpoint of a face in LOCAL coordinates is always the
    mean of its nodes' local coordinates, whatever the element's physical shape.

    The inverse map is not available anyway. `Quad4.get_isoparametric_coordinate_of_point`
    raises NotImplementedError upstream (only `Rect4` implements it), so the
    earlier version of `_conductive_outflow` raised on every call -- it was never
    exercised, because R1d's `_evaluate` does not call `conservation`.
    """
    conn = np.asarray(template.face_connectivity)
    return _REF_NODES_2D[conn].mean(axis=1)


def _conductive_outflow(planar, temp, kappa) -> float:
    """-integral_Gamma k (grad T).n dGamma over every boundary face.

    A ONE-SIDED estimate from the element-interior gradient, not the consistent
    nodal flux the weak form would recover, so it carries discretisation-level
    accuracy and its residual should not be read as the energy error of C.
    """
    template = planar.mesh.elem_template
    conn = np.asarray(template.face_connectivity)
    coords = np.asarray(planar.mesh.elem_node_coords)
    elem_nodes = np.asarray(planar.mesh.elem_nodes)
    kappa = np.asarray(kappa)
    mid_local = face_midpoints_local(template)

    total = 0.0
    for tag, pairs in planar.elem_faces.items():
        if tag is Face.NONE:
            continue
        for e, f in pairs:
            pts = coords[e][conn[f]]
            edge = pts[1] - pts[0]
            length = float(np.linalg.norm(edge))
            normal = np.array([edge[1], -edge[0]]) / length
            if np.dot(normal, pts.mean(axis=0) - coords[e].mean(axis=0)) < 0:
                normal = -normal
            grad_n = np.asarray(
                template.get_gradient_shape_function_physical(
                    jnp.asarray(mid_local[f]), jnp.asarray(coords[e])
                )
            )
            grad_t = grad_n.T @ temp[elem_nodes[e]]
            total += -float(kappa[e]) * float(np.dot(grad_t, normal)) * length
    return total


def temperature_weighted_divergence(planar, press_vel, temperature, b_f) -> float:
    """integral b_f T (div u) dOmega.

    Zero for a pointwise divergence-free field, and the exact gap between the
    non-conservative convection form the residual uses and the boundary enthalpy
    flux, since div(T u) = u.grad T + T div u. Global mass balance can hold to
    machine precision while this term does not vanish, so it is reported
    separately rather than folded into an energy residual.
    """
    mesh = planar.mesh
    shp = jax.vmap(mesh.elem_template.shape_functions)(mesh.gauss_pts)
    vel = jnp.asarray(press_vel).reshape(-1, 3)[:, 1:]

    def per_element(node_ids, node_coords):
        grad_n = jax.vmap(
            mesh.elem_template.get_gradient_shape_function_physical, in_axes=(0, None)
        )(mesh.gauss_pts, node_coords)
        _, det = jax.vmap(
            mesh.elem_template.compute_jacobian_and_determinant, in_axes=(0, None)
        )(mesh.gauss_pts, node_coords)
        u = vel[node_ids]
        t = jnp.asarray(temperature)[node_ids]
        div_u = jnp.einsum("gnd, nd -> g", grad_n, u)
        t_g = jnp.einsum("gn, n -> g", shp, t)
        return jnp.einsum("g, g, g, g -> ", t_g, div_u, mesh.gauss_weights, det)

    per = jax.vmap(per_element)(
        jnp.asarray(mesh.elem_nodes), mesh.elem_node_coords
    )
    return float(b_f * jnp.sum(per))


def analyse(
    spec: _z.Zhao2DSpec,
    options: CaseOptions = CaseOptions(),
    solid_fraction=None,
    settings: dict | None = None,
    with_thermal: bool = True,
) -> dict:
    """Solve a fixed design and report Zhao's metrics plus the diagnostics.

    `solid_fraction` defaults to the reference field named by `options`.
    `with_thermal=False` runs the flow alone, which is all Psi needs.
    """
    settings = settings or default_solver_settings()
    flow_mesh = _z.build_mesh(spec, dofs_per_node=3)
    s = (
        _z.reference_solid_fraction(flow_mesh, spec, options.reference)
        if solid_fraction is None
        else jnp.asarray(solid_fraction)
    )

    flow, press_vel, alpha, report = solve_flow(spec, flow_mesh, s, options, settings)
    psi = float(flow.dissipated_power(press_vel, alpha))

    out = {
        "label": options.label(),
        "psi": psi,
        "psi_over_reported": psi / spec.reported_psi_0,
        "speed_max": float(
            jnp.max(jnp.linalg.norm(press_vel.reshape(-1, 3)[:, 1:], axis=1))
        ),
        "pressure_range": (
            float(jnp.min(press_vel[0::3])),
            float(jnp.max(press_vel[0::3])),
        ),
        **_z.fluid_fractions(flow_mesh, s),
        **report,
    }

    if not with_thermal:
        out["state"] = {"press_vel": press_vel, "alpha": alpha, "s": s}
        return out

    thermal_mesh = _z.build_mesh(spec, dofs_per_node=1)
    elem_vel = flow.element_velocities(press_vel)
    thermal, temperature, kappa, q_source, treport = solve_thermal(
        spec, thermal_mesh, s, elem_vel, options, settings
    )
    compliance = float(thermal.thermal_compliance(temperature, elem_vel, kappa))

    out.update(
        compliance=compliance,
        compliance_over_reported=compliance / spec.reported_c_0,
        temperature_max=float(jnp.max(temperature)),
        temperature_mean=float(jnp.mean(temperature)),
        **treport,
        **conservation(spec, thermal_mesh, press_vel, temperature, q_source, kappa),
    )
    out["state"] = {
        "press_vel": press_vel,
        "temperature": temperature,
        "alpha": alpha,
        "kappa": kappa,
        "s": s,
    }
    out["meshes"] = (flow_mesh, thermal_mesh)
    return out
