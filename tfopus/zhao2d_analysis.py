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
    capacity 900 are never used. They are left at zero rather than filled in, so
    that any code path that starts using them fails loudly instead of silently
    adopting an untested two-phase rule.
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


def _edge_integral(planar, tag: Face, per_face):
    """Sum per_face(nodes, outward_normal, length) over the tagged edges.

    A Quad4 face is a straight two-node segment, so the trapezoid rule is exact
    for the bilinear trace.
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


def conservation(spec, planar, press_vel, temperature, q_source) -> dict:
    """Mass and energy budgets measured on the discrete solution."""
    vel = np.asarray(press_vel).reshape(-1, 3)[:, 1:]
    temp = np.asarray(temperature)

    def flux(nodes, normal, length):
        return float(np.dot(vel[nodes].mean(axis=0), normal)) * length

    def enthalpy(nodes, normal, length):
        return (
            float(np.dot(vel[nodes].mean(axis=0), normal))
            * length
            * float(temp[nodes].mean())
            * spec.b_f
        )

    inflow = -_edge_integral(planar, Face.INLET, flux)
    outflow = _edge_integral(planar, Face.OUTLET, flux)
    wall_leak = _edge_integral(planar, Face.WALL, flux) + _edge_integral(
        planar, Face.SYMMETRY, flux
    )
    nominal = spec.inlet_speed * spec.inlet_half_width

    heat_in = float(jnp.sum(q_source * jnp.asarray(planar.elem_area)))
    net_enthalpy = _edge_integral(planar, Face.OUTLET, enthalpy) + _edge_integral(
        planar, Face.INLET, enthalpy
    )
    return {
        "mass_in": inflow,
        "mass_out": outflow,
        "mass_imbalance_rel": abs(inflow - outflow) / max(abs(inflow), 1e-300),
        "wall_leak_rel": abs(wall_leak) / max(abs(inflow), 1e-300),
        "inlet_flux_over_nominal": inflow / nominal,
        "heat_in": heat_in,
        "enthalpy_net_out": net_enthalpy,
        "energy_imbalance_rel": abs(net_enthalpy - heat_in) / max(abs(heat_in), 1e-300),
    }


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
        **conservation(spec, thermal_mesh, press_vel, temperature, q_source),
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
