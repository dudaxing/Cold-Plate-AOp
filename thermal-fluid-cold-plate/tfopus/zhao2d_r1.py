"""R1: the frozen configuration, and a differentiable objective chain.

Two things live here that R0 deliberately did not have.

**A frozen configuration.** `zhao2d_analysis.CaseOptions` still defaults to
`ZHAO_FORM` for both physics, and its `label()` records the reference domain,
outlet, source region and alpha_max but NOT the formulation switches. So a run
can be reproducible-looking and still be a different discretisation. `R1Config`
carries every switch, and `fingerprint()` serialises all of them, so a stored
reference value can be checked against the configuration that produced it.

**A differentiable objective.** `zhao2d_analysis.analyse()` is a reporting
entry point: it converts to Python floats and computes diagnostics in NumPy, so
`jax.grad(analyse)` cannot work. `objective_and_constraint` below is the same
physics with everything kept in JAX arrays, and the reporting left outside.

The chain:

    x (design variables on the design domain)
      -> scatter, tabs pinned to fluid
      -> filter (sparse, KD-tree)  -> projection
      -> s (num_elems,)
      -> alpha(s), kappa(s)
      -> press_vel = newton(flow, alpha)          implicit diff
      -> element velocities
      -> T = newton(thermal, u, kappa, Q)         implicit diff
      -> Psi(press_vel, alpha),  C(T, u, kappa)
      -> J = w Psi/Psi_0 + (1-w) C/C_0,  g = v_f/v_f_max - 1

alpha_max continuation moves the STATE, never the denominators: Psi_0 and C_0
are computed once on the frozen reference and then held, so J stays a ratio
against one fixed scale rather than a moving one.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import jax
import jax.numpy as jnp

import toflux.src.solver as _solver

from tfopus import design as _design
from tfopus import elements as _elements
from tfopus import fe_flow as _fe_flow
from tfopus import fe_thermal as _fe_thermal
from tfopus import materials as _materials
from tfopus import zhao2d as _z
from tfopus import zhao2d_analysis as _za


# The R1 main line, decided 2026-09-22. Every switch is explicit; none is
# inherited from a module default.
R1_FLOW_FORM = _fe_flow.FlowForm(
    brinkman_in_tau=True,  # tau_u keeps the reactive limit
    brinkman_in_supg_residual=True,  # alpha*u stays in the SUPG/PSPG residual
    viscous_form="symmetric",  # matches the traction outlet and the Psi integral
)
R1_THERMAL_FORM = _fe_thermal.ThermalForm(
    tau="two_limit",  # finite as u -> 0, unlike h/(2|u|)
    stabilise_source=True,  # Q enters the strong residual, so a balanced
    #                          field gets no spurious stabilisation
    supg_heat_capacity=True,  # S_T carries b_f, so it can be added to K_c,T
)


class VolumeDomain:
    """Which domain equation 26's v_f is measured over."""

    DESIGN = "design"
    WHOLE = "whole"


@dataclasses.dataclass(frozen=True)
class R1Config:
    """Everything that fixes the discrete problem. Stored beside every result.

    `volume_domain` defaults to the design domain. Equation 26 writes
    (1/|Omega|) integral over Omega, but with fluid tabs the whole-domain
    fraction of the reference state is 0.4231, so a whole-domain bound of 0.4
    would make the paper's own reference model infeasible. Restricting the
    constraint to the design domain makes it exactly 0.4. Both are always
    reported; this only chooses which one MMA sees.
    """

    reference: str = _z.ReferenceField.TABS_FLUID.value
    outlet: str = _z.OutletKind.TRACTION.value
    source: str = _z.SourceRegion.WHOLE_DOMAIN.value
    alpha_max_reference: float = 1.0e6
    element_length_mode: str = "min_edge"
    volume_domain: str = VolumeDomain.DESIGN
    max_fluid_fraction: float = 0.40
    weight: float = 0.5

    # Not fixed by the paper. The CBS band epsilon = 0.75h is an interface
    # smoothing width for a geometric parametrisation, NOT a density filter
    # radius, so it is not reused here. This value is provisional: it is large
    # enough to suppress checkerboarding on this mesh and has not been chosen
    # by a length-scale study.
    filter_radius_elements: float = 2.0
    projection_beta: float = 0.0

    flow_form: _fe_flow.FlowForm = R1_FLOW_FORM
    thermal_form: _fe_thermal.ThermalForm = R1_THERMAL_FORM

    def fingerprint(self) -> str:
        """Stable JSON of every switch, for binding a stored reference value."""
        payload = {
            f.name: (
                dataclasses.asdict(getattr(self, f.name))
                if dataclasses.is_dataclass(getattr(self, f.name))
                else getattr(self, f.name)
            )
            for f in dataclasses.fields(self)
        }
        return json.dumps(payload, sort_keys=True)

    @property
    def reference_field(self) -> _z.ReferenceField:
        return _z.ReferenceField(self.reference)

    @property
    def outlet_kind(self) -> _z.OutletKind:
        return _z.OutletKind(self.outlet)

    @property
    def source_region(self) -> _z.SourceRegion:
        return _z.SourceRegion(self.source)


@dataclasses.dataclass(frozen=True)
class ReferenceValues:
    """Frozen normalisation. Bound to the configuration that produced it."""

    psi_0: float
    c_0: float
    fingerprint: str
    spec_element_size: float
    flow_residual_relative: float
    thermal_residual_relative: float

    def check(self, config: R1Config, spec: _z.Zhao2DSpec) -> None:
        if self.fingerprint != config.fingerprint():
            raise ValueError(
                "the frozen reference was computed under a different "
                "configuration; recompute it rather than reusing it.\n"
                f"stored: {self.fingerprint}\ncurrent: {config.fingerprint()}"
            )
        if not np.isclose(self.spec_element_size, spec.element_size, rtol=1e-12):
            raise ValueError(
                f"reference was frozen at h={self.spec_element_size}, "
                f"running at h={spec.element_size}"
            )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "ReferenceValues":
        return ReferenceValues(**json.loads(text))


class NotConverged(RuntimeError):
    """A state that must not reach the objective, the gradient or MMA."""


# --------------------------------------------------------------------------
# The differentiable problem
# --------------------------------------------------------------------------


class Zhao2DProblem:
    """Meshes, solvers and the design map, built once and reused every call."""

    def __init__(
        self,
        spec: _z.Zhao2DSpec,
        config: R1Config = R1Config(),
        solver_settings: dict | None = None,
    ):
        self.spec = spec
        self.config = config
        self.settings = solver_settings or _za.default_solver_settings()

        self.flow_mesh = _z.build_mesh(spec, dofs_per_node=3)
        self.thermal_mesh = _z.build_mesh(spec, dofs_per_node=1)
        self.h_elem = _elements.element_lengths(
            self.flow_mesh.mesh, config.element_length_mode
        )

        self.flow_bc = _z.build_flow_bc(self.flow_mesh, spec, config.outlet_kind)
        self.thermal_bc = _z.build_thermal_bc(self.thermal_mesh, spec)
        self.q_source = _z.heat_source_field(
            self.thermal_mesh, spec, config.source_region
        )

        self.flow = _fe_flow.FlowSolver(
            self.flow_mesh.mesh,
            self.flow_bc.bc,
            density=spec.fluid_density,
            viscosity=spec.fluid_viscosity,
            solver_settings=self.settings,
            elem_length=self.h_elem,
            form=config.flow_form,
        )
        self.thermal = _fe_thermal.ThermalSolver(
            self.thermal_mesh.mesh,
            self.thermal_bc,
            b_f=spec.b_f,
            solver_settings=self.settings,
            elem_length=self.h_elem,
            form=config.thermal_form,
        )

        self.flow_x0 = (
            jnp.zeros((self.flow_mesh.mesh.num_dofs,))
            .at[self.flow_bc.bc["fixed_dofs"]]
            .set(self.flow_bc.bc["dirichlet_values"])
        )
        self.thermal_x0 = (
            jnp.zeros((self.thermal_mesh.mesh.num_dofs,))
            .at[self.thermal_bc["fixed_dofs"]]
            .set(self.thermal_bc["dirichlet_values"])
        )

        self.design_elements = np.nonzero(self.flow_mesh.design_mask)[0]
        self.area = jnp.asarray(self.flow_mesh.elem_area)
        self.filter_matrix = _design.build_surface_filter(
            np.asarray(self.flow_mesh.elem_centres)[self.design_elements],
            radius=config.filter_radius_elements * spec.element_size,
            kind="linear",
        )

    @property
    def num_design(self) -> int:
        return len(self.design_elements)

    # -- design map ---------------------------------------------------------

    def solid_fraction(self, x):
        """Design vector -> (num_elems,) solid fraction, tabs pinned to fluid.

        The filter acts only among design elements, so it cannot bleed material
        into the fixed inlet/outlet tabs. The tabs are pinned AFTER the filter
        for the same reason.
        """
        filtered = self.filter_matrix @ jnp.asarray(x)
        projected = _design.heaviside_projection(
            filtered, self.config.projection_beta
        )
        return jnp.zeros(self.flow_mesh.num_elems).at[self.design_elements].set(
            projected
        )

    def fluid_fraction(self, x):
        """v_f on the domain the constraint is measured over."""
        gamma = 1.0 - self.solid_fraction(x)
        if self.config.volume_domain == VolumeDomain.WHOLE:
            return jnp.sum(gamma * self.area) / jnp.sum(self.area)
        mask = jnp.asarray(self.flow_mesh.design_mask)
        return jnp.sum(jnp.where(mask, gamma * self.area, 0.0)) / jnp.sum(
            jnp.where(mask, self.area, 0.0)
        )

    # -- states -------------------------------------------------------------

    def solve_states(self, s, alpha_max: float):
        """(press_vel, temperature, alpha, kappa), differentiable in `s`."""
        material = _za.build_material(self.spec, alpha_max)
        alpha = _materials.brinkman_penalty(s, material)
        kappa = _materials.conductivity(s, material)

        press_vel = _solver.modified_newton_raphson_solve(
            self.flow, self.flow_x0, alpha
        )
        elem_vel = self.flow.element_velocities(press_vel)
        temperature = _solver.modified_newton_raphson_solve(
            self.thermal, self.thermal_x0, elem_vel, kappa, self.q_source
        )
        return press_vel, temperature, alpha, kappa

    def metrics(self, s, alpha_max: float):
        """(Psi, C) as JAX scalars. Zhao equations 24 and 23."""
        press_vel, temperature, alpha, kappa = self.solve_states(s, alpha_max)
        psi = self.flow.dissipated_power(press_vel, alpha)
        c = self.thermal.thermal_compliance(
            temperature, self.flow.element_velocities(press_vel), kappa
        )
        return psi, c

    def residual_norms(self, s, alpha_max: float) -> dict:
        """Recomputed residuals at the converged states. NOT differentiable.

        Upstream's Newton loop prints "converged" unconditionally and stores the
        residual from before its last step, so this is the only honest measure.
        """
        press_vel, temperature, alpha, kappa = self.solve_states(s, alpha_max)
        press_vel = jax.lax.stop_gradient(press_vel)
        temperature = jax.lax.stop_gradient(temperature)
        elem_vel = self.flow.element_velocities(press_vel)
        fr, _ = self.flow.get_residual_and_tangent_stiffness(press_vel, alpha)
        f0, _ = self.flow.get_residual_and_tangent_stiffness(self.flow_x0, alpha)
        tr, _ = self.thermal.get_residual_and_tangent_stiffness(
            temperature, elem_vel, kappa, self.q_source
        )
        t0, _ = self.thermal.get_residual_and_tangent_stiffness(
            self.thermal_x0, elem_vel, kappa, self.q_source
        )
        return {
            "flow": float(jnp.linalg.norm(fr) / jnp.maximum(jnp.linalg.norm(f0), 1e-300)),
            "thermal": float(
                jnp.linalg.norm(tr) / jnp.maximum(jnp.linalg.norm(t0), 1e-300)
            ),
        }

    def require_converged(self, s, alpha_max: float, tol: float = 1e-8) -> dict:
        """Raise unless both states converged. Call before trusting a gradient.

        The implicit function theorem gives dJ/ds from R = 0. At a state that
        did not reach R = 0 the returned gradient is not the sensitivity of
        anything, so an unconverged state must not reach MMA -- it would be
        accepted silently otherwise.
        """
        norms = self.residual_norms(s, alpha_max)
        bad = {k: v for k, v in norms.items() if not (v <= tol)}
        if bad:
            raise NotConverged(
                f"relative residual above {tol:g}: {bad}; "
                "the implicit-function-theorem gradient is not valid here"
            )
        return norms

    # -- objective ----------------------------------------------------------

    def objective_and_constraint(self, x, reference: ReferenceValues, alpha_max: float):
        """(J, g) as JAX scalars, differentiable end to end in `x`.

        g <= 0 is feasible:  g = v_f / v_f_max - 1.
        """
        reference.check(self.config, self.spec)
        s = self.solid_fraction(x)
        psi, c = self.metrics(s, alpha_max)
        w = self.config.weight
        j = w * psi / reference.psi_0 + (1.0 - w) * c / reference.c_0
        g = self.fluid_fraction(x) / self.config.max_fluid_fraction - 1.0
        return j, g


def freeze_reference(
    spec: _z.Zhao2DSpec,
    config: R1Config = R1Config(),
    solver_settings: dict | None = None,
) -> tuple[ReferenceValues, dict]:
    """Compute Psi_0 and C_0 once, on the configured reference field.

    Returns the frozen values plus the raw report. Full precision, bound to the
    configuration fingerprint -- never rebuilt from a rounded printed number.
    """
    problem = Zhao2DProblem(spec, config, solver_settings)
    s = _z.reference_solid_fraction(problem.flow_mesh, spec, config.reference_field)
    norms = problem.residual_norms(s, config.alpha_max_reference)
    psi, c = problem.metrics(s, config.alpha_max_reference)

    values = ReferenceValues(
        psi_0=float(psi),
        c_0=float(c),
        fingerprint=config.fingerprint(),
        spec_element_size=spec.element_size,
        flow_residual_relative=norms["flow"],
        thermal_residual_relative=norms["thermal"],
    )
    report = {
        "psi_0": values.psi_0,
        "c_0": values.c_0,
        **{f"residual_{k}": v for k, v in norms.items()},
        **_z.fluid_fractions(problem.flow_mesh, s),
    }
    return values, report
