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
import pathlib

import numpy as np
import jax
import jax.numpy as jnp

import toflux.src.solver as _solver

from tfopus import design as _design
from tfopus import projection as _projection
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


class Projection:
    """How the filtered design becomes the solid fraction s.

    VOLUME_PRESERVING is Xu, Cai & Cheng (2010), Eqs. (19) and (21): the
    threshold eta is solved every call so that the projected volume equals
    the filtered volume, so beta never moves the volume constraint (see
    tfopus/projection.py). It is the default from R1l on.

    TANH is the fixed-threshold tanh form, eta = 0.5, that R1b to R1k and
    their records used. With it, raising beta moves the volume: R1k's
    terminal design goes from g = -1.5e-4 at beta 8 to +8.0e-3 at beta 16.
    Scripts that reproduce those records pin it.

    Neither is Zhao's: the paper's relaxed Heaviside (its Eq. 9) acts on a CBS
    level-set function, which a density method does not have.
    """

    VOLUME_PRESERVING = "volume_preserving"
    TANH = "tanh"


@dataclasses.dataclass(frozen=True)
class R1Config:
    """Everything that fixes the discrete problem. Stored beside every result.

    `volume_domain` defaults to the design domain. Equation 26 writes
    (1/|Omega|) integral over Omega, but with fluid tabs the whole-domain
    fraction of the reference state is 0.4231, so a whole-domain bound of 0.4
    would put this project's reference state outside the constraint.

    That is a reason for THIS reproduction's convention, not evidence about
    Zhao's: a normalisation reference is allowed to be infeasible, and the
    paper's own 3D reference has v_f = 0.616, well past its bound. Both
    fractions are always reported; this only chooses which one MMA sees.
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
    projection: str = Projection.VOLUME_PRESERVING

    flow_form: _fe_flow.FlowForm = R1_FLOW_FORM
    thermal_form: _fe_thermal.ThermalForm = R1_THERMAL_FORM

    def fingerprint(self) -> str:
        """Stable JSON of every switch. Recorded with a run; does NOT gate reuse.

        Use `reference_identity` for deciding whether a stored Psi_0 / C_0 may
        be reused: this one changes when beta or the filter radius moves, which
        are stages of the optimisation and have no effect on the reference.
        """
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


# Fields of Zhao2DSpec that the reference computation actually depends on:
# the geometry it is solved on, the materials, the loads and boundary values,
# the interpolation exponents and the reference density itself. `reported_*`
# are comparison values that enter nothing, so they are excluded -- binding
# them would invalidate a reference for a change that cannot move it.
_REFERENCE_SPEC_FIELDS = (
    "inlet_half_width",
    "tab_length",
    "design_half_width",
    "design_height",
    "element_size",
    "inlet_speed",
    "heat_source",
    "inlet_temperature",
    "fluid_density",
    "fluid_viscosity",
    "fluid_heat_capacity",
    "fluid_conductivity",
    "solid_conductivity",
    "q_alpha",
    "q_kappa",
    "reference_gamma",
)

# Fields of R1Config that the reference depends on. Deliberately absent:
# filter_radius_elements, projection_beta, volume_domain, max_fluid_fraction
# and weight. The reference is a directly specified uniform physical density --
# `freeze_reference` never routes it through the filter or the projection --
# and Psi_0 / C_0 are states, not objective weights.
_REFERENCE_CONFIG_FIELDS = (
    "reference",
    "outlet",
    "source",
    "alpha_max_reference",
    "element_length_mode",
    "flow_form",
    "thermal_form",
)


def reference_identity(spec: _z.Zhao2DSpec, config: R1Config) -> str:
    """Canonical JSON of everything Psi_0 and C_0 depend on, and nothing else.

    Two separate identities are needed, and conflating them fails both ways.
    Binding the whole `R1Config` rejects a legitimate projection step, which is
    a stage of the optimisation the reference is deliberately independent of;
    binding only `R1Config` silently accepts a changed heat source, inlet speed
    or geometry, which change the reference problem entirely.
    """
    payload = {
        "spec": {f: getattr(spec, f) for f in _REFERENCE_SPEC_FIELDS},
        "config": {
            f: (
                dataclasses.asdict(getattr(config, f))
                if dataclasses.is_dataclass(getattr(config, f))
                else getattr(config, f)
            )
            for f in _REFERENCE_CONFIG_FIELDS
        },
    }
    return json.dumps(payload, sort_keys=True)


@dataclasses.dataclass(frozen=True)
class ReferenceValues:
    """Frozen normalisation, bound to the reference problem that produced it.

    `run_fingerprint` records the full configuration of the freezing run for
    provenance; only `identity` decides whether the values may be reused.
    """

    psi_0: float
    c_0: float
    identity: str
    run_fingerprint: str
    spec_element_size: float
    flow_residual_relative: float
    thermal_residual_relative: float

    def check(self, config: R1Config, spec: _z.Zhao2DSpec) -> None:
        """Raise unless this reference belongs to the problem being run.

        Changing beta, the filter radius or the volume-constraint domain passes:
        those are optimisation stages, and the denominators are held fixed
        across them on purpose.
        """
        current = reference_identity(spec, config)
        if self.identity != current:
            stored = json.loads(self.identity)
            now = json.loads(current)
            diffs = [
                f"{group}.{k}: {stored[group][k]!r} -> {now[group][k]!r}"
                for group in ("spec", "config")
                for k in stored[group]
                if stored[group][k] != now[group][k]
            ]
            raise ValueError(
                "the frozen reference belongs to a different reference "
                "problem; recompute it rather than reusing it:\n  "
                + "\n  ".join(diffs or ["(fields differ in shape)"])
            )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "ReferenceValues":
        return ReferenceValues(**json.loads(text))


class NotConverged(RuntimeError):
    """A state that must not reach the objective, the gradient or MMA."""


class DegenerateProjection(RuntimeError):
    """A design at which the volume-preserving projection's root is degenerate.

    With no filtered density strictly between 0 and 1, Eq. (21) holds for
    every eta: eta is not unique, and its implicit derivative, which the
    gradient is built on, does not apply. That is about eta, not a claim that
    s has no derivative (a lone element is mapped to itself, ds/drho = 1).
    The value is still defined -- s equals the filtered design -- so
    value-only evaluations proceed; a gradient does not reach MMA.
    """


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

    def solid_fraction(self, x, beta: float | None = None):
        """Design vector -> (num_elems,) solid fraction, tabs pinned to fluid.

        The filter acts only among design elements, so it cannot bleed material
        into the fixed inlet/outlet tabs. The tabs are pinned AFTER the filter
        for the same reason.

        `beta` overrides the configured projection sharpness. It is a call
        argument rather than a rebuilt problem because it is the one stage
        parameter that touches nothing else: the mesh, the filter, the solvers
        and the boundary conditions are all independent of it, so a
        continuation step must not pay for rebuilding them.

        Which projection is `config.projection` (see `Projection`). With the
        default, volume-preserving one, the design-domain volume of s equals
        that of the filtered design for every beta.
        """
        projected, _ = self._project(x, beta)
        return jnp.zeros(self.flow_mesh.num_elems).at[self.design_elements].set(
            projected
        )

    def projection_threshold(self, x, beta: float | None = None) -> float:
        """The eta the projection used for this design: 0.5 for TANH; for
        VOLUME_PRESERVING the root of Xu et al.'s Eq. (21), NaN at beta = 0."""
        return float(self._project(x, beta)[1])

    def projection_root(self, x, beta: float | None = None) -> dict:
        """Whether the implicit derivative the gradient is built on applies here.

        Only the volume-preserving projection at beta > 0 has a root to worry
        about: it needs some filtered density strictly between 0 and 1, or
        eta is not unique and its implicit derivative does not apply. TANH, and
        beta = 0, are always differentiable.
        """
        b = self.config.projection_beta if beta is None else beta
        if self.config.projection != Projection.VOLUME_PRESERVING or b <= 0.0:
            return {"eta": self.projection_threshold(x, beta), "slope": None,
                    "nondegenerate": True}
        filtered = self.filter_matrix @ jnp.asarray(x)
        volumes = self.area[self.design_elements]
        eta = _projection.volume_preserving_eta(filtered, volumes, b)
        slope = _projection.root_slope(filtered, volumes, b, eta)
        return {"eta": float(eta), "slope": slope,
                "nondegenerate": _projection.root_is_nondegenerate(filtered, volumes, b, eta)}

    def _project(self, x, beta):
        filtered = self.filter_matrix @ jnp.asarray(x)
        b = self.config.projection_beta if beta is None else beta
        if self.config.projection == Projection.VOLUME_PRESERVING:
            # Eq. (21)'s volumes: the design-domain elements the projection acts on
            return _projection.volume_preserving_projection(
                filtered, self.area[self.design_elements], b)
        if self.config.projection == Projection.TANH:
            return _design.heaviside_projection(filtered, b), jnp.asarray(0.5)
        raise ValueError(f"unknown projection {self.config.projection!r}")

    def fluid_fraction(self, x, beta: float | None = None):
        """v_f on the domain the constraint is measured over."""
        gamma = 1.0 - self.solid_fraction(x, beta)
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

    def thermal_velocity(self, press_vel):
        """The velocity the thermal solve sees, in the thermal solver's layout.

        Here the flow's own element velocities; the dual-mesh subclass maps the
        flow onto its thermal mesh instead. Everything that forms C goes through
        this one method, so no caller can hand the thermal side the wrong mesh's
        velocity.
        """
        return self.flow.element_velocities(press_vel)

    def metrics(self, s, alpha_max: float):
        """(Psi, C) as JAX scalars. Zhao equations 24 and 23."""
        press_vel, temperature, alpha, kappa = self.solve_states(s, alpha_max)
        psi = self.flow.dissipated_power(press_vel, alpha)
        c = self.thermal.thermal_compliance(
            temperature, self.thermal_velocity(press_vel), kappa
        )
        return psi, c

    def residual_norms(self, s, alpha_max: float) -> dict:
        """Recomputed residuals at the converged states. NOT differentiable.

        Upstream's Newton loop prints "converged" unconditionally and stores the
        residual from before its last step, so this is the only honest measure.
        """
        press_vel, temperature, alpha, kappa = self.solve_states(s, alpha_max)
        return self.residual_norms_at(press_vel, temperature, alpha, kappa)

    def residual_norms_at(self, press_vel, temperature, alpha, kappa) -> dict:
        """The same measure for states already in hand, without re-solving.

        What a gate needs when the states it judges are the ones a caller is
        about to use: a second solve would gate a different computation.
        """
        press_vel = jax.lax.stop_gradient(press_vel)
        temperature = jax.lax.stop_gradient(temperature)
        vel_t = self.thermal_velocity(press_vel)
        fr, _ = self.flow.get_residual_and_tangent_stiffness(press_vel, alpha)
        f0, _ = self.flow.get_residual_and_tangent_stiffness(self.flow_x0, alpha)
        tr, _ = self.thermal.get_residual_and_tangent_stiffness(
            temperature, vel_t, kappa, self.q_source
        )
        t0, _ = self.thermal.get_residual_and_tangent_stiffness(
            self.thermal_x0, vel_t, kappa, self.q_source
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

    def check_reference(self, reference: ReferenceValues) -> None:
        """Raise unless `reference` was frozen for THIS model.

        Subclasses whose model is more than the spec and the R1 config -- the
        dual-mesh one adds a thermal mesh -- override it, so a caller such as
        the driver checks the right identity without knowing which model it has.
        """
        reference.check(self.config, self.spec)

    def objective_and_constraint(self, x, reference: ReferenceValues, alpha_max: float):
        """(J, g) as JAX scalars, differentiable end to end in `x`.

        g <= 0 is feasible:  g = v_f / v_f_max - 1.
        """
        self.check_reference(reference)
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
    # Gate, not merely measure: a reference frozen from an unconverged state
    # would silently become the denominator of every result that follows.
    norms = problem.require_converged(s, config.alpha_max_reference)
    psi, c = problem.metrics(s, config.alpha_max_reference)

    values = ReferenceValues(
        psi_0=float(psi),
        c_0=float(c),
        identity=reference_identity(spec, config),
        run_fingerprint=config.fingerprint(),
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


REFERENCE_FILE = pathlib.Path(__file__).resolve().parent / "zhao2d_reference.json"


def load_reference(
    spec: _z.Zhao2DSpec,
    config: R1Config = R1Config(),
    path: pathlib.Path | None = None,
) -> ReferenceValues:
    """Read the frozen reference from disk and check it belongs here."""
    path = path or REFERENCE_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run scripts/zhao2d_freeze_reference.py --write"
        )
    values = ReferenceValues.from_json(path.read_text(encoding="utf-8"))
    values.check(config, spec)
    return values


def verify_reference_against_state(
    problem: "Zhao2DProblem", reference: ReferenceValues, rtol: float = 1e-9
) -> dict:
    """Re-solve the physical reference state and check BOTH ratios are 1.

    Separately, not through J. J = w + (1-w) = 1 whenever the two errors happen
    to cancel, and with w = 0.5 an equal and opposite pair does exactly that,
    so J alone cannot detect a swapped or stale pair of denominators.

    The state re-solved here is the PHYSICAL reference density, never the design
    vector: with beta > 0 the projection moves x = 0.6 to P_beta(0.6) != 0.6, so
    requiring J(x = 0.6) = 1 after a projection step would be wrong.
    """
    s = _z.reference_solid_fraction(
        problem.flow_mesh, problem.spec, problem.config.reference_field
    )
    problem.require_converged(s, problem.config.alpha_max_reference)
    psi, c = problem.metrics(s, problem.config.alpha_max_reference)
    ratios = {
        "psi_over_psi_0": float(psi) / reference.psi_0,
        "c_over_c_0": float(c) / reference.c_0,
    }
    bad = {k: v for k, v in ratios.items() if not np.isclose(v, 1.0, rtol=rtol)}
    if bad:
        raise ValueError(
            f"the loaded reference does not reproduce its own state: {bad}"
        )
    return ratios
