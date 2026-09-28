"""R1q: the check layer as a differentiable model of the coarse design.

The candidates are ranked on the check layer D: flow on h/2, temperature on
h/8, the flow and thermal sides exactly R1m's. This module makes D a model of
the SAME design variables as the development model:

    x (5000 raw values on h) -> filter, projection -> s_D     on the design mesh h
    s_F = E_DF s_D            each design cell copied to its r_F^2 flow cells
    R_F(U_F, s_F) = 0         flow on h / r_F
    s_T = E_FT E_DF s_D       the same density on the thermal mesh
    u_T = P_FT u_F            the flow's exact Q1 prolongation
    R_T(T, u_T, s_T) = 0      temperature on h / (r_F r_T)
    Psi on F, C on T, the volume constraint on the design mesh

The design side is inherited unchanged from `Zhao2DProblem` built on h: the
design elements, their areas, the filter at its physical radius
(filter_radius_elements x h) and the projection, eta solved on the coarse
design volume. Building the existing problem on h/2 instead would make the
flow cells the design variables -- four times as many -- and halve the filter's
physical radius, changing the design space and the regularisation as well.

The whole chain is traced, so reverse mode sends density sensitivities back
through E^T (a gather's transpose: children are summed, not averaged), and the
flow, the materials, tau and eta all keep their dependence on the design. No
saved state enters the gradient; reusing saved states is for checking given
designs.

This model has no frozen reference of its own. It is evaluated on the
development model's Psi_0 and C_0 as a declared common scale (`CommonScale`),
which records that reference's identity -- checked, when it is made, to be the
development model's -- and this model's. `check_reference` stays strict: an
ordinary reference is accepted only if it was frozen for THIS model.

`run` (from R1r) optimises on this model: `zhao2d_driver.run_loop` with
`evaluate` on the declared common scale, over the same coarse design vector.
Nothing is relabelled for it, and the old driver entry is unchanged.

`thermal_path="linear"` (from R1s) solves the temperature, for a given flow
and density, in one linear solve (`tfopus.affine_solve`) instead of upstream's
Newton loop. The residual, the gate and the implicit derivative are the same.

`migrate_legacy_checkpoint` (from R1v) re-signs a checkpoint made before
c31b3c2, whose binding did not yet name the optimisation problem, with today's
binding; its state is carried over untouched.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import time

import numpy as np
import jax
import jax.numpy as jnp

import toflux.src.solver as _solver

from tfopus import affine_solve as _affine
from tfopus import elements as _elements
from tfopus import fe_flow as _fe_flow
from tfopus import fe_thermal as _fe_thermal
from tfopus import materials as _materials
from tfopus import zhao2d as _z
from tfopus import zhao2d_analysis as _za
from tfopus import zhao2d_driver as _driver
from tfopus import zhao2d_dual as _dual
from tfopus import zhao2d_r1 as _r1
from tfopus import zhao2d_refine as _refine


THERMAL_PATHS = ("newton", "linear")


class Zhao2DFineFlowProblem(_r1.Zhao2DProblem):
    """The design on h; the flow on h / r_F; the temperature nested in the flow mesh.

    `flow_mesh` is the flow's mesh, as everywhere else in this code;
    `design_mesh` is the mesh the design variables, the filter and the volume
    constraint live on. With `flow_refinement` 1 the two coincide and the model
    is `Zhao2DDualProblem` with the same thermal refinement.

    `thermal_path` is how the temperature is solved for a given flow and
    density. The default, "newton", is upstream's modified Newton loop, which
    every run up to R1r used. "linear" is `affine_solve`: one linear solve, the
    same residual, the same implicit derivative (R1s). It is a solver choice,
    not a model change, so it is not part of the identity; records name it.
    """

    def __init__(
        self,
        spec: _z.Zhao2DSpec,
        config: _r1.R1Config = _r1.R1Config(),
        flow_refinement: int = 2,
        thermal_refinement: int = 4,
        thermal_quadrature: int = 3,
        solver_settings: dict | None = None,
        thermal_path: str = "newton",
    ):
        if thermal_path not in THERMAL_PATHS:
            raise ValueError(f"thermal_path must be one of {THERMAL_PATHS}, got {thermal_path!r}")
        # The design side: everything a Zhao2DProblem on h builds for it.
        super().__init__(spec, config, solver_settings)
        self.design_mesh = self.flow_mesh
        self.thermal_path = thermal_path
        self.flow_refinement = int(flow_refinement)
        self.thermal_refinement = int(thermal_refinement)
        self.thermal_quadrature = int(thermal_quadrature)
        if self.flow_refinement < 1 or self.flow_refinement != flow_refinement:
            raise ValueError(f"flow_refinement must be a positive integer, got {flow_refinement}")

        # The flow on h / r_F, built as a Zhao2DProblem on that spec builds it --
        # the same mesh, element lengths, boundary conditions and solver as R1m's.
        self.flow_spec = _refine.refine_spec(spec, self.flow_refinement)
        self.flow_mesh = _z.build_mesh(self.flow_spec, dofs_per_node=3)
        self.h_elem = _elements.element_lengths(self.flow_mesh.mesh, config.element_length_mode)
        self.flow_bc = _z.build_flow_bc(self.flow_mesh, self.flow_spec, config.outlet_kind)
        self.flow = _fe_flow.FlowSolver(
            self.flow_mesh.mesh,
            self.flow_bc.bc,
            density=spec.fluid_density,
            viscosity=spec.fluid_viscosity,
            solver_settings=self.settings,
            elem_length=self.h_elem,
            form=config.flow_form,
        )
        self.flow_x0 = (
            jnp.zeros((self.flow_mesh.mesh.num_dofs,))
            .at[self.flow_bc.bc["fixed_dofs"]]
            .set(self.flow_bc.bc["dirichlet_values"])
        )
        self.design_to_flow = _dual.build_nested_maps(
            self.design_mesh, self.flow_mesh, self.flow_refinement)

        # The temperature, nested in the flow mesh as in Zhao2DDualProblem.
        self.thermal_spec = _refine.refine_spec(self.flow_spec, self.thermal_refinement)
        self.thermal_mesh = _z.build_mesh(
            self.thermal_spec, dofs_per_node=1, gauss_order=self.thermal_quadrature)
        self.maps = _dual.build_nested_maps(self.flow_mesh, self.thermal_mesh,
                                            self.thermal_refinement)
        self.thermal_bc = _z.build_thermal_bc(self.thermal_mesh, self.flow_spec)
        self.q_source = _z.heat_source_field(self.thermal_mesh, self.flow_spec,
                                             config.source_region)
        self.h_thermal = _elements.element_lengths(self.thermal_mesh.mesh,
                                                   config.element_length_mode)
        self.thermal = _fe_thermal.ThermalSolver(
            self.thermal_mesh.mesh,
            self.thermal_bc,
            b_f=spec.b_f,
            solver_settings=self.settings,
            elem_length=self.h_thermal,
            form=config.thermal_form,
        )
        self.thermal_x0 = (
            jnp.zeros((self.thermal_mesh.mesh.num_dofs,))
            .at[self.thermal_bc["fixed_dofs"]]
            .set(self.thermal_bc["dirichlet_values"])
        )
        self.nesting = self._check_nesting()

    def _check_nesting(self) -> dict:
        """What must be invariant from the design mesh down; raises otherwise."""
        design, flow, thermal = self.design_mesh, self.flow_mesh, self.thermal_mesh
        p_df, p_ft = self.design_to_flow.parents, self.maps.parents
        area = {name: np.asarray(m.elem_area) for name, m in
                (("design", design), ("flow", flow), ("thermal", thermal))}
        if not np.allclose(np.bincount(p_df, weights=area["flow"], minlength=design.num_elems),
                           area["design"], rtol=1e-12):
            raise RuntimeError("flow cells do not tile their design cells")
        if not np.allclose(np.bincount(p_ft, weights=area["thermal"], minlength=flow.num_elems),
                           area["flow"], rtol=1e-12):
            raise RuntimeError("thermal cells do not tile their flow cells")
        if not (np.array_equal(flow.design_mask, design.design_mask[p_df])
                and np.array_equal(thermal.design_mask, flow.design_mask[p_ft])):
            raise RuntimeError("the design/tab partition is not preserved")
        q = np.asarray(_z.heat_source_field(design, self.spec, self.config.source_region))
        heat_design = float(np.sum(q * area["design"]))
        heat_thermal = float(np.sum(np.asarray(self.q_source) * area["thermal"]))
        if not np.isclose(heat_design, heat_thermal, rtol=1e-12):
            raise RuntimeError(f"total heat source changed: {heat_design} -> {heat_thermal}")
        return {
            "area_design_mesh": float(area["design"].sum()),
            "area_flow_mesh": float(area["flow"].sum()),
            "area_thermal_mesh": float(area["thermal"].sum()),
            "heat_source_design_mesh": heat_design,
            "heat_source_thermal_mesh": heat_thermal,
            "design_cells": int(design.design_mask.sum()),
            "design_cells_flow_mesh": int(flow.design_mask.sum()),
            "design_cells_thermal_mesh": int(thermal.design_mask.sum()),
        }

    # -- the design map, on the design mesh -----------------------------------

    def solid_fraction(self, x, beta: float | None = None):
        """Design vector -> solid fraction on the DESIGN mesh, tabs pinned to fluid."""
        projected, _ = self._project(x, beta)
        return jnp.zeros(self.design_mesh.num_elems).at[self.design_elements].set(projected)

    def fluid_fraction(self, x, beta: float | None = None):
        """v_f over the constraint's domain, on the design mesh."""
        gamma = 1.0 - self.solid_fraction(x, beta)
        if self.config.volume_domain == _r1.VolumeDomain.WHOLE:
            return jnp.sum(gamma * self.area) / jnp.sum(self.area)
        mask = jnp.asarray(self.design_mesh.design_mask)
        return jnp.sum(jnp.where(mask, gamma * self.area, 0.0)) / jnp.sum(
            jnp.where(mask, self.area, 0.0))

    # -- the maps, applied ----------------------------------------------------

    def flow_density(self, s):
        """s_F = E_DF s_D: the design mesh's density on the flow mesh."""
        return self.design_to_flow.density(s)

    def thermal_density(self, s):
        """s_T = E_FT E_DF s_D: the design mesh's density on the thermal mesh."""
        return self.maps.density(self.design_to_flow.density(s))

    def thermal_nodal_velocity(self, press_vel):
        """(n_thermal_nodes, dim): u_T = P_FT u_F."""
        u_f = jnp.asarray(press_vel).reshape(-1, self.flow.num_fields)[:, 1:]
        return self.maps.velocity(u_f)

    def thermal_velocity(self, press_vel):
        """(n_thermal_elems, nodes_per_elem * dim), the thermal solver's layout."""
        u_t = self.thermal_nodal_velocity(press_vel)
        nodes = jnp.asarray(self.thermal_mesh.mesh.elem_nodes)
        return u_t[nodes].reshape(self.thermal_mesh.num_elems, -1)

    def flow_material(self, s, alpha_max: float):
        """alpha(E_DF s) on the flow mesh."""
        return _materials.brinkman_penalty(self.flow_density(s),
                                           _za.build_material(self.spec, alpha_max))

    def thermal_conductivity(self, s, alpha_max: float):
        """kappa(E_DT s) on the thermal mesh."""
        return _materials.conductivity(self.thermal_density(s),
                                       _za.build_material(self.spec, alpha_max))

    # -- states ------------------------------------------------------------------

    def _thermal_state(self, elem_velocity, kappa_t):
        """T for the thermal mesh's velocity and conductivity, by `thermal_path`."""
        solve = (_solver.modified_newton_raphson_solve if self.thermal_path == "newton"
                 else _affine.affine_solve)
        return solve(self.thermal, self.thermal_x0, elem_velocity, kappa_t, self.q_source)

    def solve_thermal(self, press_vel, s, alpha_max: float):
        """T on the thermal mesh for a GIVEN flow state and design-mesh density."""
        return self._thermal_state(self.thermal_velocity(press_vel),
                                   self.thermal_conductivity(s, alpha_max))

    def solve_states(self, s, alpha_max: float):
        """(press_vel, temperature, alpha_F, kappa_T) for the design-mesh density s.

        alpha is on the flow mesh and kappa on the thermal mesh, as the
        inherited `metrics` and `residual_norms_at` expect.
        """
        alpha = self.flow_material(s, alpha_max)
        press_vel = _solver.modified_newton_raphson_solve(self.flow, self.flow_x0, alpha)
        kappa_t = self.thermal_conductivity(s, alpha_max)
        temperature = self._thermal_state(self.thermal_velocity(press_vel), kappa_t)
        return press_vel, temperature, alpha, kappa_t

    # -- identity -------------------------------------------------------------------

    def model_identity(self) -> str:
        """The single-mesh identity plus the flow and the thermal discretisation."""
        return fine_flow_identity(self.spec, self.config, self.flow_refinement,
                                  self.thermal_refinement, self.thermal_quadrature)

    def reference_identity(self) -> str:
        """What a reference frozen for THIS model would carry."""
        return self.model_identity()

    def check_reference(self, reference: _r1.ReferenceValues) -> None:
        """Refuse any reference not frozen for THIS model -- the development one included."""
        if isinstance(reference, CommonScale):
            raise TypeError(
                "a CommonScale is not a reference of this model; optimise on it with "
                "zhao2d_fineflow.run, which checks it with check_common_scale")
        if reference.identity != self.reference_identity():
            raise ValueError(
                "this reference was not frozen for the fine-flow model (flow refinement "
                f"{self.flow_refinement}, thermal refinement {self.thermal_refinement}, "
                f"quadrature {self.thermal_quadrature}); to evaluate it on another model's "
                "constants, make a CommonScale, which says so")

    def check_common_scale(self, scale: "CommonScale") -> None:
        """Raise unless `scale` was made for THIS model from the model it names."""
        if scale.target_identity != self.model_identity():
            raise ValueError("this common scale was made for another model")
        expected = development_identity(self.spec, self.config,
                                        scale.source_model["thermal_refinement"],
                                        scale.source_model["thermal_quadrature"])
        if scale.source_identity != expected:
            raise ValueError("the common scale's source is not the development model it names")


# --------------------------------------------------------------------------
# The common scale
# --------------------------------------------------------------------------


def fine_flow_identity(spec: _z.Zhao2DSpec, config: _r1.R1Config, flow_refinement: int,
                       thermal_refinement: int, thermal_quadrature: int) -> str:
    """`Zhao2DFineFlowProblem.model_identity()` of that model, without building it."""
    payload = json.loads(_r1.reference_identity(spec, config))
    payload["flow_mesh"] = {"refinement": int(flow_refinement)}
    payload["thermal_mesh"] = {
        "refinement": int(thermal_refinement),
        "quadrature": int(thermal_quadrature),
        "element_length_mode": config.element_length_mode,
    }
    return json.dumps(payload, sort_keys=True)


def development_identity(spec: _z.Zhao2DSpec, config: _r1.R1Config,
                         thermal_refinement: int, thermal_quadrature: int) -> str:
    """`Zhao2DDualProblem.reference_identity()` of that model, without building it."""
    payload = json.loads(_r1.reference_identity(spec, config))
    payload["thermal_mesh"] = {
        "refinement": int(thermal_refinement),
        "quadrature": int(thermal_quadrature),
        "element_length_mode": config.element_length_mode,
    }
    return json.dumps(payload, sort_keys=True)


@dataclasses.dataclass(frozen=True)
class CommonScale:
    """Another model's frozen Psi_0 and C_0, used as a stated common scale.

    Not the normalisation of the model it is used on, and not passed off as one:
    it keeps the source reference's own identity, checked to be the development
    model it names, beside the identity of the model it was made for.
    """

    psi_0: float
    c_0: float
    source_file: str
    source_sha256: str
    source_identity: str
    source_model: dict
    target_identity: str


def common_scale(problem: Zhao2DFineFlowProblem, path: pathlib.Path,
                 thermal_refinement: int = 4, thermal_quadrature: int = 3) -> CommonScale:
    """The development model's reference, as a common scale for `problem`.

    Refuses a file whose identity is not the development model (flow h,
    thermal h / thermal_refinement) of `problem`'s spec and configuration.
    """
    path = pathlib.Path(path)
    values = _r1.ReferenceValues.from_json(path.read_text(encoding="utf-8"))
    expected = development_identity(problem.spec, problem.config, thermal_refinement,
                                    thermal_quadrature)
    if values.identity != expected:
        raise ValueError(f"{path} is not the development model's reference (thermal "
                         f"refinement {thermal_refinement}, quadrature {thermal_quadrature})")
    return CommonScale(
        psi_0=values.psi_0,
        c_0=values.c_0,
        source_file=path.as_posix(),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_identity=values.identity,
        source_model={"kind": "development: flow h, thermal nested",
                      "thermal_refinement": int(thermal_refinement),
                      "thermal_quadrature": int(thermal_quadrature)},
        target_identity=problem.model_identity(),
    )


# --------------------------------------------------------------------------
# The evaluation entry
# --------------------------------------------------------------------------


def evaluate(problem: Zhao2DFineFlowProblem, scale: CommonScale, x, alpha_max: float,
             beta: float, gradient: bool = True, tol: float = 1e-8):
    """J, Psi, C, g, their gradients and the states, from ONE traced forward solve.

    J = w Psi / Psi_0 + (1 - w) C / C_0 on the common scale. With `gradient`,
    one forward pass is differentiated twice in reverse mode, for Psi and for
    C, and dJ is formed from the two; dg is the constraint's own gradient. The
    scale is checked against this model before anything is solved; a
    degenerate projection root refuses a gradient; the residual gate is applied
    to the states the solve returned.

    Returns (record, (s_D, press_vel, temperature), grads), grads None without
    `gradient`, else {"J", "psi", "c", "g"} as numpy arrays.
    """
    problem.check_common_scale(scale)
    config = problem.config
    w = config.weight
    root = problem.projection_root(x, beta)
    if gradient and not root["nondegenerate"]:
        raise _r1.DegenerateProjection(
            f"the projection's root is degenerate at this design (beta {beta:g}); "
            "no gradient")

    def metrics(v):
        s = problem.solid_fraction(v, beta)
        press_vel, temperature, alpha, kappa = problem.solve_states(s, alpha_max)
        psi = problem.flow.dissipated_power(press_vel, alpha)
        c = problem.thermal.thermal_compliance(
            temperature, problem.thermal_velocity(press_vel), kappa)
        return (psi, c), (s, press_vel, temperature, alpha, kappa)

    def constraint(v):
        return problem.fluid_fraction(v, beta) / config.max_fluid_fraction - 1.0

    x = jnp.asarray(x)
    timing, t0 = {}, time.perf_counter()
    if gradient:
        (psi, c), vjp, aux = jax.vjp(metrics, x, has_aux=True)
        jax.block_until_ready((psi, c, aux))
        timing["forward"], t0 = time.perf_counter() - t0, time.perf_counter()
        d_psi = jax.block_until_ready(vjp((jnp.ones_like(psi), jnp.zeros_like(c)))[0])
        timing["reverse_psi"], t0 = time.perf_counter() - t0, time.perf_counter()
        d_c = jax.block_until_ready(vjp((jnp.zeros_like(psi), jnp.ones_like(c)))[0])
        timing["reverse_c"] = time.perf_counter() - t0
        g, d_g = jax.value_and_grad(constraint)(x)
        d_j = w * d_psi / scale.psi_0 + (1.0 - w) * d_c / scale.c_0
        grads = {"J": np.asarray(d_j), "psi": np.asarray(d_psi), "c": np.asarray(d_c),
                 "g": np.asarray(d_g)}
    else:
        (psi, c), aux = metrics(x)
        jax.block_until_ready((psi, c, aux))
        timing["forward"] = time.perf_counter() - t0
        g, grads = constraint(x), None
    s, press_vel, temperature, alpha, kappa = aux

    norms = problem.residual_norms_at(press_vel, temperature, alpha, kappa)
    bad = {k: v for k, v in norms.items() if not v <= tol}
    if bad:
        raise _r1.NotConverged(
            f"relative residual above {tol:g}: {bad}; this value and gradient are not used")

    psi, c, g = float(psi), float(c), float(g)
    record = {
        "alpha_max": float(alpha_max),
        "beta": float(beta),
        "J_common_scale": w * psi / scale.psi_0 + (1.0 - w) * c / scale.c_0,
        "psi": psi,
        "compliance": c,
        "constraint_g": g,
        **_z.fluid_fractions(problem.design_mesh, s),
        "grey_fraction": float(jnp.mean((s > 0.05) & (s < 0.95))),
        "flow_residual_relative": norms["flow"],
        "thermal_residual_relative": norms["thermal"],
        "projection": config.projection,
        "projection_eta": float(root["eta"]) if np.isfinite(root["eta"]) else None,
        "projection_root_slope": root["slope"],
        "scale": {"psi_0": scale.psi_0, "c_0": scale.c_0, "source_file": scale.source_file,
                  "source_sha256": scale.source_sha256},
        "thermal_path": problem.thermal_path,
        "timing_s": timing,
    }
    state = (np.asarray(s), np.asarray(press_vel), np.asarray(temperature))
    return record, state, grads


# --------------------------------------------------------------------------
# The optimisation entry
# --------------------------------------------------------------------------


def run(problem: Zhao2DFineFlowProblem, scale: CommonScale, phases: list, move_limit: float = 0.1,
        budget: int | None = None, on_iteration=None, initial_design=None, resume=None,
        stop_after: int | None = None):
    """The driver's MMA loop on this model, J on the declared common scale.

    The scale is checked against this model before anything is solved, and so
    is `initial_design`: the raw design variables, `problem.num_design` of them
    on the design mesh. Every iterate is one `evaluate` -- one forward solve,
    its two reverse passes, the gate on that solve's states -- and MMA is given
    J_common_scale with its gradient, and g with its own. The loop, its
    terminal pairing and its stop reasons are `zhao2d_driver.run_loop`'s.

    MMA's state after the last update comes back as `RunResult.mma`, bound to
    `run_binding(problem, scale)` and the run's MMA parameters, schedule and
    budget. `resume` continues from such a checkpoint, and `stop_after` pauses
    a run without a terminal evaluation; see `zhao2d_driver.run_loop`.

    Returns a `zhao2d_driver.RunResult`; its records carry J_common_scale, and
    no J_self.
    """
    problem.check_common_scale(scale)

    def evaluator(x, alpha_max, beta, gradient=True):
        record, state, grads = evaluate(problem, scale, x, alpha_max, beta, gradient)
        if grads is None:
            return record, state, record["J_common_scale"], None, None
        return record, state, record["J_common_scale"], grads["J"], grads["g"]

    return _driver.run_loop(problem, evaluator, phases, move_limit, budget, on_iteration,
                            initial_design, binding=run_binding(problem, scale),
                            resume=resume, stop_after=stop_after)


def run_binding(problem: Zhao2DFineFlowProblem, scale: CommonScale) -> str:
    """What this entry's MMA state belongs to: the model, the optimisation problem, the scale.

    `model_identity` is the reference's identity, which deliberately leaves out
    the objective's weight, the volume constraint, the projection and the
    filter: a reference is a density given directly. An MMA state is not: its
    history and asymptotes belong to one objective, one constraint and one map
    from x to s. So the binding adds the optimisation problem's own identity --
    the whole configuration and spec, the weight, the volume domain and bound,
    the projection, the physical filter radius and the design elements -- and
    leaves `model_identity`, and every reference and scale made with it, alone.
    """
    return json.dumps({
        "model": json.loads(problem.model_identity()),
        "thermal_path": problem.thermal_path,
        "scale": {"psi_0": scale.psi_0, "c_0": scale.c_0, "source_sha256": scale.source_sha256,
                  "source_identity": json.loads(scale.source_identity)},
        "optimisation": optimisation_identity(problem),
    }, sort_keys=True)


def optimisation_identity(problem: _r1.Zhao2DProblem) -> dict:
    """The optimisation problem a run's MMA state belongs to, as `run_binding` names it.

    Read from the design side only: the configuration, the spec and the design
    elements. A `Zhao2DFineFlowProblem` builds its design side exactly as a
    `_r1.Zhao2DProblem` on h does, so the block can be formed from one of those
    without building the fine meshes (R1v's migration does).
    """
    config, spec = problem.config, problem.spec
    design_elements = np.ascontiguousarray(np.asarray(problem.design_elements, dtype=np.int64))
    return {
        "weight": config.weight,
        "volume_domain": config.volume_domain,
        "max_fluid_fraction": config.max_fluid_fraction,
        "projection": config.projection,
        "filter": {"kind": "linear", "radius_elements": config.filter_radius_elements,
                   "radius_m": config.filter_radius_elements * spec.element_size},
        "design_map": {"num_design": problem.num_design,
                       "design_elements_sha256": hashlib.sha256(
                           design_elements.tobytes()).hexdigest()},
        "config": json.loads(config.fingerprint()),
        "spec": dataclasses.asdict(spec),
    }


# The entry `run_binding` signed with from R1t until c31b3c2: no optimisation block.
LEGACY_ENTRY_KEYS = ("model", "scale", "thermal_path")


def migrate_legacy_checkpoint(checkpoint: "_driver.MMACheckpoint",
                              design_problem: _r1.Zhao2DProblem,
                              config_fingerprint: str) -> "_driver.MMACheckpoint":
    """A checkpoint signed before its binding named the optimisation problem, signed with today's.

    From R1t until c31b3c2, `run_binding` named the model, the thermal path and
    the scale only, so a checkpoint signed then -- of those saved in results/,
    R1t's is the only one -- is refused by every run today. This re-signs such
    a checkpoint, and nothing else: the state, the KKT residual, the updates,
    MMA's parameters, the schedule and the budget are carried over untouched,
    and the entry only gains the optimisation block `run_binding` forms today.

    The block is formed from `design_problem`: the design side of the model the
    checkpoint was made on, a `_r1.Zhao2DProblem` on h -- the constructor a
    `Zhao2DFineFlowProblem` runs for its own design side -- which the caller
    builds from the run's own record and the source it was run with, not from
    today's defaults. Refused unless:

    * the binding is a legacy one: its entry names the model, the scale and the
      thermal path, and nothing else;
    * `design_problem`'s configuration has the fingerprint the run recorded,
      `config_fingerprint`, exactly;
    * its spec and configuration give, through today's identity code, the
      model identity the binding names -- at the binding's own refinements --
      and the identity of the scale's source;
    * it has the checkpoint's number of design variables.

    The spec fields outside the model identity (the paper's reported values,
    the alpha_max schedule's constants, the spec's own weight and volume
    bound) cannot be checked against a legacy binding, which never named
    them; the caller shows they are the run's.
    """
    old = json.loads(checkpoint.binding)
    entry = old.get("entry")
    if not isinstance(entry, dict) or sorted(entry) != sorted(LEGACY_ENTRY_KEYS):
        raise ValueError(
            "not a legacy binding: its entry names "
            f"{sorted(entry) if isinstance(entry, dict) else entry}, not exactly "
            f"{sorted(LEGACY_ENTRY_KEYS)}; only a checkpoint signed without the optimisation "
            "problem is migrated")
    config, spec = design_problem.config, design_problem.spec
    if config.fingerprint() != config_fingerprint:
        raise ValueError("the design problem's configuration is not the one the run recorded")
    model, source = entry["model"], entry["scale"]["source_identity"]
    rebuilt = fine_flow_identity(spec, config, model["flow_mesh"]["refinement"],
                                 model["thermal_mesh"]["refinement"],
                                 model["thermal_mesh"]["quadrature"])
    if json.loads(rebuilt) != model:
        raise ValueError("the spec and configuration given do not rebuild the model identity "
                         "the checkpoint's binding names")
    if json.loads(development_identity(spec, config, source["thermal_mesh"]["refinement"],
                                       source["thermal_mesh"]["quadrature"])) != source:
        raise ValueError("the spec and configuration given do not rebuild the identity of the "
                         "scale's source that the checkpoint's binding names")
    if design_problem.num_design != checkpoint.num_design_var:
        raise ValueError(f"the design problem has {design_problem.num_design} design variables; "
                         f"the checkpoint {checkpoint.num_design_var}")
    entry = dict(entry, optimisation=optimisation_identity(design_problem))
    return _driver.MMACheckpoint(
        state_array=np.array(checkpoint.state_array, dtype=np.float64, copy=True),
        kktnorm=checkpoint.kktnorm, updates_done=checkpoint.updates_done,
        num_design_var=checkpoint.num_design_var,
        binding=json.dumps(dict(old, entry=entry), sort_keys=True))
