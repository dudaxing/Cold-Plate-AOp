"""R1g: a thermal mesh finer than the design and flow mesh, differentiable end to end.

R1f found that, on the R1d design, most of the refinement change in C comes
from the temperature approximation space: with the velocity function and the
parent tau held fixed, enriching the temperature space alone moved C by +4410
of a net +5462 along that path. So the temperature gets its own finer, nested
mesh, while the design variables and the flow stay where they are:

    x -> filter, projection -> s_D                    design = flow mesh, h
      -> alpha(s_D) -> Newton(flow)    [implicit]  -> u_F
      -> s_T = E s_D                  each thermal element copies its parent
      -> u_T = P u_F                  the coarse Q1 velocity at the thermal nodes
      -> kappa(s_T), tau_T(u_T, kappa, h_T)   recomputed at every evaluation
      -> Newton(thermal) [implicit]  -> T_T          thermal mesh, h / r
      -> Psi(u_F, alpha) on the flow mesh,  C(T_T, u_T, kappa) on the thermal mesh
      -> g on the design domain, unchanged

This changes how the temperature is discretised, not the physical task: the
same design, velocity function, materials, loads, boundaries, objective and
constraint. The coarse flow is a cost decision, not a claim that it is exact --
re-solving it on the fine mesh moved C by -2.5% in R1f.

**The maps.** E and P are fixed by the geometry, so their index and weight
tables are built once, in NumPy, and APPLIED with JAX gathers. That is what
makes them differentiable, and it fixes what their transposes are: the reverse
derivative of a gather is a scatter-ADD, so a parent's design sensitivity is the
SUM of its children's (E^T), and a coarse node's velocity sensitivity collects
every thermal node that interpolates from it, with its weight (P^T). Averaging
instead of summing would scale the thermal part of the gradient by 1/r^2 and
nothing else would look wrong.

`zhao2d_thermal_study.extend_velocity` evaluates the same P in NumPy, for static
diagnostics. It converts to NumPy, so it cannot sit inside a traced chain.

**tau is not frozen.** R1f's frozen-tau analysis was a counterfactual for
attribution. Here tau_T comes from the formula on the thermal mesh at every
evaluation, from the current u_T and kappa, and its dependence on both is part
of the gradient.

**Reference values.** Psi_0 and C_0 were frozen on the single-mesh model. They
are not this model's normalisation, and `objective_and_constraint` refuses them
rather than silently accepting a reference whose identity does not mention the
thermal mesh. For a model to be optimised, `freeze_reference` computes its own
Psi_0 and C_0 into a versioned file beside the single-mesh one, and
`load_reference` returns them only to the model they were frozen for. The
single-mesh values stay a stated common reporting scale, via
`reporting_objective`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib

import numpy as np
import jax
import jax.numpy as jnp

import toflux.src.solver as _solver

from tfopus import elements as _elements
from tfopus import fe_thermal as _fe_thermal
from tfopus import materials as _materials
from tfopus import zhao2d as _z
from tfopus import zhao2d_analysis as _za
from tfopus import zhao2d_r1 as _r1
from tfopus import zhao2d_refine as _refine

# Quad4 reference-node positions in `build_mesh`'s node order:
# (i, j), (i+1, j), (i+1, j+1), (i, j+1).
_REF_NODES = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


# --------------------------------------------------------------------------
# The nested maps
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class NestedMaps:
    """E (element density) and P (nodal velocity) from a coarse to a nested mesh.

    parents       (n_fine_elems,)    coarse element containing each fine one
    node_cells    (n_fine_nodes, 4)  coarse nodes of a cell containing each
                                     fine node
    node_weights  (n_fine_nodes, 4)  Q1 shape-function values there
    """

    factor: int
    parents: np.ndarray
    node_cells: np.ndarray
    node_weights: np.ndarray
    num_coarse_elems: int
    num_coarse_nodes: int

    @property
    def num_fine_elems(self) -> int:
        return len(self.parents)

    @property
    def num_fine_nodes(self) -> int:
        return len(self.node_cells)

    def density(self, s_coarse):
        """s_T = E s_D. A gather, so its transpose accumulates children."""
        return jnp.asarray(s_coarse)[self.parents]

    def velocity(self, u_coarse):
        """u_T = P u_F, (n_coarse_nodes, dim) -> (n_fine_nodes, dim)."""
        u = jnp.asarray(u_coarse)
        return jnp.einsum("nk, nkd -> nd", self.node_weights, u[self.node_cells])

    def matrices(self):
        """E and P as explicit SciPy matrices -- for verification only."""
        import scipy.sparse as sp

        n_f_e, n_f_n = self.num_fine_elems, self.num_fine_nodes
        e = sp.csr_matrix(
            (np.ones(n_f_e), (np.arange(n_f_e), self.parents)),
            shape=(n_f_e, self.num_coarse_elems),
        )
        p = sp.csr_matrix(
            (
                self.node_weights.ravel(),
                (np.repeat(np.arange(n_f_n), 4), self.node_cells.ravel()),
            ),
            shape=(n_f_n, self.num_coarse_nodes),
        )
        return e, p

    def identity(self) -> dict:
        """What the maps are, compactly enough to store beside a result."""

        def digest(*arrays):
            h = hashlib.sha256()
            for a in arrays:
                h.update(np.ascontiguousarray(a).tobytes())
            return h.hexdigest()

        return {
            "factor": self.factor,
            "coarse_elements": self.num_coarse_elems,
            "coarse_nodes": self.num_coarse_nodes,
            "fine_elements": self.num_fine_elems,
            "fine_nodes": self.num_fine_nodes,
            "parents_sha256": digest(self.parents.astype(np.int64)),
            "node_map_sha256": digest(
                self.node_cells.astype(np.int64), self.node_weights
            ),
        }


def _lattice(coords: np.ndarray, origin: np.ndarray, h: float) -> np.ndarray:
    """Integer lattice index of each point; raises if a point is off the lattice."""
    idx = np.rint((coords - origin) / h).astype(np.int64)
    if not np.allclose(origin + idx * h, coords, rtol=0.0, atol=1e-9 * h):
        raise RuntimeError("mesh nodes are not on the expected lattice")
    return idx


def build_nested_maps(
    coarse: _z.PlanarMesh, fine: _z.PlanarMesh, factor: int
) -> NestedMaps:
    """Index and weight tables for E and P, from the geometry alone.

    Everything is done in integers on the FINE lattice, so there is no floor()
    of a coordinate that should be an integer and is not quite. A fine node on
    a coarse cell edge lies in two or four coarse cells; any of them gives the
    same value, because the coarse field is continuous, so the first valid
    candidate is taken.
    """
    r = int(factor)
    if r < 1 or r != factor:
        raise ValueError(f"factor must be a positive integer, got {factor}")
    h_c = float(np.sqrt(np.asarray(coarse.elem_area)[0]))
    h_f = float(np.sqrt(np.asarray(fine.elem_area)[0]))
    if not np.isclose(h_c / h_f, r, rtol=1e-12):
        raise ValueError(f"element sizes {h_c:g} and {h_f:g} are not in ratio {r}")

    parents = _refine.parent_of_each_fine_element(coarse, fine)
    counts = np.bincount(parents, minlength=coarse.num_elems)
    if not np.all(counts == r * r):
        raise RuntimeError(f"expected {r * r} children per parent, got {set(counts)}")

    cc = np.asarray(coarse.mesh.nodes.coords)
    fc = np.asarray(fine.mesh.nodes.coords)
    origin = cc.min(axis=0)
    coarse_nodes = np.asarray(coarse.mesh.elem_nodes)

    lat_c = _lattice(cc, origin, h_f)  # coarse nodes, on the fine lattice
    lat_f = _lattice(fc, origin, h_f)
    lower_left = lat_c[coarse_nodes].min(axis=1)  # (n_c_elems, 2), multiples of r

    # The weights below assume build_mesh's node order; check rather than trust.
    expected = lower_left[:, None, :] + (r * (_REF_NODES + 1.0) / 2.0).astype(np.int64)
    if not np.array_equal(lat_c[coarse_nodes], expected):
        raise RuntimeError("coarse element node order is not the expected Quad4 order")

    cell_ij = lower_left // r
    table = np.full(cell_ij.max(axis=0) + 2, -1, dtype=np.int64)
    table[cell_ij[:, 0], cell_ij[:, 1]] = np.arange(coarse.num_elems)

    base = lat_f // r
    chosen = np.full(len(fc), -1, dtype=np.int64)
    for di, dj in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
        ci, cj = base[:, 0] + di, base[:, 1] + dj
        inside = (ci >= 0) & (cj >= 0) & (ci < table.shape[0]) & (cj < table.shape[1])
        cell = np.full(len(fc), -1, dtype=np.int64)
        cell[inside] = table[ci[inside], cj[inside]]
        a = lat_f[:, 0] - r * ci
        b = lat_f[:, 1] - r * cj
        ok = (cell >= 0) & (a >= 0) & (a <= r) & (b >= 0) & (b <= r) & (chosen < 0)
        chosen[ok] = cell[ok]
    if np.any(chosen < 0):
        raise RuntimeError(f"{int(np.sum(chosen < 0))} fine nodes lie in no coarse cell")

    # ... and that the template's N_k is the one that is 1 at reference node k
    shape = jax.vmap(coarse.mesh.elem_template.shape_functions)
    if not np.allclose(np.asarray(shape(jnp.asarray(_REF_NODES))), np.eye(4)):
        raise RuntimeError("Quad4 shape functions are not in the expected node order")

    local = (lat_f - lower_left[chosen]).astype(float)  # in [0, r]
    xi_eta = -1.0 + 2.0 * local / r
    weights = np.asarray(shape(jnp.asarray(xi_eta)))

    return NestedMaps(
        factor=r,
        parents=parents,
        node_cells=coarse_nodes[chosen],
        node_weights=weights,
        num_coarse_elems=coarse.num_elems,
        num_coarse_nodes=len(cc),
    )


# --------------------------------------------------------------------------
# The dual-mesh problem
# --------------------------------------------------------------------------


def reporting_objective(psi, c, scale: _r1.ReferenceValues, weight: float):
    """J* = w Psi/Psi_0 + (1-w) C/C_0 on a STATED reporting scale.

    Not a normalised objective of the dual-mesh model: `scale` is the
    single-mesh frozen reference, used here only so results on different
    thermal meshes are expressed in one unit. A dual-mesh optimisation needs its
    own versioned reference first.
    """
    return weight * psi / scale.psi_0 + (1.0 - weight) * c / scale.c_0


class Zhao2DDualProblem(_r1.Zhao2DProblem):
    """Zhao2DProblem with the temperature on a nested mesh r times finer.

    Everything on the design and flow side -- meshes, filter, projection,
    volume constraint, flow solver, the convergence gate -- is inherited
    unchanged. Only the thermal members are rebuilt, and the states are wired
    through the maps.
    """

    def __init__(
        self,
        spec: _z.Zhao2DSpec,
        config: _r1.R1Config = _r1.R1Config(),
        thermal_refinement: int = 2,
        thermal_quadrature: int = 3,
        solver_settings: dict | None = None,
    ):
        super().__init__(spec, config, solver_settings)
        self.thermal_refinement = int(thermal_refinement)
        self.thermal_quadrature = int(thermal_quadrature)
        self.thermal_spec = _refine.refine_spec(spec, self.thermal_refinement)

        # Replace the single-mesh thermal members built by the parent.
        self.thermal_mesh = _z.build_mesh(
            self.thermal_spec, dofs_per_node=1, gauss_order=self.thermal_quadrature
        )
        self.maps = build_nested_maps(
            self.flow_mesh, self.thermal_mesh, self.thermal_refinement
        )
        self.thermal_bc = _z.build_thermal_bc(self.thermal_mesh, spec)
        self.q_source = _z.heat_source_field(
            self.thermal_mesh, spec, config.source_region
        )
        # tau_T uses the THERMAL mesh's own element length, not the flow mesh's
        self.h_thermal = _elements.element_lengths(
            self.thermal_mesh.mesh, config.element_length_mode
        )
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
        """What must be invariant between the two meshes; raises otherwise."""
        coarse, fine, parents = self.flow_mesh, self.thermal_mesh, self.maps.parents
        area_c = np.asarray(coarse.elem_area)
        area_f = np.asarray(fine.elem_area)
        child_area = np.bincount(parents, weights=area_f, minlength=coarse.num_elems)
        q_c = np.asarray(
            _z.heat_source_field(coarse, self.spec, self.config.source_region)
        )
        heat_c = float(np.sum(q_c * area_c))
        heat_f = float(np.sum(np.asarray(self.q_source) * area_f))

        if not np.allclose(child_area, area_c, rtol=1e-12):
            raise RuntimeError("children do not tile their parents")
        if not np.array_equal(fine.design_mask, coarse.design_mask[parents]):
            raise RuntimeError("the design/tab partition is not preserved")
        if not np.isclose(heat_c, heat_f, rtol=1e-12):
            raise RuntimeError(f"total heat source changed: {heat_c} -> {heat_f}")
        return {
            "area_flow_mesh": float(area_c.sum()),
            "area_thermal_mesh": float(area_f.sum()),
            "heat_source_flow_mesh": heat_c,
            "heat_source_thermal_mesh": heat_f,
            "design_cells_flow_mesh": int(coarse.design_mask.sum()),
            "design_cells_thermal_mesh": int(fine.design_mask.sum()),
        }

    # -- the maps, applied --------------------------------------------------

    def thermal_nodal_velocity(self, press_vel):
        """(n_thermal_nodes, dim): u_T = P u_F."""
        u_f = jnp.asarray(press_vel).reshape(-1, self.flow.num_fields)[:, 1:]
        return self.maps.velocity(u_f)

    def thermal_velocity(self, press_vel):
        """(n_thermal_elems, nodes_per_elem * dim), the thermal solver's layout."""
        u_t = self.thermal_nodal_velocity(press_vel)
        nodes = jnp.asarray(self.thermal_mesh.mesh.elem_nodes)
        return u_t[nodes].reshape(self.thermal_mesh.num_elems, -1)

    def thermal_conductivity(self, s, alpha_max: float):
        """kappa(E s) on the thermal mesh."""
        material = _za.build_material(self.spec, alpha_max)
        return _materials.conductivity(self.maps.density(s), material)

    def solve_thermal(self, press_vel, s, alpha_max: float):
        """T on the thermal mesh for a GIVEN flow state. Differentiable in both.

        For fixed-design studies that solve the flow once and reuse it; the
        optimisation chain goes through `solve_states`.
        """
        return _solver.modified_newton_raphson_solve(
            self.thermal,
            self.thermal_x0,
            self.thermal_velocity(press_vel),
            self.thermal_conductivity(s, alpha_max),
            self.q_source,
        )

    # -- states, metrics, residuals -----------------------------------------

    def solve_states(self, s, alpha_max: float):
        """(press_vel, temperature, alpha, kappa_T). kappa is on the THERMAL mesh."""
        material = _za.build_material(self.spec, alpha_max)
        alpha = _materials.brinkman_penalty(s, material)
        press_vel = _solver.modified_newton_raphson_solve(
            self.flow, self.flow_x0, alpha
        )
        kappa_t = _materials.conductivity(self.maps.density(s), material)
        temperature = _solver.modified_newton_raphson_solve(
            self.thermal,
            self.thermal_x0,
            self.thermal_velocity(press_vel),
            kappa_t,
            self.q_source,
        )
        return press_vel, temperature, alpha, kappa_t

    def metrics(self, s, alpha_max: float):
        """(Psi on the flow mesh, C on the thermal mesh) as JAX scalars."""
        press_vel, temperature, alpha, kappa_t = self.solve_states(s, alpha_max)
        psi = self.flow.dissipated_power(press_vel, alpha)
        c = self.thermal.thermal_compliance(
            temperature, self.thermal_velocity(press_vel), kappa_t
        )
        return psi, c

    # `residual_norms` and `residual_norms_at` are the parent's: they measure
    # the thermal residual through `thermal_velocity`, which is this class's.

    def evaluate(self, x, alpha_max: float, beta: float | None = None) -> dict:
        """Psi, C, g and both residuals from ONE solve. Reporting, not traced.

        The finite-difference side of a gradient check needs every perturbed
        state gated for convergence; doing that through `require_converged`
        solves each state twice.
        """
        s = self.solid_fraction(x, beta)
        press_vel, temperature, alpha, kappa_t = self.solve_states(s, alpha_max)
        psi = self.flow.dissipated_power(press_vel, alpha)
        c = self.thermal.thermal_compliance(
            temperature, self.thermal_velocity(press_vel), kappa_t
        )
        g = self.fluid_fraction(x, beta) / self.config.max_fluid_fraction - 1.0
        return {
            "psi": float(psi),
            "c": float(c),
            "g": float(g),
            "residuals": self.residual_norms_at(press_vel, temperature, alpha, kappa_t),
        }

    # -- objective ----------------------------------------------------------

    def reference_identity(self) -> str:
        """The single-mesh identity plus the thermal discretisation."""
        payload = json.loads(_r1.reference_identity(self.spec, self.config))
        payload["thermal_mesh"] = {
            "refinement": self.thermal_refinement,
            "quadrature": self.thermal_quadrature,
            "element_length_mode": self.config.element_length_mode,
        }
        return json.dumps(payload, sort_keys=True)

    def check_reference(self, reference: _r1.ReferenceValues) -> None:
        """Refuse any reference not frozen for THIS thermal model.

        The inherited check compares only the spec and the R1 config, neither of
        which mentions the thermal mesh -- so it would accept the single-mesh
        denominators, or another thermal mesh's, without complaint.
        """
        if reference.identity != self.reference_identity():
            raise ValueError(
                "this reference was not frozen for the dual-mesh thermal model "
                f"(refinement {self.thermal_refinement}, quadrature "
                f"{self.thermal_quadrature}); the single-mesh Psi_0/C_0 are a "
                "reporting scale here, not a normalisation. Freeze a new, "
                "versioned reference for this model before optimising with it."
            )

    def objective_and_constraint(self, x, reference: _r1.ReferenceValues, alpha_max: float):
        """(J, g), refusing any reference not frozen for THIS thermal model."""
        self.check_reference(reference)
        s = self.solid_fraction(x)
        psi, c = self.metrics(s, alpha_max)
        w = self.config.weight
        j = w * psi / reference.psi_0 + (1.0 - w) * c / reference.c_0
        g = self.fluid_fraction(x) / self.config.max_fluid_fraction - 1.0
        return j, g


# --------------------------------------------------------------------------
# This model's own reference, versioned
# --------------------------------------------------------------------------


def reference_file(thermal_refinement: int, thermal_quadrature: int,
                   version: int = 1) -> pathlib.Path:
    """Where a dual-mesh model's frozen reference lives -- never the single-mesh file.

    The name carries the thermal mesh and a version, so a refrozen reference
    for the same model is a new file beside the old one, not an overwrite.
    """
    return (pathlib.Path(__file__).resolve().parent
            / f"zhao2d_reference_dual_r{thermal_refinement}q{thermal_quadrature}"
              f"_v{version}.json")


def freeze_reference(spec: _z.Zhao2DSpec, config: _r1.R1Config = _r1.R1Config(),
                     thermal_refinement: int = 4, thermal_quadrature: int = 3,
                     solver_settings: dict | None = None, tol: float = 1e-8):
    """Psi_0 and C_0 for THIS dual-mesh model, bound to its identity.

    The reference state is the configured physical density -- gamma = 0.4 in
    the design domain, the tabs fluid -- set directly, never through the filter
    or the projection, at alpha_max_reference. It is solved ONCE; the gate is
    applied to the states that solve returned, and Psi_0 and C_0 are taken from
    those same states.

    With w fixed, a new C_0 changes how C is weighed against Psi in J: it is a
    different objective, not the old one in new units.

    Returns (ReferenceValues, report, problem).
    """
    problem = Zhao2DDualProblem(spec, config, thermal_refinement, thermal_quadrature,
                                solver_settings)
    s = _z.reference_solid_fraction(problem.flow_mesh, spec, config.reference_field)
    alpha_max = config.alpha_max_reference
    press_vel, temperature, alpha, kappa_t = problem.solve_states(s, alpha_max)
    norms = problem.residual_norms_at(press_vel, temperature, alpha, kappa_t)
    bad = {k: v for k, v in norms.items() if not v <= tol}
    if bad:
        raise _r1.NotConverged(
            f"the reference state did not converge ({bad}); nothing frozen"
        )
    psi = float(problem.flow.dissipated_power(press_vel, alpha))
    c = float(problem.thermal.thermal_compliance(
        temperature, problem.thermal_velocity(press_vel), kappa_t))
    values = _r1.ReferenceValues(
        psi_0=psi,
        c_0=c,
        identity=problem.reference_identity(),
        run_fingerprint=config.fingerprint(),
        spec_element_size=spec.element_size,
        flow_residual_relative=norms["flow"],
        thermal_residual_relative=norms["thermal"],
    )
    report = {
        "psi_0": psi,
        "c_0": c,
        **{f"residual_{k}": v for k, v in norms.items()},
        **_z.fluid_fractions(problem.flow_mesh, s),
        "t_max": float(jnp.max(temperature)),
        "thermal_elements": int(problem.thermal_mesh.num_elems),
    }
    return values, report, problem


def load_reference(problem: Zhao2DDualProblem, path: pathlib.Path | None = None):
    """A frozen dual-mesh reference, only if it belongs to `problem`."""
    path = path or reference_file(problem.thermal_refinement, problem.thermal_quadrature)
    if not pathlib.Path(path).is_file():
        raise FileNotFoundError(
            f"{path} not found; freeze it with scripts/zhao2d_freeze_dual_reference.py"
        )
    values = _r1.ReferenceValues.from_json(pathlib.Path(path).read_text(encoding="utf-8"))
    problem.check_reference(values)
    return values
