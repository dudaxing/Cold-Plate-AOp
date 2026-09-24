"""Zhao's 2D heat sink (section 4.1), rebuilt as a density-method case.

Zhao et al., *Applied Thermal Engineering* **291** (2026) 130088,
doi:10.1016/j.applthermaleng.2026.130088. The CBS parametrisation is
deliberately not implemented: the design field is a per-element solid fraction.
Everything downstream of the pseudo-density -- interpolations, governing
equations, objective, constraint -- is kept.

**Geometry.** Figure 7 is a schematic with dimensions but no coordinates. The
reconstruction below is an interpretation, and `Zhao2DSpec.provenance()` prints
which numbers came from the paper and which did not. One independent check
supports it: at h = 1e-4 the half model has

    50 x 100 (design) + 2 x (10 x 10) (inlet and outlet tabs) = 5200

elements, and the paper states 5200. That is evidence for the interpretation,
not proof of it -- a different split could give the same total.

**What is genuinely undetermined**, and what R0 exists to measure:

1.  The reference field. The paper says "a uniform pseudo-density field
    distribution gamma = 0.4" without saying whether the non-design inlet and
    outlet tabs share it or stay fluid. `ReferenceField` builds both.
2.  The volume-fraction domain. Equation 26 writes
    v_f = (1/|Omega|) integral gamma dOmega over Omega, but the design freedom
    lives only in the design domain. Both are reported, always.
3.  alpha_max at the reference. Continuation runs 1e6 -> 1e7 and the paper does
    not say which value the reference model used.
4.  The outlet. Figure 7 writes p = 0; the weak form's natural condition is a
    traction. See `OutletKind`.
5.  Where the heat source acts. The text says "across the entire domain"; the
    figure prints Q = 1e8 inside the design domain. See `SourceRegion`.

Nothing here tunes any of these to hit the paper's numbers. They are enumerated,
computed and reported.
"""

from __future__ import annotations

import dataclasses
import enum

import numpy as np
import jax
import jax.numpy as jnp

import toflux.src.bc as _bc
import toflux.src.mesher as _mesher

from tfopus import elements as _elements
from tfopus import fe_flow as _fe_flow
from tfopus import fe_thermal as _fe_thermal
from tfopus import materials as _materials
from tfopus.mesh import Face, Region


class ReferenceField(enum.Enum):
    """Which uniform field the paper's Psi_0 / C_0 were computed on."""

    TABS_FLUID = "tabs_fluid"  # gamma = 0.4 in the design domain, 1 in the tabs
    UNIFORM_ALL = "uniform_all"  # gamma = 0.4 over the whole physical domain


class OutletKind(enum.Enum):
    """How "p = 0" at the outlet is discretised."""

    TRACTION = "traction"  # natural zero external traction; nothing constrained
    PINNED = "pinned"  # p pinned nodally to 0 on the outlet face


class SourceRegion(enum.Enum):
    """Where the uniform volumetric heat source acts."""

    WHOLE_DOMAIN = "whole_domain"  # section 4.1 text
    DESIGN_ONLY = "design_only"  # figure 7(b) annotation


# --------------------------------------------------------------------------
# Specification
# --------------------------------------------------------------------------

# field -> (source, note). "paper" means the value is printed in Zhao; the rest
# are reconstruction choices this module is responsible for.
_PROVENANCE = {
    "inlet_half_width": ("reconstruction", "fig 7(b) marks 0.001 beside the tab"),
    "tab_length": ("reconstruction", "fig 7(a) L; fig 7(b) marks 0.001"),
    "design_half_width": ("reconstruction", "fig 7(a) shows a square design domain"),
    "design_height": ("reconstruction", "fig 7(a) 10L; fig 7(b) marks 0.01"),
    "element_size": ("derived", "h giving the paper's 5200 elements"),
    "inlet_speed": ("paper", "section 4.1: inlet velocity 0.2"),
    "heat_source": ("paper", "fig 7(b): Q = 1e8"),
    "inlet_temperature": ("paper", "fig 7(b): T = 0"),
    "fluid_density": ("paper", "table 1"),
    "fluid_viscosity": ("paper", "table 1"),
    "fluid_heat_capacity": ("paper", "table 1"),
    "fluid_conductivity": ("paper", "table 1"),
    "solid_conductivity": ("paper", "table 1"),
    "q_alpha": ("paper", "section 3.1: q_alpha = q_kappa = 0.2"),
    "q_kappa": ("paper", "section 3.1: q_alpha = q_kappa = 0.2"),
    "alpha_max_initial": ("paper", "section 4.1: alpha_max from 1e6"),
    "alpha_max_final": ("paper", "section 4.1: ... to 1e7"),
    "alpha_max_growth": ("paper", "section 4.1: alpha_max = 1e6 * 1.03^iter"),
    "max_fluid_fraction": ("paper", "section 4.1: 40%"),
    "weight": ("paper", "section 4.1: w = 0.5"),
    "reference_gamma": ("paper", "section 4.1: uniform gamma = 0.4"),
    "reported_psi_0": ("paper", "section 4.1: Psi_0 = 20,816"),
    "reported_c_0": ("paper", "section 4.1: C_0 = 0.0456"),
    "reported_reynolds": ("paper", "section 4.1: Re = 200"),
    "reported_num_elements": ("paper", "section 4.1: 5200 square elements"),
}

_GAPS = [
    "The reference field's extent (design domain only, or the tabs too) is not stated.",
    "alpha_max at the reference model is not stated; continuation spans 1e6 to 1e7.",
    "Equation 26's |Omega| for v_f is not resolved against the non-design tabs.",
    "The discrete form of the p = 0 outlet is not disclosed.",
    "The heat source region: the text says the entire domain, fig 7(b) marks the "
    "design domain.",
    "Table 1 lists solid density 2700 and heat capacity 900, but section 2 allows "
    "the fluid rho*c over the whole domain; the solid pair is therefore unused.",
    "h_e for the stabilisation parameters is named but never defined.",
]


@dataclasses.dataclass(frozen=True)
class Zhao2DSpec:
    """Geometry and physics for the 2D heat sink, all dimensionless.

    The paper states table 1's properties "are assumed to be dimensionless", so
    no SI interpretation is attached to any length here.
    """

    # geometry (half model; symmetry plane at x = 0)
    inlet_half_width: float = 0.001
    tab_length: float = 0.001
    design_half_width: float = 0.005
    design_height: float = 0.010
    element_size: float = 1.0e-4

    # loading
    inlet_speed: float = 0.2
    heat_source: float = 1.0e8
    inlet_temperature: float = 0.0

    # table 1
    fluid_density: float = 1000.0
    fluid_viscosity: float = 0.001
    fluid_heat_capacity: float = 4180.0
    fluid_conductivity: float = 0.61
    solid_conductivity: float = 237.0

    # interpolation
    q_alpha: float = 0.2
    q_kappa: float = 0.2
    alpha_max_initial: float = 1.0e6
    alpha_max_final: float = 1.0e7
    alpha_max_growth: float = 1.03

    # optimisation
    max_fluid_fraction: float = 0.40
    weight: float = 0.5
    reference_gamma: float = 0.4

    # reported, for comparison only -- never a target to tune toward
    reported_psi_0: float = 20816.0
    reported_c_0: float = 0.0456
    reported_reynolds: float = 200.0
    reported_num_elements: int = 5200

    @property
    def b_f(self) -> float:
        """Advective volumetric heat capacity rho*c, used everywhere."""
        return self.fluid_density * self.fluid_heat_capacity

    def alpha_max(self, iteration: int) -> float:
        """alpha_max = 1e6 * 1.03^iter, capped at 1e7 (section 4.1)."""
        return min(
            self.alpha_max_final,
            self.alpha_max_initial * self.alpha_max_growth**iteration,
        )

    def reynolds(self, length_scale: float | None = None) -> float:
        """rho * U * L / mu. Defaults to the inlet half width.

        With L = the inlet half width 0.001 this returns 200, matching the
        paper. The full inlet width 0.002 would give 400, so the paper's Re
        identifies the half width as its characteristic length -- a consistency
        check on the geometry, not an input.
        """
        L = self.inlet_half_width if length_scale is None else length_scale
        return self.fluid_density * self.inlet_speed * L / self.fluid_viscosity

    def provenance(self) -> str:
        """Per-field: paper value, derived, or reconstruction choice."""
        lines = ["Zhao 2D heat sink -- field provenance", ""]
        for field in dataclasses.fields(self):
            src, note = _PROVENANCE.get(field.name, ("unrecorded", ""))
            value = getattr(self, field.name)
            lines.append(f"  {field.name:24s} = {value!s:<12s} [{src}] {note}")
        lines += ["", "Recorded gaps in the paper:"]
        lines += [f"  - {g}" for g in _GAPS]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------


@dataclasses.dataclass
class PlanarMesh:
    """A 2D structured mesh plus the tags the case driver addresses it by."""

    mesh: _mesher.Mesh
    region: np.ndarray  # (num_elems,) Region
    elem_faces: dict  # Face -> [(elem, local_face), ...]
    elem_centres: np.ndarray  # (num_elems, 2)
    elem_area: np.ndarray  # (num_elems,)

    @property
    def num_elems(self) -> int:
        return self.mesh.num_elems

    @property
    def design_mask(self) -> np.ndarray:
        return self.region == int(Region.DESIGN)

    def face_count(self) -> dict:
        return {f.name: len(v) for f, v in self.elem_faces.items() if v}


def build_mesh(spec: Zhao2DSpec, dofs_per_node: int, gauss_order: int = 2) -> PlanarMesh:
    """The half model of figure 7(b): design rectangle plus two inlet/outlet tabs.

    Built by masking a structured grid over the bounding box, then renumbering
    the surviving nodes. Cells are kept by CENTRE, so the tab/design interface
    lands exactly on a grid line when the tab dimensions are multiples of h.
    """
    h = spec.element_size
    x0, x1 = 0.0, spec.design_half_width
    y0, y1 = -spec.tab_length, spec.design_height + spec.tab_length

    nx = int(round((x1 - x0) / h))
    ny = int(round((y1 - y0) / h))
    for name, extent in (
        ("design_half_width", spec.design_half_width),
        ("design_height", spec.design_height),
        ("tab_length", spec.tab_length),
        ("inlet_half_width", spec.inlet_half_width),
    ):
        if abs(extent / h - round(extent / h)) > 1e-9:
            raise ValueError(f"{name}={extent} is not a whole number of h={h}")

    xs = x0 + h * np.arange(nx + 1)
    ys = y0 + h * np.arange(ny + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    all_nodes = np.stack([gx.ravel(), gy.ravel()], axis=1)

    def node_id(i, j):
        return i * (ny + 1) + j

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    cx = x0 + h * (ii + 0.5)
    cy = y0 + h * (jj + 0.5)

    in_design = (cy > 0.0) & (cy < spec.design_height)
    in_tab = cx < spec.inlet_half_width
    in_inlet_tab = in_tab & (cy > spec.design_height)
    in_outlet_tab = in_tab & (cy < 0.0)
    keep = in_design | in_inlet_tab | in_outlet_tab

    region = np.where(in_design[keep], int(Region.DESIGN), int(Region.PORT_FLUID))

    # Quad4 node order: (i,j), (i+1,j), (i+1,j+1), (i,j+1) -- counter-clockwise.
    quads = np.stack(
        [
            node_id(ii, jj),
            node_id(ii + 1, jj),
            node_id(ii + 1, jj + 1),
            node_id(ii, jj + 1),
        ],
        axis=1,
    )[keep]

    used = np.unique(quads)
    renumber = np.full(all_nodes.shape[0], -1, dtype=int)
    renumber[used] = np.arange(len(used))
    elem_nodes = renumber[quads]

    mesh = _mesher.Mesh(
        nodes=_mesher.Nodes(
            coords=jnp.asarray(all_nodes[used]), dof_per_node=dofs_per_node
        ),
        elem_nodes=elem_nodes,
        elem_template=_elements.Quad4(),
        gauss_order=gauss_order,
    )

    return PlanarMesh(
        mesh=mesh,
        region=region,
        elem_faces=_tag_faces(mesh, spec),
        elem_centres=np.stack([cx[keep], cy[keep]], axis=1),
        elem_area=np.asarray(mesh.elem_volume),
    )


def _tag_faces(mesh: _mesher.Mesh, spec: Zhao2DSpec) -> dict:
    """Tag every boundary face by where its midpoint sits."""
    conn = np.asarray(mesh.elem_template.face_connectivity)
    coords = np.asarray(mesh.elem_node_coords)
    elems, faces = np.nonzero(np.asarray(mesh.boundary_faces))
    tol = 0.25 * spec.element_size

    y_in = spec.design_height + spec.tab_length
    y_out = -spec.tab_length

    out = {f: [] for f in Face}
    for e, f in zip(elems, faces):
        mid = coords[e][conn[f]].mean(axis=0)
        if abs(mid[1] - y_in) < tol and mid[0] < spec.inlet_half_width:
            tag = Face.INLET
        elif abs(mid[1] - y_out) < tol and mid[0] < spec.inlet_half_width:
            tag = Face.OUTLET
        elif abs(mid[0]) < tol:
            tag = Face.SYMMETRY
        else:
            tag = Face.WALL
        out[tag].append((int(e), int(f)))
    return out


def face_nodes(planar: PlanarMesh, tag: Face) -> np.ndarray:
    """Unique global node indices on all faces carrying `tag`."""
    conn = np.asarray(planar.mesh.elem_template.face_connectivity)
    elem_nodes = np.asarray(planar.mesh.elem_nodes)
    pairs = planar.elem_faces[tag]
    if not pairs:
        return np.zeros(0, dtype=int)
    return np.unique(np.concatenate([elem_nodes[e][conn[f]] for e, f in pairs]))


# --------------------------------------------------------------------------
# Boundary conditions
# --------------------------------------------------------------------------


@dataclasses.dataclass
class FlowBC:
    bc: dict
    inlet_nodes: np.ndarray
    wall_nodes: np.ndarray
    symmetry_nodes: np.ndarray
    outlet_nodes: np.ndarray
    outlet_kind: OutletKind


def build_flow_bc(
    planar: PlanarMesh,
    spec: Zhao2DSpec,
    outlet: OutletKind = OutletKind.TRACTION,
) -> FlowBC:
    """Inlet v = -U, no-slip walls, u_x = 0 on the symmetry plane.

    The inlet rim shares nodes with the wall. No-slip is applied first and the
    inlet second, so the inlet wins on shared nodes and the discrete inflow is
    exactly U times the inlet width. The alternative (wall wins) would remove
    one element's worth of flux from a ten-element inlet -- 10% of the mass
    flow -- which is why the rule is fixed here rather than left to dictionary
    ordering.
    """
    mesh = planar.mesh
    dof = mesh.nodes.dof_per_node
    if dof != 3:
        raise ValueError(f"flow mesh needs 3 dofs per node, got {dof}")

    inlet = face_nodes(planar, Face.INLET)
    wall = face_nodes(planar, Face.WALL)
    symmetry = face_nodes(planar, Face.SYMMETRY)
    outlet_nodes = face_nodes(planar, Face.OUTLET)

    values: dict[int, float] = {}

    # symmetry: only the normal component u_x
    for nd in symmetry:
        values[dof * nd + 1] = 0.0
    # walls: no slip
    for nd in wall:
        values[dof * nd + 1] = 0.0
        values[dof * nd + 2] = 0.0
    # inlet: wins on the rim
    for nd in inlet:
        values[dof * nd + 1] = 0.0
        values[dof * nd + 2] = -spec.inlet_speed

    if outlet is OutletKind.PINNED:
        for nd in outlet_nodes:
            values[dof * nd + 0] = 0.0

    fixed = np.array(sorted(values), dtype=int)
    bc = _bc.BCDict(
        elem_forces=jnp.zeros((mesh.num_elems, mesh.num_dofs_per_elem)),
        fixed_dofs=fixed,
        free_dofs=np.setdiff1d(np.arange(mesh.num_dofs), fixed),
        dirichlet_values=np.array([values[d] for d in fixed]),
    )
    return FlowBC(bc, inlet, wall, symmetry, outlet_nodes, outlet)


def build_thermal_bc(planar: PlanarMesh, spec: Zhao2DSpec) -> dict:
    """T prescribed at the inlet; adiabatic walls and outlet are natural."""
    mesh = planar.mesh
    if mesh.nodes.dof_per_node != 1:
        raise ValueError("thermal mesh needs one dof per node")
    inlet = face_nodes(planar, Face.INLET)
    return _bc.BCDict(
        elem_forces=jnp.zeros((mesh.num_elems, mesh.num_dofs_per_elem)),
        fixed_dofs=inlet,
        free_dofs=np.setdiff1d(np.arange(mesh.num_dofs), inlet),
        dirichlet_values=np.full(len(inlet), spec.inlet_temperature),
    )


# --------------------------------------------------------------------------
# Fields
# --------------------------------------------------------------------------


def reference_solid_fraction(
    planar: PlanarMesh, spec: Zhao2DSpec, kind: ReferenceField
) -> jnp.ndarray:
    """(num_elems,) solid fraction s = 1 - gamma for a reference field.

    TABS_FLUID  gamma = 0.4 in the design domain, gamma = 1 in the tabs
    UNIFORM_ALL gamma = 0.4 everywhere
    """
    s_design = 1.0 - spec.reference_gamma
    if kind is ReferenceField.UNIFORM_ALL:
        return jnp.full(planar.num_elems, s_design)
    return jnp.where(jnp.asarray(planar.design_mask), s_design, 0.0)


def heat_source_field(
    planar: PlanarMesh, spec: Zhao2DSpec, where: SourceRegion
) -> jnp.ndarray:
    """(num_elems,) volumetric source.

    Independent of the design field: multiplying Q by the density would let the
    optimiser shrink the total heat load by adding fluid, which is a different
    problem from the one the paper poses.
    """
    if where is SourceRegion.WHOLE_DOMAIN:
        return jnp.full(planar.num_elems, spec.heat_source)
    return jnp.where(jnp.asarray(planar.design_mask), spec.heat_source, 0.0)


def element_adjacency(planar: PlanarMesh):
    """Face-neighbour graph of the structured masked grid.

    Face neighbours only, not diagonals: two cells touching at a single corner
    share one node and carry no channel, so counting them as connected would
    report a flow path where the discretisation has none.

    The cell index is `floor(centre / h)`, NOT `round`. Centres sit at
    (i + 1/2) h, so `centre / h` is always a half-integer and Python's round
    does banker's rounding on every one of them -- i and i+1 collapse onto the
    same key for odd i. That silently dropped 160 of 5200 elements from the
    graph and reported a fully fluid domain as disconnected.
    """
    import scipy.sparse as sp

    centres = np.asarray(planar.elem_centres)
    h = float(np.sqrt(np.asarray(planar.elem_area)[0]))
    index = {
        (int(np.floor(x / h)), int(np.floor(y / h))): i
        for i, (x, y) in enumerate(centres)
    }
    if len(index) != len(centres):
        raise RuntimeError(
            f"cell index collided: {len(centres)} elements mapped to "
            f"{len(index)} keys"
        )
    rows, cols = [], []
    for (i, j), e in index.items():
        for nb in ((i + 1, j), (i, j + 1)):
            if nb in index:
                rows += [e, index[nb]]
                cols += [index[nb], e]
    n = len(centres)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def fluid_connectivity(planar: PlanarMesh, fluid_mask, inlet_elems, outlet_elems):
    """Connected components of the fluid phase, and whether the ports join.

    Returns (num_components, component_sizes, connected, inlet_ids, outlet_ids).
    """
    import scipy.sparse.csgraph as csgraph

    fluid = np.asarray(fluid_mask, dtype=bool)
    graph = element_adjacency(planar)
    num, labels = csgraph.connected_components(
        graph[fluid][:, fluid], directed=False
    )
    sizes = np.bincount(labels, minlength=num)
    lookup = {e: labels[k] for k, e in enumerate(np.nonzero(fluid)[0])}
    inlet_ids = sorted({lookup[e] for e in inlet_elems if e in lookup})
    outlet_ids = sorted({lookup[e] for e in outlet_elems if e in lookup})
    connected = bool(set(inlet_ids) & set(outlet_ids))
    return num, sizes, connected, inlet_ids, outlet_ids


def fluid_fractions(planar: PlanarMesh, s) -> dict:
    """Area-weighted fluid fraction over the whole domain and the design domain.

    Both are reported because equation 26's |Omega| is not resolved against the
    non-design tabs.
    """
    area = jnp.asarray(planar.elem_area)
    gamma = 1.0 - jnp.asarray(s)
    design = jnp.asarray(planar.design_mask)
    return {
        "v_f_whole_domain": float(jnp.sum(gamma * area) / jnp.sum(area)),
        "v_f_design_domain": float(
            jnp.sum(jnp.where(design, gamma * area, 0.0))
            / jnp.sum(jnp.where(design, area, 0.0))
        ),
    }
