"""Boundary conditions, with the shared-node resolution made explicit.

**The outlet.** With the symmetric viscous form used by `fe_flow`, integrating
the momentum equation by parts produces a traction term:

    integral [ rho v.(u.grad)u + 2 mu eps(v):eps(u) - p div v + alpha v.u ] dV
      - integral_Gamma v . (sigma n) dA,        sigma = -p I + 2 mu eps(u)

What appears naturally is the TRACTION, not a pressure value. This module
implements the outlet as a zero external pressure traction,

    sigma n = -p_out n,   p_out = 0,

which means: no essential constraint at all on the outlet -- not on the three
velocity components, and not on pressure either -- and no boundary integral
added, since the term is p_out * integral v.n dA = 0. The outlet traction fixes
the pressure level, so **no pressure pin and no zero-mean constraint is added**;
doing so would over-determine the problem.

This gives, at the outlet, p = p_out + 2 mu n.eps(u)n. The local static pressure
equals p_out only where the normal viscous stress happens to vanish. Pinning
p = 0 pointwise instead is a DIFFERENT boundary-value problem, and equal-order
stabilisation does not make the two equivalent. Zhou states only that a
zero-pressure BC is imposed and does not disclose the discrete form, so this is
recorded as a reproduction choice, not a recovered author setting.

One consequence for verification: fully developed duct flow u = (U(y,z), 0, 0)
has outlet traction (-p, mu U_y, mu U_z). The tangential traction is generally
NOT zero, so a zero-traction outlet is inconsistent with holding the analytic
profile exactly up to the outlet plane. The analytic benchmark must either apply
the analytic traction or compare inside a fully developed interior region.

**Shared nodes.** Faces are tagged uniquely, but nodes are not: the inlet rim
sits on the wall. In a continuous Q1 space those nodes cannot carry both
u = -U n and u = 0, and resolving it by dictionary insertion order would make
the answer depend on the order boundary conditions happen to be listed in. The
rule is therefore explicit and reported:

    EdgeRule.INLET_WINS  inlet velocity on every inlet-face node, including the
                         rim. The discrete inlet flux is then exactly U * A_in;
                         the price is a velocity jump across one element at the
                         inlet-wall junction.
    EdgeRule.WALL_WINS   no-slip on the rim. Continuous, but on a mesh this thin
                         it removes a large share of the flux: with two elements
                         through a 0.5 mm wall, the through-thickness trapezoid
                         of (0, U, 0) is U*h/2, i.e. half, before the lateral
                         rim is even counted.

Neither is silently "corrected" by rescaling the velocity or switching to a
parabolic profile. `inlet_flux_report` integrates the actual discrete boundary
field so the mass flow that was really imposed is always a measured number.
"""

from __future__ import annotations

import dataclasses
import enum

import numpy as np
import jax.numpy as jnp

import toflux.src.bc as _bc

from tfopus import mesh as _mesh

Face = _mesh.Face


class EdgeRule(enum.Enum):
    """Who wins on a node shared by the inlet/outlet rim and a wall."""

    INLET_WINS = "inlet_wins"
    WALL_WINS = "wall_wins"


def face_nodes(shell: _mesh.ShellMesh, tag: Face) -> np.ndarray:
    """Unique global node indices on all faces carrying `tag`."""
    conn = np.asarray(shell.mesh.elem_template.face_connectivity)
    elem_nodes = np.asarray(shell.mesh.elem_nodes)
    pairs = shell.elem_faces[tag]
    if not pairs:
        return np.zeros(0, dtype=int)
    nodes = [elem_nodes[e][conn[f]] for e, f in pairs]
    return np.unique(np.concatenate(nodes))


def face_outward_normals(shell: _mesh.ShellMesh, tag: Face) -> np.ndarray:
    """(num_faces, 3) unit outward normals, oriented away from the element centre."""
    conn = np.asarray(shell.mesh.elem_template.face_connectivity)
    coords = np.asarray(shell.mesh.elem_node_coords)
    out = []
    for e, f in shell.elem_faces[tag]:
        fc = coords[e][conn[f]]
        n = np.cross(fc[1] - fc[0], fc[2] - fc[0])
        n = n / np.linalg.norm(n)
        if np.dot(n, fc.mean(axis=0) - coords[e].mean(axis=0)) < 0:
            n = -n
        out.append(n)
    return np.asarray(out)


def face_areas(shell: _mesh.ShellMesh, tag: Face) -> np.ndarray:
    """(num_faces,) areas of the tagged faces, by splitting each quad into two triangles."""
    conn = np.asarray(shell.mesh.elem_template.face_connectivity)
    coords = np.asarray(shell.mesh.elem_node_coords)
    out = []
    for e, f in shell.elem_faces[tag]:
        q = coords[e][conn[f]]
        a = 0.5 * np.linalg.norm(np.cross(q[1] - q[0], q[2] - q[0]))
        b = 0.5 * np.linalg.norm(np.cross(q[2] - q[0], q[3] - q[0]))
        out.append(a + b)
    return np.asarray(out)


@dataclasses.dataclass
class FlowBC:
    """Assembled flow constraints plus the bookkeeping needed to audit them."""

    bc: dict
    inlet_nodes: np.ndarray
    inlet_velocity: np.ndarray  # (num_inlet_nodes, 3) as actually imposed
    outlet_nodes: np.ndarray
    wall_nodes: np.ndarray
    symmetry_nodes: np.ndarray
    edge_rule: EdgeRule


def build_flow_bc(
    shell: _mesh.ShellMesh,
    inlet_speed: float,
    edge_rule: EdgeRule = EdgeRule.INLET_WINS,
    symmetry_component: int | None = 2,
) -> FlowBC:
    """Flow constraints for a shell mesh.

    Args:
      shell: the mesh, already face-tagged.
      inlet_speed: the magnitude of the prescribed normal inflow, m/s. What it
        means physically depends on `edge_rule`: under INLET_WINS it is the
        uniform normal speed over the whole inlet face, so the section average
        equals it; under WALL_WINS it is the plateau value away from the rim and
        the section average is lower. `inlet_flux_report` gives the number that
        was actually imposed.
      edge_rule: how inlet-rim nodes shared with a wall are resolved.
      symmetry_component: velocity component held at zero on symmetry faces, or
        None if the mesh has no symmetry plane. The cylinder's symmetry plane is
        z = const with n = e_z, so only u_z is constrained -- the two tangential
        components stay free, as a symmetry condition requires.
    """
    mesh = shell.mesh
    dim = mesh.num_dim
    dof_per_node = mesh.nodes.dof_per_node
    if dof_per_node != dim + 1:
        raise ValueError(f"expected {dim + 1} flow dofs per node, got {dof_per_node}")

    wall = face_nodes(shell, Face.WALL)
    inlet = face_nodes(shell, Face.INLET)
    outlet = face_nodes(shell, Face.OUTLET)
    symmetry = face_nodes(shell, Face.SYMMETRY)

    # Node-averaged inward velocity on the inlet, from the face normals.
    normals = face_outward_normals(shell, Face.INLET)
    conn = np.asarray(mesh.elem_template.face_connectivity)
    elem_nodes = np.asarray(mesh.elem_nodes)
    acc = np.zeros((mesh.num_nodes, dim))
    cnt = np.zeros(mesh.num_nodes)
    for (e, f), n in zip(shell.elem_faces[Face.INLET], normals):
        for nd in elem_nodes[e][conn[f]]:
            acc[nd] += -inlet_speed * n  # inflow is opposite the outward normal
            cnt[nd] += 1
    inlet_vel = np.zeros((mesh.num_nodes, dim))
    hit = cnt > 0
    inlet_vel[hit] = acc[hit] / cnt[hit, None]

    # Constraints, applied lowest priority first so later writes win.
    values = {}  # global dof -> value

    def put(nodes, comps, vals):
        for nd, v in zip(nodes, vals):
            for c in comps:
                values[dof_per_node * nd + 1 + c] = v[c]

    if symmetry_component is not None and symmetry.size:
        put(symmetry, [symmetry_component], np.zeros((len(symmetry), dim)))

    if edge_rule is EdgeRule.INLET_WINS:
        put(wall, range(dim), np.zeros((len(wall), dim)))
        put(inlet, range(dim), inlet_vel[inlet])
    else:
        put(inlet, range(dim), inlet_vel[inlet])
        put(wall, range(dim), np.zeros((len(wall), dim)))

    # The outlet carries no essential constraint at all: zero external pressure
    # traction is the natural condition, and pinning p there would change the
    # boundary-value problem (see the module docstring).

    fixed = np.array(sorted(values), dtype=int)
    bc = _bc.BCDict(
        elem_forces=jnp.zeros((mesh.num_elems, mesh.num_dofs_per_elem)),
        fixed_dofs=fixed,
        free_dofs=np.setdiff1d(np.arange(mesh.num_dofs), fixed),
        dirichlet_values=np.array([values[d] for d in fixed]),
    )

    imposed = np.zeros((mesh.num_nodes, dim))
    for nd in inlet:
        for c in range(dim):
            key = dof_per_node * nd + 1 + c
            imposed[nd, c] = values.get(key, 0.0)

    return FlowBC(
        bc=bc,
        inlet_nodes=inlet,
        inlet_velocity=imposed[inlet],
        outlet_nodes=outlet,
        wall_nodes=wall,
        symmetry_nodes=symmetry,
        edge_rule=edge_rule,
    )


def build_thermal_bc(shell: _mesh.ShellMesh, inlet_temperature: float) -> dict:
    """Temperature constraints: Dirichlet on the inlet, natural everywhere else.

    Adiabatic walls, zero conductive flux at the outlet and zero normal flux on
    a symmetry plane are all the natural condition for this weak form, so they
    need no constraint. Zero CONDUCTIVE flux at the outlet does not stop
    advection carrying enthalpy out.
    """
    mesh = shell.mesh
    if mesh.nodes.dof_per_node != 1:
        raise ValueError("thermal mesh must have one dof per node")
    inlet = face_nodes(shell, Face.INLET)
    return _bc.BCDict(
        elem_forces=jnp.zeros((mesh.num_elems, mesh.num_dofs_per_elem)),
        fixed_dofs=inlet,
        free_dofs=np.setdiff1d(np.arange(mesh.num_dofs), inlet),
        dirichlet_values=np.full(len(inlet), inlet_temperature),
    )


def surface_flux(shell: _mesh.ShellMesh, tag: Face, nodal_velocity: np.ndarray) -> float:
    """integral u.n dA over the tagged faces, from the discrete nodal field.

    Bilinear on each face, integrated with the same 2x2 rule used elsewhere, so
    this is the flux the discretisation actually carries -- not speed * area.
    """
    conn = np.asarray(shell.mesh.elem_template.face_connectivity)
    elem_nodes = np.asarray(shell.mesh.elem_nodes)
    coords = np.asarray(shell.mesh.elem_node_coords)
    normals = face_outward_normals(shell, tag)
    areas = face_areas(shell, tag)

    total = 0.0
    for ((e, f), n, a) in zip(shell.elem_faces[tag], normals, areas):
        nodes = elem_nodes[e][conn[f]]
        # mean of the four corner values = the bilinear face average
        u_mean = nodal_velocity[nodes].mean(axis=0)
        total += float(np.dot(u_mean, n)) * a
    return total


def inlet_flux_report(
    shell: _mesh.ShellMesh, flow_bc: FlowBC, inlet_speed: float
) -> dict:
    """What mass flow the discrete inlet constraint actually imposes.

    `nominal` is speed * area, the number one would quote from the paper;
    `imposed` is the integral of the discrete boundary field. The ratio is the
    figure that must never be papered over by rescaling the inlet speed.
    """
    dim = shell.mesh.num_dim
    nodal = np.zeros((shell.mesh.num_nodes, dim))
    nodal[flow_bc.inlet_nodes] = flow_bc.inlet_velocity
    area = float(face_areas(shell, Face.INLET).sum())
    imposed = -surface_flux(shell, Face.INLET, nodal)  # inflow is positive
    return {
        "edge_rule": flow_bc.edge_rule.value,
        "inlet_area_m2": area,
        "nominal_flux_m3_s": inlet_speed * area,
        "imposed_flux_m3_s": imposed,
        "ratio_imposed_over_nominal": imposed / (inlet_speed * area),
        "num_inlet_nodes": int(len(flow_bc.inlet_nodes)),
    }
