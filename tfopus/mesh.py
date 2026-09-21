"""Swept thin-shell meshes for the Zhou conformal-cooling cases.

Both geometries are the same object: a structured (n1 x n2 x n_thick) grid in a
parametric box, pushed through an analytic map into 3D. The map only builds a
fixed mesh -- it carries no design freedom, which is what separates this from
the feature-driven parametrisation the cases originally used.

    cylinder sector   X(theta, z, zeta) = ((R + t*zeta) cos th, (R + t*zeta) sin th, z)
    spherical cap     X(a, b, zeta)     = (R + t*zeta) * normalize(a, b, 1)

The sphere uses a gnomonic (cubed-sphere) patch rather than a (polar, azimuth)
parametrisation on purpose: the polar chart collapses an entire parametric edge
onto the pole, producing zero-volume elements. The gnomonic patch is
non-degenerate everywhere and still flattens to a 2D parametric square, which is
the setting Zhou's design description assumes.

Region tags follow Zhou section 4.1: the heat source and the design freedom live
in the design domain only; the inlet/outlet fluid domains are non-designable.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Callable

import numpy as np
import jax.numpy as jnp

import toflux.src.mesher as _mesher

from tfopus import elements as _elements


class Region(enum.IntEnum):
    """Element regions. Only DESIGN carries design variables and the heat source."""

    DESIGN = 0
    PORT_FLUID = 1  # non-design inlet/outlet channel: s = 0, Q = 0
    PORT_SOLID = 2  # non-design surround that walls the ports in: s = 1, Q = 0


class Face(enum.IntEnum):
    """Boundary face tags."""

    NONE = 0
    INLET = 1
    OUTLET = 2
    SYMMETRY = 3
    WALL = 4


# Reference-cube node coordinates, in the Hex8 isoparametric order.
_REF_NODES = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=float,
)


@dataclasses.dataclass
class ShellMesh:
    """A swept shell mesh plus everything the case driver needs to address it.

    Attributes:
      mesh: upstream `toflux` Mesh carrying the corrected Hex8 template.
      n1, n2, n_thick: element counts along the two surface directions and the
        wall thickness.
      surface_id: (num_elems,) index of the surface column each element sits in.
        Elements sharing a column share one design variable.
      region: (num_elems,) `Region` tag.
      elem_faces: dict mapping `Face` -> list of (elem, local_face) pairs.
      surface_centres: (num_surface_cells, 3) physical mid-surface centre of each
        column, used as the metric for the density filter.
      column_volume: (num_surface_cells,) total volume of each column.
    """

    mesh: _mesher.Mesh
    n1: int
    n2: int
    n_thick: int
    surface_id: np.ndarray
    region: np.ndarray
    elem_faces: dict
    surface_centres: np.ndarray
    column_volume: np.ndarray

    @property
    def num_surface_cells(self) -> int:
        return self.n1 * self.n2

    @property
    def num_elems(self) -> int:
        return self.mesh.num_elems

    def face_count(self) -> dict:
        return {f.name: len(v) for f, v in self.elem_faces.items() if v}


def structured_hex_connectivity(n1: int, n2: int, n3: int) -> np.ndarray:
    """Hex8 connectivity for an (n1, n2, n3) element grid.

    Node index is i*(n2+1)*(n3+1) + j*(n3+1) + k. Node order follows the Hex8
    isoparametric convention: 0-3 the zeta-minus face, 4-7 the zeta-plus face.
    """
    s2, s3 = n2 + 1, n3 + 1

    def nid(i, j, k):
        return (i * s2 + j) * s3 + k

    i, j, k = np.meshgrid(
        np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij"
    )
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    return np.stack(
        [
            nid(i, j, k),
            nid(i + 1, j, k),
            nid(i + 1, j + 1, k),
            nid(i, j + 1, k),
            nid(i, j, k + 1),
            nid(i + 1, j, k + 1),
            nid(i + 1, j + 1, k + 1),
            nid(i, j + 1, k + 1),
        ],
        axis=1,
    )


def _face_side_table(template) -> dict:
    """Local face index -> (parametric axis, side) it lies on.

    Read off the template rather than hard-coded, so the mapping stays honest if
    upstream reorders `face_connectivity`. axis 0,1,2 = (p1, p2, zeta);
    side 0 = minus, 1 = plus.
    """
    table = {}
    for f, nodes in enumerate(np.asarray(template.face_connectivity)):
        coords = _REF_NODES[nodes]
        for axis in range(3):
            col = coords[:, axis]
            if np.allclose(col, col[0]):
                table[f] = (axis, 0 if col[0] < 0 else 1)
                break
        else:  # pragma: no cover - a Hex8 face is always axis-constant
            raise RuntimeError("face %d is not constant along any axis" % f)
    return table


def build_shell_mesh(
    param_map: Callable[..., np.ndarray],
    p1: np.ndarray,
    p2: np.ndarray,
    n_thick: int,
    dofs_per_node: int,
    region_fn: Callable[..., np.ndarray],
    face_fn: Callable[..., Face],
    gauss_order: int = 2,
) -> ShellMesh:
    """Build a swept shell mesh from an analytic parametric map.

    Args:
      param_map: (p1, p2, zeta) flat arrays -> (N, 3) physical coordinates.
      p1, p2: node coordinate vectors along the two surface directions.
      n_thick: number of element layers through the wall.
      dofs_per_node: 4 for the flow system (p, u, v, w), 1 for temperature.
      region_fn: (p1_centre, p2_centre) flat arrays -> `Region` array, one entry
        per surface column.
      face_fn: (axis, side, phys_centre, param_centre) -> `Face` tag for one
        boundary face. `param_centre` is (p1, p2, zeta) at the face centre,
        which is usually the easier frame to write the tagging rule in.
      gauss_order: quadrature points per direction.

    Raises:
      ValueError: if the parametrisation is left-handed. The residual assembly
        uses the signed det(J), so a negative orientation would silently flip
        the sign of the whole system; the caller must order p1/p2 right-handed.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    n1, n2 = len(p1) - 1, len(p2) - 1
    zeta = np.linspace(0.0, 1.0, n_thick + 1)

    g1, g2, g3 = np.meshgrid(p1, p2, zeta, indexing="ij")
    coords = np.asarray(param_map(g1.ravel(), g2.ravel(), g3.ravel()), dtype=float)

    elem_nodes = structured_hex_connectivity(n1, n2, n_thick)
    template = _elements.Hex8()

    _, det = template.compute_jacobian_and_determinant(
        jnp.zeros(3), jnp.asarray(coords[elem_nodes[0]])
    )
    if float(det) <= 0:
        raise ValueError(
            "left-handed parametrisation (det J = %.3e at the first element). "
            "Swap or reverse p1/p2 so that (p1, p2, zeta) is right-handed."
            % float(det)
        )

    mesh = _mesher.Mesh(
        nodes=_mesher.Nodes(coords=jnp.asarray(coords), dof_per_node=dofs_per_node),
        elem_nodes=elem_nodes,
        elem_template=template,
        gauss_order=gauss_order,
    )

    # Surface column bookkeeping: element (i, j, k) belongs to column i*n2 + j.
    ii, jj, _ = np.meshgrid(
        np.arange(n1), np.arange(n2), np.arange(n_thick), indexing="ij"
    )
    surface_id = (ii * n2 + jj).ravel()

    c1 = 0.5 * (p1[:-1] + p1[1:])
    c2 = 0.5 * (p2[:-1] + p2[1:])
    cc1, cc2 = np.meshgrid(c1, c2, indexing="ij")
    column_region = np.asarray(region_fn(cc1.ravel(), cc2.ravel()), dtype=int)
    region = column_region[surface_id]

    surface_centres = np.asarray(
        param_map(cc1.ravel(), cc2.ravel(), np.full(cc1.size, 0.5)), dtype=float
    )
    column_volume = np.bincount(
        surface_id, weights=np.asarray(mesh.elem_volume), minlength=n1 * n2
    )

    elem_faces = _tag_boundary_faces(
        mesh, template, p1, p2, zeta, n1, n2, n_thick, face_fn
    )

    return ShellMesh(
        mesh=mesh,
        n1=n1,
        n2=n2,
        n_thick=n_thick,
        surface_id=surface_id,
        region=region,
        elem_faces=elem_faces,
        surface_centres=surface_centres,
        column_volume=column_volume,
    )


def _tag_boundary_faces(mesh, template, p1, p2, zeta, n1, n2, n3, face_fn) -> dict:
    """Hand every boundary face to `face_fn` and collect the tags."""
    side_of = _face_side_table(template)
    face_conn = np.asarray(template.face_connectivity)
    elems, faces = np.nonzero(np.asarray(mesh.boundary_faces))

    idx = np.arange(n1 * n2 * n3)
    cell_index = np.stack([idx // (n2 * n3), (idx // n3) % n2, idx % n3], axis=1)
    extent = (n1, n2, n3)
    axis_nodes = (p1, p2, zeta)

    node_coords = np.asarray(mesh.elem_node_coords)
    out = {f: [] for f in Face}
    for elem, face in zip(elems, faces):
        axis, side = side_of[int(face)]
        cell = cell_index[elem, axis]
        # A genuine boundary face sits on the outer layer of its own axis.
        if (side == 0 and cell != 0) or (side == 1 and cell != extent[axis] - 1):
            continue
        # Parametric centre: cell midpoint along the two in-face axes, and the
        # exact boundary value along the face's own axis.
        param = np.empty(3)
        for a in range(3):
            c = cell_index[elem, a]
            nodes_a = axis_nodes[a]
            if a == axis:
                param[a] = nodes_a[0] if side == 0 else nodes_a[-1]
            else:
                param[a] = 0.5 * (nodes_a[c] + nodes_a[c + 1])
        centre = node_coords[elem][face_conn[face]].mean(axis=0)
        out[face_fn(axis, side, centre, param)].append((int(elem), int(face)))
    return out


# --------------------------------------------------------------------------
# Analytic maps
# --------------------------------------------------------------------------


def cylinder_map(inner_radius: float, thickness: float):
    """X(theta, z, zeta) for a cylindrical shell sector.

    (theta, z, zeta) is right-handed: d/dtheta x d/dz points along -r_hat, and
    with zeta outward the triple product is positive.
    """

    def _map(theta, z, zeta):
        r = inner_radius + thickness * zeta
        return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)

    return _map


def spherical_cap_map(inner_radius: float, thickness: float):
    """X(alpha, beta, zeta) for an equiangular gnomonic patch of a spherical shell.

    (alpha, beta) are the two angular offsets from the patch centre, in radians,
    so the parametric box is an angle box and a uniform parametric grid has
    uniform arc length along the two centre lines. Grading in tangent space
    instead would over-refine the patch edges by (1 + tan^2) -- a factor of 4
    across a 120 degree patch.

    The direction is the gnomonic one, (tan alpha, tan beta, 1) normalised, which
    stays non-degenerate over the whole patch.
    """

    def _map(alpha, beta, zeta):
        r = inner_radius + thickness * zeta
        d = np.stack([np.tan(alpha), np.tan(beta), np.ones_like(alpha)], axis=1)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        return d * r[:, None]

    return _map
