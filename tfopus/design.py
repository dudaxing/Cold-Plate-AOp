"""Design field: surface columns -> filtered -> projected -> extruded to volume.

    x (design columns)
      -> H   surface density filter, physical distance on the mid-surface
      -> P   Heaviside projection
      -> scatter into all columns, with the non-design ports pinned
      -> E   extrude down each column
      -> s_volume

Zhou holds the pseudo-density constant through the wall thickness, which is what
E encodes. The reverse path is E^T: a column's sensitivity is the SUM over its
volume elements, not their mean -- each volume element carries its own
integration weight already.

The filter is built with a KD-tree, not a dense pairwise distance matrix.
Upstream's `create_density_filter` forms an (N, N) dense array, which for the
7412 columns here would be 440 MB and for a finer mesh far worse.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import scipy.spatial as spspatial

from tfopus import mesh as _mesh


def build_surface_filter(
    centres: np.ndarray, radius: float, kind: str = "linear"
) -> jsparse.BCOO:
    """Row-normalised density filter over surface columns, as a sparse matrix.

    Args:
      centres: (num_columns, 3) physical mid-surface centres.
      radius: cutoff, in metres. Physical length, not parametric distance.
      kind: "linear" (cone) or "constant".
    """
    tree = spspatial.cKDTree(centres)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    # symmetric neighbours plus the diagonal
    rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(len(centres))])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(len(centres))])
    dist = np.linalg.norm(centres[rows] - centres[cols], axis=1)

    if kind == "linear":
        w = np.maximum(0.0, 1.0 - dist / radius)
    elif kind == "constant":
        w = np.ones_like(dist)
    else:
        raise ValueError(f"unknown filter kind {kind!r}")

    row_sum = np.bincount(rows, weights=w, minlength=len(centres))
    w = w / row_sum[rows]
    return jsparse.BCOO(
        (jnp.asarray(w), jnp.asarray(np.stack([rows, cols], axis=1))),
        shape=(len(centres), len(centres)),
    )


def heaviside_projection(rho, beta: float, eta: float = 0.5):
    """Smoothed Heaviside, the standard tanh form. beta -> 0 is the identity."""
    if beta <= 0.0:
        return rho
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta))
    den = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
    return num / den


@dataclasses.dataclass
class DesignMap:
    """Design vector -> volume solid fraction, for one shell mesh.

    Attributes:
      shell: the mesh this map belongs to.
      design_columns: indices of the columns that carry design variables.
      pinned: (num_columns,) solid fraction for the non-design columns; NaN on
        design columns.
      filter_matrix: row-normalised surface filter, over ALL columns.
      num_design: number of design variables.
    """

    shell: _mesh.ShellMesh
    design_columns: np.ndarray
    pinned: np.ndarray
    filter_matrix: jsparse.BCOO

    @property
    def num_design(self) -> int:
        return len(self.design_columns)

    @property
    def design_volume(self) -> np.ndarray:
        """Column volumes of the design columns, for the volume constraint."""
        return self.shell.column_volume[self.design_columns]

    def to_volume(self, x, beta: float = 0.0):
        """Design vector -> (num_elems,) solid fraction on the volume mesh."""
        return self.to_columns(x, beta)[self.shell.surface_id]

    def to_columns(self, x, beta: float = 0.0):
        """Design vector -> (num_columns,) solid fraction, ports pinned."""
        full = jnp.asarray(self.pinned)
        scattered = full.at[self.design_columns].set(x)
        filtered = self.filter_matrix @ scattered
        projected = heaviside_projection(filtered, beta)
        # Re-pin after projection: the ports are fixed boundary data, and the
        # filter would otherwise bleed design values into them.
        return projected.at[self.non_design_columns].set(
            jnp.asarray(self.pinned)[self.non_design_columns]
        )

    @property
    def non_design_columns(self) -> np.ndarray:
        mask = np.ones(len(self.pinned), dtype=bool)
        mask[self.design_columns] = False
        return np.nonzero(mask)[0]

    def fluid_fraction(self, x, beta: float = 0.0):
        """Zhou equation 27: volume-weighted fluid fraction over the design domain.

        Zhou writes v_f = (1/|Omega|) integral gamma dOmega without stating
        whether Omega includes the non-design ports. Restricting it to the design
        domain is this reproduction's convention, recorded here; the port volume
        is reported separately by the case driver.
        """
        s_cols = self.to_columns(x, beta)[self.design_columns]
        vol = jnp.asarray(self.design_volume)
        return jnp.sum((1.0 - s_cols) * vol) / jnp.sum(vol)


def build_design_map(
    shell: _mesh.ShellMesh, filter_radius: float, filter_kind: str = "linear"
) -> DesignMap:
    """Design map for a shell mesh: design columns free, port columns pinned."""
    n_cols = shell.num_surface_cells
    column_region = np.empty(n_cols, dtype=int)
    column_region[shell.surface_id] = shell.region  # region is column-constant

    design_columns = np.nonzero(column_region == _mesh.Region.DESIGN)[0]
    pinned = np.zeros(n_cols)
    pinned[column_region == _mesh.Region.PORT_FLUID] = 0.0  # fluid
    pinned[column_region == _mesh.Region.PORT_SOLID] = 1.0  # solid

    return DesignMap(
        shell=shell,
        design_columns=design_columns,
        pinned=pinned,
        filter_matrix=build_surface_filter(
            shell.surface_centres, filter_radius, filter_kind
        ),
    )


def extrusion_transpose(shell: _mesh.ShellMesh, g_volume):
    """E^T: accumulate volume-element sensitivities onto their surface column.

    A sum, never a mean: the volume gradients already carry their own
    integration weights, so averaging over the layers would divide the
    sensitivity by the layer count.
    """
    return jnp.zeros(shell.num_surface_cells).at[shell.surface_id].add(g_volume)


def heat_source_field(shell: _mesh.ShellMesh, q0: float) -> jnp.ndarray:
    """(num_elems,) volumetric heat source: q0 in the design domain, 0 in the ports.

    Zhou applies the source over the fixed design domain, independent of the
    density, so it must NOT be multiplied by s -- that would make the total heat
    load shrink as the optimiser adds fluid.
    """
    return jnp.where(
        jnp.asarray(shell.region) == int(_mesh.Region.DESIGN), q0, 0.0
    )
