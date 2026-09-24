"""Corrected isoparametric elements.

Upstream TOFLUX's `element.py` has three defects that only surface on meshes
that are not axis-aligned boxes. Its own 2D work never hits them, because both
`GridMesh` and `grid_mesh_brep` emit axis-aligned rectangles, where the Jacobian
is diagonal and the error vanishes identically. Swept cylindrical and spherical
meshes are not axis-aligned, so all three must be fixed before any curved-shell
result means anything.

1.  `get_gradient_shape_function_physical` contracts the wrong index of the
    inverse Jacobian, in BOTH `Quad4` and `Hex8`.

    With `jac[d, i] = dx_i/dxi_d` (upstream's convention), the chain rule gives

        dN_n/dx_i = sum_d dN_n/dxi_d * dxi_d/dx_i,   dxi_d/dx_i = inv(jac)[i, d]

    Upstream writes `einsum("di, nd -> ni", inv(jac), gradN)`, i.e. it uses
    `inv(jac)[d, i]` -- the transpose. Correct is `einsum("id, nd -> ni", ...)`.
    A linear-field patch test on a sheared element fails by 0.16 (Quad4) and
    0.35 (Hex8); with the fix both drop to ~1e-16.

2.  `Hex8.elem_volume` sums six tetrahedra whose vertex sets are
    (0,1,3,4) (1,2,3,6) (1,4,5,6) (3,4,6,7) (1,3,4,6) (1,4,5,6)
    -- (1,4,5,6) appears twice, so a unit cube integrates to 7/6. Replaced by
    Gauss quadrature of |det J|, which is also what the residuals already use
    and is correct for trilinearly distorted elements.

3.  `Hex8.get_gradient_shape_function_physical` is missing `@classmethod`
    (`Quad4`'s has it). It happens to work when called on an instance, which is
    how `fe_fluid` calls it, and raises when called on the class.

`Hex8` additionally has no `diag_length`, which `mesher.Mesh.__post_init__`
requires, so it is added here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import toflux.src.element as _upstream
import toflux.src.utils as _utils


def _gradient_shape_function_physical(cls, gauss_pt, node_coords):
    """dN_n/dx_i at one isoparametric point. See defect 1 above."""
    grad_iso = cls.shape_function_gradients_isoparametric(gauss_pt)  # (n, d)
    jac, _ = cls.compute_jacobian_and_determinant(gauss_pt, node_coords)
    return jnp.einsum("id, nd -> ni", jnp.linalg.inv(jac), grad_iso)


def _gauss_volume(cls, node_coords, order: int = 2) -> jax.Array:
    """Volume as the Gauss quadrature of |det J|."""
    pts, wts = _utils.gauss_integ_points_weights(order=order, dimension=cls.dimension)
    _, det = jax.vmap(cls.compute_jacobian_and_determinant, in_axes=(0, None))(
        pts, node_coords
    )
    return jnp.einsum("g, g -> ", wts, jnp.abs(det))


class Quad4(_upstream.Quad4):
    """Upstream Quad4 with the inverse-Jacobian contraction corrected."""

    @classmethod
    def get_gradient_shape_function_physical(cls, gauss_pt, node_coords):
        return _gradient_shape_function_physical(cls, gauss_pt, node_coords)


class Hex8(_upstream.Hex8):
    """Upstream Hex8 with the gradient, the volume and `diag_length` corrected."""

    @property
    def dimension(self) -> int:
        return 3

    @classmethod
    def get_gradient_shape_function_physical(cls, gauss_pt, node_coords):
        return _gradient_shape_function_physical(cls, gauss_pt, node_coords)

    @classmethod
    def elem_volume(cls, node_coords: jax.Array) -> jax.Array:
        return _gauss_volume(cls(), node_coords)

    @staticmethod
    def diag_length(node_coords: jax.Array) -> jax.Array:
        """Mean of the four body diagonals.

        This is the `h_e` that feeds the stabilisation parameters, and it
        follows upstream Quad4's convention of averaging the diagonals rather
        than taking a single one. Node order is 0-3 bottom face, 4-7 top face,
        so the body diagonals pair node k with node ((k + 2) % 4) + 4.
        """
        pairs = jnp.array([[0, 6], [1, 7], [2, 4], [3, 5]])
        d = node_coords[pairs[:, 0], :] - node_coords[pairs[:, 1], :]
        return jnp.mean(jnp.linalg.norm(d, axis=1))


def min_edge_length(node_coords: jax.Array, template=None) -> jax.Array:
    """Shortest edge of one element.

    This is the characteristic length `h_e` that the stabilisation parameters
    use. Zhou names h_e but never defines it, and the choice is not cosmetic:
    tau_3 = rho h^2 / (12 mu) and its thermal twin 4k/(b_f h^2) both scale as
    h^2, so an h that is dominated by the LONGEST edge over-stabilises whenever
    the flow gradient lives across the shortest one -- which is exactly the
    situation in a thin swept shell.

    Measured on a plane-Poiseuille duct (see validation/test_stabilisation.py):
    switching from the mean body diagonal to the shortest edge cuts the velocity
    error by 36x at Re = 0.15, 18x at Re = 150 and 10x at Re = 1500, restores
    second-order convergence in the diffusive regime, and removes a solution
    blow-up that the diagonal produces at Re = 1500.
    """
    tmpl = template or Hex8()
    edges = jnp.asarray(tmpl.edge_connectivity)
    d = node_coords[edges[:, 0], :] - node_coords[edges[:, 1], :]
    return jnp.min(jnp.linalg.norm(d, axis=1))


def element_lengths(mesh, mode: str = "min_edge") -> jax.Array:
    """(num_elems,) characteristic lengths for the stabilisation parameters.

    Modes: "min_edge" (default, see above), "diagonal" (upstream's mean body
    diagonal, kept so the comparison stays runnable), "cube_root_volume".
    """
    if mode == "min_edge":
        return jax.vmap(lambda c: min_edge_length(c, mesh.elem_template))(
            mesh.elem_node_coords
        )
    if mode == "diagonal":
        return mesh.elem_diag_length
    if mode == "cube_root_volume":
        return jnp.asarray(mesh.elem_volume) ** (1.0 / mesh.num_dim)
    raise ValueError(f"unknown element-length mode {mode!r}")
