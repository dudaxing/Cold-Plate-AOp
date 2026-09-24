"""Dimension-agnostic stabilised Navier-Stokes-Brinkman, in SI units.

Strong form, Zhou equation 14:

    rho (u . grad) u - mu div(grad u + grad u^T) + grad p + alpha(s) u = 0
    div u = 0

Why this is not upstream's `fe_fluid.FluidSolver`:

*   Upstream packs the element residual with `res_mom[0::2]` / `res_mom[1::2]`,
    which hard-codes two velocity components. In 3D it tries to column_stack
    arrays of length 8 and 12 and raises.
*   Upstream's two SUPG momentum terms contract the wrong indices (its
    `res_brink_stab` and `res_conv_stab`; its `res_press_stab` and the whole
    PSPG block are correct). Writing the stabilisation as
    `tau * (u.grad N) * r_i` against a strong residual `r` computed once, as
    below, makes that class of mistake unrepresentable.
*   Upstream's `tau_1` omits the sum over nodes when evaluating the element
    centroid velocity, so `ue` is the sum of squared nodal contributions rather
    than the squared centroid speed -- a factor of 4 low for a uniform flow on
    Q1/Hex8.

Zhou's stabilisation parameter (equation 20) is algebraically identical to
upstream's tau, so the tau_1 defect applies to these cases directly:

    tau_u = [ (2|u|/h)^2 + (12 mu/(rho h^2))^2 + (alpha/rho)^2 ]^(-1/2)

`h` is the element characteristic length. Zhou names it but does not define it;
this implementation uses `Hex8.diag_length`, the mean body diagonal, which
follows upstream's 2D convention of averaging the element diagonals. For a cube
of side a that is a*sqrt(3). Recorded as a reproduction choice.

Degrees of freedom are ordered (p, u, v, w) per node, matching upstream.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

import toflux.src.solver as _solver

# Replaces upstream's solve with one whose host callback never calls JAX; the
# original can deadlock an eager backward pass. See tfopus/_callback_solve.py.
from tfopus import _callback_solve  # noqa: F401
from tfopus import elements as _elements
from tfopus import sparse_bc as _sparse_bc


@dataclasses.dataclass(frozen=True)
class FlowForm:
    """Which printed formulation the momentum weak form follows.

    Zhou and Zhao write the same strong form but different discretisations, and
    the differences are not cosmetic in a Brinkman-penalised domain. Each field
    below names the paper equation it comes from, so a case can state its
    formulation instead of inheriting whichever one happened to be coded first.

    Attributes:
      brinkman_in_tau: include the reactive limit (alpha/rho)^2 in tau.
        Zhou equation 20 has it; Zhao equation 16 has only the convective and
        diffusive limits. In a solid cell alpha/rho dominates by orders of
        magnitude, so this sets tau there almost by itself.
      brinkman_in_supg_residual: include alpha*u in the strong residual that
        SUPG and PSPG weight. Zhou's does; Zhao's S_uu/S_up/S_pu/S_pp
        (equation 15) weight only rho(u.grad)u + grad p.
      viscous_form: "symmetric" uses mu grad v : (grad u + grad u^T), which is
        what the traction-based outlet in `bc3d` is derived from and what the
        dissipation integral (Zhou 25 / Zhao 24) measures. "laplacian" uses
        mu grad v : grad u, which is Zhao equation 13 as printed. For a
        divergence-free field with natural boundary conditions the two have the
        same continuous solution; discretely, and at the outlet traction, they
        do not. Switching to "laplacian" does NOT change the dissipation
        functional, which always uses the symmetric form.
    """

    brinkman_in_tau: bool = True
    brinkman_in_supg_residual: bool = True
    viscous_form: str = "symmetric"

    def __post_init__(self):
        if self.viscous_form not in ("symmetric", "laplacian"):
            raise ValueError(f"unknown viscous_form {self.viscous_form!r}")


ZHOU_FORM = FlowForm()
ZHAO_FORM = FlowForm(
    brinkman_in_tau=False,  # Zhao eq 16
    brinkman_in_supg_residual=False,  # Zhao eq 15
    viscous_form="laplacian",  # Zhao eq 13
)


class FlowSolver(_solver.NonlinearProblem):
    """Stabilised NS-Brinkman on a 2D or 3D mesh, taking element-wise alpha."""

    def __init__(
        self,
        mesh,
        bc,
        density: float,
        viscosity: float,
        solver_settings,
        elem_length=None,
        tau_scale: float = 1.0,
        form: FlowForm = ZHOU_FORM,
    ):
        """
        Args:
          elem_length: (num_elems,) characteristic length h_e for the
            stabilisation. Defaults to the mesh's mean body diagonal. Zhou names
            h_e but never defines it, and the choice is NOT cosmetic on
            anisotropic elements: the body diagonal is dominated by the longest
            edge, while the flow gradient may live across the shortest, which
            inflates the diffusive limit tau_3 = rho h^2 / (12 mu).
          tau_scale: diagnostic multiplier on tau. 1.0 for production; smaller
            values isolate how much of a given error is stabilisation.
          form: which paper's discretisation to follow. See `FlowForm`.
        """
        super().__init__(solver_settings=solver_settings)
        self.form = form
        self.elem_length = (
            _elements.element_lengths(mesh, "min_edge")
            if elem_length is None
            else elem_length
        )
        self.tau_scale = tau_scale
        self.mesh = mesh
        self.bc = bc
        self.density = density
        self.viscosity = viscosity
        self.dim = mesh.num_dim
        self.num_fields = self.dim + 1
        self.node_id_jac = np.stack((mesh.iK, mesh.jK)).astype(np.int32).T
        self.shp_fn = jax.vmap(mesh.elem_template.shape_functions)(mesh.gauss_pts)
        self.shp_centre = mesh.elem_template.shape_functions(jnp.zeros(self.dim))

    # -- pieces -------------------------------------------------------------

    def _split(self, press_vel):
        """(num_dofs_per_elem,) -> (pressure per node, velocity (n, dim))."""
        block = press_vel.reshape(-1, self.num_fields)
        return block[:, 0], block[:, 1:]

    def _tau(self, vel_nodes, alpha, h):
        """Zhou equation 20, or Zhao equation 16 without the reactive limit.

        `vel_nodes` is (nodes_per_elem, dim).
        """
        u_centre = jnp.einsum("n, nd -> d", self.shp_centre, vel_nodes)
        speed_sq = jnp.einsum("d, d -> ", u_centre, u_centre)
        inv_tau_conv_sq = 4.0 * speed_sq / h**2
        inv_tau_diff = 12.0 * self.viscosity / (self.density * h**2)
        inv_sq = inv_tau_conv_sq + inv_tau_diff**2
        if self.form.brinkman_in_tau:
            inv_sq = inv_sq + (alpha / self.density) ** 2
        return self.tau_scale * inv_sq ** (-0.5)

    def element_residual(self, press_vel, alpha, node_coords, h):
        """Element residual, ordered (p, u, v[, w]) per node."""
        pressure, vel_nodes = self._split(press_vel)

        grad_n = jax.vmap(
            self.mesh.elem_template.get_gradient_shape_function_physical,
            in_axes=(0, None),
        )(self.mesh.gauss_pts, node_coords)  # (g, n, d)
        _, det_j = jax.vmap(
            self.mesh.elem_template.compute_jacobian_and_determinant, in_axes=(0, None)
        )(self.mesh.gauss_pts, node_coords)  # (g,)
        w = self.mesh.gauss_weights * det_j  # (g,) integration weight

        tau = self._tau(vel_nodes, alpha, h)

        u_g = jnp.einsum("gn, nd -> gd", self.shp_fn, vel_nodes)  # (g, d)
        p_g = jnp.einsum("gn, n -> g", self.shp_fn, pressure)  # (g,)
        grad_u = jnp.einsum("gnd, ni -> gid", grad_n, vel_nodes)  # d u_i / d x_d
        grad_p = jnp.einsum("gnd, n -> gd", grad_n, pressure)  # (g, d)

        # TRUNCATED strong-form momentum residual, component i: the viscous
        # part -div(2 mu eps(u)) is not included, matching the thermal solver's
        # treatment (see its module docstring for why that is an approximation
        # to be verified, not an identity). The Galerkin viscous term above is
        # complete. Computed once and reused by both SUPG and PSPG, so the two
        # cannot disagree.
        conv = self.density * jnp.einsum("gd, gid -> gi", u_g, grad_u)
        r_mom = conv + grad_p  # (g, i)
        if self.form.brinkman_in_supg_residual:
            r_mom = r_mom + alpha * u_g

        # The contraction below is einsum("gnd, gdi -> gni"), i.e. it sums
        # dN_n/dx_d against index d of `viscous_grad`. grad_u is stored as
        # [g, i, d] = du_i/dx_d, so:
        #   symmetric  mu dN/dx_d (du_d/dx_i + du_i/dx_d) -> the symmetric sum,
        #              which is index-order agnostic;
        #   laplacian  mu dN/dx_d du_i/dx_d               -> needs [g, d, i],
        #              so grad_u must be transposed. Passing grad_u unswapped
        #              would give dN/dx_d du_d/dx_i, the OTHER half of the
        #              symmetric form, which is not the Laplacian.
        if self.form.viscous_form == "symmetric":
            viscous_grad = grad_u + jnp.swapaxes(grad_u, 1, 2)
        else:  # "laplacian", Zhao eq 13
            viscous_grad = jnp.swapaxes(grad_u, 1, 2)

        # Galerkin momentum
        mom = (
            jnp.einsum("gn, gi -> gni", self.shp_fn, conv)
            + self.viscosity * jnp.einsum("gnd, gdi -> gni", grad_n, viscous_grad)
            - jnp.einsum("gni, g -> gni", grad_n, p_g)
            + alpha * jnp.einsum("gn, gi -> gni", self.shp_fn, u_g)
        )
        # SUPG: tau * (u . grad N_n) * r_i
        u_dot_grad_n = jnp.einsum("gd, gnd -> gn", u_g, grad_n)  # (g, n)
        mom = mom + tau * jnp.einsum("gn, gi -> gni", u_dot_grad_n, r_mom)

        # Galerkin continuity + PSPG: (tau/rho) * grad N_n . r
        div_u = jnp.einsum("gdd -> g", grad_u)
        cont = jnp.einsum("gn, g -> gn", self.shp_fn, div_u) + (
            tau / self.density
        ) * jnp.einsum("gni, gi -> gn", grad_n, r_mom)

        res_mom = jnp.einsum("gni, g -> ni", mom, w)  # (n, dim)
        res_cont = jnp.einsum("gn, g -> n", cont, w)  # (n,)
        return jnp.concatenate([res_cont[:, None], res_mom], axis=1).ravel()

    # -- assembly -----------------------------------------------------------

    def get_residual_and_tangent_stiffness(self, press_vel, alpha):
        args = (
            press_vel[self.mesh.elem_dof_mat],
            alpha,
            self.mesh.elem_node_coords,
            self.elem_length,
        )
        elem_res = jax.vmap(self.element_residual)(*args)
        residual = jnp.zeros((self.mesh.num_dofs,))
        residual = residual.at[self.mesh.elem_dof_mat].add(elem_res)
        residual = residual.at[self.bc["fixed_dofs"]].set(0.0)

        elem_jac = jax.vmap(jax.jacfwd(self.element_residual, argnums=0))(*args)
        assembled = jsparse.BCOO(
            (elem_jac.flatten(), self.node_id_jac),
            shape=(self.mesh.num_dofs, self.mesh.num_dofs),
        ).T
        return residual, _sparse_bc.apply_dirichlet_bc(
            assembled, self.bc["fixed_dofs"]
        )

    # -- diagnostics --------------------------------------------------------

    def element_dissipation(self, press_vel, alpha, node_coords):
        """Zhou equation 25, per element.

            Psi = integral [ mu/2 (grad u + grad u^T):(grad u + grad u^T)
                             + alpha u.u ] dV

        Written out rather than reusing upstream's `compute_elem_dissipated_power`,
        which evaluates half of this. Deriving Psi by doubling upstream would be
        correct only if the reference Psi_0 were doubled in the same step; keeping
        one definition removes that trap.
        """
        _, vel_nodes = self._split(press_vel)
        grad_n = jax.vmap(
            self.mesh.elem_template.get_gradient_shape_function_physical,
            in_axes=(0, None),
        )(self.mesh.gauss_pts, node_coords)
        _, det_j = jax.vmap(
            self.mesh.elem_template.compute_jacobian_and_determinant, in_axes=(0, None)
        )(self.mesh.gauss_pts, node_coords)

        u_g = jnp.einsum("gn, nd -> gd", self.shp_fn, vel_nodes)
        grad_u = jnp.einsum("gnd, ni -> gid", grad_n, vel_nodes)
        strain2 = grad_u + jnp.swapaxes(grad_u, 1, 2)

        integrand = 0.5 * self.viscosity * jnp.einsum(
            "gid, gid -> g", strain2, strain2
        ) + alpha * jnp.einsum("gd, gd -> g", u_g, u_g)
        return jnp.einsum("g, g, g -> ", integrand, self.mesh.gauss_weights, det_j)

    def dissipated_power(self, press_vel, alpha):
        """Total Psi over the mesh."""
        return jnp.sum(
            jax.vmap(self.element_dissipation)(
                press_vel[self.mesh.elem_dof_mat], alpha, self.mesh.elem_node_coords
            )
        )

    def element_velocities(self, press_vel):
        """(num_elems, nodes_per_elem * dim) nodal velocities, for the thermal solve."""
        block = press_vel[self.mesh.elem_dof_mat].reshape(
            self.mesh.num_elems, -1, self.num_fields
        )
        return block[:, :, 1:].reshape(self.mesh.num_elems, -1)
