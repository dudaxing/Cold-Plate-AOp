"""Stabilised convection-conduction with a volumetric source, in SI units.

Strong form, Zhou equation 16, with the advective heat capacity held at the
fluid value over the whole domain (see `materials.ThermoFluidMaterial`):

    b_f (u . grad T) - div(k(s) grad T) = Q,    b_f = (rho c_p)_fluid

**On the printed stabilisation.** Zhou equation 22 prints the thermal SUPG block
as

    S_T = sum_e integral tau_T (u.grad N)^T (u.grad N) dV

with tau_T from equation 23 carrying units of seconds. Against the printed
advection and conduction blocks, which are both in W, that term evaluates to
K m^3 / s: it is short exactly one factor of volumetric heat capacity. It cannot
be added to a dimensional system as printed. Zhou also does not print the
assembly of the source vector Q, so the printed form does not establish that the
authors omitted a stabilised source term either.

This implementation therefore uses the residual form, which is dimensionally
consistent throughout. With the element strong residual

    r_T = b_f (u . grad T) - Q

the weighted residual for a test function v is

    R_T(v) = integral [ b_f v (u.grad T) + k grad v . grad T - v Q ] dV
           + sum_e integral tau_T (u . grad v) r_T dV

which contributes the advective stabilisation matrix and the stabilised source
load that Zhou's printed equation 22 lacks:

    S_T,adv = sum_e integral b_f tau_T (u.grad N)^T (u.grad N) dV
    F_T,SUPG(v) = sum_e integral tau_T (u.grad v) Q dV

**This is a TRUNCATED residual, not the full one.** The conduction part of the
strong residual, -div(k grad T), is dropped from r_T. The Galerkin part keeps
the full conduction term, so no physics is removed from the state equation, but
the stabilisation is then built on a truncated operator and must not be
described as a fully residual-consistent SUPG. Zhou's printed S_T likewise
contains only the convective operator, which is the literature precedent for
keeping the simplified form -- it is not a proof that the diffusive strong
residual vanishes.

It does not vanish, not even on affine elements. Under the shear map
x = xi + a*eta, y = eta, z = zeta, the trilinear reference field T = xi*eta maps
to T = x*y - a*y^2, whose physical Laplacian is -2a. Vanishing pure second
derivatives in reference coordinates say nothing about the physical Laplacian
once the map is not axis-aligned, and swept shell elements are not.

So: the element-interior diffusive strong residual is NOT yet in the
stabilisation, and the size of that approximation on non-axis-aligned and swept
meshes has to be established by verification, not assumed. `validation/` covers
it with an analytic field that has a non-zero physical Laplacian on a sheared
mesh; a pure-conduction test cannot, because at u = 0 the streamline weight
(u . grad v) vanishes too, and a linear-field patch test cannot, because a
linear field has zero Laplacian by construction.

Every appearance of the advective heat capacity uses the single `b_f` from the
material object, so the residual, tau_T, the thermal compliance and the port
enthalpy accounting cannot drift apart.

The system is linear in T for a given velocity field, but it is posed as a
`NonlinearProblem` so that it goes through the same implicitly-differentiated
Newton solve as the flow; it converges in one iteration.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

import toflux.src.solver as _solver

from tfopus import elements as _elements
from tfopus import sparse_bc as _sparse_bc


@dataclasses.dataclass(frozen=True)
class ThermalForm:
    """Which printed thermal stabilisation a case follows.

    Attributes:
      tau: "two_limit" is Zhou equation 23,
        tau_T = [(2|u|/h)^2 + (4k/(b_f h^2))^2]^(-1/2), which stays finite as
        u -> 0. "convective" is Zhao equation 18, tau_T = h/(2|u|), which does
        NOT: it diverges wherever the Brinkman penalty has driven the velocity
        to zero, i.e. over the whole solid phase. Every term it multiplies
        carries at least one factor of u, so the products have finite limits,
        but tau itself must be cut off -- see `tau_velocity_floor`.
      stabilise_source: add the source to the strong residual, giving the
        stabilised load sum_e integral tau (u.grad v) Q dV. Zhou's and Zhao's
        printed stabilisation matrices both omit it, so it is off for a literal
        reproduction and on for a residual-consistent scheme. With Zhao's
        divergent tau this term does NOT vanish as u -> 0 -- it tends to
        (h/2)|grad v|Q with an ill-defined sign -- which is a second reason the
        literal Zhao form keeps it off.
      supg_heat_capacity: multiply the SUPG advection block by b_f. Both papers
        print S_T without it (Zhou eq 22, Zhao eq 18) while printing the
        Galerkin advection block K_c,T WITH rho*c. As printed the two cannot be
        added: S_T comes out short by exactly one volumetric heat capacity. True
        reproduces the printed form; the default repairs it.
      tau_velocity_floor: speeds below this fraction of a reference speed give
        tau = 0 rather than a division by zero. Only used by "convective".
    """

    tau: str = "two_limit"
    stabilise_source: bool = True
    supg_heat_capacity: bool = True
    tau_velocity_floor: float = 1e-12

    def __post_init__(self):
        if self.tau not in ("two_limit", "convective"):
            raise ValueError(f"unknown tau form {self.tau!r}")


ZHOU_FORM = ThermalForm()
ZHAO_FORM = ThermalForm(
    tau="convective",  # Zhao eq 18
    stabilise_source=False,  # Zhao eq 17-18: Q enters as the Galerkin load only
    supg_heat_capacity=True,  # the printed S_T is short one rho*c; repaired
)


class ThermalSolver(_solver.NonlinearProblem):
    """Convection-conduction with SUPG and a volumetric source."""

    def __init__(
        self,
        mesh,
        bc,
        b_f: float,
        solver_settings,
        elem_length=None,
        form: ThermalForm = ZHOU_FORM,
        tau_elem=None,
    ):
        """
        Args:
          tau_elem: (num_elems,) stabilisation parameter supplied per element,
            overriding the formula. This exists for ONE diagnostic purpose:
            refining the mesh changes both the temperature space and tau at the
            same time, so a refined run with tau frozen at the parent values
            separates the two. It is a counterfactual, not a production mode --
            the frozen values are not the ones the formula would give on this
            mesh, so a solution obtained with it is not the scheme's solution.
        """
        super().__init__(solver_settings=solver_settings)
        self.form = form
        self.tau_elem = None if tau_elem is None else jnp.asarray(tau_elem)
        self.elem_length = (
            _elements.element_lengths(mesh, "min_edge")
            if elem_length is None
            else elem_length
        )
        self.mesh = mesh
        self.bc = bc
        self.b_f = b_f
        self.dim = mesh.num_dim
        self.node_id_jac = np.stack((mesh.iK, mesh.jK)).astype(np.int32).T
        self.shp_fn = jax.vmap(mesh.elem_template.shape_functions)(mesh.gauss_pts)
        self.shp_centre = mesh.elem_template.shape_functions(jnp.zeros(self.dim))

    def _tau(self, vel_nodes, k, h):
        """Zhou equation 23 or Zhao equation 18. See `ThermalForm`."""
        u_centre = jnp.einsum("n, nd -> d", self.shp_centre, vel_nodes)
        speed_sq = jnp.einsum("d, d -> ", u_centre, u_centre)

        if self.form.tau == "two_limit":
            inv_tau_conv_sq = 4.0 * speed_sq / h**2
            inv_tau_diff = 4.0 * k / (self.b_f * h**2)
            return (inv_tau_conv_sq + inv_tau_diff**2) ** (-0.5)

        # "convective": tau_T = h / (2|u|), singular at u = 0. Cut off rather
        # than regularised by adding to the denominator: every term tau
        # multiplies carries a factor of u, so tau = 0 is the correct limit of
        # the PRODUCT, whereas h/(2(|u| + eps)) would leave an O(h/eps) tail.
        floor = self.form.tau_velocity_floor
        speed = jnp.sqrt(jnp.where(speed_sq > floor**2, speed_sq, 1.0))
        return jnp.where(speed_sq > floor**2, h / (2.0 * speed), 0.0)

    def element_residual(
        self, temperature, velocity, k, q_source, node_coords, h, tau_given
    ):
        """Element residual. `velocity` is flat (nodes_per_elem * dim,).

        `tau_given` is used only when the solver was constructed with
        `tau_elem`; the branch is on a Python attribute, not a traced value, so
        the formula path is unchanged when it is not.
        """
        vel_nodes = velocity.reshape(-1, self.dim)

        grad_n = jax.vmap(
            self.mesh.elem_template.get_gradient_shape_function_physical,
            in_axes=(0, None),
        )(self.mesh.gauss_pts, node_coords)  # (g, n, d)
        _, det_j = jax.vmap(
            self.mesh.elem_template.compute_jacobian_and_determinant, in_axes=(0, None)
        )(self.mesh.gauss_pts, node_coords)
        w = self.mesh.gauss_weights * det_j

        tau = tau_given if self.tau_elem is not None else self._tau(vel_nodes, k, h)

        u_g = jnp.einsum("gn, nd -> gd", self.shp_fn, vel_nodes)
        grad_t = jnp.einsum("gnd, n -> gd", grad_n, temperature)
        u_dot_grad_t = jnp.einsum("gd, gd -> g", u_g, grad_t)
        u_dot_grad_n = jnp.einsum("gd, gnd -> gn", u_g, grad_n)

        # Strong residual, conduction part dropped (see the module docstring).
        # The advective heat capacity in front of it is `supg_heat_capacity`;
        # the source term is `stabilise_source`. See `ThermalForm`.
        supg_b = self.b_f if self.form.supg_heat_capacity else 1.0
        r_t = supg_b * u_dot_grad_t
        if self.form.stabilise_source:
            r_t = r_t - q_source

        galerkin = (
            self.b_f * jnp.einsum("gn, g -> gn", self.shp_fn, u_dot_grad_t)
            + k * jnp.einsum("gnd, gd -> gn", grad_n, grad_t)
            - q_source * self.shp_fn
        )
        supg = tau * jnp.einsum("gn, g -> gn", u_dot_grad_n, r_t)
        return jnp.einsum("gn, g -> n", galerkin + supg, w)

    def get_residual_and_tangent_stiffness(self, temperature, elem_velocity, k, q_source):
        args = (
            temperature[self.mesh.elem_dof_mat],
            elem_velocity,
            k,
            q_source,
            self.mesh.elem_node_coords,
            self.elem_length,
            self.tau_elem
            if self.tau_elem is not None
            else jnp.zeros(self.mesh.num_elems),
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

    # -- objective ----------------------------------------------------------

    def element_compliance(self, temperature, velocity, k, node_coords):
        """Zhou equation 24, per element.

            C = integral [ b_f T (u . grad T) + k grad T . grad T ] dV

        The SUPG block S_T is deliberately absent: Zhou defines C as
        T^T (K_c,T + K_v,T) T. That is a statement about the METRIC only. The
        temperature it is evaluated at still comes from the stabilised system, so
        the sensitivity of C must be taken through S_T and through the stabilised
        source load -- which the implicit differentiation does automatically,
        since both are in the residual.
        """
        vel_nodes = velocity.reshape(-1, self.dim)
        grad_n = jax.vmap(
            self.mesh.elem_template.get_gradient_shape_function_physical,
            in_axes=(0, None),
        )(self.mesh.gauss_pts, node_coords)
        _, det_j = jax.vmap(
            self.mesh.elem_template.compute_jacobian_and_determinant, in_axes=(0, None)
        )(self.mesh.gauss_pts, node_coords)

        u_g = jnp.einsum("gn, nd -> gd", self.shp_fn, vel_nodes)
        t_g = jnp.einsum("gn, n -> g", self.shp_fn, temperature)
        grad_t = jnp.einsum("gnd, n -> gd", grad_n, temperature)

        integrand = self.b_f * t_g * jnp.einsum(
            "gd, gd -> g", u_g, grad_t
        ) + k * jnp.einsum("gd, gd -> g", grad_t, grad_t)
        return jnp.einsum("g, g, g -> ", integrand, self.mesh.gauss_weights, det_j)

    def thermal_compliance(self, temperature, elem_velocity, k):
        """Total C over the mesh, in W K."""
        return jnp.sum(
            jax.vmap(self.element_compliance)(
                temperature[self.mesh.elem_dof_mat],
                elem_velocity,
                k,
                self.mesh.elem_node_coords,
            )
        )

    def compliance_decomposition(
        self, temperature, elem_velocity, k, q_source
    ) -> dict:
        """C split into its parts, plus the stabilisation terms it excludes.

        Taking the test function to be T_h itself in the discrete residual --
        legitimate here because T_h vanishes on the Dirichlet boundary, the
        inlet value being zero -- gives an identity that must hold exactly:

            C + D_SUPG - F_SUPG = L_Q

            C      = integral [ b_f T (u.grad T) + k |grad T|^2 ]   (Zhao eq 23)
            L_Q    = integral Q T
            D_SUPG = sum_e integral b_f tau (u.grad T)^2
            F_SUPG = sum_e integral tau Q (u.grad T)

        So C is not an independent quantity: it is L_Q minus the net
        stabilisation work. That makes `net_supg_over_c` the honest measure of
        how much of the reported compliance is shaped by the stabilisation --
        without implying the error IS that size, or that the stabilisation
        should be removed.

        `closure` is the residual of the identity and should be at solver
        precision. It is computed with the SAME quadrature as the residual, so
        a non-zero value means the state or the assembly disagrees, not that
        the integration rules differ.
        """
        temp_elem = temperature[self.mesh.elem_dof_mat]

        def per_element(temp, velocity, k_e, q_e, node_coords, h, tau_given):
            vel_nodes = velocity.reshape(-1, self.dim)
            grad_n = jax.vmap(
                self.mesh.elem_template.get_gradient_shape_function_physical,
                in_axes=(0, None),
            )(self.mesh.gauss_pts, node_coords)
            _, det_j = jax.vmap(
                self.mesh.elem_template.compute_jacobian_and_determinant,
                in_axes=(0, None),
            )(self.mesh.gauss_pts, node_coords)
            w = self.mesh.gauss_weights * det_j

            tau = (
                tau_given if self.tau_elem is not None
                else self._tau(vel_nodes, k_e, h)
            )
            u_g = jnp.einsum("gn, nd -> gd", self.shp_fn, vel_nodes)
            t_g = jnp.einsum("gn, n -> g", self.shp_fn, temp)
            grad_t = jnp.einsum("gnd, n -> gd", grad_n, temp)
            u_dot_grad_t = jnp.einsum("gd, gd -> g", u_g, grad_t)

            adv = jnp.einsum("g, g, g -> ", self.b_f * t_g, u_dot_grad_t, w)
            diff = k_e * jnp.einsum("gd, gd, g -> ", grad_t, grad_t, w)
            load = q_e * jnp.einsum("g, g -> ", t_g, w)
            supg_d = self.b_f * tau * jnp.einsum(
                "g, g, g -> ", u_dot_grad_t, u_dot_grad_t, w
            )
            supg_f = tau * q_e * jnp.einsum("g, g -> ", u_dot_grad_t, w)
            return adv, diff, load, supg_d, supg_f

        args = (
            temp_elem,
            elem_velocity,
            k,
            q_source,
            self.mesh.elem_node_coords,
            self.elem_length,
            self.tau_elem
            if self.tau_elem is not None
            else jnp.zeros(self.mesh.num_elems),
        )
        adv, diff, load, supg_d, supg_f = jax.vmap(per_element)(*args)

        c_adv, c_diff = float(jnp.sum(adv)), float(jnp.sum(diff))
        l_q, d_supg, f_supg = (
            float(jnp.sum(load)),
            float(jnp.sum(supg_d)),
            float(jnp.sum(supg_f)),
        )
        c = c_adv + c_diff
        return {
            "c_advective": c_adv,
            "c_diffusive": c_diff,
            "compliance": c,
            "l_q": l_q,
            "d_supg": d_supg,
            "f_supg": f_supg,
            "closure": c + d_supg - f_supg - l_q,
            "closure_relative": abs(c + d_supg - f_supg - l_q) / max(abs(l_q), 1e-300),
            "net_supg_over_c": (d_supg - f_supg) / c if c != 0 else float("nan"),
        }
