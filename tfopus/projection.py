"""The volume-preserving projection of Xu, Cai & Cheng (2010).

S. Xu, Y. Cai, G. Cheng, "Volume preserving nonlinear density filter based on
heaviside functions", Struct Multidisc Optim 41 (2010) 495-505.

Their Eq. (19) rescales Sigmund's modified Heaviside (their Eq. 15) onto
[0, eta] and Guest's Heaviside (their Eq. 11) onto [eta, 1]:

    H(rho) = eta [exp(-beta (1 - rho/eta)) - (1 - rho/eta) exp(-beta)]      rho <= eta
    H(rho) = (1 - eta) [1 - exp(-beta (rho - eta)/(1 - eta))
                        + (rho - eta) exp(-beta)/(1 - eta)] + eta            rho > eta

so H(0) = 0, H(eta) = eta, H(1) = 1; H is the identity at beta = 0 and a step
at eta as beta grows. eta is not a free parameter: their Eq. (21) fixes it so
that the projected volume equals the filtered one,

    sum_i v_i H(rho_i; eta) = sum_i v_i rho_i,

and their Appendix A shows the root is unique in ]0, 1[ (f(eta) decreases,
Eqs. 26-28). The volume a constraint sees is then the filtered volume whatever
beta is, so beta can be chosen for sharpness alone.

One deliberate difference from the paper. Its sensitivities use the chain rule
(13) with dH/drho of Eq. (20), holding eta fixed; but eta moves with the design
through (21). Here the derivative includes that, by implicit differentiation
of the root, so it is the derivative of the map actually used -- the one a
finite-difference check measures -- and the volume's derivative is exactly the
filtered volume's, which the eta-fixed chain rule does not give.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def xu_heaviside(rho, beta: float, eta):
    """Eq. (19), for 0 <= rho <= 1 and 0 < eta < 1.

    Each branch is evaluated only on its own side of eta (the argument clamped
    to it), so neither overflows at large beta nor leaks a NaN into the
    derivative through the branch jnp.where does not select. The clamps are
    jnp.where, not jnp.minimum/maximum: at rho == eta those split the
    derivative between their two arguments and would halve dH/drho there.
    """
    below_eta = rho <= eta
    lo = jnp.where(below_eta, rho, eta)
    hi = jnp.where(below_eta, eta, rho)
    e = jnp.exp(-beta)
    below = eta * (jnp.exp(-beta * (1.0 - lo / eta)) - (1.0 - lo / eta) * e)
    above = (1.0 - eta) * (1.0 - jnp.exp(-beta * (hi - eta) / (1.0 - eta))
                           + (hi - eta) * e / (1.0 - eta)) + eta
    return jnp.where(below_eta, below, above)


def volume_preserving_eta(rho, volumes, beta: float, iterations: int = 64):
    """The eta of Eq. (21), by bisection; differentiable in rho and the volumes.

    f(eta) = sum v H(rho; eta) - sum v rho decreases, with f(0+) > 0 and
    f(1-) < 0 whenever some rho lies strictly between 0 and 1 (Appendix A), so
    bisection on ]0, 1[ cannot miss the root; 64 halvings reach double
    precision and never evaluate H at eta = 0 or 1. `jax.lax.custom_root`
    supplies d(eta)/d(rho) implicitly. If no rho is intermediate, neither f nor
    H depends on eta, and the derivative through eta is zero, not 0/0.
    """
    rho = jnp.asarray(rho)
    volumes = jnp.asarray(volumes)

    def f(eta):
        return jnp.sum(volumes * xu_heaviside(rho, beta, eta)) - jnp.sum(volumes * rho)

    def bisect(func, _):
        def halve(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            above = func(mid) > 0.0  # f decreases: the root lies above mid
            return jnp.where(above, mid, lo), jnp.where(above, hi, mid)

        zero = jnp.asarray(0.0, rho.dtype)
        lo, hi = jax.lax.fori_loop(0, iterations, halve, (zero, zero + 1.0))
        return 0.5 * (lo + hi)

    def tangent_solve(g, y):
        slope = g(1.0)
        safe = jnp.where(slope != 0.0, slope, 1.0)
        return jnp.where(slope != 0.0, y / safe, 0.0)

    return jax.lax.custom_root(f, jnp.asarray(0.5, rho.dtype), bisect, tangent_solve)


def volume_preserving_projection(rho, volumes, beta: float):
    """(projected, eta): Eq. (19) at the eta of Eq. (21). beta <= 0 is the identity.

    At beta = 0 Eq. (19) is the identity for every eta, so eta is undefined and
    returned as NaN; nothing downstream depends on it.
    """
    rho = jnp.asarray(rho)
    if beta <= 0.0:
        return rho, jnp.asarray(jnp.nan, rho.dtype)
    eta = volume_preserving_eta(rho, volumes, beta)
    return xu_heaviside(rho, beta, eta), eta
