"""The volume-preserving projection of Xu, Cai & Cheng (2010), and its use in R1.

Eq. (19) is checked against the paper's own closed forms -- its derivative (20)
and the eta-derivatives (27) and (28) of its Appendix A -- so that the
implementation is tested against formulas it does not share code with. Then
Eq. (21): the projected volume equals the filtered volume for any beta, eta is
the unique root, and the derivative, which includes eta's dependence on the
design, matches finite differences. Last, the R1 problem: the constraint no
longer moves with beta under the default, and the TANH setting still reproduces
the records made with it.
"""

import dataclasses
import json
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tfopus import projection as P  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
BETAS = (0.1, 1.0, 8.0, 16.0, 64.0, 200.0)


def _field(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rho = np.clip(rng.uniform(-0.2, 1.2, n), 0.0, 1.0)  # includes exact 0s and 1s
    return jnp.asarray(rho), jnp.asarray(rng.uniform(0.5, 1.5, n))


# -- Eq. (19) against the paper's closed forms -------------------------------------


@pytest.mark.parametrize("eta", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("beta", [0.5, 8.0, 64.0])
def test_eq19_passes_through_0_eta_1_and_is_continuous(eta, beta):
    H = lambda r: float(P.xu_heaviside(jnp.asarray(r), beta, eta))  # noqa: E731
    assert H(0.0) == pytest.approx(0.0, abs=1e-15)
    assert H(eta) == pytest.approx(eta, rel=1e-14)
    assert H(1.0) == pytest.approx(1.0, rel=1e-14)
    assert H(eta - 1e-12) == pytest.approx(H(eta + 1e-12), abs=1e-9)


@pytest.mark.parametrize("eta", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("beta", [0.5, 8.0, 64.0])
def test_eq19_derivative_is_eq20_and_eta_derivatives_are_eqs27_28(eta, beta):
    rho = jnp.linspace(0.0, 1.0, 101)
    d_rho = jax.vmap(jax.grad(lambda r: P.xu_heaviside(r, beta, eta)))(rho)
    d_eta = jax.vmap(jax.grad(lambda e, r: P.xu_heaviside(r, beta, e)), (None, 0))(eta, rho)
    below = np.asarray(rho) <= eta
    r = np.asarray(rho)
    eq20 = np.where(below, beta * np.exp(-beta * (1 - r / eta)) + np.exp(-beta),
                    beta * np.exp(-beta * (r - eta) / (1 - eta)) + np.exp(-beta))
    q = np.minimum(r, eta) / eta
    eq27 = np.exp(-beta) * (np.exp(beta * q) - 1 - beta * q * np.exp(beta * q))
    p = (np.maximum(r, eta) - eta) / (1 - eta)
    eq28 = np.exp(-beta * p) * (-beta * (1 - r) / (1 - eta) + 1
                                - np.exp(-beta * (1 - r) / (1 - eta)))
    assert np.allclose(np.asarray(d_rho), eq20, rtol=1e-12, atol=1e-12)
    assert np.allclose(np.asarray(d_eta), np.where(below, eq27, eq28), rtol=1e-10, atol=1e-12)
    assert np.all(np.asarray(d_eta) <= 1e-15)  # Appendix A: H never increases with eta
    assert np.all(np.asarray(d_rho) > 0.0)


@pytest.mark.parametrize("beta", [0.5, 8.0, 64.0])
def test_the_derivative_at_rho_equal_eta_is_beta_plus_exp_minus_beta(beta):
    """Eq. (20) at the join: both sides give beta + e^-beta. A min/max clamp
    would split this derivative at the tie and return half of it."""
    eta = 0.5  # exactly representable, so rho == eta really is a tie
    d = float(jax.grad(lambda r: P.xu_heaviside(r, beta, eta))(jnp.asarray(eta)))
    assert d == pytest.approx(beta + np.exp(-beta), rel=1e-14)


def test_beta_zero_is_the_identity_for_any_eta():
    rho, vol = _field()
    for eta in (0.1, 0.5, 0.9):
        assert np.allclose(np.asarray(P.xu_heaviside(rho, 0.0, eta)), np.asarray(rho),
                           rtol=0, atol=1e-15)
    s, eta = P.volume_preserving_projection(rho, vol, 0.0)
    assert bool(jnp.all(s == rho)) and np.isnan(float(eta))


# -- Eq. (21): volume preservation and the root ------------------------------------------


@pytest.mark.parametrize("beta", BETAS)
def test_the_projected_volume_is_the_filtered_volume(beta):
    rho, vol = _field()
    s, eta = P.volume_preserving_projection(rho, vol, beta)
    assert 0.0 < float(eta) < 1.0
    assert float(jnp.sum(vol * s)) == pytest.approx(float(jnp.sum(vol * rho)), rel=1e-13)


def test_eta_is_the_unique_root():
    rho, vol = _field()
    beta = 16.0
    grid = jnp.linspace(1e-3, 1 - 1e-3, 999)
    f = jax.vmap(lambda e: jnp.sum(vol * P.xu_heaviside(rho, beta, e)) - jnp.sum(vol * rho))(grid)
    f = np.asarray(f)
    assert np.all(np.diff(f) < 0) and f[0] > 0 > f[-1]  # decreasing, one sign change
    eta = float(P.volume_preserving_eta(rho, vol, beta))
    assert grid[np.argmin(np.abs(f))] == pytest.approx(eta, abs=2e-3)


def test_large_beta_is_nearly_binary_at_the_same_volume():
    rho, vol = _field()
    s, _ = P.volume_preserving_projection(rho, vol, 200.0)
    m_nd = float(jnp.mean(4 * s * (1 - s)))  # the paper's Eq. (22), as a fraction
    assert m_nd < 0.02
    assert float(jnp.sum(vol * s)) == pytest.approx(float(jnp.sum(vol * rho)), rel=1e-13)


# -- derivatives, including eta's --------------------------------------------------------


@pytest.mark.parametrize("beta", BETAS)
def test_the_derivative_matches_central_differences(beta):
    rho, vol = _field()
    rng = np.random.default_rng(1)
    c = jnp.asarray(rng.normal(size=rho.shape))
    d = jnp.asarray(rng.choice([-1.0, 1.0], rho.shape))

    def L(r):
        return jnp.dot(c, P.volume_preserving_projection(r, vol, beta)[0])

    ad = float(jnp.dot(jax.grad(L)(rho), d))
    best = min(abs(ad - (float(L(rho + h * d)) - float(L(rho - h * d))) / (2 * h)) / abs(ad)
               for h in (1e-4, 1e-5, 1e-6, 1e-7))
    assert best < 1e-7


@pytest.mark.parametrize("beta", BETAS)
def test_the_projected_volume_has_the_filtered_volumes_derivative(beta):
    rho, vol = _field()
    g = jax.grad(lambda r: jnp.sum(vol * P.volume_preserving_projection(r, vol, beta)[0]))(rho)
    assert np.allclose(np.asarray(g), np.asarray(vol), rtol=1e-11, atol=1e-12)


def test_holding_eta_fixed_would_get_the_volume_gradient_wrong():
    """The paper's chain rule (13) with (20) omits d(eta)/d(rho). For the
    volume -- the constraint -- the exact derivative is v; with eta held fixed
    it is v H'(rho), nearly zero away from eta and about beta near it."""
    rho, vol = _field()
    beta = 16.0
    eta = P.volume_preserving_eta(rho, vol, beta)
    fixed = jax.grad(lambda r: jnp.sum(vol * P.xu_heaviside(r, beta, jax.lax.stop_gradient(eta))))(rho)
    exact = jax.grad(lambda r: jnp.sum(vol * P.volume_preserving_projection(r, vol, beta)[0]))(rho)
    assert np.allclose(np.asarray(exact), np.asarray(vol), rtol=1e-11)
    assert float(jnp.linalg.norm(fixed - vol) / jnp.linalg.norm(vol)) > 0.5


def test_a_field_with_no_intermediate_density_stays_finite():
    rho = jnp.concatenate([jnp.zeros(10), jnp.ones(10)])
    vol = jnp.ones(20)
    s, _ = P.volume_preserving_projection(rho, vol, 8.0)
    g = jax.grad(lambda r: jnp.sum(P.volume_preserving_projection(r, vol, 8.0)[0] ** 2))(rho)
    assert bool(jnp.all(s == rho)) and bool(jnp.all(jnp.isfinite(g)))


# -- in the R1 problem ---------------------------------------------------------------------


SPEC = dataclasses.replace(z.Zhao2DSpec(), element_size=5.0e-4)


@pytest.fixture(scope="module")
def problems():
    return (r1.Zhao2DProblem(SPEC, r1.R1Config()),
            r1.Zhao2DProblem(SPEC, r1.R1Config(projection=r1.Projection.TANH)))


def test_the_default_is_volume_preserving_and_the_reference_does_not_see_it():
    assert r1.R1Config().projection == r1.Projection.VOLUME_PRESERVING
    tanh = r1.R1Config(projection=r1.Projection.TANH)
    assert r1.reference_identity(SPEC, r1.R1Config()) == r1.reference_identity(SPEC, tanh)
    assert r1.R1Config().fingerprint() != tanh.fingerprint()  # recorded with every run


def test_under_the_default_beta_does_not_move_the_constraint(problems):
    vp, tanh = problems
    x = jnp.asarray(np.random.default_rng(3).uniform(0.2, 0.9, vp.num_design))
    filtered = float(vp.fluid_fraction(x, 0.0))
    for beta in (1.0, 8.0, 16.0, 64.0):
        assert float(vp.fluid_fraction(x, beta)) == pytest.approx(filtered, rel=1e-12)
        assert 0.0 < vp.projection_threshold(x, beta) < 1.0
    # the tanh projection, by contrast, moves it
    assert abs(float(tanh.fluid_fraction(x, 16.0)) - filtered) > 1e-3
    assert tanh.projection_threshold(x, 16.0) == 0.5


def test_the_constraints_gradient_is_the_filtered_volumes_for_every_beta(problems):
    vp, _ = problems
    x = jnp.asarray(np.random.default_rng(4).uniform(0.2, 0.9, vp.num_design))
    g0 = jax.grad(lambda v: vp.fluid_fraction(v, 0.0))(x)
    for beta in (8.0, 64.0):
        g = jax.grad(lambda v: vp.fluid_fraction(v, beta))(x)
        assert np.allclose(np.asarray(g), np.asarray(g0), rtol=1e-9, atol=1e-15)


def test_tanh_still_reproduces_r1ds_saved_design():
    """The records made before the default changed stay reproducible."""
    meta = json.loads((REPO / "results" / "zhao2d_r1d_main.json").read_text(encoding="utf-8"))
    fields = np.load(REPO / "results" / "zhao2d_r1d_main_fields.npz")
    problem = r1.Zhao2DProblem(z.Zhao2DSpec(), r1.R1Config(projection=r1.Projection.TANH))
    s = np.asarray(problem.solid_fraction(jnp.asarray(fields["design"]), meta["final_beta"]))
    assert np.array_equal(s, np.asarray(fields["solid_fraction"]))
