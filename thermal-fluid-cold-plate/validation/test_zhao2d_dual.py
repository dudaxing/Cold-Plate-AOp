"""R1g: the dual-mesh chain -- the maps, their transposes, the wiring, the gradient.

Each test targets one way the chain could be silently wrong:

  - P that is not the coarse function (a projection, or the wrong node order);
  - E^T that averages children instead of summing them;
  - a heat source whose total grows with the number of thermal elements;
  - the thermal solve still using the flow mesh's element length for tau;
  - a gradient that misses tau's dependence on u and kappa;
  - the single-mesh reference silently accepted as this model's normalisation.
"""

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tfopus import fe_thermal as fe_thermal  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402
from tfopus import zhao2d_refine as ref  # noqa: E402
from tfopus import zhao2d_thermal_study as ts  # noqa: E402

# 10 x 20 design + 2 x (2 x 2) tabs = 208 flow cells; wiring, not accuracy.
SPEC = dataclasses.replace(z.Zhao2DSpec(), element_size=5.0e-4)
STEPS = (1e-4, 1e-5, 1e-6)
TOL = 1e-5  # the existing gradient-check threshold, not re-tuned here


def _meshes(r):
    coarse = z.build_mesh(SPEC, dofs_per_node=1)
    fine = z.build_mesh(ref.refine_spec(SPEC, r), dofs_per_node=1)
    return coarse, fine, dual.build_nested_maps(coarse, fine, r)


@pytest.fixture(scope="module", params=[2, 4], ids=["r2", "r4"])
def nested(request):
    return _meshes(request.param)


def _bilinear(xy):
    # scaled so every term is O(1) on a 5e-3 x 1.2e-2 domain
    x, y = xy[:, 0] / 1e-3, xy[:, 1] / 1e-3
    return np.stack([1.5 + 2.0 * x - 0.7 * y + 0.3 * x * y,
                     -0.4 + 0.2 * x + 1.1 * y - 0.25 * x * y], axis=1)


# -- P: the velocity map ------------------------------------------------------


def test_p_reproduces_constants_and_full_bilinear_fields(nested):
    """Q1 holds a + bx + cy + dxy exactly; so must its extension, xy term included."""
    coarse, fine, maps = nested
    cc = np.asarray(coarse.mesh.nodes.coords)
    fc = np.asarray(fine.mesh.nodes.coords)

    const = np.tile([0.3, -1.7], (len(cc), 1))
    assert np.allclose(np.asarray(maps.velocity(const)), [0.3, -1.7], atol=1e-14)

    out = np.asarray(maps.velocity(_bilinear(cc)))
    assert np.max(np.abs(out - _bilinear(fc))) < 1e-12


def test_p_is_exact_on_shared_nodes(nested):
    coarse, fine, maps = nested
    rng = np.random.default_rng(0)
    u = rng.normal(size=(coarse.mesh.num_nodes, 2))
    out = np.asarray(maps.velocity(u))
    report = ts.check_extension(coarse, fine, u, out, tol=1e-14)
    assert report["shared_nodes"] == coarse.mesh.num_nodes


def test_p_agrees_with_the_static_numpy_extension(nested):
    """Two independent implementations of the same P must agree to rounding."""
    coarse, fine, maps = nested
    rng = np.random.default_rng(1)
    u = rng.normal(size=(coarse.mesh.num_nodes, 2))
    assert np.max(np.abs(
        np.asarray(maps.velocity(u)) - ts.extend_velocity(coarse, fine, u)
    )) < 1e-13


def _locate(planar, h, origin, pts):
    """(element, local coords) of each point, on a structured masked grid."""
    centres = np.asarray(planar.elem_centres)
    table = {
        (int(np.floor((cx - origin[0]) / h)), int(np.floor((cy - origin[1]) / h))): e
        for e, (cx, cy) in enumerate(centres)
    }
    elems, local = [], []
    for x, y in pts:
        i, j = int(np.floor((x - origin[0]) / h)), int(np.floor((y - origin[1]) / h))
        elems.append(table[(i, j)])
        local.append([2.0 * ((x - origin[0]) / h - i) - 1.0,
                      2.0 * ((y - origin[1]) / h - j) - 1.0])
    return np.array(elems), np.array(local)


def test_p_embeds_the_coarse_space_at_interior_points(nested):
    """The fine interpolant of P u IS the coarse function, not just at nodes.

    For nested rectangles a bilinear function restricted to a child is still
    bilinear, so this must hold for arbitrary nodal values, to rounding.
    """
    coarse, fine, maps = nested
    rng = np.random.default_rng(2)
    u = rng.normal(size=(coarse.mesh.num_nodes, 2))
    u_fine = np.asarray(maps.velocity(u))

    origin = np.asarray(coarse.mesh.nodes.coords).min(axis=0)
    h_c = SPEC.element_size
    h_f = h_c / maps.factor
    e_c = rng.integers(0, coarse.num_elems, 300)
    lo = np.asarray(coarse.mesh.elem_node_coords)[e_c].min(axis=1)
    pts = lo + rng.uniform(0.0, 1.0, (300, 2)) * h_c

    shape = jax.vmap(coarse.mesh.elem_template.shape_functions)
    ec, lc = _locate(coarse, h_c, origin, pts)
    ef, lf = _locate(fine, h_f, origin, pts)
    val_c = np.einsum("pn, pnd -> pd", np.asarray(shape(jnp.asarray(lc))),
                      u[np.asarray(coarse.mesh.elem_nodes)[ec]])
    val_f = np.einsum("pn, pnd -> pd", np.asarray(shape(jnp.asarray(lf))),
                      u_fine[np.asarray(fine.mesh.elem_nodes)[ef]])
    assert np.max(np.abs(val_c - val_f)) < 1e-12


# -- E: the density map -------------------------------------------------------


def test_e_preserves_endpoints_partition_and_area(nested):
    coarse, fine, maps = nested
    for value in (0.0, 1.0):
        out = np.asarray(maps.density(jnp.full(coarse.num_elems, value)))
        assert np.all(out == value)

    assert np.array_equal(fine.design_mask, coarse.design_mask[maps.parents])
    s_ref = z.reference_solid_fraction(coarse, SPEC, z.ReferenceField.TABS_FLUID)
    s_fine = np.asarray(maps.density(s_ref))
    assert np.all(s_fine[~fine.design_mask] == 0.0), "tabs picked up material"

    child_area = np.bincount(maps.parents, weights=np.asarray(fine.elem_area),
                             minlength=coarse.num_elems)
    assert np.allclose(child_area, np.asarray(coarse.elem_area), rtol=1e-13)


@pytest.mark.parametrize("region", list(z.SourceRegion))
def test_total_heat_source_does_not_change_with_the_thermal_mesh(nested, region):
    coarse, fine, _ = nested
    q_c = np.asarray(z.heat_source_field(coarse, SPEC, region))
    q_f = np.asarray(z.heat_source_field(fine, SPEC, region))
    assert np.sum(q_f * np.asarray(fine.elem_area)) == pytest.approx(
        np.sum(q_c * np.asarray(coarse.elem_area)), rel=1e-13
    )


# -- the transposes -----------------------------------------------------------


def test_e_transpose_sums_children_and_matches_the_explicit_matrix(nested):
    """E^T must ACCUMULATE: averaging would scale the thermal gradient by 1/r^2."""
    coarse, fine, maps = nested
    e_mat, _ = maps.matrices()
    rng = np.random.default_rng(3)
    s = jnp.asarray(rng.uniform(size=coarse.num_elems))
    _, vjp = jax.vjp(maps.density, s)

    ones = np.asarray(vjp(jnp.ones(fine.num_elems))[0])
    assert np.all(ones == maps.factor**2)

    ct = rng.normal(size=fine.num_elems)
    back = np.asarray(vjp(jnp.asarray(ct))[0])
    assert np.allclose(back, e_mat.T @ ct, rtol=1e-13, atol=1e-13)

    a = rng.normal(size=coarse.num_elems)
    assert float(np.dot(np.asarray(maps.density(a)), ct)) == pytest.approx(
        float(np.dot(a, back)), rel=1e-12
    )


def test_p_transpose_matches_the_explicit_matrix(nested):
    coarse, fine, maps = nested
    _, p_mat = maps.matrices()
    assert np.allclose(np.asarray(p_mat.sum(axis=1)).ravel(), 1.0, atol=1e-14), (
        "rows of P must sum to one: the extension of a constant is that constant"
    )
    rng = np.random.default_rng(4)
    u = jnp.asarray(rng.normal(size=(coarse.mesh.num_nodes, 2)))
    _, vjp = jax.vjp(maps.velocity, u)
    ct = rng.normal(size=(fine.mesh.num_nodes, 2))
    back = np.asarray(vjp(jnp.asarray(ct))[0])
    assert np.allclose(back, p_mat.T @ ct, rtol=1e-12, atol=1e-13)
    assert float(np.sum(np.asarray(maps.velocity(u)) * ct)) == pytest.approx(
        float(np.sum(np.asarray(u) * back)), rel=1e-12
    )


# -- the problem --------------------------------------------------------------


@pytest.fixture(scope="module")
def problem_r2():
    return dual.Zhao2DDualProblem(SPEC, r1.R1Config(), thermal_refinement=2,
                                  thermal_quadrature=3)


@pytest.fixture(scope="module")
def frozen():
    values, _ = r1.freeze_reference(SPEC, r1.R1Config())
    return values


def _grey(problem, seed=5):
    return jnp.asarray(np.random.default_rng(seed).uniform(0.15, 0.85, problem.num_design))


def test_refinement_one_reproduces_the_single_mesh_chain():
    """Identity maps must leave the old chain unchanged -- values and gradients.

    This is the regression that the wiring (velocity layout, kappa on the
    thermal mesh, tau length) is the old one when the meshes coincide, at the
    old chain's own quadrature.
    """
    old = r1.Zhao2DProblem(SPEC, r1.R1Config())
    new = dual.Zhao2DDualProblem(SPEC, r1.R1Config(), thermal_refinement=1,
                                 thermal_quadrature=2)
    x = _grey(old)
    beta, alpha_max = 2.0, 1.0e7

    s = old.solid_fraction(x, beta)
    psi_o, c_o = old.metrics(s, alpha_max)
    psi_n, c_n = new.metrics(s, alpha_max)
    assert float(psi_n) == pytest.approx(float(psi_o), rel=1e-13)
    assert float(c_n) == pytest.approx(float(c_o), rel=1e-12)

    def c_of(problem):
        return lambda v: problem.metrics(problem.solid_fraction(v, beta), alpha_max)[1]

    g_o = np.asarray(jax.grad(c_of(old))(x))
    g_n = np.asarray(jax.grad(c_of(new))(x))
    assert np.allclose(g_n, g_o, rtol=1e-10, atol=1e-10 * np.abs(g_o).max())


def test_thermal_solver_uses_the_thermal_mesh_and_the_formula_tau(problem_r2):
    """tau from the formula at h/r, every evaluation -- never frozen, never h."""
    assert problem_r2.thermal.tau_elem is None
    h = np.asarray(problem_r2.thermal.elem_length)
    assert np.allclose(h, SPEC.element_size / 2)
    assert problem_r2.thermal_mesh.num_elems == 4 * problem_r2.flow_mesh.num_elems
    n = problem_r2.nesting
    assert n["heat_source_thermal_mesh"] == pytest.approx(
        n["heat_source_flow_mesh"], rel=1e-13
    )


def test_single_mesh_reference_is_refused(problem_r2, frozen):
    """The inherited check would accept it: spec and config are unchanged."""
    x = jnp.full(problem_r2.num_design, 0.6)
    frozen.check(problem_r2.config, problem_r2.spec)  # the trap: this passes
    with pytest.raises(ValueError, match="not frozen for the dual-mesh"):
        problem_r2.objective_and_constraint(x, frozen, 1.0e6)

    own = dataclasses.replace(frozen, identity=problem_r2.reference_identity())
    j, g = problem_r2.objective_and_constraint(x, own, 1.0e6)
    assert np.isfinite(float(j)) and np.isfinite(float(g))


# -- the gradient -------------------------------------------------------------


def _directional_check(problem, scale, x0, direction, alpha_max, beta):
    """AD (reverse mode, what MMA uses) against central FD, per quantity.

    Returns {quantity: min over step of relative error}. Every perturbed state
    is gated: a difference across an unconverged solve is not a derivative.
    """
    w = problem.config.weight

    def quantity(i):
        def f(x):
            s = problem.solid_fraction(x, beta)
            psi, c = problem.metrics(s, alpha_max)
            g = problem.fluid_fraction(x, beta) / problem.config.max_fluid_fraction - 1
            return (psi, c, g, dual.reporting_objective(psi, c, scale, w))[i]
        return f

    ad = [float(jnp.dot(jax.grad(quantity(i))(x0), direction)) for i in range(4)]
    best = [np.inf] * 4
    for h in STEPS:
        vals = []
        for sign in (1.0, -1.0):
            e = problem.evaluate(x0 + sign * h * direction, alpha_max, beta)
            assert max(e["residuals"].values()) < 1e-8, e["residuals"]
            vals.append((e["psi"], e["c"], e["g"],
                         float(dual.reporting_objective(e["psi"], e["c"], scale, w))))
        for i in range(4):
            fd = (vals[0][i] - vals[1][i]) / (2 * h)
            best[i] = min(best[i], abs(ad[i] - fd) / max(abs(fd), 1e-300))
    return dict(zip(("Psi", "C", "g", "J"), best)), ad


@pytest.mark.parametrize("seed", [11, 12])
def test_total_gradient_matches_finite_differences(problem_r2, frozen, seed):
    """Psi, C, g and J separately, projected, grey, through E, P and tau(k, u)."""
    rng = np.random.default_rng(seed)
    x0 = _grey(problem_r2, seed)
    direction = jnp.asarray(rng.choice([-1.0, 1.0], problem_r2.num_design))
    errors, _ = _directional_check(problem_r2, frozen, x0, direction, 1.0e7, 8.0)
    assert max(errors.values()) < TOL, errors


def test_gradient_includes_tau_dependence_on_velocity_and_conductivity(problem_r2):
    """Freezing tau changes the derivative, and FD sides with the full one.

    If the chain dropped tau's dependence on u_T and kappa -- e.g. by
    evaluating it outside the traced computation -- the AD result would be the
    frozen one, and it would disagree with finite differences by this margin.
    """
    alpha_max, beta = 1.0e7, 8.0
    x0 = _grey(problem_r2, 21)
    direction = jnp.asarray(np.random.default_rng(22).choice([-1.0, 1.0],
                                                             problem_r2.num_design))
    s0 = problem_r2.solid_fraction(x0, beta)
    pv0, _, _, kappa0 = problem_r2.solve_states(s0, alpha_max)
    tau0 = ts.element_tau(problem_r2.thermal, problem_r2.thermal_velocity(pv0), kappa0)

    frozen_tau = dual.Zhao2DDualProblem(SPEC, r1.R1Config(), 2, 3)
    frozen_tau.thermal = fe_thermal.ThermalSolver(
        frozen_tau.thermal_mesh.mesh, frozen_tau.thermal_bc, b_f=SPEC.b_f,
        solver_settings=frozen_tau.settings, elem_length=frozen_tau.h_thermal,
        form=frozen_tau.config.thermal_form, tau_elem=tau0,
    )

    def c_of(problem):
        return lambda x: problem.metrics(problem.solid_fraction(x, beta), alpha_max)[1]

    full = float(jnp.dot(jax.grad(c_of(problem_r2))(x0), direction))
    no_tau = float(jnp.dot(jax.grad(c_of(frozen_tau))(x0), direction))
    h = 1e-5
    fd = (problem_r2.evaluate(x0 + h * direction, alpha_max, beta)["c"]
          - problem_r2.evaluate(x0 - h * direction, alpha_max, beta)["c"]) / (2 * h)

    assert abs(full - fd) / abs(fd) < TOL
    assert abs(no_tau - fd) / abs(fd) > 100 * TOL, (
        "freezing tau barely moved the derivative, so this test cannot see "
        "whether the tau path is present"
    )


# -- the regression anchor ----------------------------------------------------


@pytest.mark.slow
def test_r1d_design_at_h_over_2_reproduces_r1f_c_row_not_d():
    """Saved design, coarse flow re-solved, h_T = h/2, 3x3: R1f's C row.

    C (33233.13) used the coarse velocity extended; D (32400.12) used a flow
    re-solved on the fine mesh. The dual-mesh model keeps the coarse flow, so
    landing on D would mean a fine flow had crept back in.
    """
    import pathlib

    results = pathlib.Path(__file__).resolve().parent.parent / "results"
    fields = results / "zhao2d_r1d_main_fields.npz"
    if not fields.is_file():
        pytest.skip("R1d fields not present")
    s = jnp.asarray(np.load(fields)["solid_fraction"])

    problem = dual.Zhao2DDualProblem(z.Zhao2DSpec(), r1.R1Config(),
                                     thermal_refinement=2, thermal_quadrature=3)
    psi, c = problem.metrics(s, 1.0e7)
    assert float(c) == pytest.approx(33233.13278907864, rel=1e-10)
    assert abs(float(c) / 32400.115711650746 - 1.0) > 0.02
