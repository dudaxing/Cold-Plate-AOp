"""Zhao's 2D heat sink: geometry reconstruction, interpolation, formulation options.

These pin the things R0 has to get right before any reference number means
anything. They do NOT assert that the paper's Psi_0 / C_0 are reproduced --
whether they can be is a measurement, reported by
`scripts/zhao2d_reference_study.py`, not a property of this implementation.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tfopus import elements as elements  # noqa: E402
from tfopus import fe_flow as fe_flow  # noqa: E402
from tfopus import materials as materials  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus.mesh import Face, Region  # noqa: E402

SPEC = z.Zhao2DSpec()


# -- geometry ---------------------------------------------------------------


def test_element_count_matches_the_paper():
    """50x100 design + 2x(10x10) tabs = 5200, the count section 4.1 reports."""
    pm = z.build_mesh(SPEC, dofs_per_node=3)
    assert pm.num_elems == SPEC.reported_num_elements
    assert int(pm.design_mask.sum()) == 5000
    assert int((~pm.design_mask).sum()) == 200


def test_coarse_mesh_halves_consistently():
    """h -> 2h gives 25x50 + 2x(5x5) = 1300, the obvious coarse companion."""
    import dataclasses

    coarse = dataclasses.replace(SPEC, element_size=SPEC.element_size * 2)
    assert z.build_mesh(coarse, dofs_per_node=1).num_elems == 1300


def test_non_conforming_element_size_is_rejected():
    """A tab that is not a whole number of elements must fail, not round."""
    import dataclasses

    bad = dataclasses.replace(SPEC, element_size=SPEC.element_size * 3)
    with pytest.raises(ValueError, match="whole number"):
        z.build_mesh(bad, dofs_per_node=1)


def test_areas_and_face_tags():
    pm = z.build_mesh(SPEC, dofs_per_node=3)
    h = SPEC.element_size
    assert np.allclose(pm.elem_area, h * h, rtol=1e-6)

    total = SPEC.design_half_width * SPEC.design_height + 2 * (
        SPEC.inlet_half_width * SPEC.tab_length
    )
    assert float(np.sum(pm.elem_area)) == pytest.approx(total, rel=1e-6)

    counts = pm.face_count()
    n_tab = int(round(SPEC.inlet_half_width / h))
    assert counts["INLET"] == n_tab
    assert counts["OUTLET"] == n_tab
    # symmetry runs the full height of the model, tabs included
    assert counts["SYMMETRY"] == int(
        round((SPEC.design_height + 2 * SPEC.tab_length) / h)
    )


def test_no_duplicate_or_orphan_nodes():
    pm = z.build_mesh(SPEC, dofs_per_node=1)
    used = np.unique(np.asarray(pm.mesh.elem_nodes))
    assert used.size == pm.mesh.num_nodes
    assert used.min() == 0 and used.max() == pm.mesh.num_nodes - 1
    coords = np.asarray(pm.mesh.nodes.coords)
    assert np.unique(coords, axis=0).shape[0] == coords.shape[0]


def test_reynolds_identifies_the_inlet_half_width():
    """Re = rho U D / mu returns the paper's 200 only for D = the HALF width.

    This is a consistency check on the geometry reading, not an input: the full
    width would give 400.
    """
    assert SPEC.reynolds() == pytest.approx(SPEC.reported_reynolds)
    assert SPEC.reynolds(2 * SPEC.inlet_half_width) == pytest.approx(
        2 * SPEC.reported_reynolds
    )


# -- interpolation ----------------------------------------------------------


def test_interpolation_endpoints():
    mat = materials.ThermoFluidMaterial(
        fluid=materials.Phase(
            "f",
            SPEC.fluid_density,
            SPEC.fluid_conductivity,
            SPEC.fluid_heat_capacity,
            SPEC.fluid_viscosity,
        ),
        solid=materials.Phase("s", 0.0, SPEC.solid_conductivity, 0.0),
        alpha_min=0.0,
        alpha_max=SPEC.alpha_max_initial,
        q_alpha_zhou=SPEC.q_alpha,
        q_k_zhou=SPEC.q_kappa,
    )
    s = jnp.array([0.0, 1.0])  # fluid, solid
    assert np.allclose(materials.brinkman_penalty(s, mat), [0.0, SPEC.alpha_max_initial])
    assert np.allclose(
        materials.conductivity(s, mat),
        [SPEC.fluid_conductivity, SPEC.solid_conductivity],
    )


def test_reference_field_interpolates_to_the_analytic_values():
    """At gamma = 0.4 with q = 0.2 the RAMP factor is exactly 0.2.

        q(1-gamma)/(gamma+q) = 0.2*0.6/0.6 = 0.2
        alpha = 0.2 * alpha_max
        kappa = 0.61 + (237 - 0.61)*0.2 = 47.888

    Analytic, so no tolerance argument is needed.
    """
    mat = materials.ThermoFluidMaterial(
        fluid=materials.Phase(
            "f", 1000.0, SPEC.fluid_conductivity, 4180.0, 0.001
        ),
        solid=materials.Phase("s", 0.0, SPEC.solid_conductivity, 0.0),
        alpha_min=0.0,
        alpha_max=1.0e6,
        q_alpha_zhou=0.2,
        q_k_zhou=0.2,
    )
    s = jnp.array([1.0 - 0.4])
    assert float(materials.brinkman_penalty(s, mat)[0]) == pytest.approx(0.2e6)
    assert float(materials.conductivity(s, mat)[0]) == pytest.approx(47.888)


def test_zhao_and_toflux_ramp_parameters_are_reciprocal():
    """q_toflux = 1/q_zhao. Passing 0.2 to TOFLUX's convex RAMP is wrong."""
    s = jnp.linspace(0.0, 1.0, 11)
    a = materials.ramp_zhou(s, 0.0, 1.0e6, q_zhou=0.2)
    b = materials.ramp_toflux_convex(s, 0.0, 1.0e6, q_toflux=5.0)
    assert np.allclose(a, b)
    assert not np.allclose(a, materials.ramp_toflux_convex(s, 0.0, 1.0e6, 0.2))


# -- fields and bookkeeping -------------------------------------------------


def test_fluid_fractions_split_design_and_whole_domain():
    """The tabs are 200 of 5200 cells, so the two conventions differ by 2.3 pp.

    (0.4*5000 + 1.0*200)/5200 = 0.423077. Reporting one number as "the" volume
    fraction would silently pick a side of equation 26's ambiguity.
    """
    pm = z.build_mesh(SPEC, dofs_per_node=1)
    s = z.reference_solid_fraction(pm, SPEC, z.ReferenceField.TABS_FLUID)
    f = z.fluid_fractions(pm, s)
    assert f["v_f_design_domain"] == pytest.approx(0.4, rel=1e-9)
    assert f["v_f_whole_domain"] == pytest.approx((0.4 * 5000 + 200) / 5200, rel=1e-9)

    s_all = z.reference_solid_fraction(pm, SPEC, z.ReferenceField.UNIFORM_ALL)
    f_all = z.fluid_fractions(pm, s_all)
    assert f_all["v_f_whole_domain"] == pytest.approx(0.4, rel=1e-9)


def test_heat_source_is_independent_of_the_design_field():
    """Q must not be scaled by density, or the optimiser could shrink the load."""
    pm = z.build_mesh(SPEC, dofs_per_node=1)
    q = z.heat_source_field(pm, SPEC, z.SourceRegion.WHOLE_DOMAIN)
    assert np.allclose(q, SPEC.heat_source)

    q_design = z.heat_source_field(pm, SPEC, z.SourceRegion.DESIGN_ONLY)
    assert float(jnp.sum(q_design > 0)) == 5000
    total = float(jnp.sum(q_design * jnp.asarray(pm.elem_area)))
    assert total == pytest.approx(
        SPEC.heat_source * SPEC.design_half_width * SPEC.design_height, rel=1e-6
    )


def test_flow_bc_inlet_wins_on_the_shared_rim():
    """The inlet rim node carries the inlet velocity, not no-slip.

    With ten inlet faces, letting the wall win would drop a whole element's
    worth of the imposed profile -- 10% of the mass flow on this mesh.
    """
    pm = z.build_mesh(SPEC, dofs_per_node=3)
    fb = z.build_flow_bc(pm, SPEC)
    dof = pm.mesh.nodes.dof_per_node
    lookup = dict(zip(fb.bc["fixed_dofs"].tolist(), fb.bc["dirichlet_values"].tolist()))

    shared = np.intersect1d(fb.inlet_nodes, fb.wall_nodes)
    assert shared.size > 0, "expected the inlet rim to touch the wall"
    for nd in shared:
        assert lookup[dof * int(nd) + 2] == pytest.approx(-SPEC.inlet_speed)


def test_traction_outlet_constrains_nothing_there():
    """The natural condition adds no essential constraint; PINNED adds pressure."""
    pm = z.build_mesh(SPEC, dofs_per_node=3)
    dof = pm.mesh.nodes.dof_per_node
    free = z.build_flow_bc(pm, SPEC, z.OutletKind.TRACTION)
    pinned = z.build_flow_bc(pm, SPEC, z.OutletKind.PINNED)

    outlet_pressure_dofs = {dof * int(n) for n in free.outlet_nodes}
    assert not (outlet_pressure_dofs & set(free.bc["fixed_dofs"].tolist()))
    assert outlet_pressure_dofs <= set(pinned.bc["fixed_dofs"].tolist())


# -- formulation options ----------------------------------------------------


def _one_element_solver(form, tau_scale=0.0, viscosity=1.0, density=1.0):
    """A single unit Quad4 with everything but the viscous term switched off."""
    import toflux.src.bc as tf_bc
    import toflux.src.mesher as tf_mesher

    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    mesh = tf_mesher.Mesh(
        nodes=tf_mesher.Nodes(coords=coords, dof_per_node=3),
        elem_nodes=np.array([[0, 1, 2, 3]]),
        elem_template=elements.Quad4(),
        gauss_order=2,
    )
    bc = tf_bc.BCDict(
        elem_forces=jnp.zeros((1, mesh.num_dofs_per_elem)),
        fixed_dofs=np.zeros(0, dtype=int),
        free_dofs=np.arange(mesh.num_dofs),
        dirichlet_values=np.zeros(0),
    )
    return fe_flow.FlowSolver(
        mesh,
        bc,
        density=density,
        viscosity=viscosity,
        solver_settings={"linear": {}, "nonlinear": {}},
        elem_length=jnp.ones(1),
        tau_scale=tau_scale,
        form=form,
    )


def _shear_residual(form):
    """Viscous residual for u = (y, 0): pure shear, divergence free.

    This field also makes the convective term vanish identically --
    (u.grad)u = (y * du_x/dx, 0) = 0 -- so with `tau_scale = 0` and alpha = 0
    the element residual is the viscous term alone, whatever the density is.

    grad u = [[0, 1], [0, 0]] (du_x/dy = 1), so

        laplacian  dN/dx_d du_i/dx_d      -> x: dN/dy,  y: 0
        symmetric  dN/dx_d (du_d/dx_i + du_i/dx_d)
                                          -> x: dN/dy,  y: dN/dx

    The y component is what separates them.
    """
    solver = _one_element_solver(form)
    coords = np.asarray(solver.mesh.elem_node_coords[0])
    press_vel = jnp.asarray(
        np.stack([np.zeros(4), coords[:, 1], np.zeros(4)], axis=1).ravel()
    )
    res = solver.element_residual(press_vel, 0.0, solver.mesh.elem_node_coords[0], 1.0)
    return np.asarray(res).reshape(4, 3)


def test_laplacian_and_symmetric_viscous_forms_differ_as_derived():
    lap = _shear_residual(
        fe_flow.FlowForm(viscous_form="laplacian", brinkman_in_supg_residual=False)
    )
    sym = _shear_residual(
        fe_flow.FlowForm(viscous_form="symmetric", brinkman_in_supg_residual=False)
    )
    # x-momentum identical, y-momentum only in the symmetric form
    assert np.allclose(lap[:, 1], sym[:, 1], atol=1e-14)
    assert np.allclose(lap[:, 2], 0.0, atol=1e-14)
    assert np.linalg.norm(sym[:, 2]) > 0.1

    # and the symmetric y-residual is the integral of dN/dx, which on the unit
    # square is +-1/2 per node.
    assert np.allclose(np.sort(sym[:, 2]), np.sort([-0.5, 0.5, 0.5, -0.5]), atol=1e-12)


def test_tau_forms_agree_with_the_closed_form():
    """Both tau forms match their definitions exactly, at ANY parameters."""
    vel = jnp.zeros((4, 2))
    alpha, h, rho, mu = 2.0e5, 0.1, 1000.0, 1.0
    zhou = _one_element_solver(fe_flow.ZHOU_FORM, tau_scale=1.0, density=rho)
    zhao = _one_element_solver(fe_flow.ZHAO_FORM, tau_scale=1.0, density=rho)

    inv_diff = 12.0 * mu / (rho * h**2)
    inv_react = alpha / rho
    assert float(zhou._tau(vel, alpha, h)) == pytest.approx(
        (inv_diff**2 + inv_react**2) ** -0.5, rel=1e-12
    )
    assert float(zhao._tau(vel, alpha, h)) == pytest.approx(1.0 / inv_diff, rel=1e-12)


def test_dropping_the_reactive_limit_is_negligible_at_the_papers_scale():
    """The switch ratio is bounded, and at Zhao's own numbers the bound is 1.4%.

    Guards a retracted claim. A ~167x ratio was once quoted for this switch; it
    came from mu = 1.0, a thousand times Zhao's table 1 value. The ratio has a
    closed form that needs no solver:

        tau_0 / tau_r = sqrt(1 + (alpha tau_0 / rho)^2)
                      <= sqrt(1 + (alpha h^2 / 12 mu)^2)     since tau_0 <= rho h^2 / 12 mu

    At alpha = 2e5 (the reference field's value at gamma = 0.4, alpha_max = 1e6)
    and Zhao's mu = 0.001 that bound is 1.0138 for the element edge and 1.0541
    for the diagonal.
    """
    rho, mu = SPEC.fluid_density, SPEC.fluid_viscosity
    alpha = 0.2 * SPEC.alpha_max_initial  # RAMP factor at gamma = 0.4, q = 0.2
    zhou = _one_element_solver(fe_flow.ZHOU_FORM, tau_scale=1.0, density=rho,
                               viscosity=mu)
    zhao = _one_element_solver(fe_flow.ZHAO_FORM, tau_scale=1.0, density=rho,
                               viscosity=mu)

    for h, expected_bound in (
        (SPEC.element_size, 1.0138),
        (SPEC.element_size * np.sqrt(2.0), 1.0541),
    ):
        for speed in (0.0, 0.02, 0.2):
            vel = jnp.full((4, 2), speed / np.sqrt(2.0))
            ratio = float(zhao._tau(vel, alpha, h)) / float(zhou._tau(vel, alpha, h))
            bound = np.sqrt(1.0 + (alpha * h**2 / (12.0 * mu)) ** 2)
            assert bound == pytest.approx(expected_bound, rel=1e-3)
            assert 1.0 <= ratio <= bound * (1 + 1e-12), (
                f"ratio {ratio} outside [1, {bound}] at h={h}, |u|={speed}"
            )
        # at zero velocity the bound is attained
        ratio0 = float(zhao._tau(jnp.zeros((4, 2)), alpha, h)) / float(
            zhou._tau(jnp.zeros((4, 2)), alpha, h)
        )
        assert ratio0 == pytest.approx(
            np.sqrt(1.0 + (alpha * h**2 / (12.0 * mu)) ** 2), rel=1e-12
        )


def test_brinkman_in_supg_residual_changes_the_residual():
    base = fe_flow.FlowForm(brinkman_in_supg_residual=True, brinkman_in_tau=False)
    off = fe_flow.FlowForm(brinkman_in_supg_residual=False, brinkman_in_tau=False)
    press_vel = jnp.asarray(np.tile([0.0, 0.3, 0.1], 4))
    kw = dict(tau_scale=1.0, density=1000.0)
    r_on = _one_element_solver(base, **kw).element_residual(
        press_vel, 2.0e5, jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), 1.0
    )
    r_off = _one_element_solver(off, **kw).element_residual(
        press_vel, 2.0e5, jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), 1.0
    )
    assert not np.allclose(r_on, r_off)
