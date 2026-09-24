"""Mesh refinement carrying the same design, and the repaired heat diagnostic.

A refinement study only means something if the design really is unchanged. The
transfer is therefore checked on the quantities that must be invariant -- area,
the design/tab partition, both fluid fractions -- rather than assumed from the
fact that the code ran.
"""

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_refine as ref  # noqa: E402

SPEC = dataclasses.replace(z.Zhao2DSpec(), element_size=5.0e-4)


@pytest.fixture(scope="module")
def meshes():
    fine_spec = ref.refine_spec(SPEC, 2)
    return (
        z.build_mesh(SPEC, dofs_per_node=1),
        z.build_mesh(fine_spec, dofs_per_node=1),
        fine_spec,
    )


def _random_design(planar, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.0, 1.0, planar.num_elems)
    s[~planar.design_mask] = 0.0
    return jnp.asarray(s)


# -- the refinement itself --------------------------------------------------


def test_refine_spec_only_changes_the_element_size():
    fine = ref.refine_spec(SPEC, 2)
    assert fine.element_size == SPEC.element_size / 2
    for f in dataclasses.fields(SPEC):
        if f.name != "element_size":
            assert getattr(fine, f.name) == getattr(SPEC, f.name)


def test_each_coarse_cell_has_exactly_four_children(meshes):
    coarse, fine, _ = meshes
    parents = ref.parent_of_each_fine_element(coarse, fine)
    counts = np.bincount(parents, minlength=coarse.num_elems)
    assert set(counts.tolist()) == {4}
    assert fine.num_elems == 4 * coarse.num_elems


def test_transfer_preserves_area_partition_and_both_fractions(meshes):
    coarse, fine, _ = meshes
    s_c = _random_design(coarse)
    s_f = ref.refine_design(coarse, fine, s_c)
    report = ref.check_transfer(coarse, fine, s_c, s_f)  # raises on any change
    assert report["children_per_parent"] == [4]
    assert report["coarse_v_f_design_domain"] == pytest.approx(
        report["fine_v_f_design_domain"], rel=1e-12
    )
    assert report["coarse_v_f_whole_domain"] == pytest.approx(
        report["fine_v_f_whole_domain"], rel=1e-12
    )


def test_every_child_carries_its_parents_value_exactly(meshes):
    coarse, fine, _ = meshes
    s_c = _random_design(coarse, seed=3)
    s_f = np.asarray(ref.refine_design(coarse, fine, s_c))
    parents = ref.parent_of_each_fine_element(coarse, fine)
    assert np.array_equal(s_f, np.asarray(s_c)[parents])


def test_tabs_stay_fluid_through_refinement(meshes):
    coarse, fine, _ = meshes
    s_f = ref.refine_design(coarse, fine, _random_design(coarse))
    assert np.allclose(np.asarray(s_f)[~fine.design_mask], 0.0)


def test_check_transfer_rejects_a_tampered_design(meshes):
    coarse, fine, _ = meshes
    s_c = _random_design(coarse)
    s_f = np.array(ref.refine_design(coarse, fine, s_c))  # copy: jax arrays are read-only
    s_f[int(np.nonzero(fine.design_mask)[0][0])] = 1.0  # one cell moved
    with pytest.raises(RuntimeError, match="v_f"):
        ref.check_transfer(coarse, fine, s_c, jnp.asarray(s_f))


def test_threshold_then_refine_equals_refine_then_threshold_here(meshes):
    """Both orders agree because the transfer is piecewise constant.

    Worth pinning: it is the reason a binary design can be defined on the parent
    mesh and still be exactly representable on the child mesh, so the two
    analyses differ only by discretisation and not by the polygon.
    """
    coarse, fine, _ = meshes
    s_c = _random_design(coarse, seed=7)
    a = ref.refine_design(coarse, fine, ref.threshold_design(coarse, s_c, 0.5))
    b = ref.threshold_design(fine, ref.refine_design(coarse, fine, s_c), 0.5)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_refine_spec_rejects_a_non_integer_factor():
    with pytest.raises(ValueError):
        ref.refine_spec(SPEC, 1.5)


# -- undershoot -------------------------------------------------------------


def test_undershoot_counts_only_below_the_inlet_value():
    t = jnp.array([0.0, 1.0, -0.25, -1e-14, 5.0])
    out = ref.undershoot(t, inlet_temperature=0.0)
    assert out["nodes_below_inlet"] == 1  # -1e-14 is inside the tolerance
    assert out["min_temperature"] == pytest.approx(-0.25)
    assert out["undershoot_depth"] == pytest.approx(0.25)


# -- the repaired boundary flux diagnostic ----------------------------------


def test_face_midpoints_are_read_off_the_connectivity(meshes):
    """The inverse isoparametric map is not available and is not needed."""
    coarse, _, _ = meshes
    mid = za.face_midpoints_local(coarse.mesh.elem_template)
    assert np.allclose(
        np.sort(mid, axis=0),
        np.sort(np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), axis=0),
    )


def test_upstream_inverse_map_still_raises(meshes):
    """Pin why the midpoints are derived rather than inverted.

    If upstream ever implements this, the derivation stays correct -- but this
    test failing is the signal to re-read that decision.
    """
    coarse, _, _ = meshes
    with pytest.raises(NotImplementedError):
        coarse.mesh.elem_template.get_isoparametric_coordinate_of_point(
            jnp.array([0.0, 0.0]), coarse.mesh.elem_node_coords[0]
        )


def test_conductive_outflow_is_exact_for_a_linear_field(meshes):
    """T = a + b.x has a constant gradient, so the flux integral is analytic.

    With kappa = 1 and a closed boundary, -integral grad T . n over the whole
    boundary is zero for any linear field by the divergence theorem. This is the
    check that the repaired path integrates the right thing, not merely that it
    stops raising.
    """
    coarse, _, _ = meshes
    nodes = np.asarray(coarse.mesh.nodes.coords)
    temp = 3.0 + 2.0 * nodes[:, 0] - 1.5 * nodes[:, 1]
    total = za._conductive_outflow(
        coarse, temp, np.ones(coarse.num_elems)
    )
    scale = 2.5 * np.sqrt(np.asarray(coarse.elem_area)[0]) * coarse.num_elems
    assert abs(total) < 1e-10 * max(scale, 1.0)


def test_temperature_weighted_divergence_vanishes_for_a_uniform_field(meshes):
    """u constant => div u = 0 pointwise, so the term is zero for any T."""
    coarse, _, _ = meshes
    n = coarse.mesh.num_nodes
    press_vel = np.zeros((n, 3))
    press_vel[:, 1] = 0.7  # uniform u_x
    press_vel[:, 2] = -0.3  # uniform u_y
    temp = np.asarray(coarse.mesh.nodes.coords)[:, 0] * 11.0 + 4.0
    value = za.temperature_weighted_divergence(
        coarse, jnp.asarray(press_vel.ravel()), jnp.asarray(temp), b_f=4.18e6
    )
    assert abs(value) < 1e-6
