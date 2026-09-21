"""Z0: the swept shell meshes are geometrically what they claim to be.

Nothing downstream is meaningful if the mesh has negative Jacobians, loses
volume, or mislabels a boundary. These are the contract for the case geometries.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tfopus import elements as E
from tfopus import geometry as G
from tfopus import mesh as M

R, T = 0.0195, 0.0005


def _total_volume(sm):
    return float(np.sum(np.asarray(sm.mesh.elem_volume)))


def _plain(param_map, p1, p2, n_thick=2):
    return M.build_shell_mesh(
        param_map, p1, p2, n_thick, 4,
        lambda a, b: np.zeros_like(a, dtype=int),
        lambda ax, sd, x, q: M.Face.WALL,
    )


# --------------------------------------------------------------------------
# Raw swept meshes against closed-form volumes
# --------------------------------------------------------------------------


def test_cylinder_volume_converges_second_order():
    """Straight-edged elements under-fill a curved sector by the chord deficit.

    The deficit is a discretisation error, not a defect, so what matters is that
    it falls as O(h^2).
    """
    arc, L = np.pi, 0.010
    exact = 0.5 * arc * ((R + T) ** 2 - R ** 2) * L
    errs = []
    for n in (24, 48, 96):
        sm = _plain(M.cylinder_map(R, T), np.linspace(0, arc, n + 1), np.linspace(0, L, 9))
        errs.append(abs(_total_volume(sm) - exact) / exact)
    rates = [np.log2(a / b) for a, b in zip(errs, errs[1:])]
    assert all(1.8 < r < 2.2 for r in rates), f"rates {rates} from errors {errs}"


def test_sphere_volume_converges_second_order():
    A = np.radians(60.0)
    # solid angle of an equiangular gnomonic square patch of half-angle A
    tA = np.tan(A)
    omega = 4.0 * np.arcsin(tA ** 2 / (1.0 + tA ** 2))
    exact = omega / 3.0 * ((R + T) ** 3 - R ** 3)
    errs = []
    for n in (20, 40, 80):
        g = np.linspace(-A, A, n + 1)
        sm = _plain(M.spherical_cap_map(R, T), g, g)
        errs.append(abs(_total_volume(sm) - exact) / exact)
    rates = [np.log2(a / b) for a, b in zip(errs, errs[1:])]
    assert all(1.8 < r < 2.2 for r in rates), f"rates {rates} from errors {errs}"


def test_sphere_nodes_lie_on_the_two_spheres():
    A = np.radians(60.0)
    g = np.linspace(-A, A, 21)
    sm = _plain(M.spherical_cap_map(R, T), g, g, n_thick=2)
    radii = np.linalg.norm(np.asarray(sm.mesh.nodes.coords), axis=1)
    assert radii.min() == pytest.approx(R, rel=1e-12)
    assert radii.max() == pytest.approx(R + T, rel=1e-12)


def test_left_handed_parametrisation_is_rejected():
    """A negative det(J) would silently flip the sign of the whole residual."""
    arc = np.pi
    with pytest.raises(ValueError, match="left-handed"):
        _plain(M.cylinder_map(R, T), np.linspace(arc, 0, 9), np.linspace(0, 0.01, 5))


def test_equiangular_grading_keeps_element_size_uniform():
    """Grading the sphere in tangent space would stretch elements by (1+tan^2)."""
    sm = G.build_sphere(G.SphereSpec(), 4)
    d = jax.vmap(E.Hex8.diag_length)(sm.mesh.elem_node_coords)
    assert float(d.max() / d.min()) < 1.6


# --------------------------------------------------------------------------
# Case geometries
# --------------------------------------------------------------------------

CASES = {
    "cylinder_A": lambda: G.build_cylinder(G.CylinderSpec(), 4),
    "cylinder_B": lambda: G.build_cylinder(G.CylinderSpec(axial_scope="full"), 4),
    "sphere": lambda: G.build_sphere(G.SphereSpec(), 4),
}


@pytest.fixture(scope="module", params=sorted(CASES))
def case_mesh(request):
    return request.param, CASES[request.param]()


def test_all_element_volumes_positive(case_mesh):
    _, sm = case_mesh
    assert float(np.asarray(sm.mesh.elem_volume).min()) > 0.0


def test_jacobian_positive_at_every_gauss_point(case_mesh):
    """Positive volume is not enough; the signed det enters the residual."""
    _, sm = case_mesh
    _, det = jax.vmap(
        jax.vmap(E.Hex8.compute_jacobian_and_determinant, in_axes=(0, None)),
        in_axes=(None, 0),
    )(sm.mesh.gauss_pts, sm.mesh.elem_node_coords)
    assert float(jnp.min(det)) > 0.0


def test_every_boundary_face_is_tagged_exactly_once(case_mesh):
    _, sm = case_mesh
    tagged = [f for v in sm.elem_faces.values() for f in v]
    assert len(tagged) == len(set(tagged)), "a face got two tags"
    assert len(tagged) == int(np.asarray(sm.mesh.boundary_faces).sum())
    assert sm.elem_faces[M.Face.NONE] == []


def test_inlet_and_outlet_are_present_and_equal_sized(case_mesh):
    _, sm = case_mesh
    n_in = len(sm.elem_faces[M.Face.INLET])
    n_out = len(sm.elem_faces[M.Face.OUTLET])
    assert n_in > 0 and n_in == n_out


def test_regions_partition_the_mesh(case_mesh):
    _, sm = case_mesh
    counts = [int((sm.region == r).sum()) for r in M.Region]
    assert sum(counts) == sm.num_elems
    assert all(c > 0 for c in counts), "a region is empty"


def test_design_region_is_column_consistent(case_mesh):
    """Region is a surface property: a column is all design or all not."""
    _, sm = case_mesh
    for r in M.Region:
        cols = np.unique(sm.surface_id[sm.region == r])
        other = np.unique(sm.surface_id[sm.region != r])
        assert not np.intersect1d(cols, other).size, f"{r.name} splits a column"


def test_column_volumes_sum_to_the_mesh_volume(case_mesh):
    _, sm = case_mesh
    assert sm.column_volume.sum() == pytest.approx(_total_volume(sm), rel=1e-12)
    assert sm.column_volume.min() > 0.0


def test_cylinder_A_matches_the_reported_dof_count():
    """Zhou reports ~110,070 mesh DOFs. This is what fixes the reconstruction.

    180 degree arc, 10 mm modelled axially, 0.3 mm surface mesh, 2 layers
    through the wall, 4 flow dofs + 1 thermal dof per node.
    """
    sm = CASES["cylinder_A"]()
    dofs = 5 * sm.mesh.num_nodes
    assert abs(dofs - 110_070) / 110_070 < 0.06, f"{dofs} dofs"
