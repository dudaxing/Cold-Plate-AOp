"""The element corrections in `tfopus.elements`, against upstream TOFLUX.

Each test states what upstream does, so that if upstream is ever fixed these
turn into redundant-but-passing checks rather than silent confusion.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import toflux.src.element as upstream
from tfopus import elements as fixed

# A unit cube and a sheared (affine, non-axis-aligned) hex, in Hex8 node order.
CUBE = jnp.array(
    [[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
)
REF_HEX = jnp.array(
    [[-1.0, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
)
SHEAR3 = jnp.array([[1.0, 0.3, 0.1], [0.0, 1.1, 0.25], [0.2, 0.0, 0.9]])
REF_QUAD = jnp.array([[-1.0, -1], [1, -1], [1, 1], [-1, 1]])
SHEAR2 = jnp.array([[1.0, 0.4], [0.25, 1.2]])


def _patch_residual(template, nodes, gauss_pt):
    """max |sum_n gradN[n,:] x_n - I|.

    Reproducing the gradient of the linear field f_i(x) = x_i exactly is the
    minimum any isoparametric gradient must satisfy.
    """
    grad = template.get_gradient_shape_function_physical(gauss_pt, nodes)
    dim = nodes.shape[1]
    return float(jnp.max(jnp.abs(jnp.einsum("ni, nj -> ji", grad, nodes) - jnp.eye(dim))))


def test_hex8_volume_upstream_double_counts_a_tetrahedron():
    """Upstream's tet table lists vertex set (1,4,5,6) twice, giving 7/6."""
    sets = [tuple(sorted(t)) for t in
            [[0, 1, 3, 4], [1, 2, 3, 6], [1, 5, 4, 6], [4, 6, 7, 3], [1, 6, 3, 4], [4, 5, 6, 1]]]
    assert len(set(sets)) == 5, "upstream tet table is no longer degenerate"
    assert float(upstream.Hex8.elem_volume(CUBE)) == pytest.approx(7.0 / 6.0)


def test_hex8_volume_is_corrected():
    assert float(fixed.Hex8.elem_volume(CUBE)) == pytest.approx(1.0, abs=1e-12)
    sheared = REF_HEX @ SHEAR3.T
    exact = 8.0 * abs(float(jnp.linalg.det(SHEAR3)))
    assert float(fixed.Hex8.elem_volume(sheared)) == pytest.approx(exact, rel=1e-12)


@pytest.mark.parametrize(
    "up, fix, ref, shear, gp",
    [
        (upstream.Hex8(), fixed.Hex8(), REF_HEX, SHEAR3, jnp.array([0.1, -0.2, 0.3])),
        (upstream.Quad4(), fixed.Quad4(), REF_QUAD, SHEAR2, jnp.array([0.2, -0.1])),
    ],
    ids=["hex8", "quad4"],
)
def test_gradient_patch_test(up, fix, ref, shear, gp):
    """Upstream fails the patch test on sheared elements; the fix passes it.

    Quad4 is included deliberately: the same defect is there, and it is
    invisible in all of upstream's 2D work only because GridMesh and
    grid_mesh_brep emit axis-aligned rectangles.
    """
    nodes = ref @ shear.T
    assert _patch_residual(up, nodes, gp) > 1e-2
    assert _patch_residual(fix, nodes, gp) < 1e-12


def test_gradient_is_correct_upstream_on_axis_aligned_elements():
    """Why upstream's 2D results are unaffected: a diagonal Jacobian is symmetric."""
    rect = jnp.array([[0.0, 0], [2, 0], [2, 3], [0, 3]])
    assert _patch_residual(upstream.Quad4(), rect, jnp.array([0.2, -0.1])) < 1e-14


def test_hex8_gradient_is_a_classmethod():
    """Upstream's lacks the decorator, so it only works when called on an instance."""
    import inspect

    assert isinstance(
        inspect.getattr_static(upstream.Hex8, "get_gradient_shape_function_physical"),
        type(lambda: None),
    )
    assert isinstance(
        inspect.getattr_static(fixed.Hex8, "get_gradient_shape_function_physical"),
        classmethod,
    )
    # the fixed one works off the class, not just an instance
    fixed.Hex8.get_gradient_shape_function_physical(jnp.zeros(3), CUBE)


def test_hex8_diag_length():
    """Upstream Hex8 has no diag_length at all, which mesher.Mesh requires."""
    assert not hasattr(upstream.Hex8, "diag_length")
    assert float(fixed.Hex8.diag_length(CUBE)) == pytest.approx(np.sqrt(3.0))
    # scales linearly with the element
    assert float(fixed.Hex8.diag_length(2.0 * CUBE)) == pytest.approx(2 * np.sqrt(3.0))


def test_gradient_fix_is_the_transpose_only():
    """The fix swaps which index of inv(J) is contracted -- nothing else."""
    nodes = REF_HEX @ SHEAR3.T
    gp = jnp.array([0.1, -0.2, 0.3])
    up = upstream.Hex8().get_gradient_shape_function_physical(gp, nodes)
    fx = fixed.Hex8.get_gradient_shape_function_physical(gp, nodes)
    gN = fixed.Hex8.shape_function_gradients_isoparametric(gp)
    jac, _ = fixed.Hex8.compute_jacobian_and_determinant(gp, nodes)
    assert np.allclose(up, jnp.einsum("di, nd -> ni", jnp.linalg.inv(jac), gN))
    assert np.allclose(fx, jnp.einsum("id, nd -> ni", jnp.linalg.inv(jac), gN))
