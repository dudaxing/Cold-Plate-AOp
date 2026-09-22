"""The element adjacency graph and the fluid-connectivity diagnostic.

These exist because a broken adjacency graph does not raise, does not look
wrong, and produces a physically meaningful-sounding answer: "the thresholded
design does not connect inlet to outlet". The first version indexed cells by
`round(centre / h)`, and centres sit at (i + 1/2) h, so every key was a
half-integer and Python's banker's rounding collapsed i and i+1 for odd i. 160
of 5200 elements were silently dropped from the graph and a fully fluid domain
was reported as disconnected.

The decisive test is the degenerate one: if every cell is fluid, there is
exactly one component. No amount of staring at an optimised design would have
told us that; one line of all-True does.
"""

import dataclasses

import numpy as np
import pytest

from tfopus import zhao2d as z
from tfopus.mesh import Face

SPEC = dataclasses.replace(z.Zhao2DSpec(), element_size=5.0e-4)


@pytest.fixture(scope="module")
def planar():
    return z.build_mesh(SPEC, dofs_per_node=1)


@pytest.fixture(scope="module")
def main_mesh():
    """The real 5200-element mesh: the one the diagnostic actually ran on."""
    return z.build_mesh(z.Zhao2DSpec(), dofs_per_node=1)


def test_cell_index_does_not_collide(main_mesh):
    """Every element must get its own key. Centres are at half-integer x/h."""
    graph = z.element_adjacency(main_mesh)  # raises on collision
    assert graph.shape == (main_mesh.num_elems, main_mesh.num_elems)


def test_round_would_have_collided(main_mesh):
    """Pin the actual defect, so a 'tidy-up' back to round() fails here."""
    centres = np.asarray(main_mesh.elem_centres)
    h = float(np.sqrt(np.asarray(main_mesh.elem_area)[0]))
    with_round = {(round(x / h), round(y / h)) for x, y in centres}
    with_floor = {(int(np.floor(x / h)), int(np.floor(y / h))) for x, y in centres}
    assert len(with_floor) == main_mesh.num_elems
    assert len(with_round) < main_mesh.num_elems


@pytest.mark.parametrize("mesh_name", ["planar", "main_mesh"])
def test_all_fluid_is_exactly_one_component(request, mesh_name):
    """The test that would have caught it. The mesh itself is connected."""
    mesh = request.getfixturevalue(mesh_name)
    num, sizes, connected, inlet_ids, outlet_ids = z.fluid_connectivity(
        mesh,
        np.ones(mesh.num_elems, dtype=bool),
        np.unique([e for e, _ in mesh.elem_faces[Face.INLET]]),
        np.unique([e for e, _ in mesh.elem_faces[Face.OUTLET]]),
    )
    assert num == 1, f"a fully fluid domain split into {num} components"
    assert sizes[0] == mesh.num_elems
    assert connected
    assert inlet_ids == outlet_ids == [0]


def test_all_solid_has_no_fluid_and_no_path(planar):
    num, sizes, connected, inlet_ids, outlet_ids = z.fluid_connectivity(
        planar,
        np.zeros(planar.num_elems, dtype=bool),
        np.unique([e for e, _ in planar.elem_faces[Face.INLET]]),
        np.unique([e for e, _ in planar.elem_faces[Face.OUTLET]]),
    )
    assert num == 0 and not connected
    assert inlet_ids == [] and outlet_ids == []


def test_edge_count_matches_a_structured_grid(planar):
    """Interior faces of the masked grid, counted independently.

    A graph that dropped cells would also drop edges, so this cross-checks the
    adjacency against the mesh's own boundary-face bookkeeping:

        2 * interior_faces = sum over cells of (4 - boundary faces of that cell)
    """
    graph = z.element_adjacency(planar)
    boundary_per_cell = np.asarray(planar.mesh.boundary_faces).sum(axis=1)
    expected_directed = int((4 - boundary_per_cell).sum())
    assert graph.nnz == expected_directed


def test_a_severed_column_disconnects_the_ports(planar):
    """A solid wall across the full width must break every path."""
    centres = np.asarray(planar.elem_centres)
    h = SPEC.element_size
    mid = SPEC.design_height / 2.0
    fluid = ~(np.abs(centres[:, 1] - mid) < h)  # one solid row across everything
    _, _, connected, _, _ = z.fluid_connectivity(
        planar,
        fluid,
        np.unique([e for e, _ in planar.elem_faces[Face.INLET]]),
        np.unique([e for e, _ in planar.elem_faces[Face.OUTLET]]),
    )
    assert not connected


def test_diagonal_contact_is_not_a_connection(planar):
    """Two cells meeting at a corner share one node and carry no channel."""
    centres = np.asarray(planar.elem_centres)
    h = SPEC.element_size
    idx = {
        (int(np.floor(x / h)), int(np.floor(y / h))): i
        for i, (x, y) in enumerate(centres)
    }
    graph = z.element_adjacency(planar)
    (i, j) = next(k for k in idx if (k[0] + 1, k[1] + 1) in idx)
    a, b = idx[(i, j)], idx[(i + 1, j + 1)]
    assert graph[a, b] == 0, "diagonal neighbours must not be adjacent"
    assert graph[a, idx[(i + 1, j)]] == 1, "face neighbours must be adjacent"
