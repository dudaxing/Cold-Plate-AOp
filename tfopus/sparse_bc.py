"""Dirichlet application that scales to 3D meshes.

Upstream's `toflux.src.bc.apply_dirichlet_bc` selects the constrained entries
with `jnp.isin(row_indices, fixed_dofs)`, which materialises an
(nnz x num_fixed) boolean array. That is fine for upstream's 2D examples and
impossible here: the cylinder case has 15.2 M nonzeros and 47 k constrained
dofs, i.e. 714 GB. It is the reason a 33 k-dof duct run dies with
RESOURCE_EXHAUSTED while a 7 k-dof one passes.

A boolean lookup table over the dofs costs O(num_dofs) to build and O(nnz) to
apply, and is otherwise identical: zero the constrained rows and columns, then
append unit diagonal entries, relying on BCOO summing duplicate indices.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.experimental.sparse as jsparse


def apply_dirichlet_bc(jacobian: jsparse.BCOO, fixed_dofs) -> jsparse.BCOO:
    """Zero the constrained rows/columns and put 1 on their diagonal."""
    data, indices = jacobian.data, jacobian.indices
    num_dofs = jacobian.shape[0]
    fixed = jnp.asarray(fixed_dofs)

    is_fixed = jnp.zeros(num_dofs, dtype=bool).at[fixed].set(True)
    touched = is_fixed[indices[:, 0]] | is_fixed[indices[:, 1]]
    data = jnp.where(touched, 0.0, data)

    diag = jnp.stack([fixed, fixed], axis=1)
    return jsparse.BCOO(
        (
            jnp.concatenate([data, jnp.ones(fixed.shape[0], dtype=data.dtype)]),
            jnp.concatenate([indices, diag]),
        ),
        shape=jacobian.shape,
    )
