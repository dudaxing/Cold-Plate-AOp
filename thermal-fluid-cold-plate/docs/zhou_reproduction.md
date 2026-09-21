# Zhou conformal-cooling cases, density method

Reproducing Zhou et al., *Appl. Sci.* **16**, 7255 (2026),
[doi:10.3390/app16147255](https://doi.org/10.3390/app16147255) with TOFLUX's
density method. The feature-driven parametrisation (BSOF, B-spline control
points, offset radii) is deliberately **not** implemented: the design field is a
per-surface-column solid fraction, swept through the wall.

Scope: 10 optimizations (7 cylinder weights, 2 cylinder initialisations, 1
sphere) plus fixed-geometry comparisons (straight channel, sphere initial
reference).

## Stage status

| Stage | Content | State |
|---|---|---|
| Z0 | Geometry interpretation, swept meshes, region/face tagging | **done** |
| Z1 | 3D flow + thermal kernels, source term, coupled gradient | **in progress** |
| Z2 | Cylinder w = 0.92 end-to-end | |
| Z3 | Remaining cylinder weights, initialisations, straight channel | |
| Z4 | Sphere w = 0.95 and its initial reference | |

## Geometry reconstruction

Zhou gives the radius, thickness and mesh size but never the circumferential
angle, and the axial 10 mm is ambiguous between body text and figure 3. The
reported **~110,070 mesh DOFs** is what pins it down:

| interpretation | grid | nodes | DOFs (5/node) |
|---|---|---|---|
| **A**: 180° arc × 10 mm modelled (full 20 mm) | 218 × 34 × 2 | 22,995 | **114,975** |
| **B**: 10 mm is the full length → model 5 mm | 218 × 17 × 2 | 11,826 | 59,130 |

A is within 4.5% of the reported count; B is half of it. Both are built, A as the
main line and B as a scope comparison, per the user's decision.

The sphere uses an **equiangular gnomonic (cubed-sphere) patch**, not a polar
cap: the polar chart collapses a whole parametric edge onto the pole, giving
zero-volume elements. Grading in angle rather than tangent keeps element size
uniform (size ratio 1.46 across the patch instead of ~4).

## Units and conventions

- Zhou's γ = 1 is **fluid**; TOFLUX's s = 1 is **solid**. γ = 1 − s.
- RAMP: Zhou's qα and TOFLUX's convex-RAMP q are reciprocal, **q_T = 1/q_Z**.
  The sphere's qα = q_k = 0.2 is q_T = 5 in TOFLUX's form.
- Zhou Ψ (eq 25) is exactly **2×** TOFLUX's `compute_elem_dissipated_power`.
- Zhou τ_u (eq 20) is **identical** to TOFLUX's τ — so upstream's τ₁ defect
  applies directly.
- Zhou's C (eq 24) excludes the SUPG matrix S_T; only K_c,T + K_v,T.

## Recorded paper gaps

`CylinderSpec.provenance()` and `SphereSpec.provenance()` print, per field,
whether a number is from the paper or a reproduction choice, plus the list of
gaps. Notably: the sphere's section 4.2 average temperature (303.98 K) exceeds
table 6's maximum (300.99 K); both are recorded, neither is a target.


## Z1: what the stabilisation study established

The 3D flow kernel converges (Newton in 8-9 iterations to 1e-12), but the
characteristic length `h_e` that feeds the stabilisation parameters turned out
to set the accuracy ORDER, not just a constant. Zhou names h_e and never defines
it. Plane-Poiseuille duct, relative velocity error on the mid plane:

| Re | h_e | 20x2x4 | 40x4x8 | 80x8x16 | order |
|---|---|---|---|---|---|
| 0.15 | body diagonal | 8.44e-1 | 5.62e-1 | 2.41e-1 | ~0.9 |
| 0.15 | **shortest edge** | 6.20e-2 | 1.56e-2 | 3.91e-3 | **1.99 / 2.00** |
| 150 | body diagonal | 1.83e-1 | 9.59e-2 | 5.04e-2 | 0.93 / 0.93 |
| 150 | **shortest edge** | 1.88e-2 | 7.78e-3 | 2.75e-3 | 1.27 / 1.50 |
| 1500 | body diagonal | 1.80e-2 | **1.25e+01 diverged** | 5.27e-3 | - |
| 1500 | **shortest edge** | 1.95e-3 | 1.03e-3 | 5.49e-4 | 0.92 / 0.91 |

Two separate results. First, the body diagonal is not merely less accurate: at
Re = 1500 it produced dp/dx = +3.44e6 against an analytic -1.2e4 on the middle
mesh of the sequence, between two meshes that look fine. That is a robustness
failure a single-resolution run cannot detect. The shortest edge is stable
across all three Reynolds numbers and all three meshes, and is now the default.

Second, the shortest edge recovers second order only in the DIFFUSIVE regime.
At Re = 150 the rate is 1.27-1.50 and at Re = 1500 it is ~0.91. This matches
the truncated strong residual: where tau is the convective limit h/(2|u|) the
truncation enters at O(tau) = O(h), and where tau is the diffusive limit
rho h^2/(12 mu) it enters at O(h^2). Zhou's cases run near Re = 200, i.e. in the
1.3-1.5 band. **Recorded as a limitation, not a target.**

Adding the viscous term back on this box mesh would change nothing: a trilinear
field on an axis-aligned box has identically zero Laplacian. Only a sheared or
swept mesh can discriminate, which is what the outstanding thermal verification
is for.

The real shell meshes are anisotropic enough for this to matter:

| | diagonal / min edge, median | tau_3 inflation |
|---|---|---|
| cylinder A | 1.96 | 3.9x |
| sphere | 2.79 (max 3.32) | 7.8x (max 11.0x) |

## Z1: upstream blocker found

`toflux.src.bc.apply_dirichlet_bc` selects constrained entries with
`jnp.isin(row_indices, fixed_dofs)`, materialising an (nnz x num_fixed) boolean
array. For the cylinder case that is 15.2 M x 47 k = **714 GB**; a 33 k-dof duct
run already dies with RESOURCE_EXHAUSTED. `tfopus/sparse_bc.py` replaces it with
a boolean lookup table over the dofs, O(nnz), bit-identical output.
