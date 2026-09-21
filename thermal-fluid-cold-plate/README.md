# Cold-Plate-AOp

Density-based topology optimization for thermal–fluid problems, built on
[TOFLUX](https://github.com/UW-ERSL/TOFLUX)'s differentiable JAX finite-element
core.

Two published case sets are being reproduced with a **density parametrisation**.
In both, the original feature-driven parametrisation is deliberately *not*
implemented — everything downstream of the pseudo-density (interpolations,
governing equations, objective, constraint) is kept.

| Target | Original parametrisation | Reproduced as | Status |
|---|---|---|---|
| Zhou et al., *Appl. Sci.* **16**, 7255 (2026) — conformal cooling | BSOF B-spline offset surfaces | per-surface-column solid fraction swept through the wall | geometry + meshes done |
| Zhao et al., *Appl. Therm. Eng.* **291** (2026) 130088 — cold plate / heat sink | CBS closed B-spline features | per-element solid fraction | 2D fixed-design analysis done |

Per-case detail, including the reconstruction choices and the gaps found in each
paper: [`docs/zhou_reproduction.md`](docs/zhou_reproduction.md),
[`docs/zhao_reproduction.md`](docs/zhao_reproduction.md).

## Layout

| Path | Contents |
|---|---|
| `tfopus/` | the library: corrected elements, dimension-agnostic stabilised flow and thermal kernels, materials and interpolations, meshes, boundary conditions, design mapping |
| `tfopus/zhao2d*.py` | Zhao's 2D heat sink: geometry reconstruction and fixed-design analysis |
| `validation/` | the test suite — analytic solutions, gradient checks, formulation comparisons |
| `scripts/` | upstream checkout, and the reference studies whose numbers the docs quote |
| `docs/` | per-case reconstruction notes and findings |

Zhao's and Zhou's discretisations differ in three places — the stabilisation
parameter's reactive limit, whether αu enters the SUPG residual, and symmetric
versus Laplacian viscous form. These are explicit switches
(`tfopus.fe_flow.FlowForm`, `tfopus.fe_thermal.ThermalForm`) with a named
constant per paper, not inherited defaults. Measured at the figure-7 scale,
only the second one moves the answer (−14% in Ψ); the other two are under 0.2%.
See `scripts/zhao2d_form_attribution.py`.

## Findings so far

**Zhao's reported normalisation constants are transposed.** Section 4.1 gives
Ψ₀ = 20,816 and C₀ = 0.0456; computing them on the geometry figure 7 prints
gives those two magnitudes the other way round. Across the 16 combinations of
the choices the paper leaves open, the best agreement with the labels as
printed is off by 4 × 10⁵, and the best agreement with them swapped is 1.06 —
both constants within 6% simultaneously. Details and the rejected alternative
(a length-scale reading) are in
[`docs/zhao_reproduction.md`](docs/zhao_reproduction.md).

**Upstream TOFLUX has four defects** that the validation suite pins down, two of
which only surface on meshes that are not axis-aligned boxes. They are applied
as source substitutions against a pristine checkout rather than a fork, so the
suite fails loudly if upstream changes those lines. See
`validation/variants.py` and `tfopus/elements.py`.

## Setup

Upstream TOFLUX ships without a LICENSE file, so it is **not** vendored here.
`scripts/setup_toflux.py` extracts it from the paper's supplementary-material
archive into `external/` (gitignored) and makes its `petsc4py` import optional —
upstream imports it at module scope without declaring it as a dependency.

```bash
pip install -r requirements-dev.txt
python scripts/setup_toflux.py --zip /path/to/158_2026_4252_MOESM1_ESM.zip
pytest                                   # add -m "not slow" to skip refinement studies
```

`TOFLUX_ZIP` sets the archive path and `TOFLUX_ROOT` the checkout location. The
suite uses SciPy's sparse direct solver, so PETSc and PARDISO are not needed.

## Reproducing the reported studies

```bash
python scripts/zhao2d_reference_study.py --provenance   # the option sweep
python scripts/zhao2d_swap_test.py                      # the transposition test
```

`Zhao2DSpec.provenance()` prints, per field, whether a number comes from the
paper, is derived, or is a reconstruction choice — plus the list of gaps the
paper leaves open.
