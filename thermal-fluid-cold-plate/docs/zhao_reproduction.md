# Zhao 2D heat sink, density method

Reproducing Zhao et al., *Applied Thermal Engineering* **291** (2026) 130088,
[doi:10.1016/j.applthermaleng.2026.130088](https://doi.org/10.1016/j.applthermaleng.2026.130088)
with a density-based parametrisation. The CBS feature parametrisation, the
adaptive insertion strategy and the Case 1–13 algorithm comparison are
deliberately **not** implemented: the design field is a per-element solid
fraction. Everything downstream of the pseudo-density — interpolations,
governing equations, objective, constraint — is kept.

## Stage status

| Stage | Content | State |
|---|---|---|
| R0 | 2D fixed-design analysis, reference-scale verification | **done** |
| R1a | frozen configuration and normalisation | **done** |
| R1b | total-gradient verification | **done** — worst AD/FD 1.7e-7 |
| R1c | short MMA trial run (mechanism check) | **done** |
| R1d | full 2D optimisation | awaiting release |
| R2 | 3D extruded analysis, straight-channel reference (fig 15) | |
| R3 | 3D cold plate optimisation | |

R1d is deliberately not started: the short run is a mechanism check stopped at a
fixed iteration count, and the filter radius has not been chosen by a
length-scale study (see `R1Config.filter_radius_elements`).

## R0 headline: the reported Ψ₀ and C₀ are transposed

Section 4.1 reports, for a uniform γ = 0.4 reference model,
**Ψ₀ = 20,816** and **C₀ = 0.0456**. This implementation, on the geometry
figure 7 prints, gets values of those two magnitudes — the other way round.

`scripts/zhao2d_swap_test.py` sweeps the 16 combinations of the four choices
section 4.1 leaves open and scores each one against both labellings. The score
is the worst of the two ratios, so 1.00 is a perfect match either way:

| labelling | best worst-of-two ratio |
|---|---|
| as printed | **4.0 × 10⁵** |
| swapped | **1.06** |

The best swapped match is γ = 0.4 over the **whole** domain, α_max = 10⁶, the
heat source over the whole domain, and the stabilisation parameter **including**
its reactive limit:

| quantity | computed | paper | ratio |
|---|---|---|---|
| Ψ | 0.048286 | C₀ = 0.0456 | 1.059 |
| C | 20,224 | Ψ₀ = 20,816 | 1.029 |

Both land within 6% **simultaneously**, across five orders of magnitude in
opposite directions. A coincidence of that shape is not plausible; a
transposition is.

Three independent checks agree with the reading this selects:

- **Element count.** 50 × 100 design + 2 × (10 × 10) tabs = 5200, exactly the
  number section 4.1 reports, at h = 10⁻⁴.
- **Reynolds number.** ρUD/μ with D the inlet *half* width gives exactly 200.
  The full width would give 400.
- **Temperature range.** T_max ≈ 9.5 against figure 11's 0–12 colourbar. The
  temperature field is what rules out the alternative reading below.

Nothing in `tfopus/` relabels anything. `Zhao2DSpec.reported_psi_0` and
`reported_c_0` keep the paper's own labels, and the swap lives only in the study
script's scoring.

### The alternative that was tested and rejected

Before the transposition was visible, the gap looked like a length-scale
problem: Ψ is dominated by ∫αu·u, which scales with area, so a domain 1000×
larger raises it by 10⁶ and leaves the 5200-element count unchanged (a ratio).
That reading also recovers Re = 200 if μ is read as a kinematic viscosity, and
it does bring Ψ to 1.30 × 20,816.

It is rejected because it takes the temperature with it: at ×1000 the computed
T_max is 4.4 × 10⁴ against figure 11's colourbar maximum of 12, and C lands at
2 × 10¹³. The transposition explains both constants at once; the length scale
explains one and destroys the other.

### What R0 selected, and how firmly

| choice | selected | how firmly |
|---|---|---|
| reference field extent | γ = 0.4 over the **whole** domain, tabs included | 1.06 vs 1.44 for fluid tabs — clear, and it is also the reading that makes eq 26's v_f come out at exactly 0.4 |
| α_max at the reference | **10⁶**, the continuation start | Ψ ∝ α_max exactly, and 10⁷ is off by 9.7× |
| heat source region | whole domain (the §4.1 text, not fig 7's annotation) | weak: 1.06 vs 1.09, within the residual disagreement |
| momentum stabilisation | αu **kept** in the SUPG/PSPG residual | 1.06 vs 1.10, and see below |

### What in the momentum form actually matters, and a retracted claim

`ZHAO_FORM` differs from `ZHOU_FORM` in three places at once, so the difference
between them attributes to nothing on its own.
`scripts/zhao2d_form_attribution.py` varies each independently at the figure-7
scale and the selected reference state:

| switch, one at a time from `ZHOU_FORM` | ΔΨ | ΔC |
|---|---|---|
| drop the reactive limit in τ (Zhao eq 16) | −0.01% | +0.00% |
| drop αu from the SUPG/PSPG residual (Zhao eq 15) | **−14.25%** | **+12.66%** |
| Laplacian instead of symmetric viscous (Zhao eq 13) | +0.11% | −0.00% |

So the one consequential choice is whether the Brinkman term enters the
*stabilisation residual*. τ's reactive limit and the viscous form are both
negligible here. Using Zhao equation 15 literally moves the match from 1.06 to
1.10 — the transposition conclusion is unaffected either way, which is worth
saying explicitly: it does not rest on this switch.

**Retracted.** An earlier version of this document claimed that dropping the
reactive limit "inflates τ by a factor of ~167" at α = 2 × 10⁵, and credited it
with the bulk of the Zhao/Zhou difference. That factor came from a synthetic
unit-element configuration with h = 0.1 and **μ = 1.0** — a thousand times
Zhao's table 1 viscosity — and describes no case in the paper. The switch ratio
has a closed form that needs no solver,

    τ₀ / τ_r = √(1 + (α τ₀ / ρ)²) ≤ √(1 + (α h² / 12μ)²)   since τ₀ ≤ ρh²/12μ,

which at α = 2 × 10⁵, μ = 0.001 gives **1.0138** for h = 10⁻⁴ and **1.0541**
for the diagonal — 1.4% and 5.4%, not 167×.
`scripts/tau_decomposition.py` prints the full decomposition against that bound
at all three scales, and `test_zhao2d.py` now asserts the bound at the paper's
own parameters rather than at synthetic ones. The 1.7 × 10⁵ ratio that does
exist belongs to the ×1000 reading, which is rejected above.

## Implementation

Zhao's discretisation differs from Zhou's in three ways that are *not*
cosmetic under Brinkman penalisation, so `tfopus.fe_flow.FlowForm` and
`tfopus.fe_thermal.ThermalForm` make each one an explicit switch rather than an
inherited default. `ZHOU_FORM` and `ZHAO_FORM` name the two papers.

| | Zhou | Zhao |
|---|---|---|
| τ_u | conv + diff + reactive (eq 20) | conv + diff only (eq 16) |
| SUPG/PSPG strong residual | includes αu | ρ(u·∇)u + ∇p only (eq 15) |
| viscous Galerkin term | symmetric ∇v:(∇u + ∇uᵀ) | Laplacian ∇v:∇u (eq 13) |
| τ_T | two-limit (eq 23) | h/(2‖u‖) (eq 18), singular at u = 0 |

Zhao's τ_T diverges wherever the Brinkman penalty has driven u to zero, i.e.
over the whole solid phase. Every term it multiplies carries a factor of u, so
the products have finite limits, but τ itself is cut off below a velocity floor
rather than regularised by adding ε to the denominator — adding ε would leave an
O(h/ε) tail instead of the correct zero limit.

**Scope of the attribution above.** `zhao2d_form_attribution.py` varies only
`FlowForm`. It says nothing about τ_T, because it never changed it — an earlier
version of this document generalised its conclusion across all four rows, which
it does not support. The thermal switches are measured separately by
`scripts/zhao2d_freeze_reference.py`, one change at a time on the same
converged flow field:

| thermal change, on one flow field | ΔC₀ |
|---|---|
| τ_T convective → two-limit | +0.16% |
| then stabilised source off → on | +0.14% |

So both thermal switches are small here too, but that is a measurement, not an
inference from the flow study.

Both papers print the thermal SUPG block S_T without a ρc factor while printing
the Galerkin advection block K_c,T with one, so as printed the two cannot be
added: S_T is short exactly one volumetric heat capacity. The default repairs
it; `ThermalForm.supg_heat_capacity` can reproduce the printed form.

## Recorded gaps in the paper

`Zhao2DSpec.provenance()` prints, per field, whether a number is from the paper,
derived, or a reconstruction choice, plus this list:

- The reference field's extent (design domain only, or the tabs too).
- α_max at the reference model; continuation spans 10⁶ to 10⁷.
- Equation 26's |Ω| for v_f against the non-design tabs. Both are always
  reported: with fluid tabs they differ by 2.3 percentage points
  (0.4231 vs 0.4000).
- The discrete form of the p = 0 outlet. Implemented as a zero external
  traction, the weak form's natural condition; `OutletKind.PINNED` is the
  alternative. On this problem the two agree to 0.3% in Ψ.
- The heat source region: the §4.1 text says the entire domain, figure 7(b)
  annotates the design domain.
- h_e for the stabilisation parameters is named but never defined.
- Table 1 lists solid density 2700 and heat capacity 900, but §2 puts the fluid
  ρc over the whole domain, so the solid pair is unused. `build_material`
  leaves them at zero as a marker. Zero is not a guard:
  `Phase.volumetric_heat_capacity` returns 0.0 without raising, so a code path
  that started using a two-phase ρc would get zero silently. What protects
  against that is that every consumer takes the single `b_f` off the material
  object, not a defensive mechanism.

## R1: the frozen configuration

Decided 2026-09-22. **The reference state used for R1 is not the one that best
matched the paper.** R0's best match was γ = 0.4 over the whole domain; R1 uses
fluid tabs, because the reference state should be defined by the physics of the
problem, not chosen for agreement with a number being investigated. The two
serve different purposes and are both kept.

| | value |
|---|---|
| reference field | γ = 0.4 on the design domain, tabs pure fluid (`TABS_FLUID`) |
| α_max at the reference | 10⁶, the continuation start |
| heat source | whole domain, independent of the design |
| outlet | zero external traction |
| h_e | element edge (`min_edge`) |
| flow form | symmetric viscous, αu **in** the SUPG/PSPG residual, τ_u **with** the reactive limit |
| thermal form | τ_T two-limit, stabilised source **on**, SUPG advection carries b_f |
| volume constraint domain | design domain |
| objective | J = 0.5 Ψ/Ψ₀ + 0.5 C/C₀, self-computed denominators |

The flow and thermal forms are **not** a literal transcription of equations
13–18 — αu in the stabilisation residual and the reactive limit in τ_u are both
additions to what Zhao prints, and the stabilised source is too. That is a
stated modelling choice: the goal is Zhao's geometry, physics and metrics under
a declared stabilisation, not a reproduction of the printed discrete operators.
`ZHAO_FORM` and `ThermalForm(tau="convective", stabilise_source=False)` remain
available for the printed-formula diagnostics.

### Frozen normalisation

Computed once at h = 10⁻⁴ under exactly that configuration and then held —
α_max continuation moves the state, never the denominators:

    Psi_0_self = 0.03157912835073678
    C_0_self   = 20332.91587569144

Stored with the configuration fingerprint in `tfopus/zhao2d_reference.json`;
`ReferenceValues.check` refuses a reference frozen under different settings or a
different mesh. As a cheap staleness detector, J at the reference state is
exactly 1 by construction, and `test_zhao2d_r1.py` asserts it.

Ψ₀ is identical across all three thermal variants (spread exactly 0), which is
the cross-check that the coupling really is one way. C₀ moves +0.31% in total
from R0's thermal setting, attributed above.

Every result also carries the parallel figures against the transposed paper
constants (Ψ/0.0456, C/20816), which are reporting only and never the objective.

### Why not the paper's constants as the denominators

Changing a denominator is not a change of units: with

    J = (w/Ψ₀) Ψ + ((1-w)/C₀) C

the two denominators set the relative weight of the terms. Evaluating the same
raw metrics against the transposed paper scale instead corresponds to an
effective dissipation weight near 0.585 rather than 0.5. Self-computed
denominators keep J consistent with the solver that produces it; the paper scale
is kept alongside for comparison with tables 4 and 7.

One caveat on that comparison: table 4's arithmetic
(0.5 × 0.6312 + 0.5 × 1.1794 = 0.9053) is self-consistent, but because both
weights are 0.5 it stays self-consistent if the two columns are swapped. It
confirms the arithmetic, not which column is which.

## Running it

```bash
python scripts/zhao2d_reference_study.py --provenance   # the sweep, both stages
python scripts/zhao2d_swap_test.py                      # the transposition test
pytest validation/test_zhao2d.py
```

## Running R1

```bash
python scripts/zhao2d_freeze_reference.py --write   # A/B/C thermal study, freezes Psi_0 and C_0
python scripts/zhao2d_gradient_check.py             # Psi, C, g and J against finite differences
python scripts/zhao2d_short_run.py --iterations 20  # short MMA mechanism check
```
