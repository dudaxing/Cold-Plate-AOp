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
| R1d | 2D optimisation, 300-update budget | **done** — budget limited, not converged |
| R1e | fixed-design mesh and heat check | **done** |
| R1f | thermal space / stabilisation / velocity separation | **done** — closed with the benchmark and label corrections below |
| R1g | dual-mesh thermal model: differentiable chain, fixed-design h/2 vs h/4 | **done** — h/4 still drifts; stopped as the contract says; record wording corrected in R1h |
| R1h | fixed design: flow h vs h/2 on the common thermal meshes h/2 and h/4 | **done**, closed in the review of d71ab66 — on this design, refining the flow h → h/2 does not remove the thermal drift (+8.6% against +8.9%); flow replacement −2.5% to −2.7% of C |
| R1i | fixed design: h_T = h/8 on the saved coarse flow, one thermal state | proposed in the review of d71ab66; awaiting authorisation |
| R2 | 3D extruded analysis, straight-channel reference (fig 15) | not authorised |

## R1d: the 300-update run

5200 elements, filter radius 2×10⁻⁴, Zhao's unmodified 1.03 α ramp to the 10⁷
cap at n = 78 inside a 100-step β = 0 phase, then β = 1, 2, 4, 8 at the cap.
80.9 minutes.

| | self scale | paper-interpreted scale |
|---|---|---|
| J | 0.891231 | 0.8060 |
| Ψ/Ψ₀ | 0.4545 | 0.3147 |
| C/C₀ | 1.3280 | 1.2972 |

Raw Ψ = 0.0143511, C = 27002.4. v_f design 0.399978, constraint active.

**Not converged.** `stop_reason: phase_end`, `converged_by: None`. The objective
fell because dissipation fell 54.6% against the reference while thermal
compliance *rose* 32.8% — a trade-off, not an improvement in both. And the
reference is at α_max = 10⁶ while the final design is at 10⁷, so that pair is
not a controlled comparison of two geometries at one penalty.

Two convergence numbers need care, and neither supports a statement about
distance from an optimum:

- The final design step 0.271 is an **L² norm over 5000 variables**, i.e. an RMS
  change of 0.0038 per variable — not a 27% change in anything.
- `kkt_norm` ≈ 2.6×10⁻³ is an **internal mixed-iterate diagnostic**: upstream's
  `_kktcheck` evaluates at the updated `xmma` while still using the objective,
  constraint and gradients supplied at the previous point. It is not a
  same-point KKT residual for the terminal design. Claiming convergence later
  requires re-evaluating value, gradients and multipliers at one point.

The saved state carries the design and the PDE fields but **not** MMA's
asymptotes, `xold1/xold2` or multipliers, so a later continuation is a
refinement segment warm-started from x₃₀₀ with MMA re-initialised — not a
resumption of the same trajectory.

### Binary diagnostic, and a retracted one

The run's own output reported 65 fluid components with inlet and outlet
disconnected. **That is retracted**: `element_adjacency` indexed cells by
`round(centre/h)`, centres sit at (i+½)h, and banker's rounding collapsed i with
i+1 for odd i, dropping 160 of 5200 elements from the graph. It surfaced by
pushing the threshold to τ = 1.0001, where every cell is fluid and the answer
must be one component — it still said disconnected. Corrected, on the same
stored design:

| threshold | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| fluid components | 1 | 1 | 1 | 1 | 1 |
| v_f design | 0.3884 | 0.3946 | **0.3988** | 0.4048 | 0.4112 |

All connected. Note that **0.6 and 0.7 exceed the 40% bound**, so they are
topology-sensitivity probes, not candidate designs.

At threshold 0.5: Ψ −5.65%, C +2.38%, **J +0.33%**. That +0.33% is the net of
two opposing contributions, ΔJ_Ψ = −0.01285 and ΔJ_C = +0.01581, which largely
cancel — so it understates how much the thermal response moves. Measured
separately, **T_max rises 16.0%** (13.832 → 16.047). The defensible statement is:

> At this mesh, this finite Brinkman penalty and the frozen objective, the
> thresholded design stays connected and most of the combined-objective gain is
> retained; thermal compliance and peak temperature both worsen.

Not: that performance is independent of grey material, or that peak temperature
is insensitive. T_max is not an objective here and no hot-spot constraint is
added retroactively.

The grey figure quoted throughout is the fraction of cells with
0.05 < s < 0.95: 0.0640 over the whole domain, 0.0666 over the design domain.

## R1e: the fixed-design mesh check, and what it costs the R1d conclusion

The R1d design, unchanged, re-analysed at h and h/2 (5200 → 20800 elements).
The binary field is thresholded on the parent mesh and transferred, so both
meshes describe the same polygon; `check_transfer` asserts area, the design/tab
partition and both fluid fractions are identical (v_f design 0.39997847 on both).

| design / mesh | Ψ | C | T_max | T_min | undershoot nodes |
|---|---|---|---|---|---|
| continuous / h | 0.01435114 | 27002.4 | 13.832 | −0.3156 | 18 |
| continuous / h/2 | 0.01406784 | **32417.5** | 15.290 | **0.0000** | **0** |
| binary / h | 0.01353975 | 27645.4 | 16.047 | −0.6087 | 26 |
| binary / h/2 | 0.01358559 | **37008.7** | 18.521 | −0.2275 | 11 |

**Ψ is close to mesh converged; C is not.** One refinement moves Ψ by −1.97%
(continuous) and +0.34% (binary), but moves C by **+20.05%** and **+33.87%**.

**The undershoot is a discretisation artefact and it refines away** — completely
for the continuous design, and from −0.609 to −0.227 (26 nodes to 11) for the
binary one. It is not a defect in the source term or the stabilised residual.
Both conservation diagnostics also improve under refinement: the energy residual
1.00e-2 → 6.4e-3 and 4.13e-2 → 8.4e-3, and ∫b_f T ∇·u relative to the source
3.29e-2 → 6.99e-3 and 6.64e-2 → 2.59e-2. Everything moves toward zero, which is
what a consistent discretisation with an under-resolved solution looks like.

### The reason

The element Péclet number, Pe_e = b_f |u at the element centre| h / (2κ), over
cells with s < 0.5, inlet and outlet tabs included (2194 of 5200 at h). The mask
matters: over the whole domain the median is 0.0027, four orders of magnitude
away, because the solid cells carry no flow. R1f's table lists every mask.

| mesh | median | p90 | max | cells with Pe_e > 1 |
|---|---|---|---|---|
| h = 10⁻⁴ | 47.3 | 77.4 | 104.2 | 95.2% |
| h/2 = 5×10⁻⁵ | 23.8 | 38.7 | 51.5 | 91.9% |

With b_f = 4.18×10⁶ against κ_f = 0.61, the thermal layers are far thinner than
either mesh resolves, so SUPG is carrying the temperature solution and C is
measuring a boundary layer it cannot see. h/2 halves Pe_e and is still ~24.

"SUPG is carrying the solution" was a reading of the Péclet numbers when it was
written. R1f measures the size of the stabilisation terms on the solution —
(D_SUPG − F_SUPG)/C is 18.49% at h and 9.40% at h/2, a ratio of terms, not an
error — and separates which of the three changes moves C.

### This changes the R1d binarisation conclusion

Continuous → binary, measured on each mesh against the same reporting scale:

| mesh | ΔΨ | ΔC | ΔJ* | ΔT_max |
|---|---|---|---|---|
| h = 10⁻⁴ | −5.65% | +2.38% | **+0.33%** | +16.0% |
| h/2 | −3.43% | +14.16% | **+10.32%** | +21.1% |

So "thresholding costs 0.33% of J" is **a coarse-mesh result, not a property of
the design**: on one refinement it becomes 10.3%, a factor of 30. The R1d
statement should be read as holding at h = 10⁻⁴ and not beyond it.

J* here is the raw metrics divided by the ORIGINAL h = 10⁻⁴ frozen denominators,
used as a common reporting scale so the two meshes are comparable. It is not a
fine-mesh reference and the fine-mesh J* is not a normalised objective;
`ReferenceValues.check` is not relaxed to pretend otherwise.

### What this implies, and what it does not

The optimisation in R1d minimised a J whose thermal half carries a
discretisation error of the same order as the differences being optimised. That
is a statement about the mesh, not about the method, the implementation or the
gradients — all of which were verified independently. It does mean a converged
2D result at h = 10⁻⁴ would not be worth much, and that a mesh or formulation
decision comes before any further optimisation.

It does not say which way the optimum moves, because the design was held fixed.
Nothing here re-opens the frozen configuration or the reference values.

## R1f: which part of the refinement moves C

R1e changed three things at once — the temperature space, τ (which shrinks with
h), and the velocity (re-solved on the fine mesh). Four analyses on the same
continuous design change one at a time, with 3×3 thermal quadrature throughout
(2×2 is not exact for the SUPG term; the production baseline stays 2×2):

| | C | τ median | T_max | T_min | undershoot nodes |
|---|---|---|---|---|---|
| **A** h, u_h, τ_h | 26938.33 | 4.503e-5 | 13.828 | −0.3101 | 18 |
| **B** h/2, u_h extended, **τ frozen** | 31348.68 | 4.503e-5 | 15.130 | **0.0000** | **0** |
| **C** h/2, u_h extended, τ_{h/2} | 33233.13 | 1.126e-5 | 15.541 | −0.0987 | 1 |
| **D** h/2, u_{h/2}, τ_{h/2} | 32400.12 | 1.126e-5 | 15.289 | 0.0000 | 0 |

| step | ΔC | share of the total |
|---|---|---|
| temperature space, A → B | **+4410.34** | **+80.7%** |
| stabilisation coefficient, B → C | +1884.46 | +34.5% |
| velocity input, C → D | −833.02 | −15.3% |
| total, A → D | +5461.78 | |

**The temperature approximation space dominates.** The shares exceed 100%
because the velocity update partly cancels the other two; they sum along this
path and are not a path-independent budget. Nor is 80.7% the share of the true
error that comes from the temperature mesh: its denominator is a net change with
cancellation in it, and nothing says another design would keep the proportion.
What it does support is the order of work — improve the temperature space first.

Two internal checks: B's τ median is *identical* to A's, so the freeze worked;
C and D's whole-domain median is a quarter of it, which is τ ∝ h² in the
diffusive limit with h halved. That quarter is a property of the whole-domain
**median**, not of every element: the median cell is solid, where diffusion
dominates. Paired child against parent (same velocity function, R1g), the ratio
is

| τ(h/2) / τ(h) | median | p10 | p90 |
|---|---|---|---|
| whole domain | 0.2500 | 0.2500 | 0.524 |
| fluid, parent s < 0.5 | **0.4973** | 0.349 | 0.605 |

as the formula says it should be: for the same local velocity and conductivity,
τ(h/2)/τ(h) = ½ √(Pe² + 1) / √(Pe² + 4), with Pe the parent's element Péclet
number — ¼ where diffusion dominates, ½ where convection does. In the channels
the stabilisation coefficient roughly halved, it did not quarter. (An earlier
version of this paragraph read the quarter as applying to the channels too.)

And the undershoot is removed by the finer space, not by τ — B, with the larger
frozen τ, has none, while C with the smaller τ has one.

The net stabilisation term, measured against C, falls with refinement. It comes
from the identity C + D_SUPG − F_SUPG = L_Q (exact because T_h vanishes on the
Dirichlet boundary, the inlet value being zero):

| | L_Q | D_SUPG | F_SUPG | (D−F)/C | closure |
|---|---|---|---|---|---|
| A | 31919.82 | 5042.41 | 60.91 | **18.49%** | 1.1e-16 |
| B | 35357.17 | 4069.91 | 61.42 | 12.79% | 2.1e-16 |
| C | 36356.02 | 3154.05 | 31.16 | **9.40%** | 2.0e-16 |
| D | 35490.78 | 3120.00 | 29.33 | 9.54% | 2.1e-16 |

The identity is a statement about the discrete balance on one solution. C's
definition (Zhao eq 23) contains no τ; τ reaches C only through the state
equation, and changing τ changes T and with it L_Q, D_SUPG and F_SUPG together.
So (D − F)/C = 18.49% is a ratio of terms on this solution. It is not an error,
not the change C would undergo if the stabilisation were removed, and not a
statement that 18% of the performance is set by τ — an earlier wording of this
section said that and is withdrawn. The objective stays Zhao's C; adding the
SUPG terms to it to "correct" the number would change the task.

A and R1e's h row are the same analysis at different quadrature: C = 26938.33
(3×3) against 27002.4 (2×2), 0.24% apart. Every comparison within R1f is at 3×3,
and the production objective is unchanged at 2×2.

The two authorised binary controls, A′ = 26278.58 at h and C′ = 34904.70 at h/2,
threshold the density and recompute κ but keep the **continuous design's
velocity** (`zhao2d_thermal_separation.py` passes `ev_coarse` and
`ev_fine_ext`). They are the thermal response to a binary conductivity field
with the flow held at the continuous design's — not the binary design's
performance with its own flow, and not comparable with R1e's full binary states
(27645.4 and 37008.7), which re-solve the flow on the binary design. What they
do show is the same direction of change under refinement as the continuous
case. The JSON keys still read `binary, u_h`; the script's labels now say
`binary k, continuous-design u`.

### An independent accuracy reference

Ranking stabilisation variants by which gives a smaller C on the cold plate
would be selection bias, since there is no exact answer there.
`tfopus/advection_benchmark.py` (driven by
`scripts/zhao2d_advection_benchmark.py`) solves a convection–diffusion problem
with a known exact solution, using the same element and residual: an
exponential layer of width L/Pe at a Dirichlet outlet, Pe = 1000. It checks the
method on a known answer. It does not size the cold plate's thermal mesh — see
"What this points at".

**Corrected after review.** The first version had two measurement faults. Its
L² denominator, the norm of the exact solution, was integrated with a fixed
10-point Gauss rule that cannot follow a layer thinner than the element: it came
out 51% low at nx = 10 and 9% low at nx = 20, so the relative errors on those
meshes were not comparable with the rest. The denominator is now the closed
form, ‖T‖ = 0.0111803398875, and the numerator uses composite Gauss accepted
only once doubling the subdivision changes it by less than 10⁻¹¹; a test pins
the norm as mesh-independent. And it reported one "h" for two different
lengths: Pe used hx = L/nx while τ used the min edge, which on the first two
meshes is hy. The ∫k|∇T|² column was never affected.

**Anisotropic series**, ny = 8 — the original meshes, lengths now separate:

| nx | hx | h_τ | Pe_x | Pe_τ | ‖T_h − T‖/‖T‖ | as first reported | ∫k\|∇T\|² error | undershoot |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.1 | 0.03125 | 50 | **15.6** | **8.00** | 11.40 | 94.0% | 0.50 |
| 20 | 0.05 | 0.03125 | 25 | **15.6** | **5.12** | 5.35 | 94.0% | 0.20 |
| 40 | 0.025 | 0.025 | 12.5 | 12.5 | 3.81 | 3.81 | 92.6% | 0 |
| 80 | 0.0125 | 0.0125 | 6.25 | 6.25 | 2.51 | 2.51 | 86.1% | 0 |
| 160 | 0.00625 | 0.00625 | 3.13 | 3.13 | 1.54 | 1.54 | 74.9% | 0 |
| 320 | 0.003125 | 0.003125 | 1.56 | 1.56 | 0.81 | 0.81 | 56.8% | 0 |

**Square series**, ny = nx/4, so hx = hy = h_τ — the one comparable with the
cold plate's square elements:

| nx | h | Pe_e | ‖T_h − T‖/‖T‖ | ∫k\|∇T\|² error | undershoot |
|---|---|---|---|---|---|
| 8 | 0.125 | 62.5 | 9.00 | 98.4% | 0 |
| 16 | 0.0625 | 31.3 | 6.28 | 96.9% | 0 |
| 32 | 0.03125 | 15.6 | 4.32 | 94.0% | 0 |
| 64 | 0.0156 | 7.81 | 2.88 | 88.6% | 0 |
| 128 | 0.00781 | 3.91 | 1.82 | 79.1% | 0 |
| 256 | 0.00391 | 1.95 | 1.02 | 63.5% | 0 |
| 512 | 0.00195 | 0.98 | 0.46 | 40.6% | 0 |

The ∫k|∇T|² errors are underestimates: the discrete layer is smeared over an
element, so its gradient — and the dissipation it carries — is too small.

Observed orders between successive meshes, against hx:

| series | L² | ∫k\|∇T\|² |
|---|---|---|
| anisotropic | 0.64, 0.43, 0.60, 0.71, 0.92 | −0.00, 0.02, 0.11, 0.20, 0.40 |
| square | 0.52, 0.54, 0.58, 0.67, 0.83, 1.15 | 0.02, 0.04, 0.09, 0.16, 0.32, 0.65 |

These replace the earlier "0.49–1.09" for L², which came from the faulty
denominator. The earlier statement that the integral "does not improve at all"
over the first refinement also needs narrowing: in the anisotropic series h_τ
does not change over nx = 10 → 20 while the aspect ratio does, so that stall
cannot be put down to layer resolution. The square series, where only h
changes, shows the slow convergence is real — order 0.02 from Pe_e 62.5 to 31.3,
still under 0.7 at Pe_e ≈ 1 — without that confound.

The anisotropic series also undershoots, by 0.50 and 0.20 on its first two
meshes; the square series never does. With the min edge as h_τ on elements 3.2
and 1.6 times longer streamwise, τ is sized by the cross-stream edge and the
streamwise stabilisation is too weak. The cold plate's elements are square, so
this does not carry over, and its min-edge definition is unchanged.

Caveat: the layer sits at a Dirichlet outlet, while the cold plate has a
volumetric source, adiabatic walls and interior channel/solid layers. The
benchmark bounds nothing about the cold plate's percentages.

### Péclet, with the mask stated

`b_f |u at element centre| h / (2κ)`, h = 10⁻⁴:

| mask | n | median | p90 | max | > 1 |
|---|---|---|---|---|---|
| whole domain | 5200 | **0.0027** | 64.50 | 104.16 | 40.2% |
| s < 0.5, tabs included | 2194 | **47.35** | 77.45 | 104.16 | 95.2% |
| s < 0.5, design domain only | 1994 | 45.26 | 74.87 | 103.63 | 94.7% |

A bare "median Pe" is not checkable: the first two differ by four orders of
magnitude.

### What this points at

A → B dominating says the temperature approximation space is where to start,
not that the stabilisation formula is irrelevant. The cheap consequence is that
**thermal resolution can be raised on its own mesh**, leaving the design and flow
meshes where they are — R1g builds exactly that.

Keeping the coarse flow is a cost decision, not a finding that it is accurate:
C → D is −833.02, −2.51% of the C row's compliance. Small beside A → B, not zero,
and it stays in the error discussion.

Nor does it say τ is fine: B → C is +34.5% of the net change.

How fine the thermal mesh must be is decided by two different checks with
different jobs. The analytic benchmark asks whether the method gets a known
answer right; it cannot be converted into a cold-plate mesh size, because an
outlet layer at a Dirichlet boundary is not the cold plate's interior
channel-and-solid problem. The cold plate's own fixed-design refinement — same
physics, design and scheme, only the thermal mesh changing — is a legitimate
discretisation check of the metric that matters. What would be biased is
comparing different *schemes* on one mesh and keeping the one with the smaller
C. (An earlier version of this paragraph said the question belonged to the
benchmark and not to the cold plate's C; that conflated the two.)

### An environment fault worth knowing about

Repeated large sparse solves through `jax.pure_callback` crash the interpreter
with Windows heap corruption (0xC0000374) and **no traceback**, immediately after
OpenBLAS reports exceeding its precompiled thread count on this 32-core machine.
The shell sees exit code 0 and a truncated log, so it is indistinguishable from
a clean finish — it killed an R1f run after the first of four analyses and then
a full test run. `tfopus/_threads.py` sets a default of 8 BLAS threads before
NumPy loads, and is imported from `tfopus/__init__.py` and
`validation/conftest.py` as well as the entry scripts. It is a default, not a
cap: an explicit value in the environment wins. It also only works if it runs
before the BLAS library starts, since OpenBLAS reads the variable once; if NumPy
was imported first it warns rather than pretend. This is a mitigation that has
worked on this machine, not a root-cause proof.

Revisited after R1h. The warning comes from a table of 50 buffer slots in
OpenBLAS 0.3.30's allocator: each LAPACK call in flight holds one -- jaxlib's
CPU LAPACK, behind the element Jacobian inverse, is SciPy's OpenBLAS -- and each
worker of OpenBLAS's own pool holds one for life, 23 at the build's default of
24 threads and 7 at 8. Once the table has overflowed, each further overflow
adds a record to a 512-entry heap array that has no bound check, so enough of
them write past it. On the R1d/R1h anchors a pool of 24 printed the
warning in both of two runs and one segfaulted at exit; a pool of 8 never
printed it in three. The routing proposed in the conformal-cooling repository,
upstream's spsolve on one dedicated thread, was evaluated and not adopted: the
anchors are bit-identical with it, but the solves never overlap, so it leaves
the slot count unchanged, and a pool of 24 still printed the warning. The
mechanism is read from the source and fits every failure seen, but no crash has
been caught with a native stack. `tfopus/_threads.py` has the detail.

## R1g: the temperature on a mesh of its own

Decided in the review of df39ce6: the design variables and the flow stay on the
h = 10⁻⁴ mesh, the temperature gets an independent nested mesh, and the chain
stays differentiable end to end. R1g builds that chain and checks it on the R1d
design at h_T = h/2 (the development candidate) and h_T = h/4 (one resolution
check). Neither is declared resolved in advance, and nothing is optimised.

### The chain

    x → filter, projection → s_D                        design = flow mesh, h
      → α(s_D) → Newton(flow)    [implicit] → u_F
      → s_T = E s_D,   u_T = P u_F                    fixed nested maps
      → κ(s_T), τ_T(u_T, κ, h_T)   recomputed at every evaluation
      → Newton(thermal) [implicit] → T_T              thermal mesh, h/r
      → Ψ(u_F, α) on the flow mesh,  C(T_T, u_T, κ) on the thermal mesh,
        g on the design domain

`tfopus/zhao2d_dual.py`. **E** copies each parent's physical density to its r²
children. **P** evaluates the coarse Q1 velocity at the thermal nodes — the same
function, not a projection. Both are index/weight tables built once from the
geometry and applied as JAX gathers, so their transposes are scatter-adds: a
parent's design sensitivity is the **sum** of its children's, never their
average. τ comes from the formula on the thermal mesh, with the thermal mesh's
own element length, at every evaluation; R1f's frozen τ stays a diagnostic.
`Zhao2DDualProblem` inherits the design and flow side of `Zhao2DProblem`
unchanged and rebuilds only the thermal members.

The single-mesh Ψ₀ and C₀ are **refused** as this model's normalisation.
`objective_and_constraint` checks an identity that includes the thermal mesh;
the inherited check compares only the spec and the R1 config, neither of which
mentions the thermal mesh, and would have accepted them. Here they are only a
stated reporting scale, J*. A dual-mesh optimisation needs its own versioned
reference first, and `tfopus/zhao2d_reference.json` is untouched.

### Verification

| check | result |
|---|---|
| P reproduces constants and a + bx + cy + dxy, r = 2 and 4 | error < 10⁻¹² |
| fine interpolant of P u equals the coarse function at 300 random interior points, random u | < 10⁻¹² |
| P against R1f's independent NumPy extension | < 10⁻¹³ |
| E keeps 0/1 endpoints, the design/tab partition and fluid tabs; children tile parents | exact |
| total heat source on the thermal mesh, both source regions | equal to 10⁻¹³ |
| E^T 1 = r² per parent; vjp of E = explicit E^T; ⟨Ea, b⟩ = ⟨a, E^T b⟩ | exact; 10⁻¹³ |
| vjp of P = explicit P^T; rows of P sum to one; ⟨Pu, v⟩ = ⟨u, P^T v⟩ | 10⁻¹² |
| r = 1 at 2×2: Ψ, C and ∇C equal the single-mesh chain | 10⁻¹³, 10⁻¹², 10⁻¹⁰ |
| total gradient, 208 flow / 832 thermal cells, α = 10⁷, β = 8, grey field, two ±1 directions: Ψ, C, g, J separately | best-step error < 10⁻⁵ |
| freezing τ changes dC·d by 0.24%; the unfrozen chain matches finite differences to 4×10⁻¹⁰, the frozen one misses by 2.4×10⁻³ at every step | τ(u, κ) is in the gradient — small in this direction, 5000× the noise |
| the single-mesh reference is refused, although the inherited check passes it | yes |
| **R1b's gradient-check protocol on the dual chain**, h_T = h/2: 1300 flow / 5200 thermal cells, 3 stages (α 10⁶ and 10⁷, β 0 and 8) × uniform and grey fields × 6 coordinate probes + 2 ±1 directions, Ψ, C, g and J separately, every perturbed state gated | summary **5.7×10⁻⁷** (threshold 10⁻⁵; R1b single-mesh 1.7×10⁻⁷); directions ≤ 4.3×10⁻⁹ |

The one entry above 10⁻⁷ is J at a coordinate probe where its Ψ and C terms
cancel about 300-fold (dJ = −3.4×10⁻⁶); Ψ and C themselves agree to 3×10⁻¹⁰
and 1×10⁻⁹ there. Full table: `results/zhao2d_r1g_gradient_check.txt`.

### Anchors

| | this chain | R1f | relative difference |
|---|---|---|---|
| h_T = h, 3×3 | 26938.332194702118 | A 26938.332194702118 | 0 |
| h_T = h/2, 3×3 | 33233.13278907866 | **C** 33233.13278907864 | 7×10⁻¹⁶ |
| h_T = h/2, against the fine-flow row | | D 32400.115711650746 | +2.57% |

The flow is re-solved from the saved design and matches the saved R1d state bit
for bit, so landing on C rather than D also confirms that no fine-mesh flow
crept in. The C anchor is a slow regression test.

### h, h/2, h/4 on the same coarse flow

`scripts/zhao2d_dual_check.py`: the flow is solved once and reused, so the
levels differ only in how the temperature is discretised. 3×3 quadrature
throughout.

| | h | h/2 | h/4 |
|---|---|---|---|
| thermal elements | 5,200 | 20,800 | 83,200 |
| **C** | 26938.33 | 33233.13 | **36180.16** |
| c_advective | 17403.91 | 18320.14 | 18565.18 |
| c_diffusive | 9534.42 | 14912.99 | 17614.98 |
| L_Q | 31919.82 | 36356.02 | 37686.52 |
| D_SUPG − F_SUPG | 4981.49 | 3122.89 | 1506.36 |
| (D − F)/C | 18.49% | 9.40% | 4.16% |
| T_max | 13.828 | 15.541 | 15.951 |
| T_min (nodes below inlet) | −0.310 (18) | −0.099 (1) | 0 (0) |
| fluid Pe_e, median (s < 0.5, tabs incl.) | 47.3 | 23.6 | 11.8 |
| J* (single-mesh scale) | 0.8897 | 1.0445 | 1.1169 |

| step | C | c_diffusive | c_advective | L_Q | T_max | J* |
|---|---|---|---|---|---|---|
| h → h/2 | +23.37% | +56.41% | +5.26% | +13.90% | +12.39% | +17.40% |
| h/2 → h/4 | **+8.87%** | **+18.12%** | +1.34% | +3.66% | +2.64% | +6.94% |

**h/4 still drifts**: C moves another +8.87%, and the diffusive half carries 92%
of that step. The ratio of successive changes, (C_h/2 − C_h)/(C_h/4 − C_h/2),
is 2.14 — about first order if the sequence were asymptotic, which three levels
cannot show. As the stage contract says, this is recorded and the stage stops:
no h/8, no optimisation. Neither h/2 nor h/4 is declared adequate.

### The coarse velocity's divergence, weighted by a finer temperature

What holds exactly, for these polynomial fields at 3×3 quadrature, is the
divergence theorem, with H the boundary enthalpy flux and D_T = ∫ b_f T ∇·u:

    H = ∫ b_f u·∇T + D_T,        H − Q = D_T + [∫ b_f u·∇T − Q]

The bracket is the discrete equation tested with v = 1, i.e. the reaction at
the Dirichlet inlet: −0.036, −0.068, −0.072 at h, h/2, h/4 — small against
5200, not zero. So D_T is close to H − Q without being it; the inlet's
conduction and the weak form do not vanish from the balance.

| | h | h/2 | h/4 |
|---|---|---|---|
| ∫ (∇·u)² (the coarse flow's own divergence) | 0.166659 | 0.166659 | 0.166659 |
| D_T, share of the heat input | 3.26% | 7.56% | **8.97%** |
| −½ ∫ b_f T² ∇·u, share of C | −3.02% | −5.86% | **−6.73%** |

u_T = P u_F carries the coarse flow's discrete divergence, and P is an exact
embedding: ∫(∇·u)² is the same on all three thermal meshes (independently
recomputed in review to the last digits, and in R1h on the flow mesh itself,
0.166659443741465). The thermal refinement does not
create or enlarge the divergence; what changes is the temperature that weights
it. For comparison, R1e's flow re-solved at h/2 gave D_T = 0.70% of the source.

The second row is an algebraic split of C's advective half: element by element
∫ b_f T u·∇T = ½∮ b_f T² u·n − ½∫ b_f T² ∇·u (closes to 10⁻¹⁵). It is a share of
**this** solution, not an error, and not what C would change by if the velocity
were divergence-free — replacing the flow moves T, the boundary term and the
SUPG terms too. It cannot be added to the +8.87% thermal step. R1f's C → D
(−2.51% of C) is the complete measured response to replacing the flow at a
common thermal mesh; nothing is missing from it. R1h measures that response at
h/4 as well.

(`conservation()`'s `energy_imbalance_rel`, 0.96% → 6.28% → 8.30%, mixes this
with a one-sided estimate of boundary conduction; the identity above is the
clean statement.)

### τ, paired element by element

| ratio to the parent's τ at h | whole domain, median | fluid (parent s < 0.5): median [p10, p90] |
|---|---|---|
| h/2 | 0.2500 | 0.4973 [0.349, 0.605] |
| h/4 | 0.0625 | 0.2479 [0.147, 0.292] |

The diffusive limit, (1/r)², holds for the median cell, which is solid; in the
channels τ scales nearer 1/r, the convective limit. This is the correction to
R1f's "a quarter" above.

### Where the design gradient points

| | ‖∇J*‖ | cos with h | cos with h/2 |
|---|---|---|---|
| h | 7.74×10⁻³ | | |
| h/2 | 3.82×10⁻² | 0.17 | |
| h/4 | 5.76×10⁻² | 0.04 | **0.976** |

The gradient is of J* (single-mesh scale, α = 10⁷, β = 8) with respect to the
raw design vector x, not of C alone. At the R1d design, the thermal model at h
and the finer ones disagree on which way to move: the gradient at h is nearly
orthogonal to both others. h/2 and h/4 agree on direction (12.6° apart) but not
on size: ‖g_h/4‖/‖g_h/2‖ = 1.507 and ‖g_h/4 − g_h/2‖/‖g_h/2‖ = 0.573. Direction
agreement is positive local evidence, not a converged gradient, and says
nothing yet about whether a constrained MMA step would be the same on the two
models, or where an optimisation on either would end.

### Cost

| | h | h/2 | h/4 |
|---|---|---|---|
| thermal solve, repeat | 0.96 s | 2.49 s | 9.49 s |
| value + gradient of J* through the whole chain, first / repeat | 17.0 / 5.0 s | 13.0 / 7.4 s | 23.6 / 18.1 s |
| one-time build (mesh, maps, solver) | — | 10.8 s | 33.1 s |
| peak working set of the process so far (cumulative) | 2.2 GB | 3.0 GB | 4.5 GB |

Measured on this machine (Windows, CPU, float64, SciPy sparse direct, 8 BLAS
threads). The working set is the process's peak up to that point, across every
level run before it — not the memory one model needs in isolation. The coarse
flow solve is 7.8 s on first call and 3.0 s repeated. The repeat
value-and-gradient is the chain's cost per evaluation: ×1.5 at h/2 and ×3.6 at
h/4 against the same chain at h. R1d's measured 16.2 s per MMA iteration is
more than one value-and-gradient because its driver also solves the state twice
more (the convergence gate and the reported metrics); from these parts, the same
driver would cost roughly 20–25 s per iteration at h/2 and 45–50 s at h/4.
Those are estimates from measured components, not measurements.

**The existing driver is not dual-mesh aware.** `zhao2d_driver._evaluate` builds
the thermal velocity with `flow.element_velocities`, the flow mesh's layout, so
it fails on shape for r > 1; and it forms J itself, so it would bypass the
reference refusal above. It is unchanged here, since no optimisation is
authorised, and must be adapted before any dual-mesh run.

### What R1g says, and what it does not

- The dual-mesh chain is right: the maps are exact, their transposes
  accumulate, r = 1 reproduces the old chain, the total gradient matches finite
  differences at the existing threshold, and both anchors reproduce R1f.
- On the R1d design, C is not stable between h/2 and h/4 (+8.9%), mostly in its
  diffusive half. Neither mesh is declared adequate.
- The coarse flow's divergence is fixed (∫(∇·u)² the same on every thermal
  mesh); a finer temperature weights it more: D_T is 8.97% of the source and
  −½D_T2 is −6.7% of C at h/4. Those are algebraic shares of the solution, not
  the cost of the coarse flow, which only replacing the flow can measure (R1h).
- h/2 and h/4 agree on the design gradient's direction (cos 0.976) but not its
  size (norm ratio 1.51); h does not agree on either.
- Not done, by the contract: optimisation, h/8, a new reference, any change to
  the physical task or the stabilisation formula.

## R1h: the flow mesh, at a fixed thermal mesh

Decided in the review of 05df809: on the R1d continuous design, replace the flow
solved at h by the flow solved at h/2, on the common thermal meshes h/2 and h/4.
One fine flow, verified rather than re-solved; two new thermal states (four
thermal solve calls: each state solved twice, the second for timing); no MMA, no
new reference, no change to the thermal residual, the stabilisation, the
physics or the boundary conditions.
`tfopus/zhao2d_flow_study.py`, `scripts/zhao2d_flow_mesh_check.py`, record
`results/zhao2d_r1h_matrix.json` (and `.log`).

| flow \ thermal | h/2 | h/4 |
|---|---|---|
| **h** | R1g level 2: states reused, re-evaluated and re-gated | R1g level 4: likewise |
| **h/2** | thermal solved here; must reproduce R1f's D row | thermal solved here — **new** |

Nothing is composed that did not exist: the flow solver on the refined mesh,
R1g's nested maps from that flow mesh to a thermal mesh, and the thermal solver.
The physical density is R1d's, copied from the parent element to every finer
mesh (no filter, no projection, no design variable on a fine mesh); the h/4
thermal mesh receives the same density whether it is reached from the h or the
h/2 flow mesh, and each column's two cells use literally the same thermal mesh,
source and Dirichlet nodes. Flow 2×2, thermal 3×3.

### The fine flow: verified, not re-solved

R1f's h/2 flow was cached as a bare array (`results/zhao2d_r1f_fine_flow.npz`,
untracked, `press_vel` only) with no record of what it solves. Before reuse it
was checked against this problem: 63,423 dofs; every Dirichlet value carried
exactly; relative residual **1.6×10⁻¹⁴** under this density, α = 10⁷, the
boundary conditions and properties (gate 10⁻⁸); Ψ = 0.014067841749016196,
**identical** to R1e's separate solve of the same problem. It is now stored as
`results/zhao2d_r1h_fine_flow.npz` together with its identity — geometry and
mesh hashes, flow form, outlet, material, α_max, density hash, Dirichlet hash,
solver settings — and `load_flow_state` refuses it for any other problem and
re-gates it on load. The saved file was read back through that check before the
thermal solves used it.

A rerun takes the cheapest trustworthy source first (`resolve_flow_state`):
this identity file, then the bare R1f cache verified as above, and a flow solve
only if neither qualifies; the identity file is never rewritten when it was the
input. The script also refuses to overwrite R1h records already in its output
directory without `--overwrite`, so a reproduction goes to a directory of its
own (`--out DIR`) and the committed records stay what the text cites. (Until
d71ab66's review the script looked only for the R1f cache, so on a checkout
without it a rerun would have re-solved the flow and replaced this file.)

### Same load on both flow meshes

| | flow h | flow h/2 |
|---|---|---|
| inlet nodes | 11 | 21 |
| inlet velocity on every inlet node | (0, −0.2) exactly | (0, −0.2) exactly |
| inflow / U·(inlet half width) | 1 | 1 − 1 ulp |
| rim nodes (tab wall, symmetry) | inlet wins, (0, −0.2) | inlet wins, (0, −0.2) |
| tangential slip on the tab wall next to the rim | one element, 1×10⁻⁴ | one element, 5×10⁻⁵ |

The two inlet traces are the same function (difference 0), and u_T = P u_F
carries it unchanged to every thermal inlet node. The rim slip is the one
discrete difference in the inflow boundary: inlet-wins keeps the flux exact on
every mesh at the price of a slip one flow element long on the wall, so it is
part of what replacing the flow replaces. Mass balance closes to 6×10⁻¹⁵ on
both flows; wall and symmetry fluxes are exactly zero.

### The matrix

| | flow h, T h/2 | flow h, T h/4 | flow h/2, T h/2 | flow h/2, T h/4 |
|---|---|---|---|---|
| **C** | 33233.132789 | 36180.160296 | 32400.115712 | **35194.669709** |
| c_advective | 18320.1415 | 18565.1760 | 17688.6760 | 17910.6500 |
| c_diffusive | 14912.9913 | 17614.9843 | 14711.4397 | 17284.0197 |
| L_Q | 36356.0220 | 37686.5193 | 35490.7844 | 36685.5029 |
| D_SUPG − F_SUPG | 3122.8892 | 1506.3590 | 3090.6686 | 1490.8332 |
| (D − F)/C | 9.40% | 4.16% | 9.54% | 4.24% |
| T_max | 15.540621 | 15.950727 | 15.288763 | 15.694987 |
| T_min (nodes below inlet) | −0.0987 (1) | 0 (0) | 0 (0) | 0 (0) |
| **Ψ** | 0.0143511431 | 0.0143511431 | 0.0140678417 | 0.0140678417 |
| J* (single-mesh scale) | 1.044450 | 1.116919 | 1.019480 | 1.088200 |
| \|R\|/\|R₀\| flow, thermal | 1.3e-14, 1.0e-12 | 1.3e-14, 3.9e-12 | 1.6e-14, 9.9e-13 | 1.6e-14, 3.9e-12 |

Anchors, relative: flow h × T h/2 against R1g level 2 **0**, against R1f's C
row 7×10⁻¹⁶; flow h × T h/4 against R1g level 4 **0**; flow h/2 × T h/2 against
R1f's **D row 0**; Ψ of the h flow against R1g 0, of the h/2 flow against R1e 0.

### The two mesh changes, each as a complete response

| | ΔC | c_advective | c_diffusive | T_max | Ψ | J* |
|---|---|---|---|---|---|---|
| flow h → h/2, at T h/2 | −833.02 (**−2.51%**) | −3.45% | −1.35% | −1.62% | −1.97% | −2.39% |
| flow h → h/2, at T h/4 | −985.49 (**−2.72%**) | −3.53% | −1.88% | −1.60% | −1.97% | −2.57% |
| T h/2 → h/4, on flow h | +2947.03 (**+8.87%**) | +1.34% | +18.12% | +2.64% | 0 | +6.94% |
| T h/2 → h/4, on flow h/2 | +2794.55 (**+8.63%**) | +1.25% | +17.49% | +2.66% | 0 | +6.74% |

Interaction (the flow replacement at h/4 minus at h/2, equally the thermal step
on flow h/2 minus on flow h): −152.47 in C, −0.4% of C.

Each row is the complete response to one mesh change with everything else held;
a flow replacement includes every change in the velocity field, not only its
divergence. These are differences between two meshes, not errors against an
exact solution.

- **This flow refinement does not remove the thermal drift.** On the R1d
  design, with the current thermal residual and these two flow meshes, the
  thermal step h/2 → h/4 is +8.63% on the fine flow and +8.87% on the coarse
  one: close, and the largest observed mesh difference is still on the thermal
  side. That is one design and two flow meshes; u_h/2 is not a known exact
  flow, so this is not a statement that the drift is independent of the flow
  mesh. The thermal step also changes two things at once, the temperature space
  and τ_T recomputed on h_T, as the frozen rules require. Its split — 92% in
  c_diffusive — is one algebraic decomposition of the objective difference, not
  an error budget; the same steps are equally ΔL_Q − Δ(D_SUPG − F_SUPG) =
  1330.50 + 1616.53 (coarse flow) and 1194.72 + 1599.84 (fine flow).
- **Replacing the flow moves C by −2.5% to −2.7%**, about a third of the thermal
  step, with nearly the same share at both thermal meshes. Most of it is in the
  advective half. Ψ moves −1.97% (R1e's figure), T_max −1.6%. The interaction,
  −152.47, is 0.4% of C but 18% of the flow replacement at T h/2: nearly
  additive on this design, not a correction factor to carry to another one.

### Divergence and the identities, on both flows

| | flow h, T h/2 | flow h, T h/4 | flow h/2, T h/2 | flow h/2, T h/4 |
|---|---|---|---|---|
| ∫ (∇·u)², flow mesh = thermal mesh | 0.1666594 | 0.1666594 | 0.0598732 | 0.0598732 |
| D_T = ∫ b_f T ∇·u, share of the source | 7.56% | 8.97% | 0.70% | 1.27% |
| −½ D_T2, share of C | −5.86% | −6.73% | −0.53% | −0.97% |
| ∫ b_f u·∇T − Q = Dirichlet reaction | −0.06813 | −0.07184 | −0.06497 | −0.07187 |
| closures: H, C_adv, C + D − F − L_Q (relative) | ≤ 5×10⁻¹⁵ | ≤ 7×10⁻¹⁵ | ≤ 6×10⁻¹⁵ | ≤ 7×10⁻¹⁵ |

∫(∇·u)² is a property of each flow: equal on its own mesh and on every thermal
mesh it is sampled on, because P is an exact embedding. The fine flow's is 0.36
of the coarse flow's.

**The bracket is the Dirichlet reaction.** The Q1 shape functions sum to one,
so summing the assembled thermal residual over all nodes tests the discrete
equation with v = 1, which leaves ∫ b_f u·∇T − Q; the free-node residuals
vanish at convergence, so it equals r_D, the residual at the inlet (Dirichlet)
nodes before they are replaced. They agree to 2×10⁻¹² in every cell, and the
value barely depends on the flow. Writing Q_cond,out = −r_D (about 0.07) for the
heat the weak form conducts out through the inlet,

    H − Q = D_T + r_D,        i.e.        H + Q_cond,out − Q = D_T

So D_T/Q is the global heat-balance deficit of the discrete solution as defined
here: 7.6–9.0% of the source on the coarse flow, 0.7–1.3% on the fine one,
produced by the velocity's discrete divergence in the non-conservative
convection term. It is not an error in C and not the change in C, and two
further limits apply. Q_cond,out is the weak form's own residual reaction, not
a continuous heat flux checked against anything independent. And the deficit
is a statement about the global balance — it bears on the flow-weighted mean
outlet temperature that the balance implies — not an error estimate for the
outlet temperature field, T_max, C or local fluxes: 1.3% on the fine flow does
not make that a "1% accurate" model.

**The shares are not the flow's effect on C.** At T h/4, replacing the flow
lifts −½D_T2 from −2435.2 to −340.6 (+2094.6), but the boundary term
½∮ b_f T² u·n falls by 2749.2 and c_advective falls by 654.5; C falls by 985.5.
Read as the coarse flow's error, the −6.7% share would have predicted C rising
by about 6%; the complete response is −2.7%. This is the R1g correction above,
now measured.

The one-sided boundary-conduction estimate is recorded per boundary
(`boundary.*.conduction_out`): at the inlet it is 0.073–0.075 against the
reaction's 0.065–0.072, and on the adiabatic walls and symmetry plane, where the
weak form's flux is zero, it is −67 to −70 at T h/2 and −35 to −36 at T h/4 —
halving with the thermal mesh, on either flow. It is a diagnostic of the
one-sided gradient, not of the heat balance, and is never substituted for r_D.

### Cost

| | flow h/2, T h/2 | flow h/2, T h/4 |
|---|---|---|
| thermal solve, first / repeat | 5.7 / 2.4 s | 12.3 / 9.3 s |
| one-time build (both meshes, maps, solvers) | 19.7 s | 41.0 s |

The thermal solves cost what R1g's did on the coarse flow (2.5 s and 9.5 s
repeated): the driving flow does not change the thermal problem's size. The
fine flow was not re-solved, so its solve time was not measured here; the flow
problem has 63,423 dofs against 16,113 at h, and a design → fine flow → fine
temperature gradient chain does not exist yet. The whole run took 194 s; the
process's cumulative peak working set was 2969 MiB, with all four problems
built — not the memory any one model needs. Measured on this machine, CPU,
float64, 8 BLAS threads.

### What R1h says, and what it does not

- On the R1d design, with the current thermal residual and these two flow
  meshes, refining the flow h → h/2 does not remove the h/2 → h/4 thermal drift
  (+8.6% against +8.9%); the largest observed mesh difference is on the thermal
  side, where the temperature space and τ_T change together. Neither thermal
  mesh is shown adequate.
- The flow replacement changes C by −2.5% to −2.7% and Ψ by −2.0% at this
  design, a nearly constant share across the two thermal meshes — not a
  correction factor for other designs.
- The fine flow closes the discrete global heat balance much better: the
  deficit D_T/Q falls from 7.6–9.0% to 0.7–1.3%. A balance statement, not an
  accuracy figure.
- Not shown: the drift beyond h/4, the fine-flow chain's gradient, any design
  other than R1d's, anything about the optimum. Two meshes per direction are not
  an exact solution or a convergence proof.

**For the choice of the production model (limited evidence, one design).** On
C and Ψ, the largest observed mesh difference at this design is thermal; the
flow replacement is a third of it and nearly additive. Keeping the flow on the
design mesh h is therefore a reasonable candidate development chain, with the
offsets stated for this design only — not a validated production flow mesh.
The decision still open is thermal: h/2 and h/4 differ by 8.6–8.9% on either
flow.

If the reproduction needs the discrete global heat balance closed better than
the coarse flow's 7.6–9.0%, the options are a finer flow (about 4× the flow
dofs, plus a new gradient chain) or a different convection form — and the
second is not one option. Writing the convection term as
A_θ = b_f (u·∇T + θ T ∇·u), the same residual sum gives

    H − Q − r_D = (1 − θ) D_T

so the conservative form (θ = 1) closes the balance by construction, while the
common skew-symmetric split (θ = ½) does not; and since T and D_T change with
the form, θ = ½ does not predict halving the deficit either. It may help
stability or divergence pollution, which is a different property. None of this
is authorised or implemented.

The review of d71ab66 closed R1h and proposed R1i: on the same design and the
same saved coarse flow, one thermal analysis at h_T = h/8 (332,800 elements,
334,161 nodes), to see whether the step shrinks from Δ₂₄ = C_h/4 − C_h/2 to
Δ₄₈ = C_h/8 − C_h/4. The ratio |Δ₄₈|/|Δ₂₄| would be an observation, not an
error estimate or a pass mark. Not run; it awaits authorisation.

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
python scripts/zhao2d_freeze_reference.py --write   # freezes Psi_0 and C_0
python scripts/zhao2d_gradient_check.py             # Psi, C, g and J against finite differences
python scripts/zhao2d_optimise.py --coarse --budget 25   # R1c mechanism check
python scripts/zhao2d_optimise.py --budget 300           # R1d, the main case (~80 min)
python scripts/zhao2d_binary_diagnostic.py               # R1d thresholding and connectivity
python scripts/zhao2d_refine_check.py                    # R1e, h vs h/2 on a fixed design
python scripts/zhao2d_thermal_separation.py              # R1f, analyses A/B/C/D
python scripts/zhao2d_thermal_separation.py --binary     # R1f binary controls A', C'
python scripts/zhao2d_advection_benchmark.py --pe 1000   # R1f accuracy reference
python scripts/zhao2d_dual_check.py                      # R1g, h / h/2 / h/4 on one flow
python scripts/zhao2d_gradient_check.py --thermal-refinement 2   # R1g gradients
python scripts/zhao2d_flow_mesh_check.py --out DIR       # R1h rerun; keeps results/ unless --overwrite
```

`zhao2d_short_run.py` is retired to a pointer: it had its own optimisation loop
with the terminal-pairing defect, and both entry points now share
`tfopus/zhao2d_driver.py`.
