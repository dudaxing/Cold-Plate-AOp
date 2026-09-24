"""The optimisation loop: phased continuation, a budget, and honest endings.

Shared by the short mechanism check and the full run so the terminal-pairing
fix exists once.

**Terminal pairing.** The loop evaluates x_n, records it, then asks MMA for
x_{n+1}. Saving `state.x` after the loop therefore saves a design one update
ahead of the last metrics recorded -- a design that has never been solved,
never passed the convergence gate, and has no objective value. This driver
re-evaluates the final design and appends it as its own record, so every saved
design is paired with metrics from the same iterate.

**Stop reasons.** Upstream `mma.py` sets `is_converged = True` for three
different reasons -- the design step falling below `step_tol`, the KKT residual
falling below `kkt_tol`, and simply reaching `max_iter` -- so the flag cannot
distinguish a converged design from an exhausted budget. `_mma_convergence`
re-derives the two genuine criteria and ignores the third. The driver reports
one of:

    converged          step_tol or kkt_tol fired, on its own
    phase_end          the schedule ran to its end without either firing
    budget_exhausted   the global budget ran out first

Only `converged` reached in the FINAL phase, at the final alpha_max and beta,
describes a converged design; `converged` in an earlier phase means that phase
settled under its own continuation parameters.

Reading the KKT residual needs care: `MMAState` declares `kkt_norm`, but
`update_mma` assigns `mma_state.kktnorm` -- a different attribute created on the
fly. The declared field therefore keeps its initial 1000.0 for the whole run,
and code that reads the documented name sees a number that never moves.
`_kkt_norm` reads the one that is actually written.

**One continuation parameter at a time.** alpha_max ramps with beta = 0; beta
then steps with alpha_max pinned at its cap. `validate_schedule` refuses a
schedule that moves both across a phase boundary, because a jump in J would
then be unattributable.

**One solve per iterate, for any model.** `evaluate` takes J, its gradient and
the reported states from a single traced solve and gates THOSE states; it forms
C through `problem.thermal_velocity` and checks the reference with
`problem.check_reference`. So the same driver serves the single-mesh model and
the dual-mesh one, and a reference frozen for another model -- for a dual-mesh
problem, the single-mesh one -- is refused at the entry. Earlier, the gate ran
on one solve, the reported values came from a second, the gradient from a
third through a different function, and C used the flow mesh's velocity layout
whatever the thermal mesh was.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

import toflux.src.mma as _mma

from tfopus import zhao2d as _z
from tfopus import zhao2d_r1 as _r1

# Zhao's constants under the transposition, for the parallel report only.
PAPER_PSI_0_INTERPRETED = 0.0456
PAPER_C_0_INTERPRETED = 20816.0


@dataclasses.dataclass(frozen=True)
class Phase:
    """One continuation stage.

    `alpha_max` is either a constant or a function of the GLOBAL step index, so
    Zhao's 1e6 * 1.03^n can be written exactly as printed.
    """

    name: str
    iterations: int
    beta: float
    alpha_max: float | Callable[[int], float]
    note: str = ""

    def alpha_at(self, global_step: int) -> float:
        return (
            self.alpha_max(global_step)
            if callable(self.alpha_max)
            else self.alpha_max
        )


def zhao_alpha_ramp(spec: _z.Zhao2DSpec, steps: int | None = None):
    """Section 4.1: alpha_max = 1e6 * 1.03^iter, capped at 1e7 (n = 78).

    Returns (callable, growth, reaches_cap_at).

    If `steps` is shorter than 79 the paper's rate cannot reach the cap before
    the projection phases begin, which would make the first beta boundary move
    BOTH parameters -- exactly what `validate_schedule` refuses. Rather than
    letting a rehearsal silently run a different schedule shape, the growth rate
    is compressed so the cap lands on the last step of the alpha phase, and the
    deviation is returned so the caller can report it.
    """
    span = spec.alpha_max_final / spec.alpha_max_initial
    paper_steps = int(np.ceil(np.log(span) / np.log(spec.alpha_max_growth)))
    if steps is None or steps > paper_steps:
        return spec.alpha_max, spec.alpha_max_growth, paper_steps

    growth = span ** (1.0 / (steps - 1)) if steps > 1 else span

    def ramp(n: int) -> float:
        return min(spec.alpha_max_final, spec.alpha_max_initial * growth**n)

    return ramp, growth, steps - 1


FULL_BUDGET = 300
_ALPHA_SHARE = 100 / FULL_BUDGET  # reaches the cap at n = 78, then settles
_BETAS = (1.0, 2.0, 4.0, 8.0)


def r1d_schedule(spec: _z.Zhao2DSpec, budget: int = FULL_BUDGET) -> list[Phase]:
    """alpha first at beta = 0, then beta 1 -> 2 -> 4 -> 8 at alpha_max = 1e7.

    At the full budget the alpha phase gets 100 steps, enough to reach the cap
    at n = 78 and settle there before any projection starts, so no phase
    boundary moves both parameters.

    A smaller budget scales every phase proportionally rather than taking the
    shortfall out of the beta phases -- doing that silently produced negative
    iteration counts, and even bounded at zero it would have turned a rehearsal
    into a different schedule from the one it is rehearsing.
    """
    if budget < len(_BETAS) + 1:
        raise ValueError(
            f"budget {budget} cannot cover one step per phase "
            f"({len(_BETAS) + 1} phases)"
        )
    # floor, not round: at the minimum budget rounding up the alpha share
    # starves the projection phases, so the guard above would state a minimum
    # the function cannot actually deliver.
    alpha_steps = max(1, int(budget * _ALPHA_SHARE))
    remaining = budget - alpha_steps
    per_beta, extra = divmod(remaining, len(_BETAS))
    assert per_beta >= 1, (budget, alpha_steps, remaining)

    ramp, growth, cap_at = zhao_alpha_ramp(spec, alpha_steps)
    note = f"growth {growth:.6g}/step, cap at n = {cap_at}" + (
        ""
        if growth == spec.alpha_max_growth
        else f"  [COMPRESSED from the paper's {spec.alpha_max_growth} so the "
             "cap lands inside this phase; a rehearsal, not the paper schedule]"
    )
    phases = [
        Phase("alpha-ramp", alpha_steps, beta=0.0, alpha_max=ramp, note=note)
    ]
    for i, beta in enumerate(_BETAS):
        phases.append(
            Phase(
                f"beta-{beta:g}",
                per_beta + (extra if i == len(_BETAS) - 1 else 0),
                beta=beta,
                alpha_max=spec.alpha_max_final,
            )
        )
    assert sum(p.iterations for p in phases) == budget
    return phases


def validate_schedule(phases: list[Phase], spec: _z.Zhao2DSpec) -> None:
    """Refuse a schedule that changes alpha_max and beta across one boundary."""
    for prev, nxt in zip(phases, phases[1:]):
        a_before = prev.alpha_at(sum(p.iterations for p in phases[: phases.index(prev) + 1]) - 1)
        a_after = nxt.alpha_at(sum(p.iterations for p in phases[: phases.index(prev) + 1]))
        if not np.isclose(a_before, a_after) and prev.beta != nxt.beta:
            raise ValueError(
                f"phase boundary {prev.name} -> {nxt.name} moves both alpha_max "
                f"({a_before:.3g} -> {a_after:.3g}) and beta "
                f"({prev.beta} -> {nxt.beta}); a jump in J would not be "
                "attributable to either"
            )


@dataclasses.dataclass
class RunResult:
    history: list[dict]
    terminal: dict
    stop_reason: str
    # Which genuine MMA criterion fired, if stop_reason is "converged".
    converged_by: str | None
    # True only if convergence was reached in the LAST phase of the schedule,
    # i.e. at the final alpha_max and beta. Convergence in an earlier phase
    # says that phase settled, not that the design is final.
    converged_at_final_stage: bool
    design: np.ndarray
    solid_fraction: np.ndarray
    press_vel: np.ndarray
    temperature: np.ndarray
    # The model the run actually ENDED under. Not phases[-1]: a run stopped by
    # the budget never reaches the last phase, and evaluating its final design
    # at the last phase's alpha_max and beta would compare two different models.
    final_alpha_max: float
    final_beta: float
    final_phase: str


def evaluate(problem, reference, x, alpha_max, beta, gradient: bool = True,
             tol: float = 1e-8):
    """Every reported quantity for one iterate, and its gradients, from ONE solve.

    One traced forward solve produces J and, as auxiliary output, the states it
    solved; `jax.value_and_grad` differentiates that same solve. So the value
    MMA sees, the gradient it is given and the states that are reported and
    saved cannot come from different solves -- nor from different
    discretisations: C is formed through `problem.thermal_velocity`, which for a
    dual-mesh problem maps the flow onto the thermal mesh.

    The reference is checked against THIS problem before anything is solved
    (for a dual-mesh problem the identity includes the thermal mesh), and the
    residual gate is applied to the states that solve returned: an unconverged
    state raises `NotConverged`, and its gradient is never used.

    Returns (record, (s, press_vel, temperature), dJ, dg); the gradients are
    None when `gradient` is False.
    """
    problem.check_reference(reference)
    config = problem.config
    w = config.weight

    def objective(v):
        s = problem.solid_fraction(v, beta)
        press_vel, temperature, alpha, kappa = problem.solve_states(s, alpha_max)
        psi = problem.flow.dissipated_power(press_vel, alpha)
        c = problem.thermal.thermal_compliance(
            temperature, problem.thermal_velocity(press_vel), kappa
        )
        j = w * psi / reference.psi_0 + (1.0 - w) * c / reference.c_0
        return j, (s, press_vel, temperature, alpha, kappa, psi, c)

    def constraint(v):
        return problem.fluid_fraction(v, beta) / config.max_fluid_fraction - 1.0

    if gradient:
        (j, aux), dj = jax.value_and_grad(objective, has_aux=True)(x)
        g, dg = jax.value_and_grad(constraint)(x)
    else:
        (j, aux), g, dj, dg = objective(x), constraint(x), None, None
    s, press_vel, temperature, alpha, kappa, psi, c = aux

    norms = problem.residual_norms_at(press_vel, temperature, alpha, kappa)
    bad = {k: v for k, v in norms.items() if not v <= tol}
    if bad:
        raise _r1.NotConverged(
            f"relative residual above {tol:g}: {bad}; this iterate's value and "
            "gradient are not used"
        )

    fractions = _z.fluid_fractions(problem.flow_mesh, s)
    record = {
        "alpha_max": float(alpha_max),
        "beta": float(beta),
        "J_self": float(j),
        "psi": float(psi),
        "compliance": float(c),
        "psi_over_psi0_self": float(psi) / reference.psi_0,
        "c_over_c0_self": float(c) / reference.c_0,
        "J_paper_interpreted": (
            w * float(psi) / PAPER_PSI_0_INTERPRETED
            + (1 - w) * float(c) / PAPER_C_0_INTERPRETED
        ),
        "psi_over_paper": float(psi) / PAPER_PSI_0_INTERPRETED,
        "c_over_paper": float(c) / PAPER_C_0_INTERPRETED,
        "constraint_g": float(g),
        "grey_fraction": float(
            jnp.mean((s > 0.05) & (s < 0.95))
        ),
        **fractions,
        "flow_residual_relative": norms["flow"],
        "thermal_residual_relative": norms["thermal"],
    }
    state = (np.asarray(s), np.asarray(press_vel), np.asarray(temperature))
    return record, state, dj, dg


def _evaluate(problem, reference, x, alpha_max, beta):
    """(record, state) for one iterate, without gradients. See `evaluate`."""
    record, state, _, _ = evaluate(problem, reference, x, alpha_max, beta,
                                   gradient=False)
    return record, state


def _kkt_norm(state) -> float:
    """The KKT residual upstream actually writes (`kktnorm`, not `kkt_norm`)."""
    value = getattr(state, "kktnorm", None)
    if value is None:  # pragma: no cover - only if upstream fixes the typo
        value = getattr(state, "kkt_norm", float("nan"))
    return float(value)


def _mma_convergence(state, params) -> str | None:
    """Which genuine criterion fired, if any. `max_iter` is not one of them.

    Upstream folds three conditions into one boolean, and one of them is just
    the iteration cap, so `state.is_converged` is True on the last step of every
    run regardless of the design.
    """
    if state.epoch > 1 and state.change_design_var < params.step_tol:
        return "step_tol"
    if _kkt_norm(state) < params.kkt_tol:
        return "kkt_tol"
    return None


def run(
    problem: "_r1.Zhao2DProblem",
    reference: "_r1.ReferenceValues",
    phases: list[Phase],
    move_limit: float = 0.1,
    budget: int | None = None,
    on_iteration: Callable[[dict], None] | None = None,
) -> RunResult:
    """Run the phased schedule. Every state is gated before MMA sees it.

    A reference not frozen for `problem` is refused here, before the first
    solve -- for a dual-mesh problem that includes the single-mesh reference.
    """
    spec = problem.spec
    problem.check_reference(reference)
    validate_schedule(phases, spec)
    budget = budget or sum(p.iterations for p in phases)

    n = problem.num_design
    params = _mma.MMAParams(
        max_iter=budget,
        kkt_tol=1e-6,
        step_tol=1e-6,
        move_limit=move_limit,
        num_design_var=n,
        num_cons=1,
        lower_bound=np.zeros((n, 1)),
        upper_bound=np.ones((n, 1)),
    )
    state = _mma.init_mma(np.full((n, 1), 1.0 - spec.reference_gamma), params)

    history: list[dict] = []
    stop_reason = "phase_end"
    converged_by: str | None = None
    step = 0

    for phase in phases:
        for _ in range(phase.iterations):
            if step >= budget:
                stop_reason = "budget_exhausted"
                break
            alpha_max = phase.alpha_at(step)
            x = jnp.asarray(state.x.reshape(-1))
            t0 = time.time()

            record, _, dj, dg = evaluate(problem, reference, x, alpha_max, phase.beta)

            record.update(
                iteration=step,
                phase=phase.name,
                seconds=time.time() - t0,
                model_changed=bool(
                    history
                    and (
                        record["alpha_max"] != history[-1]["alpha_max"]
                        or record["beta"] != history[-1]["beta"]
                    )
                ),
            )
            history.append(record)
            if on_iteration:
                on_iteration(record)

            state = _mma.update_mma(
                state,
                params,
                record["J_self"],
                np.asarray(dj).reshape((-1, 1)),
                np.array([record["constraint_g"]]),
                np.asarray(dg).reshape((1, -1)),
            )
            step += 1

            record["kkt_norm"] = _kkt_norm(state)
            record["design_step_norm"] = float(state.change_design_var)
            fired = _mma_convergence(state, params)
            record["mma_criterion"] = fired
            if fired is not None:
                stop_reason = "converged"
                converged_by = fired
                break
        if stop_reason in ("budget_exhausted", "converged"):
            break

    # The design MMA last produced has never been solved. Evaluate it so that
    # what gets saved is paired with metrics from the same iterate -- and under
    # the model the run ended in, which is the last phase it actually REACHED.
    if not history:
        raise RuntimeError("no iteration completed; nothing to finalise")
    final_alpha_max = history[-1]["alpha_max"]
    final_beta = history[-1]["beta"]
    final_phase = history[-1]["phase"]

    x_final = jnp.asarray(state.x.reshape(-1))
    terminal, (s, press_vel, temperature), _, _ = evaluate(
        problem, reference, x_final, final_alpha_max, final_beta, gradient=False
    )
    terminal.update(iteration=step, phase=final_phase, terminal=True)

    return RunResult(
        history=history,
        terminal=terminal,
        stop_reason=stop_reason,
        converged_by=converged_by,
        converged_at_final_stage=bool(
            stop_reason == "converged" and final_phase == phases[-1].name
        ),
        design=np.asarray(x_final),
        solid_fraction=s,
        press_vel=press_vel,
        temperature=temperature,
        final_alpha_max=final_alpha_max,
        final_beta=final_beta,
        final_phase=final_phase,
    )
