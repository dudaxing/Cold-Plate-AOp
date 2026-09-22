"""The optimisation loop: phased continuation, a budget, and honest endings.

Shared by the short mechanism check and the full run so the terminal-pairing
fix exists once.

**Terminal pairing.** The loop evaluates x_n, records it, then asks MMA for
x_{n+1}. Saving `state.x` after the loop therefore saves a design one update
ahead of the last metrics recorded -- a design that has never been solved,
never passed the convergence gate, and has no objective value. This driver
re-evaluates the final design and appends it as its own record, so every saved
design is paired with metrics from the same iterate.

**Stop reasons.** Upstream `mma.py` sets `is_converged = True` when it merely
reaches `max_iter`, so that flag cannot certify anything. The driver reports
one of:

    converged          the optimiser's own criteria fired inside a phase
    phase_end          the phase ran out of its allocated iterations
    budget_exhausted   the global budget ran out first

Only `converged` in the FINAL phase, at the final alpha_max and beta, describes
a converged design.

**One continuation parameter at a time.** alpha_max ramps with beta = 0; beta
then steps with alpha_max pinned at its cap. `validate_schedule` refuses a
schedule that moves both across a phase boundary, because a jump in J would
then be unattributable.
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


def _evaluate(problem, reference, x, alpha_max, beta):
    """Solve, gate, and return every reported quantity for one iterate."""
    config = problem.config
    s = problem.solid_fraction(x, beta)
    norms = problem.require_converged(s, alpha_max)
    press_vel, temperature, alpha, kappa = problem.solve_states(s, alpha_max)
    psi = problem.flow.dissipated_power(press_vel, alpha)
    c = problem.thermal.thermal_compliance(
        temperature, problem.flow.element_velocities(press_vel), kappa
    )
    g = problem.fluid_fraction(x, beta) / config.max_fluid_fraction - 1.0
    w = config.weight
    j = w * psi / reference.psi_0 + (1.0 - w) * c / reference.c_0

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
    return record, state


def _gradients(problem, reference, x, alpha_max, beta):
    config = problem.config

    def objective(v):
        s = problem.solid_fraction(v, beta)
        psi, c = problem.metrics(s, alpha_max)
        w = config.weight
        return w * psi / reference.psi_0 + (1.0 - w) * c / reference.c_0

    def constraint(v):
        return problem.fluid_fraction(v, beta) / config.max_fluid_fraction - 1.0

    return jax.grad(objective)(x), jax.grad(constraint)(x)


def run(
    problem: "_r1.Zhao2DProblem",
    reference: "_r1.ReferenceValues",
    phases: list[Phase],
    move_limit: float = 0.1,
    budget: int | None = None,
    on_iteration: Callable[[dict], None] | None = None,
) -> RunResult:
    """Run the phased schedule. Every state is gated before MMA sees it."""
    spec = problem.spec
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
    step = 0

    for phase in phases:
        for _ in range(phase.iterations):
            if step >= budget:
                stop_reason = "budget_exhausted"
                break
            alpha_max = phase.alpha_at(step)
            x = jnp.asarray(state.x.reshape(-1))
            t0 = time.time()

            record, _ = _evaluate(problem, reference, x, alpha_max, phase.beta)
            dj, dg = _gradients(problem, reference, x, alpha_max, phase.beta)

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
        if stop_reason == "budget_exhausted":
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
    terminal, (s, press_vel, temperature) = _evaluate(
        problem, reference, x_final, final_alpha_max, final_beta
    )
    terminal.update(iteration=step, phase=final_phase, terminal=True)

    return RunResult(
        history=history,
        terminal=terminal,
        stop_reason=stop_reason,
        design=np.asarray(x_final),
        solid_fraction=s,
        press_vel=press_vel,
        temperature=temperature,
        final_alpha_max=final_alpha_max,
        final_beta=final_beta,
        final_phase=final_phase,
    )
