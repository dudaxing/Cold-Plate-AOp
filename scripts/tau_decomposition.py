"""What the stabilisation parameter actually is, term by term, at a stated scale.

Written because a ratio quoted in an earlier report -- "dropping the reactive
limit inflates tau by 167x" -- came from a synthetic unit-element configuration
(h = 0.1, mu = 1.0), not from Zhao's geometry and Zhao's table 1. At the
figure-7 scale the true factor is about 1.4%. The two are not close, and no
error message would ever have surfaced the difference, so the decomposition is
printed here rather than asserted from memory.

    tau_adv   = h / (2|u|)
    tau_diff  = rho h^2 / (12 mu)
    tau_react = rho / alpha
    tau_0 = (tau_adv^-2 + tau_diff^-2)^-1/2            Zhao eq 16
    tau_r = (tau_adv^-2 + tau_diff^-2 + tau_react^-2)^-1/2   Zhou eq 20 / Alexandersen

The switch ratio has a closed-form bound that needs no solver:

    tau_0 / tau_r = sqrt(1 + (alpha tau_0 / rho)^2) <= sqrt(1 + (alpha h^2 / 12 mu)^2)

because tau_0 <= tau_diff. Every row below is checked against it.

    python scripts/tau_decomposition.py
"""

from __future__ import annotations

import math

# Zhao table 1, dimensionless.
RHO = 1000.0
MU = 0.001
# alpha at the reference field: alpha_max * q(1-gamma)/(gamma+q) with
# alpha_max = 1e6, gamma = 0.4, q = 0.2  ->  0.2 * alpha_max.
ALPHA_REF = 0.2e6

H_FIGURE = 1.0e-4  # element edge at the figure-7 scale
SCALES = {
    "figure 7 (edge h)": H_FIGURE,
    "figure 7 (diagonal h)": H_FIGURE * math.sqrt(2.0),
    "x1000 reading (edge h)": H_FIGURE * 1000.0,
}


def taus(h: float, speed: float, rho: float, mu: float, alpha: float) -> dict:
    inv_adv_sq = (2.0 * speed / h) ** 2
    inv_diff = 12.0 * mu / (rho * h**2)
    inv_react = alpha / rho
    tau0 = (inv_adv_sq + inv_diff**2) ** -0.5
    taur = (inv_adv_sq + inv_diff**2 + inv_react**2) ** -0.5
    return {
        "tau_adv": float("inf") if speed == 0 else h / (2.0 * speed),
        "tau_diff": 1.0 / inv_diff,
        "tau_react": 1.0 / inv_react,
        "tau_0": tau0,
        "tau_r": taur,
        "ratio": tau0 / taur,
        "bound": math.sqrt(1.0 + (alpha * h**2 / (12.0 * mu)) ** 2),
    }


def main() -> None:
    print(f"rho = {RHO},  mu = {MU},  alpha = {ALPHA_REF:.3g} "
          f"(= 0.2 * alpha_max at gamma = 0.4)\n")
    head = (f"{'case':>24} {'|u|':>8} {'tau_adv':>11} {'tau_diff':>11} "
            f"{'tau_react':>11} {'tau_0':>11} {'tau_r':>11} {'ratio':>10} "
            f"{'bound':>10}")
    print(head)
    print("-" * len(head))

    for label, h in SCALES.items():
        for speed in (0.0, 0.02, 0.2):
            t = taus(h, speed, RHO, MU, ALPHA_REF)
            assert t["ratio"] <= t["bound"] * (1 + 1e-12), "ratio exceeded its bound"
            print(f"{label:>24} {speed:8.3g} {t['tau_adv']:11.4g} "
                  f"{t['tau_diff']:11.4g} {t['tau_react']:11.4g} {t['tau_0']:11.4g} "
                  f"{t['tau_r']:11.4g} {t['ratio']:10.4g} {t['bound']:10.4g}")
        print()

    print("The synthetic configuration the retracted 167x came from "
          "(h = 0.1, mu = 1.0):")
    t = taus(0.1, 0.0, RHO, 1.0, ALPHA_REF)
    print(f"    tau_diff = {t['tau_diff']:.6g}, tau_react = {t['tau_react']:.6g}, "
          f"ratio = {t['ratio']:.6g}")
    print("    mu = 1.0 is 1000x Zhao's table 1 value, so this row describes "
          "no case in the paper.")


if __name__ == "__main__":
    main()
