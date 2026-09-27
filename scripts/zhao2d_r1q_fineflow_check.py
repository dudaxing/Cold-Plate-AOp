"""R1q: the check layer as a differentiable model of the coarse design -- checked, not optimised.

The contract proposed in the review of b230f58. The model is
`zhao2d_fineflow.Zhao2DFineFlowProblem`: the design on h (5000 raw variables,
the filter at its physical radius 2e-4, the volume-preserving projection at
beta = 32 with eta solved on the coarse design volume), the flow on h/2 and the
temperature on h/8 -- R1m's check layer -- evaluated on the development
model's Psi_0 and C_0 as a declared common scale.

Checked in stages; a failure writes the record and exits non-zero before the
next solve:

1. identities: the flow and thermal sides are R1m's check layer (mesh, source
   and Dirichlet digests); the design side has 5000 variables; the common
   scale's source is the development model's reference;
2. anchors, nothing solved: R1n's and R1o's given binary designs, with the
   check-layer states R1n and R1o saved -- the new model's copy of each design
   to the flow mesh is the saved one, the flows re-verify, and Psi and C
   reproduce the records;
3. the main point, R1o's raw continuous terminal x (not its binary mask): its
   root is non-degenerate, and eta and the constraint reproduce R1o's terminal
   record; then ONE value-and-gradient evaluation (one forward solve, two
   reverse passes);
4. the existing directional-difference protocol: 2 directions fixed here
   (random signs, seeds 11 and 12, on the variables at least the largest step
   away from both bounds), 2 steps (1e-4, 1e-5) x 2 signs = 8 perturbed
   evaluations, each re-solving the flow and the temperature and gated at 1e-8.
   For each direction and quantity (Psi, C, g, J) the best step must agree to
   1e-5 relative; every step's absolute and relative difference is reported.

No MMA, no change of beta, q or thermal form, no new reference, no h/16.

    python scripts/zhao2d_r1q_fineflow_check.py [--inputs results] [--out DIR] [--overwrite]
"""

from __future__ import annotations

import pathlib
import sys

# Default the BLAS thread count BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import faulthandler
import json
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))
sys.path.insert(0, str(REPO / "scripts"))

from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_binary as zb  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_fineflow as ff  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file  # noqa: E402
from zhao2d_r1m_flow_check import mesh_identity, thermal_identity  # noqa: E402

ALPHA_MAX, BETA = 1.0e7, 32.0
FLOW_REFINEMENT, THERMAL_REFINEMENT, QUADRATURE = 2, 4, 3
GATE, ANCHOR_TOL = 1e-8, 1e-12
STEPS, SEEDS, TOL = (1e-4, 1e-5), (11, 12), 1e-5
RECORD = "zhao2d_r1q_fineflow_check.json"
FIELDS = "zhao2d_r1q_fields.npz"
STACKS = "zhao2d_r1q_stacks.log"
INPUTS = ("zhao2d_r1m_flow_check.json", "zhao2d_r1n_beta32.json", "zhao2d_r1n_fields.npz",
          "zhao2d_r1o.json", "zhao2d_r1o_fields.npz")
QUANTITIES = {"psi": "psi", "c": "compliance", "g": "constraint_g", "J": "J_common_scale"}


def load_json(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=pathlib.Path, default=REPO / "results")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    inputs, out = args.inputs, args.out
    existing = [n for n in (RECORD, FIELDS) if (out / n).exists()]
    if existing and not args.overwrite:
        sys.exit(f"{out} already holds {', '.join(existing)}, which is cited evidence; write "
                 "elsewhere with --out DIR, or pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    stacks = open(out / STACKS, "w", encoding="utf-8")
    faulthandler.dump_traceback_later(7200, repeat=True, file=stacks)  # longer than a healthy run
    t_start = time.perf_counter()

    spec = z.Zhao2DSpec()
    config = r1.R1Config(projection=r1.Projection.VOLUME_PRESERVING)
    m_rec = load_json(inputs / "zhao2d_r1m_flow_check.json")
    recs = {"r1n": load_json(inputs / "zhao2d_r1n_beta32.json"),
            "r1o": load_json(inputs / "zhao2d_r1o.json")}
    fields = {"r1n": np.load(inputs / "zhao2d_r1n_fields.npz"),
              "r1o": np.load(inputs / "zhao2d_r1o_fields.npz")}
    x = np.asarray(fields["r1o"]["design"], dtype=np.float64)
    given = {name: {"s": fields[name]["solid_fraction_binary"],
                    "s_flow": fields[name]["new_solid_fraction_flow_h2"],
                    "press_vel": fields[name]["new_press_vel_check"],
                    "temperature": fields[name]["new_temperature_check"],
                    "cell": recs[name]["cells"]["new/check"]} for name in recs}

    record = {
        "note": ("R1q: the check layer (flow h/2, thermal h/8) as a differentiable model of the "
                 "5000 coarse design variables, on the development model's Psi_0 and C_0 as a "
                 "declared common scale; anchored on R1n's and R1o's saved check-layer states and "
                 "checked at R1o's raw terminal design against central differences. No MMA."),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in INPUTS},
        "model": {"design": "h, 5000 raw variables", "flow_refinement": FLOW_REFINEMENT,
                  "thermal_refinement": THERMAL_REFINEMENT, "thermal_quadrature": QUADRATURE,
                  "alpha_max": ALPHA_MAX, "beta": BETA, "projection": config.projection},
        "criterion": {"steps": list(STEPS), "seeds": list(SEEDS), "best_step_relative_tolerance": TOL,
                      "gate": GATE, "fixed_before_the_run": True},
    }
    for script in (pathlib.Path(__file__).resolve(), REPO / "scripts" / "zhao2d_r1m_flow_check.py"):
        record["provenance"]["source_sha256"][script.relative_to(REPO).as_posix()] = sha256_file(script)
    failures, checkpoints, cost, saved = [], [], {"evaluations": []}, {}

    def require(ok, what: str) -> bool:
        if not ok:
            failures.append(what)
        return bool(ok)

    def finish() -> None:
        record["checkpoints"] = checkpoints
        if saved:
            np.savez_compressed(out / FIELDS, **saved)
            record["fields"] = {"file": FIELDS, "arrays": sorted(saved)}
        record["cost"] = {**cost, "peak_working_set_mb": peak_working_set_mb(),
                          "peak_working_set_note": "the whole process's, cumulative",
                          "wall_clock_total_s": time.perf_counter() - t_start}
        faulthandler.cancel_dump_traceback_later()
        stacks.close()
        if (out / STACKS).stat().st_size == 0:
            (out / STACKS).unlink()
        (out / RECORD).write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")

    def checkpoint(stage: str) -> None:
        """Stop here, with the record written, if anything so far has failed."""
        checkpoints.append({"stage": stage, "failures": list(failures)})
        if failures:
            record["stopped"] = f"{stage}: checks failed; nothing after it was solved"
            finish()
            sys.exit(f"CHECKS FAILED at {stage}; nothing after it was solved: {failures}")
        print(f"checkpoint passed: {stage}", flush=True)

    def gated_evaluate(label: str, point, gradient: bool):
        """ff.evaluate, timed; a failed gate is recorded and stops the run here."""
        t0 = time.perf_counter()
        try:
            out_ = ff.evaluate(problem, scale, jnp.asarray(point), ALPHA_MAX, BETA, gradient, GATE)
        except (r1.NotConverged, r1.DegenerateProjection) as exc:
            require(False, f"{label}: {type(exc).__name__}: {exc}")
            checkpoint(label)
        seconds = time.perf_counter() - t0
        cost["evaluations"].append({"label": label, "gradient": gradient, "seconds": seconds})
        return out_, seconds

    # -- inputs -----------------------------------------------------------------------
    require(np.array_equal(fields["r1o"]["design"], fields["r1o"]["designs"][-1]),
            "the main point is not R1o's last saved design row")
    for name, d in given.items():
        require(not any(c["failures"] for c in recs[name]["checkpoints"]),
                f"{name}: its record has failed checks")
        sha = d["cell"]["state_sha256"]
        require(fs.digest(d["s"]) == recs[name]["export"]["binary_sha256"],
                f"{name}: s is not its exported binary design")
        for mine, theirs in (("s_flow", "s"), ("press_vel", "press_vel"), ("temperature", "temperature")):
            require(fs.digest(d[mine]) == sha[theirs], f"{name}: the saved {mine} is not the recorded one")
    record["main_point"] = {"source": "R1o's raw continuous terminal design",
                            "sha256_float64": fs.digest(x)}
    checkpoint("inputs")

    # -- 1. the model and its identities ------------------------------------------------
    t0 = time.perf_counter()
    problem = ff.Zhao2DFineFlowProblem(spec, config, FLOW_REFINEMENT, THERMAL_REFINEMENT, QUADRATURE)
    cost["build_s"] = time.perf_counter() - t0
    print(f"built: design {problem.design_mesh.num_elems} el ({problem.num_design} variables), flow "
          f"{problem.flow_mesh.num_elems} el, thermal {problem.thermal_mesh.num_elems} el "
          f"({cost['build_s']:.1f} s)", flush=True)
    rows = m_rec["rows"]["flow_h2"]
    record["identities"] = {"flow_mesh": mesh_identity(problem.flow_mesh),
                            "thermal_mesh": thermal_identity(problem),
                            "design_mesh": mesh_identity(problem.design_mesh),
                            "num_design": problem.num_design,
                            "filter_radius": config.filter_radius_elements * spec.element_size,
                            "model_identity": json.loads(problem.model_identity()),
                            "nesting": problem.nesting}
    require(record["identities"]["flow_mesh"] == rows["flow_mesh"], "the flow side is not R1m's check layer")
    require(record["identities"]["thermal_mesh"] == rows["thermal_mesh"],
            "the thermal side is not R1m's check layer")
    require(record["identities"]["design_mesh"] == m_rec["rows"]["flow_h"]["flow_mesh"],
            "the design mesh is not the development model's h mesh")
    require(problem.num_design == 5000, f"{problem.num_design} design variables, not 5000")
    try:
        scale = ff.common_scale(problem, dual.reference_file(4, QUADRATURE), 4, QUADRATURE)
    except ValueError as exc:
        require(False, f"the common scale: {exc}")
        checkpoint("identities and the common scale")
    record["scale"] = {"psi_0": scale.psi_0, "c_0": scale.c_0, "source_file": scale.source_file,
                       "source_sha256": scale.source_sha256, "source_model": scale.source_model,
                       "declared": "the development model's constants, used as a common scale; "
                                   "not a reference of this model"}
    for name in recs:
        require(all(recs[name]["yardstick"][k] == v for k, v in (("psi_0", scale.psi_0), ("c_0", scale.c_0))),
                f"the scale is not the one {name} was ranked on")
    checkpoint("identities and the common scale")

    # -- 2. anchors: the given binary designs on the new model, nothing solved ----------
    w = config.weight
    anchors = {}
    for name, d in given.items():
        s = jnp.asarray(d["s"])
        pv, temp = jnp.asarray(d["press_vel"]), jnp.asarray(d["temperature"])
        s_flow = np.asarray(problem.flow_density(s))
        a = {"copy_to_flow_mesh_is_the_saved_one": bool(np.array_equal(s_flow, d["s_flow"]))}
        anchors[name] = a
        require(a["copy_to_flow_mesh_is_the_saved_one"], f"{name}: E_DF s is not the saved flow-mesh density")
        try:
            a["flow_verification"] = fs.verify_flow_state(problem, pv, s_flow, ALPHA_MAX, GATE)
        except (ValueError, r1.NotConverged) as exc:
            require(False, f"{name}: the saved flow does not verify: {exc}")
            continue
        alpha = problem.flow_material(s, ALPHA_MAX)
        kappa = problem.thermal_conductivity(s, ALPHA_MAX)
        psi = float(problem.flow.dissipated_power(pv, alpha))
        c = float(problem.thermal.thermal_compliance(temp, problem.thermal_velocity(pv), kappa))
        norms = problem.residual_norms_at(pv, temp, alpha, kappa)
        a.update(psi=psi, compliance=c, J_common_scale=w * psi / scale.psi_0 + (1 - w) * c / scale.c_0,
                 residuals=norms, gate_passed=zb.gate_passed(norms, GATE),
                 anchor=zb.anchor_status({"psi": psi, "compliance": c},
                                         {"psi": d["cell"]["psi"], "compliance": d["cell"]["compliance"],
                                          "source": f"{name}'s check-layer record"}, ANCHOR_TOL))
        require(a["gate_passed"], f"{name}: a residual fails the gate {norms}")
        require(a["anchor"]["reproduced"], f"{name}: does not reproduce its record: {a['anchor']}")
        print(f"anchor {name}: Psi {psi:.9e}  C {c:.6f}  J {a['J_common_scale']:.9f}  |R| "
              f"{norms['flow']:.1e}/{norms['thermal']:.1e}  "
              f"{'ok' if a['anchor']['reproduced'] else 'MISMATCH'}", flush=True)
    record["anchors"] = anchors
    checkpoint("anchors: the given binary designs, nothing solved")

    # -- 3. the main point: identity, then one value and gradient ------------------------
    root = problem.projection_root(jnp.asarray(x), BETA)
    g0 = float(problem.fluid_fraction(jnp.asarray(x), BETA)) / config.max_fluid_fraction - 1.0
    term = recs["r1o"]["terminal"]
    record["main_point"].update(
        root=root, g=g0, eta_vs_r1o_terminal=root["eta"] - term["projection_eta"],
        g_vs_r1o_terminal=g0 - term["constraint_g"])
    require(root["nondegenerate"], f"the main point's projection root is degenerate: {root}")
    require(abs(root["eta"] - term["projection_eta"]) <= 1e-12,
            f"eta {root['eta']} is not R1o's terminal {term['projection_eta']}")
    require(abs(g0 - term["constraint_g"]) <= 1e-12,
            f"the constraint {g0} is not R1o's terminal {term['constraint_g']}")
    checkpoint("main point: root, eta and volume")

    (main, state, grads), seconds = gated_evaluate("main point: value and gradient", x, True)
    record["main_point"].update(evaluation=main, seconds=seconds,
                                development_model_J=term["J_self"],
                                gradient_norms={k: float(np.linalg.norm(v)) for k, v in grads.items()},
                                gradient_sha256={k: fs.digest(v) for k, v in grads.items()})
    saved.update({f"gradient_{k}": v for k, v in grads.items()})
    saved.update(main_point_x=x, main_point_s=state[0], main_point_press_vel=state[1],
                 main_point_temperature=state[2])
    print(f"main point: J {main['J_common_scale']:.9f} (development model {term['J_self']:.9f})  "
          f"Psi {main['psi']:.9e}  C {main['compliance']:.6f}  g {main['constraint_g']:+.3e}  |R| "
          f"{main['flow_residual_relative']:.1e}/{main['thermal_residual_relative']:.1e}  "
          f"({seconds:.1f} s, one forward solve and two reverse passes)", flush=True)
    checkpoint("main point: value and gradient")

    # -- 4. directional differences ------------------------------------------------------
    free = (x >= STEPS[0]) & (x <= 1.0 - STEPS[0])
    directions = []
    for seed in SEEDS:
        d = np.random.default_rng(seed).choice([-1.0, 1.0], x.size) * free
        ad = {q: float(np.dot(grads[q], d)) for q in QUANTITIES}
        rows_ = []
        for h in STEPS:
            vals = {}
            for sign in (1.0, -1.0):
                (rec_, _, _), secs = gated_evaluate(f"seed {seed}, step {h:g}, sign {sign:+g}",
                                                    x + sign * h * d, False)
                vals[sign] = rec_
                print(f"  seed {seed} step {h:.0e} sign {sign:+.0f}: J {rec_['J_common_scale']:.12f}  "
                      f"|R| {rec_['flow_residual_relative']:.1e}/{rec_['thermal_residual_relative']:.1e}"
                      f"  ({secs:.1f} s)", flush=True)
            row = {"step": h}
            for q, key in QUANTITIES.items():
                fd = (vals[1.0][key] - vals[-1.0][key]) / (2 * h)
                row[q] = {"fd": fd, "ad": ad[q], "abs_error": abs(ad[q] - fd),
                          "rel_error": abs(ad[q] - fd) / max(abs(fd), 1e-300)}
            rows_.append(row)
        best = {q: min(r[q]["rel_error"] for r in rows_) for q in QUANTITIES}
        directions.append({"seed": seed, "active_components": int(free.sum()), "ad": ad,
                           "steps": rows_, "best_relative_error": best,
                           "pass": {q: best[q] <= TOL for q in QUANTITIES}})
        print(f"seed {seed}: best-step relative error " + "  ".join(
            f"{q} {best[q]:.1e}" for q in QUANTITIES), flush=True)
    record["directions"] = directions
    record["verdict"] = {
        "all_pass": all(all(d["pass"].values()) for d in directions),
        "worst_best_step_relative_error": max(max(d["best_relative_error"].values())
                                              for d in directions)}
    cost["solves"] = {"forward_flow": 1 + 2 * len(STEPS) * len(SEEDS),
                      "forward_thermal": 1 + 2 * len(STEPS) * len(SEEDS),
                      "reverse_passes": 2, "mma": 0}
    finish()

    print(f"\nverdict: {'PASS' if record['verdict']['all_pass'] else 'FAIL'} at {TOL:g}; worst "
          f"best-step relative error {record['verdict']['worst_best_step_relative_error']:.1e}")
    print(f"wrote {out / RECORD} and {out / FIELDS}; total {record['cost']['wall_clock_total_s']:.0f} s")
    if not record["verdict"]["all_pass"]:
        sys.exit("the total gradient does not pass the directional-difference criterion")


if __name__ == "__main__":
    main()
